"""Per-frame multimodal tokenizer.

Turns one (frame + YOLO detections + hand cards + elixir) snapshot into a single
sequence of token embeddings, plus a token-type tag and an attention mask. This
sequence is what gets fed to the Perceiver encoder as cross-attention keys/values.

Token types
-----------
0 PATCH    image patch token  (ViT patchify of the arena crop)
1 OBJECT   YOLO object token  (class + bbox + conf + side + track)
2 CARD     one hand-card token per slot
3 ELIXIR   single elixir-count token
4 RTG      return-to-go token
5 PREV_A   previous-action token  (card_idx + xy)
6 PREV_R   previous-reward token

The encoder is shape-agnostic — it cross-attends a fixed-size latent array to
this variable-length sequence, so we don't need to pad to a global maximum.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# Stable token-type ids — keep in sync with downstream code.
TOK_PATCH = 0
TOK_OBJECT = 1
TOK_CARD = 2
TOK_ELIXIR = 3
TOK_RTG = 4
TOK_PREV_A = 5
TOK_PREV_R = 6
N_TOKEN_TYPES = 7


@dataclass
class FrameTokens:
    """Stacked token batch ready for the Perceiver."""

    tokens: torch.Tensor          # (B, L, D)
    type_ids: torch.Tensor        # (B, L) long
    mask: torch.Tensor            # (B, L) bool — True means real, False means pad


class PatchEmbed(nn.Module):
    """Standard ViT-style 2D conv patchifier."""

    def __init__(self, image_size: Tuple[int, int], patch_size: int, d_model: int, in_channels: int = 3):
        super().__init__()
        H, W = image_size
        assert H % patch_size == 0 and W % patch_size == 0, (
            f"image_size {image_size} must divide patch_size {patch_size}"
        )
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.n_patches = (H // patch_size) * (W // patch_size)
        self.register_buffer("pos_embed", _sin_cos_2d(H // patch_size, W // patch_size, d_model), persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, 3, H, W) in [0, 1]
        x = self.proj(images)                     # (B, D, h, w)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = x + self.pos_embed.to(x.dtype)
        return x


def _sin_cos_2d(h: int, w: int, dim: int) -> torch.Tensor:
    """2D sin/cos positional embedding shared across the patch grid."""
    assert dim % 4 == 0, "patch embed dim must be divisible by 4"
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    half = dim // 2
    freqs = torch.exp(torch.arange(0, half, 2, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / max(half - 1, 1)))
    pe_x = torch.zeros(h, w, half)
    pe_y = torch.zeros(h, w, half)
    pe_x[..., 0::2] = torch.sin(grid_x[..., None] * freqs)
    pe_x[..., 1::2] = torch.cos(grid_x[..., None] * freqs)
    pe_y[..., 0::2] = torch.sin(grid_y[..., None] * freqs)
    pe_y[..., 1::2] = torch.cos(grid_y[..., None] * freqs)
    return rearrange(torch.cat([pe_y, pe_x], dim=-1), "h w d -> 1 (h w) d")


class FourierBBoxEmbed(nn.Module):
    """Cheap continuous bbox encoder: random-Fourier features + MLP."""

    def __init__(self, d_model: int, n_freqs: int = 16):
        super().__init__()
        self.register_buffer(
            "freqs",
            2.0 ** torch.arange(n_freqs).float() * torch.pi,
            persistent=False,
        )
        self.proj = nn.Sequential(
            nn.Linear(4 * 2 * n_freqs, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, bbox: torch.Tensor) -> torch.Tensor:
        # bbox: (..., 4) cx, cy, w, h in [0, 1]
        x = bbox.unsqueeze(-1) * self.freqs                       # (..., 4, F)
        x = torch.cat([x.sin(), x.cos()], dim=-1)                  # (..., 4, 2F)
        x = rearrange(x, "... c f -> ... (c f)")
        return self.proj(x)


class FrameTokenizer(nn.Module):
    """Build a single-frame multimodal token sequence.

    All inputs may be batched (B, ...). Tensors are expected pre-batched by the
    dataset loader; the model sees them as a single timestep.
    """

    def __init__(
        self,
        d_model: int,
        image_size: Tuple[int, int],
        patch_size: int,
        n_classes: int,
        max_objects: int,
        n_cards: int,
        n_hand: int,
        n_elixir: int,
        max_track_ids: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_objects = max_objects
        self.n_hand = n_hand

        self.patch = PatchEmbed(image_size=image_size, patch_size=patch_size, d_model=d_model)

        self.cls_emb = nn.Embedding(n_classes + 1, d_model, padding_idx=0)
        self.bbox_emb = FourierBBoxEmbed(d_model)
        self.conf_proj = nn.Linear(1, d_model)
        self.side_emb = nn.Embedding(3, d_model)            # 0 unknown, 1 ally, 2 enemy
        self.track_emb = nn.Embedding(max_track_ids + 1, d_model, padding_idx=0)

        # Card / elixir tokens
        self.card_emb = nn.Embedding(n_cards + 1, d_model, padding_idx=0)
        self.card_slot_emb = nn.Embedding(n_hand, d_model)
        self.elixir_emb = nn.Embedding(n_elixir + 1, d_model)

        # Scalar tokens (RTG, prev reward) projected through small MLPs
        self.rtg_proj = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.prev_r_proj = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))

        # Previous action: card slot + grid cell (cell embedded via Fourier on (row, col))
        self.prev_a_card_emb = nn.Embedding(n_hand + 1, d_model, padding_idx=0)        # 0 = no-op
        self.prev_a_pos_emb = FourierBBoxEmbed(d_model, n_freqs=8)

        # Token-type bias added to every token regardless of source
        self.type_emb = nn.Embedding(N_TOKEN_TYPES, d_model)

    def forward(
        self,
        images: torch.Tensor,                           # (B, 3, H, W)
        det_cls: torch.Tensor,                          # (B, M) long
        det_bbox: torch.Tensor,                         # (B, M, 4) float
        det_conf: torch.Tensor,                         # (B, M) float
        det_side: torch.Tensor,                         # (B, M) long
        det_track: torch.Tensor,                        # (B, M) long
        det_mask: torch.Tensor,                         # (B, M) bool
        cards: torch.Tensor,                            # (B, n_hand) long
        elixir: torch.Tensor,                           # (B,) long
        rtg: torch.Tensor,                              # (B,) float
        prev_a_card: torch.Tensor,                      # (B,) long  (0 = no-op)
        prev_a_pos: torch.Tensor,                       # (B, 2) float in [0, 1]
        prev_r: torch.Tensor,                           # (B,) float
    ) -> FrameTokens:
        B = images.shape[0]
        device = images.device

        # 1) Patch tokens
        patch_tok = self.patch(images)                                     # (B, P, D)
        patch_mask = torch.ones(patch_tok.shape[:2], dtype=torch.bool, device=device)
        patch_type = torch.full(patch_tok.shape[:2], TOK_PATCH, dtype=torch.long, device=device)

        # 2) Object tokens — combine class/bbox/conf/side/track
        obj_tok = (
            self.cls_emb(det_cls + 1)                                      # +1 because 0 is padding
            + self.bbox_emb(det_bbox)
            + self.conf_proj(det_conf.unsqueeze(-1))
            + self.side_emb(det_side.clamp(0, 2))
            + self.track_emb(det_track.clamp_max(self.track_emb.num_embeddings - 1))
        )
        # Zero out padded slots so they contribute nothing even if the mask leaks
        obj_tok = obj_tok * det_mask.unsqueeze(-1).to(obj_tok.dtype)
        obj_type = torch.full(obj_tok.shape[:2], TOK_OBJECT, dtype=torch.long, device=device)

        # 3) Card tokens
        slot_idx = torch.arange(self.n_hand, device=device)
        card_tok = self.card_emb(cards.clamp_min(0)) + self.card_slot_emb(slot_idx)[None]   # (B, n_hand, D)
        card_mask = (cards >= 0)
        card_type = torch.full(card_tok.shape[:2], TOK_CARD, dtype=torch.long, device=device)

        # 4) Elixir token
        elx_tok = self.elixir_emb(elixir.clamp_min(0)).unsqueeze(1)                          # (B, 1, D)
        elx_mask = torch.ones((B, 1), dtype=torch.bool, device=device)
        elx_type = torch.full((B, 1), TOK_ELIXIR, dtype=torch.long, device=device)

        # 5) RTG token
        rtg_tok = self.rtg_proj(rtg.unsqueeze(-1)).unsqueeze(1)                              # (B, 1, D)
        rtg_mask = torch.ones((B, 1), dtype=torch.bool, device=device)
        rtg_type = torch.full((B, 1), TOK_RTG, dtype=torch.long, device=device)

        # 6) Previous action token
        prev_a_tok = (
            self.prev_a_card_emb(prev_a_card.clamp_min(0))
            + self.prev_a_pos_emb(F.pad(prev_a_pos, (0, 2)))                                  # pad to 4-dim for shared module
        ).unsqueeze(1)
        prev_a_mask = torch.ones((B, 1), dtype=torch.bool, device=device)
        prev_a_type = torch.full((B, 1), TOK_PREV_A, dtype=torch.long, device=device)

        # 7) Previous reward token
        prev_r_tok = self.prev_r_proj(prev_r.unsqueeze(-1)).unsqueeze(1)
        prev_r_mask = torch.ones((B, 1), dtype=torch.bool, device=device)
        prev_r_type = torch.full((B, 1), TOK_PREV_R, dtype=torch.long, device=device)

        # Concatenate
        toks = torch.cat([patch_tok, obj_tok, card_tok, elx_tok, rtg_tok, prev_a_tok, prev_r_tok], dim=1)
        masks = torch.cat([patch_mask, det_mask, card_mask, elx_mask, rtg_mask, prev_a_mask, prev_r_mask], dim=1)
        types = torch.cat([patch_type, obj_type, card_type, elx_type, rtg_type, prev_a_type, prev_r_type], dim=1)

        toks = toks + self.type_emb(types)
        return FrameTokens(tokens=toks, type_ids=types, mask=masks)


def flatten_time_batch(images: torch.Tensor, *others: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Helper: collapse (B, T, ...) into (B*T, ...)."""
    B, T = images.shape[:2]
    out = [images.reshape(B * T, *images.shape[2:])]
    for x in others:
        out.append(x.reshape(B * T, *x.shape[2:]))
    return tuple(out)
