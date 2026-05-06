"""End-to-end Clash Royale agent: tokenizer → Perceiver → Decision Transformer → heads.

Used by both training and (future) inference. Inference helpers consume a single
trajectory window (B=1, T=seq_len) and return the predicted action for the last
timestep.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from configs.policy_config import FullConfig
from models.policy.decision_transformer import DecisionTransformer
from models.policy.perceiver_encoder import PerceiverEncoder
from models.policy.policy_heads import HeadOutput, PolicyHeads
from perception.tokenizer import FrameTokenizer


@dataclass
class TrajectoryBatch:
    """A batch of trajectory windows (B, T, ...)."""

    images: torch.Tensor         # (B, T, 3, H, W)
    det_cls: torch.Tensor        # (B, T, M)
    det_bbox: torch.Tensor       # (B, T, M, 4)
    det_conf: torch.Tensor       # (B, T, M)
    det_side: torch.Tensor       # (B, T, M)
    det_track: torch.Tensor      # (B, T, M)
    det_mask: torch.Tensor       # (B, T, M) bool
    cards: torch.Tensor          # (B, T, n_hand)
    elixir: torch.Tensor         # (B, T)
    rtg: torch.Tensor            # (B, T)
    prev_a_card: torch.Tensor    # (B, T)
    prev_a_pos: torch.Tensor     # (B, T, 2)
    prev_r: torch.Tensor         # (B, T)
    timesteps: torch.Tensor      # (B, T) absolute frame index

    def to(self, device) -> "TrajectoryBatch":
        return TrajectoryBatch(**{k: v.to(device) for k, v in self.__dict__.items()})


class ClashRoyaleAgent(nn.Module):
    def __init__(self, cfg: FullConfig):
        super().__init__()
        self.cfg = cfg
        tk = cfg.tokenizer
        m = cfg.model

        self.tokenizer = FrameTokenizer(
            d_model=m.d_model,
            image_size=tk.image_size,
            patch_size=tk.patch_size,
            n_classes=tk.n_classes,
            max_objects=tk.max_objects,
            n_cards=tk.n_cards,
            n_hand=tk.n_hand,
            n_elixir=tk.n_elixir,
            max_track_ids=tk.max_track_ids,
        )
        self.encoder = PerceiverEncoder(
            d_model=m.perceiver.d_model,
            n_latents=m.perceiver.n_latents,
            n_cross_heads=m.perceiver.n_cross_heads,
            n_self_heads=m.perceiver.n_self_heads,
            n_self_layers=m.perceiver.n_self_layers,
            n_cross_blocks=m.perceiver.n_cross_blocks,
            ff_mult=m.perceiver.ff_mult,
            dropout=m.perceiver.dropout,
            share_weights_across_blocks=m.perceiver.share_weights_across_blocks,
        )
        self.dt = DecisionTransformer(
            d_model=m.d_model,
            n_heads=m.n_heads,
            n_layers=m.n_layers,
            ff_mult=m.ff_mult,
            dropout=m.dropout,
            seq_len=m.seq_len,
            n_latents=m.perceiver.n_latents,
            max_timesteps=m.max_timesteps,
        )
        self.heads = PolicyHeads(
            d_model=m.d_model,
            n_hand=tk.n_hand,
            grid_rows=tk.grid_rows,
            grid_cols=tk.grid_cols,
            max_delay=tk.max_delay,
            pos_mode=m.heads.pos_mode,
            enable_delay=m.heads.enable_delay,
        )

    def encode_frames(self, batch: TrajectoryBatch) -> torch.Tensor:
        """Per-timestep encode → (B, T, n_latents, D)."""
        B, T = batch.images.shape[:2]

        def flat(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(B * T, *x.shape[2:])

        ft = self.tokenizer(
            images=flat(batch.images),
            det_cls=flat(batch.det_cls),
            det_bbox=flat(batch.det_bbox),
            det_conf=flat(batch.det_conf),
            det_side=flat(batch.det_side),
            det_track=flat(batch.det_track),
            det_mask=flat(batch.det_mask),
            cards=flat(batch.cards),
            elixir=flat(batch.elixir),
            rtg=flat(batch.rtg),
            prev_a_card=flat(batch.prev_a_card),
            prev_a_pos=flat(batch.prev_a_pos),
            prev_r=flat(batch.prev_r),
        )
        latents = self.encoder(ft.tokens, ft.mask)                     # (B*T, K, D)
        K, D = latents.shape[-2], latents.shape[-1]
        return latents.reshape(B, T, K, D)

    def forward(self, batch: TrajectoryBatch) -> HeadOutput:
        latents = self.encode_frames(batch)
        h = self.dt(latents, batch.timesteps)
        return self.heads(h)

    @torch.no_grad()
    def predict_last(self, batch: TrajectoryBatch) -> Dict[str, torch.Tensor]:
        """Convenience: forward + argmax of the last timestep only."""
        out = self.forward(batch)
        select = out.select_logits[:, -1].argmax(-1)
        if out.pos_logits is not None:
            flat = out.pos_logits[:, -1].argmax(-1)
            row = flat // self.cfg.tokenizer.grid_cols
            col = flat % self.cfg.tokenizer.grid_cols
            pos = torch.stack([row, col], dim=-1)
        else:
            pos = (out.pos_xy[:, -1] * torch.tensor(
                [self.cfg.tokenizer.grid_rows, self.cfg.tokenizer.grid_cols],
                device=out.pos_xy.device,
                dtype=out.pos_xy.dtype,
            )).long()
        ret = {"select": select, "pos": pos}
        if out.delay_logits is not None:
            ret["delay"] = out.delay_logits[:, -1].argmax(-1)
        return ret
