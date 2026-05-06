"""Decision Transformer over per-timestep Perceiver latents.

Sequence layout per sample of length T (T = seq_len):

    [ S_1, ..., S_K, ACT_1 ]   (timestep 1)
    [ S_1, ..., S_K, ACT_2 ]   (timestep 2)
    ...

where S_1..S_K are the K = n_latents Perceiver outputs and ACT is a single
"action slot" reserved at every timestep — the Transformer reads through
the slot at the end of every step to produce the action prediction. Causal
mask enforces left-to-right autoregression across timesteps.

Time and timestep embeddings are added to all tokens within a step. The
slot order is the same at every timestep so positional embedding is the
combination of (timestep_idx, slot_idx_within_step).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class DecisionTransformer(nn.Module):
    """Causal Transformer over (latents + action-slot) tokens across T timesteps."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_mult: int,
        dropout: float,
        seq_len: int,
        n_latents: int,
        max_timesteps: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.n_latents = n_latents
        self.tokens_per_step = n_latents + 1                     # +1 action slot
        self.total_tokens = seq_len * self.tokens_per_step

        # Learned embedding for each slot position within a timestep
        self.slot_embed = nn.Embedding(self.tokens_per_step, d_model)
        # Absolute timestep embedding (frame index in episode)
        self.time_embed = nn.Embedding(max_timesteps + 1, d_model)
        # The "action slot" is its own learned vector that the model reads through
        self.act_slot = nn.Parameter(torch.randn(d_model) * 0.02)

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            CausalBlock(d_model, n_heads, ff_mult, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        # Causal mask reused across forward calls
        mask = torch.triu(torch.full((self.total_tokens, self.total_tokens), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, latents: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        latents:    (B, T, K, D)   per-timestep Perceiver outputs
        timesteps:  (B, T)         absolute frame index per step
        Returns:    (B, T, D)      hidden state at the action slot of each step
        """
        B, T, K, D = latents.shape
        assert T == self.seq_len, f"DT expects seq_len={self.seq_len}, got T={T}"
        assert K == self.n_latents, f"DT expects n_latents={self.n_latents}, got K={K}"

        # Action slot vector for every step
        act_tok = self.act_slot[None, None, None, :].expand(B, T, 1, D)
        x = torch.cat([latents, act_tok], dim=2)                                 # (B, T, K+1, D)

        # Slot positional embedding
        slot_idx = torch.arange(self.tokens_per_step, device=latents.device)
        x = x + self.slot_embed(slot_idx)[None, None]

        # Absolute timestep embedding
        t_emb = self.time_embed(timesteps.clamp_max(self.time_embed.num_embeddings - 1))   # (B, T, D)
        x = x + t_emb[..., None, :]

        x = x.reshape(B, T * self.tokens_per_step, D)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, self.causal_mask)
        x = self.ln_f(x)

        # Pull the action-slot output (last slot) of each timestep
        x = x.reshape(B, T, self.tokens_per_step, D)
        return x[:, :, -1, :]
