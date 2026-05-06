"""Perceiver IO style multimodal encoder.

A fixed-size latent array cross-attends a long, variable-length sequence of
multimodal tokens (image patches + YOLO objects + cards + scalar tokens).
A stack of latent self-attention layers processes the compressed
representation. Output is the latent array, used as the per-timestep state
for the Decision Transformer.

Reference: Jaegle et al., "Perceiver IO: A General Architecture for Structured Inputs & Outputs" (2021).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = d_model * mult
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    """Latents (Q) attend over inputs (K, V) — Perceiver-style bottleneck."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, latents: torch.Tensor, kv: torch.Tensor, kv_mask: torch.Tensor) -> torch.Tensor:
        B, N, D = latents.shape
        M = kv.shape[1]
        q = self.q_proj(self.norm_q(latents)).view(B, N, self.h, self.dh).transpose(1, 2)
        kn = self.norm_kv(kv)
        k = self.k_proj(kn).view(B, M, self.h, self.dh).transpose(1, 2)
        v = self.v_proj(kn).view(B, M, self.h, self.dh).transpose(1, 2)
        # (B, h, N, M)
        attn_bias = None
        if kv_mask is not None:
            attn_bias = torch.zeros((B, 1, 1, M), device=latents.device, dtype=q.dtype)
            attn_bias = attn_bias.masked_fill(~kv_mask[:, None, None, :], float("-inf"))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=0.0)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.drop(self.out(out))


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(self.norm(x)).view(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.drop(self.out(out))


class PerceiverBlock(nn.Module):
    """One cross-attention block followed by a stack of self-attention layers."""

    def __init__(self, d_model: int, n_cross_heads: int, n_self_heads: int, n_self_layers: int,
                 ff_mult: int, dropout: float):
        super().__init__()
        self.cross_attn = CrossAttention(d_model, n_cross_heads, dropout)
        self.cross_ff = FeedForward(d_model, ff_mult, dropout)
        self.cross_ff_norm = nn.LayerNorm(d_model)

        self.self_layers = nn.ModuleList()
        for _ in range(n_self_layers):
            self.self_layers.append(nn.ModuleDict({
                "attn": SelfAttention(d_model, n_self_heads, dropout),
                "ff": FeedForward(d_model, ff_mult, dropout),
                "ff_norm": nn.LayerNorm(d_model),
            }))

    def forward(self, latents: torch.Tensor, kv: torch.Tensor, kv_mask: torch.Tensor) -> torch.Tensor:
        latents = latents + self.cross_attn(latents, kv, kv_mask)
        latents = latents + self.cross_ff(self.cross_ff_norm(latents))
        for layer in self.self_layers:
            latents = latents + layer["attn"](latents)
            latents = latents + layer["ff"](layer["ff_norm"](latents))
        return latents


class PerceiverEncoder(nn.Module):
    """Compresses a multimodal token cloud into a fixed (n_latents, d_model) array."""

    def __init__(
        self,
        d_model: int,
        n_latents: int,
        n_cross_heads: int,
        n_self_heads: int,
        n_self_layers: int,
        n_cross_blocks: int,
        ff_mult: int,
        dropout: float,
        share_weights_across_blocks: bool = True,
    ):
        super().__init__()
        self.n_latents = n_latents
        self.d_model = d_model
        self.n_cross_blocks = n_cross_blocks
        self.share = share_weights_across_blocks

        self.latent = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)

        if share_weights_across_blocks:
            self.block = PerceiverBlock(d_model, n_cross_heads, n_self_heads, n_self_layers, ff_mult, dropout)
        else:
            self.blocks = nn.ModuleList([
                PerceiverBlock(d_model, n_cross_heads, n_self_heads, n_self_layers, ff_mult, dropout)
                for _ in range(n_cross_blocks)
            ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, kv: torch.Tensor, kv_mask: torch.Tensor) -> torch.Tensor:
        """
        kv:      (B, M, D) input tokens (already type-augmented by the tokenizer)
        kv_mask: (B, M) bool, True = real token
        Returns latents: (B, n_latents, D)
        """
        B = kv.shape[0]
        latents = self.latent[None].expand(B, -1, -1)
        if self.share:
            for _ in range(self.n_cross_blocks):
                latents = self.block(latents, kv, kv_mask)
        else:
            for block in self.blocks:
                latents = block(latents, kv, kv_mask)
        return self.final_norm(latents)
