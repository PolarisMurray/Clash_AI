"""Policy output heads: card selection, deployment position, optional delay.

Loss helper combines the active heads with configurable weights and supports
masking out timesteps where no human action is available (action_mask).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HeadOutput:
    select_logits: torch.Tensor                  # (B, T, n_hand + 1)   last index = no-op
    pos_logits: Optional[torch.Tensor]            # (B, T, R*C) if pos_mode == "grid"
    pos_xy: Optional[torch.Tensor]                # (B, T, 2) in [0,1] if pos_mode == "xy"
    delay_logits: Optional[torch.Tensor]          # (B, T, max_delay + 1) if enable_delay


class PolicyHeads(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_hand: int,
        grid_rows: int,
        grid_cols: int,
        max_delay: int,
        pos_mode: str = "grid",
        enable_delay: bool = True,
    ):
        super().__init__()
        assert pos_mode in ("grid", "xy")
        self.pos_mode = pos_mode
        self.enable_delay = enable_delay
        self.n_hand = n_hand
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.max_delay = max_delay

        self.select_head = nn.Linear(d_model, n_hand + 1)        # +1 = no-op
        if pos_mode == "grid":
            self.pos_head = nn.Linear(d_model, grid_rows * grid_cols)
        else:
            self.pos_head = nn.Linear(d_model, 2)
        if enable_delay:
            self.delay_head = nn.Linear(d_model, max_delay + 1)
        else:
            self.delay_head = None

    def forward(self, h: torch.Tensor) -> HeadOutput:
        # h: (B, T, D)
        select = self.select_head(h)
        if self.pos_mode == "grid":
            pos_logits = self.pos_head(h)
            pos_xy = None
        else:
            pos_logits = None
            pos_xy = torch.sigmoid(self.pos_head(h))
        delay = self.delay_head(h) if self.delay_head is not None else None
        return HeadOutput(select_logits=select, pos_logits=pos_logits, pos_xy=pos_xy, delay_logits=delay)


def policy_loss(
    out: HeadOutput,
    targets: Dict[str, torch.Tensor],
    action_mask: torch.Tensor,
    loss_w: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    """
    targets keys:
      - 'select'  (B, T) long   value in [0, n_hand]   (n_hand = no-op)
      - 'pos_idx' (B, T) long   value in [0, R*C)       (used in grid mode)
      - 'pos_xy'  (B, T, 2) float                       (used in xy mode)
      - 'delay'   (B, T) long   value in [0, max_delay]
    action_mask (B, T) bool — True = timestep contributes to loss.
    loss_w: {'select': float, 'pos': float, 'delay': float}
    Returns dict with 'loss', 'loss_select', 'loss_pos', 'loss_delay', plus accuracies.
    """
    metrics: Dict[str, torch.Tensor] = {}
    B, T = action_mask.shape
    flat_mask = action_mask.reshape(-1)
    n = flat_mask.sum().clamp_min(1).float()

    # ---- select ----
    sel_logits = out.select_logits.reshape(-1, out.select_logits.shape[-1])
    sel_target = targets["select"].reshape(-1)
    loss_select_per = F.cross_entropy(sel_logits, sel_target, reduction="none")
    loss_select = (loss_select_per * flat_mask).sum() / n
    sel_pred = sel_logits.argmax(-1)
    acc_select = ((sel_pred == sel_target).float() * flat_mask).sum() / n
    metrics["loss_select"] = loss_select
    metrics["acc_select"] = acc_select

    # ---- position ----
    if out.pos_logits is not None:
        pos_logits = out.pos_logits.reshape(-1, out.pos_logits.shape[-1])
        pos_target = targets["pos_idx"].reshape(-1)
        loss_pos_per = F.cross_entropy(pos_logits, pos_target, reduction="none")
        loss_pos = (loss_pos_per * flat_mask).sum() / n
        pos_pred = pos_logits.argmax(-1)
        acc_pos = ((pos_pred == pos_target).float() * flat_mask).sum() / n
    else:
        pos_pred_xy = out.pos_xy.reshape(-1, 2)
        pos_target_xy = targets["pos_xy"].reshape(-1, 2)
        loss_pos_per = F.mse_loss(pos_pred_xy, pos_target_xy, reduction="none").mean(-1)
        loss_pos = (loss_pos_per * flat_mask).sum() / n
        acc_pos = torch.tensor(0.0, device=loss_pos.device)
    metrics["loss_pos"] = loss_pos
    metrics["acc_pos"] = acc_pos

    # ---- delay ----
    if out.delay_logits is not None:
        delay_logits = out.delay_logits.reshape(-1, out.delay_logits.shape[-1])
        delay_target = targets["delay"].reshape(-1)
        loss_delay_per = F.cross_entropy(delay_logits, delay_target, reduction="none")
        loss_delay = (loss_delay_per * flat_mask).sum() / n
        delay_pred = delay_logits.argmax(-1)
        acc_delay = ((delay_pred == delay_target).float() * flat_mask).sum() / n
    else:
        loss_delay = torch.tensor(0.0, device=loss_select.device)
        acc_delay = torch.tensor(0.0, device=loss_select.device)
    metrics["loss_delay"] = loss_delay
    metrics["acc_delay"] = acc_delay

    total = (
        loss_w.get("select", 1.0) * loss_select
        + loss_w.get("pos", 1.0) * loss_pos
        + loss_w.get("delay", 0.5) * loss_delay
    )
    metrics["loss"] = total
    return metrics
