"""Extract scalars from a TensorBoard run and render them as a single PNG dashboard.

Usage: python -m scripts.visualize_training --logs outputs/policy_runs/demo/tb --out outputs/visualizations/04_training_curves.png
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_scalars(log_dir: Path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    for tag in acc.Tags().get("scalars", []):
        events = acc.Scalars(tag)
        out[tag] = (np.array([e.step for e in events]),
                    np.array([e.value for e in events]))
    return out


def smooth(y, w=10):
    if len(y) < w: return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, default=Path("outputs/policy_runs/demo/tb"))
    ap.add_argument("--out", type=Path, default=Path("outputs/visualizations/04_training_curves.png"))
    args = ap.parse_args()

    scalars = read_scalars(args.logs)
    if not scalars:
        raise SystemExit(f"no scalars under {args.logs}")
    print(f"[tb] {len(scalars)} tags from {args.logs}")
    for k in scalars: print(f"  {k}: {len(scalars[k][0])} pts")

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.32)

    def panel(ax, tag, color, title=None, smooth_w=10):
        if tag not in scalars:
            ax.set_axis_off(); return
        x, y = scalars[tag]
        ax.plot(x, y, color=color, alpha=0.25, linewidth=0.9, label="raw")
        ys = smooth(y, smooth_w)
        if len(ys) > 0:
            ax.plot(x[-len(ys):], ys, color=color, linewidth=2.0, label=f"smooth(w={smooth_w})")
        ax.set_title(title or tag, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        ax.set_xlabel("step", fontsize=9)

    panel(fig.add_subplot(gs[0, 0]), "train/loss", "#0f172a", "Total loss")
    panel(fig.add_subplot(gs[0, 1]), "train/loss_select", "#f97316", "Select loss")
    panel(fig.add_subplot(gs[0, 2]), "train/loss_pos", "#3b82f6", "Position loss")

    panel(fig.add_subplot(gs[1, 0]), "train/loss_delay", "#ef4444", "Delay loss")
    panel(fig.add_subplot(gs[1, 1]), "train/acc_select", "#16a34a", "Select acc")
    panel(fig.add_subplot(gs[1, 2]), "train/acc_pos", "#0ea5e9", "Position acc")

    panel(fig.add_subplot(gs[2, 0]), "train/acc_delay", "#a855f7", "Delay acc")
    panel(fig.add_subplot(gs[2, 1]), "train/lr", "#0f766e", "Learning rate")

    # 9th panel: stacked overview
    ax = fig.add_subplot(gs[2, 2])
    for tag, c in [("train/loss_select", "#f97316"),
                   ("train/loss_pos", "#3b82f6"),
                   ("train/loss_delay", "#ef4444")]:
        if tag in scalars:
            x, y = scalars[tag]
            ys = smooth(y, 10)
            ax.plot(x[-len(ys):], ys, color=c, linewidth=1.8, label=tag.split("/")[-1])
    ax.set_title("Component losses (smoothed)")
    ax.set_xlabel("step", fontsize=9); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle("Perceiver-DT training curves — synthetic 5-epoch demo run",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.text(0.5, 0.005,
             "Synthetic random trajectories · model scaled down to ~5.4M params · 640 steps · seq_len=4 · batch=4 · device=MPS",
             ha="center", fontsize=10, color="#475569", style="italic")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[tb] wrote {args.out}")


if __name__ == "__main__":
    main()
