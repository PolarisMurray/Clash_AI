"""Visualize the offline replay dataset: action timing, RTG curve, weighted sampling.

Produces:
    06_replay_dataset.png    4-panel summary using a synthetic trajectory
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.policy_config import default_config
from data.replay_dataset import ReplayDataset, build_random_trajectory


def main():
    out = ROOT / "outputs/visualizations/06_replay_dataset.png"
    cfg = default_config(seq_len=30, batch_size=4, num_workers=0)
    cfg.__post_init__()

    n_frames = 240
    trajs = [build_random_trajectory(cfg, n_frames, seed=k) for k in range(3)]
    ds = ReplayDataset(trajs, cfg, split="train")

    is_action = np.asarray(ds._is_action, dtype=np.int32)
    weights = ds._sample_weights
    print(f"[ds] {len(ds)} samples, {is_action.sum()} action frames, "
          f"action ratio {is_action.mean():.2%}")

    # Sample a batch and grab the RTG curve from one trajectory
    rtg = trajs[0].rtg
    rew = trajs[0].reward
    action_idx = np.where(trajs[0].action_card != cfg.tokenizer.n_hand)[0]

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.32)

    # 1) reward + RTG over time for one trajectory
    ax = fig.add_subplot(gs[0, 0:2])
    ax.plot(rew, color="#94a3b8", linewidth=1.0, label="per-frame reward", alpha=0.6)
    ax.plot(rtg, color="#ef4444", linewidth=2.0, label="return-to-go (RTG)")
    for k, idx in enumerate(action_idx):
        ax.axvline(idx, color="#f97316", linewidth=0.8, alpha=0.4,
                   label="action frame" if k == 0 else None)
    ax.set_title(f"Trajectory 0  ·  reward / RTG / actions  ({n_frames} frames)")
    ax.set_xlabel("frame"); ax.set_ylabel("value"); ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # 2) action distribution
    ax = fig.add_subplot(gs[0, 2])
    counts = np.bincount(np.concatenate([t.action_card for t in trajs]),
                         minlength=cfg.tokenizer.n_hand + 1)
    labels = [f"slot {i}" for i in range(cfg.tokenizer.n_hand)] + ["no-op"]
    bars = ax.bar(labels, counts,
                  color=["#a855f7"] * cfg.tokenizer.n_hand + ["#94a3b8"])
    ax.set_title("Action histogram  (across all 3 trajectories)")
    ax.set_ylabel("# frames"); ax.tick_params(axis="x", labelsize=9)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + counts.max() * 0.01,
                str(c), ha="center", fontsize=9)

    # 3) sample weights (with action frames marked)
    ax = fig.add_subplot(gs[1, 0:2])
    ax.plot(weights, color="#0ea5e9", linewidth=1.0)
    for k, idx in enumerate(np.where(is_action == 1)[0]):
        ax.axvline(idx, color="#f97316", linewidth=0.6, alpha=0.6,
                   label="action frame" if k == 0 else None)
    ax.set_title("Per-sample weight (action-frame focused)  ·  WeightedRandomSampler")
    ax.set_xlabel("flat sample index across 3 trajectories")
    ax.set_ylabel("weight"); ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # 4) RTG distribution across all trajectories
    ax = fig.add_subplot(gs[1, 2])
    all_rtg = np.concatenate([t.rtg for t in trajs])
    ax.hist(all_rtg, bins=40, color="#ef4444", alpha=0.85, edgecolor="white")
    ax.set_title(f"RTG distribution  (n={len(all_rtg)})")
    ax.set_xlabel("return-to-go"); ax.set_ylabel("# frames"); ax.grid(alpha=0.3)

    fig.suptitle("Offline replay dataset — synthetic trajectories",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.text(0.5, 0.005,
             f"3 random trajectories · {n_frames} frames each · "
             f"weighted_sampling={cfg.data.weighted_sampling} · seq_len={cfg.data.seq_len}",
             ha="center", fontsize=10, color="#475569", style="italic")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[ds] wrote {out}")


if __name__ == "__main__":
    main()
