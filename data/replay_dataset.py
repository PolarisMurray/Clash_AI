"""Offline replay dataset for the Clash Royale Decision Transformer.

A "trajectory" on disk is a single .npz file with fixed keys. Each row of the
arrays corresponds to one frame in chronological order:

    images       (N, 3, H, W)  uint8 or float in [0, 1]
    det_cls      (N, M)        int64    YOLO class ids per frame, padded with 0
    det_bbox     (N, M, 4)     float32  cx, cy, w, h in [0, 1]
    det_conf     (N, M)        float32  detection confidences
    det_side     (N, M)        int64    0=unknown, 1=ally, 2=enemy
    det_track    (N, M)        int64    optional, 0 if no tracker
    det_mask     (N, M)        bool     True for real detections
    cards        (N, n_hand)   int64    card-name ids; -1 = unknown
    elixir       (N,)          int64    0..n_elixir; -1 unknown clamped to 0
    action_card  (N,)          int64    n_hand = no-op, else slot in [0, n_hand)
    action_pos   (N, 2)        float32  (row_norm, col_norm) in [0, 1]
    reward       (N,)          float32  per-frame scalar reward
    timestep     (N,)          int64    absolute frame index inside the episode

A bare-bones builder (`build_random_trajectory`) is provided so unit tests and
the training script can run end-to-end without a real dataset on disk.

This module deliberately does **no OCR, no ResNet classification, and no
episode cutting**. It treats trajectories as already-segmented; segmentation
is the upstream collector's job.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from configs.policy_config import DataConfig, FullConfig, TokenizerConfig
from models.policy.agent import TrajectoryBatch


REQUIRED_KEYS = (
    "images", "det_cls", "det_bbox", "det_conf", "det_side", "det_track",
    "det_mask", "cards", "elixir", "action_card", "action_pos", "reward", "timestep",
)


@dataclass
class Trajectory:
    """One episode's worth of preprocessed frames + labels."""

    images: np.ndarray
    det_cls: np.ndarray
    det_bbox: np.ndarray
    det_conf: np.ndarray
    det_side: np.ndarray
    det_track: np.ndarray
    det_mask: np.ndarray
    cards: np.ndarray
    elixir: np.ndarray
    action_card: np.ndarray
    action_pos: np.ndarray
    reward: np.ndarray
    timestep: np.ndarray
    rtg: np.ndarray                          # computed from reward backwards

    @classmethod
    def from_npz(cls, path: Path) -> "Trajectory":
        with np.load(str(path), allow_pickle=False) as z:
            data = {k: z[k] for k in REQUIRED_KEYS}
        rtg = np.flip(np.cumsum(np.flip(data["reward"].astype(np.float32)))).copy()
        return cls(rtg=rtg, **data)

    def __len__(self) -> int:
        return self.images.shape[0]


def discover_trajectories(root: Path) -> List[Path]:
    root = Path(root)
    if not root.exists():
        return []
    if root.is_file():
        return [root]
    return sorted(root.rglob("*.npz"))


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr))


def _normalize_image(img: np.ndarray, size: Tuple[int, int]) -> torch.Tensor:
    """(3, H, W) at the configured size in [0, 1] float."""
    H, W = size
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    if img.shape[-2:] != (H, W):
        # Lazy resize via torch — keeps the dataset numpy-only at rest
        t = torch.from_numpy(img).unsqueeze(0)
        t = torch.nn.functional.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
        return t.squeeze(0).contiguous()
    return torch.from_numpy(img).contiguous()


class ReplayDataset(Dataset):
    """Sliding-window samples over one or more trajectories.

    For trajectories shorter than seq_len, the start of the window is
    left-padded with a repeat of the first frame (a common DT trick).
    """

    def __init__(
        self,
        trajectories: Sequence[Trajectory],
        cfg: FullConfig,
        split: str = "train",
    ):
        self.cfg = cfg
        self.tcfg: TokenizerConfig = cfg.tokenizer
        self.dcfg: DataConfig = cfg.data
        self.split = split
        if not trajectories:
            raise ValueError("ReplayDataset received no trajectories")
        self.trajectories = list(trajectories)
        self.seq_len = cfg.data.seq_len

        self._index: List[Tuple[int, int]] = []      # (traj_idx, end_frame) (inclusive)
        self._is_action: List[bool] = []
        for ti, traj in enumerate(self.trajectories):
            n = len(traj)
            for end in range(n):
                self._index.append((ti, end))
                self._is_action.append(traj.action_card[end] != self.tcfg.n_hand)

        self._sample_weights = self._compute_weights()

    def _compute_weights(self) -> np.ndarray:
        """Up-weight action frames + nearby frames so action transitions are sampled often."""
        is_action = np.asarray(self._is_action, dtype=np.float32)
        if not self.dcfg.weighted_sampling:
            return np.ones_like(is_action)
        n_action = is_action.sum()
        if n_action == 0:
            return np.ones_like(is_action)
        action_ratio = max(n_action / len(is_action), 1e-3)
        weights = is_action / action_ratio + (1.0 - is_action) / max(1.0 - action_ratio, 1e-3)
        # Spread weight to frames just before each action so the model gets buildup context.
        if self.dcfg.action_focus_window > 0:
            base = weights.copy()
            for idx in np.where(is_action > 0)[0]:
                lo = max(0, idx - self.dcfg.action_focus_window)
                for j in range(lo, idx):
                    alpha = 1.0 / (idx - j + 1)
                    weights[j] = max(weights[j], alpha / action_ratio)
            weights = np.maximum(weights, base * 0.5)
        return weights

    def sampler(self) -> WeightedRandomSampler:
        return WeightedRandomSampler(self._sample_weights.tolist(), num_samples=len(self), replacement=True)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        traj_idx, end = self._index[idx]
        traj = self.trajectories[traj_idx]
        L = self.seq_len
        start = end - L + 1
        # Indices into the trajectory; repeat the first frame for left padding
        idxs = np.arange(start, end + 1)
        idxs = np.clip(idxs, 0, len(traj) - 1)

        sample = self._gather(traj, idxs)
        if self.dcfg.lr_flip and torch.rand(1).item() < 0.5:
            sample = self._flip(sample)
        if self.dcfg.card_shuffle and torch.rand(1).item() < 0.5:
            sample = self._shuffle_hand(sample)
        return sample

    def _gather(self, traj: Trajectory, idxs: np.ndarray):
        H, W = self.tcfg.image_size
        images = np.stack([_normalize_image(traj.images[i], (H, W)).numpy() for i in idxs], axis=0)

        cards = traj.cards[idxs].copy()
        cards[cards < 0] = 0  # 0 is the padding token in the embedding
        elixir = traj.elixir[idxs].copy()
        elixir[elixir < 0] = 0

        # Previous action / reward
        prev_idxs = np.maximum(idxs - 1, 0)
        prev_a_card = traj.action_card[prev_idxs].copy()
        prev_a_pos = traj.action_pos[prev_idxs].copy()
        prev_r = traj.reward[prev_idxs].copy()

        rtg = traj.rtg[idxs].astype(np.float32) / max(self.dcfg.rtg_scale, 1e-6)

        action_card = traj.action_card[idxs].copy()
        action_pos = traj.action_pos[idxs].copy()
        # delay = number of frames until the next action; -1 if none in window
        delay = np.full(len(idxs), self.tcfg.max_delay, dtype=np.int64)
        for k, i in enumerate(idxs):
            for j in range(i, min(len(traj), i + self.tcfg.max_delay + 1)):
                if traj.action_card[j] != self.tcfg.n_hand:
                    delay[k] = j - i
                    break
        action_mask = (action_card != self.tcfg.n_hand) | (delay < self.tcfg.max_delay)

        # Convert continuous (row, col) targets into a flat grid index (used in grid mode)
        pos_idx = (np.clip(action_pos[:, 0], 0, 0.999) * self.tcfg.grid_rows).astype(np.int64) * self.tcfg.grid_cols
        pos_idx += (np.clip(action_pos[:, 1], 0, 0.999) * self.tcfg.grid_cols).astype(np.int64)

        return {
            "images": _to_tensor(images).float(),
            "det_cls": _to_tensor(traj.det_cls[idxs].astype(np.int64)),
            "det_bbox": _to_tensor(traj.det_bbox[idxs].astype(np.float32)),
            "det_conf": _to_tensor(traj.det_conf[idxs].astype(np.float32)),
            "det_side": _to_tensor(traj.det_side[idxs].astype(np.int64)),
            "det_track": _to_tensor(traj.det_track[idxs].astype(np.int64)),
            "det_mask": _to_tensor(traj.det_mask[idxs].astype(np.bool_)),
            "cards": _to_tensor(cards.astype(np.int64)),
            "elixir": _to_tensor(elixir.astype(np.int64)),
            "rtg": _to_tensor(rtg),
            "prev_a_card": _to_tensor(prev_a_card.astype(np.int64)),
            "prev_a_pos": _to_tensor(prev_a_pos.astype(np.float32)),
            "prev_r": _to_tensor(prev_r.astype(np.float32)),
            "timesteps": _to_tensor(traj.timestep[idxs].astype(np.int64)),
            "target_select": _to_tensor(action_card.astype(np.int64)),
            "target_pos_idx": _to_tensor(pos_idx.astype(np.int64)),
            "target_pos_xy": _to_tensor(action_pos.astype(np.float32)),
            "target_delay": _to_tensor(delay.astype(np.int64)),
            "action_mask": _to_tensor(action_mask.astype(np.bool_)),
        }

    def _flip(self, sample):
        sample["images"] = torch.flip(sample["images"], dims=[-1])
        bbox = sample["det_bbox"].clone()
        bbox[..., 0] = 1.0 - bbox[..., 0]                         # mirror cx
        sample["det_bbox"] = bbox
        sample["target_pos_xy"][..., 1] = 1.0 - sample["target_pos_xy"][..., 1]
        # Recompute pos_idx to match flipped col
        pos = sample["target_pos_xy"]
        rows = self.tcfg.grid_rows
        cols = self.tcfg.grid_cols
        pos_idx = (pos[..., 0].clamp(0, 0.999) * rows).long() * cols + (pos[..., 1].clamp(0, 0.999) * cols).long()
        sample["target_pos_idx"] = pos_idx
        sample["prev_a_pos"][..., 1] = 1.0 - sample["prev_a_pos"][..., 1]
        return sample

    def _shuffle_hand(self, sample):
        n_hand = self.tcfg.n_hand
        perm = torch.randperm(n_hand)
        # cards: (T, n_hand)
        sample["cards"] = sample["cards"][:, perm]
        # remap target_select if not no-op
        inv = torch.argsort(perm)
        sel = sample["target_select"]
        no_op = sel == n_hand
        new_sel = torch.where(no_op, sel, inv[sel.clamp_max(n_hand - 1)])
        sample["target_select"] = new_sel
        prev = sample["prev_a_card"]
        prev_no_op = prev == 0                                    # 0 is padding/no-op for prev embedding
        new_prev = torch.where(prev_no_op, prev, inv[(prev - 1).clamp_min(0)] + 1)
        sample["prev_a_card"] = new_prev
        return sample


def collate(batch: List[dict]) -> Tuple[TrajectoryBatch, dict]:
    """Stack list-of-dicts into a TrajectoryBatch + targets dict."""
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}
    traj = TrajectoryBatch(
        images=out["images"],
        det_cls=out["det_cls"],
        det_bbox=out["det_bbox"],
        det_conf=out["det_conf"],
        det_side=out["det_side"],
        det_track=out["det_track"],
        det_mask=out["det_mask"],
        cards=out["cards"],
        elixir=out["elixir"],
        rtg=out["rtg"],
        prev_a_card=out["prev_a_card"],
        prev_a_pos=out["prev_a_pos"],
        prev_r=out["prev_r"],
        timesteps=out["timesteps"],
    )
    targets = {
        "select": out["target_select"],
        "pos_idx": out["target_pos_idx"],
        "pos_xy": out["target_pos_xy"],
        "delay": out["target_delay"],
        "action_mask": out["action_mask"],
    }
    return traj, targets


def build_dataloader(cfg: FullConfig, trajectories: Iterable[Trajectory], split: str = "train") -> DataLoader:
    ds = ReplayDataset(list(trajectories), cfg=cfg, split=split)
    sampler = ds.sampler() if (split == "train" and cfg.data.weighted_sampling) else None
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        sampler=sampler,
        shuffle=(sampler is None and split == "train"),
        num_workers=cfg.data.num_workers,
        collate_fn=collate,
        drop_last=(split == "train"),
        persistent_workers=(cfg.data.num_workers > 0),
    )


# ----------------------------- synthetic helpers ------------------------------ #

def build_random_trajectory(cfg: FullConfig, n_frames: int, seed: int = 0) -> Trajectory:
    """Random trajectory used only for shape-checks and quick smoke tests."""
    rng = np.random.default_rng(seed)
    tk = cfg.tokenizer
    H, W = tk.image_size
    M = tk.max_objects
    images = rng.integers(0, 255, size=(n_frames, 3, H, W), dtype=np.uint8)
    det_cls = rng.integers(0, tk.n_classes, size=(n_frames, M), dtype=np.int64)
    det_bbox = rng.random((n_frames, M, 4), dtype=np.float32)
    det_conf = rng.random((n_frames, M), dtype=np.float32)
    det_side = rng.integers(0, 3, size=(n_frames, M), dtype=np.int64)
    det_track = rng.integers(0, tk.max_track_ids, size=(n_frames, M), dtype=np.int64)
    det_mask = rng.random((n_frames, M)) < 0.5
    cards = rng.integers(0, tk.n_cards, size=(n_frames, tk.n_hand), dtype=np.int64)
    elixir = rng.integers(0, tk.n_elixir, size=(n_frames,), dtype=np.int64)
    action_card = np.full(n_frames, tk.n_hand, dtype=np.int64)                  # all no-op by default
    action_idxs = rng.choice(n_frames, size=max(1, n_frames // 8), replace=False)
    action_card[action_idxs] = rng.integers(0, tk.n_hand, size=action_idxs.size, dtype=np.int64)
    action_pos = rng.random((n_frames, 2), dtype=np.float32)
    reward = rng.normal(0.0, 0.1, size=n_frames).astype(np.float32)
    reward[action_idxs] += rng.normal(0.5, 0.2, size=action_idxs.size).astype(np.float32)
    timestep = np.arange(n_frames, dtype=np.int64)

    rtg = np.flip(np.cumsum(np.flip(reward))).copy()
    return Trajectory(
        images=images, det_cls=det_cls, det_bbox=det_bbox, det_conf=det_conf,
        det_side=det_side, det_track=det_track, det_mask=det_mask,
        cards=cards, elixir=elixir, action_card=action_card, action_pos=action_pos,
        reward=reward, timestep=timestep, rtg=rtg,
    )


if __name__ == "__main__":
    from configs.policy_config import default_config
    cfg = default_config(seq_len=8, batch_size=2, num_workers=0)
    cfg.data.seq_len = 8
    cfg.model.seq_len = 8
    cfg.__post_init__()
    trajs = [build_random_trajectory(cfg, 24, seed=i) for i in range(2)]
    loader = build_dataloader(cfg, trajs, split="train")
    batch, targets = next(iter(loader))
    print("images:", batch.images.shape)
    print("targets.select:", targets["select"].shape)
