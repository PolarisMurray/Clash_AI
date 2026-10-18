"""Regression tests for the policy pipeline. Run: python -m pytest -q tests"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.policy_config import FullConfig, default_config
from data.replay_dataset import ReplayDataset, Trajectory, build_dataloader, build_random_trajectory
from models.policy.agent import ClashRoyaleAgent
from models.policy.policy_heads import HeadOutput, policy_loss
from perception.tokenizer import FrameTokenizer


def tiny_config(**overrides) -> FullConfig:
    cfg = default_config(**{**dict(seq_len=4, batch_size=2, num_workers=0, lr_flip=False, card_shuffle=False), **overrides})
    cfg.tokenizer.image_size = (32, 32)
    cfg.tokenizer.max_objects = 4
    cfg.model.n_layers = 1
    cfg.model.perceiver.n_latents = 8
    cfg.model.perceiver.n_self_layers = 1
    cfg.model.perceiver.n_cross_blocks = 1
    return cfg


def noop_trajectory(cfg: FullConfig, n: int, seed: int = 0) -> Trajectory:
    traj = build_random_trajectory(cfg, n, seed=seed)
    traj.action_card[:] = cfg.tokenizer.n_hand
    return traj


def test_forward_backward_and_checkpoint_roundtrip(tmp_path):
    cfg = tiny_config()
    batch, targets = next(iter(build_dataloader(cfg, [build_random_trajectory(cfg, 16)])))
    agent = ClashRoyaleAgent(cfg)
    action_mask = targets.pop("action_mask")
    metrics = policy_loss(agent(batch), targets, action_mask, {})
    metrics["loss"].backward()
    assert torch.isfinite(metrics["loss"])

    path = tmp_path / "ckpt.pt"
    torch.save({"state_dict": agent.state_dict(), "config": cfg.to_dict()}, path)
    loaded = ClashRoyaleAgent.from_checkpoint(path)
    agent.eval()
    with torch.no_grad():
        assert torch.allclose(agent(batch).select_logits, loaded(batch).select_logits)


def test_first_frame_has_no_previous_action():
    cfg = tiny_config()
    traj = noop_trajectory(cfg, 12)
    traj.action_card[0] = 2
    sample = ReplayDataset([traj], cfg)[0]
    assert sample["prev_a_card"][-1] == 0
    assert sample["prev_a_pos"][-1].abs().sum() == 0
    assert sample["target_select"][-1] == 2


def test_previous_action_encoding():
    cfg = tiny_config()
    traj = noop_trajectory(cfg, 12)
    traj.action_card[5] = 0
    ds = ReplayDataset([traj], cfg)
    assert ds[6]["prev_a_card"][-1] == 1          # slot 0 -> 1
    assert ds[8]["prev_a_card"][-1] == 0          # no-op -> 0
    assert ds[8]["prev_a_pos"][-1].abs().sum() == 0


def test_left_padding_is_masked_out_of_the_loss():
    cfg = tiny_config()
    traj = noop_trajectory(cfg, 12)
    traj.action_card[0] = 1
    mask = ReplayDataset([traj], cfg)[0]["action_mask"]
    assert mask.tolist() == [False, False, False, True]


def test_position_loss_ignores_noop_targets():
    n_hand = 5
    out = HeadOutput(select_logits=torch.zeros(1, 2, n_hand + 1), pos_logits=torch.randn(1, 2, 4),
                     pos_xy=None, delay_logits=None)
    base = {"select": torch.tensor([[n_hand, 1]]), "pos_xy": torch.zeros(1, 2, 2),
            "delay": torch.zeros(1, 2, dtype=torch.long)}
    mask = torch.ones(1, 2, dtype=torch.bool)
    a = policy_loss(out, dict(base, pos_idx=torch.tensor([[3, 0]])), mask, {})["loss_pos"]
    b = policy_loss(out, dict(base, pos_idx=torch.tensor([[0, 0]])), mask, {})["loss_pos"]
    assert torch.equal(a, b)


def test_card_zero_is_not_the_unknown_card():
    cfg = tiny_config()
    tk = cfg.tokenizer
    tok = FrameTokenizer(32, tk.image_size, tk.patch_size, tk.n_classes, tk.max_objects,
                         tk.n_cards, tk.n_hand, tk.n_elixir, tk.max_track_ids)
    B = 2
    args = dict(
        images=torch.zeros(B, 3, *tk.image_size), det_cls=torch.zeros(B, tk.max_objects, dtype=torch.long),
        det_bbox=torch.zeros(B, tk.max_objects, 4), det_conf=torch.zeros(B, tk.max_objects),
        det_side=torch.zeros(B, tk.max_objects, dtype=torch.long),
        det_track=torch.tensor([[0, 65, 300, 9999]] * B), det_mask=torch.ones(B, tk.max_objects, dtype=torch.bool),
        elixir=torch.zeros(B, dtype=torch.long), rtg=torch.zeros(B), prev_a_card=torch.zeros(B, dtype=torch.long),
        prev_a_pos=torch.zeros(B, 2), prev_r=torch.zeros(B),
    )
    cards = torch.tensor([[0] * tk.n_hand, [-1] * tk.n_hand])
    ft = tok(cards=cards, **args)
    n_patch = tok.patch.n_patches + tk.max_objects
    card_tokens = ft.tokens[:, n_patch:n_patch + tk.n_hand]
    assert not torch.allclose(card_tokens[0], card_tokens[1])
    assert ft.mask[0, n_patch:n_patch + tk.n_hand].all() and not ft.mask[1, n_patch:n_patch + tk.n_hand].any()
    obj = ft.tokens[0, tok.patch.n_patches:n_patch]
    assert len({tuple(t.round(decimals=4).tolist()) for t in obj[1:]}) == 3   # distinct track ids stay distinct


def test_card_shuffle_keeps_next_card_and_selected_card():
    cfg = tiny_config(card_shuffle=True)
    n_hand = cfg.tokenizer.n_hand
    traj = noop_trajectory(cfg, 8)
    traj.cards[:] = np.arange(10, 10 + n_hand)
    traj.action_card[3] = 2
    ds = ReplayDataset([traj], cfg)
    torch.manual_seed(0)
    for _ in range(50):
        s = ds._shuffle_hand(ds._gather(traj, np.arange(4), np.zeros(4, dtype=bool)))
        assert s["cards"][0, -1] == 10 + n_hand - 1
        assert s["cards"][3, s["target_select"][3]] == 12


def test_sampler_focus_window_stays_inside_trajectory():
    cfg = tiny_config()
    a, b = noop_trajectory(cfg, 20, seed=1), noop_trajectory(cfg, 20, seed=2)
    b.action_card[0] = 1
    w = ReplayDataset([a, b], cfg)._sample_weights
    assert np.allclose(w[:20], w[0])


def test_yolo_detector_rejects_mismatched_class_names():
    pytest.importorskip("ultralytics")
    from perception.yolo_detector import YOLODetector
    coco = ROOT / "models/detection/yolo26s.pt"
    if not coco.exists():
        pytest.skip("COCO base weights not present")
    with pytest.raises(ValueError, match="not trained on this label set"):
        YOLODetector(coco, class_names=[f"c{i}" for i in range(155)])
