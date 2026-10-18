# Clash Royale AI — Perceiver-DT

A **screen-only** Clash Royale agent. It observes raw gameplay frames, detects
units with a fine-tuned YOLO26 detector, fuses everything into multimodal
tokens, compresses each frame with a Perceiver IO encoder, and predicts the
next action with a Decision Transformer trained offline from replays.

```
                    ┌─────────────────────────────────────────────────────┐
   raw frame ──►    │  1. YOLO26 (fine-tuned, 155 Clash Royale classes)   │
                    │     bbox / class / conf / side / track              │
                    │                                                     │
                    │  2. FrameTokenizer                                  │
                    │     image patches │ object │ cards │ elixir │ rtg  │
                    │     prev_action  │ prev_reward                     │
                    │                                                     │
                    │  3. Perceiver IO encoder                            │
                    │     latents (K, D)  ← fixed-size compression        │
                    │                                                     │
                    │  4. Decision Transformer (causal, T timesteps)      │
                    │     reads through one action slot per timestep     │
                    │                                                     │
                    │  5. Policy heads                                    │
                    │     • select_head   → which card / no-op            │
                    │     • pos_head      → 36×64 grid (or xy regression) │
                    │     • delay_head    → categorical delay (optional) │
                    └─────────────────────────────────────────────────────┘
```

---

## Layout

```
configs/policy_config.py         one config tree shared by all stages
data/replay_dataset.py           .npz trajectory loader, sampler, frames→trajectory builder
perception/yolo_detector.py      YOLO wrapper (frame → structured detections)
perception/tokenizer.py          multimodal frame tokenizer
models/detection/
  yolo26s.pt                     COCO-pretrained YOLO26s — base for fine-tuning only
  clash_royale_yolo26s.pt        fine-tuned on the 155 Clash Royale classes (used by default)
models/policy/
  perceiver_encoder.py           Perceiver IO latent bottleneck
  decision_transformer.py        causal Transformer over latents + action slot
  policy_heads.py                select / pos / delay heads + policy_loss()
  agent.py                       ClashRoyaleAgent (tokenizer→encoder→DT→heads)
training/train_policy.py         policy training loop (cosine warmup, TensorBoard / wandb)
training/detection/train.py      YOLO fine-tuning on training/detection/prepared_dataset
scripts/visualize_*.py           one script per figure in outputs/visualizations
tests/test_pipeline.py           regression tests
```

---

## Quick start

### 1. Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Tests and smoke run (no data needed)

```bash
python -m pytest -q tests
python -m training.train_policy --smoke --out outputs/policy_runs/smoke
```

The smoke run trains a scaled-down model (5.4M params) on synthetic
trajectories for one epoch in about 10 seconds and writes
`outputs/policy_runs/smoke/ckpt/epoch_000.pt`.

### 3. Detect units on a screenshot

```bash
python -m perception.yolo_detector \
    --image training/detection/prepared_dataset/yolo_detection/images/val/OYASSU_20230212_episodes/4/02385.jpg
```

### 4. Fine-tune the detector (optional)

```bash
python training/detection/train.py --skip-prepare --epochs 30 --batch 16 --device mps
cp training/detection/runs/cr_yolo26s/weights/best.pt models/detection/clash_royale_yolo26s.pt
```

`--time 1.0` limits training to one hour instead of a fixed epoch count.
The prepared dataset (5,551 train / 1,388 val frames, 155 classes) is in the
repo; `source_dataset/` is only needed to rebuild it.

### 5. Train the policy on real trajectories

Put `.npz` trajectories (schema below) under a folder, then:

```bash
python -m training.train_policy --data outputs/replay_dataset --epochs 10 --batch-size 16 --seq-len 30
```

TensorBoard logs land in `outputs/policy_runs/perceiver_dt/tb`.

### 6. Figures

```bash
python -m scripts.visualize_yolo          # 02_yolo_grid.png, 03_yolo_stats.png, yolo_overlays/
python -m scripts.visualize_training      # 04_training_curves.png from outputs/policy_runs/demo/tb
python -m scripts.visualize_pipeline      # 05_pipeline_shapes.png (real frames → action)
python -m scripts.visualize_dataset       # 06_replay_dataset.png
```

---

## Trajectory format (.npz)

`data.replay_dataset.Trajectory` reads .npz files with a fixed schema:

| key            | shape           | dtype   | meaning                                     |
| -------------- | --------------- | ------- | ------------------------------------------- |
| `images`       | (N, 3, H, W)    | uint8   | frame stream (any size — auto-resized)      |
| `det_cls`      | (N, M)          | int64   | YOLO class id per slot, 0 if unused          |
| `det_bbox`     | (N, M, 4)       | float32 | normalized cx, cy, w, h                      |
| `det_conf`     | (N, M)          | float32 | detection confidence                         |
| `det_side`     | (N, M)          | int64   | 0=unknown, 1=ally, 2=enemy                   |
| `det_track`    | (N, M)          | int64   | tracker id, 0 if no tracker                  |
| `det_mask`     | (N, M)          | bool    | True = real detection (not padding)          |
| `cards`        | (N, n_hand)     | int64   | card-name ids per slot, -1 = unknown; last slot = next card |
| `elixir`       | (N,)            | int64   | 0..10, -1 unknown                            |
| `action_card`  | (N,)            | int64   | slot in [0, n_hand); n_hand = no-op          |
| `action_pos`   | (N, 2)          | float32 | (row_norm, col_norm) in [0, 1]               |
| `reward`       | (N,)            | float32 | per-frame scalar reward                      |
| `timestep`     | (N,)            | int64   | absolute frame index in the episode          |

Return-to-go is computed at load time (`np.flip(np.cumsum(np.flip(reward)))`).
Windows that start before frame 0 are left-padded by repeating the first
frame; padded steps are excluded from the loss.

`trajectory_from_frames(frames, detections, cfg)` turns consecutive frames
plus their YOLO output into a `Trajectory` (cards / elixir / actions left as
unknown / no-op for you to fill in), and `Trajectory.to_npz(path)` saves it.
`build_random_trajectory(cfg, n_frames)` generates a synthetic sample.

---

## Config

A single `FullConfig` bundles four sub-configs:

| sub        | covers                                                  |
| ---------- | ------------------------------------------------------- |
| tokenizer  | image size / patch size / vocab sizes / grid / max objs |
| model      | DT depth / heads + nested PerceiverConfig + heads cfg  |
| data       | seq_len, weighted sampling, augmentations               |
| train      | optimizer / scheduler / logging / checkpointing         |

`default_config(seq_len=30, batch_size=16, lr=6e-4, ...)` accepts flat
overrides and routes them by attribute name. Same key on multiple subs
(e.g. `seq_len`) is set on **all** of them.

---

## Inference

```python
from models.policy.agent import ClashRoyaleAgent
agent = ClashRoyaleAgent.from_checkpoint("outputs/policy_runs/demo/ckpt.pt")
action = agent.predict_last(batch)   # {"select", "pos" (row, col), "delay"}
```

Build `batch` from the last `seq_len` frames with `YOLODetector`,
`trajectory_from_frames`, `ReplayDataset` and `collate`;
`scripts/visualize_pipeline.py` does exactly this on real frames.

---

## License

MIT, same as upstream. See [LICENSE](LICENSE).
