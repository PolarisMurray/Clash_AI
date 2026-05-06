# Clash Royale AI — Perceiver-DT (refactored)

A **screen-only** Clash Royale agent. Observes raw gameplay frames, detects
units with YOLO, fuses everything into multimodal tokens, compresses with a
Perceiver IO encoder, and predicts the next action with a Decision Transformer.

This is the simplified pipeline that replaces the OCR-coupled, episode-cutting,
ResNet-classifier-driven legacy stack.

```
                    ┌─────────────────────────────────────────────────────┐
   raw frame ──►    │  1. YOLOv8                                          │
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

No spatial-then-temporal hard-coding. The Perceiver compresses each frame's
multimodal cloud; the Decision Transformer attends across time. Spatial,
temporal, object, and card relationships are all learned jointly.

---

## Layout (new files only)

```
configs/
  policy_config.py           # one config tree shared by all stages
data/
  replay_dataset.py          # offline-RL .npz trajectory loader + sampler
perception/
  yolo_detector.py           # thin YOLOv8 wrapper (no OCR / no episode cut)
  tokenizer.py               # multimodal frame tokenizer (PyTorch)
models/policy/
  perceiver_encoder.py       # Perceiver IO latent bottleneck
  decision_transformer.py    # causal Transformer over latents + action slot
  policy_heads.py            # select / pos / delay heads + policy_loss()
  agent.py                   # ClashRoyaleAgent (tokenizer→encoder→DT→heads)
training/
  train_policy.py            # training loop with cosine warmup + tb/wandb
README_NEW.md                # this file
```

The legacy code (`training/policy/dt.py`, `training/policy/starformer*.py`,
`perception/sar/*`, `perception/ocr/*`, `data_pipeline/detection_dataset/cut_episodes.py`,
`training/classification/*`) is left untouched on disk and can be removed once
the new path is validated. None of the new modules import from them.

---

## Quick start

### 1. Install

The new code only needs PyTorch + Ultralytics + einops. JAX / PaddleOCR are
not required.

```bash
python3 -m pip install torch torchvision ultralytics einops opencv-python pyyaml numpy
# optional:
python3 -m pip install tensorboard wandb
```

### 2. Smoke test (no data needed)

```bash
python3 -m training.train_policy --smoke --out outputs/policy_runs/smoke
```

This runs the full pipeline (Perceiver + DT + heads) with synthetic random
trajectories and a tiny model. Should finish in under a minute on CPU and
produce a checkpoint at `outputs/policy_runs/smoke/ckpt/epoch_000.pt`.

### 3. Real training run

Build trajectories using `perception/yolo_detector.py` to convert raw
gameplay videos into the .npz schema described below, drop them under
`outputs/replay_dataset/`, then:

```bash
python3 -m training.train_policy \
    --data outputs/replay_dataset \
    --epochs 10 \
    --batch-size 16 \
    --seq-len 30 \
    --device auto
```

TensorBoard logs land in `outputs/policy_runs/perceiver_dt/tb`.

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
| `cards`        | (N, n_hand)     | int64   | card-name ids per slot, -1 = unknown         |
| `elixir`       | (N,)            | int64   | 0..10, -1 unknown                            |
| `action_card`  | (N,)            | int64   | slot in [0, n_hand); n_hand = no-op          |
| `action_pos`   | (N, 2)          | float32 | (row_norm, col_norm) in [0, 1]               |
| `reward`       | (N,)            | float32 | per-frame scalar reward                      |
| `timestep`     | (N,)            | int64   | absolute frame index in the episode          |

Return-to-go is computed at load time (`np.flip(np.cumsum(np.flip(reward)))`).
Trajectories shorter than `seq_len` are left-padded by repeating the first
frame at sample time, so you don't need to drop short games.

`build_random_trajectory(cfg, n_frames)` generates a synthetic sample with
the correct schema — useful for unit tests.

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

## What was removed (and why)

| removed                                              | replacement                              |
| ---------------------------------------------------- | ----------------------------------------- |
| PaddleOCR timer / center-text                        | none — trajectories are pre-segmented     |
| `data_pipeline/detection_dataset/cut_episodes.py`   | upstream collector responsibility         |
| ResNet card classifier                               | card embedded directly from name id       |
| ResNet elixir classifier                             | elixir embedded as int (0..10)            |
| `perception/sar/*` (SARBuilder, OCR-coupled)        | `data/replay_dataset.py` reads .npz       |
| Spatial-attention-then-temporal-attention StARformer | unified Perceiver + Decision Transformer  |
| Multiple parallel YOLO detectors                     | single `YOLODetector` wrapper             |
| Hard-coded 32×18 deployment grid                     | configurable (default 36×64 per spec)     |

---

## Inference (sketch)

The `ClashRoyaleAgent` is checkpoint-only-portable. To plug it into the
existing local-inference entry point (`inference/run.py`), build a sliding
window of the last `seq_len` frames + their YOLO detections, push them
through `agent.predict_last(batch)`, then map `select` / `pos` to a card
name and pixel coordinate. (Helper code lives in `models/policy/agent.py`.)

The legacy YOLO + hardcoded-strategy advisor still works as before; the new
model is opt-in.

---

## License

MIT, same as upstream. See [LICENSE](LICENSE).
