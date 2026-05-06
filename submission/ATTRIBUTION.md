# Attribution

This file lists every external source I reused, every paper I implemented
from, and clearly separates "what I wrote" from "what I took as-is."

---

## Code I wrote myself

All 13 Python files of the new pipeline + the four visualization scripts
+ the SVG architecture diagram. Roughly 1,500 lines.

| File | Lines | Notes |
|------|-------|-------|
| `configs/policy_config.py`           | 150  | Config dataclasses |
| `data/replay_dataset.py`             | 270  | Trajectory loader, weighted sampling, augmentations |
| `perception/yolo_detector.py`        | 180  | YOLOv8 wrapper class |
| `perception/tokenizer.py`            | 200  | Multimodal tokenizer (image patch / objects / cards / scalar tokens) |
| `models/policy/perceiver_encoder.py` | 130  | Perceiver IO encoder |
| `models/policy/decision_transformer.py` | 90  | Causal Transformer with action slot |
| `models/policy/policy_heads.py`      | 130  | Three heads + masked-loss helper |
| `models/policy/agent.py`             | 130  | End-to-end agent wrapper |
| `training/train_policy.py`           | 250  | Training loop with cosine warmup, grad-accum, TB logging |
| `scripts/visualize_yolo.py`          | 130  | Figure 2 + 3 |
| `scripts/visualize_training.py`      | 80   | Figure 4 |
| `scripts/visualize_pipeline.py`      | 170  | Figure 5 |
| `scripts/visualize_dataset.py`       | 100  | Figure 6 |
| `docs/figures/architecture.svg`      | —    | Figure 1, hand-coded SVG |

I did get help drafting this code with **Claude Code** (Anthropic's
coding assistant). Every file was reviewed and edited by me; the smoke
test, the figures, and the integration testing are my own work. I'd
treat this the same way I'd treat using StackOverflow or a textbook —
I understand every line and could rewrite it from scratch if asked.

---

## Code I reused as-is

### From the upstream repo *KataCR* (https://github.com/wty-yy/KataCR)

KataCR is the open-source repo I started from. License: MIT.
After cleanup, only **two artefacts** remain in the project:

1. **`models/detection/yolo26s.pt`** — a YOLOv8-s weight file trained on
   155 Clash Royale classes. I use it as a frozen detector. I did **not**
   retrain it.
2. **`tests/sample_screenshots/1/*.jpg`** — 344 screenshots used for
   testing my new wrapper. Image content only; no associated code.

I deleted the rest of KataCR (PaddleOCR pipeline, ResNet card / elixir
classifiers, episode cutter, JAX StARformer, SARBuilder glue,
data-pipeline scripts). See REPORT.md §10.

---

## Libraries used

Standard scientific-Python stack. No bespoke dependencies.

| Library | Used for |
|---------|----------|
| **PyTorch** ≥ 2.2          | All neural-network code |
| **Ultralytics** ≥ 8.1      | YOLOv8 inference |
| **einops** ≥ 0.8           | Tensor reshaping in the tokenizer |
| **OpenCV** (`cv2`) ≥ 4.9   | Image reading and resizing |
| **Pillow**                 | Drawing bounding-box overlays |
| **NumPy** ≥ 1.26           | Array ops in the dataset layer |
| **matplotlib** ≥ 3.8       | All figure rendering |
| **TensorBoard** ≥ 2.16     | Training logs (figure 4) |
| **PyYAML**                 | Reading `data.yaml` |

Plus, for figure rendering only:

- **librsvg** (Homebrew) — used to convert the SVG architecture diagram
  to PNG once (figure 1).

---

## Papers implemented from

I implemented two architectures from their published papers, in PyTorch.
Neither uses any open-source reference implementation directly.

### 1. Perceiver IO

> Jaegle, A., Borgeaud, S., Alayrac, J. B., et al. (2021).
> **Perceiver IO: A General Architecture for Structured Inputs & Outputs.**
> *arXiv:2107.14795* — https://arxiv.org/abs/2107.14795

What I took: the **latent-bottleneck cross-attention pattern**. A small
fixed set of learned latent vectors cross-attends a variable-length
input, then a stack of self-attention layers refines them.

What I changed: I share weights across cross→self blocks (Perceiver IO
suggests this as an option) to keep the parameter count down for a small
local run.

### 2. Decision Transformer

> Chen, L., Lu, K., Rajeswaran, A., et al. (2021).
> **Decision Transformer: Reinforcement Learning via Sequence Modeling.**
> *NeurIPS 2021* — https://arxiv.org/abs/2106.01345

What I took: the **return-to-go conditioning** and the **causal sequence
of (RTG, state, action)** layout.

What I changed: instead of one state token per timestep, my "state" is
the **K-element latent array from the Perceiver**, plus a reserved
"action slot" token whose hidden output becomes the action prediction.
This is closer in spirit to *Trajectory Transformer* (Janner et al.
2021) but adapted to multimodal latents.

---

## Class structure conventions

Where my code follows community PyTorch conventions (e.g.
`nn.MultiheadAttention` API, AdamW + cosine warmup), I followed the
standard recipes from:

- The **Annotated Transformer** (Rush, 2018) for attention layout.
- The **GPT-2 / nanoGPT** code style (Karpathy) for the causal block
  structure in `decision_transformer.py`.
- The **Hugging Face Transformers** library for the param-decay split
  pattern in `training/train_policy.py` (`split_decay_params`).

These influenced the *shape* of the code, not the algorithms — the
algorithms come from the two papers above.

---

## What is and isn't novel

To be clear about contribution scope:

**Novel in this project:**

- The combination of (Perceiver IO encoder) + (Decision Transformer over
  K-latent + action-slot tokens) for screen-based RTS gameplay.
- The unified multimodal token schema (Stage 3 of Figure 1) and the
  token-type-bias trick.
- Replacing KataCR's StARformer / OCR / ResNet stack with a single
  PyTorch pipeline ~10× simpler.

**Not novel:**

- Either underlying architecture (Perceiver IO, Decision Transformer).
- The YOLO detector itself.
- The choice of Clash Royale as a benchmark (KataCR did this first).

If you'd like a thorough literature comparison I'm happy to add one;
the report keeps it short to stay focused on what I built.
