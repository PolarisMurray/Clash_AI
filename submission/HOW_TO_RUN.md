# How to Run / Reproduce Everything

Every figure and every result in `REPORT.md` came from a single command,
which I list below. Run them from the **project root** (the directory
*above* this `submission/` folder).

---

## 0. Setup (one-time)

```bash
# Python 3.11+ recommended (tested on 3.14 with MPS).
python3 -m pip install -r requirements.txt
```

Required: `torch`, `ultralytics`, `einops`, `opencv-python`, `Pillow`,
`numpy`, `matplotlib`, `tensorboard`, `PyYAML`.

If you want figure 1 re-rendered from SVG → PNG, you'll also need the
`librsvg` system library (Homebrew on macOS: `brew install librsvg`).

---

## 1. 30-second smoke test

Confirms the whole pipeline forward + backward + checkpoint works.

```bash
python3 -m training.train_policy --smoke --out outputs/policy_runs/smoke
```

Expected: ~256 training steps in ~30s, total loss falling from ≈11 → ≈8,
checkpoint saved at `outputs/policy_runs/smoke/ckpt/epoch_000.pt`.

---

## 2. The five-epoch demo training run (produces figure 4)

```bash
# Already-generated TB logs live under outputs/policy_runs/demo/tb/
# To re-run the training:
python3 -c "
import sys; sys.path.insert(0, '.')
import torch, math
from pathlib import Path
from configs.policy_config import default_config
from data.replay_dataset import build_random_trajectory, build_dataloader
from models.policy.agent import ClashRoyaleAgent
from models.policy.policy_heads import policy_loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

cfg = default_config(seq_len=4, batch_size=4, epochs=5, num_workers=0)
cfg.model.perceiver.n_self_layers = 1; cfg.model.perceiver.n_cross_blocks = 1
cfg.model.n_layers = 2; cfg.model.perceiver.n_latents = 16
cfg.tokenizer.image_size = (64, 64); cfg.tokenizer.patch_size = 16
cfg.tokenizer.max_objects = 8; cfg.__post_init__()

torch.manual_seed(0)
trajs = [build_random_trajectory(cfg, 64, seed=i) for i in range(8)]
loader = build_dataloader(cfg, trajs, split='train')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
agent = ClashRoyaleAgent(cfg).to(device)
opt = AdamW(agent.parameters(), lr=cfg.train.lr, betas=cfg.train.betas, weight_decay=cfg.train.weight_decay)
total = cfg.train.epochs * len(loader)
sch = LambdaLR(opt, lr_lambda=lambda s: (s/200) if s<200 else 0.1+0.9*0.5*(1+math.cos(math.pi*min(1,(s-200)/(total-200)))))
tb = SummaryWriter('outputs/policy_runs/demo/tb')
step = 0
loss_w = {'select':1.0,'pos':1.0,'delay':0.5}
for epoch in range(cfg.train.epochs):
    for batch, targets in loader:
        batch = batch.to(device); targets = {k: v.to(device) for k, v in targets.items()}
        am = targets.pop('action_mask')
        m = policy_loss(agent(batch), targets, am, loss_w)
        opt.zero_grad(set_to_none=True); m['loss'].backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        opt.step(); sch.step(); step += 1
        for k in ['loss','loss_select','loss_pos','loss_delay','acc_select','acc_pos','acc_delay']:
            tb.add_scalar(f'train/{k}', float(m[k].detach()), step)
        tb.add_scalar('train/lr', sch.get_last_lr()[0], step)
tb.close()
"
# Then render the curves:
python3 -m scripts.visualize_training
```

Output: `outputs/visualizations/04_training_curves.png`.

---

## 3. YOLO on real screenshots (produces figures 2 + 3 + the overlay folder)

```bash
python3 -m scripts.visualize_yolo --frames 12 --device mps --conf 0.15
```

Outputs:
- `outputs/visualizations/02_yolo_grid.png`  (figure 2)
- `outputs/visualizations/03_yolo_stats.png` (figure 3)
- `outputs/visualizations/yolo_overlays/*.png` (12 single-frame overlays)
- `outputs/visualizations/yolo_summary.json`

Drop `--device mps` if you're on CPU/CUDA.

---

## 4. End-to-end forward pass with real-data shapes (produces figure 5)

```bash
python3 -m scripts.visualize_pipeline
```

Output: `outputs/visualizations/05_pipeline_shapes.png`.

This script picks one screenshot from `tests/sample_screenshots/1/`,
runs YOLO on it, builds a synthetic 4-frame trajectory window, runs
the **whole agent** forward, and plots:

- the screenshot with bounding boxes
- the actual tensor shape at every stage
- the three head outputs

---

## 5. Replay-dataset properties (produces figure 6)

```bash
python3 -m scripts.visualize_dataset
```

Output: `outputs/visualizations/06_replay_dataset.png`.

---

## 6. Architecture diagram (produces figure 1)

The SVG (`docs/figures/architecture.svg`) is hand-written and version-
controlled. To re-render the PNG:

```bash
rsvg-convert -w 2400 docs/figures/architecture.svg -o outputs/visualizations/01_architecture.png
```

---

## 7. Run a real (full-config) training run

The default config is bigger than what runs on a laptop. On a real GPU:

```bash
# Drop trained .npz trajectories into outputs/replay_dataset/, then:
python3 -m training.train_policy \
    --data outputs/replay_dataset \
    --epochs 10 \
    --batch-size 16 \
    --seq-len 30 \
    --device cuda
```

The .npz schema is documented in `data/replay_dataset.py` — the docstring
at the top lists every required key with its shape and dtype.

---

## File-to-command mapping

| Figure / artifact                   | Producing command                              |
|-------------------------------------|------------------------------------------------|
| `01_architecture.png`               | §6 — `rsvg-convert ...`                        |
| `02_yolo_grid.png`                  | §3 — `scripts.visualize_yolo`                  |
| `03_yolo_stats.png`                 | §3 — same script                               |
| `04_training_curves.png`            | §2 — demo training + `scripts.visualize_training` |
| `05_pipeline_shapes.png`            | §4 — `scripts.visualize_pipeline`              |
| `06_replay_dataset.png`             | §5 — `scripts.visualize_dataset`               |
| `outputs/policy_runs/smoke/ckpt/`   | §1 — `training.train_policy --smoke`           |
| `outputs/policy_runs/demo/`         | §2 — demo training                             |
