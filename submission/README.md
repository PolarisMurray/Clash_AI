# Submission — Final Project

**A Screen-Based Clash Royale Agent with a Unified Multimodal Perceiver-Decision Transformer**

This folder is the grading package. It is self-contained — read these
files in order and you'll have the whole story.

---

## What to read, in this order

| # | File | Why |
|---|------|-----|
| 1 | **[REPORT.md](REPORT.md)** | The main written report. Tells the full story of what I built, why, and how well it works. ~10-min read. |
| 2 | **[figures/](figures/)** | All six visualizations referenced in the report, plus the SVG source of the architecture diagram. |
| 3 | **[HOW_TO_RUN.md](HOW_TO_RUN.md)** | Exact commands to reproduce every figure and the training run. |
| 4 | **[ATTRIBUTION.md](ATTRIBUTION.md)** | What I wrote, what I reused, every external source cited. |
| 5 | **[PRESENTATION.md](PRESENTATION.md)** | Speaker notes for the project expo (~10 min talk script). |

---

## Where the code lives

The submission folder is the **report**. The **code** is at the project
root, one directory up from this folder. Layout (only the new pipeline
— legacy code has been removed):

```
clash-royale-ai/
├── configs/                       Configuration tree
│   └── policy_config.py
├── data/                          Offline-RL replay dataset
│   └── replay_dataset.py
├── perception/                    Perception layer
│   ├── yolo_detector.py           YOLOv8 wrapper
│   └── tokenizer.py               Multimodal tokenizer
├── models/
│   ├── detection/                 Pre-trained YOLO weights (kept from upstream)
│   │   ├── yolo26s.pt
│   │   └── data.yaml
│   └── policy/                    The new policy model (everything I wrote)
│       ├── perceiver_encoder.py   Perceiver IO
│       ├── decision_transformer.py
│       ├── policy_heads.py
│       └── agent.py               End-to-end wrapper
├── training/
│   ├── train_policy.py            Training loop
│   └── detection/train.py         (Optional) re-train YOLO
├── scripts/                       Visualization scripts (one per figure)
│   ├── visualize_yolo.py
│   ├── visualize_training.py
│   ├── visualize_pipeline.py
│   └── visualize_dataset.py
├── tests/sample_screenshots/      344 sample frames, used for testing
├── outputs/                       Generated artifacts
│   ├── visualizations/            All 6 figures + speech draft
│   └── policy_runs/               Smoke + demo training checkpoints + TB logs
├── docs/figures/architecture.svg  Source of figure 1
├── README.md                      Top-level project README
└── requirements.txt               Slim dependency list
```

Total Python source: **13 files, ~1,500 lines**, all written by me.

---

## Quick verification

If you'd like a 30-second sanity check that the pipeline runs:

```bash
cd ..                              # to the project root
python3 -m training.train_policy --smoke
```

This runs the full forward + backward pass on a tiny synthetic dataset
in under a minute and saves a checkpoint. No data download needed.

---

## Figures at a glance

| File | Shows |
|------|-------|
| `figures/01_architecture.png`        | The 6-stage pipeline diagram |
| `figures/02_yolo_grid.png`           | YOLO running on 12 sample screenshots |
| `figures/03_yolo_stats.png`          | Class / confidence / count statistics |
| `figures/04_training_curves.png`     | Real training curves (5 epochs, 640 steps) |
| `figures/05_pipeline_shapes.png`     | Real screenshot end-to-end with tensor shapes |
| `figures/06_replay_dataset.png`      | Offline-RL data: reward, RTG, sampling weights |
| `figures/yolo_overlay_example.png`   | One annotated frame as a hero example |
| `figures/architecture.svg`           | Vector source of figure 1 |

---

