"""Run a real frame end-to-end through the new pipeline and visualize tensor shapes.

Produces:
    05_pipeline_shapes.png   one figure showing actual data shapes/sizes at each stage

This grounds the architecture diagram in real numbers.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.policy_config import default_config
from data.replay_dataset import build_random_trajectory, build_dataloader
from models.policy.agent import ClashRoyaleAgent
from perception.yolo_detector import YOLODetector, load_class_names


def shape_str(t: torch.Tensor) -> str:
    return "(" + ", ".join(str(x) for x in t.shape) + ")"


def draw_box(ax, x, y, w, h, color, label, sublabel=None, fontsize=10):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                  linewidth=1.4, edgecolor=color, facecolor=color + "22")
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.65, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=color)
    if sublabel:
        ax.text(x + w / 2, y + h * 0.30, sublabel, ha="center", va="center",
                fontsize=fontsize - 1, color="#475569")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screenshot", type=Path, default=ROOT / "tests/sample_screenshots/1/02325.jpg")
    ap.add_argument("--weights", type=Path, default=ROOT / "models/detection/yolo26s.pt")
    ap.add_argument("--data-yaml", type=Path, default=ROOT / "models/detection/data.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/visualizations/05_pipeline_shapes.png")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    print("[pipeline] using screenshot:", args.screenshot)
    frame = cv2.imread(str(args.screenshot))
    if frame is None:
        raise SystemExit(f"could not read {args.screenshot}")

    names = load_class_names(args.data_yaml)
    yolo = YOLODetector(args.weights, conf=0.2, device=args.device, class_names=names)
    fd = yolo.predict(frame)
    print(f"[pipeline] {len(fd.detections)} detections")

    # build the full agent (small config so it fits on CPU/MPS quickly)
    cfg = default_config(seq_len=4, batch_size=1, num_workers=0)
    cfg.model.perceiver.n_self_layers = 1
    cfg.model.perceiver.n_cross_blocks = 1
    cfg.model.n_layers = 2
    cfg.model.perceiver.n_latents = 16
    cfg.tokenizer.image_size = (64, 64)
    cfg.tokenizer.patch_size = 16
    cfg.tokenizer.max_objects = 8
    cfg.__post_init__()

    trajs = [build_random_trajectory(cfg, 16, seed=0)]
    loader = build_dataloader(cfg, trajs, split="train")
    batch, targets = next(iter(loader))
    agent = ClashRoyaleAgent(cfg)
    agent.eval()

    with torch.no_grad():
        # tokens for one batched-flat frame
        B, T = batch.images.shape[:2]
        flat_kwargs = dict(
            images=batch.images.reshape(B * T, *batch.images.shape[2:]),
            det_cls=batch.det_cls.reshape(B * T, -1),
            det_bbox=batch.det_bbox.reshape(B * T, -1, 4),
            det_conf=batch.det_conf.reshape(B * T, -1),
            det_side=batch.det_side.reshape(B * T, -1),
            det_track=batch.det_track.reshape(B * T, -1),
            det_mask=batch.det_mask.reshape(B * T, -1),
            cards=batch.cards.reshape(B * T, -1),
            elixir=batch.elixir.reshape(B * T),
            rtg=batch.rtg.reshape(B * T),
            prev_a_card=batch.prev_a_card.reshape(B * T),
            prev_a_pos=batch.prev_a_pos.reshape(B * T, 2),
            prev_r=batch.prev_r.reshape(B * T),
        )
        ft = agent.tokenizer(**flat_kwargs)
        latents = agent.encoder(ft.tokens, ft.mask)
        latents_seq = latents.reshape(B, T, *latents.shape[1:])
        h = agent.dt(latents_seq, batch.timesteps)
        out = agent.heads(h)

    n_patches = (cfg.tokenizer.image_size[0] // cfg.tokenizer.patch_size) ** 2
    info = {
        "frame_HWC": frame.shape,
        "n_yolo_dets": len(fd.detections),
        "image_tensor": batch.images,
        "det_cls": batch.det_cls,
        "cards": batch.cards,
        "tokens": ft.tokens,
        "tokens_mask": ft.mask,
        "n_patches": n_patches,
        "latents": latents,
        "dt_hidden": h,
        "select": out.select_logits,
        "pos": out.pos_logits,
        "delay": out.delay_logits,
    }

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.2], hspace=0.35, wspace=0.25)

    # Top-left: real screenshot with YOLO boxes
    ax_img = fig.add_subplot(gs[0, 0])
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax_img.imshow(img_rgb)
    for d in fd.detections:
        x1, y1, x2, y2 = d.bbox_xyxy.astype(int)
        ax_img.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           fill=False, edgecolor="#16a34a", linewidth=1.6))
        ax_img.text(x1, max(0, y1 - 4), d.cls_name, fontsize=8, color="#16a34a", fontweight="bold")
    ax_img.set_title(f"Real screenshot · {len(fd.detections)} YOLO detections", fontsize=11)
    ax_img.axis("off")

    # Top-right block: tokenizer / encoder / DT shapes via boxes
    ax = fig.add_subplot(gs[0, 1:])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Tensor shapes through the new pipeline (B=1, T=4 demo config)", fontsize=12, loc="left", fontweight="bold")

    draw_box(ax, 0.05, 2.4, 1.7, 1.1, "#3b82f6", "Image",        shape_str(info["image_tensor"]))
    draw_box(ax, 1.85, 2.4, 1.7, 1.1, "#16a34a", "YOLO dets",    shape_str(info["det_cls"]))
    draw_box(ax, 3.65, 2.4, 1.7, 1.1, "#a855f7", "Cards",        shape_str(info["cards"]))
    draw_box(ax, 5.45, 2.4, 1.7, 1.1, "#94a3b8", "Token mask",   shape_str(info["tokens_mask"]))

    # arrow row
    for x in (1.78, 3.58, 5.38, 7.18):
        ax.annotate("", xy=(x + 0.05, 1.85), xytext=(x - 0.1, 2.4),
                    arrowprops=dict(arrowstyle="->", color="#334155"))

    draw_box(ax, 0.05, 0.7, 3.4, 1.1, "#0ea5e9", "Tokenizer output", shape_str(info["tokens"]),
             fontsize=11)
    draw_box(ax, 3.65, 0.7, 2.4, 1.1, "#7c3aed", "Perceiver latents", shape_str(info["latents"]),
             fontsize=11)
    draw_box(ax, 6.25, 0.7, 2.6, 1.1, "#0f766e", "DT action-slot hidden", shape_str(info["dt_hidden"]),
             fontsize=11)

    ax.annotate("", xy=(3.65, 1.25), xytext=(3.45, 1.25), arrowprops=dict(arrowstyle="->", color="#334155"))
    ax.annotate("", xy=(6.25, 1.25), xytext=(6.05, 1.25), arrowprops=dict(arrowstyle="->", color="#334155"))

    ax.text(8.95, 1.25, "→  policy heads", fontsize=11, color="#0f172a", fontweight="bold", va="center")

    # Bottom: head outputs as bar plots
    sel = info["select"][0, -1].numpy()
    pos_logits = info["pos"][0, -1].numpy()
    delay = info["delay"][0, -1].numpy()

    ax_sel = fig.add_subplot(gs[1, 0])
    sel_labels = [f"slot {i}" for i in range(len(sel) - 1)] + ["no-op"]
    ax_sel.bar(sel_labels, sel, color=["#a855f7"] * (len(sel) - 1) + ["#94a3b8"])
    ax_sel.set_title("Select head logits  (random init)", fontsize=11)
    ax_sel.set_ylabel("logit")
    ax_sel.tick_params(axis="x", labelsize=9)

    ax_pos = fig.add_subplot(gs[1, 1])
    pos_grid = pos_logits.reshape(cfg.tokenizer.grid_rows, cfg.tokenizer.grid_cols)
    im = ax_pos.imshow(pos_grid, cmap="Blues", aspect="auto")
    ax_pos.set_title(f"Position-head logits  ({pos_grid.shape[0]}×{pos_grid.shape[1]} grid)", fontsize=11)
    ax_pos.set_xticks([]); ax_pos.set_yticks([])
    plt.colorbar(im, ax=ax_pos, fraction=0.04)

    ax_del = fig.add_subplot(gs[1, 2])
    ax_del.bar(range(len(delay)), delay, color="#ef4444")
    ax_del.set_title("Delay head logits  (frames ahead)", fontsize=11)
    ax_del.set_xlabel("Δt bin"); ax_del.set_ylabel("logit")

    ax_info = fig.add_subplot(gs[1, 3])
    ax_info.axis("off")
    ax_info.set_title("Pipeline summary", fontsize=11, loc="left", fontweight="bold")
    rows = [
        ("frame size",          f"{info['frame_HWC'][1]}×{info['frame_HWC'][0]}"),
        ("yolo detections",     f"{info['n_yolo_dets']} (conf ≥ 0.2)"),
        ("image patches",       f"{info['n_patches']}  ({cfg.tokenizer.image_size[0]}//{cfg.tokenizer.patch_size})²"),
        ("max object slots",    f"{cfg.tokenizer.max_objects}"),
        ("tokens / frame",      f"{info['tokens'].shape[1]}"),
        ("d_model",             f"{cfg.model.d_model}"),
        ("perceiver latents",   f"{cfg.model.perceiver.n_latents}"),
        ("dt seq_len T",        f"{cfg.model.seq_len}"),
        ("grid head",           f"{cfg.tokenizer.grid_rows}×{cfg.tokenizer.grid_cols}={cfg.tokenizer.grid_rows*cfg.tokenizer.grid_cols} cells"),
        ("delay head",          f"{cfg.tokenizer.max_delay+1} bins"),
    ]
    y = 0.95
    for k, v in rows:
        ax_info.text(0.02, y, k, fontsize=10, color="#475569")
        ax_info.text(0.50, y, v, fontsize=10, color="#0f172a", fontweight="bold")
        y -= 0.085

    fig.suptitle("End-to-end forward pass — real screenshot → tokens → Perceiver → DT → heads",
                 fontsize=14, fontweight="bold", y=0.995)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[pipeline] wrote {args.out}")


if __name__ == "__main__":
    main()
