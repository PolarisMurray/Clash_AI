"""Run real frames end-to-end through the new pipeline and visualize tensor shapes.

Produces:
    05_pipeline_shapes.png   one figure showing actual data shapes/sizes at each stage

A window of seq_len consecutive real frames ending at --frame goes through
YOLO -> trajectory -> tokenizer -> Perceiver -> Decision Transformer -> heads.
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
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.policy_config import default_config
from data.replay_dataset import ReplayDataset, collate, trajectory_from_frames
from models.policy.agent import ClashRoyaleAgent
from perception.yolo_detector import DEFAULT_WEIGHTS, YOLODetector

DEFAULT_FRAMES = ROOT / "training/detection/prepared_dataset/yolo_detection/images/val/OYASSU_20230212_episodes/4"


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


def build_agent(ckpt: Path) -> ClashRoyaleAgent:
    if ckpt.exists():
        print("[pipeline] loading policy checkpoint:", ckpt)
        return ClashRoyaleAgent.from_checkpoint(ckpt)
    print("[pipeline] no checkpoint, using a small random-init policy")
    cfg = default_config(seq_len=4, batch_size=1, num_workers=0)
    cfg.model.perceiver.n_self_layers = 1
    cfg.model.perceiver.n_cross_blocks = 1
    cfg.model.n_layers = 2
    cfg.model.perceiver.n_latents = 16
    cfg.tokenizer.image_size = (64, 64)
    cfg.tokenizer.max_objects = 8
    return ClashRoyaleAgent(cfg).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES)
    ap.add_argument("--frame", default="02385", help="stem of the last frame in the window")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--ckpt", type=Path, default=ROOT / "outputs/policy_runs/demo/ckpt.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/visualizations/05_pipeline_shapes.png")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    agent = build_agent(args.ckpt)
    cfg = agent.cfg
    cfg.data.lr_flip = cfg.data.card_shuffle = False
    T = cfg.model.seq_len

    files = sorted(args.frames_dir.glob("*.jpg"))
    stems = [f.stem for f in files]
    if args.frame not in stems:
        raise SystemExit(f"frame {args.frame} not in {args.frames_dir}")
    end = stems.index(args.frame)
    window = files[max(0, end - T + 1): end + 1]
    print(f"[pipeline] frames {window[0].stem}..{window[-1].stem} from {args.frames_dir}")

    yolo = YOLODetector(args.weights, conf=args.conf, device=args.device)
    frames_bgr = [cv2.imread(str(f)) for f in window]
    dets = [yolo.predict(f) for f in frames_bgr]
    fd = dets[-1]
    frame = frames_bgr[-1]
    print(f"[pipeline] {len(fd.detections)} detections on the last frame")

    traj = trajectory_from_frames([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr], dets, cfg)
    batch, _ = collate([ReplayDataset([traj], cfg)[len(traj) - 1]])

    with torch.no_grad():
        B, T = batch.images.shape[:2]
        ft = agent.tokenizer(
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
        latents = agent.encoder(ft.tokens, ft.mask)
        h = agent.dt(latents.reshape(B, T, *latents.shape[1:]), batch.timesteps)
        out = agent.heads(h)
        action = agent.predict_last(batch)
    sel = int(action["select"][0])
    row, col = (int(v) for v in action["pos"][0])
    sel_name = "no-op" if sel == cfg.tokenizer.n_hand else f"card slot {sel}"
    print(f"[pipeline] predicted action: {sel_name} at grid cell (row={row}, col={col}), "
          f"delay={int(action['delay'][0]) if 'delay' in action else '-'}")

    n_patches = (cfg.tokenizer.image_size[0] // cfg.tokenizer.patch_size) * (cfg.tokenizer.image_size[1] // cfg.tokenizer.patch_size)

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.2], hspace=0.35, wspace=0.25)

    # Top-left: real screenshot with YOLO boxes
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    for d in fd.detections:
        x1, y1, x2, y2 = d.bbox_xyxy.astype(int)
        ax_img.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           fill=False, edgecolor="#16a34a", linewidth=1.2))
    ax_img.set_title(f"Frame {window[-1].stem} · {len(fd.detections)} YOLO detections", fontsize=11)
    ax_img.axis("off")

    # Top-right block: tokenizer / encoder / DT shapes via boxes
    ax = fig.add_subplot(gs[0, 1:])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title(f"Tensor shapes through the pipeline (B=1, T={T}, real frames)", fontsize=12, loc="left", fontweight="bold")

    draw_box(ax, 0.05, 2.4, 1.7, 1.1, "#3b82f6", "Image",        shape_str(batch.images))
    draw_box(ax, 1.85, 2.4, 1.7, 1.1, "#16a34a", "YOLO dets",    shape_str(batch.det_cls))
    draw_box(ax, 3.65, 2.4, 1.7, 1.1, "#a855f7", "Cards",        shape_str(batch.cards))
    draw_box(ax, 5.45, 2.4, 1.7, 1.1, "#94a3b8", "Token mask",   shape_str(ft.mask))

    # arrow row
    for x in (1.78, 3.58, 5.38, 7.18):
        ax.annotate("", xy=(x + 0.05, 1.85), xytext=(x - 0.1, 2.4),
                    arrowprops=dict(arrowstyle="->", color="#334155"))

    draw_box(ax, 0.05, 0.7, 3.4, 1.1, "#0ea5e9", "Tokenizer output", shape_str(ft.tokens), fontsize=11)
    draw_box(ax, 3.65, 0.7, 2.4, 1.1, "#7c3aed", "Perceiver latents", shape_str(latents), fontsize=11)
    draw_box(ax, 6.25, 0.7, 2.6, 1.1, "#0f766e", "DT action-slot hidden", shape_str(h), fontsize=11)

    ax.annotate("", xy=(3.65, 1.25), xytext=(3.45, 1.25), arrowprops=dict(arrowstyle="->", color="#334155"))
    ax.annotate("", xy=(6.25, 1.25), xytext=(6.05, 1.25), arrowprops=dict(arrowstyle="->", color="#334155"))

    ax.text(8.95, 1.25, "→  policy heads", fontsize=11, color="#0f172a", fontweight="bold", va="center")

    # Bottom: head outputs at the last timestep
    sel_logits = out.select_logits[0, -1].numpy()
    pos_logits = out.pos_logits[0, -1].numpy() if out.pos_logits is not None else None
    delay = out.delay_logits[0, -1].numpy() if out.delay_logits is not None else None

    ax_sel = fig.add_subplot(gs[1, 0])
    sel_labels = [f"slot {i}" for i in range(len(sel_logits) - 1)] + ["no-op"]
    ax_sel.bar(sel_labels, sel_logits, color=["#a855f7"] * (len(sel_logits) - 1) + ["#94a3b8"])
    ax_sel.set_title("Select head logits", fontsize=11)
    ax_sel.set_ylabel("logit")
    ax_sel.tick_params(axis="x", labelsize=9)

    ax_pos = fig.add_subplot(gs[1, 1])
    if pos_logits is not None:
        pos_grid = pos_logits.reshape(cfg.tokenizer.grid_rows, cfg.tokenizer.grid_cols)
        im = ax_pos.imshow(pos_grid, cmap="Blues", aspect="auto")
        ax_pos.set_title(f"Position-head logits  ({pos_grid.shape[0]}×{pos_grid.shape[1]} grid)", fontsize=11)
        plt.colorbar(im, ax=ax_pos, fraction=0.04)
    ax_pos.set_xticks([]); ax_pos.set_yticks([])

    ax_del = fig.add_subplot(gs[1, 2])
    if delay is not None:
        ax_del.bar(range(len(delay)), delay, color="#ef4444")
    ax_del.set_title("Delay head logits  (frames ahead)", fontsize=11)
    ax_del.set_xlabel("Δt bin"); ax_del.set_ylabel("logit")

    ax_info = fig.add_subplot(gs[1, 3])
    ax_info.axis("off")
    ax_info.set_title("Pipeline summary", fontsize=11, loc="left", fontweight="bold")
    rows = [
        ("frame size",          f"{frame.shape[1]}×{frame.shape[0]}"),
        ("yolo detections",     f"{len(fd.detections)} (conf ≥ {args.conf})"),
        ("image patches",       f"{n_patches}"),
        ("max object slots",    f"{cfg.tokenizer.max_objects}"),
        ("tokens / frame",      f"{ft.tokens.shape[1]}"),
        ("d_model",             f"{cfg.model.d_model}"),
        ("perceiver latents",   f"{cfg.model.perceiver.n_latents}"),
        ("dt seq_len T",        f"{cfg.model.seq_len}"),
        ("grid head",           f"{cfg.tokenizer.grid_rows}×{cfg.tokenizer.grid_cols}={cfg.tokenizer.grid_size} cells"),
        ("predicted action",    f"{sel_name} @ ({row}, {col})"),
    ]
    y = 0.95
    for k, v in rows:
        ax_info.text(0.02, y, k, fontsize=10, color="#475569")
        ax_info.text(0.50, y, v, fontsize=10, color="#0f172a", fontweight="bold")
        y -= 0.085

    fig.suptitle("End-to-end forward pass — real frames → YOLO → tokens → Perceiver → DT → heads",
                 fontsize=14, fontweight="bold", y=0.995)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[pipeline] wrote {args.out}")


if __name__ == "__main__":
    main()
