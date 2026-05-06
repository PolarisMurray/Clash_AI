"""Generate a set of YOLO-detection visualizations using the NEW perception wrapper.

Outputs (under --out):
    yolo_overlay_<frame>.png   one annotated screenshot per chosen frame
    yolo_grid.png              4-panel grid of those screenshots
    yolo_stats.png             class distribution + confidence histogram + boxes/frame
    yolo_summary.json          per-frame detection counts + top classes

Run: python -m scripts.visualize_yolo --frames 8 --device mps
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.yolo_detector import YOLODetector, load_class_names


# ---------------- color palette per class id (stable seed) ---------------- #
def class_colors(n: int, seed: int = 42) -> List[tuple]:
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 230, size=(n, 3))
    return [tuple(int(x) for x in c) for c in base]


def draw_overlay(frame_bgr: np.ndarray, fd, colors) -> np.ndarray:
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except OSError:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    for d in fd.detections:
        x1, y1, x2, y2 = [int(v) for v in d.bbox_xyxy]
        c = colors[d.cls_id % len(colors)]
        side_tag = {1: "ally", 2: "enemy"}.get(d.side, "")
        draw.rectangle([x1, y1, x2, y2], outline=c + (255,), width=2)
        label = f"{d.cls_name} {d.conf:.2f}"
        # background pill behind text
        tw = draw.textlength(label, font=font_small)
        draw.rectangle([x1, max(0, y1 - 16), x1 + tw + 6, y1], fill=c + (220,))
        draw.text((x1 + 3, max(0, y1 - 15)), label, font=font_small, fill=(255, 255, 255))
        if side_tag:
            draw.text((x2 - 28, y2 - 14), side_tag, font=font_small, fill=c + (255,))

    # header
    head = f"{len(fd.detections)} detections    {fd.width}x{fd.height}"
    tw = draw.textlength(head, font=font)
    draw.rectangle([8, 8, 8 + tw + 12, 30], fill=(0, 0, 0, 180))
    draw.text((14, 11), head, font=font, fill=(255, 255, 255))
    return np.array(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "models/detection/yolo26s.pt")
    ap.add_argument("--data-yaml", type=Path, default=ROOT / "models/detection/data.yaml")
    ap.add_argument("--screenshots", type=Path, default=ROOT / "tests/sample_screenshots/1")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/visualizations")
    ap.add_argument("--frames", type=int, default=8, help="how many frames to sample evenly")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(args.screenshots.glob("*.jpg"))
    if not files:
        raise SystemExit(f"no .jpg under {args.screenshots}")
    idxs = np.linspace(0, len(files) - 1, args.frames).astype(int)
    chosen = [files[i] for i in idxs]
    print(f"[yolo] using {len(chosen)}/{len(files)} frames from {args.screenshots}")

    names = load_class_names(args.data_yaml)
    det = YOLODetector(args.weights, conf=args.conf, device=args.device, class_names=names)
    colors = class_colors(len(names))

    overlays = []
    summary = []
    cls_counter = Counter()
    confs = []
    n_per_frame = []

    for f in chosen:
        frame = cv2.imread(str(f))
        fd = det.predict(frame)
        ov = draw_overlay(frame, fd, colors)
        out_path = args.out / f"yolo_overlay_{f.stem}.png"
        Image.fromarray(ov).save(out_path)
        overlays.append((f.stem, ov, fd))
        summary.append({
            "frame": f.stem,
            "n_det": len(fd.detections),
            "top_classes": Counter(d.cls_name for d in fd.detections).most_common(5),
        })
        cls_counter.update(d.cls_name for d in fd.detections)
        confs.extend(d.conf for d in fd.detections)
        n_per_frame.append(len(fd.detections))
        print(f"  {f.stem}: {len(fd.detections)} dets")

    # grid
    cols = 4
    rows = int(np.ceil(len(overlays) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 5.4))
    axes = np.atleast_2d(axes).reshape(-1)
    for ax, (stem, ov, fd) in zip(axes, overlays):
        ax.imshow(ov)
        ax.set_title(f"frame {stem}  ·  {len(fd.detections)} det", fontsize=11)
        ax.axis("off")
    for ax in axes[len(overlays):]:
        ax.axis("off")
    fig.suptitle("New YOLO wrapper — sample-screenshot detections", fontsize=14, y=0.995)
    fig.tight_layout()
    grid_path = args.out / "02_yolo_grid.png"
    fig.savefig(grid_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[yolo] grid → {grid_path}")

    # stats triptych
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    top = cls_counter.most_common(15)
    if top:
        labels, counts = zip(*top)
        ax1.barh(range(len(labels)), counts, color="#16a34a")
        ax1.set_yticks(range(len(labels)))
        ax1.set_yticklabels(labels)
        ax1.invert_yaxis()
        ax1.set_xlabel("count across sampled frames")
        ax1.set_title(f"Top classes (15 / {len(cls_counter)})")
    if confs:
        ax2.hist(confs, bins=30, color="#3b82f6", edgecolor="white")
        ax2.set_xlabel("confidence")
        ax2.set_ylabel("# detections")
        ax2.set_title(f"Confidence distribution (n={len(confs)})")
    ax3.plot(n_per_frame, marker="o", color="#f97316")
    ax3.set_xlabel("frame index (sampled)")
    ax3.set_ylabel("# detections")
    ax3.set_title(f"Detections per frame  ·  mean={np.mean(n_per_frame):.1f}")
    ax3.set_xticks(range(len(n_per_frame)))
    ax3.set_xticklabels([s for s, _, _ in overlays], rotation=45, ha="right", fontsize=9)
    ax3.grid(alpha=0.3)
    fig.suptitle("YOLO detection statistics — sample screenshots", fontsize=14)
    fig.tight_layout()
    stats_path = args.out / "03_yolo_stats.png"
    fig.savefig(stats_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[yolo] stats → {stats_path}")

    json_path = args.out / "yolo_summary.json"
    json_path.write_text(json.dumps({
        "screenshots_dir": str(args.screenshots),
        "n_frames_used": len(chosen),
        "n_classes_seen": len(cls_counter),
        "n_detections_total": int(sum(n_per_frame)),
        "mean_detections_per_frame": float(np.mean(n_per_frame)),
        "per_frame": summary,
    }, indent=2))
    print(f"[yolo] summary → {json_path}")


if __name__ == "__main__":
    main()
