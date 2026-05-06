from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PART2 = ROOT / "source_dataset/images/part2"
YOLO_DATASET = ROOT / "prepared_dataset/yolo_detection"
RUNS_DIR = ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the Clash Royale part2 dataset and train a standard YOLOv8 model."
    )
    parser.add_argument("--model", default="yolo26s.pt", help="YOLO model checkpoint or yaml.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="mps", help='Use "mps" on Apple Silicon, "cpu", or a CUDA id.')
    parser.add_argument("--workers", type=int, default=0, help="0 is usually safest on macOS.")
    parser.add_argument("--name", default="cr_yolo26s")
    parser.add_argument("--project", default=str(RUNS_DIR))
    parser.add_argument(
        "--image-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How to place images into the YOLO dataset. hardlink saves disk space.",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Only build prepared_dataset/yolo_detection, do not train.")
    parser.add_argument("--skip-prepare", action="store_true", help="Train using an existing prepared_dataset/yolo_detection.")
    parser.add_argument("--check-labels", action="store_true", help="Only verify that nc matches the names list in data.yaml.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting an existing YOLO run name.")
    return parser.parse_args()


def load_class_names() -> list[str]:
    yaml_path = SOURCE_PART2 / "ClashRoyale_detection.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    real_names: dict[int, str] = {}
    for key, value in data["names"].items():
        idx = int(key)
        name = str(value)
        if name.startswith("pad_") or name == "pad_belong":
            continue
        real_names[idx] = name

    max_idx = max(real_names)
    names = [real_names.get(i, f"class_{i}") for i in range(max_idx + 1)]
    return names


def place_image(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "symlink":
        os.symlink(src, dst)
        return

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def convert_label(src_txt: Path, dst_txt: Path) -> int:
    dst_txt.parent.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    if src_txt.exists():
        for raw in src_txt.read_text(encoding="utf-8").splitlines():
            parts = raw.split()
            if len(parts) >= 5:
                rows.append(" ".join(parts[:5]))

    dst_txt.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def prepare_split(split: str, image_mode: str) -> tuple[int, int]:
    annotation_file = SOURCE_PART2 / f"{split}_annotation.txt"
    if not annotation_file.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotation_file}")

    image_count = 0
    box_count = 0
    for raw in annotation_file.read_text(encoding="utf-8").splitlines():
        rel = raw.strip().removeprefix("./")
        if not rel:
            continue

        src_img = SOURCE_PART2 / rel
        src_txt = src_img.with_suffix(".txt")
        if not src_img.exists():
            raise FileNotFoundError(f"Missing image: {src_img}")

        dst_img = YOLO_DATASET / "images" / split / rel
        dst_txt = YOLO_DATASET / "labels" / split / Path(rel).with_suffix(".txt")

        place_image(src_img, dst_img, image_mode)
        box_count += convert_label(src_txt, dst_txt)
        image_count += 1

    return image_count, box_count


def write_data_yaml(names: list[str]) -> Path:
    data_yaml = YOLO_DATASET / "data.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "path": str(YOLO_DATASET),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return data_yaml


def prepare_dataset(image_mode: str) -> Path:
    if not SOURCE_PART2.exists():
        raise FileNotFoundError(f"Missing source dataset: {SOURCE_PART2}")

    names = load_class_names()
    train_images, train_boxes = prepare_split("train", image_mode)
    val_images, val_boxes = prepare_split("val", image_mode)
    data_yaml = write_data_yaml(names)

    print(f"YOLO dataset ready: {YOLO_DATASET}")
    print(f"Classes: {len(names)}")
    print(f"Train: {train_images} images, {train_boxes} boxes")
    print(f"Val: {val_images} images, {val_boxes} boxes")
    print(f"Config: {data_yaml}")
    return data_yaml


def train(args: argparse.Namespace, data_yaml: Path) -> None:
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
    )


def check_labels(data_yaml: Path) -> None:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data["names"]
    nc = data["nc"]
    print("nc =", nc)
    print("number of names =", len(names))
    if nc != len(names):
        raise SystemExit("ERROR: nc does not match names length")
    print("OK")


def main() -> None:
    args = parse_args()
    data_yaml = YOLO_DATASET / "data.yaml"

    if args.check_labels:
        check_labels(data_yaml)
        return

    if not args.skip_prepare:
        data_yaml = prepare_dataset(args.image_mode)

    if args.prepare_only:
        return

    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing YOLO data config: {data_yaml}. Run without --skip-prepare first.")

    train(args, data_yaml)


if __name__ == "__main__":
    main()
