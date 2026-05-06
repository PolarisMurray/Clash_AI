"""Thin YOLOv8 wrapper used by the new policy pipeline.

Replaces the old multi-detector / OCR-coupled perception code.
Only responsibility: take a BGR frame → return a structured Detection list.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


@dataclass
class Detection:
    """One YOLO bounding box on one frame."""

    cls_id: int
    cls_name: str
    conf: float
    bbox_xyxy: np.ndarray            # shape (4,), float, pixel coords
    bbox_norm: np.ndarray            # shape (4,), float, normalized cx, cy, w, h in [0, 1]
    side: int = 0                    # 0=unknown, 1=ally (lower half), 2=enemy (upper half)
    track_id: int = 0                # 0 = unassigned

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cls_id": int(self.cls_id),
            "cls_name": str(self.cls_name),
            "conf": float(self.conf),
            "bbox_xyxy": self.bbox_xyxy.tolist(),
            "bbox_norm": self.bbox_norm.tolist(),
            "side": int(self.side),
            "track_id": int(self.track_id),
        }


@dataclass
class FrameDetections:
    detections: List[Detection] = field(default_factory=list)
    width: int = 0
    height: int = 0

    def to_array(self, max_objects: int) -> Dict[str, np.ndarray]:
        """Pad / clip to a fixed length matrix consumable by the tokenizer."""
        n = min(len(self.detections), max_objects)
        cls_ids = np.zeros(max_objects, dtype=np.int64)
        bbox = np.zeros((max_objects, 4), dtype=np.float32)
        conf = np.zeros(max_objects, dtype=np.float32)
        side = np.zeros(max_objects, dtype=np.int64)
        track = np.zeros(max_objects, dtype=np.int64)
        mask = np.zeros(max_objects, dtype=np.bool_)
        for i, d in enumerate(self.detections[:n]):
            cls_ids[i] = d.cls_id
            bbox[i] = d.bbox_norm
            conf[i] = d.conf
            side[i] = d.side
            track[i] = d.track_id
            mask[i] = True
        return {
            "cls_ids": cls_ids,
            "bbox_norm": bbox,
            "conf": conf,
            "side": side,
            "track": track,
            "mask": mask,
        }


class YOLODetector:
    """Loads YOLOv8 once, runs predict, returns FrameDetections.

    No OCR. No episode cutting. No card classifier.
    Tracking is optional and uses ultralytics' built-in tracker if requested.
    """

    def __init__(
        self,
        weights: Path,
        conf: float = 0.25,
        imgsz: int = 640,
        device: Optional[str] = None,
        tracker: Optional[str] = None,
        class_names: Optional[Sequence[str]] = None,
        side_split_y: float = 0.5,
    ):
        from ultralytics import YOLO

        if not Path(weights).exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")
        self._model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.tracker = tracker
        self.side_split_y = side_split_y

        if class_names is not None:
            self.class_names = {i: str(n) for i, n in enumerate(class_names)}
        else:
            raw = getattr(self._model, "names", {}) or {}
            if isinstance(raw, dict):
                self.class_names = {int(k): str(v) for k, v in raw.items()}
            else:
                self.class_names = {i: str(v) for i, v in enumerate(raw)}

    def _label(self, cid: int) -> str:
        return self.class_names.get(int(cid), str(int(cid)))

    def _classify_side(self, cy_norm: float) -> int:
        # Lower half = ally (1), upper half = enemy (2). 0 = unknown for spells / global UI.
        return 1 if cy_norm >= self.side_split_y else 2

    def predict(self, frame_bgr: np.ndarray) -> FrameDetections:
        h, w = frame_bgr.shape[:2]
        if self.tracker is not None:
            result = self._model.track(
                source=frame_bgr,
                conf=self.conf,
                imgsz=self.imgsz,
                device=self.device,
                tracker=self.tracker,
                persist=True,
                verbose=False,
            )[0]
        else:
            result = self._model.predict(
                source=frame_bgr,
                conf=self.conf,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )[0]
        return self._parse(result, h, w)

    def predict_batch(self, frames_bgr: Iterable[np.ndarray]) -> List[FrameDetections]:
        return [self.predict(f) for f in frames_bgr]

    def _parse(self, result: Any, h: int, w: int) -> FrameDetections:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return FrameDetections(width=w, height=h)
        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(int)
        ids = boxes.id.detach().cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None else None

        out: List[Detection] = []
        for i, (b, c, ci) in enumerate(zip(xyxy, conf, cls)):
            x1, y1, x2, y2 = (float(v) for v in b)
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
            cy_norm = cy / max(h, 1)
            side = self._classify_side(cy_norm)
            tid = int(ids[i]) if ids is not None else 0
            out.append(
                Detection(
                    cls_id=int(ci),
                    cls_name=self._label(int(ci)),
                    conf=float(c),
                    bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
                    bbox_norm=np.array(
                        [cx / max(w, 1), cy_norm, bw / max(w, 1), bh / max(h, 1)],
                        dtype=np.float32,
                    ),
                    side=side,
                    track_id=tid,
                )
            )
        out.sort(key=lambda d: d.conf, reverse=True)
        return FrameDetections(detections=out, width=w, height=h)


def load_class_names(data_yaml: Path) -> List[str]:
    """Read names: from a data.yaml file."""
    import yaml
    data = yaml.safe_load(Path(data_yaml).read_text()) or {}
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names)]
    return [str(x) for x in names]


if __name__ == "__main__":
    import argparse
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/detection/yolo26s.pt")
    parser.add_argument("--data-yaml", default="models/detection/data.yaml")
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    names = load_class_names(Path(args.data_yaml))
    det = YOLODetector(Path(args.weights), device=args.device, class_names=names)
    frame = cv2.imread(args.image)
    fd = det.predict(frame)
    print(f"{len(fd.detections)} detections on a {fd.width}x{fd.height} frame")
    for d in fd.detections[:10]:
        print(f"  {d.cls_name:24s}  conf={d.conf:.2f}  side={d.side}  bbox_norm={d.bbox_norm.round(3)}")
