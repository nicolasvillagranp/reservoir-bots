"""Phase 1: YOLO fine-tuning and inference on 4 macro-classes.

Training uses yolo11n.pt (Nano) with batch=4, imgsz=640 for 4GB VRAM.
Inference applies NMS + drops bboxes < 1% of image area.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from config import DATA_DIR, MACRO_CLASSES, YOLOConfig


def train(
    cfg: YOLOConfig | None = None,
    data_yaml: Path | None = None,
    project: str | Path = "runs/train",
    name: str = "theker",
    exist_ok: bool = True,
    **overrides: object,
) -> Path:
    """Fine-tune YOLO on our 4-class dataset.

    Args:
        cfg: Training hyperparameters. Defaults to YOLOConfig().
        data_yaml: Path to dataset.yaml. Defaults to data/dataset.yaml.
        project: Ultralytics project directory for saving runs.
        name: Run name within project.
        exist_ok: Allow overwriting existing run directory.
        **overrides: Extra kwargs forwarded to model.train().

    Returns:
        Path to the best.pt weights file.
    """
    cfg = cfg or YOLOConfig()
    data_yaml = data_yaml or (DATA_DIR / "dataset.yaml")

    model = YOLO(cfg.model_weights)
    model.train(
        data=str(data_yaml),
        epochs=cfg.epochs,
        batch=cfg.batch,
        imgsz=cfg.imgsz,
        device=cfg.device,
        project=str(project),
        name=name,
        exist_ok=exist_ok,
        **overrides,
    )
    best_pt = Path(project) / name / "weights" / "best.pt"
    return best_pt


def predict_objects(
    img_path: str | Path,
    model_path: str | Path,
    conf_thresh: float = 0.25,
    min_area_frac: float = 0.01,
) -> list[dict]:
    """Run inference and return filtered detections.

    Drops any bbox occupying < min_area_frac of the image area.

    Args:
        img_path: Path to input image.
        model_path: Path to .pt weights.
        conf_thresh: Minimum confidence threshold for NMS.
        min_area_frac: Minimum bbox area as fraction of image area.

    Returns:
        List of dicts with keys: class_id, class_name, confidence,
        bbox_xyxy (absolute pixels), bbox_xywh (absolute pixels).
    """
    model = YOLO(str(model_path))
    results = model.predict(
        source=str(img_path),
        conf=conf_thresh,
        verbose=False,
    )

    detections: list[dict] = []
    for result in results:
        img_h, img_w = result.orig_shape
        img_area = img_h * img_w
        boxes = result.boxes

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            box_area = (x2 - x1) * (y2 - y1)

            if box_area / img_area < min_area_frac:
                continue

            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())

            # class_name depends on whether model is fine-tuned (0-3) or base COCO
            class_name = model.names.get(cls_id, str(cls_id))

            detections.append({
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": conf,
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
            })

    return detections
