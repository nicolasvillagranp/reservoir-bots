"""Phase 1: YOLO fine-tuning and inference on 4 macro-classes.

Usage: uv run python -m src.phases.phase1_vision
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from ultralytics import YOLO

from src.config import DATA_DIR, FINETUNED_DIR, IMAGE_DIR, LABEL_DIR, MODE, YOLOConfig


def train(
    cfg: YOLOConfig | None = None,
    data_yaml: Path | None = None,
    project: str | Path = FINETUNED_DIR,
    name: str = "yolo",
    exist_ok: bool = True,
    **overrides: object,
) -> Path:
    """Fine-tune YOLO on our 4-class dataset.

    Returns:
        Path to the best.pt weights file.
    """
    cfg = cfg or YOLOConfig()
    data_yaml = data_yaml or (DATA_DIR / "dataset.yaml")

    model = YOLO(cfg.model_weights)
    print(f"Training: epochs={cfg.epochs}, batch={cfg.batch}, imgsz={cfg.imgsz}")
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
    """
    model = YOLO(str(model_path))
    results = model.predict(source=str(img_path), conf=conf_thresh, verbose=False)

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
            class_name = model.names.get(cls_id, str(cls_id))

            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                }
            )

    return detections


def _write_dataset_yaml(path: Path, images_subdir: str) -> None:
    """Write a YOLO dataset.yaml. YOLO auto-resolves labels/ from images/."""
    path.write_text(
        f"path: {DATA_DIR.as_posix()}\n"
        f"train: {images_subdir}\n"
        f"val: images/val\n"
        f"\nnc: 4\n"
        f"names:\n"
        f"  0: HUMAN\n"
        f"  1: VEHICLE\n"
        f"  2: OBSTACLE\n"
        f"  3: CONTEXT\n"
    )


def prepare_dataset() -> Path:
    """Build dataset.yaml. In test mode, create tiny subset; in production, use full data."""
    if MODE == "test":
        print("Test mode: creating tiny subset of the dataset for quick iteration.")
        cfg = YOLOConfig()
        n = cfg.subset_size or 50
        tiny_img = IMAGE_DIR / "tiny"
        tiny_lbl = LABEL_DIR / "tiny"
        tiny_img.mkdir(parents=True, exist_ok=True)
        tiny_lbl.mkdir(parents=True, exist_ok=True)

        print(f"Selecting {n} random non-empty labels from {LABEL_DIR / 'train'}...")

        train_labels = sorted((LABEL_DIR / "train").glob("*.txt"))

        print(f"Found {len(train_labels)} total label files. Filtering out empty ones...")
        non_empty = [lp for lp in train_labels if lp.read_text().strip()]

        print(f"Found {len(non_empty)} non-empty labels. Shuffling and selecting {n}...")
        random.seed(42)
        random.shuffle(non_empty)
        selected = non_empty[:n]

        for i, lbl in enumerate(selected, 1):
            print(f"Copying {i}/{len(selected)}: {lbl.name}", end="\r")
            shutil.copy2(lbl, tiny_lbl / lbl.name)
            for ext in (".jpg", ".jpeg", ".png"):
                img_src = IMAGE_DIR / "train" / f"{lbl.stem}{ext}"
                if img_src.exists():
                    shutil.copy2(img_src, tiny_img / img_src.name)
                    break
        print(f"\nCopied {len(selected)} labels and images to tiny subset.")

        yaml_path = DATA_DIR / "dataset.yaml"
        _write_dataset_yaml(yaml_path, "images/tiny")
        print(f"Test mode: {len(selected)}-image tiny subset -> {yaml_path}")
        return yaml_path

    # Production: full dataset
    yaml_path = DATA_DIR / "dataset.yaml"
    if not yaml_path.exists():
        _write_dataset_yaml(yaml_path, "images/train")
        print(f"Created full dataset.yaml -> {yaml_path}")
    return yaml_path


def main() -> None:
    """Fine-tune YOLO with current config (test or production mode)."""
    cfg = YOLOConfig()
    print(f"=== Phase 1: YOLO fine-tuning ({cfg.epochs} epochs, mode={MODE}) ===")
    data_yaml = prepare_dataset()
    best_pt = train(cfg=cfg, data_yaml=data_yaml)
    last_pt = best_pt.parent / "last.pt"
    weights = best_pt if best_pt.exists() else last_pt
    print(f"Weights saved to {weights}")
    print("=== Phase 1 complete ===")


if __name__ == "__main__":
    main()
