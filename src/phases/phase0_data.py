"""Phase 0: Parse COCO annotations → YOLO labels + pre-download YOLO weights.

Usage: uv run python -m src.phases.phase0_data
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from src.config import (
    DATA_DIR,
    LABEL_DIR,
    PRETRAINED_DIR,
    YOLOConfig,
    raw_name_to_macro_id,
)

logger = logging.getLogger(__name__)


def load_coco(annotation_path: Path) -> dict:
    """Load a COCO-format JSON file."""
    with open(annotation_path) as f:
        return json.load(f)


def build_category_lookup(coco: dict) -> dict[int, str]:
    """Map raw COCO category IDs to their names."""
    return {cat["id"]: cat["name"] for cat in coco["categories"]}


def build_image_lookup(coco: dict) -> dict[int, dict]:
    """Map image IDs to their metadata (file_name, width, height)."""
    return {img["id"]: img for img in coco["images"]}


def coco_bbox_to_yolo(
    bbox: list[float], img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """Convert COCO [x_tl, y_tl, w, h] (absolute) to YOLO [x_c, y_c, w, h] (normalized)."""
    x_tl, y_tl, bw, bh = bbox
    x_center = max(0.0, min(1.0, (x_tl + bw / 2.0) / img_w))
    y_center = max(0.0, min(1.0, (y_tl + bh / 2.0) / img_h))
    w_norm = max(0.0, min(1.0, bw / img_w))
    h_norm = max(0.0, min(1.0, bh / img_h))
    return x_center, y_center, w_norm, h_norm


def convert_split(annotation_path: Path, label_out_dir: Path) -> dict[str, int]:
    """Convert one COCO split to YOLO .txt labels."""
    coco = load_coco(annotation_path)
    cat_lookup = build_category_lookup(coco)
    img_lookup = build_image_lookup(coco)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    stats = {
        "images_total": len(img_lookup),
        "images_with_labels": 0,
        "annotations_mapped": 0,
        "annotations_dropped": 0,
    }

    for img_id, img_meta in img_lookup.items():
        fname = img_meta["file_name"]
        img_w, img_h = img_meta["width"], img_meta["height"]
        stem = Path(fname).stem

        lines: list[str] = []
        for ann in anns_by_image.get(img_id, []):
            raw_name = cat_lookup.get(ann["category_id"])
            if raw_name is None:
                stats["annotations_dropped"] += 1
                continue
            macro_id = raw_name_to_macro_id(raw_name)
            if macro_id is None:
                stats["annotations_dropped"] += 1
                continue
            xc, yc, wn, hn = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
            lines.append(f"{macro_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
            stats["annotations_mapped"] += 1

        label_path = label_out_dir / f"{stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        if lines:
            stats["images_with_labels"] += 1

    return stats


def convert_all() -> None:
    """Convert both train and val splits."""
    for split in ("train", "val"):
        ann_path = DATA_DIR / "annotations" / f"{split}.json"
        if not ann_path.exists():
            logger.warning("Annotation file not found: %s", ann_path)
            continue
        out_dir = LABEL_DIR / split
        logger.info("Converting %s -> %s", ann_path, out_dir)
        stats = convert_split(ann_path, out_dir)
        print(f"[{split}] {stats}")


def predownload_yolo() -> None:
    """Ensure YOLO weights exist in models/pretrained/."""
    cfg = YOLOConfig()
    weights_path = Path(cfg.model_weights)
    if weights_path.exists():
        print(f"YOLO weights already at {weights_path}")
        return
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading YOLO weights to {weights_path}...")
    from ultralytics import YOLO
    YOLO("yolo11n.pt")
    # ultralytics downloads to CWD; move if needed
    cwd_pt = Path("yolo11n.pt")
    if cwd_pt.exists() and not weights_path.exists():
        cwd_pt.rename(weights_path)
    print(f"YOLO weights ready at {weights_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("=== Phase 0: Data preprocessing + YOLO download ===")
    convert_all()
    predownload_yolo()
    print("=== Phase 0 complete ===")


if __name__ == "__main__":
    main()
