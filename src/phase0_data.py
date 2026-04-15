"""Phase 0: Parse COCO annotations and convert to YOLO label format.

Reads train.json / val.json from data/annotations/.
Writes per-image .txt files to data/labels/{train,val}/.
Images stay in place — YOLO finds them via parallel folder structure.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from config import (
    DATA_DIR,
    LABEL_DIR,
    MACRO_CLASSES,
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
    """Convert COCO [x_tl, y_tl, w, h] (absolute) to YOLO [x_c, y_c, w, h] (normalized).

    Args:
        bbox: COCO format [x_top_left, y_top_left, width, height] in pixels.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        (x_center, y_center, width, height) all in [0, 1].
    """
    x_tl, y_tl, bw, bh = bbox
    x_center = (x_tl + bw / 2.0) / img_w
    y_center = (y_tl + bh / 2.0) / img_h
    w_norm = bw / img_w
    h_norm = bh / img_h
    # Clamp to [0, 1] — guards against annotations slightly out of bounds
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))
    return x_center, y_center, w_norm, h_norm


def convert_split(
    annotation_path: Path,
    label_out_dir: Path,
) -> dict[str, int]:
    """Convert one COCO split to YOLO .txt labels.

    Args:
        annotation_path: Path to train.json or val.json.
        label_out_dir: Directory to write .txt label files.

    Returns:
        Stats dict with keys: images_total, images_with_labels,
        annotations_mapped, annotations_dropped.
    """
    coco = load_coco(annotation_path)
    cat_lookup = build_category_lookup(coco)
    img_lookup = build_image_lookup(coco)

    label_out_dir.mkdir(parents=True, exist_ok=True)

    # Group annotations by image_id
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
            raw_cat_id = ann["category_id"]
            raw_name = cat_lookup.get(raw_cat_id)
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

        # Write .txt even if empty (YOLO treats empty file as "no objects")
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

        logger.info(
            "[%s] images=%d (with_labels=%d) | mapped=%d dropped=%d",
            split,
            stats["images_total"],
            stats["images_with_labels"],
            stats["annotations_mapped"],
            stats["annotations_dropped"],
        )
        # Also print for quick feedback when run as script
        print(f"[{split}] {stats}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    convert_all()
