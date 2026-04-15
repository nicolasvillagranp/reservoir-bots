"""Phase 0 Milestone: verify COCO→YOLO conversion and visual bbox check."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# Allow imports from src/
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config import IMAGE_DIR, LABEL_DIR, MACRO_CLASSES, OUTPUT_DIR

# Colors per macro-class (BGR)
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 255),    # HUMAN — red
    1: (255, 165, 0),  # VEHICLE — blue-ish
    2: (0, 255, 255),  # OBSTACLE — yellow
    3: (0, 255, 0),    # CONTEXT — green
}


def find_nonempty_label(split: str = "train") -> tuple[Path, Path]:
    """Find first label file with at least one annotation and its matching image."""
    label_dir = LABEL_DIR / split
    image_dir = IMAGE_DIR / split

    for label_path in sorted(label_dir.glob("*.txt")):
        text = label_path.read_text().strip()
        if not text:
            continue
        # Match image by stem
        stem = label_path.stem
        for ext in (".jpg", ".jpeg", ".png"):
            img_path = image_dir / f"{stem}{ext}"
            if img_path.exists():
                return label_path, img_path
    raise FileNotFoundError(f"No non-empty label with matching image in {split}")


def parse_yolo_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse YOLO .txt label into list of (class_id, xc, yc, w, h) tuples."""
    entries: list[tuple[int, float, float, float, float]] = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        cls = int(parts[0])
        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        entries.append((cls, xc, yc, w, h))
    return entries


def yolo_to_abs(
    xc: float, yc: float, wn: float, hn: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """Un-normalize YOLO coords to absolute pixel (x1, y1, x2, y2)."""
    bw = wn * img_w
    bh = hn * img_h
    x1 = int((xc * img_w) - bw / 2)
    y1 = int((yc * img_h) - bh / 2)
    x2 = int((xc * img_w) + bw / 2)
    y2 = int((yc * img_h) + bh / 2)
    return x1, y1, x2, y2


def test_label_class_ids() -> None:
    """Assert all class IDs in a converted label are in {0,1,2,3}."""
    label_path, _ = find_nonempty_label("train")
    entries = parse_yolo_label(label_path)
    assert len(entries) > 0, "Label file is empty"
    for cls, xc, yc, w, h in entries:
        assert 0 <= cls <= 3, f"Invalid class ID {cls} in {label_path}"
        assert 0.0 <= xc <= 1.0, f"xc out of range: {xc}"
        assert 0.0 <= yc <= 1.0, f"yc out of range: {yc}"
        assert 0.0 <= w <= 1.0, f"w out of range: {w}"
        assert 0.0 <= h <= 1.0, f"h out of range: {h}"
    print(f"PASS: {label_path.name} — {len(entries)} boxes, all class IDs 0-3")


def test_draw_bboxes() -> None:
    """Draw bounding boxes on image and save for visual verification."""
    label_path, img_path = find_nonempty_label("train")
    entries = parse_yolo_label(label_path)

    img = cv2.imread(str(img_path))
    assert img is not None, f"Failed to load image: {img_path}"
    img_h, img_w = img.shape[:2]

    for cls, xc, yc, wn, hn in entries:
        x1, y1, x2, y2 = yolo_to_abs(xc, yc, wn, hn, img_w, img_h)
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label_text = f"{MACRO_CLASSES[cls]}({cls})"
        cv2.putText(img, label_text, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    out_path = OUTPUT_DIR / "test_p0_draw.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"PASS: Saved annotated image to {out_path}")


def test_multiple_labels_stats() -> None:
    """Spot-check that labels exist for both splits and classes span 0-3."""
    for split in ("train", "val"):
        label_dir = LABEL_DIR / split
        txt_files = list(label_dir.glob("*.txt"))
        assert len(txt_files) > 0, f"No label files in {label_dir}"

        # Sample first 200 non-empty labels and collect class distribution
        class_counts: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        checked = 0
        for lp in txt_files[:500]:
            text = lp.read_text().strip()
            if not text:
                continue
            for line in text.splitlines():
                cls = int(line.split()[0])
                assert 0 <= cls <= 3, f"Bad class {cls} in {lp}"
                class_counts[cls] += 1
            checked += 1
            if checked >= 200:
                break

        print(f"PASS [{split}]: {len(txt_files)} labels | class dist (sample): {class_counts}")


if __name__ == "__main__":
    test_label_class_ids()
    test_draw_bboxes()
    test_multiple_labels_stats()
    print("\n=== All Phase 0 tests passed ===")
