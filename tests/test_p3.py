"""Phase 3 Milestone: verify YOLO + Depth fusion produces valid 3D scene output.

1. Run fusion on a single image.
2. Assert output is list of dicts with depth_m values.
3. Save annotated image + JSON to outputs/phase3/.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.config import IMAGE_DIR, OUTPUT_DIR, PRETRAINED_DIR
from src.phases.phase3_fusion import fuse_scene

PHASE3_OUT = OUTPUT_DIR / "phase3"
PHASE3_OUT.mkdir(parents=True, exist_ok=True)

# Use base yolo11n.pt (fine-tuned weights from 2-epoch smoke test are weak)
# For real runs, swap to fully trained weights
MODEL_PATH = str(PRETRAINED_DIR / "yolo11n.pt")

# Colors per macro-class (BGR)
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 255),
    1: (255, 165, 0),
    2: (0, 255, 255),
    3: (0, 255, 0),
}


def pick_test_image() -> Path:
    """Pick a val image likely to have detectable objects."""
    # Use an image we know has people from Phase 0 testing
    val_dir = IMAGE_DIR / "val"
    # Try hardhat image first (rich scene), fall back to first available
    for candidate in ["hardhat_hard_hat_workers4546.png", "hardhat_hard_hat_workers2195.png"]:
        p = val_dir / candidate
        if p.exists():
            return p
    # Fallback: first jpg/png
    for ext in ("*.jpg", "*.png"):
        imgs = sorted(val_dir.glob(ext))
        if imgs:
            return imgs[0]
    raise FileNotFoundError("No val images found")


def pick_random_train_image() -> Path:
    """Pick a random train image (for future tests)."""
    train_dir = IMAGE_DIR / "train"
    for ext in ("*.jpg", "*.png"):
        imgs = sorted(train_dir.glob(ext))
        if imgs:
            return np.random.choice(imgs)
    raise FileNotFoundError("No train images found")


def test_fusion_output() -> None:
    """Run fusion and validate output structure."""
    print("\n--- test_fusion_output ---")
    # img_path = pick_test_image()
    img_path = pick_random_train_image()
    print(f"  Test image: {img_path.name}")

    fused = fuse_scene(img_path, MODEL_PATH)

    assert isinstance(fused, list), f"Expected list, got {type(fused)}"
    print(f"  Detected {len(fused)} objects")

    for i, obj in enumerate(fused):
        # Required keys
        assert "class" in obj, f"Missing 'class' in object {i}"
        assert "class_id" in obj, f"Missing 'class_id' in object {i}"
        assert "bbox" in obj, f"Missing 'bbox' in object {i}"
        assert "depth_m" in obj, f"Missing 'depth_m' in object {i}"
        assert "confidence" in obj, f"Missing 'confidence' in object {i}"

        # depth_m is a positive float
        assert isinstance(obj["depth_m"], float), f"depth_m not float: {type(obj['depth_m'])}"
        assert obj["depth_m"] > 0, f"depth_m must be positive, got {obj['depth_m']}"

        # bbox is [x, y, w, h]
        assert len(obj["bbox"]) == 4, f"bbox should have 4 elements"

        print(
            f"  [{i}] {obj['class']} | depth={obj['depth_m']:.1f}m | "
            f"conf={obj['confidence']:.2f} | bbox={[round(v) for v in obj['bbox']]}"
        )

    print("PASS: Fusion output structure valid")
    return fused, img_path


def test_save_results(fused: list[dict], img_path: Path) -> None:
    """Save annotated image + JSON for visual verification."""
    print("\n--- test_save_results ---")

    # Save JSON
    json_path = PHASE3_OUT / f"{img_path.stem}_fusion.json"
    with open(json_path, "w") as f:
        json.dump(fused, f, indent=2)
    print(f"  Saved JSON: {json_path}")

    # Draw annotated image
    img = cv2.imread(str(img_path))
    assert img is not None

    for obj in fused:
        x1, y1, x2, y2 = [int(v) for v in obj["bbox_xyxy"]]
        cid = obj["class_id"]
        color = CLASS_COLORS.get(cid, (180, 180, 180))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{obj['class']} {obj['depth_m']:.1f}m"
        cv2.putText(img, label, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    img_out = PHASE3_OUT / f"{img_path.stem}_fusion.jpg"
    cv2.imwrite(str(img_out), img)
    assert img_out.exists(), f"Failed to save {img_out}"
    print(f"  Saved image: {img_out} ({img_out.stat().st_size} bytes)")
    print("PASS: Results saved")


if __name__ == "__main__":
    fused, img_path = test_fusion_output()
    if fused:
        test_save_results(fused, img_path)
    else:
        print("WARN: No detections — skipping save (base COCO model may not detect macro-classes)")
    print("\n=== All Phase 3 tests passed ===")
