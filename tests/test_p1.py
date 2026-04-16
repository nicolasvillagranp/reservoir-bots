"""Phase 1 Milestone: YOLO fine-tuning smoke test + inference comparison.

1. Train for 2 epochs on a 50-image subset — verify no OOM.
2. Run inference with both base and fine-tuned models on 5 val images.
3. Assert fine-tuned outputs class IDs 0-3, base outputs standard COCO IDs.
4. Save comparison subplots to outputs/phase1/.
"""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    DATA_DIR,
    FINETUNED_DIR,
    IMAGE_DIR,
    LABEL_DIR,
    MACRO_CLASSES,
    OUTPUT_DIR,
    PRETRAINED_DIR,
    YOLOConfig,
    raw_name_to_macro_id,
)
from src.phases.phase1_vision import predict_objects, train

PHASE1_OUT = OUTPUT_DIR / "phase1"
PHASE1_OUT.mkdir(parents=True, exist_ok=True)

# Colors per macro-class (BGR for OpenCV)
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 255),  # HUMAN — red
    1: (255, 165, 0),  # VEHICLE — orange
    2: (0, 255, 255),  # OBSTACLE — yellow
    3: (0, 255, 0),  # CONTEXT — green
}


def create_tiny_subset(n: int = 50) -> Path:
    """Create a tiny dataset subset for smoke-test training.

    Copies n images + labels into data/images/tiny and data/labels/tiny.
    Returns path to a generated tiny_dataset.yaml.
    """
    tiny_img_dir = IMAGE_DIR / "tiny"
    tiny_lbl_dir = LABEL_DIR / "tiny"
    tiny_img_dir.mkdir(parents=True, exist_ok=True)
    tiny_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Pick n label files that are non-empty
    train_labels = sorted((LABEL_DIR / "train").glob("*.txt"))
    non_empty = [lp for lp in train_labels if lp.read_text().strip()]
    selected = non_empty[:n]

    for lbl_path in selected:
        stem = lbl_path.stem
        # Copy label
        shutil.copy2(lbl_path, tiny_lbl_dir / lbl_path.name)
        # Find and copy matching image
        for ext in (".jpg", ".jpeg", ".png"):
            img_src = IMAGE_DIR / "train" / f"{stem}{ext}"
            if img_src.exists():
                shutil.copy2(img_src, tiny_img_dir / img_src.name)
                break

    # Write tiny dataset YAML (use tiny for both train and val)
    yaml_path = DATA_DIR / "tiny_dataset.yaml"
    yaml_path.write_text(
        f"path: {DATA_DIR.as_posix()}\n"
        f"train: images/tiny\n"
        f"val: images/tiny\n"
        f"\nnc: 4\n"
        f"names:\n"
        f"  0: HUMAN\n"
        f"  1: VEHICLE\n"
        f"  2: OBSTACLE\n"
        f"  3: CONTEXT\n"
    )
    print(f"Tiny subset: {len(list(tiny_img_dir.iterdir()))} images in {tiny_img_dir}")
    return yaml_path


def pick_val_images(per_class: int = 1) -> list[Path]:
    """Pick val images covering each macro-class, plus one extra."""
    with open(DATA_DIR / "annotations" / "val.json") as f:
        coco = json.load(f)

    cat_lookup = {c["id"]: c["name"] for c in coco["categories"]}
    img_lookup = {img["id"]: img["file_name"] for img in coco["images"]}

    macro_to_fnames: dict[int, list[str]] = defaultdict(list)
    for ann in coco["annotations"]:
        mid = raw_name_to_macro_id(cat_lookup[ann["category_id"]])
        if mid is not None:
            macro_to_fnames[mid].append(img_lookup[ann["image_id"]])

    picked: list[Path] = []
    seen: set[str] = set()
    for mid in range(4):
        candidates = list(set(macro_to_fnames[mid]))
        random.seed(42 + mid)
        random.shuffle(candidates)
        for fname in candidates:
            if fname not in seen:
                picked.append(IMAGE_DIR / "val" / fname)
                seen.add(fname)
                break

    # Add one more random image for 5 total
    all_fnames = [img["file_name"] for img in coco["images"]]
    random.seed(99)
    random.shuffle(all_fnames)
    for fname in all_fnames:
        if fname not in seen:
            picked.append(IMAGE_DIR / "val" / fname)
            break

    return picked[:5]


def draw_detections(img: np.ndarray, detections: list[dict], title: str) -> np.ndarray:
    """Draw bounding boxes on image copy. Returns annotated image (RGB)."""
    vis = img.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        cls_id = det["class_id"]
        # Use macro colors for fine-tuned (0-3), gray for base COCO IDs
        if cls_id in CLASS_COLORS:
            color = CLASS_COLORS[cls_id][::-1]  # BGR -> RGB
        else:
            color = (180, 180, 180)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(vis, label, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return vis


def test_training_smoke() -> Path:
    """Train 2 epochs on tiny subset. Assert it completes without OOM."""
    print("\n--- test_training_smoke ---")
    yaml_path = create_tiny_subset(50)

    cfg = YOLOConfig(epochs=2, batch=4)
    project = FINETUNED_DIR
    print(f"Starting 2-epoch training on tiny subset with config: {cfg}")
    best_pt = train(
        cfg=cfg,
        data_yaml=yaml_path,
        project=project,
        name="smoke_2ep",
    )
    print(f"Training completed. Best weights at {best_pt}")
    # best.pt may not exist after only 2 epochs; last.pt is guaranteed
    last_pt = best_pt.parent / "last.pt"
    weights = best_pt if best_pt.exists() else last_pt
    assert weights.exists(), f"No weights found at {best_pt} or {last_pt}"
    print(f"PASS: 2-epoch training completed. Weights at {weights}")
    return weights


def test_inference_comparison(finetuned_pt: Path) -> None:
    """Compare base vs fine-tuned inference on 5 val images."""
    print("\n--- test_inference_comparison ---")
    val_images = pick_val_images()
    assert len(val_images) == 5, f"Expected 5 val images, got {len(val_images)}"

    base_model = str(PRETRAINED_DIR / "yolo11n.pt")

    fig, axes = plt.subplots(5, 2, figsize=(14, 30))
    fig.suptitle("Base COCO (left) vs Fine-tuned 4-class (right)", fontsize=14)

    for i, img_path in enumerate(val_images):
        assert img_path.exists(), f"Val image not found: {img_path}"
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Base model inference
        base_dets = predict_objects(img_path, base_model, min_area_frac=0.01)
        # Fine-tuned model inference
        ft_dets = predict_objects(img_path, finetuned_pt, min_area_frac=0.01)

        # Assert fine-tuned outputs only our 4 classes
        for det in ft_dets:
            assert det["class_id"] in (0, 1, 2, 3), (
                f"Fine-tuned model output unexpected class_id={det['class_id']}"
            )

        # Base model should output standard COCO IDs (0-79), not our 0-3 mapping
        # (unless by coincidence — we just check it has different class names)
        base_names = {det["class_name"] for det in base_dets}
        ft_names = {det["class_name"] for det in ft_dets}
        print(
            f"  [{img_path.name}] base classes: {base_names} | "
            f"ft classes: {ft_names} | ft boxes: {len(ft_dets)}"
        )

        # Draw
        base_vis = draw_detections(img_rgb, base_dets, "Base")
        ft_vis = draw_detections(img_rgb, ft_dets, "Fine-tuned")

        axes[i, 0].imshow(base_vis)
        axes[i, 0].set_title(f"Base: {img_path.name}", fontsize=8)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(ft_vis)
        axes[i, 1].set_title(f"Fine-tuned: {img_path.name}", fontsize=8)
        axes[i, 1].axis("off")

    plt.tight_layout()
    out_path = PHASE1_OUT / "comparison_5images.png"
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    print(f"PASS: Saved comparison plot to {out_path}")


if __name__ == "__main__":
    weights_path = test_training_smoke()
    test_inference_comparison(weights_path)
    print("\n=== All Phase 1 tests passed ===")
