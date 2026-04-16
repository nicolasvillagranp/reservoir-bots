"""Phase 2 Milestone: verify ZoeDepth produces valid metric depth maps.

1. Run depth estimation on a test image.
2. Assert output shape matches (H, W) of input.
3. Assert mean depth is realistic (0.5 - 50.0 meters).
4. Generate heatmap subplots for images from different macro-classes.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import DATA_DIR, IMAGE_DIR, MACRO_CLASSES, OUTPUT_DIR, raw_name_to_macro_id
from src.phases.phase2_depth import estimate_depth


def pick_images_per_class(n_per_class: int = 1) -> dict[int, Path]:
    """Pick one val image per macro-class."""
    with open(DATA_DIR / "annotations" / "val.json") as f:
        coco = json.load(f)

    cat_lookup = {c["id"]: c["name"] for c in coco["categories"]}
    img_lookup = {img["id"]: img["file_name"] for img in coco["images"]}

    macro_to_fnames: dict[int, list[str]] = defaultdict(list)
    for ann in coco["annotations"]:
        mid = raw_name_to_macro_id(cat_lookup[ann["category_id"]])
        if mid is not None:
            macro_to_fnames[mid].append(img_lookup[ann["image_id"]])

    picked: dict[int, Path] = {}
    for mid in range(4):
        candidates = list(set(macro_to_fnames[mid]))
        random.seed(77 + mid)
        random.shuffle(candidates)
        for fname in candidates:
            p = IMAGE_DIR / "val" / fname
            if p.exists():
                picked[mid] = p
                break

    return picked


def test_depth_shape_and_values() -> None:
    """Assert depth map shape matches image and values are realistic."""
    print("\n--- test_depth_shape_and_values ---")
    images = pick_images_per_class()
    # Use first available image
    img_path = next(iter(images.values()))

    img = cv2.imread(str(img_path))
    assert img is not None, f"Cannot load {img_path}"
    img_h, img_w = img.shape[:2]

    depth = estimate_depth(img_path)

    assert depth.shape == (img_h, img_w), (
        f"Shape mismatch: depth={depth.shape} vs image=({img_h}, {img_w})"
    )
    assert depth.dtype == np.float32, f"Expected float32, got {depth.dtype}"

    mean_depth = float(np.mean(depth))
    assert 0.5 <= mean_depth <= 50.0, (
        f"Mean depth {mean_depth:.2f}m outside realistic range [0.5, 50.0]"
    )
    print(
        f"PASS: {img_path.name} -> depth shape {depth.shape}, "
        f"mean={mean_depth:.2f}m, min={depth.min():.2f}m, max={depth.max():.2f}m"
    )


def test_depth_heatmaps() -> None:
    """Generate depth heatmap subplots for images across macro-classes."""
    print("\n--- test_depth_heatmaps ---")
    images = pick_images_per_class()

    n = len(images)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, (mid, img_path) in enumerate(sorted(images.items())):
        print(f"  Processing {MACRO_CLASSES[mid]}: {img_path.name}...")

        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        depth = estimate_depth(img_path)

        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_title(f"{MACRO_CLASSES[mid]} — {img_path.name}", fontsize=9)
        axes[row, 0].axis("off")

        im = axes[row, 1].imshow(depth, cmap="inferno")
        axes[row, 1].set_title(
            f"Depth (mean={np.mean(depth):.1f}m, max={np.max(depth):.1f}m)", fontsize=9
        )
        axes[row, 1].axis("off")
        fig.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "test_p2_depth.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    assert out_path.exists(), f"savefig failed — {out_path} not on disk"
    print(f"PASS: Saved depth heatmaps to {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    test_depth_shape_and_values()
    test_depth_heatmaps()
    print("\n=== All Phase 2 tests passed ===")
