"""Phase 3: 3D Scene Fusion — combine YOLO detections with ZoeDepth.

Usage: uv run python -m src.phases.phase3_fusion
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch

from src.config import (
    FINETUNED_DIR,
    IMAGE_DIR,
    MACRO_CLASSES,
    FusionConfig,
    YOLOConfig,
    raw_name_to_macro_id,
)
from src.phases.phase1_vision import predict_objects
from src.phases.phase2_depth import estimate_depth


def _center_crop_depth(
    depth_map: np.ndarray,
    bbox_xyxy: list[float],
    crop_frac: float,
) -> float:
    """Extract median depth from the central crop of a bounding box."""
    x1, y1, x2, y2 = bbox_xyxy
    bw, bh = x2 - x1, y2 - y1
    margin_x = bw * (1 - crop_frac) / 2
    margin_y = bh * (1 - crop_frac) / 2

    h, w = depth_map.shape
    cx1 = max(0, min(int(x1 + margin_x), w - 1))
    cx2 = max(cx1 + 1, min(int(x2 - margin_x), w))
    cy1 = max(0, min(int(y1 + margin_y), h - 1))
    cy2 = max(cy1 + 1, min(int(y2 - margin_y), h))

    crop = depth_map[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        cx = max(0, min(int((x1 + x2) / 2), w - 1))
        cy = max(0, min(int((y1 + y2) / 2), h - 1))
        return float(depth_map[cy, cx])

    return float(np.median(crop))


def fuse_scene(
    img_path: str | Path,
    model_path: str | Path,
    cfg: FusionConfig | None = None,
    conf_thresh: float = 0.25,
) -> list[dict]:
    """Run full YOLO + Depth fusion on a single image.

    Runs YOLO first, clears VRAM, then runs ZoeDepth.

    Returns:
        List of dicts with class, class_id, bbox, bbox_xyxy, depth_m, confidence.
    """
    cfg = cfg or FusionConfig()

    raw_dets = predict_objects(
        img_path, model_path, conf_thresh=conf_thresh,
        min_area_frac=cfg.min_bbox_area_frac,
    )

    # Remap COCO class names → macro-classes (handles both base and fine-tuned)
    detections: list[dict] = []
    for det in raw_dets:
        macro_id = raw_name_to_macro_id(det["class_name"])
        if macro_id is not None:
            det["class_id"] = macro_id
            det["class_name"] = MACRO_CLASSES[macro_id]
            detections.append(det)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not detections:
        return []

    depth_map = estimate_depth(img_path)

    fused: list[dict] = []
    for det in detections:
        depth_m = _center_crop_depth(
            depth_map, det["bbox_xyxy"], cfg.center_crop_fraction,
        )
        x1, y1, x2, y2 = det["bbox_xyxy"]
        fused.append({
            "class": det["class_name"],
            "class_id": det["class_id"],
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "bbox_xyxy": det["bbox_xyxy"],
            "depth_m": round(depth_m, 2),
            "confidence": round(det["confidence"], 3),
        })

    return fused


def main() -> None:
    """Test fusion on one val image."""
    print("=== Phase 3: Fusion verification ===")

    # Use fine-tuned model if available, else pretrained
    ft_best = FINETUNED_DIR / "yolo" / "weights" / "best.pt"
    model_path = str(ft_best) if ft_best.exists() else YOLOConfig().model_weights

    val_dir = IMAGE_DIR / "val"
    img_path = next(val_dir.glob("*.jpg"), None) or next(val_dir.glob("*.png"), None)
    if img_path is None:
        print("ERROR: No val images found")
        return

    print(f"Fusing {img_path.name} with model {model_path}...")
    fused = fuse_scene(img_path, model_path)
    print(f"Detected {len(fused)} objects:")
    for obj in fused:
        print(f"  {obj['class']} depth={obj['depth_m']:.1f}m conf={obj['confidence']:.2f}")
    print("=== Phase 3 complete ===")


if __name__ == "__main__":
    main()
