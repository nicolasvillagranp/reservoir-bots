"""Phase 3: 3D Scene Fusion — combine YOLO detections with ZoeDepth.

Strategy 3 (Center-Crop): for each bbox, sample the central 20% region
from the depth map and take the median depth as the object's distance.

VRAM note: runs YOLO first, extracts detections, then unloads before
running ZoeDepth. Both models never coexist in VRAM.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch

from config import MACRO_CLASSES, FusionConfig
from phase1_vision import predict_objects
from phase2_depth import estimate_depth


def _center_crop_depth(
    depth_map: np.ndarray,
    bbox_xyxy: list[float],
    crop_frac: float,
) -> float:
    """Extract median depth from the central crop of a bounding box.

    Args:
        depth_map: Full depth map (H, W) in meters.
        bbox_xyxy: [x1, y1, x2, y2] in absolute pixels.
        crop_frac: Fraction of bbox to keep (0.2 = central 20%).

    Returns:
        Median depth in meters for the center crop region.
    """
    x1, y1, x2, y2 = bbox_xyxy
    bw = x2 - x1
    bh = y2 - y1

    # Central crop offsets
    margin_x = bw * (1 - crop_frac) / 2
    margin_y = bh * (1 - crop_frac) / 2

    cx1 = int(x1 + margin_x)
    cy1 = int(y1 + margin_y)
    cx2 = int(x2 - margin_x)
    cy2 = int(y2 - margin_y)

    # Clamp to image bounds
    h, w = depth_map.shape
    cx1 = max(0, min(cx1, w - 1))
    cx2 = max(cx1 + 1, min(cx2, w))
    cy1 = max(0, min(cy1, h - 1))
    cy2 = max(cy1 + 1, min(cy2, h))

    crop = depth_map[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        # Fallback: single pixel at center
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cx = max(0, min(cx, w - 1))
        cy = max(0, min(cy, h - 1))
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

    Args:
        img_path: Path to input image.
        model_path: Path to fine-tuned YOLO .pt weights.
        cfg: Fusion parameters.
        conf_thresh: YOLO confidence threshold.

    Returns:
        List of dicts: {"class": str, "class_id": int, "bbox": [x,y,w,h],
                        "bbox_xyxy": [x1,y1,x2,y2], "depth_m": float,
                        "confidence": float}
    """
    cfg = cfg or FusionConfig()

    # Phase 1: YOLO detection
    print(f"Running YOLO on {img_path} with model {model_path}...")
    detections = predict_objects(
        img_path,
        model_path,
        conf_thresh=conf_thresh,
        min_area_frac=cfg.min_bbox_area_frac,
    )
    print(f"YOLO detected {len(detections)} objects.")

    # Free YOLO from VRAM before loading depth model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not detections:
        return []

    # Phase 2: Depth estimation
    print("Running ZoeDepth for depth estimation...")
    depth_map = estimate_depth(img_path)

    # Phase 3: Fuse — center-crop depth for each detection
    fused: list[dict] = []
    for det in detections:
        depth_m = _center_crop_depth(
            depth_map,
            det["bbox_xyxy"],
            cfg.center_crop_fraction,
        )
        x1, y1, x2, y2 = det["bbox_xyxy"]
        fused.append(
            {
                "class": det["class_name"],
                "class_id": det["class_id"],
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # [x, y, w, h]
                "bbox_xyxy": det["bbox_xyxy"],
                "depth_m": round(depth_m, 2),
                "confidence": round(det["confidence"], 3),
            }
        )

    return fused
