"""Phase 2: Metric depth estimation via ZoeDepth.

Usage: uv run python -m src.phases.phase2_depth
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from src.config import IMAGE_DIR, PRETRAINED_DIR, DepthConfig


def estimate_depth(
    img_path: str | Path,
    cfg: DepthConfig | None = None,
) -> np.ndarray:
    """Estimate metric depth for a single image.

    Loads model to GPU, infers, moves to CPU, clears VRAM.

    Returns:
        depth_map: np.ndarray of shape (H, W) with metric depth in meters.
    """
    cfg = cfg or DepthConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    image = Image.open(img_path).convert("RGB")
    orig_w, orig_h = image.size

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(
        cfg.model_name, cache_dir=str(PRETRAINED_DIR)
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        cfg.model_name, cache_dir=str(PRETRAINED_DIR)
    )
    model.to(device)
    model.eval()

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(0),
        size=(orig_h, orig_w),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_np = depth.cpu().numpy().astype(np.float32)

    del model, inputs, outputs, predicted_depth, depth
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return depth_np


def main() -> None:
    """Download depth model and verify on one image."""
    print("=== Phase 2: Depth model download + verification ===")

    # Find first available val image
    val_dir = IMAGE_DIR / "val"
    img_path = next(val_dir.glob("*.jpg"), None) or next(val_dir.glob("*.png"), None)
    if img_path is None:
        print("ERROR: No val images found")
        return

    print(f"Testing on {img_path.name}...")
    depth = estimate_depth(img_path)
    print(
        f"OK: shape={depth.shape}, "
        f"mean={depth.mean():.2f}m, range=[{depth.min():.2f}, {depth.max():.2f}]m"
    )
    print("=== Phase 2 complete ===")


if __name__ == "__main__":
    main()
