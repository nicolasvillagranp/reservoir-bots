"""Phase 2: Metric depth estimation via ZoeDepth.

Loads Intel/zoedepth-nyu-kitti from HuggingFace transformers.
Aggressively manages VRAM: load model -> infer -> move to CPU -> empty cache.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from config import PRETRAINED_DIR, DepthConfig


def estimate_depth(
    img_path: str | Path,
    cfg: DepthConfig | None = None,
) -> np.ndarray:
    """Estimate metric depth for a single image.

    Loads model to GPU, runs inference, moves everything to CPU,
    and clears VRAM immediately.

    Args:
        img_path: Path to input image.
        cfg: Depth model config. Defaults to DepthConfig().

    Returns:
        depth_map: np.ndarray of shape (H, W) with metric depth in meters.
    """
    cfg = cfg or DepthConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    image = Image.open(img_path).convert("RGB")
    orig_w, orig_h = image.size

    # Define your local cache directory
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    # Load model + processor and FORCE them to use the local folder
    print(f"Loading depth model {cfg.model_name} on {device}...")
    processor = AutoImageProcessor.from_pretrained(cfg.model_name, cache_dir=str(PRETRAINED_DIR))

    print("Processor loaded. Loading model weights...")
    model = AutoModelForDepthEstimation.from_pretrained(
        cfg.model_name, cache_dir=str(PRETRAINED_DIR)
    )
    model.to(device)
    model.eval()

    # Preprocess and infer
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth  # [1, h, w]

    print("Inference complete. Moving depth map to CPU and clearing VRAM...")

    # Resize to original image dimensions
    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(0),
        size=(orig_h, orig_w),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    # Move to CPU as numpy
    depth_np = depth.cpu().numpy().astype(np.float32)

    # Aggressive VRAM cleanup
    del model, inputs, outputs, predicted_depth, depth
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        depth_np.shape,
        depth_np.dtype,
        f"Depth range: {depth_np.min():.2f}m to {depth_np.max():.2f}m",
    )

    print(depth_np)

    return depth_np
