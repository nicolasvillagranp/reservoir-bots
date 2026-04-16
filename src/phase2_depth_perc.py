"""Phase 2: Relative depth estimation via Depth Anything V2.

Loads depth-anything/Depth-Anything-V2-Small-hf from HuggingFace.
Outputs a normalized float array from 0.0 (closest) to 1.0 (furthest).
Aggressively manages VRAM: load model -> infer -> move to CPU -> empty cache.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from config import DepthConfig


def estimate_depth(
    img_path: str | Path,
    cfg: DepthConfig | None = None,
) -> np.ndarray:
    """Estimate relative depth for a single image.

    Loads model to GPU, runs inference, moves everything to CPU,
    and clears VRAM immediately.

    Args:
        img_path: Path to input image.
        cfg: Depth model config. Defaults to DepthConfig().

    Returns:
        depth_pct: np.ndarray of shape (H, W) with relative depth (0.0=close, 1.0=far).
    """
    cfg = cfg or DepthConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    image = Image.open(img_path).convert("RGB")
    orig_w, orig_h = image.size

    # Define your local cache directory
    local_cache = Path("models/pretrained")
    local_cache.mkdir(parents=True, exist_ok=True)

    # Load model + processor
    print(f"Loading depth model {cfg.model_name} on {device}...")
    processor = AutoImageProcessor.from_pretrained(cfg.model_name, cache_dir=str(local_cache))

    print("Processor loaded. Loading model weights...")
    model = AutoModelForDepthEstimation.from_pretrained(cfg.model_name, cache_dir=str(local_cache))
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

    # Normalize Depth Anything Output
    # Depth Anything outputs INVERSE depth (higher values = closer to camera)
    d_min = depth_np.min()
    d_max = depth_np.max()

    if d_max > d_min:
        # 1. Normalize to 0.0 - 1.0 (where 1.0 is currently the closest pixel)
        d_norm = (depth_np - d_min) / (d_max - d_min)
        # 2. Invert it to match the required format: 0.0 = close, 1.0 = far
        depth_pct = 1.0 - d_norm
    else:
        # Failsafe: if the image is a solid flat color, default to 0.5 (MID)
        depth_pct = np.full_like(depth_np, 0.5)

    print(
        depth_pct.shape,
        depth_pct.dtype,
        f"Depth scale: {depth_pct.min():.2f} (closest) to {depth_pct.max():.2f} (furthest)",
    )

    return depth_pct
