"""Run full inference pipeline on a single image and save all outputs.

Produces:
    outputs/<image_id>/visualization.png   — 1x2 subplot: YOLO detections + depth heatmap
    outputs/<image_id>/scene_3d.json       — 3D scene (objects, tags, proxy coordinates)
    outputs/<image_id>/prediction.json     — GNN action, confidence, reasoning

Usage:
    uv run python scripts/example_inference.py --image-id 21216
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# ── project imports ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_DIR,
    IMAGE_DIR,
    MACRO_CLASSES,
    FusionConfig,
    raw_name_to_macro_id,
)
from src.phases.phase1_vision import predict_objects_with_model
from src.phases.phase2_depth import load_depth_components, estimate_depth_with_components
from src.phases.phase4_symbolic import (
    SymbolicConfig,
    _approx_depth_from_bbox,
    _bin_depth,
    _bin_horizontal,
    build_scene_graph,
)
from src.phases.phase5_gnn import NavigationGNN
from src.main import gnn_predict, rule_based_action, format_reasoning

# ── color palette for YOLO boxes (per macro-class) ──────────────────────────
MACRO_COLORS: dict[int, str] = {
    0: "#FF3333",  # HUMAN — red
    1: "#3399FF",  # VEHICLE — blue
    2: "#FFAA00",  # OBSTACLE — orange
    3: "#33CC33",  # CONTEXT — green
}


# ── helpers ──────────────────────────────────────────────────────────────────


def find_image(image_id: int) -> tuple[dict, Path, str]:
    """Locate image metadata and file path across all splits.

    Returns:
        (image_meta, image_path, split)
    """
    for split in ("train", "val", "test"):
        ann_path = DATA_DIR / "annotations" / f"{split}.json"
        if not ann_path.exists():
            continue
        with open(ann_path) as f:
            ann = json.load(f)
        for img in ann["images"]:
            if img["id"] == image_id:
                img_path = IMAGE_DIR / split / img["file_name"]
                if not img_path.exists():
                    raise FileNotFoundError(f"Image file missing: {img_path}")
                return img, img_path, split

    raise ValueError(f"Image ID {image_id} not found in any split")


def draw_yolo_boxes(
    ax: plt.Axes,
    image: np.ndarray,
    detections: list[dict],
) -> None:
    """Draw YOLO bounding boxes with class labels on a matplotlib axes."""
    ax.imshow(image)
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        cid = det["class_id"]
        color = MACRO_COLORS.get(cid, "#FFFFFF")
        label = f"{det['class_name']} {det['confidence']:.2f}"
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 4, label,
            fontsize=7, color="white", weight="bold",
            bbox=dict(facecolor=color, alpha=0.7, pad=1, edgecolor="none"),
        )
    ax.set_title("YOLO Detections (filtered)")
    ax.axis("off")


# ── main ─────────────────────────────────────────────────────────────────────


def run(image_id: int) -> None:
    """Full inference for one image. Saves all artifacts to outputs/<image_id>/."""

    out_dir = PROJECT_ROOT / "outputs" / str(image_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Find image
    img_meta, img_path, split = find_image(image_id)
    img_w = int(img_meta.get("width", 640))
    img_h = int(img_meta.get("height", 640))
    print(f"Image {image_id} found in '{split}': {img_meta['file_name']} ({img_w}x{img_h})")

    image_pil = Image.open(img_path).convert("RGB")
    image_np = np.array(image_pil)

    # 2. YOLO detection
    from ultralytics import YOLO

    yolo_weights = PROJECT_ROOT / "models" / "finetuned" / "yolo" / "weights" / "best.pt"
    if not yolo_weights.exists():
        yolo_weights = PROJECT_ROOT / "models" / "pretrained" / "yolo11n.pt"
    print(f"Loading YOLO from {yolo_weights.name}...")
    yolo_model = YOLO(str(yolo_weights))

    fusion_cfg = FusionConfig()
    raw_dets = predict_objects_with_model(
        img_path, yolo_model, conf_thresh=0.25, min_area_frac=fusion_cfg.min_bbox_area_frac,
    )

    # Remap to macro-classes
    mapped_dets: list[dict] = []
    for det in raw_dets:
        macro_id = raw_name_to_macro_id(det["class_name"])
        if macro_id is not None:
            det["class_id"] = macro_id
            det["class_name"] = MACRO_CLASSES[macro_id]
            mapped_dets.append(det)

    print(f"  {len(mapped_dets)} detections after filtering + macro-class remap")

    # Free YOLO from GPU before loading depth model
    del yolo_model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Depth estimation (for heatmap visualization only)
    print("Loading ZoeDepth model...")
    processor, depth_model, depth_device = load_depth_components()
    depth_map = estimate_depth_with_components(img_path, processor, depth_model, depth_device)
    print(f"  Depth map: shape={depth_map.shape}, range=[{depth_map.min():.2f}, {depth_map.max():.2f}]m")

    del depth_model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Visualization — 1x2 subplot
    fig, (ax_yolo, ax_depth) = plt.subplots(1, 2, figsize=(14, 5))
    draw_yolo_boxes(ax_yolo, image_np, mapped_dets)
    im = ax_depth.imshow(depth_map, cmap="inferno")
    ax_depth.set_title("Depth Heatmap (ZoeDepth)")
    ax_depth.axis("off")
    fig.colorbar(im, ax=ax_depth, fraction=0.046, pad=0.04, label="Depth (m)")
    fig.suptitle(f"Image {image_id} — {img_meta['file_name']}", fontsize=12, weight="bold")
    fig.tight_layout()

    viz_path = out_dir / "visualization.png"
    fig.savefig(viz_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved visualization → {viz_path}")

    # 5. 3D scene JSON — uses bbox proxy for distance, NOT depth model
    sym_cfg = SymbolicConfig()
    fused: list[dict] = []
    scene_objects: list[dict] = []
    for det in mapped_dets:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        depth_m = _approx_depth_from_bbox(det["bbox_xyxy"], img_w, img_h)
        h_bin = _bin_horizontal(det["bbox_xywh"], img_w)
        d_bin = _bin_depth(depth_m, sym_cfg)

        fused_obj = {
            "class": det["class_name"],
            "class_id": det["class_id"],
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "bbox_xyxy": det["bbox_xyxy"],
            "depth_m": depth_m,
            "confidence": round(det["confidence"], 3),
        }
        fused.append(fused_obj)

        scene_objects.append({
            "tag": det["class_name"],
            "class_id": det["class_id"],
            "confidence": round(det["confidence"], 3),
            "bbox_xyxy": [round(v, 1) for v in det["bbox_xyxy"]],
            "horizontal_bin": h_bin,
            "depth_bin": d_bin,
            "depth_m": depth_m,
        })

    scene_text = build_scene_graph(fused, img_w, sym_cfg) if fused else ""

    scene_3d = {
        "image_id": image_id,
        "file_name": img_meta["file_name"],
        "image_size": {"width": img_w, "height": img_h},
        "depth_method": "bbox_proxy",
        "objects": scene_objects,
        "scene_graph": scene_text,
    }

    scene_path = out_dir / "scene_3d.json"
    with open(scene_path, "w") as f:
        json.dump(scene_3d, f, indent=2)
    print(f"  Saved 3D scene → {scene_path}")

    # 6. GNN prediction
    gnn_weights = PROJECT_ROOT / "models" / "gnn" / "navigation_gnn.pt"
    if gnn_weights.exists() and fused:
        gnn_model = NavigationGNN()
        gnn_model.load_state_dict(torch.load(gnn_weights, map_location="cpu", weights_only=True))
        gnn_model.eval()
        action, confidence, reasoning_objs = gnn_predict(fused, gnn_model, img_w)
        method = "gnn"
        print(f"  GNN prediction: {action} ({confidence:.1%})")
    elif fused:
        action, confidence, reasoning_objs = rule_based_action(fused, img_w)
        method = "rule_based"
        print(f"  Rule-based fallback: {action} ({confidence:.1%})")
    else:
        action, confidence, reasoning_objs = "CONTINUE", 0.95, []
        method = "no_detections"
        print("  No detections — defaulting to CONTINUE")

    reasoning_str = format_reasoning(reasoning_objs)

    prediction = {
        "image_id": image_id,
        "method": method,
        "action": action,
        "confidence": round(confidence, 3),
        "reasoning": reasoning_str,
        "reasoning_objects": reasoning_objs,
    }

    pred_path = out_dir / "prediction.json"
    with open(pred_path, "w") as f:
        json.dump(prediction, f, indent=2)
    print(f"  Saved prediction → {pred_path}")

    print(f"\nDone. All outputs in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-image inference demo")
    parser.add_argument(
        "--image-id", type=int, default=21216,
        help="Image ID from train/val/test annotations (default: 21216)",
    )
    args = parser.parse_args()
    run(args.image_id)
