"""Phase 6: End-to-end pipeline — runs YOLO → Depth → Fusion → GNN on test
images and generates submission.json.

Usage:
    uv run python -m src.main --model models/finetuned/yolo/weights/best.pt
    uv run python -m src.main --model models/pretrained/yolo11n.pt --limit 10
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from src.config import (
    DATA_DIR,
    DETECTION_CATEGORIES,
    FINETUNED_DIR,
    IMAGE_DIR,
    MACRO_CLASSES,
    OUTPUT_DIR,
    PRETRAINED_DIR,
    FusionConfig,
    raw_name_to_macro_id,
)
from src.phases.phase1_vision import predict_objects
from src.phases.phase2_depth import estimate_depth
from src.phases.phase3_fusion import _center_crop_depth
from src.phases.phase4_symbolic import SymbolicConfig, _bin_depth, _bin_horizontal
from src.phases.phase5_gnn import (
    ACTION_TO_IDX,
    IDX_TO_ACTION,
    NavigationGNN,
    encode_node_features,
)

# ---------------------------------------------------------------------------
# Reasoning formatter
# ---------------------------------------------------------------------------

POSITION_LABELS = {
    "LEFT": "TO THE LEFT OF",
    "CENTER": "IN FRONT OF",
    "RIGHT": "TO THE RIGHT OF",
}


def format_reasoning(active_edges: list[dict]) -> str:
    """Convert active GNN edges into human-readable reasoning string.

    Args:
        active_edges: List of dicts with keys: class, h_bin, d_bin.

    Returns:
        Human-readable reasoning string (>= 10 chars).
    """
    if not active_edges:
        return "No immediate hazards detected. Path is clear."

    parts: list[str] = []
    for edge in active_edges[:3]:
        cls = edge["class"]
        d_bin = edge["d_bin"]
        h_bin = edge["h_bin"]
        pos = POSITION_LABELS.get(h_bin, "NEAR")
        parts.append(f"{cls} detected [{d_bin}] [{pos}] [ROBOT].")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Rule-based fallback (used when GNN not trained yet)
# ---------------------------------------------------------------------------


def rule_based_action(
    fused_objects: list[dict], img_w: int = 640,
) -> tuple[str, float, list[dict]]:
    """Deterministic rule-based action when GNN is unavailable.

    Returns:
        (action, confidence, reasoning_objects)
    """
    sym_cfg = SymbolicConfig()
    action = "CONTINUE"
    confidence = 0.9
    reasoning: list[dict] = []

    for obj in fused_objects:
        h_bin = _bin_horizontal(obj["bbox"], img_w)
        d_bin = _bin_depth(obj["depth_m"], sym_cfg)
        macro = obj["class"]

        info = {"class": macro, "h_bin": h_bin, "d_bin": d_bin, "depth_m": obj["depth_m"]}

        if macro == "HUMAN" and d_bin == "CLOSE":
            action = "STOP"
            confidence = 0.95
            reasoning.append(info)
        elif macro == "HUMAN" and h_bin == "CENTER" and d_bin == "MID":
            action = "STOP"
            confidence = 0.90
            reasoning.append(info)
        elif action != "STOP":
            if macro == "HUMAN" and d_bin == "MID":
                action = "SLOW"
                confidence = 0.85
                reasoning.append(info)
            elif macro == "VEHICLE" and d_bin in ("CLOSE", "MID"):
                action = "SLOW"
                confidence = 0.80
                reasoning.append(info)

    return action, confidence, reasoning[:3]


# ---------------------------------------------------------------------------
# GNN inference path
# ---------------------------------------------------------------------------


def gnn_predict(
    fused_objects: list[dict],
    model: NavigationGNN,
    img_w: int = 640,
) -> tuple[str, float, list[dict]]:
    """Run GNN inference on fused scene.

    Returns:
        (action, confidence, reasoning_objects)
    """
    sym_cfg = SymbolicConfig()

    node_info: list[dict] = []
    feats: list[list[float]] = []
    for obj in fused_objects:
        macro = obj["class"]
        h_bin = _bin_horizontal(obj["bbox"], img_w)
        d_bin = _bin_depth(obj["depth_m"], sym_cfg)
        feats.append(encode_node_features(macro, h_bin, d_bin, obj["depth_m"]))
        node_info.append({"class": macro, "h_bin": h_bin, "d_bin": d_bin})

    x = torch.tensor(feats, dtype=torch.float)
    n = len(feats)

    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = (
        torch.tensor([src, dst], dtype=torch.long)
        if src else torch.zeros(2, 0, dtype=torch.long)
    )

    data = Data(x=x, edge_index=edge_index, batch=torch.zeros(n, dtype=torch.long))

    model.eval()
    with torch.no_grad():
        action_logits, edge_logits = model(data)

    probs = torch.softmax(action_logits, dim=1).squeeze()
    action_idx = probs.argmax().item()
    action = IDX_TO_ACTION[action_idx]
    confidence = float(probs[action_idx])

    reasoning: list[dict] = []
    if edge_logits.numel() > 0:
        edge_probs = torch.sigmoid(edge_logits)
        top_k = min(3, edge_probs.numel())
        top_indices = edge_probs.topk(top_k).indices
        for idx in top_indices:
            src_node = src[idx.item()]
            reasoning.append(node_info[src_node])

    return action, confidence, reasoning


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def load_test_images() -> list[dict]:
    """Load test image metadata from test.json."""
    test_json = DATA_DIR / "annotations" / "test.json"
    with open(test_json) as f:
        data = json.load(f)
    return data["images"]


def run_pipeline(
    yolo_model_path: str | Path,
    gnn_model_path: str | Path | None = None,
    limit: int | None = None,
    output_path: str | Path | None = None,
) -> dict:
    """Run full pipeline on test images and generate submission JSON.

    Args:
        yolo_model_path: Path to YOLO .pt weights.
        gnn_model_path: Path to trained GNN .pt. If None, uses rule-based fallback.
        limit: Process only first N images (for testing).
        output_path: Where to write submission.json.

    Returns:
        Submission dict.
    """
    output_path = Path(output_path or (OUTPUT_DIR / "submission.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fusion_cfg = FusionConfig()

    gnn_model = None
    if gnn_model_path and Path(gnn_model_path).exists():
        gnn_model = NavigationGNN()
        gnn_model.load_state_dict(torch.load(gnn_model_path, map_location="cpu"))
        gnn_model.eval()
        print(f"GNN loaded from {gnn_model_path}")
    else:
        print("No GNN model — using rule-based fallback")

    test_images = load_test_images()
    if limit:
        test_images = test_images[:limit]

    detections_list: list[dict] = []
    predictions_list: list[dict] = []
    det_id = 1

    for img_meta in tqdm(test_images, desc="Processing test images"):
        image_id = img_meta["id"]
        fname = img_meta["file_name"]
        img_w = img_meta["width"]
        img_path = IMAGE_DIR / "test" / fname

        if not img_path.exists():
            print(f"WARN: {img_path} not found, skipping")
            continue

        # Phase 1: YOLO detection + remap to macro-classes
        raw_dets = predict_objects(
            img_path, yolo_model_path,
            conf_thresh=0.25, min_area_frac=fusion_cfg.min_bbox_area_frac,
        )

        mapped_dets: list[dict] = []
        for det in raw_dets:
            macro_id = raw_name_to_macro_id(det["class_name"])
            if macro_id is not None:
                det["class_id"] = macro_id
                det["class_name"] = MACRO_CLASSES[macro_id]
                mapped_dets.append(det)
        raw_dets = mapped_dets

        if not raw_dets:
            predictions_list.append({
                "image_id": image_id,
                "action": "CONTINUE",
                "confidence": 0.95,
                "reasoning": "No objects detected in scene. Path is clear.",
            })
            continue

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Phase 2: Depth
        depth_map = estimate_depth(img_path)

        # Phase 3: Fuse
        fused: list[dict] = []
        for det in raw_dets:
            depth_m = _center_crop_depth(
                depth_map, det["bbox_xyxy"], fusion_cfg.center_crop_fraction,
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

            detections_list.append({
                "id": det_id,
                "image_id": image_id,
                "category_id": det["class_id"],
                "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                "score": round(det["confidence"], 3),
            })
            det_id += 1

        # Phase 4/5: Action prediction
        if gnn_model:
            action, conf, reasoning_objs = gnn_predict(fused, gnn_model, img_w)
        else:
            action, conf, reasoning_objs = rule_based_action(fused, img_w)

        reasoning_str = format_reasoning(reasoning_objs)

        predictions_list.append({
            "image_id": image_id,
            "action": action,
            "confidence": round(conf, 3),
            "reasoning": reasoning_str,
        })

    submission = {
        "team_name": "theker_robotics",
        "detection_categories": DETECTION_CATEGORIES,
        "detections": detections_list,
        "predictions": predictions_list,
    }

    with open(output_path, "w") as f:
        json.dump(submission, f, indent=2)

    print(f"\nSubmission saved: {output_path}")
    print(f"  Images processed: {len(predictions_list)}")
    print(f"  Total detections: {len(detections_list)}")

    return submission


def main() -> None:
    """Run end-to-end pipeline with CLI args."""
    parser = argparse.ArgumentParser(description="Run end-to-end pipeline")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO .pt weights")
    parser.add_argument("--gnn", type=str, default=None, help="Path to GNN .pt weights")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N images")
    parser.add_argument("--output", type=str, default=None, help="Output submission.json path")
    args = parser.parse_args()

    run_pipeline(args.model, args.gnn, args.limit, args.output)


if __name__ == "__main__":
    main()
