"""Phase 4: Symbolic Graph Builder + Claude LLM Distillation.

Generates synthetic GNN training dataset by running fusion + Claude on
train/val images and saving labeled graphs to data/gnn_dataset/.

Usage: uv run python -m src.phases.phase4_symbolic
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import anthropic

from src.config import (
    DATA_DIR,
    FusionConfig,
    IMAGE_DIR,
    MACRO_CLASSES,
    MODE,
    OUTPUT_DIR,
    SymbolicConfig,
    YOLOConfig,
    raw_name_to_macro_id,
)

# ---------------------------------------------------------------------------
# Spatial binning helpers
# ---------------------------------------------------------------------------


def _bin_horizontal(bbox_xywh: list[float], img_w: int) -> str:
    """Bin object horizontal center into LEFT / CENTER / RIGHT."""
    x, _, w, _ = bbox_xywh
    cx = x + w / 2
    third = img_w / 3
    if cx < third:
        return "LEFT"
    elif cx < 2 * third:
        return "CENTER"
    return "RIGHT"


def _bin_depth(depth_m: float, cfg: SymbolicConfig) -> str:
    """Bin depth into CLOSE / MID / FAR."""
    if depth_m < cfg.depth_close:
        return "CLOSE"
    elif depth_m < cfg.depth_far:
        return "MID"
    return "FAR"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


@dataclass
class SceneNode:
    """A single object in the scene graph."""

    label: str
    macro_class: str
    h_bin: str
    d_bin: str
    depth_m: float


def build_scene_graph(
    fused_objects: list[dict],
    img_w: int = 640,
    cfg: SymbolicConfig | None = None,
) -> str:
    """Convert fused detections into a text scene description."""
    cfg = cfg or SymbolicConfig()

    class_counters: dict[str, int] = {}
    nodes: list[SceneNode] = []
    for obj in fused_objects:
        macro = obj.get("class", MACRO_CLASSES.get(obj.get("class_id", -1), "UNKNOWN"))
        class_counters[macro] = class_counters.get(macro, 0) + 1
        label = f"{macro}_{class_counters[macro]}"
        h_bin = _bin_horizontal(obj["bbox"], img_w)
        d_bin = _bin_depth(obj["depth_m"], cfg)
        nodes.append(
            SceneNode(
                label=label,
                macro_class=macro,
                h_bin=h_bin,
                d_bin=d_bin,
                depth_m=obj["depth_m"],
            )
        )

    lines: list[str] = []
    for node in nodes:
        lines.append(f"{node.label} is {node.h_bin} and {node.d_bin}.")

    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            if abs(a.depth_m - b.depth_m) < cfg.nearness_threshold and a.h_bin == b.h_bin:
                lines.append(f"{a.label} is NEAR {b.label}.")

    return " ".join(lines)


# ---------------------------------------------------------------------------
# Claude API distillation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a safety-critical navigation decision system for an autonomous robot.

Given a text description of a scene with objects and their spatial relations, output a JSON object with exactly two keys:
- "action": one of "STOP", "SLOW", or "CONTINUE"
- "reasoning_edges": a list of 1 to 3 strings, each corresponding to a node or edge from the scene that most influenced your decision. Try to output the least number of edges needed to justify the action.

Rules:
- reasoning_edges must reference actual nodes/edges from the input scene
- Maximum 3 reasoning_edges

Example Rules:
- STOP if any HUMAN is CLOSE or CENTER+MID
- SLOW if any HUMAN is MID, or any VEHICLE is CLOSE/MID
- CONTINUE only if no immediate danger

In general, understand the scene and use logical rules to determine the safest action. Prioritize human safety above all else, then robot safety, then efficiency. If in doubt, choose the safer action.

Example 1:
Input: "HUMAN_1 is CENTER and CLOSE. OBSTACLE_1 is RIGHT and FAR."
Output: {"action": "STOP", "reasoning_edges": ["HUMAN_1_CENTER_CLOSE"]}

Example 2:
Input: "HUMAN_1 is LEFT and MID. FORKLIFT_1 is LEFT and MID. HUMAN_1 is NEAR FORKLIFT_1. OBSTACLE_1 is CENTER and FAR. TRAFFIC_SIGN_1 is RIGHT and CLOSE."
Output: {"action": "SLOW", "reasoning_edges": ["HUMAN_1_LEFT_MID", "FORKLIFT_1_LEFT_MID", "HUMAN_1_NEAR_FORKLIFT_1"]}

Output ONLY the JSON object. No markdown, no explanation."""


def query_claude(scene_text: str, cfg: SymbolicConfig | None = None) -> dict:
    """Send scene graph to Claude and parse structured response."""
    cfg = cfg or SymbolicConfig()
    client = anthropic.Anthropic()

    message = client.messages.create(
        model=cfg.claude_model,
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": scene_text}],
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    result = json.loads(raw)
    if result.get("action") not in ("STOP", "SLOW", "CONTINUE"):
        raise ValueError(f"Invalid action: {result.get('action')}")
    if not isinstance(result.get("reasoning_edges"), list):
        raise ValueError("reasoning_edges must be a list")
    return result


# ---------------------------------------------------------------------------
# Rule-based labeling fallback (no API key needed)
# ---------------------------------------------------------------------------


def rule_based_label(fused_objects: list[dict], img_w: int = 640) -> dict:
    """Generate action + reasoning_edges via deterministic rules."""
    cfg = SymbolicConfig()
    action = "CONTINUE"
    edges: list[str] = []

    class_counters: dict[str, int] = {}
    for obj in fused_objects:
        macro = obj.get("class", "UNKNOWN")
        class_counters[macro] = class_counters.get(macro, 0) + 1
        label = f"{macro}_{class_counters[macro]}"
        h_bin = _bin_horizontal(obj["bbox"], img_w)
        d_bin = _bin_depth(obj["depth_m"], cfg)

        if macro == "HUMAN" and d_bin == "CLOSE":
            action = "STOP"
            edges.append(f"{label}_{h_bin}_{d_bin}")
        elif macro == "HUMAN" and h_bin == "CENTER" and d_bin == "MID":
            action = "STOP"
            edges.append(f"{label}_{h_bin}_{d_bin}")
        elif action != "STOP" and macro == "HUMAN" and d_bin == "MID":
            action = "SLOW"
            edges.append(f"{label}_{h_bin}_{d_bin}")
        elif action != "STOP" and macro == "VEHICLE" and d_bin in ("CLOSE", "MID"):
            action = "SLOW"
            edges.append(f"{label}_{h_bin}_{d_bin}")

    return {"action": action, "reasoning_edges": edges[:3] or ["CLEAR_PATH"]}


def _load_split_images(split: str, n_images: int | None) -> list[tuple[dict, Path]]:
    """Load image metadata + existing image paths for a split."""
    with open(DATA_DIR / "annotations" / f"{split}.json") as f:
        ann_data = json.load(f)

    cfg = SymbolicConfig()
    images = list(ann_data["images"])
    sample_fraction = max(0.0, min(1.0, cfg.dataset_fraction))

    if sample_fraction < 1.0 or n_images:
        random.seed(42)
        random.shuffle(images)

    if sample_fraction < 1.0 and images:
        keep = max(1, int(len(images) * sample_fraction))
        images = images[:keep]

    if n_images:
        images = images[:n_images]

    samples: list[tuple[dict, Path]] = []
    for img_meta in images:
        img_path = IMAGE_DIR / split / img_meta["file_name"]
        if img_path.exists():
            samples.append((img_meta, img_path))

    return samples


def _map_to_macro_classes(raw_dets: list[dict]) -> list[dict]:
    """Remap detector outputs into the 4 macro-classes."""
    mapped: list[dict] = []
    for det in raw_dets:
        macro_id = raw_name_to_macro_id(det["class_name"])
        if macro_id is None:
            continue
        mapped_det = dict(det)
        mapped_det["class_id"] = macro_id
        mapped_det["class_name"] = MACRO_CLASSES[macro_id]
        mapped.append(mapped_det)
    return mapped


def _samples_signature(samples: list[tuple[dict, Path]]) -> str:
    """Create stable signature for the sampled image IDs."""
    ids = ",".join(str(img_meta["id"]) for img_meta, _ in samples)
    return hashlib.sha1(ids.encode("utf-8")).hexdigest()


def _load_stage_checkpoint(path: Path, expected_meta: dict[str, object]) -> dict[int, list] | None:
    """Load stage checkpoint if metadata exactly matches expectations."""
    if not path.exists():
        return None

    try:
        with open(path) as f:
            payload = json.load(f)
    except Exception:
        return None

    meta = payload.get("meta")
    data = payload.get("data")
    if not isinstance(meta, dict) or not isinstance(data, dict):
        return None

    for k, v in expected_meta.items():
        if meta.get(k) != v:
            return None

    try:
        return {int(k): v for k, v in data.items()}
    except Exception:
        return None


def _save_stage_checkpoint(path: Path, meta: dict[str, object], data: dict[int, list]) -> None:
    """Write stage checkpoint payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_data = {str(k): v for k, v in data.items()}
    with open(path, "w") as f:
        json.dump({"meta": meta, "data": serializable_data}, f)


def _approx_depth_from_bbox(
    bbox_xyxy: list[float],
    img_w: int,
    img_h: int,
) -> float:
    """Fast geometric depth approximation from bbox position and size."""
    x1, y1, x2, y2 = bbox_xyxy
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    img_area = max(1.0, float(img_w * img_h))
    area_frac = min(1.0, (bw * bh) / img_area)
    bottom_norm = max(0.0, min(1.0, y2 / max(1.0, float(img_h))))

    # Favor lower/larger boxes as closer, then map to pseudo-metric meters.
    area_term = min(1.0, (area_frac / 0.15) ** 0.5)
    closeness = 0.65 * bottom_norm + 0.35 * area_term
    depth_m = 12.0 - 11.0 * closeness
    return round(max(1.0, min(12.0, depth_m)), 2)


def _generate_gnn_dataset_production(
    split: str,
    n_images: int | None,
    use_claude: bool,
    out_path: Path,
) -> Path:
    """Optimized production dataset generation with staged model execution."""
    import torch
    from src.phases.phase3_fusion import _center_crop_depth

    samples = _load_split_images(split, n_images)
    total = len(samples)
    cfg = SymbolicConfig()
    sample_sig = _samples_signature(samples)

    if total == 0:
        out_path.write_text("")
        print(f"Saved 0 labeled scenes to {out_path}")
        return out_path

    fusion_cfg = FusionConfig()

    yolo_cfg = YOLOConfig()
    ft_weights = Path(yolo_cfg.finetuned_weights)
    yolo_model_path = ft_weights if ft_weights.exists() else Path(yolo_cfg.model_weights)
    if not ft_weights.exists():
        print(
            "WARN: Fine-tuned YOLO weights missing in production; "
            f"falling back to base model: {yolo_model_path}"
        )

    print(
        f"[{split}] Production optimization enabled "
        f"({total} images, fraction={cfg.dataset_fraction * 100:.1f}%, "
        f"depth={cfg.depth_strategy})"
    )

    checkpoint_dir = DATA_DIR / "gnn_dataset" / "checkpoints"
    yolo_ckpt = checkpoint_dir / f"{split}_yolo_detections.json"
    depth_ckpt = checkpoint_dir / f"{split}_depth_values_{cfg.depth_strategy}.json"

    yolo_meta: dict[str, object] = {
        "stage": "yolo",
        "split": split,
        "sample_signature": sample_sig,
        "sample_count": total,
        "model_path": str(yolo_model_path),
    }

    detections_by_image_id: dict[int, list[dict]] | None = None
    yolo_skip_count = 0
    depth_skip_count = 0
    if cfg.enable_stage_checkpoints:
        detections_by_image_id = _load_stage_checkpoint(yolo_ckpt, yolo_meta)
        if detections_by_image_id is not None:
            sample_ids = {img_meta["id"] for img_meta, _ in samples}
            if sample_ids.issubset(detections_by_image_id.keys()):
                print(f"[{split}] Stage 1/3: Loaded YOLO checkpoint {yolo_ckpt}")
            else:
                detections_by_image_id = None

    if detections_by_image_id is None:
        from ultralytics import YOLO

        from src.phases.phase1_vision import predict_objects_with_model

        print(f"[{split}] Stage 1/3: YOLO detections with model {yolo_model_path}")
        detections_by_image_id = {}
        yolo_model = YOLO(str(yolo_model_path))
        for i, (img_meta, img_path) in enumerate(samples, start=1):
            print(f"  [YOLO][{i}/{total}] {img_meta['file_name']}...", end="\r")
            try:
                raw_dets = predict_objects_with_model(
                    img_path,
                    yolo_model,
                    conf_thresh=0.25,
                    min_area_frac=fusion_cfg.min_bbox_area_frac,
                )
            except ValueError as e:
                msg = str(e).replace("\n", " ").strip()
                if "need at least one array to stack" not in msg.lower():
                    raise
                yolo_skip_count += 1
                print(
                    f"\n  WARN: YOLO could not decode {img_meta['file_name']}: {msg}. "
                    "Skipping image."
                )
                detections_by_image_id[img_meta["id"]] = []
                continue
            detections_by_image_id[img_meta["id"]] = _map_to_macro_classes(raw_dets)
        print()

        del yolo_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if cfg.enable_stage_checkpoints:
            _save_stage_checkpoint(yolo_ckpt, yolo_meta, detections_by_image_id)
            print(f"[{split}] Saved YOLO checkpoint -> {yolo_ckpt}")

    assert detections_by_image_id is not None

    depth_meta: dict[str, object] = {
        "stage": "depth",
        "split": split,
        "sample_signature": sample_sig,
        "sample_count": total,
        "depth_strategy": cfg.depth_strategy,
    }
    depth_values_by_image_id: dict[int, list[float]] | None = None
    if cfg.enable_stage_checkpoints:
        depth_values_by_image_id = _load_stage_checkpoint(depth_ckpt, depth_meta)
        if depth_values_by_image_id is not None:
            sample_ids = {img_meta["id"] for img_meta, _ in samples}
            if sample_ids.issubset(depth_values_by_image_id.keys()):
                print(f"[{split}] Stage 2/3: Loaded depth checkpoint {depth_ckpt}")
            else:
                depth_values_by_image_id = None

    if depth_values_by_image_id is None:
        depth_values_by_image_id = {}
        if cfg.depth_strategy == "bbox_approx":
            print(f"[{split}] Stage 2/3: Depth via fast bbox approximation")
            for i, (img_meta, _) in enumerate(samples, start=1):
                print(f"  [DepthApprox][{i}/{total}] {img_meta['file_name']}...", end="\r")
                mapped_dets = detections_by_image_id.get(img_meta["id"], [])
                if not mapped_dets:
                    depth_values_by_image_id[img_meta["id"]] = []
                    continue

                img_w = int(img_meta.get("width", 640))
                img_h = int(img_meta.get("height", 640))
                depth_values_by_image_id[img_meta["id"]] = [
                    _approx_depth_from_bbox(det["bbox_xyxy"], img_w, img_h)
                    for det in mapped_dets
                ]
            print()

        elif cfg.depth_strategy == "model":
            from src.phases.phase2_depth import estimate_depth_with_components, load_depth_components

            print(f"[{split}] Stage 2/3: Depth inference (single model pass)")
            processor, depth_model, depth_device = load_depth_components()
            for i, (img_meta, img_path) in enumerate(samples, start=1):
                print(f"  [Depth][{i}/{total}] {img_meta['file_name']}...", end="\r")
                mapped_dets = detections_by_image_id.get(img_meta["id"], [])
                if not mapped_dets:
                    depth_values_by_image_id[img_meta["id"]] = []
                    continue

                try:
                    depth_map = estimate_depth_with_components(
                        img_path, processor, depth_model, depth_device,
                    )
                except OSError as e:
                    depth_skip_count += 1
                    msg = str(e).replace("\n", " ").strip()
                    print(
                        f"\n  WARN: Depth could not decode {img_meta['file_name']}: {msg}. "
                        "Skipping image."
                    )
                    depth_values_by_image_id[img_meta["id"]] = []
                    continue

                depth_values: list[float] = []
                for det in mapped_dets:
                    depth_m = _center_crop_depth(
                        depth_map,
                        det["bbox_xyxy"],
                        fusion_cfg.center_crop_fraction,
                    )
                    depth_values.append(round(depth_m, 2))
                depth_values_by_image_id[img_meta["id"]] = depth_values
            print()

            del depth_model, processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        else:
            raise ValueError(
                f"Unknown depth_strategy '{cfg.depth_strategy}'. "
                "Use 'bbox_approx' or 'model'."
            )

        if cfg.enable_stage_checkpoints:
            _save_stage_checkpoint(depth_ckpt, depth_meta, depth_values_by_image_id)
            print(f"[{split}] Saved depth checkpoint -> {depth_ckpt}")

    assert depth_values_by_image_id is not None

    print(f"[{split}] Stage 3/3: Fusion labels + scene graph serialization")
    scenes: list[dict] = []
    for i, (img_meta, _) in enumerate(samples, start=1):
        print(f"  [Label][{i}/{total}] {img_meta['file_name']}...", end="\r")
        mapped_dets = detections_by_image_id.get(img_meta["id"], [])
        depth_values = depth_values_by_image_id.get(img_meta["id"], [])
        if not mapped_dets or not depth_values:
            continue

        fused: list[dict] = []
        for det, depth_m in zip(mapped_dets, depth_values):
            x1, y1, x2, y2 = det["bbox_xyxy"]
            fused.append(
                {
                    "class": det["class_name"],
                    "class_id": det["class_id"],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "bbox_xyxy": det["bbox_xyxy"],
                    "depth_m": depth_m,
                    "confidence": round(det["confidence"], 3),
                }
            )

        img_w = img_meta.get("width", 640)
        scene_text = build_scene_graph(fused, img_w)

        if use_claude and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                label = query_claude(scene_text)
            except Exception as e:
                print(f"\n  Claude error: {e}, falling back to rules")
                label = rule_based_label(fused, img_w)
        else:
            label = rule_based_label(fused, img_w)

        scenes.append(
            {
                "image": img_meta["file_name"],
                "objects": fused,
                "scene_text": scene_text,
                "action": label["action"],
                "reasoning_edges": label["reasoning_edges"],
            }
        )
    print()

    with open(out_path, "w") as f:
        for scene in scenes:
            f.write(json.dumps(scene) + "\n")

    if yolo_skip_count or depth_skip_count:
        print(
            f"[{split}] Skipped unreadable/problematic images: "
            f"yolo={yolo_skip_count}, depth={depth_skip_count}"
        )

    print(f"Saved {len(scenes)} labeled scenes to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------


def generate_gnn_dataset(
    split: str = "train",
    n_images: int | None = None,
    use_claude: bool = False,
) -> Path:
    """Run fusion on images from *split* and save labeled scenes.

    Args:
        split: "train" or "val".
        n_images: Cap number of images (for test mode).
        use_claude: Use Claude API for labeling (else rule-based).

    Returns:
        Path to output JSONL file.
    """
    out_dir = DATA_DIR / "gnn_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"scenes_{split}.jsonl"

    if MODE == "production":
        return _generate_gnn_dataset_production(split, n_images, use_claude, out_path)

    from src.phases.phase3_fusion import fuse_scene

    images = _load_split_images(split, n_images)

    # Always use base YOLO — fuse_scene remaps to macro-classes.
    # Fine-tuned model may be too weak (e.g. 2-epoch smoke test).
    model_path = YOLOConfig().model_weights

    scenes: list[dict] = []
    for i, (img_meta, img_path) in enumerate(images):
        print(f"  [{split}][{i + 1}/{len(images)}] {img_meta['file_name']}...")
        fused = fuse_scene(img_path, model_path)
        if not fused:
            continue

        img_w = img_meta.get("width", 640)
        scene_text = build_scene_graph(fused, img_w)

        if use_claude and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                label = query_claude(scene_text)
            except Exception as e:
                print(f"  Claude error: {e}, falling back to rules")
                label = rule_based_label(fused, img_w)
        else:
            label = rule_based_label(fused, img_w)

        scenes.append(
            {
                "image": img_meta["file_name"],
                "objects": fused,
                "scene_text": scene_text,
                "action": label["action"],
                "reasoning_edges": label["reasoning_edges"],
            }
        )

    with open(out_path, "w") as f:
        for scene in scenes:
            f.write(json.dumps(scene) + "\n")

    print(f"Saved {len(scenes)} labeled scenes to {out_path}")
    return out_path


def main() -> None:
    """Generate synthetic GNN training dataset for both splits."""
    cfg = SymbolicConfig()
    n = 20 if MODE == "test" else None
    use_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))
    print(
        "=== Phase 4: Generating GNN dataset "
        f"(n={n or 'all'}/split, frac={cfg.dataset_fraction * 100:.1f}%, "
        f"depth={cfg.depth_strategy}, checkpoints={cfg.enable_stage_checkpoints}, "
        f"claude={use_claude}) ==="
    )
    for split in ("train", "val"):
        generate_gnn_dataset(split=split, n_images=n, use_claude=use_claude)
    print("=== Phase 4 complete ===")


if __name__ == "__main__":
    main()
