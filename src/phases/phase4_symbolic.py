"""Phase 4: Symbolic Graph Builder + Claude LLM Distillation.

Generates synthetic GNN training dataset by running fusion + Claude on
train/val images and saving labeled graphs to data/gnn_dataset/.

Usage: uv run python -m src.phases.phase4_symbolic
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import anthropic

from src.config import (
    DATA_DIR,
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
    from src.phases.phase3_fusion import fuse_scene

    out_dir = DATA_DIR / "gnn_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"scenes_{split}.jsonl"

    with open(DATA_DIR / "annotations" / f"{split}.json") as f:
        ann_data = json.load(f)

    images = ann_data["images"]
    if n_images:
        random.seed(42)
        random.shuffle(images)
        images = images[:n_images]

    # Always use base YOLO — fuse_scene remaps to macro-classes.
    # Fine-tuned model may be too weak (e.g. 2-epoch smoke test).
    model_path = YOLOConfig().model_weights

    scenes: list[dict] = []
    for i, img_meta in enumerate(images):
        img_path = IMAGE_DIR / split / img_meta["file_name"]
        if not img_path.exists():
            continue

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
    n = 20 if MODE == "test" else None
    use_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))
    print(f"=== Phase 4: Generating GNN dataset (n={n or 'all'}/split, claude={use_claude}) ===")
    for split in ("train", "val"):
        generate_gnn_dataset(split=split, n_images=n, use_claude=use_claude)
    print("=== Phase 4 complete ===")


if __name__ == "__main__":
    main()
