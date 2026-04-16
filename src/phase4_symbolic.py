"""Phase 4: Symbolic Graph Builder + Claude LLM Distillation.

Converts fused 3D scene objects into a text graph, then queries Claude
for navigation action + reasoning edges.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import anthropic

from config import MACRO_CLASSES, SymbolicConfig

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

    label: str  # e.g. "HUMAN_1"
    macro_class: str  # e.g. "HUMAN"
    h_bin: str  # LEFT / CENTER / RIGHT
    d_bin: str  # CLOSE / MID / FAR
    depth_m: float


def build_scene_graph(
    fused_objects: list[dict],
    img_w: int = 640,
    cfg: SymbolicConfig | None = None,
) -> str:
    """Convert fused detections into a text scene description.

    Generates self-relations (position + distance) and cross-relations
    (NEAR pairs) for objects.

    Args:
        fused_objects: Output from phase3_fusion.fuse_scene().
        img_w: Image width for horizontal binning.
        cfg: Symbolic config for thresholds.

    Returns:
        Multi-line text scene description for Claude.
    """
    cfg = cfg or SymbolicConfig()

    # Build nodes with unique labels per class
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

    # Self-relations
    lines: list[str] = []
    for node in nodes:
        lines.append(f"{node.label} is {node.h_bin} and {node.d_bin}.")

    # Cross-relations: NEAR pairs
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


def query_claude(
    scene_text: str,
    cfg: SymbolicConfig | None = None,
) -> dict:
    """Send scene graph to Claude and parse structured response.

    Args:
        scene_text: Text scene from build_scene_graph().
        cfg: Config with model name.

    Returns:
        Dict with "action" and "reasoning_edges" keys.

    Raises:
        ValueError: If response doesn't match expected schema.
    """
    cfg = cfg or SymbolicConfig()
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

    message = client.messages.create(
        model=cfg.claude_model,
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": scene_text}],
    )

    raw = message.content[0].text.strip()

    # Parse JSON — strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    result = json.loads(raw)

    # Validate schema
    if result.get("action") not in ("STOP", "SLOW", "CONTINUE"):
        raise ValueError(f"Invalid action: {result.get('action')}")
    if not isinstance(result.get("reasoning_edges"), list):
        raise ValueError("reasoning_edges must be a list")

    return result
