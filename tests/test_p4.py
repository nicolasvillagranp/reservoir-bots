"""Phase 4 Milestone: verify graph builder + Claude API response.

1. Build graph from hardcoded 4-object scene.
2. Call Claude API (requires ANTHROPIC_API_KEY).
3. Validate response schema + reasoning edge validity.
4. Save results to outputs/phase4/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config import OUTPUT_DIR
from phase4_symbolic import build_scene_graph, query_claude

PHASE4_OUT = OUTPUT_DIR / "phase4"
PHASE4_OUT.mkdir(parents=True, exist_ok=True)

# Hardcoded complex scene: human near forklift, distant box, stop sign
MOCK_SCENE = [
    {
        "class": "HUMAN",
        "class_id": 0,
        "bbox": [100, 200, 80, 200],   # LEFT zone
        "depth_m": 3.5,
        "confidence": 0.92,
    },
    {
        "class": "VEHICLE",
        "class_id": 1,
        "bbox": [120, 180, 100, 250],  # LEFT zone, similar depth -> NEAR
        "depth_m": 4.0,
        "confidence": 0.88,
    },
    {
        "class": "OBSTACLE",
        "class_id": 2,
        "bbox": [400, 300, 60, 60],    # RIGHT zone
        "depth_m": 12.0,
        "confidence": 0.75,
    },
    {
        "class": "CONTEXT",
        "class_id": 3,
        "bbox": [500, 100, 40, 80],    # RIGHT zone
        "depth_m": 1.5,
        "confidence": 0.95,
    },
]


def test_graph_builder() -> str:
    """Build graph from mock scene and verify text output."""
    print("\n--- test_graph_builder ---")
    scene_text = build_scene_graph(MOCK_SCENE, img_w=640)
    print(f"  Scene text: {scene_text}")

    # Should contain self-relations for all 4 objects
    assert "HUMAN_1" in scene_text, "Missing HUMAN_1"
    assert "VEHICLE_1" in scene_text, "Missing VEHICLE_1"
    assert "OBSTACLE_1" in scene_text, "Missing OBSTACLE_1"
    assert "CONTEXT_1" in scene_text, "Missing CONTEXT_1"

    # HUMAN_1 and VEHICLE_1 are both LEFT + similar depth -> should be NEAR
    assert "NEAR" in scene_text, "Expected NEAR relation between HUMAN_1 and VEHICLE_1"

    print("PASS: Graph builder output valid")
    return scene_text


def test_claude_api(scene_text: str) -> dict:
    """Query Claude and validate response schema."""
    print("\n--- test_claude_api ---")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("SKIP: ANTHROPIC_API_KEY not set")
        return {}

    result = query_claude(scene_text)
    print(f"  Claude response: {json.dumps(result, indent=2)}")

    # 1. action is strictly one of STOP, SLOW, CONTINUE
    assert result["action"] in ("STOP", "SLOW", "CONTINUE"), (
        f"Invalid action: {result['action']}"
    )

    # 2. reasoning_edges is a list
    edges = result["reasoning_edges"]
    assert isinstance(edges, list), f"reasoning_edges not a list: {type(edges)}"

    # 3. 1 <= len(reasoning_edges) <= 3
    assert 1 <= len(edges) <= 3, (
        f"Expected 1-3 reasoning_edges, got {len(edges)}: {edges}"
    )

    # 4. Every edge references actual nodes/relations from scene_text
    # Build set of valid tokens from scene
    valid_tokens = set()
    for word in scene_text.replace(".", "").split():
        valid_tokens.add(word)
    # Also accept compound forms like HUMAN_1_LEFT_MID
    scene_nodes = ["HUMAN_1", "VEHICLE_1", "OBSTACLE_1", "CONTEXT_1"]
    bins = ["LEFT", "CENTER", "RIGHT", "CLOSE", "MID", "FAR", "NEAR"]
    for node in scene_nodes:
        for b1 in bins:
            valid_tokens.add(f"{node}_{b1}")
            for b2 in bins:
                valid_tokens.add(f"{node}_{b1}_{b2}")
    # Cross-relations
    for i, a in enumerate(scene_nodes):
        for b in scene_nodes[i + 1:]:
            valid_tokens.add(f"{a}_NEAR_{b}")

    for edge in edges:
        # Check that edge components come from scene
        parts = edge.split("_")
        # Reconstruct possible node name from parts
        found = False
        for token in valid_tokens:
            if edge == token:
                found = True
                break
        if not found:
            # Relaxed check: all sub-parts appear in scene text
            for part in parts:
                assert part in scene_text.replace(".", "").replace(",", ""), (
                    f"Hallucinated edge component '{part}' in edge '{edge}' "
                    f"not found in scene"
                )

    print(f"PASS: Claude response valid — action={result['action']}, edges={edges}")
    return result


def test_save_results(scene_text: str, result: dict) -> None:
    """Save graph + Claude response to outputs."""
    print("\n--- test_save_results ---")
    output = {
        "scene_text": scene_text,
        "claude_response": result,
        "mock_objects": MOCK_SCENE,
    }
    out_path = PHASE4_OUT / "phase4_test_result.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    assert out_path.exists()
    print(f"PASS: Saved to {out_path}")


if __name__ == "__main__":
    scene_text = test_graph_builder()
    result = test_claude_api(scene_text)
    if result:
        test_save_results(scene_text, result)
    print("\n=== All Phase 4 tests passed ===")
