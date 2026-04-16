"""Phase 6 Milestone: verify end-to-end pipeline generates valid submission.

1. Run pipeline on 3 test images.
2. Assert predictions.json has exactly 3 entries in predictions.
3. Assert detection_categories array at root.
4. Assert every reasoning string is >= 10 chars.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from config import DETECTION_CATEGORIES, OUTPUT_DIR, PRETRAINED_DIR
from run_pipeline import run_pipeline

PHASE6_OUT = OUTPUT_DIR / "phase6"
PHASE6_OUT.mkdir(parents=True, exist_ok=True)


def test_pipeline_3_images() -> dict:
    """Run pipeline on 3 test images and validate output."""
    print("\n--- test_pipeline_3_images ---")

    model_path = PRETRAINED_DIR / "yolo11n.pt"
    output_path = PHASE6_OUT / "test_submission.json"

    submission = run_pipeline(
        yolo_model_path=str(model_path),
        gnn_model_path=None,  # rule-based fallback
        limit=3,
        output_path=str(output_path),
    )

    assert output_path.exists(), f"Submission file not created at {output_path}"
    print(f"PASS: Submission file created ({output_path.stat().st_size} bytes)")
    return submission


def test_predictions_count(submission: dict) -> None:
    """Assert exactly 3 predictions."""
    print("\n--- test_predictions_count ---")
    preds = submission["predictions"]
    assert len(preds) == 3, f"Expected 3 predictions, got {len(preds)}"
    print(f"PASS: {len(preds)} predictions")


def test_detection_categories(submission: dict) -> None:
    """Assert detection_categories array at root matches spec."""
    print("\n--- test_detection_categories ---")
    cats = submission.get("detection_categories")
    assert cats is not None, "Missing detection_categories"
    assert isinstance(cats, list), f"Expected list, got {type(cats)}"
    assert len(cats) == 4, f"Expected 4 categories, got {len(cats)}"

    expected_ids = {0, 1, 2, 3}
    actual_ids = {c["id"] for c in cats}
    assert actual_ids == expected_ids, f"Category IDs mismatch: {actual_ids}"
    print(f"PASS: detection_categories valid — {[c['name'] for c in cats]}")


def test_reasoning_strings(submission: dict) -> None:
    """Assert every reasoning string is a str with >= 10 chars."""
    print("\n--- test_reasoning_strings ---")
    for pred in submission["predictions"]:
        r = pred["reasoning"]
        assert isinstance(r, str), f"reasoning not str: {type(r)}"
        assert len(r) >= 10, f"reasoning too short ({len(r)} chars): '{r}'"
        print(f"  [{pred['image_id']}] action={pred['action']} "
              f"conf={pred['confidence']:.2f} reasoning='{r[:60]}...'")
    print("PASS: All reasoning strings valid")


def test_prediction_schema(submission: dict) -> None:
    """Assert each prediction has required keys and valid action."""
    print("\n--- test_prediction_schema ---")
    valid_actions = {"STOP", "SLOW", "CONTINUE"}
    for pred in submission["predictions"]:
        assert "image_id" in pred, "Missing image_id"
        assert "action" in pred, "Missing action"
        assert "confidence" in pred, "Missing confidence"
        assert "reasoning" in pred, "Missing reasoning"
        assert pred["action"] in valid_actions, f"Invalid action: {pred['action']}"
        assert 0.0 <= pred["confidence"] <= 1.0, f"Confidence out of range: {pred['confidence']}"
    print("PASS: All prediction schemas valid")


def test_detection_schema(submission: dict) -> None:
    """Assert detections have valid structure."""
    print("\n--- test_detection_schema ---")
    for det in submission.get("detections", []):
        assert "id" in det, "Missing id"
        assert "image_id" in det, "Missing image_id"
        assert "category_id" in det, "Missing category_id"
        assert "bbox" in det, "Missing bbox"
        assert "score" in det, "Missing score"
        assert len(det["bbox"]) == 4, f"bbox should have 4 elements: {det['bbox']}"
        assert det["category_id"] in (0, 1, 2, 3), f"Invalid category_id: {det['category_id']}"
    print(f"PASS: {len(submission.get('detections', []))} detections, all valid")


if __name__ == "__main__":
    submission = test_pipeline_3_images()
    test_predictions_count(submission)
    test_detection_categories(submission)
    test_reasoning_strings(submission)
    test_prediction_schema(submission)
    test_detection_schema(submission)
    print("\n=== All Phase 6 tests passed ===")
