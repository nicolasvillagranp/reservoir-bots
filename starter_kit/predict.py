"""THEKER Hackathon -- Prediction Template.

This script loads the dataset, runs a decision function on each image,
and writes a submission-ready JSON file.  A simple baseline is provided
so participants can submit immediately, then iterate.

Key design points:
    - Train/val have annotations; test has NONE (you must detect).
    - Labels are raw and heterogeneous (27 categories).  Preprocessing
      is part of the challenge.  The baseline groups them for you as a
      starting point.

Usage::

    # Run on val (has annotations -- for development)
    python predict.py --data-dir ../data --split val --output predictions.json

    # Run on test (no annotations -- for submission)
    python predict.py --data-dir ../data --split test --output submission.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils import (
    bbox_area,
    bbox_height_ratio,
    build_annotation_index,
    build_image_index,
    load_annotations,
)

# ---------------------------------------------------------------------------
# Label preprocessing -- GROUP THE 27 RAW LABELS
# ---------------------------------------------------------------------------
# The dataset has 27 messy labels from multiple sources. This baseline
# groups them into 4 decision-relevant categories. You should refine this
# mapping or build your own.

PERSON_LABELS = {"person", "hat", "helmet", "head"}
VEHICLE_LABELS = {"bicycle", "car", "motorcycle", "bus", "train", "truck", "forklift"}
OBSTACLE_LABELS = {
    "suitcase", "chair", "barrel", "crate", "box", "handcart", "ladder",
    "Box", "Barrel", "Container", "Ladder", "Suitcase",
}
SAFETY_LABELS = {"cone", "Traffic sign", "Stop sign", "Traffic light"}


def classify_annotation(ann: dict, categories: dict[int, str]) -> str | None:
    """Map a raw annotation to a decision group.

    Args:
        ann: Annotation dict with ``category_id``.
        categories: Mapping of category_id -> category name.

    Returns:
        One of ``"person"``, ``"vehicle"``, ``"obstacle"``,
        ``"safety_marker"``, or ``None`` if unrecognized.
    """
    name = categories.get(ann["category_id"], "")
    if name in PERSON_LABELS:
        return "person"
    if name in VEHICLE_LABELS:
        return "vehicle"
    if name in OBSTACLE_LABELS:
        return "obstacle"
    if name in SAFETY_LABELS:
        return "safety_marker"
    return None


# ---------------------------------------------------------------------------
# Decision function -- REPLACE THIS WITH YOUR OWN LOGIC
# ---------------------------------------------------------------------------

def make_decision(
    image_record: Dict[str, Any],
    annotations: List[Dict[str, Any]],
    categories: Dict[int, str],
) -> Tuple[str, float, str]:
    """Decide whether the vehicle should STOP, SLOW, or CONTINUE.

    This baseline preprocesses the 27 raw labels into 4 groups, then
    applies simple spatial rules.  Participants should replace or extend
    this with learned models, VLMs, and richer reasoning.

    Args:
        image_record: COCO image dict (``id``, ``width``, ``height``, ``file_name``).
        annotations: Annotation dicts for this image (may be empty for test).
        categories: Mapping of category_id -> category name from the annotations file.

    Returns:
        Tuple of (action, confidence, reasoning).
    """
    img_h = image_record["height"]
    img_w = image_record["width"]
    img_area = img_h * img_w

    # Group annotations
    persons, vehicles, obstacles, markers = [], [], [], []
    for ann in annotations:
        group = classify_annotation(ann, categories)
        if group == "person":
            persons.append(ann)
        elif group == "vehicle":
            vehicles.append(ann)
        elif group == "obstacle":
            obstacles.append(ann)
        elif group == "safety_marker":
            markers.append(ann)

    # -----------------------------------------------------------------------
    # --- YOUR LOGIC HERE ---
    # The rules below are a minimal baseline.  Replace or extend them.
    # -----------------------------------------------------------------------

    # Rule 1: Large person (close range) -> STOP
    for ann in persons:
        if bbox_height_ratio(ann["bbox"], img_h) > 0.25:
            return (
                "STOP", 0.90,
                "Person detected at close range (height ratio > 0.25).",
            )

    # Rule 2: Large vehicle -> STOP
    for ann in vehicles:
        if bbox_area(ann["bbox"]) > 0.15 * img_area:
            return (
                "STOP", 0.85,
                "Vehicle occupying >15% of image area.",
            )

    # Rule 3: Any person or vehicle present -> SLOW
    if persons or vehicles:
        return (
            "SLOW", 0.70,
            f"Detected {len(persons)} person(s) and {len(vehicles)} vehicle(s).",
        )

    # Rule 4: Safety marker present -> SLOW
    if markers:
        return (
            "SLOW", 0.60,
            f"Safety marker(s) detected ({len(markers)}).",
        )

    # Rule 5: Obstacles covering >40% of image width -> SLOW
    if obstacles:
        x_ranges = sorted((ann["bbox"][0], ann["bbox"][0] + ann["bbox"][2]) for ann in obstacles)
        merged = [x_ranges[0]]
        for start, end in x_ranges[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        total_width = sum(e - s for s, e in merged)
        if total_width > 0.40 * img_w:
            return (
                "SLOW", 0.65,
                f"Obstacles cover {total_width / img_w:.0%} of image width.",
            )

    # Rule 6: Nothing significant -> CONTINUE
    return ("CONTINUE", 0.80, "No significant hazards detected.")

    # -----------------------------------------------------------------------
    # --- END OF YOUR LOGIC ---
    # -----------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_predictions(
    data_dir: Path,
    output_path: Path,
    team_name: str,
    split: str = "test",
) -> None:
    """Load the dataset, predict on every image, and write submission JSON.

    For train/val splits, annotations are available and used by the baseline.
    For the test split, annotations are empty -- the baseline will predict
    CONTINUE for every image (since it has no detections).  Participants
    should add their own detection pipeline for the test split.

    Args:
        data_dir: Root data directory.
        output_path: Where to write the submission JSON.
        team_name: Team name for the submission.
        split: Dataset split to run on.
    """
    ann_path = data_dir / "annotations" / f"{split}.json"
    if not ann_path.exists():
        print(f"ERROR: Annotation file not found: {ann_path}")
        sys.exit(1)

    print(f"Loading annotations from {ann_path} ...")
    coco_data = load_annotations(ann_path)

    # Build category lookup from the file itself
    categories = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
    print(f"Categories: {len(categories)} raw labels")

    image_index = build_image_index(coco_data)
    annotation_index = build_annotation_index(coco_data)

    has_annotations = len(coco_data.get("annotations", [])) > 0
    if not has_annotations:
        print(f"WARNING: {split} split has no annotations. "
              "The baseline will predict CONTINUE for all images. "
              "Add your own detection pipeline for meaningful predictions.")

    print(f"Found {len(image_index)} images in the {split} split.")

    predictions: List[Dict[str, Any]] = []
    all_detections: List[Dict[str, Any]] = []

    for image_id, image_record in sorted(image_index.items()):
        anns = annotation_index.get(image_id, [])

        # --- If you have your own detector, run it here for test images ---
        # if not has_annotations:
        #     anns = my_detector.detect(image_path)
        # -----------------------------------------------------------------

        action, confidence, reasoning = make_decision(
            image_record, anns, categories,
        )

        predictions.append({
            "image_id": image_id,
            "action": action,
            "confidence": round(confidence, 4),
            "reasoning": reasoning,
        })

        # Collect detections for scoring (re-emit the annotations we used)
        for ann in anns:
            all_detections.append({
                "image_id": image_id,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "score": 1.0,  # ground-truth annotations have implicit score 1.0
            })

    submission: Dict[str, Any] = {
        "team_name": team_name,
        "predictions": predictions,
    }

    # Include detections (for val, these are just the given annotations;
    # for test, you should populate this with your detector's output)
    if all_detections:
        submission["detections"] = all_detections
        submission["detection_categories"] = [
            {"id": cid, "name": cname}
            for cid, cname in sorted(categories.items())
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(submission, fh, indent=2)

    print(f"Wrote {len(predictions)} predictions to {output_path}")
    if all_detections:
        print(f"Included {len(all_detections)} detections for scoring.")
    else:
        print("No detections included (test set has no annotations).")

    # Summary
    counts: Dict[str, int] = {}
    for p in predictions:
        counts[p["action"]] = counts.get(p["action"], 0) + 1
    print("Action distribution:")
    for act in ("STOP", "SLOW", "CONTINUE"):
        c = counts.get(act, 0)
        print(f"  {act:10s}: {c:5d}  ({c / len(predictions):.1%})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="THEKER Hackathon -- Generate predictions for submission.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("../data"))
    parser.add_argument("--output", type=Path, default=Path("predictions.json"))
    parser.add_argument("--team-name", type=str, default="your_team")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    args = parser.parse_args()
    run_predictions(args.data_dir, args.output, args.team_name, args.split)
