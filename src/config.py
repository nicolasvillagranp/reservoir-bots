"""Global configuration: macro-classes, category mappings, paths, and thresholds.

Set PIPELINE_MODE=production for full DGX training.
Default is 'test' (small subset, few epochs).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Mode: "test" (laptop, small subset) or "production" (DGX, full dataset)
# ---------------------------------------------------------------------------
MODE = os.environ.get("PIPELINE_MODE", "test")

print(f"Running in {MODE} mode. Set PIPELINE_MODE=production for full training.")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
LABEL_DIR = DATA_DIR / "labels"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
FINETUNED_DIR = MODELS_DIR / "finetuned"
GNN_DIR = MODELS_DIR / "gnn"

# ---------------------------------------------------------------------------
# Macro-Class Ontology (4 classes)
# ---------------------------------------------------------------------------
MACRO_CLASSES: dict[int, str] = {
    0: "HUMAN",
    1: "VEHICLE",
    2: "OBSTACLE",
    3: "CONTEXT",
}

MACRO_CLASS_TO_ID: dict[str, int] = {v: k for k, v in MACRO_CLASSES.items()}

# ---------------------------------------------------------------------------
# Raw COCO Category → Macro-Class mapping
#
# 27 raw categories from the hackathon COCO annotations, mapped into 4 bins.
# Keys are the *category names* (lowercased, stripped) found in train.json /
# val.json.  We map by name rather than raw ID so the mapping survives
# re-numbering across annotation files.
# ---------------------------------------------------------------------------
RAW_TO_MACRO: dict[str, str] = {
    # ── Macro-class self-mapping (fine-tuned model outputs these) ──
    "human": "HUMAN",
    "vehicle": "VEHICLE",
    "obstacle": "OBSTACLE",
    "context": "CONTEXT",
    # ── HUMAN (0) ──────────────────────────────────────────────
    "person": "HUMAN",
    "head": "HUMAN",
    "hat": "HUMAN",
    "helmet": "HUMAN",
    # ── VEHICLE (1) ───────────────────────────────────────────
    "car": "VEHICLE",
    "truck": "VEHICLE",
    "bus": "VEHICLE",
    "train": "VEHICLE",
    "motorcycle": "VEHICLE",
    "bicycle": "VEHICLE",
    "forklift": "VEHICLE",
    "handcart": "VEHICLE",
    # ── OBSTACLE (2) ──────────────────────────────────────────
    "box": "OBSTACLE",
    "crate": "OBSTACLE",
    "barrel": "OBSTACLE",
    "suitcase": "OBSTACLE",
    "chair": "OBSTACLE",
    "ladder": "OBSTACLE",
    "container": "OBSTACLE",
    # ── CONTEXT (3) ───────────────────────────────────────────
    "cone": "CONTEXT",
    "stop sign": "CONTEXT",
    "traffic sign": "CONTEXT",
    "traffic light": "CONTEXT",
}


def raw_name_to_macro_id(raw_name: str) -> int | None:
    """Map a raw COCO category name to a macro-class ID (0-3).

    Returns None if the name is not in the mapping (unknown category).
    """
    macro = RAW_TO_MACRO.get(raw_name.strip().lower())
    if macro is None:
        return None
    return MACRO_CLASS_TO_ID[macro]


# ---------------------------------------------------------------------------
# YOLO Training
# ---------------------------------------------------------------------------
@dataclass
class YOLOConfig:
    """YOLO training hyperparameters. Adapts to MODE automatically."""

    model_weights: str = str(PRETRAINED_DIR / "yolo11n.pt")
    finetuned_weights: str = str(FINETUNED_DIR / "yolo/weights/best.pt")
    imgsz: int = 640
    batch: int = 128
    epochs: int = 2 if MODE == "test" else 50
    device: str = "cuda"
    subset_size: int | None = 50 if MODE == "test" else None


# ---------------------------------------------------------------------------
# Depth Estimation
# ---------------------------------------------------------------------------
@dataclass
class DepthConfig:
    """ZoeDepth configuration."""

    model_name: str = "Intel/zoedepth-nyu-kitti"
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Fusion Thresholds
# ---------------------------------------------------------------------------
@dataclass
class FusionConfig:
    """Phase 3 fusion parameters."""

    center_crop_fraction: float = 0.2
    min_bbox_area_frac: float = 0.01


# ---------------------------------------------------------------------------
# Symbolic / Graph Thresholds
# ---------------------------------------------------------------------------
@dataclass
class SymbolicConfig:
    """Phase 4 graph-building thresholds."""

    depth_close: float = 2.0
    depth_far: float = 7.0
    nearness_threshold: float = 1.5
    horizontal_bins: tuple[str, ...] = ("LEFT", "CENTER", "RIGHT")
    claude_model: str = "claude-sonnet-4-6"
    dataset_fraction: float = 0.05 if MODE == "production" else 1.0
    depth_strategy: str = "bbox_approx" if MODE == "production" else "model"
    enable_stage_checkpoints: bool = True


# ---------------------------------------------------------------------------
# GNN Training
# ---------------------------------------------------------------------------
@dataclass
class GNNConfig:
    """Phase 5 GNN training hyperparameters."""

    hidden: int = 64
    lr: float = 0.01
    batch_size: int = 32
    epochs: int = 100 if MODE == "test" else 500
    save_path: str = str(GNN_DIR / "navigation_gnn.pt")


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------
DETECTION_CATEGORIES: list[dict[str, int | str]] = [
    {"id": 0, "name": "person-like"},
    {"id": 1, "name": "vehicle-like"},
    {"id": 2, "name": "obstacle-like"},
    {"id": 3, "name": "safety-marker-like"},
]
