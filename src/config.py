"""Global configuration: macro-classes, category mappings, paths, and thresholds."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
LABEL_DIR = DATA_DIR / "labels"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

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
    # ── HUMAN (0) ──────────────────────────────────────────────
    "person":           "HUMAN",
    "head":             "HUMAN",
    "hat":              "HUMAN",
    "helmet":           "HUMAN",
    # ── VEHICLE (1) ───────────────────────────────────────────
    "car":              "VEHICLE",
    "truck":            "VEHICLE",
    "bus":              "VEHICLE",
    "train":            "VEHICLE",
    "motorcycle":       "VEHICLE",
    "bicycle":          "VEHICLE",
    "forklift":         "VEHICLE",
    "handcart":         "VEHICLE",
    # ── OBSTACLE (2) ──────────────────────────────────────────
    "box":              "OBSTACLE",
    "crate":            "OBSTACLE",
    "barrel":           "OBSTACLE",
    "cone":             "CONTEXT",
    "suitcase":         "OBSTACLE",
    "chair":            "OBSTACLE",
    "ladder":           "OBSTACLE",
    "container":        "OBSTACLE",
    # ── CONTEXT (3) ───────────────────────────────────────────
    "stop sign":        "CONTEXT",
    "traffic sign":     "CONTEXT",
    "traffic light":    "CONTEXT",
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
# YOLO Training Defaults (VRAM-safe for 4 GB)
# ---------------------------------------------------------------------------
@dataclass
class YOLOConfig:
    """YOLO training hyperparameters tuned for 4 GB VRAM."""

    model_weights: str = "yolo11n.pt"
    imgsz: int = 640
    batch: int = 4
    epochs: int = 50
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Depth Estimation Defaults
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

    center_crop_fraction: float = 0.2  # central 20% of bbox for depth sampling
    min_bbox_area_frac: float = 0.01   # drop boxes < 1% of image area


# ---------------------------------------------------------------------------
# Symbolic / Graph Thresholds
# ---------------------------------------------------------------------------
@dataclass
class SymbolicConfig:
    """Phase 4 graph-building thresholds."""

    depth_close: float = 2.0   # meters
    depth_far: float = 7.0     # meters
    nearness_threshold: float = 1.5  # meter diff to count as "NEAR"
    horizontal_bins: tuple[str, ...] = ("LEFT", "CENTER", "RIGHT")
    claude_model: str = "claude-3-5-sonnet-20241022"


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------
DETECTION_CATEGORIES: list[dict[str, int | str]] = [
    {"id": 0, "name": "person-like"},
    {"id": 1, "name": "vehicle-like"},
    {"id": 2, "name": "obstacle-like"},
    {"id": 3, "name": "safety-marker-like"},
]
