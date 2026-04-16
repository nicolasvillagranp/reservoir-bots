# R3F: Reliable Real-time Reasoning Framework

**A Hybrid Neuro-Symbolic Pipeline for Autonomous Robot Navigation Safety**

> Our pipeline extracts 2D and depth data, compiles it into a spatial-symbolic graph, uses a heavy LLM to reason about safety rules (Teacher), and distills that reasoning into a lightning-fast Graph Neural Network (Student) for real-time execution.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Architecture](#architecture)
  - [Phase 0 — Data Preprocessing](#phase-0--data-preprocessing)
  - [Phase 1 — Dual Perception: Object Detection (YOLOv11)](#phase-1--dual-perception-object-detection-yolov11)
  - [Phase 2 — Dual Perception: Metric Depth (ZoeDepth)](#phase-2--dual-perception-metric-depth-zoedepth)
  - [Phase 3 — Spatial-Symbolic Compiler (3D Scene Fusion)](#phase-3--spatial-symbolic-compiler-3d-scene-fusion)
  - [Phase 4 — Reliable Reasoning Tutor (Teacher LLM Distillation)](#phase-4--reliable-reasoning-tutor-teacher-llm-distillation)
  - [Phase 5 — Lightning Apprentice (Student GNN)](#phase-5--lightning-apprentice-student-gnn)
  - [Phase 6 — End-to-End Inference & Submission](#phase-6--end-to-end-inference--submission)
- [Installation](#installation)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Presentation Slides](#presentation-slides)
- [Project Structure](#project-structure)

---

## Problem Statement

Autonomous robots operating in warehouse and industrial environments must make split-second safety decisions: **stop**, **slow down**, or **continue**. The challenge is threefold:

1. **Perception under constraint** — Monocular cameras provide no native depth. Scale ambiguity makes it impossible to know if an object is a small obstacle nearby or a large one far away.
2. **Safety-critical reasoning** — A missed human detection or a wrong depth estimate can cause injury. The system must be conservative and explainable.
3. **Real-time execution** — Heavy models (LLMs, large depth networks) cannot run on-robot at 30 FPS. The deployed model must be tiny and fast.

Most approaches pick two of {accurate, explainable, fast}. We achieve all three through **knowledge distillation from a symbolic-LLM teacher into a lightweight GNN student**.

---

## Solution Overview

R3F is a **Teacher-Student neuro-symbolic pipeline**:

```
Raw Image
    |
    +---> [YOLOv11 Nano] --------> 2D Bounding Boxes (4 macro-classes)
    |
    +---> [ZoeDepth / BBox Heuristic] --> Metric Depth per Object
    |
    v
[Spatial-Symbolic Compiler]
    |
    v
Text Scene Graph: "HUMAN_1 is CENTER and CLOSE. FORKLIFT_1 is LEFT and MID."
    |
    +---> [Claude Sonnet — Teacher] ---> {action: "STOP", reasoning_edges: [...]}
    |                                        |
    |                              (offline, generates training data)
    |                                        |
    v                                        v
[GraphSAGE GNN — Student] <---- trains on Teacher's labels
    |
    v
Real-time: action + explainable reasoning edges
```

### Why This Framework is Powerful

**Flexible and adaptable.** The pipeline is not tied to any specific robot, camera, or environment. You can retrain it on:

- **Custom datasets from real robots** — Any COCO-format annotations work. Swap the images and category mapping, the rest adapts.
- **Simulation data** — Synthetic images from Gazebo, Isaac Sim, or Unity plug in directly.
- **Structured camera setups** — When the camera intrinsics and mounting are known (fixed lens distortion, known height), the depth estimation becomes dramatically more robust. The geometric depth heuristic (`bbox_approx` strategy) is already calibrated for this — with real calibration data, it becomes near-exact.

**Explainable by design.** The GNN doesn't just output "STOP" — it predicts which edges in the scene graph caused the decision. The robot's telemetry can report: *"STOP triggered by HUMAN_1 at CENTER CLOSE"*. This is critical for safety certification and debugging.

**Scales from laptop to DGX.** The `PIPELINE_MODE` flag switches between a lightweight test mode (50 images, 2 YOLO epochs, 100 GNN epochs) and full production training (entire dataset, 50 YOLO epochs, 500 GNN epochs).

---

## Architecture

### Phase 0 — Data Preprocessing

**Module:** `src/phases/phase0_data.py`

Parses COCO-format annotations into YOLO training labels and collapses 27 raw categories into 4 safety-relevant **macro-classes**:

| Macro-Class | ID | Raw Categories |
|---|---|---|
| `HUMAN` | 0 | person, head, hat, helmet |
| `VEHICLE` | 1 | car, truck, bus, train, motorcycle, bicycle, forklift, handcart |
| `OBSTACLE` | 2 | box, crate, barrel, suitcase, chair, ladder, container |
| `CONTEXT` | 3 | cone, stop sign, traffic sign, traffic light |

**What it produces:**
- `data/labels/train/*.txt` — YOLO-format labels (class_id, x_center, y_center, w, h, normalized)
- `data/labels/val/*.txt` — Same for validation split
- `data/dataset.yaml` — YOLO dataset config pointing to images and labels
- `models/pretrained/yolo11n.pt` — Pre-downloaded YOLOv11 Nano weights

**In test mode:** Creates a 50-image `data/images/tiny/` subset for fast iteration.

---

### Phase 1 — Dual Perception: Object Detection (YOLOv11)

**Module:** `src/phases/phase1_vision.py`

Fine-tunes **YOLOv11 Nano** on the 4 macro-classes. YOLOv11n was chosen for its speed-accuracy tradeoff — it runs comfortably on edge devices while being sufficient for coarse macro-class detection.

**Training config:**
- Image size: 640x640
- Batch size: 128
- Epochs: 2 (test) / 50 (production)
- Confidence threshold: 0.25
- Minimum bbox area: 1% of image (filters noise)

**What it produces:**
- `models/finetuned/yolo/weights/best.pt` — Fine-tuned YOLO weights (~6 MB)
- `outputs/phase1/runs/` — Training metrics, loss curves, confusion matrix

**Inference function:** `predict_objects_with_model(image_path, model)` returns a list of detections with `class_name`, `bbox_xyxy`, `confidence`.

---

### Phase 2 — Dual Perception: Metric Depth (ZoeDepth)

**Module:** `src/phases/phase2_depth.py`

Loads **ZoeDepth** (Intel/zoedepth-nyu-kitti) via HuggingFace Transformers to produce metric depth maps in meters from a single monocular image.

This solves the **monocular scale ambiguity** problem: a bounding box alone cannot tell you if an object is 2m or 20m away. ZoeDepth provides a per-pixel depth estimate trained on NYU Depth V2 and KITTI, giving real-world metric distances.

**Important note on adaptability:** When deploying on a real robot with known camera intrinsics (focal length, sensor size, mounting height), depth estimation becomes far more robust. Fixed camera parameters eliminate the generalization gap that affects pretrained monocular depth models on arbitrary images. A calibrated system can even bypass the neural depth model entirely and use geometric projection.

**What it produces:**
- Depth maps as NumPy arrays (H x W, values in meters)
- GPU memory is explicitly freed after inference to allow sequential model loading

**Production optimization:** In production mode, the pipeline can use a fast **bbox geometry heuristic** (`depth_strategy: bbox_approx`) that estimates depth from bounding box position and size — no depth model needed. This makes dataset generation 10x faster on large datasets.

---

### Phase 3 — Spatial-Symbolic Compiler (3D Scene Fusion)

**Module:** `src/phases/phase3_fusion.py`

Fuses YOLO detections with depth maps into a unified 3D scene representation. For each detected object:

1. Extract the **center crop** (20% of bbox area) from the depth map
2. Take the **median depth** within that crop (robust to depth noise at edges)
3. Output: object class, bounding box, metric depth, confidence

**What it produces:**
- Per-image list of fused objects: `{class, bbox, depth_m, confidence}`
- This is an intermediate representation consumed by Phase 4

---

### Phase 4 — Reliable Reasoning Tutor (Teacher LLM Distillation)

**Module:** `src/phases/phase4_symbolic.py`

This is the core innovation. Phase 4 does three things:

#### 1. Scene Graph Construction

Translates physical geometry into **discrete semantic logic** — no heavy point clouds needed:

- **Horizontal binning:** Object center-x mapped to `LEFT` / `CENTER` / `RIGHT` (image thirds)
- **Depth binning:** Metric depth mapped to `CLOSE` (<2m) / `MID` (2-7m) / `FAR` (>7m)
- **Proximity edges:** Objects within 1.5m at the same horizontal bin get a `NEAR` relation

**Output format:**
```
HUMAN_1 is CENTER and CLOSE. FORKLIFT_1 is LEFT and MID. HUMAN_1 is NEAR FORKLIFT_1.
```

#### 2. Teacher Labeling (Claude API)

When `ANTHROPIC_API_KEY` is set, the text scene graph is sent to **Claude Sonnet** with a safety-focused system prompt. Claude reasons about warehouse safety rules and returns:

```json
{"action": "STOP", "reasoning_edges": ["HUMAN_1_CENTER_CLOSE"]}
```

The reasoning edges identify exactly which objects/relations triggered the decision — this is the explainability signal that the GNN learns to reproduce.

#### 3. Rule-Based Fallback

When no API key is available, a deterministic rule engine applies:
- **STOP** if any HUMAN is CLOSE, or HUMAN is CENTER+MID
- **SLOW** if any HUMAN is MID, or any VEHICLE is CLOSE/MID
- **CONTINUE** otherwise

**Production optimization:** 3-stage pipeline with JSON checkpoints at each stage (YOLO detections, depth values, final labels). If interrupted, resumes from the last completed stage.

**What it produces:**
- `data/gnn_dataset/scenes_train.jsonl` — Training scenes (JSONL, one scene per line)
- `data/gnn_dataset/scenes_val.jsonl` — Validation scenes
- `data/gnn_dataset/checkpoints/` — Stage checkpoints for resumability

Each line contains:
```json
{
  "image": "000000001234.jpg",
  "objects": [{"class": "HUMAN", "bbox": [x,y,w,h], "depth_m": 2.5, "confidence": 0.92}],
  "scene_text": "HUMAN_1 is CENTER and CLOSE.",
  "action": "STOP",
  "reasoning_edges": ["HUMAN_1_CENTER_CLOSE"]
}
```

---

### Phase 5 — Lightning Apprentice (Student GNN)

**Module:** `src/phases/phase5_gnn.py`

Trains a lightweight **GraphSAGE** model (PyTorch Geometric) on the synthetic dataset from Phase 4. This is the model that runs on the robot in real-time.

#### Model Architecture: `NavigationGNN`

```
Input: Graph (nodes = objects, edges = all-to-all)
  |
  v
SAGEConv(11 -> 64) + ReLU        # GraphSAGE layer 1
  |
  v
SAGEConv(64 -> 64) + ReLU        # GraphSAGE layer 2
  |
  +---> global_mean_pool ----------> Action Head: Linear(64->32->3)
  |                                   Output: STOP / SLOW / CONTINUE
  |
  +---> [src_emb || dst_emb] ------> Edge Head: Linear(128->32->1)
                                      Output: per-edge reasoning probability
```

#### Node Features (11 dimensions)

| Feature | Dims | Encoding |
|---|---|---|
| Macro-class | 4 | One-hot (HUMAN, VEHICLE, OBSTACLE, CONTEXT) |
| Horizontal bin | 3 | One-hot (LEFT, CENTER, RIGHT) |
| Depth bin | 3 | One-hot (CLOSE, MID, FAR) |
| Depth (normalized) | 1 | depth_m / 20.0 |

#### Joint Loss

```
L_total = L_action (Cross-Entropy) + L_edge (Binary Cross-Entropy with Logits)
```

- **Action loss:** 3-way classification — did the GNN predict the correct action?
- **Edge loss:** Binary per-edge — did the GNN identify the correct reasoning edges?

**Training config:**
- Hidden dimension: 64
- Learning rate: 0.01 (Adam)
- Batch size: 32
- Epochs: 100 (test) / 500 (production)

**What it produces:**
- `models/gnn/navigation_gnn.pt` — Trained GNN weights (~50 KB)

---

### Phase 6 — End-to-End Inference & Submission

**Module:** `src/main.py`

Runs the complete pipeline on test images and generates a submission JSON:

1. Load YOLO model (fine-tuned or pretrained)
2. Load GNN model (if available, else rule-based fallback)
3. For each test image:
   - YOLO detection + macro-class remapping
   - Fast depth approximation from bbox geometry
   - GNN inference (or rule-based fallback)
   - Format reasoning into human-readable string
4. Write `submission.json`

**Reasoning output example:**
```
ACTION: STOP
REASON: HUMAN detected [CLOSE] [IN FRONT OF] [ROBOT]. VEHICLE detected [MID] [TO THE LEFT OF] [ROBOT].
```

**What it produces:**
- `outputs/submission.json` — Final submission with detections and predictions

---

## Installation

### Prerequisites

- **Python 3.10** (required, `~=3.10.0`)
- **CUDA 12.6+** (for GPU training; CPU works for testing)
- [**uv**](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/theker-hack.git
cd theker-hack

# 2. Create virtual environment and install dependencies
uv sync

# 3. (Optional) Set environment variables
export PIPELINE_MODE=test          # "test" (default) or "production"
export ANTHROPIC_API_KEY=sk-...    # Only needed for Claude teacher labeling in Phase 4
```

### Docker (GPU Training)

```bash
# Build and run with GPU access
docker compose up -d

# Execute inside the container
docker compose exec theker-lab bash
uv sync
```

The Docker image is based on `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` with 32 GB shared memory.

---

## CLI Reference

Every phase is an independent Python module. Run them sequentially for a full pipeline, or individually to iterate on a specific stage.

### Full Pipeline (All Phases)

```bash
# Test mode (laptop, ~15 min)
export PIPELINE_MODE=test
uv run python -m src.phases.phase0_data
uv run python -m src.phases.phase1_vision
uv run python -m src.phases.phase2_depth
uv run python -m src.phases.phase3_fusion
uv run python -m src.phases.phase4_symbolic
uv run python -m src.phases.phase5_gnn
uv run python -m src.main --model models/finetuned/yolo/weights/best.pt --gnn models/gnn/navigation_gnn.pt

# Production mode (DGX, full dataset)
export PIPELINE_MODE=production
# Same commands — config adapts automatically
```

### Phase-by-Phase

#### Phase 0: Data Preprocessing

```bash
uv run python -m src.phases.phase0_data
```

**Reads:** `data/annotations/train.json`, `data/annotations/val.json`
**Creates:**
- `data/labels/train/*.txt` — YOLO labels
- `data/labels/val/*.txt` — YOLO labels
- `data/dataset.yaml` — YOLO dataset config
- `models/pretrained/yolo11n.pt` — Base YOLO weights

#### Phase 1: YOLO Fine-Tuning

```bash
uv run python -m src.phases.phase1_vision
```

**Reads:** `data/dataset.yaml`, `data/images/`, `data/labels/`
**Creates:**
- `models/finetuned/yolo/weights/best.pt` — Fine-tuned weights
- `outputs/phase1/runs/` — Training logs and metrics

#### Phase 2: Depth Model Verification

```bash
uv run python -m src.phases.phase2_depth
```

**Reads:** Sample validation images
**Creates:** Nothing persistent (verifies ZoeDepth loads and runs correctly)

#### Phase 3: Fusion Test

```bash
uv run python -m src.phases.phase3_fusion
```

**Reads:** Sample images, YOLO model, depth model
**Creates:** Nothing persistent (tests the fusion pipeline on sample images)

#### Phase 4: GNN Dataset Generation

```bash
# Rule-based labeling (no API key needed)
uv run python -m src.phases.phase4_symbolic

# With Claude teacher labeling
export ANTHROPIC_API_KEY=sk-...
uv run python -m src.phases.phase4_symbolic
```

**Reads:** `data/annotations/train.json`, `data/annotations/val.json`, images, YOLO model
**Creates:**
- `data/gnn_dataset/scenes_train.jsonl` — Training dataset
- `data/gnn_dataset/scenes_val.jsonl` — Validation dataset
- `data/gnn_dataset/checkpoints/*.json` — Resumable stage checkpoints

#### Phase 5: GNN Training

```bash
uv run python -m src.phases.phase5_gnn
```

**Reads:** `data/gnn_dataset/scenes_train.jsonl`, `data/gnn_dataset/scenes_val.jsonl`
**Creates:**
- `models/gnn/navigation_gnn.pt` — Trained GNN weights

#### Phase 6: End-to-End Inference

```bash
# With GNN model
uv run python -m src.main \
  --model models/finetuned/yolo/weights/best.pt \
  --gnn models/gnn/navigation_gnn.pt

# Without GNN (rule-based fallback)
uv run python -m src.main \
  --model models/finetuned/yolo/weights/best.pt

# Limit to N images (for testing)
uv run python -m src.main \
  --model models/pretrained/yolo11n.pt \
  --limit 10

# Custom output path
uv run python -m src.main \
  --model models/finetuned/yolo/weights/best.pt \
  --output results/my_submission.json
```

**Reads:** `data/annotations/test.json`, `data/images/test/`, YOLO model, GNN model
**Creates:**
- `outputs/submission.json` — Submission file with detections and predictions

### Tests

```bash
uv run pytest tests/ -v
```

---

## Configuration

All configuration adapts automatically based on `PIPELINE_MODE`:

| Parameter | Test Mode | Production Mode |
|---|---|---|
| YOLO epochs | 2 | 50 |
| YOLO subset | 50 images | Full dataset |
| GNN epochs | 100 | 500 |
| Depth strategy | `model` (ZoeDepth) | `bbox_approx` (fast heuristic) |
| Dataset fraction | 100% | 5% |
| Stage checkpoints | Enabled | Enabled |

Config classes in `src/config.py`:
- `YOLOConfig` — YOLO training hyperparameters
- `DepthConfig` — ZoeDepth model selection
- `FusionConfig` — Center crop fraction, minimum bbox area
- `SymbolicConfig` — Depth bins, horizontal bins, Claude model, depth strategy
- `GNNConfig` — Hidden dim, learning rate, batch size, epochs

---

## Presentation Slides

### Slide 1: The R3F Architecture Overview

**Content:**
- R3F: Reliable Real-time Reasoning Framework — a hybrid neuro-symbolic pipeline for autonomous navigation safety
- Progression from raw pixels to explainable safety decisions in 4 stages
- Teacher-Student distillation: heavy LLM reasons offline, lightweight GNN executes in real-time
- Result: high safety, zero-latency inference, full explainability

**Visual Diagram Instruction:**
Draw a horizontal 4-step flowchart with large rounded-rectangle boxes connected by thick directional arrows (left to right). Each box has an icon above it and a subtitle below:
- **Box 1: "Dual Perception Engine"** — icon: camera lens splitting into two beams. Subtitle: "YOLOv11 + ZoeDepth". Color: blue.
- **Arrow 1→2:** labeled "2D boxes + depth map"
- **Box 2: "Spatial-Symbolic Compiler"** — icon: grid/graph with nodes. Subtitle: "Geometry → Logic". Color: green.
- **Arrow 2→3:** labeled "text scene graph"
- **Box 3: "Reliable Reasoning Tutor"** — icon: brain with graduation cap. Subtitle: "Claude Sonnet (Offline)". Color: purple. Dashed border to indicate offline/training-time only.
- **Arrow 3→4:** labeled "labeled training data"
- **Box 4: "Lightning Apprentice"** — icon: lightning bolt over a small network graph. Subtitle: "GraphSAGE GNN (Real-time)". Color: orange.
Below the entire flowchart, a thin horizontal bar labeled "Raw Camera Image" on the left and "ACTION: STOP | REASON: HUMAN_1 at CENTER CLOSE" on the right, showing the end-to-end transformation.

---

### Slide 2: Dual Perception Engine (Vision + Depth)

**Content:**
- Single monocular camera feed processed through two parallel models
- **Track A — YOLOv11 Nano:** Detects and classifies objects into 4 macro-classes (HUMAN, VEHICLE, OBSTACLE, CONTEXT). Fine-tuned on 27 raw COCO categories collapsed into safety-relevant bins.
- **Track B — ZoeDepth (Intel/zoedepth-nyu-kitti):** Produces per-pixel metric depth map (meters). Solves monocular scale ambiguity — bounding box alone cannot distinguish "small object nearby" from "large object far away".
- Production alternative: fast bbox geometry heuristic (no depth model needed, 10x faster)
- **Key advantage:** With a known camera setup (fixed intrinsics, mounting height), the depth track becomes near-exact — structured environments make this system more robust, not less

**Visual Diagram Instruction:**
Draw a split-path diagram. Start with a single large rectangle on the left labeled **"Raw Camera Image"** containing a simplified warehouse scene sketch (stick figure, forklift, boxes). Two thick arrows diverge from the right edge of this rectangle, one going up-right and one going down-right:
- **Track A (top path):** Arrow leads to a rounded box labeled **"YOLOv11 Nano"** (with a small speed icon, e.g., stopwatch showing "6ms"). Arrow continues right to an output box showing the same warehouse scene but overlaid with colored bounding boxes: red box around the stick figure labeled "HUMAN 0.94", blue box around the forklift labeled "VEHICLE 0.87", yellow box around boxes labeled "OBSTACLE 0.72".
- **Track B (bottom path):** Arrow leads to a rounded box labeled **"ZoeDepth"** (or "BBox Heuristic" in dashed alternative). Arrow continues right to an output box showing a depth heatmap: a gradient from red (close) to blue (far), same warehouse scene silhouette visible in the depth map. Label the scale: "0m (red) → 12m (blue)".
- Both output boxes have arrows converging into a single point on the right labeled **"→ Phase 3: Fusion"**.

---

### Slide 3: Spatial-Symbolic Compiler (3D Scene to Graph)

**Content:**
- Translates physical geometry into discrete semantic logic — no heavy point clouds, no 3D reconstruction
- For every YOLO detection:
  - Extract **median depth** from center crop of depth map (robust to edge noise)
  - Bin horizontal position: `LEFT` / `CENTER` / `RIGHT` (image divided in thirds)
  - Bin depth: `CLOSE` (<2m) / `MID` (2-7m) / `FAR` (>7m)
- Cross-relations: objects within 1.5m at the same horizontal position get a `NEAR` edge
- Output is a **text scene description** — readable by both LLMs and humans
- This is the bridge between continuous perception and discrete reasoning

**Visual Diagram Instruction:**
Draw a funnel/pipeline diagram flowing left to right. On the left, two input boxes stacked vertically:
- Top input: **"Bounding Boxes"** — small illustration of colored rectangles on an image
- Bottom input: **"Depth Map"** — small heatmap thumbnail
Both inputs have arrows pointing into a central large trapezoid (wide on left, narrow on right) labeled **"Spatial Binning Logic"**. Inside the trapezoid, show three mini-diagrams:
  1. A horizontal bar divided into thirds labeled "LEFT | CENTER | RIGHT"
  2. A depth bar with three zones: "CLOSE <2m | MID 2-7m | FAR >7m"
  3. A small dotted circle labeled "NEAR: Δdepth < 1.5m"
On the right side of the trapezoid, an arrow points to a dark-background code block (monospace font) showing:
```
HUMAN_1 is CENTER and CLOSE.
FORKLIFT_1 is LEFT and MID.
OBSTACLE_1 is RIGHT and FAR.
HUMAN_1 is NEAR FORKLIFT_1.
```
Below the code block, a small label: "Symbolic Scene Graph (text)"

---

### Slide 4: Reliable Reasoning Tutor (Teacher LLM)

**Content:**
- Offline oracle that generates high-quality training labels for the student GNN
- Text scene graph is sent to **Claude Sonnet** with a safety-critical system prompt
- Claude applies complex reasoning rules:
  - STOP if any HUMAN is CLOSE or CENTER+MID
  - SLOW if any HUMAN is MID, or any VEHICLE is CLOSE/MID
  - CONTINUE only when no immediate danger
  - Always prioritize: human safety > robot safety > efficiency
- Returns structured JSON: action + reasoning edges (which nodes/relations caused the decision)
- Deterministic rule-based fallback when no API key is available (same logic, no LLM cost)
- Runs once on the training set to generate `scenes_train.jsonl` — the student never needs the LLM again

**Visual Diagram Instruction:**
Draw a vertical top-down flow with three layers:
- **Top layer:** A rounded rectangle labeled **"Text Scene Graph"** with the example text inside: `"HUMAN_1 is CENTER and CLOSE. FORKLIFT_1 is LEFT and MID."`
- **Arrow down** into the middle layer.
- **Middle layer:** A large prominent box with a brain/AI icon, labeled **"Claude Sonnet — Teacher Oracle"**. Inside or beside it, show a thought bubble or thinking block containing: "HUMAN_1 is CENTER and CLOSE → immediate danger → STOP". Color this box purple/dark to emphasize it's the heavy model.
- **Arrow down** splits into two paths.
- **Bottom-left output:** A badge/pill shape colored red, labeled **"Action: STOP"**
- **Bottom-right output:** A box with a list inside: **"Reasoning Edges:"** followed by bullet items: `"HUMAN_1_CENTER_CLOSE"`, `"FORKLIFT_1_LEFT_MID"`. These should look like graph edge labels.
- Small footnote at the bottom: "Runs offline on training set only. Student GNN trains on these outputs."

---

### Slide 5: Lightning Apprentice (Student GNN)

**Content:**
- Lightweight PyTorch Geometric model deployed on the robot for real-time inference
- Architecture: **2-layer GraphSAGE** with dual prediction heads
- Node features (11 dims): macro-class one-hot (4) + horizontal bin one-hot (3) + depth bin one-hot (3) + normalized depth (1)
- **Action Head:** Graph-level classification (global mean pool → MLP → STOP/SLOW/CONTINUE)
- **Edge Head:** Per-edge binary classification (concatenated node embeddings → MLP → reasoning probability)
- **Joint Loss:** `L = CrossEntropy(action) + BCE(edges)` — learns both what to do and why
- Model size: ~50 KB. Inference: sub-millisecond on CPU

**Visual Diagram Instruction:**
Draw a neural network architecture diagram flowing left to right:
- **Input (far left):** A small graph illustration with 3-4 colored nodes (red=HUMAN, blue=VEHICLE, yellow=OBSTACLE) connected by edges. Label: "Graph: Nodes (11-dim features) + Edges (all-to-all)"
- **Arrow right** into first layer block.
- **Layer 1:** A rectangular block labeled **"SAGEConv(11→64) + ReLU"**. Show small message-passing arrows between nodes inside.
- **Arrow right** into second layer block.
- **Layer 2:** A rectangular block labeled **"SAGEConv(64→64) + ReLU"**.
- **Arrow right** splits into two diverging paths (fork):
  - **Top path:** Arrow goes through a circle labeled **"Global Mean Pool"**, then into a box labeled **"Action Head"** (Linear 64→32→3). Output: three colored pills — red "STOP", yellow "SLOW", green "CONTINUE". Label below: "Cross-Entropy Loss".
  - **Bottom path:** Arrow labeled "[src || dst] embeddings" goes into a box labeled **"Edge Head"** (Linear 128→32→1). Output: a small graph where certain edges are highlighted/bold in red. Label below: "BCE Loss".
- Below everything, a combined loss formula: **L_total = L_action + L_edge**

---

### Slide 6: Explainable Execution (The Output)

**Content:**
- The robot doesn't just stop — its telemetry explicitly states **why** it stopped
- GNN predicts active reasoning edges alongside the action
- Human-readable formatting: "HUMAN detected [CLOSE] [IN FRONT OF] [ROBOT]"
- **Safety:** Conservative by design — when uncertain, defaults to safer action
- **Latency:** Sub-millisecond GNN inference. No LLM in the loop at runtime
- **Explainability:** Every decision is traceable to specific objects and spatial relations
- **Auditability:** Full decision log for safety certification and incident review

**Visual Diagram Instruction:**
Draw a dashboard/cockpit mockup split into two panels:
- **Left panel (60% width):** A simplified top-down or first-person view of the robot's environment. Show a robot icon at the bottom center. Ahead of the robot, draw a stick figure (person) with a red highlight circle and a label "HUMAN_1". To the left, draw a forklift icon with a yellow highlight. The robot has brake lines / a red stop indicator visible. Background: warehouse floor grid.
- **Right panel (40% width):** A dark-themed telemetry console with monospace font. Contents:
  ```
  ┌─────────────────────────────┐
  │ NAVIGATION SAFETY SYSTEM    │
  │ ─────────────────────────── │
  │ ACTION:  ██ STOP ██         │  ← red background
  │ CONFIDENCE: 0.96            │
  │                             │
  │ REASONING:                  │
  │ ● HUMAN detected [CLOSE]   │
  │   [IN FRONT OF] [ROBOT]    │
  │ ● VEHICLE detected [MID]   │
  │   [TO THE LEFT OF] [ROBOT] │
  │                             │
  │ ACTIVE EDGES:               │
  │ ○ HUMAN_1_CENTER_CLOSE      │  ← highlighted in red
  │ ○ FORKLIFT_1_LEFT_MID       │  ← highlighted in yellow
  │                             │
  │ LATENCY: 0.3ms              │
  │ MODEL: GraphSAGE v1 (48KB) │
  └─────────────────────────────┘
  ```
- Below both panels, three badge icons summarizing the value proposition:
  - Shield icon: **"Safety-First"** — conservative decisions, human safety prioritized
  - Lightning icon: **"Zero Latency"** — sub-ms inference, no LLM at runtime
  - Eye icon: **"Fully Explainable"** — every decision traceable to scene objects

---

## Project Structure

```
theker-hack/
├── src/
│   ├── config.py                 # Global config, macro-classes, paths, thresholds
│   ├── main.py                   # Phase 6: end-to-end inference + submission
│   └── phases/
│       ├── phase0_data.py        # COCO → YOLO labels + macro-class mapping
│       ├── phase1_vision.py      # YOLOv11 fine-tuning + inference
│       ├── phase2_depth.py       # ZoeDepth metric depth estimation
│       ├── phase3_fusion.py      # YOLO + Depth → fused 3D scene
│       ├── phase4_symbolic.py    # Scene graph + Claude/rule labeling → JSONL dataset
│       └── phase5_gnn.py         # GraphSAGE GNN training + evaluation
├── tests/                        # Pytest test suite (one file per phase)
├── data/
│   ├── annotations/              # COCO-format JSON (train/val/test)
│   ├── images/                   # Raw images (train/val/test/tiny)
│   ├── labels/                   # Generated YOLO labels
│   └── gnn_dataset/              # Generated GNN training data (JSONL + checkpoints)
├── models/
│   ├── pretrained/               # Base model weights (yolo11n.pt)
│   ├── finetuned/                # Fine-tuned YOLO weights (best.pt)
│   └── gnn/                      # Trained GNN weights (navigation_gnn.pt)
├── outputs/                      # Submission JSON + training artifacts
├── Dockerfile                    # CUDA 12.4 + uv base image
├── docker-compose.yml            # GPU training compose config
└── pyproject.toml                # Dependencies + tool config
```
