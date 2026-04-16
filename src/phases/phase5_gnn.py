"""Phase 5: GNN Student Model — train on synthetic dataset from Phase 4.

Usage: uv run python -m src.phases.phase5_gnn
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

from src.config import DATA_DIR, GNN_DIR, GNNConfig

# Action label mapping
ACTION_TO_IDX: dict[str, int] = {"STOP": 0, "SLOW": 1, "CONTINUE": 2}
IDX_TO_ACTION: dict[int, str] = {v: k for k, v in ACTION_TO_IDX.items()}

# Node feature dimensions: macro(4) + h_bin(3) + d_bin(3) + depth(1) = 11
NODE_FEAT_DIM = 11

MACRO_CLASSES = {"HUMAN": 0, "VEHICLE": 1, "OBSTACLE": 2, "CONTEXT": 3}
H_BINS = {"LEFT": 0, "CENTER": 1, "RIGHT": 2}
D_BINS = {"CLOSE": 0, "MID": 1, "FAR": 2}


def encode_node_features(
    macro_class: str, h_bin: str, d_bin: str, depth_m: float,
) -> list[float]:
    """Encode a scene node into a fixed-length feature vector (11 dims)."""
    feat = [0.0] * NODE_FEAT_DIM
    feat[MACRO_CLASSES.get(macro_class, 0)] = 1.0
    feat[4 + H_BINS.get(h_bin, 1)] = 1.0
    feat[7 + D_BINS.get(d_bin, 1)] = 1.0
    feat[10] = depth_m / 20.0
    return feat


def scene_to_pyg_graph(
    fused_objects: list[dict],
    action: str,
    reasoning_edges: list[str],
    img_w: int = 640,
) -> Data:
    """Convert a fused scene + labels into a PyG Data object."""
    from src.phases.phase4_symbolic import _bin_horizontal, _bin_depth, SymbolicConfig

    cfg = SymbolicConfig()
    nodes_feats: list[list[float]] = []
    node_labels: list[str] = []
    class_counters: dict[str, int] = {}

    for obj in fused_objects:
        macro = obj.get("class", "UNKNOWN")
        class_counters[macro] = class_counters.get(macro, 0) + 1
        label = f"{macro}_{class_counters[macro]}"
        node_labels.append(label)
        h_bin = _bin_horizontal(obj["bbox"], img_w)
        d_bin = _bin_depth(obj["depth_m"], cfg)
        nodes_feats.append(encode_node_features(macro, h_bin, d_bin, obj["depth_m"]))

    x = torch.tensor(nodes_feats, dtype=torch.float)
    n = len(node_labels)

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

    y = torch.tensor([ACTION_TO_IDX.get(action, 0)], dtype=torch.long)

    reasoning_set = set(reasoning_edges)
    edge_labels = []
    for i, j in zip(src, dst):
        ni, nj = node_labels[i], node_labels[j]
        is_reasoning = 0.0
        for edge_str in reasoning_set:
            if ni in edge_str and nj in edge_str:
                is_reasoning = 1.0
                break
            if f"{ni}_NEAR_{nj}" in reasoning_set or f"{nj}_NEAR_{ni}" in reasoning_set:
                is_reasoning = 1.0
                break
        edge_labels.append(is_reasoning)

    edge_y = torch.tensor(edge_labels, dtype=torch.float) if edge_labels else torch.zeros(0)
    return Data(x=x, edge_index=edge_index, y=y, edge_y=edge_y)


# ---------------------------------------------------------------------------
# GNN Model
# ---------------------------------------------------------------------------

class NavigationGNN(nn.Module):
    """Joint-loss GNN: graph classification + edge classification."""

    def __init__(
        self,
        in_channels: int = NODE_FEAT_DIM,
        hidden: int = 64,
        num_actions: int = 3,
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.action_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, num_actions),
        )
        self.edge_head = nn.Sequential(
            nn.Linear(hidden * 2, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, data: Data | Batch) -> tuple[torch.Tensor, torch.Tensor]:
        x, edge_index = data.x, data.edge_index
        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        graph_emb = global_mean_pool(x, batch)
        action_logits = self.action_head(graph_emb)

        if edge_index.numel() > 0:
            src_emb = x[edge_index[0]]
            dst_emb = x[edge_index[1]]
            edge_emb = torch.cat([src_emb, dst_emb], dim=1)
            edge_logits = self.edge_head(edge_emb).squeeze(-1)
        else:
            edge_logits = torch.zeros(0, device=x.device)

        return action_logits, edge_logits


def train_step(
    model: NavigationGNN,
    batch: Batch,
    optimizer: torch.optim.Optimizer,
    action_weight: float = 1.0,
    edge_weight: float = 1.0,
) -> dict[str, float]:
    """Single training step with joint loss."""
    model.train()
    optimizer.zero_grad()

    action_logits, edge_logits = model(batch)
    action_loss = F.cross_entropy(action_logits, batch.y)

    if edge_logits.numel() > 0 and batch.edge_y.numel() > 0:
        edge_loss = F.binary_cross_entropy_with_logits(edge_logits, batch.edge_y)
    else:
        edge_loss = torch.tensor(0.0, device=action_logits.device)

    total_loss = action_weight * action_loss + edge_weight * edge_loss
    total_loss.backward()
    optimizer.step()

    return {
        "total_loss": total_loss.item(),
        "action_loss": action_loss.item(),
        "edge_loss": edge_loss.item(),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def load_dataset() -> list[Data]:
    """Load scenes.jsonl and convert to PyG graphs."""
    scenes_path = DATA_DIR / "gnn_dataset" / "scenes.jsonl"
    if not scenes_path.exists():
        raise FileNotFoundError(f"{scenes_path} not found — run Phase 4 first")

    graphs: list[Data] = []
    with open(scenes_path) as f:
        for line in f:
            scene = json.loads(line)
            g = scene_to_pyg_graph(
                scene["objects"], scene["action"], scene["reasoning_edges"],
            )
            graphs.append(g)

    print(f"Loaded {len(graphs)} graphs from {scenes_path}")
    return graphs


def train_gnn(cfg: GNNConfig | None = None) -> Path:
    """Train GNN on synthetic dataset and save weights."""
    cfg = cfg or GNNConfig()
    GNN_DIR.mkdir(parents=True, exist_ok=True)

    graphs = load_dataset()
    if not graphs:
        raise ValueError("No graphs to train on")

    loader = DataLoader(graphs, batch_size=cfg.batch_size, shuffle=True)
    model = NavigationGNN(hidden=cfg.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        for batch in loader:
            metrics = train_step(model, batch, optimizer)
            epoch_loss += metrics["total_loss"]

        if (epoch + 1) % max(1, cfg.epochs // 10) == 0:
            print(
                f"  Epoch {epoch+1:4d}/{cfg.epochs} | "
                f"loss={epoch_loss / len(loader):.4f}"
            )

    save_path = Path(cfg.save_path)
    torch.save(model.state_dict(), save_path)
    print(f"GNN saved to {save_path}")
    return save_path


def main() -> None:
    """Train GNN on Phase 4 dataset."""
    cfg = GNNConfig()
    print(f"=== Phase 5: GNN training ({cfg.epochs} epochs) ===")
    train_gnn(cfg)
    print("=== Phase 5 complete ===")


if __name__ == "__main__":
    main()
