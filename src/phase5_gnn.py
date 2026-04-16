"""Phase 5: GNN Student Model — learns to predict actions + reasoning edges.

Joint-loss model with:
- Head 1: Graph-level classification (STOP / SLOW / CONTINUE)
- Head 2: Edge-level binary classification (is this edge part of reasoning?)

Uses GraphSAGE convolutions from PyTorch Geometric.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import SAGEConv, global_mean_pool

# Action label mapping
ACTION_TO_IDX: dict[str, int] = {"STOP": 0, "SLOW": 1, "CONTINUE": 2}
IDX_TO_ACTION: dict[int, str] = {v: k for k, v in ACTION_TO_IDX.items()}

# Node feature dimensions:
#   macro_class one-hot (4) + h_bin one-hot (3) + d_bin one-hot (3) + depth_m (1) = 11
NODE_FEAT_DIM = 11

# Encodings
MACRO_CLASSES = {"HUMAN": 0, "VEHICLE": 1, "OBSTACLE": 2, "CONTEXT": 3}
H_BINS = {"LEFT": 0, "CENTER": 1, "RIGHT": 2}
D_BINS = {"CLOSE": 0, "MID": 1, "FAR": 2}


def encode_node_features(
    macro_class: str,
    h_bin: str,
    d_bin: str,
    depth_m: float,
) -> list[float]:
    """Encode a scene node into a fixed-length feature vector.

    Returns:
        List of 11 floats: [macro_oh(4), h_bin_oh(3), d_bin_oh(3), depth(1)]
    """
    feat = [0.0] * NODE_FEAT_DIM
    # Macro-class one-hot (positions 0-3)
    feat[MACRO_CLASSES.get(macro_class, 0)] = 1.0
    # H-bin one-hot (positions 4-6)
    feat[4 + H_BINS.get(h_bin, 1)] = 1.0
    # D-bin one-hot (positions 7-9)
    feat[7 + D_BINS.get(d_bin, 1)] = 1.0
    # Normalized depth (position 10)
    feat[10] = depth_m / 20.0  # normalize roughly to [0, 1]
    return feat


def scene_to_pyg_graph(
    fused_objects: list[dict],
    action: str,
    reasoning_edges: list[str],
    img_w: int = 640,
) -> Data:
    """Convert a fused scene + Claude labels into a PyG Data object.

    Args:
        fused_objects: List from phase3_fusion output.
        action: Ground truth action label from Claude.
        reasoning_edges: List of edge strings from Claude.
        img_w: Image width for horizontal binning.

    Returns:
        PyG Data with node features, edge_index, graph label, edge labels.
    """
    from phase4_symbolic import _bin_horizontal, _bin_depth, SymbolicConfig

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

    # Fully connected edges (all pairs, both directions)
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)

    # Graph-level label
    y = torch.tensor([ACTION_TO_IDX.get(action, 0)], dtype=torch.long)

    # Edge-level labels: 1 if this edge pair is in reasoning_edges
    reasoning_set = set(reasoning_edges)
    edge_labels = []
    for i, j in zip(src, dst):
        # Check if this edge matches any reasoning edge pattern
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
    """Joint-loss GNN: graph classification + edge classification.

    Architecture:
        2x GraphSAGE conv layers -> global_mean_pool -> action head (3-class)
        Edge pairs through MLP -> reasoning head (binary)
    """

    def __init__(
        self,
        in_channels: int = NODE_FEAT_DIM,
        hidden: int = 64,
        num_actions: int = 3,
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)

        # Graph-level action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions),
        )

        # Edge-level reasoning head
        self.edge_head = nn.Sequential(
            nn.Linear(hidden * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, data: Data | Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            data: PyG Data or Batch.

        Returns:
            action_logits: [batch_size, 3] graph classification logits.
            edge_logits: [num_edges] edge reasoning logits (pre-sigmoid).
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Node embeddings
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Graph-level: pool + classify
        graph_emb = global_mean_pool(x, batch)  # [batch_size, hidden]
        action_logits = self.action_head(graph_emb)  # [batch_size, 3]

        # Edge-level: concat src+dst embeddings
        if edge_index.numel() > 0:
            src_emb = x[edge_index[0]]  # [num_edges, hidden]
            dst_emb = x[edge_index[1]]  # [num_edges, hidden]
            edge_emb = torch.cat([src_emb, dst_emb], dim=1)  # [num_edges, hidden*2]
            edge_logits = self.edge_head(edge_emb).squeeze(-1)  # [num_edges]
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
    """Single training step with joint loss.

    Returns:
        Dict with total_loss, action_loss, edge_loss.
    """
    model.train()
    optimizer.zero_grad()

    action_logits, edge_logits = model(batch)

    # Action loss: cross-entropy
    action_loss = F.cross_entropy(action_logits, batch.y)

    # Edge loss: binary cross-entropy
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
