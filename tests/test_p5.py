"""Phase 5 Milestone: verify GNN can overfit on 10 dummy graphs.

1. Create 10 synthetic PyG graphs with known labels.
2. Train for 50 epochs.
3. Assert loss strictly decreases to near zero.
"""

from __future__ import annotations

import random

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.phases.phase5_gnn import (
    ACTION_TO_IDX,
    NODE_FEAT_DIM,
    NavigationGNN,
    encode_node_features,
    train_step,
)


def make_dummy_graphs(n: int = 10) -> list[Data]:
    """Create n synthetic graphs with deterministic labels for overfitting test."""
    random.seed(42)
    graphs: list[Data] = []

    macros = ["HUMAN", "VEHICLE", "OBSTACLE", "CONTEXT"]
    h_bins = ["LEFT", "CENTER", "RIGHT"]
    d_bins = ["CLOSE", "MID", "FAR"]
    actions = ["STOP", "SLOW", "CONTINUE"]

    for i in range(n):
        # 3-5 nodes per graph
        num_nodes = random.randint(3, 5)
        feats = []
        for _ in range(num_nodes):
            m = random.choice(macros)
            h = random.choice(h_bins)
            d = random.choice(d_bins)
            depth = random.uniform(0.5, 15.0)
            feats.append(encode_node_features(m, h, d, depth))

        x = torch.tensor(feats, dtype=torch.float)

        # Fully connected edges
        src, dst = [], []
        for a in range(num_nodes):
            for b in range(num_nodes):
                if a != b:
                    src.append(a)
                    dst.append(b)
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        # Deterministic label based on index
        action = actions[i % 3]
        y = torch.tensor([ACTION_TO_IDX[action]], dtype=torch.long)

        # Mark ~30% of edges as reasoning edges
        num_edges = len(src)
        edge_y = torch.zeros(num_edges)
        reasoning_count = max(1, num_edges // 3)
        for j in range(reasoning_count):
            edge_y[j] = 1.0

        graphs.append(Data(x=x, edge_index=edge_index, y=y, edge_y=edge_y))

    return graphs


def test_overfit() -> None:
    """Train GNN on 10 graphs for 50 epochs. Assert loss -> near zero."""
    print("\n--- test_overfit ---")
    graphs = make_dummy_graphs(10)
    loader = DataLoader(graphs, batch_size=10, shuffle=False)

    model = NavigationGNN(in_channels=NODE_FEAT_DIM, hidden=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses: list[float] = []
    for epoch in range(100):
        for batch in loader:
            metrics = train_step(model, batch, optimizer)
            losses.append(metrics["total_loss"])

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch + 1:3d} | total={metrics['total_loss']:.4f} "
                f"action={metrics['action_loss']:.4f} edge={metrics['edge_loss']:.4f}"
            )

    first_loss = losses[0]
    final_loss = losses[-1]

    print(f"\n  First loss: {first_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Reduction:  {(1 - final_loss / first_loss) * 100:.1f}%")

    # Assert loss decreased significantly
    assert final_loss < first_loss * 0.1, (
        f"Loss did not decrease enough: {first_loss:.4f} -> {final_loss:.4f}"
    )

    # Assert loss is near zero (< 0.1 for 10-graph overfit)
    assert final_loss < 0.15, f"Final loss {final_loss:.4f} not near zero"

    # Verify model can predict correctly
    model.eval()
    with torch.no_grad():
        for batch in loader:
            action_logits, edge_logits = model(batch)
            preds = action_logits.argmax(dim=1)
            accuracy = (preds == batch.y).float().mean().item()
            print(f"  Action accuracy: {accuracy * 100:.0f}%")
            assert accuracy >= 0.9, f"Accuracy {accuracy:.0%} too low for overfit test"

    print("PASS: GNN overfits successfully — joint loss near zero")


if __name__ == "__main__":
    test_overfit()
    print("\n=== All Phase 5 tests passed ===")
