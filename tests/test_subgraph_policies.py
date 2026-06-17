"""Tests for subgraph generation policies.

Fixture: a triangle graph (3 nodes, 3 undirected edges = 6 directed entries)
with 4-dim node features so feature preservation can be checked.
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from src.data.subgraph_policies import (
    edge_deletion_subgraphs,
    ego_subgraphs,
    node_deletion_subgraphs,
)


@pytest.fixture
def triangle() -> Data:
    # Nodes 0, 1, 2 fully connected. Each undirected edge stored both ways.
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 0, 2],
         [1, 0, 2, 1, 2, 0]],
        dtype=torch.long,
    )
    x = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0]],
        dtype=torch.float,
    )
    return Data(x=x, edge_index=edge_index, num_nodes=3)


def test_node_deletion_count_and_size(triangle):
    bag = node_deletion_subgraphs(triangle)
    assert len(bag) == 3
    for sub in bag:
        assert sub.num_nodes == 2
        assert sub.x.shape == (2, 4)
        # Triangle minus one node leaves a single undirected edge -> 2 directed.
        assert sub.edge_index.size(1) == 2
        assert hasattr(sub, "parent_nodes")
        assert sub.parent_nodes.numel() == 2


def test_node_deletion_removes_correct_node(triangle):
    bag = node_deletion_subgraphs(triangle)
    # Subgraph i should contain every original node except i.
    for i, sub in enumerate(bag):
        kept = sub.parent_nodes.tolist()
        assert i not in kept
        assert sorted(kept) == sorted(set(range(3)) - {i})


def test_node_deletion_feature_preservation(triangle):
    bag = node_deletion_subgraphs(triangle)
    for sub in bag:
        # Each row of sub.x must equal the corresponding row of the original x.
        for new_idx, orig_idx in enumerate(sub.parent_nodes.tolist()):
            assert torch.equal(sub.x[new_idx], triangle.x[orig_idx])


def test_node_deletion_edges_relabeled(triangle):
    bag = node_deletion_subgraphs(triangle)
    for sub in bag:
        assert sub.edge_index.max().item() < sub.num_nodes
        assert sub.edge_index.min().item() >= 0


def test_edge_deletion_count(triangle):
    bag = edge_deletion_subgraphs(triangle)
    # Triangle has 3 unique undirected edges.
    assert len(bag) == 3


def test_edge_deletion_preserves_nodes_and_features(triangle):
    bag = edge_deletion_subgraphs(triangle)
    for sub in bag:
        assert sub.num_nodes == 3
        assert torch.equal(sub.x, triangle.x)
        assert sub.parent_nodes.tolist() == [0, 1, 2]


def test_edge_deletion_removes_one_undirected_edge(triangle):
    bag = edge_deletion_subgraphs(triangle)
    original_directed = triangle.edge_index.size(1)
    for sub in bag:
        # Removing one undirected edge drops both directed entries.
        assert sub.edge_index.size(1) == original_directed - 2


def test_edge_deletion_no_duplicates(triangle):
    bag = edge_deletion_subgraphs(triangle)
    # Collect the undirected edge missing from each subgraph; all must be distinct.
    full = {tuple(sorted((int(a), int(b))))
            for a, b in triangle.edge_index.t().tolist()}
    missing = []
    for sub in bag:
        present = {tuple(sorted((int(a), int(b))))
                   for a, b in sub.edge_index.t().tolist()}
        diff = full - present
        assert len(diff) == 1
        missing.append(next(iter(diff)))
    assert len(set(missing)) == len(missing)


def test_ego_subgraphs_count(triangle):
    bag = ego_subgraphs(triangle, hops=1)
    assert len(bag) == 3


def test_ego_subgraphs_one_hop_covers_triangle(triangle):
    # In a triangle every node's 1-hop neighborhood is the whole graph.
    bag = ego_subgraphs(triangle, hops=1)
    for i, sub in enumerate(bag):
        assert sub.num_nodes == 3
        assert sub.x.shape == (3, 4)
        assert int(sub.center_node.item()) == i
        assert sorted(sub.parent_nodes.tolist()) == [0, 1, 2]


def test_ego_subgraphs_zero_hop(triangle):
    bag = ego_subgraphs(triangle, hops=0)
    # 0-hop ego is just the center node, no edges.
    assert len(bag) == 3
    for i, sub in enumerate(bag):
        assert sub.num_nodes == 1
        assert int(sub.center_node.item()) == i
        assert sub.parent_nodes.tolist() == [i]
        assert sub.edge_index.size(1) == 0


def test_ego_subgraphs_feature_preservation(triangle):
    bag = ego_subgraphs(triangle, hops=1)
    for sub in bag:
        for new_idx, orig_idx in enumerate(sub.parent_nodes.tolist()):
            assert torch.equal(sub.x[new_idx], triangle.x[orig_idx])
