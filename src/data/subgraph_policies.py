"""Subgraph generation policies for DPP-based bag construction.

Each policy takes a PyG ``Data`` object and returns a list of ``Data`` subgraphs
(a "bag"). Every produced subgraph carries ``parent_nodes`` (a LongTensor of
original node indices it retains) so downstream components can map back to the
parent graph.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph


def _induced_subgraph(data: Data, keep_mask: torch.Tensor) -> Data:
    """Build an induced subgraph over nodes where ``keep_mask`` is True."""
    keep_idx = keep_mask.nonzero(as_tuple=False).view(-1)
    edge_index, edge_attr = subgraph(
        keep_idx,
        data.edge_index,
        edge_attr=getattr(data, "edge_attr", None),
        relabel_nodes=True,
        num_nodes=data.num_nodes,
    )
    sub = Data(edge_index=edge_index)
    if data.x is not None:
        sub.x = data.x[keep_idx]
    if edge_attr is not None:
        sub.edge_attr = edge_attr
    if getattr(data, "y", None) is not None:
        sub.y = data.y
    sub.num_nodes = int(keep_idx.numel())
    sub.parent_nodes = keep_idx.clone()
    return sub


def node_deletion_subgraphs(data: Data) -> list[Data]:
    """Generate a bag by deleting one node at a time.

    For a graph with N nodes, returns N subgraphs, each missing exactly one node
    (and all incident edges). Node features are reindexed to the surviving nodes.
    """
    n = data.num_nodes
    bag: list[Data] = []
    for v in range(n):
        keep = torch.ones(n, dtype=torch.bool)
        keep[v] = False
        bag.append(_induced_subgraph(data, keep))
    return bag


def edge_deletion_subgraphs(data: Data) -> list[Data]:
    """Generate a bag by deleting one undirected edge at a time.

    For each unique undirected edge {u, v}, both directed entries (u->v and
    v->u) are removed. Self-loops are treated as a single edge. Node set is
    preserved; only ``edge_index`` (and ``edge_attr``) shrinks.
    """
    edge_index = data.edge_index
    edge_attr = getattr(data, "edge_attr", None)
    num_edges = edge_index.size(1)

    seen: set[tuple[int, int]] = set()
    unique_edges: list[tuple[int, int]] = []
    for k in range(num_edges):
        u = int(edge_index[0, k].item())
        v = int(edge_index[1, k].item())
        key = (u, v) if u <= v else (v, u)
        if key in seen:
            continue
        seen.add(key)
        unique_edges.append(key)

    bag: list[Data] = []
    parent_nodes = torch.arange(data.num_nodes, dtype=torch.long)
    src = edge_index[0]
    dst = edge_index[1]

    for u, v in unique_edges:
        mask_uv = (src == u) & (dst == v)
        mask_vu = (src == v) & (dst == u)
        keep = ~(mask_uv | mask_vu)
        sub_edge_index = edge_index[:, keep]
        sub = Data(edge_index=sub_edge_index)
        if data.x is not None:
            sub.x = data.x.clone()
        if edge_attr is not None:
            sub.edge_attr = edge_attr[keep]
        if getattr(data, "y", None) is not None:
            sub.y = data.y
        sub.num_nodes = data.num_nodes
        sub.parent_nodes = parent_nodes.clone()
        bag.append(sub)

    return bag


def ego_subgraphs(data: Data, hops: int = 1) -> list[Data]:
    """Generate k-hop ego subgraphs, one per node.

    Each ego subgraph is the induced subgraph on the ``hops``-hop neighborhood
    around a center node. ``center_node`` (original index) and ``parent_nodes``
    are stored on each returned ``Data``.
    """
    n = data.num_nodes
    bag: list[Data] = []

    for center in range(n):
        node_idx, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=center,
            num_hops=hops,
            edge_index=data.edge_index,
            relabel_nodes=True,
            num_nodes=n,
        )
        sub = Data(edge_index=sub_edge_index)
        if data.x is not None:
            sub.x = data.x[node_idx]
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is not None:
            sub.edge_attr = edge_attr[edge_mask]
        if getattr(data, "y", None) is not None:
            sub.y = data.y
        sub.num_nodes = int(node_idx.numel())
        sub.parent_nodes = node_idx.clone()
        sub.center_node = torch.tensor(center, dtype=torch.long)
        bag.append(sub)

    return bag
