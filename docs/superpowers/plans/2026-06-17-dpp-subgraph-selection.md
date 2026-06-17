# DPP-Based Margin-Aware Subgraph Selection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a subgraph GNN that selects subgraphs via a Determinantal Point Process with margin-contribution as the quality signal and embedding correlation as the diversity kernel, outperforming fixed/random/centrality/expressivity-based selection (ESAN, HyMN, Policy-Learn).

**Architecture:** ESAN-style bag-of-subgraphs architecture where the subgraph bag is filtered by a DPP-based selector before aggregation. The DPP kernel has two components: (1) a quality model that scores each subgraph by its predicted margin contribution, and (2) a similarity kernel based on subgraph embeddings that penalizes redundancy. Training alternates between updating the base GNN classifier and updating the DPP quality/similarity parameters.

**Tech Stack:** PyTorch, PyTorch Geometric (PyG), DPPy (DPP sampling), OGB (benchmarks), ZINC/TU datasets.

---

## File Structure

```
src/
├── data/
│   ├── subgraph_policies.py     # Subgraph generation (node-del, edge-del, ego)
│   └── datasets.py              # ZINC, TU, OGB-molhiv loaders with subgraph bags
├── models/
│   ├── base_gnn.py              # GIN/GCN encoder (processes single subgraph)
│   ├── bag_aggregator.py        # DeepSets/attention aggregation over selected bag
│   ├── dpp_selector.py          # DPP kernel construction + sampling/selection
│   └── margin_scorer.py         # Per-subgraph margin contribution estimator
├── training/
│   ├── trainer.py               # Training loop with alternating DPP + classifier updates
│   ├── margin_utils.py          # Margin computation (distance to decision boundary)
│   └── losses.py                # Classification loss + DPP log-likelihood
├── baselines/
│   ├── esan_uniform.py          # ESAN with uniform random subgraph sampling
│   ├── esan_full.py             # ESAN with full bag (no selection)
│   └── centrality_select.py     # HyMN-style centrality-based selection
├── experiments/
│   ├── run_zinc.py              # ZINC-12K experiment script
│   ├── run_tu.py                # TU datasets experiment script
│   └── run_ogb.py               # OGB-molhiv experiment script
└── utils.py                     # Logging, seeding, config
tests/
├── test_subgraph_policies.py
├── test_dpp_selector.py
├── test_margin_scorer.py
├── test_bag_aggregator.py
├── test_trainer.py
└── test_baselines.py
configs/
├── zinc.yaml
├── tu.yaml
└── ogb.yaml
```

---

### Task 1: Project Skeleton + Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `setup.py`
- Create: `src/__init__.py`
- Create: `src/data/__init__.py`
- Create: `src/models/__init__.py`
- Create: `src/training/__init__.py`
- Create: `src/baselines/__init__.py`
- Create: `src/experiments/__init__.py`
- Create: `tests/__init__.py`
- Create: `configs/zinc.yaml`

- [ ] **Step 1: Create requirements.txt**

```
torch>=2.1.0
torch-geometric>=2.4.0
torch-scatter
torch-sparse
ogb>=1.3.6
dppy>=0.3.2
numpy>=1.24
scipy>=1.10
pyyaml>=6.0
pytest>=7.0
```

- [ ] **Step 2: Create setup.py**

```python
from setuptools import setup, find_packages

setup(
    name="dpp-subgraph-select",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
)
```

- [ ] **Step 3: Create all __init__.py files (empty)**

```bash
mkdir -p src/data src/models src/training src/baselines src/experiments tests configs
touch src/__init__.py src/data/__init__.py src/models/__init__.py src/training/__init__.py src/baselines/__init__.py src/experiments/__init__.py tests/__init__.py
```

- [ ] **Step 4: Create initial config**

```yaml
# configs/zinc.yaml
dataset:
  name: ZINC
  subset: true  # ZINC-12K subset

model:
  hidden_dim: 64
  num_layers: 4
  dropout: 0.0
  pool: sum

selection:
  policy: node_deletion  # subgraph generation policy
  budget_k: 10           # number of subgraphs to select
  dpp_temperature: 1.0   # temperature for DPP sampling

training:
  epochs: 300
  lr: 0.001
  weight_decay: 0.0
  batch_size: 128
  margin_warmup_epochs: 10  # train classifier before computing margins

evaluation:
  metric: mae  # ZINC uses MAE
  seeds: [0, 1, 2, 3, 4]
```

- [ ] **Step 5: Install and verify**

Run: `pip install -e . && pip install -r requirements.txt`
Expected: clean install, `import src` works

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: project skeleton with dependencies and config"
```

---

### Task 2: Subgraph Generation Policies

**Files:**
- Create: `src/data/subgraph_policies.py`
- Create: `tests/test_subgraph_policies.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_subgraph_policies.py
import torch
from torch_geometric.data import Data
from src.data.subgraph_policies import (
    node_deletion_subgraphs,
    edge_deletion_subgraphs,
    ego_subgraphs,
)


def make_triangle():
    """3-node triangle graph."""
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long)
    x = torch.randn(3, 4)
    return Data(x=x, edge_index=edge_index, num_nodes=3)


def test_node_deletion_count():
    g = make_triangle()
    subs = node_deletion_subgraphs(g)
    assert len(subs) == 3  # one per node


def test_node_deletion_removes_node():
    g = make_triangle()
    subs = node_deletion_subgraphs(g)
    for sub in subs:
        assert sub.num_nodes == 2


def test_edge_deletion_count():
    g = make_triangle()
    subs = edge_deletion_subgraphs(g)
    # 3 undirected edges -> 3 subgraphs
    assert len(subs) == 3


def test_ego_subgraphs_count():
    g = make_triangle()
    subs = ego_subgraphs(g, hops=1)
    assert len(subs) == 3  # one per node


def test_ego_preserves_features():
    g = make_triangle()
    subs = ego_subgraphs(g, hops=1)
    for sub in subs:
        assert sub.x.shape[1] == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_subgraph_policies.py -v`
Expected: FAIL (ImportError — module doesn't exist)

- [ ] **Step 3: Implement subgraph policies**

```python
# src/data/subgraph_policies.py
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, k_hop_subgraph


def node_deletion_subgraphs(data: Data) -> list[Data]:
    """Generate bag of subgraphs by deleting one node at a time."""
    n = data.num_nodes
    subgraphs = []
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        keep_tensor = torch.tensor(keep, dtype=torch.long)
        edge_idx, _, edge_mask = subgraph(
            keep_tensor, data.edge_index, relabel_nodes=True, num_nodes=n
        )
        x = data.x[keep_tensor]
        sub = Data(x=x, edge_index=edge_idx, num_nodes=len(keep))
        if data.edge_attr is not None and edge_mask is not None:
            sub.edge_attr = data.edge_attr[edge_mask]
        sub.parent_nodes = keep_tensor
        subgraphs.append(sub)
    return subgraphs


def edge_deletion_subgraphs(data: Data) -> list[Data]:
    """Generate bag of subgraphs by deleting one undirected edge at a time."""
    edge_index = data.edge_index
    seen = set()
    subgraphs = []
    for idx in range(edge_index.shape[1]):
        u, v = edge_index[0, idx].item(), edge_index[1, idx].item()
        edge_key = (min(u, v), max(u, v))
        if edge_key in seen:
            continue
        seen.add(edge_key)
        mask = ~(
            ((edge_index[0] == u) & (edge_index[1] == v))
            | ((edge_index[0] == v) & (edge_index[1] == u))
        )
        new_edge_index = edge_index[:, mask]
        sub = Data(x=data.x.clone(), edge_index=new_edge_index, num_nodes=data.num_nodes)
        if data.edge_attr is not None:
            sub.edge_attr = data.edge_attr[mask]
        subgraphs.append(sub)
    return subgraphs


def ego_subgraphs(data: Data, hops: int = 1) -> list[Data]:
    """Generate bag of k-hop ego subgraphs, one per node."""
    n = data.num_nodes
    subgraphs = []
    for i in range(n):
        subset, sub_edge_index, _, _ = k_hop_subgraph(
            i, hops, data.edge_index, relabel_nodes=True, num_nodes=n
        )
        x = data.x[subset]
        sub = Data(x=x, edge_index=sub_edge_index, num_nodes=len(subset))
        sub.center_node = torch.tensor([0])  # center is relabeled to 0
        sub.parent_nodes = subset
        subgraphs.append(sub)
    return subgraphs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_subgraph_policies.py -v`
Expected: all 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/subgraph_policies.py tests/test_subgraph_policies.py
git commit -m "feat: subgraph generation policies (node-del, edge-del, ego)"
```

---

### Task 3: Base GNN Encoder

**Files:**
- Create: `src/models/base_gnn.py`
- Create: `tests/test_base_gnn.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_base_gnn.py
import torch
from torch_geometric.data import Data
from src.models.base_gnn import GINEncoder


def make_graph():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.randn(3, 7)
    return Data(x=x, edge_index=edge_index, num_nodes=3)


def test_gin_output_shape():
    model = GINEncoder(in_dim=7, hidden_dim=32, out_dim=16, num_layers=3)
    g = make_graph()
    out = model(g)
    assert out.shape == (16,)  # graph-level embedding


def test_gin_batch():
    from torch_geometric.data import Batch
    model = GINEncoder(in_dim=7, hidden_dim=32, out_dim=16, num_layers=3)
    graphs = [make_graph() for _ in range(4)]
    batch = Batch.from_data_list(graphs)
    out = model(batch)
    assert out.shape == (4, 16)


def test_gin_deterministic():
    model = GINEncoder(in_dim=7, hidden_dim=32, out_dim=16, num_layers=3)
    model.eval()
    g = make_graph()
    out1 = model(g)
    out2 = model(g)
    assert torch.allclose(out1, out2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_base_gnn.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement GIN encoder**

```python
# src/models/base_gnn.py
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Data, Batch


class GINEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            dim_in = in_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(dim_in, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.project = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        if isinstance(data, Data) and not hasattr(data, "batch"):
            data = Batch.from_data_list([data])
            squeeze = True
        else:
            squeeze = False

        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)

        x = global_add_pool(x, batch)
        x = self.project(x)

        if squeeze:
            x = x.squeeze(0)
        return x
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_base_gnn.py -v`
Expected: all 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/base_gnn.py tests/test_base_gnn.py
git commit -m "feat: GIN encoder for subgraph embedding"
```

---

### Task 4: DPP Selector

**Files:**
- Create: `src/models/dpp_selector.py`
- Create: `tests/test_dpp_selector.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dpp_selector.py
import torch
from src.models.dpp_selector import DPPSelector


def test_dpp_selector_output_size():
    selector = DPPSelector(embed_dim=16, budget_k=3)
    embeddings = torch.randn(10, 16)  # 10 candidate subgraphs
    quality_scores = torch.randn(10)  # margin contribution scores
    selected_indices = selector(embeddings, quality_scores)
    assert len(selected_indices) == 3


def test_dpp_selector_indices_valid():
    selector = DPPSelector(embed_dim=16, budget_k=5)
    embeddings = torch.randn(20, 16)
    quality_scores = torch.randn(20)
    selected_indices = selector(embeddings, quality_scores)
    assert all(0 <= idx < 20 for idx in selected_indices)
    assert len(set(selected_indices)) == 5  # no duplicates


def test_dpp_selector_budget_exceeds_candidates():
    selector = DPPSelector(embed_dim=16, budget_k=10)
    embeddings = torch.randn(5, 16)  # fewer candidates than budget
    quality_scores = torch.randn(5)
    selected_indices = selector(embeddings, quality_scores)
    assert len(selected_indices) == 5  # return all


def test_dpp_selector_differentiable_relaxation():
    selector = DPPSelector(embed_dim=16, budget_k=3)
    embeddings = torch.randn(10, 16, requires_grad=True)
    quality_scores = torch.randn(10, requires_grad=True)
    soft_weights = selector.soft_select(embeddings, quality_scores)
    assert soft_weights.shape == (10,)
    loss = soft_weights.sum()
    loss.backward()
    assert quality_scores.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dpp_selector.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement DPP selector**

```python
# src/models/dpp_selector.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DPPSelector(nn.Module):
    """Select k subgraphs via a DPP with learned quality and embedding-based diversity."""

    def __init__(self, embed_dim: int, budget_k: int, temperature: float = 1.0):
        super().__init__()
        self.budget_k = budget_k
        self.temperature = temperature
        self.embed_dim = embed_dim

    def _build_L_kernel(self, embeddings: torch.Tensor, quality_scores: torch.Tensor) -> torch.Tensor:
        """Build the L-ensemble DPP kernel: L_ij = q_i * S_ij * q_j."""
        q = F.softplus(quality_scores)  # ensure positive
        embeddings_norm = F.normalize(embeddings, dim=-1)
        S = embeddings_norm @ embeddings_norm.T  # cosine similarity
        L = torch.outer(q, q) * S
        return L

    def forward(self, embeddings: torch.Tensor, quality_scores: torch.Tensor) -> list[int]:
        """Greedy MAP inference for DPP: select budget_k items."""
        n = embeddings.shape[0]
        k = min(self.budget_k, n)

        L = self._build_L_kernel(embeddings, quality_scores)

        # Greedy MAP DPP (Chen et al. 2018 — fast greedy)
        selected = []
        remaining = list(range(n))

        for _ in range(k):
            if not remaining:
                break
            if not selected:
                scores = torch.diag(L)[remaining]
            else:
                sel_tensor = torch.tensor(selected, dtype=torch.long)
                rem_tensor = torch.tensor(remaining, dtype=torch.long)
                L_sel = L[sel_tensor][:, sel_tensor]
                L_sel_inv = torch.linalg.inv(L_sel + 1e-6 * torch.eye(len(selected)))
                L_cross = L[rem_tensor][:, sel_tensor]
                diag_L = L[rem_tensor, rem_tensor]
                scores = diag_L - (L_cross @ L_sel_inv @ L_cross.T).diag()

            best_local = scores.argmax().item()
            best_global = remaining[best_local]
            selected.append(best_global)
            remaining.remove(best_global)

        return selected

    def soft_select(self, embeddings: torch.Tensor, quality_scores: torch.Tensor) -> torch.Tensor:
        """Differentiable relaxation: return soft attention weights over all candidates."""
        L = self._build_L_kernel(embeddings, quality_scores)
        # Marginal inclusion probabilities: K = L(L + I)^{-1}
        n = L.shape[0]
        K = L @ torch.linalg.inv(L + torch.eye(n, device=L.device))
        marginals = torch.diag(K)  # P(item i in DPP sample)
        # Temperature-scaled for sharper selection
        return torch.sigmoid((marginals - 0.5) / self.temperature)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dpp_selector.py -v`
Expected: all 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/dpp_selector.py tests/test_dpp_selector.py
git commit -m "feat: DPP selector with greedy MAP and differentiable relaxation"
```

---

### Task 5: Margin Scorer

**Files:**
- Create: `src/models/margin_scorer.py`
- Create: `src/training/margin_utils.py`
- Create: `tests/test_margin_scorer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_margin_scorer.py
import torch
from src.training.margin_utils import compute_margin
from src.models.margin_scorer import MarginScorer


def test_compute_margin_correct_class():
    logits = torch.tensor([[3.0, 1.0, 0.5]])  # class 0 predicted
    labels = torch.tensor([0])
    margin = compute_margin(logits, labels)
    # margin = logit[correct] - max(logit[incorrect]) = 3.0 - 1.0 = 2.0
    assert torch.allclose(margin, torch.tensor([2.0]))


def test_compute_margin_negative():
    logits = torch.tensor([[1.0, 3.0, 0.5]])  # class 1 predicted
    labels = torch.tensor([0])  # but true label is 0
    margin = compute_margin(logits, labels)
    # margin = 1.0 - 3.0 = -2.0
    assert torch.allclose(margin, torch.tensor([-2.0]))


def test_margin_scorer_output_shape():
    scorer = MarginScorer(embed_dim=16)
    subgraph_embeddings = torch.randn(10, 16)
    graph_embedding = torch.randn(16)
    scores = scorer(subgraph_embeddings, graph_embedding)
    assert scores.shape == (10,)


def test_margin_scorer_gradients():
    scorer = MarginScorer(embed_dim=16)
    subgraph_embeddings = torch.randn(10, 16, requires_grad=True)
    graph_embedding = torch.randn(16)
    scores = scorer(subgraph_embeddings, graph_embedding)
    scores.sum().backward()
    assert subgraph_embeddings.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_margin_scorer.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement margin utilities**

```python
# src/training/margin_utils.py
import torch


def compute_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute classification margin: logit[true] - max(logit[other])."""
    batch_size = logits.shape[0]
    correct_logits = logits[torch.arange(batch_size), labels]

    # Mask out the correct class to find max of others
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[torch.arange(batch_size), labels] = False
    other_logits = logits.masked_fill(~mask, float("-inf"))
    max_other = other_logits.max(dim=1).values

    return correct_logits - max_other
```

- [ ] **Step 4: Implement margin scorer**

```python
# src/models/margin_scorer.py
import torch
import torch.nn as nn


class MarginScorer(nn.Module):
    """Predict per-subgraph margin contribution given subgraph and graph embeddings."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, subgraph_embeddings: torch.Tensor, graph_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            subgraph_embeddings: (n_subgraphs, embed_dim)
            graph_embedding: (embed_dim,) — full graph context
        Returns:
            scores: (n_subgraphs,) — predicted margin contribution per subgraph
        """
        n = subgraph_embeddings.shape[0]
        context = graph_embedding.unsqueeze(0).expand(n, -1)
        combined = torch.cat([subgraph_embeddings, context], dim=-1)
        scores = self.score_net(combined).squeeze(-1)
        return scores
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_margin_scorer.py -v`
Expected: all 4 PASS

- [ ] **Step 6: Commit**

```bash
git add src/training/margin_utils.py src/models/margin_scorer.py tests/test_margin_scorer.py
git commit -m "feat: margin computation and per-subgraph margin scorer"
```

---

### Task 6: Bag Aggregator

**Files:**
- Create: `src/models/bag_aggregator.py`
- Create: `tests/test_bag_aggregator.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_bag_aggregator.py
import torch
from src.models.bag_aggregator import BagAggregator


def test_bag_aggregator_sum():
    agg = BagAggregator(embed_dim=16, method="sum")
    embeddings = torch.randn(5, 16)  # 5 selected subgraphs
    weights = torch.ones(5)
    out = agg(embeddings, weights)
    assert out.shape == (16,)


def test_bag_aggregator_weighted():
    agg = BagAggregator(embed_dim=16, method="weighted_sum")
    embeddings = torch.randn(5, 16)
    weights = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    out = agg(embeddings, weights)
    assert torch.allclose(out, embeddings[0], atol=1e-6)


def test_bag_aggregator_attention():
    agg = BagAggregator(embed_dim=16, method="attention")
    embeddings = torch.randn(5, 16)
    weights = torch.ones(5)
    out = agg(embeddings, weights)
    assert out.shape == (16,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_bag_aggregator.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement bag aggregator**

```python
# src/models/bag_aggregator.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class BagAggregator(nn.Module):
    """Aggregate a bag of subgraph embeddings into a single graph representation."""

    def __init__(self, embed_dim: int, method: str = "weighted_sum"):
        super().__init__()
        self.method = method
        self.embed_dim = embed_dim
        if method == "attention":
            self.attn = nn.Linear(embed_dim, 1)

    def forward(self, embeddings: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (n_selected, embed_dim)
            weights: (n_selected,) — DPP soft weights or uniform
        Returns:
            (embed_dim,) — aggregated graph representation
        """
        if self.method == "sum":
            return embeddings.sum(dim=0)
        elif self.method == "weighted_sum":
            w = weights.unsqueeze(-1)  # (n, 1)
            return (embeddings * w).sum(dim=0)
        elif self.method == "attention":
            attn_logits = self.attn(embeddings).squeeze(-1)  # (n,)
            attn_weights = F.softmax(attn_logits * weights, dim=0)
            return (embeddings * attn_weights.unsqueeze(-1)).sum(dim=0)
        else:
            raise ValueError(f"Unknown method: {self.method}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_bag_aggregator.py -v`
Expected: all 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/bag_aggregator.py tests/test_bag_aggregator.py
git commit -m "feat: bag aggregator (sum, weighted, attention)"
```

---

### Task 7: Full Model — DPPSubgraphGNN

**Files:**
- Create: `src/models/dpp_subgraph_gnn.py`
- Create: `tests/test_full_model.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_full_model.py
import torch
from torch_geometric.data import Data
from src.models.dpp_subgraph_gnn import DPPSubgraphGNN


def make_graph():
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long)
    x = torch.randn(4, 7)
    return Data(x=x, edge_index=edge_index, num_nodes=4)


def test_forward_shape():
    model = DPPSubgraphGNN(
        in_dim=7, hidden_dim=32, out_dim=3,
        num_layers=3, budget_k=3, policy="node_deletion"
    )
    g = make_graph()
    logits = model(g)
    assert logits.shape == (3,)  # 3 classes


def test_forward_with_labels_returns_margin():
    model = DPPSubgraphGNN(
        in_dim=7, hidden_dim=32, out_dim=3,
        num_layers=3, budget_k=3, policy="node_deletion"
    )
    g = make_graph()
    g.y = torch.tensor([1])
    logits, margin_info = model(g, return_margin_info=True)
    assert "margin" in margin_info
    assert "selected_indices" in margin_info
    assert len(margin_info["selected_indices"]) <= 3


def test_backward():
    model = DPPSubgraphGNN(
        in_dim=7, hidden_dim=32, out_dim=3,
        num_layers=3, budget_k=3, policy="node_deletion"
    )
    g = make_graph()
    logits = model(g)
    loss = logits.sum()
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_full_model.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement full model**

```python
# src/models/dpp_subgraph_gnn.py
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from src.data.subgraph_policies import node_deletion_subgraphs, edge_deletion_subgraphs, ego_subgraphs
from src.models.base_gnn import GINEncoder
from src.models.dpp_selector import DPPSelector
from src.models.margin_scorer import MarginScorer
from src.models.bag_aggregator import BagAggregator
from src.training.margin_utils import compute_margin


POLICIES = {
    "node_deletion": node_deletion_subgraphs,
    "edge_deletion": edge_deletion_subgraphs,
    "ego": lambda g: ego_subgraphs(g, hops=1),
}


class DPPSubgraphGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 4,
        budget_k: int = 10,
        policy: str = "node_deletion",
        aggregation: str = "weighted_sum",
    ):
        super().__init__()
        self.policy_fn = POLICIES[policy]
        self.budget_k = budget_k

        self.encoder = GINEncoder(in_dim, hidden_dim, hidden_dim, num_layers)
        self.margin_scorer = MarginScorer(hidden_dim)
        self.selector = DPPSelector(hidden_dim, budget_k)
        self.aggregator = BagAggregator(hidden_dim, method=aggregation)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, data: Data, return_margin_info: bool = False):
        # Generate candidate subgraphs
        subgraphs = self.policy_fn(data)
        if not subgraphs:
            # Fallback: use original graph
            graph_emb = self.encoder(data)
            logits = self.classifier(graph_emb)
            if return_margin_info:
                return logits, {"margin": None, "selected_indices": []}
            return logits

        # Encode all candidates
        batch = Batch.from_data_list(subgraphs)
        subgraph_embeddings = self.encoder(batch)  # (n_candidates, hidden_dim)

        # Compute graph-level context (mean of all subgraph embeddings)
        graph_context = subgraph_embeddings.mean(dim=0)

        # Score each subgraph's predicted margin contribution
        quality_scores = self.margin_scorer(subgraph_embeddings, graph_context)

        # Select via DPP (hard selection for inference, soft for training)
        if self.training:
            soft_weights = self.selector.soft_select(subgraph_embeddings, quality_scores)
            graph_repr = self.aggregator(subgraph_embeddings, soft_weights)
            selected_indices = self.selector(subgraph_embeddings.detach(), quality_scores.detach())
        else:
            selected_indices = self.selector(subgraph_embeddings, quality_scores)
            selected_emb = subgraph_embeddings[selected_indices]
            weights = torch.ones(len(selected_indices))
            graph_repr = self.aggregator(selected_emb, weights)

        logits = self.classifier(graph_repr)

        if return_margin_info:
            info = {
                "selected_indices": selected_indices,
                "quality_scores": quality_scores,
                "subgraph_embeddings": subgraph_embeddings,
            }
            if hasattr(data, "y") and data.y is not None:
                info["margin"] = compute_margin(logits.unsqueeze(0), data.y).item()
            else:
                info["margin"] = None
            return logits, info

        return logits
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_full_model.py -v`
Expected: all 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/dpp_subgraph_gnn.py tests/test_full_model.py
git commit -m "feat: DPPSubgraphGNN end-to-end model"
```

---

### Task 8: Training Loop with Margin-Aware DPP Loss

**Files:**
- Create: `src/training/losses.py`
- Create: `src/training/trainer.py`
- Create: `tests/test_trainer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_trainer.py
import torch
from src.training.losses import dpp_margin_loss, classification_loss


def test_classification_loss():
    logits = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 1])
    loss = classification_loss(logits, labels)
    assert loss.shape == ()
    assert loss.item() > 0


def test_dpp_margin_loss_rewards_positive_margin():
    # High quality scores + positive margins = low loss
    quality_scores = torch.tensor([1.0, 1.0, 1.0])
    actual_margins = torch.tensor([2.0, 2.0, 2.0])  # correct predictions
    loss_good = dpp_margin_loss(quality_scores, actual_margins)

    # High quality scores + negative margins = high loss
    actual_margins_bad = torch.tensor([-1.0, -1.0, -1.0])
    loss_bad = dpp_margin_loss(quality_scores, actual_margins_bad)

    assert loss_bad > loss_good
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_trainer.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement losses**

```python
# src/training/losses.py
import torch
import torch.nn.functional as F


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def dpp_margin_loss(quality_scores: torch.Tensor, actual_margins: torch.Tensor) -> torch.Tensor:
    """Encourage quality scorer to predict margin contribution accurately.

    The quality scorer should assign high scores to subgraphs that
    contribute to positive margins (correct, confident predictions)
    and low scores to subgraphs associated with negative margins.

    Loss = MSE between predicted quality and actual margin signal.
    """
    target = torch.sigmoid(actual_margins)  # normalize to [0, 1]
    predicted = torch.sigmoid(quality_scores)
    return F.mse_loss(predicted, target)
```

- [ ] **Step 4: Implement trainer**

```python
# src/training/trainer.py
import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from src.models.dpp_subgraph_gnn import DPPSubgraphGNN
from src.training.losses import classification_loss, dpp_margin_loss
from src.training.margin_utils import compute_margin


class Trainer:
    def __init__(self, model: DPPSubgraphGNN, lr: float = 1e-3, margin_weight: float = 0.1,
                 warmup_epochs: int = 10):
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.margin_weight = margin_weight
        self.warmup_epochs = warmup_epochs
        self.epoch = 0

    def train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total_cls_loss = 0.0
        total_margin_loss = 0.0
        n_samples = 0

        for data in loader:
            self.optimizer.zero_grad()

            logits, info = self.model(data, return_margin_info=True)
            labels = data.y

            # Classification loss
            cls_loss = classification_loss(logits.unsqueeze(0), labels)

            # Margin-aware DPP loss (after warmup)
            if self.epoch >= self.warmup_epochs and info.get("quality_scores") is not None:
                margin = compute_margin(logits.unsqueeze(0), labels)
                # Broadcast margin to all subgraphs (they all contributed)
                n_subs = info["quality_scores"].shape[0]
                margin_expanded = margin.expand(n_subs)
                m_loss = dpp_margin_loss(info["quality_scores"], margin_expanded)
                loss = cls_loss + self.margin_weight * m_loss
                total_margin_loss += m_loss.item()
            else:
                loss = cls_loss

            loss.backward()
            self.optimizer.step()

            total_cls_loss += cls_loss.item()
            n_samples += 1

        self.epoch += 1
        return {
            "cls_loss": total_cls_loss / max(n_samples, 1),
            "margin_loss": total_margin_loss / max(n_samples, 1),
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        correct = 0
        total = 0
        total_margin = 0.0

        for data in loader:
            logits = self.model(data)
            pred = logits.argmax(dim=-1) if logits.dim() > 0 else logits.unsqueeze(0).argmax(dim=-1)
            labels = data.y
            correct += (pred == labels).sum().item()
            total += labels.shape[0]
            margin = compute_margin(logits.unsqueeze(0), labels)
            total_margin += margin.sum().item()

        return {
            "accuracy": correct / max(total, 1),
            "avg_margin": total_margin / max(total, 1),
        }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_trainer.py -v`
Expected: all 2 PASS

- [ ] **Step 6: Commit**

```bash
git add src/training/losses.py src/training/trainer.py tests/test_trainer.py
git commit -m "feat: training loop with classification + DPP margin loss"
```

---

### Task 9: ESAN Baselines (Uniform + Full)

**Files:**
- Create: `src/baselines/esan_uniform.py`
- Create: `src/baselines/esan_full.py`
- Create: `tests/test_baselines.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_baselines.py
import torch
from torch_geometric.data import Data
from src.baselines.esan_uniform import ESANUniform
from src.baselines.esan_full import ESANFull


def make_graph():
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long)
    x = torch.randn(4, 7)
    return Data(x=x, edge_index=edge_index, num_nodes=4)


def test_esan_uniform_output():
    model = ESANUniform(in_dim=7, hidden_dim=32, out_dim=3, num_layers=3, budget_k=2)
    g = make_graph()
    logits = model(g)
    assert logits.shape == (3,)


def test_esan_full_output():
    model = ESANFull(in_dim=7, hidden_dim=32, out_dim=3, num_layers=3)
    g = make_graph()
    logits = model(g)
    assert logits.shape == (3,)


def test_esan_full_uses_all_subgraphs():
    model = ESANFull(in_dim=7, hidden_dim=32, out_dim=3, num_layers=3)
    g = make_graph()
    logits, info = model(g, return_info=True)
    assert info["n_subgraphs"] == 4  # node-deletion on 4-node graph
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_baselines.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement baselines**

```python
# src/baselines/esan_uniform.py
import torch
import torch.nn as nn
import random
from torch_geometric.data import Batch

from src.data.subgraph_policies import node_deletion_subgraphs
from src.models.base_gnn import GINEncoder


class ESANUniform(nn.Module):
    """ESAN baseline with uniform random subgraph sampling."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 4, budget_k: int = 10):
        super().__init__()
        self.budget_k = budget_k
        self.encoder = GINEncoder(in_dim, hidden_dim, hidden_dim, num_layers)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        subgraphs = node_deletion_subgraphs(data)
        if not subgraphs:
            emb = self.encoder(data)
            return self.classifier(emb)

        k = min(self.budget_k, len(subgraphs))
        selected = random.sample(subgraphs, k)
        batch = Batch.from_data_list(selected)
        embeddings = self.encoder(batch)
        graph_repr = embeddings.mean(dim=0)
        return self.classifier(graph_repr)
```

```python
# src/baselines/esan_full.py
import torch
import torch.nn as nn
from torch_geometric.data import Batch

from src.data.subgraph_policies import node_deletion_subgraphs
from src.models.base_gnn import GINEncoder


class ESANFull(nn.Module):
    """ESAN baseline using ALL subgraphs (no selection)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 4):
        super().__init__()
        self.encoder = GINEncoder(in_dim, hidden_dim, hidden_dim, num_layers)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, data, return_info: bool = False):
        subgraphs = node_deletion_subgraphs(data)
        if not subgraphs:
            emb = self.encoder(data)
            logits = self.classifier(emb)
            if return_info:
                return logits, {"n_subgraphs": 0}
            return logits

        batch = Batch.from_data_list(subgraphs)
        embeddings = self.encoder(batch)
        graph_repr = embeddings.mean(dim=0)
        logits = self.classifier(graph_repr)

        if return_info:
            return logits, {"n_subgraphs": len(subgraphs)}
        return logits
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_baselines.py -v`
Expected: all 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/baselines/esan_uniform.py src/baselines/esan_full.py tests/test_baselines.py
git commit -m "feat: ESAN baselines (uniform sampling + full bag)"
```

---

### Task 10: ZINC Experiment Script

**Files:**
- Create: `src/experiments/run_zinc.py`
- Create: `src/data/datasets.py`

- [ ] **Step 1: Implement dataset loader**

```python
# src/data/datasets.py
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader


def get_zinc_loaders(batch_size: int = 128, root: str = "data/"):
    train_dataset = ZINC(root=root, subset=True, split="train")
    val_dataset = ZINC(root=root, subset=True, split="val")
    test_dataset = ZINC(root=root, subset=True, split="test")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader
```

- [ ] **Step 2: Implement experiment script**

```python
# src/experiments/run_zinc.py
"""ZINC-12K experiment: compare DPP selection vs baselines."""
import argparse
import torch
import yaml
from pathlib import Path

from src.data.datasets import get_zinc_loaders
from src.models.dpp_subgraph_gnn import DPPSubgraphGNN
from src.baselines.esan_uniform import ESANUniform
from src.baselines.esan_full import ESANFull
from src.training.trainer import Trainer


def run_experiment(config_path: str = "configs/zinc.yaml", model_type: str = "dpp"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    train_loader, val_loader, test_loader = get_zinc_loaders()

    # ZINC node features are atom types (1-dimensional, integer)
    in_dim = 1
    hidden_dim = cfg["model"]["hidden_dim"]
    out_dim = 1  # ZINC is regression
    num_layers = cfg["model"]["num_layers"]
    budget_k = cfg["selection"]["budget_k"]

    if model_type == "dpp":
        model = DPPSubgraphGNN(
            in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
            num_layers=num_layers, budget_k=budget_k,
            policy=cfg["selection"]["policy"],
        )
    elif model_type == "uniform":
        model = ESANUniform(
            in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
            num_layers=num_layers, budget_k=budget_k,
        )
    elif model_type == "full":
        model = ESANFull(
            in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
            num_layers=num_layers,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    trainer = Trainer(
        model, lr=cfg["training"]["lr"],
        warmup_epochs=cfg["training"].get("margin_warmup_epochs", 10),
    )

    best_val = float("inf")
    for epoch in range(cfg["training"]["epochs"]):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)

        if val_metrics.get("avg_margin", 0) < best_val:
            best_val = val_metrics.get("avg_margin", 0)
            torch.save(model.state_dict(), f"checkpoints/{model_type}_best.pt")

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: cls_loss={train_metrics['cls_loss']:.4f} "
                  f"margin_loss={train_metrics['margin_loss']:.4f} "
                  f"val_acc={val_metrics['accuracy']:.4f}")

    test_metrics = trainer.evaluate(test_loader)
    print(f"\nTest: {test_metrics}")
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["dpp", "uniform", "full"], default="dpp")
    parser.add_argument("--config", default="configs/zinc.yaml")
    args = parser.parse_args()

    Path("checkpoints").mkdir(exist_ok=True)
    run_experiment(args.config, args.model)
```

- [ ] **Step 3: Commit**

```bash
git add src/data/datasets.py src/experiments/run_zinc.py
git commit -m "feat: ZINC experiment script with all model variants"
```

---

### Task 11: Centrality Baseline (HyMN-style)

**Files:**
- Create: `src/baselines/centrality_select.py`

- [ ] **Step 1: Implement centrality-based selection**

```python
# src/baselines/centrality_select.py
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.utils import degree

from src.data.subgraph_policies import node_deletion_subgraphs
from src.models.base_gnn import GINEncoder


class CentralitySelect(nn.Module):
    """HyMN-style baseline: select subgraphs by node centrality ranking."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 4, budget_k: int = 10):
        super().__init__()
        self.budget_k = budget_k
        self.encoder = GINEncoder(in_dim, hidden_dim, hidden_dim, num_layers)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def _centrality_scores(self, data) -> list[float]:
        """Compute degree centrality for each node (used to rank subgraphs)."""
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
        return deg.tolist()

    def forward(self, data):
        subgraphs = node_deletion_subgraphs(data)
        if not subgraphs:
            emb = self.encoder(data)
            return self.classifier(emb)

        # Rank by centrality of the DELETED node (delete high-centrality = informative subgraph)
        centrality = self._centrality_scores(data)
        # Sort by centrality descending — deleting high-centrality nodes creates most different subgraph
        ranked_indices = sorted(range(len(centrality)), key=lambda i: centrality[i], reverse=True)

        k = min(self.budget_k, len(subgraphs))
        selected = [subgraphs[i] for i in ranked_indices[:k]]

        batch = Batch.from_data_list(selected)
        embeddings = self.encoder(batch)
        graph_repr = embeddings.mean(dim=0)
        return self.classifier(graph_repr)
```

- [ ] **Step 2: Commit**

```bash
git add src/baselines/centrality_select.py
git commit -m "feat: centrality-based subgraph selection baseline (HyMN-style)"
```

---

### Task 12: Analysis & Ablation Scripts

**Files:**
- Create: `src/experiments/ablations.py`
- Create: `src/experiments/analyze_selection.py`

- [ ] **Step 1: Implement ablation runner**

```python
# src/experiments/ablations.py
"""Run ablations: DPP vs uniform vs centrality vs full at same budget k."""
import torch
import json
from pathlib import Path

from src.data.datasets import get_zinc_loaders
from src.models.dpp_subgraph_gnn import DPPSubgraphGNN
from src.baselines.esan_uniform import ESANUniform
from src.baselines.esan_full import ESANFull
from src.baselines.centrality_select import CentralitySelect
from src.training.trainer import Trainer


def run_ablation(budget_k: int = 5, epochs: int = 100, seed: int = 0):
    torch.manual_seed(seed)
    train_loader, val_loader, test_loader = get_zinc_loaders()

    models = {
        "dpp": DPPSubgraphGNN(in_dim=1, hidden_dim=64, out_dim=1, num_layers=4, budget_k=budget_k),
        "uniform": ESANUniform(in_dim=1, hidden_dim=64, out_dim=1, num_layers=4, budget_k=budget_k),
        "centrality": CentralitySelect(in_dim=1, hidden_dim=64, out_dim=1, num_layers=4, budget_k=budget_k),
        "full": ESANFull(in_dim=1, hidden_dim=64, out_dim=1, num_layers=4),
    }

    results = {}
    for name, model in models.items():
        print(f"\n=== Training {name} (k={budget_k}, seed={seed}) ===")
        trainer = Trainer(model, lr=1e-3)
        for epoch in range(epochs):
            trainer.train_epoch(train_loader)
        test_metrics = trainer.evaluate(test_loader)
        results[name] = test_metrics
        print(f"{name}: {test_metrics}")

    Path("results").mkdir(exist_ok=True)
    with open(f"results/ablation_k{budget_k}_seed{seed}.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_ablation(args.budget_k, args.epochs, args.seed)
```

- [ ] **Step 2: Implement selection analysis**

```python
# src/experiments/analyze_selection.py
"""Analyze WHAT the DPP selects vs other methods — the key qualitative result."""
import torch
from torch_geometric.data import Data

from src.models.dpp_subgraph_gnn import DPPSubgraphGNN
from src.baselines.centrality_select import CentralitySelect
from src.data.subgraph_policies import node_deletion_subgraphs


def compare_selections(data: Data, dpp_model: DPPSubgraphGNN, budget_k: int = 5):
    """Show which subgraphs DPP picks vs centrality on a single graph."""
    dpp_model.eval()
    with torch.no_grad():
        _, info = dpp_model(data, return_margin_info=True)

    dpp_selected = info["selected_indices"]
    quality_scores = info["quality_scores"]

    # Centrality baseline selection
    from torch_geometric.utils import degree
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
    centrality_selected = deg.argsort(descending=True)[:budget_k].tolist()

    return {
        "dpp_selected_nodes_deleted": dpp_selected,
        "centrality_selected_nodes_deleted": centrality_selected,
        "dpp_quality_scores": quality_scores.tolist(),
        "node_degrees": deg.tolist(),
        "overlap": len(set(dpp_selected) & set(centrality_selected)),
    }
```

- [ ] **Step 3: Commit**

```bash
git add src/experiments/ablations.py src/experiments/analyze_selection.py
git commit -m "feat: ablation and selection analysis scripts"
```

---

## Success Criteria

**Minimum viable result (LoG workshop):**
- DPP selection outperforms uniform selection by >0.5% accuracy or MAE on ZINC at budget k=5
- Show that DPP selects DIFFERENT subgraphs than centrality on hard examples (low-margin graphs)
- Show DPP diversity > uniform diversity (measure pairwise cosine similarity of selected embeddings)

**Strong result (LoG main / NeurIPS workshop):**
- Improvement holds across ZINC + OGB-molhiv + 2 TU datasets
- Ablation: remove diversity kernel → performance drops (proves diversity is load-bearing, not just quality)
- Ablation: remove margin scorer → performance drops (proves margin signal is load-bearing, not just DPP)
- Show examples where centrality-selected subgraphs HURT margin (Franks-Morris regime)

**Home run (NeurIPS main):**
- All of the above + theoretical guarantee: "under conditions X, DPP selection preserves margin while Policy-Learn selection may not"
- Runtime analysis showing DPP selection adds <20% overhead vs uniform at same k

---

## Timeline

| Week | Focus |
|------|-------|
| 1 | Tasks 1-3: skeleton, policies, encoder |
| 2 | Tasks 4-6: DPP selector, margin scorer, aggregator |
| 3 | Tasks 7-8: full model, training loop |
| 4 | Task 9: baselines |
| 5 | Tasks 10-11: ZINC experiments + centrality baseline |
| 6 | Task 12: ablations + analysis |
| 7-8 | TU/OGB experiments, hyperparameter tuning |
| 9-10 | Paper writing, figures, additional ablations |
