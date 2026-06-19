# CFG Composition for Discrete Flows — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Classifier-Free Guidance (CFG) composition for discrete flow matching — AND/NOT/OR over multiple CFG-trained conditions — and validate on structured synthetic data where ground-truth Boolean accuracy is computable.

**Architecture:** A single discrete flow model trained with random condition dropout (standard CFG recipe). Each property label is a separate conditioning signal. At inference, compose multiple CFG posteriors via AND (product), NOT (subtraction), OR (inclusion-exclusion) in log-probability space. Validation on data from a known Markov chain where P(property) is analytically computable.

**Tech Stack:** PyTorch, Transformer encoder, no external dependencies beyond torch/numpy/scipy/yaml/pytest.

---

## Why This Works (vs. Failed Classifier Guidance)

The previous approach trained a *separate* classifier and an *unconditional* flow on *uniform random* data. Guidance failed because:
1. Uniform data has no structure → flow generates random → guidance has nothing to steer
2. Classifier guidance requires the base model to already produce good samples

CFG composition fixes both: the flow is trained *conditionally* on structured data, with label dropout. At inference, `score_cfg(x|c) = (1+w) * score(x|c) - w * score(x|∅)` amplifies the condition. Composing multiple conditions is the novel contribution.

---

## File Structure

```
src/
├── data/
│   ├── __init__.py
│   └── synthetic.py          # MODIFY: add MarkovSequenceDataset with properties
├── models/
│   ├── __init__.py
│   ├── conditional_flow.py   # CREATE: CFG-trained conditional discrete flow
│   ├── ctmc_flow.py          # KEEP (unchanged)
│   └── prob_path_flow.py     # KEEP (unchanged)
├── guidance/
│   ├── __init__.py
│   ├── cfg_compose.py        # CREATE: CFG AND/NOT/OR composition
│   ├── compose.py            # KEEP (classifier-guidance variant)
│   ├── ctmc_guidance.py      # KEEP
│   └── prob_path_guidance.py # KEEP
├── training/
│   ├── __init__.py
│   ├── train_conditional.py  # CREATE: CFG training loop with label dropout
│   ├── train_classifier.py   # KEEP
│   └── train_flow.py         # KEEP
└── experiments/
    ├── __init__.py
    ├── run_cfg.py             # CREATE: CFG composition experiment
    └── run_synthetic.py       # KEEP (old classifier-guidance experiment)
tests/
├── test_markov_data.py       # CREATE
├── test_conditional_flow.py  # CREATE
├── test_cfg_compose.py       # CREATE
├── test_cfg_integration.py   # CREATE
└── ... (existing tests kept)
configs/
├── cfg_synthetic.yaml        # CREATE: CFG experiment config
├── synthetic.yaml            # KEEP
└── smoke.yaml                # KEEP
```

---

### Task 1: Structured Synthetic Data (Markov Chain)

**Files:**
- Modify: `src/data/synthetic.py`
- Create: `tests/test_markov_data.py`

- [ ] **Step 1: Write failing test for Markov data**

```python
# tests/test_markov_data.py
import torch
from src.data.synthetic import (
    MarkovSequenceDataset,
    property_starts_with_0,
    property_contains_pattern,
    property_no_adjacent_repeats,
)


def test_markov_dataset_shape():
    ds = MarkovSequenceDataset(n=500, K=8, L=32)
    seq, labels = ds[0]
    assert seq.shape == (32,)
    assert isinstance(labels, dict)
    assert "starts_with_0" in labels
    assert "contains_pattern" in labels
    assert "no_repeats" in labels


def test_markov_dataset_structure():
    """Markov sequences should NOT be uniform — they have transition structure."""
    ds = MarkovSequenceDataset(n=5000, K=8, L=32)
    seqs = ds.seqs
    # Adjacent token correlations should be non-zero (unlike uniform random)
    same_as_prev = (seqs[:, 1:] == seqs[:, :-1]).float().mean().item()
    # With structured transitions, this should differ from 1/K = 0.125
    assert same_as_prev > 0.15 or same_as_prev < 0.10


def test_markov_property_rates():
    """Properties should have non-trivial rates (not too rare, not too common)."""
    ds = MarkovSequenceDataset(n=5000, K=8, L=32)
    for name, labels in ds.labels.items():
        rate = labels.float().mean().item()
        assert 0.05 < rate < 0.95, f"{name} rate {rate} is too extreme"


def test_markov_conditional_labels():
    """Dataset should provide multi-hot condition vectors."""
    ds = MarkovSequenceDataset(n=100, K=8, L=32)
    seq, labels = ds[0]
    # labels should be booleans
    assert all(isinstance(v, bool) for v in labels.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_markov_data.py -v`
Expected: FAIL with ImportError (MarkovSequenceDataset not defined)

- [ ] **Step 3: Implement MarkovSequenceDataset**

Add to `src/data/synthetic.py`:

```python
def generate_markov_sequences(n: int, K: int, L: int) -> torch.Tensor:
    """Generate sequences from a structured transition matrix.
    Creates a random stochastic matrix with self-transition bias,
    producing sequences with learnable inter-position patterns.
    """
    torch.manual_seed(42)
    # Transition matrix: prefer self-transition and neighbors
    T = torch.zeros(K, K)
    for i in range(K):
        T[i, i] = 3.0  # self-transition bias
        T[i, (i + 1) % K] = 2.0  # next-state bias
        T[i, (i - 1) % K] = 1.5  # prev-state bias
    T = T + 0.1  # small uniform background
    T = T / T.sum(dim=1, keepdim=True)  # normalize rows

    # Generate sequences
    seqs = torch.zeros(n, L, dtype=torch.long)
    # Start state: biased towards 0 and 1
    start_probs = torch.ones(K) / K
    start_probs[0] = 2.0
    start_probs[1] = 1.5
    start_probs = start_probs / start_probs.sum()
    seqs[:, 0] = torch.multinomial(start_probs.expand(n, -1), 1).squeeze(-1)

    for l in range(1, L):
        prev = seqs[:, l - 1]  # (n,)
        trans_probs = T[prev]  # (n, K)
        seqs[:, l] = torch.multinomial(trans_probs, 1).squeeze(-1)

    torch.manual_seed(torch.seed())  # reset seed
    return seqs


class MarkovSequenceDataset(Dataset):
    """Structured sequences from a Markov chain with computable properties."""

    def __init__(self, n: int = 10000, K: int = 8, L: int = 32):
        self.seqs = generate_markov_sequences(n, K, L)
        self.labels = {
            "starts_with_0": property_starts_with_0(self.seqs),
            "contains_pattern": property_contains_pattern(self.seqs),
            "no_repeats": property_no_adjacent_repeats(self.seqs),
        }
        self.n_properties = len(self.labels)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        props = {k: v[idx].item() for k, v in self.labels.items()}
        return seq, props

    def get_condition_vector(self, idx) -> torch.Tensor:
        """Return multi-hot vector of active properties for this sample."""
        return torch.tensor([self.labels[k][idx].float() for k in self.labels])
```

- [ ] **Step 4: Run tests**

Run: `source .venv/bin/activate && pytest tests/test_markov_data.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/synthetic.py tests/test_markov_data.py
git commit -m "feat: structured Markov sequence dataset for CFG training"
```

---

### Task 2: Conditional Discrete Flow with CFG Dropout

**Files:**
- Create: `src/models/conditional_flow.py`
- Create: `tests/test_conditional_flow.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_conditional_flow.py
import torch
from src.models.conditional_flow import ConditionalProbPathFlow


def test_conditional_flow_forward_with_condition():
    model = ConditionalProbPathFlow(K=8, L=32, n_conditions=3, hidden_dim=64, num_layers=2)
    x_t = torch.randint(0, 8, (4, 32))
    t = torch.tensor([0.5, 0.3, 0.8, 0.1])
    cond = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=torch.float)
    logits = model(x_t, t, cond)
    assert logits.shape == (4, 32, 8)


def test_conditional_flow_unconditional():
    """Null condition (all zeros) should work — this is the 'unconditional' mode."""
    model = ConditionalProbPathFlow(K=8, L=32, n_conditions=3, hidden_dim=64, num_layers=2)
    x_t = torch.randint(0, 8, (4, 32))
    t = torch.tensor([0.5, 0.3, 0.8, 0.1])
    null_cond = torch.zeros(4, 3)
    logits = model(x_t, t, null_cond)
    assert logits.shape == (4, 32, 8)


def test_conditional_flow_cfg_dropout():
    """With dropout_prob=1.0, condition should be zeroed (unconditional)."""
    model = ConditionalProbPathFlow(K=8, L=32, n_conditions=3, hidden_dim=64, num_layers=2)
    x_t = torch.randint(0, 8, (2, 32))
    t = torch.tensor([0.5, 0.5])
    cond = torch.ones(2, 3)
    model.train()
    # With dropout=1.0, all conditions should be dropped
    logits_drop = model(x_t, t, cond, cfg_dropout_prob=1.0)
    logits_null = model(x_t, t, torch.zeros_like(cond))
    assert torch.allclose(logits_drop, logits_null, atol=1e-5)


def test_conditional_flow_different_conditions_give_different_outputs():
    model = ConditionalProbPathFlow(K=8, L=32, n_conditions=3, hidden_dim=64, num_layers=2)
    model.eval()
    x_t = torch.randint(0, 8, (1, 32))
    t = torch.tensor([0.5])
    cond_a = torch.tensor([[1.0, 0.0, 0.0]])
    cond_b = torch.tensor([[0.0, 1.0, 0.0]])
    logits_a = model(x_t, t, cond_a)
    logits_b = model(x_t, t, cond_b)
    assert not torch.allclose(logits_a, logits_b, atol=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_conditional_flow.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement ConditionalProbPathFlow**

```python
# src/models/conditional_flow.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalProbPathFlow(nn.Module):
    """Conditional discrete flow with CFG dropout.

    Trained to predict p(x_0 | x_t, t, c) where c is a multi-hot condition vector.
    At training time, c is randomly zeroed with probability cfg_dropout_prob
    to learn the unconditional p(x_0 | x_t, t) jointly.
    """

    def __init__(self, K: int, L: int, n_conditions: int,
                 hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.K = K
        self.L = L
        self.n_conditions = n_conditions

        self.embed = nn.Embedding(K, hidden_dim)
        self.pos_embed = nn.Embedding(L, hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(n_conditions, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 4,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, K)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor,
                cfg_dropout_prob: float = 0.0) -> torch.Tensor:
        """
        Args:
            x_t: (B, L) noisy token indices
            t: (B,) time values in [0, 1]
            cond: (B, n_conditions) multi-hot condition vector
            cfg_dropout_prob: probability of dropping condition (training only)
        Returns:
            (B, L, K) logits for p(x_0 | x_t, t, c)
        """
        B, L = x_t.shape

        # CFG dropout: zero the condition vector with probability cfg_dropout_prob
        if cfg_dropout_prob > 0.0 and self.training:
            drop_mask = (torch.rand(B, 1, device=x_t.device) < cfg_dropout_prob).float()
            cond = cond * (1.0 - drop_mask)

        pos_ids = torch.arange(L, device=x_t.device).unsqueeze(0).expand(B, -1)
        h = self.embed(x_t) + self.pos_embed(pos_ids)  # (B, L, hidden)
        t_emb = self.time_embed(t.unsqueeze(-1))  # (B, hidden)
        c_emb = self.cond_embed(cond)  # (B, hidden)
        h = h + t_emb.unsqueeze(1) + c_emb.unsqueeze(1)
        h = self.transformer(h)
        return self.out(h)  # (B, L, K)
```

- [ ] **Step 4: Run tests**

Run: `source .venv/bin/activate && pytest tests/test_conditional_flow.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/conditional_flow.py tests/test_conditional_flow.py
git commit -m "feat: conditional discrete flow with CFG dropout"
```

---

### Task 3: CFG Training Loop

**Files:**
- Create: `src/training/train_conditional.py`

- [ ] **Step 1: Implement training loop**

```python
# src/training/train_conditional.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data.synthetic import MarkovSequenceDataset
from src.models.conditional_flow import ConditionalProbPathFlow
from src.models.prob_path_flow import sample_from_categorical


def train_conditional_flow(config: dict, device: str = "cpu") -> ConditionalProbPathFlow:
    K = config["data"]["K"]
    L = config["data"]["L"]
    n_conditions = len(config["properties"])
    cfg_dropout = config["training"]["cfg_dropout_prob"]

    model = ConditionalProbPathFlow(
        K=K, L=L, n_conditions=n_conditions,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
    ).to(device)

    ds = MarkovSequenceDataset(n=config["data"]["n_train"], K=K, L=L)
    loader = DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    property_names = list(ds.labels.keys())

    model.train()
    for epoch in range(config["training"]["epochs"]):
        total_loss = 0.0
        for seqs, props_batch in loader:
            seqs = seqs.to(device)
            B = seqs.shape[0]

            # Build condition vectors from property labels
            cond = torch.zeros(B, n_conditions, device=device)
            for i, name in enumerate(property_names):
                for j in range(B):
                    cond[j, i] = float(props_batch[name][j])

            t = torch.rand(B, device=device)
            x_t = sample_from_categorical(seqs, t, K)
            logits = model(x_t, t, cond, cfg_dropout_prob=cfg_dropout)
            loss = F.cross_entropy(logits.reshape(-1, K), seqs.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"  CFG flow epoch {epoch+1}: loss={total_loss / len(loader):.4f}")

    return model
```

- [ ] **Step 2: Verify imports**

Run: `source .venv/bin/activate && python -c "from src.training.train_conditional import train_conditional_flow; print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add src/training/train_conditional.py
git commit -m "feat: CFG training loop with condition dropout"
```

---

### Task 4: CFG Composition Operators

**Files:**
- Create: `src/guidance/cfg_compose.py`
- Create: `tests/test_cfg_compose.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_cfg_compose.py
import torch
from src.models.conditional_flow import ConditionalProbPathFlow
from src.guidance.cfg_compose import (
    cfg_single,
    cfg_and,
    cfg_not,
    cfg_or,
    cfg_sample,
)


def _make_model():
    return ConditionalProbPathFlow(K=4, L=8, n_conditions=3, hidden_dim=32, num_layers=1)


def test_cfg_single():
    model = _make_model()
    model.eval()
    x_t = torch.randint(0, 4, (2, 8))
    t = torch.tensor([0.5, 0.5])
    # Condition on property 0
    logits = cfg_single(model, x_t, t, condition_idx=0, n_conditions=3, w=1.0)
    assert logits.shape == (2, 8, 4)


def test_cfg_and():
    model = _make_model()
    model.eval()
    x_t = torch.randint(0, 4, (2, 8))
    t = torch.tensor([0.5, 0.5])
    logits = cfg_and(model, x_t, t, condition_idxs=[0, 1], n_conditions=3, ws=[1.0, 1.0])
    assert logits.shape == (2, 8, 4)


def test_cfg_not():
    model = _make_model()
    model.eval()
    x_t = torch.randint(0, 4, (2, 8))
    t = torch.tensor([0.5, 0.5])
    logits = cfg_not(model, x_t, t, keep_idx=0, avoid_idx=2, n_conditions=3, w_keep=1.0, w_avoid=1.0)
    assert logits.shape == (2, 8, 4)


def test_cfg_or():
    model = _make_model()
    model.eval()
    x_t = torch.randint(0, 4, (2, 8))
    t = torch.tensor([0.5, 0.5])
    logits = cfg_or(model, x_t, t, condition_idxs=[0, 1], n_conditions=3, ws=[1.0, 1.0])
    assert logits.shape == (2, 8, 4)


def test_cfg_sample():
    model = _make_model()
    model.eval()
    samples = cfg_sample(model, n=10, K=4, L=8, n_conditions=3,
                         condition_idxs=[0], ws=[2.0], num_steps=10)
    assert samples.shape == (10, 8)
    assert samples.min() >= 0 and samples.max() < 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_cfg_compose.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement CFG composition**

```python
# src/guidance/cfg_compose.py
import torch
import torch.nn.functional as F
from src.models.conditional_flow import ConditionalProbPathFlow
from src.models.prob_path_flow import sample_euler_step


def _get_logits(model: ConditionalProbPathFlow, x_t: torch.Tensor,
                t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """Get model logits for given condition."""
    with torch.no_grad():
        return model(x_t, t, cond, cfg_dropout_prob=0.0)


def _null_cond(B: int, n_conditions: int, device) -> torch.Tensor:
    return torch.zeros(B, n_conditions, device=device)


def _one_hot_cond(B: int, condition_idx: int, n_conditions: int, device) -> torch.Tensor:
    cond = torch.zeros(B, n_conditions, device=device)
    cond[:, condition_idx] = 1.0
    return cond


def cfg_single(model: ConditionalProbPathFlow, x_t: torch.Tensor,
               t: torch.Tensor, condition_idx: int, n_conditions: int,
               w: float = 1.0) -> torch.Tensor:
    """Single-condition CFG: logits_cfg = (1+w)*logits_c - w*logits_uncond"""
    B = x_t.shape[0]
    device = x_t.device
    cond = _one_hot_cond(B, condition_idx, n_conditions, device)
    null = _null_cond(B, n_conditions, device)

    logits_cond = _get_logits(model, x_t, t, cond)
    logits_uncond = _get_logits(model, x_t, t, null)

    return (1 + w) * logits_cond - w * logits_uncond


def cfg_and(model: ConditionalProbPathFlow, x_t: torch.Tensor,
            t: torch.Tensor, condition_idxs: list, n_conditions: int,
            ws: list) -> torch.Tensor:
    """AND composition: product of experts in probability space.
    log p(x|A∧B) ≈ log p(x|A) + log p(x|B) - log p(x|∅)
    In logit space: cfg_A + cfg_B - logits_uncond
    """
    B = x_t.shape[0]
    device = x_t.device
    null = _null_cond(B, n_conditions, device)
    logits_uncond = _get_logits(model, x_t, t, null)

    composed = logits_uncond.clone()
    for idx, w in zip(condition_idxs, ws):
        cond = _one_hot_cond(B, idx, n_conditions, device)
        logits_c = _get_logits(model, x_t, t, cond)
        # Add the guidance signal: w * (logits_c - logits_uncond)
        composed = composed + w * (logits_c - logits_uncond)

    return composed


def cfg_not(model: ConditionalProbPathFlow, x_t: torch.Tensor,
            t: torch.Tensor, keep_idx: int, avoid_idx: int,
            n_conditions: int, w_keep: float = 1.0,
            w_avoid: float = 1.0) -> torch.Tensor:
    """NOT composition: A ∧ ¬B.
    Boost condition A, suppress condition B.
    logits = logits_uncond + w_keep*(logits_A - logits_uncond) - w_avoid*(logits_B - logits_uncond)
    """
    B = x_t.shape[0]
    device = x_t.device
    null = _null_cond(B, n_conditions, device)
    logits_uncond = _get_logits(model, x_t, t, null)

    cond_keep = _one_hot_cond(B, keep_idx, n_conditions, device)
    cond_avoid = _one_hot_cond(B, avoid_idx, n_conditions, device)
    logits_keep = _get_logits(model, x_t, t, cond_keep)
    logits_avoid = _get_logits(model, x_t, t, cond_avoid)

    return (logits_uncond
            + w_keep * (logits_keep - logits_uncond)
            - w_avoid * (logits_avoid - logits_uncond))


def cfg_or(model: ConditionalProbPathFlow, x_t: torch.Tensor,
           t: torch.Tensor, condition_idxs: list, n_conditions: int,
           ws: list) -> torch.Tensor:
    """OR composition: A ∨ B.
    Take element-wise max of guided logits (approximation of union).
    More principled: log(exp(cfg_A) + exp(cfg_B) - exp(cfg_A+B_uncond))
    Simplified: max(cfg_A, cfg_B) per position per state.
    """
    B = x_t.shape[0]
    device = x_t.device
    null = _null_cond(B, n_conditions, device)
    logits_uncond = _get_logits(model, x_t, t, null)

    guided_logits = []
    for idx, w in zip(condition_idxs, ws):
        cond = _one_hot_cond(B, idx, n_conditions, device)
        logits_c = _get_logits(model, x_t, t, cond)
        cfg_logits = logits_uncond + w * (logits_c - logits_uncond)
        guided_logits.append(cfg_logits)

    # Element-wise max over conditions (noisy-OR in logit space)
    stacked = torch.stack(guided_logits, dim=0)  # (N_conds, B, L, K)
    return stacked.max(dim=0).values


@torch.no_grad()
def cfg_sample(model: ConditionalProbPathFlow, n: int, K: int, L: int,
               n_conditions: int, condition_idxs: list, ws: list,
               num_steps: int = 50, mode: str = "and") -> torch.Tensor:
    """Generate samples with CFG composition."""
    device = next(model.parameters()).device
    x_t = torch.randint(0, K, (n, L), device=device)
    dt = 1.0 / num_steps

    model.eval()
    for step in range(num_steps):
        t_val = 1.0 - (step + 1) * dt
        t = torch.full((n,), t_val, device=device)

        if mode == "and":
            logits = cfg_and(model, x_t, t, condition_idxs, n_conditions, ws)
        elif mode == "or":
            logits = cfg_or(model, x_t, t, condition_idxs, n_conditions, ws)
        elif mode == "single":
            logits = cfg_single(model, x_t, t, condition_idxs[0], n_conditions, ws[0])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        posterior = F.softmax(logits, dim=-1)
        x_t = sample_euler_step(x_t, posterior, dt, K)

    return x_t


@torch.no_grad()
def cfg_sample_not(model: ConditionalProbPathFlow, n: int, K: int, L: int,
                   n_conditions: int, keep_idx: int, avoid_idx: int,
                   w_keep: float, w_avoid: float,
                   num_steps: int = 50) -> torch.Tensor:
    """Generate samples with NOT composition."""
    device = next(model.parameters()).device
    x_t = torch.randint(0, K, (n, L), device=device)
    dt = 1.0 / num_steps

    model.eval()
    for step in range(num_steps):
        t_val = 1.0 - (step + 1) * dt
        t = torch.full((n,), t_val, device=device)
        logits = cfg_not(model, x_t, t, keep_idx, avoid_idx, n_conditions, w_keep, w_avoid)
        posterior = F.softmax(logits, dim=-1)
        x_t = sample_euler_step(x_t, posterior, dt, K)

    return x_t
```

- [ ] **Step 4: Run tests**

Run: `source .venv/bin/activate && pytest tests/test_cfg_compose.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/guidance/cfg_compose.py tests/test_cfg_compose.py
git commit -m "feat: CFG AND/NOT/OR composition operators with sampling"
```

---

### Task 5: End-to-End CFG Experiment

**Files:**
- Create: `src/experiments/run_cfg.py`
- Create: `configs/cfg_synthetic.yaml`
- Create: `tests/test_cfg_integration.py`

- [ ] **Step 1: Create config**

```yaml
# configs/cfg_synthetic.yaml
data:
  K: 8
  L: 32
  n_train: 10000
  n_val: 2000

model:
  hidden_dim: 128
  num_layers: 3

training:
  epochs: 200
  lr: 0.001
  batch_size: 128
  cfg_dropout_prob: 0.15

guidance:
  w: 3.0
  num_steps: 50
  num_samples: 500

properties:
  - name: starts_with_0
  - name: contains_pattern
  - name: no_repeats
```

- [ ] **Step 2: Write integration test**

```python
# tests/test_cfg_integration.py
import torch
from src.data.synthetic import MarkovSequenceDataset, PROPERTIES
from src.models.conditional_flow import ConditionalProbPathFlow
from src.guidance.cfg_compose import cfg_sample, cfg_sample_not


def test_cfg_pipeline_runs():
    """Full pipeline: model forward + CFG sampling works without error."""
    model = ConditionalProbPathFlow(K=4, L=8, n_conditions=3, hidden_dim=32, num_layers=1)
    samples = cfg_sample(model, n=5, K=4, L=8, n_conditions=3,
                         condition_idxs=[0, 1], ws=[2.0, 2.0], num_steps=5, mode="and")
    assert samples.shape == (5, 8)


def test_cfg_not_pipeline_runs():
    model = ConditionalProbPathFlow(K=4, L=8, n_conditions=3, hidden_dim=32, num_layers=1)
    samples = cfg_sample_not(model, n=5, K=4, L=8, n_conditions=3,
                             keep_idx=0, avoid_idx=2, w_keep=2.0, w_avoid=2.0, num_steps=5)
    assert samples.shape == (5, 8)


def test_markov_data_has_structure():
    """Sanity: Markov data is NOT uniform random."""
    ds = MarkovSequenceDataset(n=2000, K=8, L=32)
    seqs = ds.seqs
    # Position 0 should be biased towards 0 (by construction)
    starts_0_rate = (seqs[:, 0] == 0).float().mean().item()
    assert starts_0_rate > 0.15  # should be > 1/8 = 0.125 due to start bias
```

- [ ] **Step 3: Implement experiment runner**

```python
# src/experiments/run_cfg.py
import yaml
import torch
from src.data.synthetic import MarkovSequenceDataset, PROPERTIES
from src.training.train_conditional import train_conditional_flow
from src.guidance.cfg_compose import cfg_sample, cfg_sample_not


def evaluate_boolean_accuracy(samples: torch.Tensor, property_fns: list, mode: str = "and") -> float:
    results = [fn(samples) for fn in property_fns]
    if mode == "and":
        combined = results[0]
        for r in results[1:]:
            combined = combined & r
    elif mode == "or":
        combined = results[0]
        for r in results[1:]:
            combined = combined | r
    elif mode == "not":
        combined = results[0] & (~results[1])
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return combined.float().mean().item()


def run_cfg_experiment(config_path: str = "configs/cfg_synthetic.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    K = config["data"]["K"]
    L = config["data"]["L"]
    w = config["guidance"]["w"]
    num_steps = config["guidance"]["num_steps"]
    n_samples = config["guidance"]["num_samples"]
    n_conditions = len(config["properties"])

    print("=== Composable Discrete Flows — CFG Composition ===")
    print(f"K={K}, L={L}, w={w}, steps={num_steps}")
    print()

    # Step 1: Train conditional flow with CFG dropout
    print("[1/2] Training conditional flow with CFG dropout...")
    model = train_conditional_flow(config)
    print()

    # Step 2: Evaluate
    print("[2/2] Evaluating CFG composition...")
    prop_fns = {
        "starts_with_0": PROPERTIES["starts_with_0"],
        "contains_pattern": PROPERTIES["contains_pattern"],
        "no_repeats": PROPERTIES["no_repeats"],
    }
    prop_list = list(prop_fns.keys())

    # Baseline: unconditional (w=0)
    print("\n--- Unconditional (w=0) ---")
    uncond = cfg_sample(model, n_samples, K, L, n_conditions,
                        condition_idxs=[0], ws=[0.0], num_steps=num_steps, mode="single")
    for name, fn in prop_fns.items():
        rate = fn(uncond).float().mean().item()
        print(f"  {name}: {rate*100:.1f}%")

    # Single-condition CFG
    print(f"\n--- Single-condition CFG (w={w}) ---")
    for i, name in enumerate(prop_list):
        samples = cfg_sample(model, n_samples, K, L, n_conditions,
                             condition_idxs=[i], ws=[w], num_steps=num_steps, mode="single")
        rate = prop_fns[name](samples).float().mean().item()
        print(f"  {name}: {rate*100:.1f}%")

    # AND composition
    print(f"\n--- AND(starts_with_0, contains_pattern) w={w} ---")
    samples_and = cfg_sample(model, n_samples, K, L, n_conditions,
                             condition_idxs=[0, 1], ws=[w, w],
                             num_steps=num_steps, mode="and")
    rate_a = prop_fns["starts_with_0"](samples_and).float().mean().item()
    rate_b = prop_fns["contains_pattern"](samples_and).float().mean().item()
    acc_and = evaluate_boolean_accuracy(
        samples_and, [prop_fns["starts_with_0"], prop_fns["contains_pattern"]], mode="and"
    )
    print(f"  starts_with_0: {rate_a*100:.1f}%")
    print(f"  contains_pattern: {rate_b*100:.1f}%")
    print(f"  BOTH (AND accuracy): {acc_and*100:.1f}%")

    # NOT composition
    print(f"\n--- NOT: starts_with_0 AND NOT no_repeats ---")
    samples_not = cfg_sample_not(model, n_samples, K, L, n_conditions,
                                 keep_idx=0, avoid_idx=2,
                                 w_keep=w, w_avoid=w, num_steps=num_steps)
    rate_a = prop_fns["starts_with_0"](samples_not).float().mean().item()
    rate_b = prop_fns["no_repeats"](samples_not).float().mean().item()
    acc_not = evaluate_boolean_accuracy(
        samples_not, [prop_fns["starts_with_0"], prop_fns["no_repeats"]], mode="not"
    )
    print(f"  starts_with_0: {rate_a*100:.1f}%")
    print(f"  no_repeats (should be LOW): {rate_b*100:.1f}%")
    print(f"  A AND NOT B accuracy: {acc_not*100:.1f}%")

    # OR composition
    print(f"\n--- OR(starts_with_0, contains_pattern) w={w} ---")
    samples_or = cfg_sample(model, n_samples, K, L, n_conditions,
                            condition_idxs=[0, 1], ws=[w, w],
                            num_steps=num_steps, mode="or")
    acc_or = evaluate_boolean_accuracy(
        samples_or, [prop_fns["starts_with_0"], prop_fns["contains_pattern"]], mode="or"
    )
    print(f"  A OR B accuracy: {acc_or*100:.1f}%")

    # Kill gate
    print(f"\n=== KILL GATE ===")
    print(f"  AND accuracy: {acc_and*100:.1f}% (target: >80%)")
    print(f"  NOT accuracy: {acc_not*100:.1f}% (target: >80%)")
    if acc_and >= 0.8 and acc_not >= 0.8:
        print("  PASS — proceed to Phase 2")
    elif acc_and >= 0.5 or acc_not >= 0.5:
        print("  PARTIAL — increase w, more epochs, or tune dropout")
    else:
        print("  FAIL — fundamental issue")

    return {"and": acc_and, "not": acc_not, "or": acc_or}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cfg_synthetic.yaml")
    args = parser.parse_args()
    run_cfg_experiment(args.config)
```

- [ ] **Step 4: Run integration test**

Run: `source .venv/bin/activate && pytest tests/test_cfg_integration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/experiments/run_cfg.py configs/cfg_synthetic.yaml tests/test_cfg_integration.py
git commit -m "feat: CFG composition experiment with kill-gate evaluation"
```

---

### Task 6: Smoke Test + Full Run

- [ ] **Step 1: Create fast smoke config**

```yaml
# configs/cfg_smoke.yaml
data:
  K: 4
  L: 8
  n_train: 1000
  n_val: 200

model:
  hidden_dim: 32
  num_layers: 2

training:
  epochs: 20
  lr: 0.001
  batch_size: 64
  cfg_dropout_prob: 0.15

guidance:
  w: 3.0
  num_steps: 10
  num_samples: 100

properties:
  - name: starts_with_0
  - name: contains_pattern
  - name: no_repeats
```

- [ ] **Step 2: Run smoke test**

Run: `source .venv/bin/activate && python -m src.experiments.run_cfg --config configs/cfg_smoke.yaml`
Expected: Completes without error. Numbers will be low (undertrained) but pipeline runs end-to-end.

- [ ] **Step 3: Fix any runtime errors**

- [ ] **Step 4: Commit**

```bash
git add configs/cfg_smoke.yaml
git commit -m "feat: CFG smoke test config"
```

- [ ] **Step 5: Run full experiment on dev desktop**

```bash
scp configs/cfg_synthetic.yaml src/data/synthetic.py src/models/conditional_flow.py src/training/train_conditional.py src/guidance/cfg_compose.py src/experiments/run_cfg.py dev-dsk-divkov-1b-029561b7.us-east-1.amazon.com:~/biostat/
ssh dev-dsk-divkov-1b-029561b7.us-east-1.amazon.com "source ~/miniconda3/bin/activate && cd ~/biostat && nohup python -u -m src.experiments.run_cfg --config configs/cfg_synthetic.yaml > cfg_experiment.log 2>&1 &"
```

- [ ] **Step 6: Monitor and report results**

```bash
ssh dev-dsk-divkov-1b-029561b7.us-east-1.amazon.com "tail -20 ~/biostat/cfg_experiment.log"
```

---

## Self-Review Checklist

- [x] Spec coverage: structured data ✓, conditional flow with CFG dropout ✓, AND/NOT/OR composition ✓, sampling pipeline ✓, kill gate ✓
- [x] No placeholders — all code complete
- [x] Type consistency: `ConditionalProbPathFlow` interface consistent across train_conditional.py, cfg_compose.py, and tests
- [x] Function signatures match: `cfg_sample(model, n, K, L, n_conditions, condition_idxs, ws, num_steps, mode)` used consistently
- [x] All tests have concrete assertions
- [x] Key insight addressed: CFG operates on a model that ALREADY generates good conditioned samples — unlike classifier guidance which tries to steer an unconditional model
- [x] OR composition uses element-wise max approximation (valid for logits; log-sum-exp alternative noted)
- [x] Existing code preserved — new files only, old classifier-guidance code left intact for comparison
