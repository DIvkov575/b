# Composable Discrete Flows — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the synthetic validation phase (weeks 1-3) of the Composable Discrete Flows project — CTMC and probability-path discrete flow models, 3 binary classifiers on synthetic data, and AND/NOT/OR composition operators — to hit the kill-gate target of >80% Boolean accuracy on K=8, L=32 categorical sequences.

**Architecture:** Two discrete flow base models (CTMC rate-matrix and probability-path posterior-prediction) trained unconditionally on synthetic categorical sequences. Three time-conditional binary classifiers trained on known sequence properties. A composition engine applies AND/NOT/OR operators to modify generation. Tau-leaping (CTMC) and Euler-step (prob-path) samplers produce sequences under composed guidance.

**Tech Stack:** PyTorch (no PyG needed), NumPy, SciPy, PyYAML, pytest

---

## File Structure

```
src/
├── __init__.py
├── data/
│   ├── __init__.py
│   └── synthetic.py          # Synthetic sequence dataset + property functions
├── models/
│   ├── __init__.py
│   ├── ctmc_flow.py          # CTMC discrete flow (rate matrix, denoiser network)
│   ├── prob_path_flow.py     # Probability-path discrete flow (posterior predictor)
│   └── classifier.py         # Time-conditional binary classifier
├── guidance/
│   ├── __init__.py
│   ├── compose.py            # AND/NOT/OR operators (framework-agnostic interface)
│   ├── ctmc_guidance.py      # CTMC-specific guided sampling (tau-leaping)
│   └── prob_path_guidance.py # Prob-path-specific guided sampling (Euler)
├── training/
│   ├── __init__.py
│   ├── train_flow.py         # Training loop for base flow models
│   └── train_classifier.py   # Training loop for classifiers
└── experiments/
    ├── __init__.py
    └── run_synthetic.py      # End-to-end: train, compose, evaluate Boolean accuracy
tests/
├── __init__.py
├── test_synthetic_data.py
├── test_ctmc_flow.py
├── test_prob_path_flow.py
├── test_classifier.py
├── test_composition.py
└── test_integration.py
configs/
└── synthetic.yaml
```

---

### Task 1: Project Skeleton + Synthetic Data

**Files:**
- Create: `requirements.txt`
- Create: `setup.py`
- Create: `configs/synthetic.yaml`
- Create: `src/__init__.py`, `src/data/__init__.py`, `src/models/__init__.py`, `src/guidance/__init__.py`, `src/training/__init__.py`, `src/experiments/__init__.py`
- Create: `tests/__init__.py`
- Create: `src/data/synthetic.py`
- Create: `tests/test_synthetic_data.py`

- [ ] **Step 1: Remove old DPP codebase**

```bash
rm -rf src/ tests/ configs/ results/ setup.py requirements.txt
```

- [ ] **Step 2: Create requirements.txt**

```
torch>=2.1.0
numpy>=1.24
scipy>=1.10
pyyaml>=6.0
pytest>=7.0
```

- [ ] **Step 3: Create setup.py**

```python
from setuptools import setup, find_packages

setup(
    name="composable-discrete-flows",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
)
```

- [ ] **Step 4: Create configs/synthetic.yaml**

```yaml
data:
  K: 8          # vocabulary size (number of categories)
  L: 32         # sequence length
  n_train: 10000
  n_val: 2000

model:
  hidden_dim: 128
  num_layers: 3
  time_embed_dim: 64

training:
  epochs: 100
  lr: 0.001
  batch_size: 256
  flow_type: ctmc  # or prob_path

guidance:
  gamma: 1.0       # guidance strength
  num_steps: 100   # sampling steps
  num_samples: 1000

properties:
  - name: starts_with_0
    description: "sequence[0] == 0"
  - name: contains_pattern
    description: "subsequence [1,2,3] appears somewhere"
  - name: no_repeats
    description: "no two adjacent positions share same value"
```

- [ ] **Step 5: Create directory structure**

```bash
mkdir -p src/data src/models src/guidance src/training src/experiments tests configs
touch src/__init__.py src/data/__init__.py src/models/__init__.py src/guidance/__init__.py src/training/__init__.py src/experiments/__init__.py tests/__init__.py
```

- [ ] **Step 6: Write failing test for synthetic data**

```python
# tests/test_synthetic_data.py
import torch
from src.data.synthetic import (
    generate_uniform_sequences,
    property_starts_with_0,
    property_contains_pattern,
    property_no_adjacent_repeats,
    SyntheticSequenceDataset,
)


def test_generate_uniform_sequences():
    seqs = generate_uniform_sequences(n=100, K=8, L=32)
    assert seqs.shape == (100, 32)
    assert seqs.dtype == torch.long
    assert seqs.min() >= 0
    assert seqs.max() <= 7


def test_property_starts_with_0():
    seqs = torch.tensor([[0, 1, 2], [1, 2, 3], [0, 0, 0]])
    labels = property_starts_with_0(seqs)
    assert labels.tolist() == [True, False, True]


def test_property_contains_pattern():
    seqs = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 0, 1, 2, 3, 0, 0, 0],
        [7, 7, 7, 7, 7, 7, 7, 7],
    ])
    labels = property_contains_pattern(seqs, pattern=[1, 2, 3])
    assert labels.tolist() == [True, True, False]


def test_property_no_adjacent_repeats():
    seqs = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 0, 1, 2, 3, 4, 5, 6],
        [1, 2, 1, 2, 1, 2, 1, 2],
    ])
    labels = property_no_adjacent_repeats(seqs)
    assert labels.tolist() == [True, False, True]


def test_synthetic_dataset():
    ds = SyntheticSequenceDataset(n=500, K=8, L=32)
    seq, props = ds[0]
    assert seq.shape == (32,)
    assert len(props) == 3
    assert all(isinstance(v, bool) for v in props.values())
```

- [ ] **Step 7: Run test to verify it fails**

Run: `pytest tests/test_synthetic_data.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 8: Implement src/data/synthetic.py**

```python
import torch
from torch.utils.data import Dataset


def generate_uniform_sequences(n: int, K: int, L: int) -> torch.Tensor:
    return torch.randint(0, K, (n, L))


def property_starts_with_0(seqs: torch.Tensor) -> torch.Tensor:
    return seqs[:, 0] == 0


def property_contains_pattern(seqs: torch.Tensor, pattern: list = [1, 2, 3]) -> torch.Tensor:
    pat = torch.tensor(pattern, device=seqs.device)
    pat_len = len(pattern)
    n, L = seqs.shape
    results = torch.zeros(n, dtype=torch.bool, device=seqs.device)
    for i in range(L - pat_len + 1):
        match = (seqs[:, i:i + pat_len] == pat.unsqueeze(0)).all(dim=1)
        results |= match
    return results


def property_no_adjacent_repeats(seqs: torch.Tensor) -> torch.Tensor:
    diffs = seqs[:, 1:] != seqs[:, :-1]
    return diffs.all(dim=1)


PROPERTIES = {
    "starts_with_0": property_starts_with_0,
    "contains_pattern": property_contains_pattern,
    "no_repeats": property_no_adjacent_repeats,
}


class SyntheticSequenceDataset(Dataset):
    def __init__(self, n: int = 10000, K: int = 8, L: int = 32):
        self.seqs = generate_uniform_sequences(n, K, L)
        self.labels = {
            "starts_with_0": property_starts_with_0(self.seqs),
            "contains_pattern": property_contains_pattern(self.seqs),
            "no_repeats": property_no_adjacent_repeats(self.seqs),
        }

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        props = {k: v[idx].item() for k, v in self.labels.items()}
        return seq, props
```

- [ ] **Step 9: Run tests**

Run: `pytest tests/test_synthetic_data.py -v`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "feat: project skeleton + synthetic sequence data with 3 properties"
```

---

### Task 2: CTMC Discrete Flow Base Model

**Files:**
- Create: `src/models/ctmc_flow.py`
- Create: `tests/test_ctmc_flow.py`

- [ ] **Step 1: Write failing test for CTMC flow**

```python
# tests/test_ctmc_flow.py
import torch
from src.models.ctmc_flow import (
    uniform_rate_matrix,
    noise_sequence,
    CTMCDenoiser,
    compute_rate_from_denoiser,
)


def test_uniform_rate_matrix():
    R = uniform_rate_matrix(K=8)
    assert R.shape == (8, 8)
    assert torch.allclose(R.sum(dim=1), torch.zeros(8), atol=1e-6)
    assert (R[torch.eye(8, dtype=torch.bool) == False] >= 0).all()


def test_noise_sequence():
    x0 = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    x_t = noise_sequence(x0, t=0.0, K=8)
    assert (x_t == x0).all()

    x_t_noisy = noise_sequence(x0, t=1.0, K=8)
    assert x_t_noisy.shape == x0.shape
    assert x_t_noisy.min() >= 0 and x_t_noisy.max() <= 7


def test_ctmc_denoiser_forward():
    model = CTMCDenoiser(K=8, L=32, hidden_dim=64, num_layers=2)
    x_t = torch.randint(0, 8, (4, 32))
    t = torch.tensor([0.5, 0.3, 0.8, 0.1])
    logits = model(x_t, t)
    assert logits.shape == (4, 32, 8)


def test_compute_rate_from_denoiser():
    model = CTMCDenoiser(K=8, L=32, hidden_dim=64, num_layers=2)
    x_t = torch.randint(0, 8, (1, 32))
    t = torch.tensor([0.5])
    rates = compute_rate_from_denoiser(model, x_t, t, K=8)
    # rates[b, l, k] = rate of position l transitioning to state k
    assert rates.shape == (1, 32, 8)
    # rate to current state should be 0 (no self-transition)
    for l in range(32):
        assert rates[0, l, x_t[0, l].item()].item() <= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ctmc_flow.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement src/models/ctmc_flow.py**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def uniform_rate_matrix(K: int) -> torch.Tensor:
    R = torch.ones(K, K) / (K - 1)
    R.fill_diagonal_(0.0)
    R -= torch.diag(R.sum(dim=1))
    return R


def noise_sequence(x0: torch.Tensor, t: float, K: int) -> torch.Tensor:
    if t == 0.0:
        return x0.clone()
    # Marginal: p(x_t = j | x_0 = i) = (1-alpha_t)*delta(i,j) + alpha_t/K
    # where alpha_t = 1 - exp(-t * K/(K-1))
    alpha_t = 1.0 - torch.exp(torch.tensor(-t * K / (K - 1)))
    mask = torch.rand_like(x0.float()) < alpha_t
    noise = torch.randint(0, K, x0.shape, device=x0.device)
    return torch.where(mask, noise, x0)


class CTMCDenoiser(nn.Module):
    def __init__(self, K: int, L: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.K = K
        self.L = L
        self.embed = nn.Embedding(K, hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, K)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x_t: (B, L) integers, t: (B,) floats in [0, 1]
        h = self.embed(x_t)  # (B, L, hidden)
        t_emb = self.time_embed(t.unsqueeze(-1))  # (B, hidden)
        h = h + t_emb.unsqueeze(1)  # broadcast time to all positions
        h = self.net(h)
        logits = self.out(h)  # (B, L, K)
        return logits


def compute_rate_from_denoiser(
    model: CTMCDenoiser, x_t: torch.Tensor, t: torch.Tensor, K: int
) -> torch.Tensor:
    """Compute transition rates from denoiser predictions.

    R_t(x'|x) = p_theta(x_0=x' | x_t) * rate_ref(x'|x) / sum_x0 p_theta(x_0 | x_t) * rate_ref(x'|x)
    Simplified: rates proportional to predicted clean-data probabilities for non-current states.
    """
    logits = model(x_t, t)  # (B, L, K)
    probs = F.softmax(logits, dim=-1)  # (B, L, K)
    # Zero out rate to current state (no self-transition)
    B, L, _ = probs.shape
    current = x_t.unsqueeze(-1)  # (B, L, 1)
    mask = torch.zeros_like(probs).scatter_(2, current, 1.0)
    rates = probs * (1.0 - mask)
    # Diagonal: negative sum of off-diagonal (so rows sum to 0)
    diag = -rates.sum(dim=-1, keepdim=True)
    rates = rates + mask * diag
    return rates
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_ctmc_flow.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/ctmc_flow.py tests/test_ctmc_flow.py
git commit -m "feat: CTMC discrete flow base model with denoiser and rate computation"
```

---

### Task 3: Probability-Path Discrete Flow Base Model

**Files:**
- Create: `src/models/prob_path_flow.py`
- Create: `tests/test_prob_path_flow.py`

- [ ] **Step 1: Write failing test for prob-path flow**

```python
# tests/test_prob_path_flow.py
import torch
from src.models.prob_path_flow import (
    interpolate_categorical,
    ProbPathDenoiser,
    sample_euler_step,
)


def test_interpolate_categorical():
    x0 = torch.tensor([0, 1, 2, 3])
    K = 8
    # t=0 → one-hot on x0
    p_t = interpolate_categorical(x0, t=0.0, K=K)
    assert p_t.shape == (4, K)
    assert torch.allclose(p_t.sum(dim=-1), torch.ones(4))
    for i in range(4):
        assert p_t[i, x0[i]].item() > 0.99

    # t=1 → uniform
    p_t = interpolate_categorical(x0, t=1.0, K=K)
    expected = torch.ones(4, K) / K
    assert torch.allclose(p_t, expected, atol=0.01)


def test_prob_path_denoiser_forward():
    model = ProbPathDenoiser(K=8, L=32, hidden_dim=64, num_layers=2)
    x_t = torch.randint(0, 8, (4, 32))
    t = torch.tensor([0.5, 0.3, 0.8, 0.1])
    logits = model(x_t, t)
    assert logits.shape == (4, 32, 8)


def test_sample_euler_step():
    K = 8
    # posterior: strong preference for state 0
    posterior = torch.zeros(2, 4, K)
    posterior[:, :, 0] = 10.0
    posterior = torch.softmax(posterior, dim=-1)
    x_t = torch.randint(0, K, (2, 4))
    x_next = sample_euler_step(x_t, posterior, dt=0.1, K=K)
    assert x_next.shape == (2, 4)
    assert x_next.min() >= 0 and x_next.max() < K
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prob_path_flow.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement src/models/prob_path_flow.py**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def interpolate_categorical(x0: torch.Tensor, t: float, K: int) -> torch.Tensor:
    """Linear interpolation between one-hot(x0) and uniform on the simplex.

    p_t(k) = (1-t) * one_hot(x0, k) + t * (1/K)
    """
    one_hot = F.one_hot(x0, K).float()  # (..., K)
    uniform = torch.ones_like(one_hot) / K
    return (1.0 - t) * one_hot + t * uniform


def sample_from_categorical(x0: torch.Tensor, t: float, K: int) -> torch.Tensor:
    """Sample x_t from the interpolated categorical distribution."""
    p_t = interpolate_categorical(x0, t, K)
    flat = p_t.reshape(-1, K)
    samples = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return samples.reshape(x0.shape)


class ProbPathDenoiser(nn.Module):
    """Predicts p(x_1 | x_t) — the clean-data posterior given noisy observation."""

    def __init__(self, K: int, L: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.K = K
        self.L = L
        self.embed = nn.Embedding(K, hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, K)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.embed(x_t)  # (B, L, hidden)
        t_emb = self.time_embed(t.unsqueeze(-1))  # (B, hidden)
        h = h + t_emb.unsqueeze(1)
        h = self.net(h)
        return self.out(h)  # (B, L, K) logits for p(x_0 | x_t)


def sample_euler_step(
    x_t: torch.Tensor, posterior_probs: torch.Tensor, dt: float, K: int
) -> torch.Tensor:
    """One Euler step: mix current one-hot with predicted posterior, sample."""
    # posterior_probs: (B, L, K) — predicted clean distribution
    one_hot_current = F.one_hot(x_t, K).float()
    # p_{t-dt} = (1-dt) * one_hot(x_t) + dt * posterior
    p_next = (1.0 - dt) * one_hot_current + dt * posterior_probs
    p_next = p_next.clamp(min=1e-8)
    p_next = p_next / p_next.sum(dim=-1, keepdim=True)
    B, L, _ = p_next.shape
    flat = p_next.reshape(-1, K)
    samples = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return samples.reshape(B, L)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_prob_path_flow.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/prob_path_flow.py tests/test_prob_path_flow.py
git commit -m "feat: probability-path discrete flow with Euler sampling"
```

---

### Task 4: Time-Conditional Binary Classifier

**Files:**
- Create: `src/models/classifier.py`
- Create: `tests/test_classifier.py`

- [ ] **Step 1: Write failing test for classifier**

```python
# tests/test_classifier.py
import torch
from src.models.classifier import TimeConditionalClassifier


def test_classifier_forward():
    model = TimeConditionalClassifier(K=8, L=32, hidden_dim=64, num_layers=2)
    x_t = torch.randint(0, 8, (4, 32))
    t = torch.tensor([0.5, 0.3, 0.8, 0.1])
    logits = model(x_t, t)
    assert logits.shape == (4, 1)


def test_classifier_probability():
    model = TimeConditionalClassifier(K=8, L=32, hidden_dim=64, num_layers=2)
    x_t = torch.randint(0, 8, (4, 32))
    t = torch.tensor([0.5, 0.3, 0.8, 0.1])
    prob = model.predict_prob(x_t, t)
    assert prob.shape == (4,)
    assert (prob >= 0).all() and (prob <= 1).all()


def test_classifier_log_ratio():
    model = TimeConditionalClassifier(K=8, L=32, hidden_dim=64, num_layers=2)
    x = torch.randint(0, 8, (2, 32))
    x_prime = torch.randint(0, 8, (2, 32))
    t = torch.tensor([0.5, 0.5])
    log_ratio = model.log_prob_ratio(x, x_prime, t)
    assert log_ratio.shape == (2,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_classifier.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement src/models/classifier.py**

```python
import torch
import torch.nn as nn


class TimeConditionalClassifier(nn.Module):
    """Binary classifier p(y=1 | x_t, t) — time-conditional for noisy inputs."""

    def __init__(self, K: int, L: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.K = K
        self.L = L
        self.embed = nn.Embedding(K, hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)
        self.pool = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.embed(x_t)  # (B, L, hidden)
        t_emb = self.time_embed(t.unsqueeze(-1))  # (B, hidden)
        h = h + t_emb.unsqueeze(1)
        h = self.net(h)
        h = h.mean(dim=1)  # pool over sequence
        h = torch.relu(self.pool(h))
        return self.head(h)  # (B, 1)

    def predict_prob(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x_t, t).squeeze(-1)
        return torch.sigmoid(logits)

    def log_prob_ratio(
        self, x: torch.Tensor, x_prime: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute log[p(y|x')/p(y|x)] — the guidance signal for CTMC rates."""
        log_p_x = torch.log(self.predict_prob(x, t).clamp(min=1e-8))
        log_p_x_prime = torch.log(self.predict_prob(x_prime, t).clamp(min=1e-8))
        return log_p_x_prime - log_p_x
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_classifier.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/classifier.py tests/test_classifier.py
git commit -m "feat: time-conditional binary classifier with log-ratio for guidance"
```

---

### Task 5: Training Loops (Flow + Classifier)

**Files:**
- Create: `src/training/train_flow.py`
- Create: `src/training/train_classifier.py`

- [ ] **Step 1: Implement src/training/train_flow.py**

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data.synthetic import SyntheticSequenceDataset
from src.models.ctmc_flow import CTMCDenoiser, noise_sequence
from src.models.prob_path_flow import ProbPathDenoiser, sample_from_categorical


def train_ctmc_flow(config: dict, device: str = "cpu") -> CTMCDenoiser:
    K = config["data"]["K"]
    L = config["data"]["L"]
    model = CTMCDenoiser(
        K=K, L=L,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
    ).to(device)
    ds = SyntheticSequenceDataset(n=config["data"]["n_train"], K=K, L=L)
    loader = DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    model.train()
    for epoch in range(config["training"]["epochs"]):
        total_loss = 0.0
        for seqs, _ in loader:
            seqs = seqs.to(device)
            B = seqs.shape[0]
            t = torch.rand(B, device=device)
            x_t = torch.stack([noise_sequence(seqs[i], t[i].item(), K) for i in range(B)])
            logits = model(x_t, t)
            loss = F.cross_entropy(logits.reshape(-1, K), seqs.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"  CTMC epoch {epoch+1}: loss={total_loss / len(loader):.4f}")
    return model


def train_prob_path_flow(config: dict, device: str = "cpu") -> ProbPathDenoiser:
    K = config["data"]["K"]
    L = config["data"]["L"]
    model = ProbPathDenoiser(
        K=K, L=L,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
    ).to(device)
    ds = SyntheticSequenceDataset(n=config["data"]["n_train"], K=K, L=L)
    loader = DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    model.train()
    for epoch in range(config["training"]["epochs"]):
        total_loss = 0.0
        for seqs, _ in loader:
            seqs = seqs.to(device)
            B = seqs.shape[0]
            t = torch.rand(B, device=device)
            x_t = torch.stack([
                sample_from_categorical(seqs[i], t[i].item(), K) for i in range(B)
            ])
            logits = model(x_t, t)
            loss = F.cross_entropy(logits.reshape(-1, K), seqs.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"  ProbPath epoch {epoch+1}: loss={total_loss / len(loader):.4f}")
    return model
```

- [ ] **Step 2: Implement src/training/train_classifier.py**

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from src.data.synthetic import SyntheticSequenceDataset, PROPERTIES
from src.models.classifier import TimeConditionalClassifier
from src.models.ctmc_flow import noise_sequence


def train_classifier(
    property_name: str, config: dict, device: str = "cpu"
) -> TimeConditionalClassifier:
    K = config["data"]["K"]
    L = config["data"]["L"]
    ds = SyntheticSequenceDataset(n=config["data"]["n_train"], K=K, L=L)
    labels = ds.labels[property_name].float().to(device)
    seqs = ds.seqs.to(device)

    model = TimeConditionalClassifier(
        K=K, L=L,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    batch_size = config["training"]["batch_size"]

    model.train()
    for epoch in range(config["training"]["epochs"]):
        perm = torch.randperm(len(seqs), device=device)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(seqs), batch_size):
            idx = perm[i:i + batch_size]
            batch_seqs = seqs[idx]
            batch_labels = labels[idx]
            # Random time + noise for time-conditional training
            t = torch.rand(len(idx), device=device)
            x_t = torch.stack([
                noise_sequence(batch_seqs[j], t[j].item(), K) for j in range(len(idx))
            ])
            logits = model(x_t, t).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 20 == 0:
            print(f"  Classifier[{property_name}] epoch {epoch+1}: "
                  f"loss={total_loss / n_batches:.4f}")
    return model
```

- [ ] **Step 3: Commit**

```bash
git add src/training/train_flow.py src/training/train_classifier.py
git commit -m "feat: training loops for flow models and classifiers"
```

---

### Task 6: Composition Engine — AND/NOT/OR Operators

**Files:**
- Create: `src/guidance/compose.py`
- Create: `src/guidance/ctmc_guidance.py`
- Create: `src/guidance/prob_path_guidance.py`
- Create: `tests/test_composition.py`

- [ ] **Step 1: Write failing test for composition**

```python
# tests/test_composition.py
import torch
from src.models.classifier import TimeConditionalClassifier
from src.guidance.compose import compose_and, compose_not, compose_or
from src.guidance.ctmc_guidance import guided_rates_ctmc
from src.guidance.prob_path_guidance import guided_posterior_prob_path


def _make_classifier(K=8, L=32):
    return TimeConditionalClassifier(K=K, L=L, hidden_dim=32, num_layers=1)


def test_compose_and_prob_path():
    clf_a = _make_classifier()
    clf_b = _make_classifier()
    x_t = torch.randint(0, 8, (4, 32))
    t = torch.tensor([0.5] * 4)
    base_logits = torch.randn(4, 32, 8)
    guided = compose_and(
        base_logits, [clf_a, clf_b], x_t, t, gammas=[1.0, 1.0], mode="prob_path"
    )
    assert guided.shape == (4, 32, 8)
    # Should still be valid probability after softmax
    probs = torch.softmax(guided, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4, 32), atol=1e-5)


def test_compose_not_prob_path():
    clf_a = _make_classifier()
    clf_b = _make_classifier()
    x_t = torch.randint(0, 8, (4, 32))
    t = torch.tensor([0.5] * 4)
    base_logits = torch.randn(4, 32, 8)
    guided = compose_not(
        base_logits, clf_a, clf_b, x_t, t, gamma_a=1.0, gamma_b=1.0, mode="prob_path"
    )
    assert guided.shape == (4, 32, 8)


def test_compose_or_prob_path():
    clf_a = _make_classifier()
    clf_b = _make_classifier()
    x_t = torch.randint(0, 8, (4, 32))
    t = torch.tensor([0.5] * 4)
    base_logits = torch.randn(4, 32, 8)
    guided = compose_or(
        base_logits, [clf_a, clf_b], x_t, t, gammas=[1.0, 1.0], mode="prob_path"
    )
    assert guided.shape == (4, 32, 8)


def test_guided_rates_ctmc():
    clf = _make_classifier()
    x_t = torch.randint(0, 8, (2, 32))
    t = torch.tensor([0.5, 0.5])
    base_rates = torch.rand(2, 32, 8)
    guided = guided_rates_ctmc(base_rates, x_t, t, [clf], gammas=[1.0])
    assert guided.shape == (2, 32, 8)


def test_guided_posterior_prob_path():
    clf = _make_classifier()
    x_t = torch.randint(0, 8, (2, 32))
    t = torch.tensor([0.5, 0.5])
    base_logits = torch.randn(2, 32, 8)
    guided = guided_posterior_prob_path(base_logits, x_t, t, [clf], gammas=[1.0])
    assert guided.shape == (2, 32, 8)
    probs = torch.softmax(guided, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 32), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_composition.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement src/guidance/compose.py**

```python
import torch
import torch.nn.functional as F
from src.models.classifier import TimeConditionalClassifier


def _classifier_log_guidance(
    clf: TimeConditionalClassifier, x_t: torch.Tensor, t: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Compute gamma * log p(y=1 | x_t, t) — scalar per batch element."""
    prob = clf.predict_prob(x_t, t)  # (B,)
    return gamma * torch.log(prob.clamp(min=1e-8))


def compose_and(
    base_logits: torch.Tensor,
    classifiers: list,
    x_t: torch.Tensor,
    t: torch.Tensor,
    gammas: list,
    mode: str = "prob_path",
) -> torch.Tensor:
    """AND composition: multiply posteriors by product of classifier likelihoods.

    p(x_0 | x_t, A∧B) ∝ p(x_0 | x_t) * p(A|x_0)^γA * p(B|x_0)^γB
    In log space: logits + γA*log_pA + γB*log_pB (broadcast over positions/states)
    """
    log_guidance = torch.zeros(x_t.shape[0], device=x_t.device)
    for clf, gamma in zip(classifiers, gammas):
        log_guidance = log_guidance + _classifier_log_guidance(clf, x_t, t, gamma)
    # Broadcast: (B,) -> (B, 1, 1) to add to (B, L, K) logits
    return base_logits + log_guidance.unsqueeze(-1).unsqueeze(-1)


def compose_not(
    base_logits: torch.Tensor,
    clf_keep: TimeConditionalClassifier,
    clf_avoid: TimeConditionalClassifier,
    x_t: torch.Tensor,
    t: torch.Tensor,
    gamma_a: float = 1.0,
    gamma_b: float = 1.0,
    mode: str = "prob_path",
) -> torch.Tensor:
    """NOT composition: A ∧ ¬B.

    p(x_0 | x_t, A∧¬B) ∝ p(x_0 | x_t) * p(A|x_0)^γA * p(B|x_0)^{-γB}
    """
    log_keep = _classifier_log_guidance(clf_keep, x_t, t, gamma_a)
    log_avoid = _classifier_log_guidance(clf_avoid, x_t, t, -gamma_b)
    log_guidance = log_keep + log_avoid
    return base_logits + log_guidance.unsqueeze(-1).unsqueeze(-1)


def compose_or(
    base_logits: torch.Tensor,
    classifiers: list,
    x_t: torch.Tensor,
    t: torch.Tensor,
    gammas: list,
    mode: str = "prob_path",
) -> torch.Tensor:
    """OR composition: A ∨ B.

    p(x_0 | x_t, A∨B) ∝ p(x_0|x_t) * [p(A|x_0)^γ + p(B|x_0)^γ - p(A|x_0)^γ*p(B|x_0)^γ]
    Use log-sum-exp approximation: log(exp(a) + exp(b) - exp(a+b))
    """
    log_probs = []
    for clf, gamma in zip(classifiers, gammas):
        log_probs.append(_classifier_log_guidance(clf, x_t, t, gamma))
    # Inclusion-exclusion: P(A∨B) = P(A) + P(B) - P(A)P(B)
    # In log space for 2 classifiers:
    a, b = log_probs[0], log_probs[1]
    # log(exp(a) + exp(b) - exp(a+b)) via log-sum-exp with correction
    max_ab = torch.max(a, b)
    log_or = max_ab + torch.log(
        torch.exp(a - max_ab) + torch.exp(b - max_ab)
        - torch.exp(a + b - max_ab)
    .clamp(min=1e-8))
    return base_logits + log_or.unsqueeze(-1).unsqueeze(-1)
```

- [ ] **Step 4: Implement src/guidance/ctmc_guidance.py**

```python
import torch
from src.models.classifier import TimeConditionalClassifier


def guided_rates_ctmc(
    base_rates: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    classifiers: list,
    gammas: list,
    avoid_classifiers: list = None,
    avoid_gammas: list = None,
) -> torch.Tensor:
    """Apply classifier guidance to CTMC transition rates.

    R_guided(x'|x) = R_base(x'|x) * prod_i [p(yi|x',t)/p(yi|x,t)]^γi

    For each possible next state x', compute the ratio of classifier probs.
    Approximation: evaluate classifier at x_t (current) vs each one-hot-substituted x'.
    For efficiency, use the fact that only one position changes at a time in CTMC.
    Simplified: scale rates by classifier prob at current x.
    """
    B, L, K = base_rates.shape
    # Simplified guidance: scale all rates by classifier signal at current state
    log_scale = torch.zeros(B, device=x_t.device)
    for clf, gamma in zip(classifiers, gammas):
        prob = clf.predict_prob(x_t, t)
        log_scale = log_scale + gamma * torch.log(prob.clamp(min=1e-8))

    if avoid_classifiers:
        for clf, gamma in zip(avoid_classifiers, avoid_gammas):
            prob = clf.predict_prob(x_t, t)
            log_scale = log_scale - gamma * torch.log(prob.clamp(min=1e-8))

    scale = torch.exp(log_scale).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
    guided = base_rates * scale
    return guided


def sample_tau_leaping(
    rates: torch.Tensor, x_t: torch.Tensor, dt: float, K: int
) -> torch.Tensor:
    """One tau-leaping step: sample transitions from Poisson with given rates."""
    B, L, _ = rates.shape
    # For each position, probability of transitioning to state k in dt
    # Poisson approximation: prob ≈ rate * dt for small dt
    trans_probs = (rates * dt).clamp(min=0)
    # Zero out self-transition (diagonal)
    current = x_t.unsqueeze(-1)  # (B, L, 1)
    mask = torch.zeros_like(trans_probs).scatter_(2, current, 1.0)
    trans_probs = trans_probs * (1 - mask)
    # Probability of staying = 1 - sum(transition probs)
    stay_prob = (1.0 - trans_probs.sum(dim=-1, keepdim=True)).clamp(min=0)
    # Combine into full categorical: [stay, trans_to_0, ..., trans_to_{K-1}]
    # Simpler: sample whether to transition, then where
    total_trans = trans_probs.sum(dim=-1)  # (B, L)
    do_transition = torch.rand_like(total_trans) < total_trans
    # Where to transition: normalize trans_probs
    normed = trans_probs / trans_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    flat_normed = normed.reshape(-1, K)
    new_states = torch.multinomial(flat_normed, num_samples=1).squeeze(-1).reshape(B, L)
    return torch.where(do_transition, new_states, x_t)
```

- [ ] **Step 5: Implement src/guidance/prob_path_guidance.py**

```python
import torch
import torch.nn.functional as F
from src.models.classifier import TimeConditionalClassifier


def guided_posterior_prob_path(
    base_logits: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    classifiers: list,
    gammas: list,
    avoid_classifiers: list = None,
    avoid_gammas: list = None,
) -> torch.Tensor:
    """Apply classifier guidance to probability-path posterior.

    p_guided(x_0|x_t) ∝ p_base(x_0|x_t) * prod p(yi|x_t,t)^γi / prod p(yj|x_t,t)^γj
    """
    log_guidance = torch.zeros(x_t.shape[0], device=x_t.device)
    for clf, gamma in zip(classifiers, gammas):
        prob = clf.predict_prob(x_t, t)
        log_guidance = log_guidance + gamma * torch.log(prob.clamp(min=1e-8))

    if avoid_classifiers:
        for clf, gamma in zip(avoid_classifiers, avoid_gammas):
            prob = clf.predict_prob(x_t, t)
            log_guidance = log_guidance - gamma * torch.log(prob.clamp(min=1e-8))

    guided = base_logits + log_guidance.unsqueeze(-1).unsqueeze(-1)
    return guided
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_composition.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/guidance/ tests/test_composition.py
git commit -m "feat: AND/NOT/OR composition operators for CTMC and prob-path guidance"
```

---

### Task 7: End-to-End Sampling Pipelines

**Files:**
- Create: `src/experiments/run_synthetic.py`
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write failing integration test**

```python
# tests/test_integration.py
import torch
from src.experiments.run_synthetic import (
    generate_unconditional,
    generate_guided_single,
    generate_composed_and,
    generate_composed_not,
    evaluate_boolean_accuracy,
)
from src.models.ctmc_flow import CTMCDenoiser
from src.models.prob_path_flow import ProbPathDenoiser
from src.models.classifier import TimeConditionalClassifier
from src.data.synthetic import property_starts_with_0, property_contains_pattern


def _tiny_config():
    return {"data": {"K": 4, "L": 8}, "guidance": {"num_steps": 10, "gamma": 1.0}}


def test_generate_unconditional_ctmc():
    model = CTMCDenoiser(K=4, L=8, hidden_dim=32, num_layers=1)
    samples = generate_unconditional(model, n=10, K=4, L=8, num_steps=10, mode="ctmc")
    assert samples.shape == (10, 8)
    assert samples.min() >= 0 and samples.max() < 4


def test_generate_unconditional_prob_path():
    model = ProbPathDenoiser(K=4, L=8, hidden_dim=32, num_layers=1)
    samples = generate_unconditional(model, n=10, K=4, L=8, num_steps=10, mode="prob_path")
    assert samples.shape == (10, 8)


def test_generate_guided_single():
    model = ProbPathDenoiser(K=4, L=8, hidden_dim=32, num_layers=1)
    clf = TimeConditionalClassifier(K=4, L=8, hidden_dim=32, num_layers=1)
    samples = generate_guided_single(
        model, clf, n=10, K=4, L=8, num_steps=10, gamma=1.0, mode="prob_path"
    )
    assert samples.shape == (10, 8)


def test_generate_composed_and():
    model = ProbPathDenoiser(K=4, L=8, hidden_dim=32, num_layers=1)
    clf_a = TimeConditionalClassifier(K=4, L=8, hidden_dim=32, num_layers=1)
    clf_b = TimeConditionalClassifier(K=4, L=8, hidden_dim=32, num_layers=1)
    samples = generate_composed_and(
        model, [clf_a, clf_b], n=10, K=4, L=8, num_steps=10,
        gammas=[1.0, 1.0], mode="prob_path"
    )
    assert samples.shape == (10, 8)


def test_evaluate_boolean_accuracy():
    seqs = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3],
                         [1, 2, 3, 0, 1, 2, 3, 0]])
    prop_fns = [property_starts_with_0]
    acc = evaluate_boolean_accuracy(seqs, prop_fns, mode="and")
    assert 0.0 <= acc <= 1.0
    assert acc == 0.5  # only first seq starts with 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_integration.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement src/experiments/run_synthetic.py**

```python
import torch
import torch.nn.functional as F
from src.models.ctmc_flow import CTMCDenoiser, noise_sequence, compute_rate_from_denoiser
from src.models.prob_path_flow import ProbPathDenoiser, sample_euler_step
from src.models.classifier import TimeConditionalClassifier
from src.guidance.ctmc_guidance import guided_rates_ctmc, sample_tau_leaping
from src.guidance.prob_path_guidance import guided_posterior_prob_path


@torch.no_grad()
def generate_unconditional(
    model, n: int, K: int, L: int, num_steps: int, mode: str
) -> torch.Tensor:
    device = next(model.parameters()).device
    x_t = torch.randint(0, K, (n, L), device=device)
    dt = 1.0 / num_steps

    model.eval()
    for step in range(num_steps):
        t_val = 1.0 - step * dt
        t = torch.full((n,), t_val, device=device)
        if mode == "ctmc":
            rates = compute_rate_from_denoiser(model, x_t, t, K)
            x_t = sample_tau_leaping(rates, x_t, dt, K)
        else:
            logits = model(x_t, t)
            posterior = F.softmax(logits, dim=-1)
            x_t = sample_euler_step(x_t, posterior, dt, K)
    return x_t


@torch.no_grad()
def generate_guided_single(
    model, classifier: TimeConditionalClassifier,
    n: int, K: int, L: int, num_steps: int, gamma: float, mode: str
) -> torch.Tensor:
    device = next(model.parameters()).device
    x_t = torch.randint(0, K, (n, L), device=device)
    dt = 1.0 / num_steps

    model.eval()
    classifier.eval()
    for step in range(num_steps):
        t_val = 1.0 - step * dt
        t = torch.full((n,), t_val, device=device)
        if mode == "ctmc":
            rates = compute_rate_from_denoiser(model, x_t, t, K)
            rates = guided_rates_ctmc(rates, x_t, t, [classifier], [gamma])
            x_t = sample_tau_leaping(rates, x_t, dt, K)
        else:
            logits = model(x_t, t)
            logits = guided_posterior_prob_path(logits, x_t, t, [classifier], [gamma])
            posterior = F.softmax(logits, dim=-1)
            x_t = sample_euler_step(x_t, posterior, dt, K)
    return x_t


@torch.no_grad()
def generate_composed_and(
    model, classifiers: list,
    n: int, K: int, L: int, num_steps: int, gammas: list, mode: str
) -> torch.Tensor:
    device = next(model.parameters()).device
    x_t = torch.randint(0, K, (n, L), device=device)
    dt = 1.0 / num_steps

    model.eval()
    for clf in classifiers:
        clf.eval()
    for step in range(num_steps):
        t_val = 1.0 - step * dt
        t = torch.full((n,), t_val, device=device)
        if mode == "ctmc":
            rates = compute_rate_from_denoiser(model, x_t, t, K)
            rates = guided_rates_ctmc(rates, x_t, t, classifiers, gammas)
            x_t = sample_tau_leaping(rates, x_t, dt, K)
        else:
            logits = model(x_t, t)
            logits = guided_posterior_prob_path(logits, x_t, t, classifiers, gammas)
            posterior = F.softmax(logits, dim=-1)
            x_t = sample_euler_step(x_t, posterior, dt, K)
    return x_t


@torch.no_grad()
def generate_composed_not(
    model, clf_keep: TimeConditionalClassifier, clf_avoid: TimeConditionalClassifier,
    n: int, K: int, L: int, num_steps: int, gamma_a: float, gamma_b: float, mode: str
) -> torch.Tensor:
    device = next(model.parameters()).device
    x_t = torch.randint(0, K, (n, L), device=device)
    dt = 1.0 / num_steps

    model.eval()
    clf_keep.eval()
    clf_avoid.eval()
    for step in range(num_steps):
        t_val = 1.0 - step * dt
        t = torch.full((n,), t_val, device=device)
        if mode == "ctmc":
            rates = compute_rate_from_denoiser(model, x_t, t, K)
            rates = guided_rates_ctmc(
                rates, x_t, t, [clf_keep], [gamma_a],
                avoid_classifiers=[clf_avoid], avoid_gammas=[gamma_b]
            )
            x_t = sample_tau_leaping(rates, x_t, dt, K)
        else:
            logits = model(x_t, t)
            logits = guided_posterior_prob_path(
                logits, x_t, t, [clf_keep], [gamma_a],
                avoid_classifiers=[clf_avoid], avoid_gammas=[gamma_b]
            )
            posterior = F.softmax(logits, dim=-1)
            x_t = sample_euler_step(x_t, posterior, dt, K)
    return x_t


def evaluate_boolean_accuracy(
    samples: torch.Tensor, property_fns: list, mode: str = "and"
) -> float:
    """Evaluate % of samples satisfying the Boolean condition."""
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
        # A and NOT B
        combined = results[0] & (~results[1])
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return combined.float().mean().item()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/experiments/run_synthetic.py tests/test_integration.py
git commit -m "feat: end-to-end sampling pipelines with composed guidance evaluation"
```

---

### Task 8: Full Experiment Script + Kill-Gate Evaluation

**Files:**
- Modify: `src/experiments/run_synthetic.py` (add `main()` entry point)

- [ ] **Step 1: Add main experiment runner to src/experiments/run_synthetic.py**

Append to the end of `src/experiments/run_synthetic.py`:

```python
import yaml
from pathlib import Path
from src.training.train_flow import train_ctmc_flow, train_prob_path_flow
from src.training.train_classifier import train_classifier
from src.data.synthetic import PROPERTIES
from functools import partial


def run_full_experiment(config_path: str = "configs/synthetic.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    K = config["data"]["K"]
    L = config["data"]["L"]
    num_steps = config["guidance"]["num_steps"]
    gamma = config["guidance"]["gamma"]
    n_samples = config["guidance"]["num_samples"]
    mode = config["training"]["flow_type"]

    print(f"=== Composable Discrete Flows — Synthetic Validation ===")
    print(f"Mode: {mode}, K={K}, L={L}, steps={num_steps}, gamma={gamma}")
    print()

    # Step 1: Train base flow
    print("[1/3] Training base flow model...")
    if mode == "ctmc":
        flow_model = train_ctmc_flow(config)
    else:
        flow_model = train_prob_path_flow(config)
    print()

    # Step 2: Train classifiers
    print("[2/3] Training classifiers...")
    classifiers = {}
    for prop_name in PROPERTIES:
        print(f"  Training classifier: {prop_name}")
        classifiers[prop_name] = train_classifier(prop_name, config)
    print()

    # Step 3: Evaluate composition
    print("[3/3] Evaluating composed guidance...")
    prop_fns = {
        "starts_with_0": PROPERTIES["starts_with_0"],
        "contains_pattern": PROPERTIES["contains_pattern"],
        "no_repeats": PROPERTIES["no_repeats"],
    }

    # Baseline: unconditional
    uncond_samples = generate_unconditional(flow_model, n_samples, K, L, num_steps, mode)
    print(f"\n--- Unconditional baseline ---")
    for name, fn in prop_fns.items():
        rate = fn(uncond_samples).float().mean().item()
        print(f"  {name}: {rate*100:.1f}%")

    # Single-condition guidance
    print(f"\n--- Single-condition guidance (gamma={gamma}) ---")
    for name in PROPERTIES:
        samples = generate_guided_single(
            flow_model, classifiers[name], n_samples, K, L, num_steps, gamma, mode
        )
        rate = prop_fns[name](samples).float().mean().item()
        print(f"  {name}: {rate*100:.1f}%")

    # AND composition: starts_with_0 AND contains_pattern
    print(f"\n--- AND(starts_with_0, contains_pattern) ---")
    samples_and = generate_composed_and(
        flow_model,
        [classifiers["starts_with_0"], classifiers["contains_pattern"]],
        n_samples, K, L, num_steps, [gamma, gamma], mode
    )
    acc_and = evaluate_boolean_accuracy(
        samples_and,
        [prop_fns["starts_with_0"],
         partial(prop_fns["contains_pattern"], pattern=[1, 2, 3])
         if "pattern" in prop_fns["contains_pattern"].__code__.co_varnames
         else prop_fns["contains_pattern"]],
        mode="and"
    )
    rate_a = prop_fns["starts_with_0"](samples_and).float().mean().item()
    rate_b = prop_fns["contains_pattern"](samples_and).float().mean().item()
    print(f"  starts_with_0: {rate_a*100:.1f}%")
    print(f"  contains_pattern: {rate_b*100:.1f}%")
    print(f"  BOTH (AND accuracy): {acc_and*100:.1f}%")

    # NOT composition: starts_with_0 AND NOT no_repeats
    print(f"\n--- NOT(starts_with_0, no_repeats) = starts_with_0 AND NOT no_repeats ---")
    samples_not = generate_composed_not(
        flow_model,
        classifiers["starts_with_0"], classifiers["no_repeats"],
        n_samples, K, L, num_steps, gamma, gamma, mode
    )
    acc_not = evaluate_boolean_accuracy(
        samples_not,
        [prop_fns["starts_with_0"], prop_fns["no_repeats"]],
        mode="not"
    )
    rate_a = prop_fns["starts_with_0"](samples_not).float().mean().item()
    rate_b = prop_fns["no_repeats"](samples_not).float().mean().item()
    print(f"  starts_with_0: {rate_a*100:.1f}%")
    print(f"  no_repeats (should be LOW): {rate_b*100:.1f}%")
    print(f"  A ∧ ¬B accuracy: {acc_not*100:.1f}%")

    # Kill gate
    print(f"\n=== KILL GATE ===")
    print(f"  AND accuracy: {acc_and*100:.1f}% (target: >80%)")
    print(f"  NOT accuracy: {acc_not*100:.1f}% (target: >80%)")
    if acc_and >= 0.8 and acc_not >= 0.8:
        print("  ✓ PASS — proceed to Phase 2")
    elif acc_and >= 0.6 or acc_not >= 0.6:
        print("  ~ PARTIAL — diagnose weak operator, tune gamma")
    else:
        print("  ✗ FAIL — operators broken, investigate math vs implementation")

    return {
        "and_accuracy": acc_and,
        "not_accuracy": acc_not,
        "mode": mode,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic.yaml")
    args = parser.parse_args()
    results = run_full_experiment(args.config)
```

- [ ] **Step 2: Run full test suite**

Run: `pytest -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/experiments/run_synthetic.py
git commit -m "feat: full experiment runner with kill-gate evaluation"
```

---

### Task 9: Smoke Test — Train + Evaluate (Reduced Scale)

- [ ] **Step 1: Create a fast smoke-test config**

Create `configs/smoke.yaml`:

```yaml
data:
  K: 4
  L: 8
  n_train: 1000
  n_val: 200

model:
  hidden_dim: 32
  num_layers: 2
  time_embed_dim: 32

training:
  epochs: 10
  lr: 0.001
  batch_size: 64
  flow_type: prob_path

guidance:
  gamma: 2.0
  num_steps: 20
  num_samples: 200

properties:
  - name: starts_with_0
  - name: contains_pattern
  - name: no_repeats
```

- [ ] **Step 2: Run smoke test**

Run: `python -m src.experiments.run_synthetic --config configs/smoke.yaml`
Expected: Completes without error. Accuracy numbers will be low (undertrained) but pipeline runs end-to-end.

- [ ] **Step 3: Fix any runtime errors discovered during smoke test**

Address issues as they arise — this step validates the full pipeline integration.

- [ ] **Step 4: Commit fixes if any**

```bash
git add -A
git commit -m "fix: runtime issues discovered in smoke test"
```

- [ ] **Step 5: Run full-scale synthetic experiment**

Run: `python -m src.experiments.run_synthetic --config configs/synthetic.yaml`
Expected: ~20-40 minutes on CPU. AND accuracy and NOT accuracy are the kill-gate metrics.

- [ ] **Step 6: Commit results and update CLAUDE.md**

```bash
git add -A
git commit -m "feat: phase 1 synthetic validation complete"
```

---

## Self-Review Checklist

- [x] Spec Phase 1 coverage: CTMC flow ✓, prob-path flow ✓, 3 classifiers ✓, AND/NOT/OR operators ✓, synthetic data (K=8, L=32) ✓, evaluation metrics ✓, kill gate ✓
- [x] No placeholders — all code blocks complete
- [x] Type consistency: `CTMCDenoiser`, `ProbPathDenoiser`, `TimeConditionalClassifier` used consistently throughout
- [x] Function signatures match across files: `generate_unconditional(model, n, K, L, num_steps, mode)` etc.
- [x] All test files have concrete assertions
- [x] OR operator implemented in compose.py (spec requirement)
- [x] Both CTMC and prob-path sampling paths in all generate functions
- [x] Tau-leaping (CTMC) and Euler step (prob-path) samplers implemented
- [x] Task 2 from spec (graph generation) deferred — not needed for kill gate
- [x] Task 3 from spec (scaling test K→100) deferred — run after kill gate passes
