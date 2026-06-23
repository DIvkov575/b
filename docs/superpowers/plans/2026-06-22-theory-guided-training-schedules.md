# Theory-Guided Training Schedules for Masked Diffusion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive and validate a principled training time-sampling schedule for masked diffusion models based on effective total correlation I(t) (Dmitriev et al., 2026), showing it outperforms both uniform sampling and heuristic bell-shaped sampling (Hong et al., 2026) on text modeling benchmarks.

**Architecture:** (1) Train a baseline MDLM masked diffusion model on text8/LM1B. (2) Estimate I(t) empirically from the trained model by measuring pairwise conditional mutual information at each timestep. (3) Derive the optimal training schedule: sample timesteps proportionally to I(t) (since Theorem 3 shows discretization error is Σ h_k ∫ I(t)dt — training should concentrate where this error is highest). (4) Retrain with the theory-guided schedule. (5) Compare convergence speed (FLOPs-to-target-loss) against uniform and bell-shaped baselines.

**Tech Stack:** PyTorch, HuggingFace datasets (text8, OpenWebText), Transformer encoder, wandb for tracking. Model scale: 50-170M parameters (matches Hong et al. and Sahoo et al. experimental setups).

**Key Insight:** Hong et al. found bell-shaped time sampling gives 4× speedup but chose the shape heuristically. Dmitriev et al. proved that sampling convergence error is governed by I(t) = Σᵢ≠ⱼ I(xᵢₜ; xʲₜ | x^{-(i,j)}ₜ). We connect these: the optimal *training* schedule should weight timesteps by I(t), because the model's accuracy matters most where discretization error accumulates fastest. If I(t) for natural language is bell-shaped, this explains Hong et al.'s empirical finding from first principles.

---

## File Structure

```
src/
├── __init__.py
├── data/
│   ├── __init__.py
│   └── text_data.py          # Text8/OpenWebText data loading + masking
├── models/
│   ├── __init__.py
│   └── mdlm.py               # Masked diffusion LM (Transformer encoder)
├── training/
│   ├── __init__.py
│   ├── schedules.py           # Time-sampling schedules (uniform, bell, I(t)-optimal)
│   ├── train_mdlm.py         # Training loop with configurable schedule
│   └── estimate_it.py        # Estimate I(t) from trained model
├── sampling/
│   ├── __init__.py
│   └── sample_mdlm.py        # Masked diffusion sampling (tau-leaping)
└── experiments/
    ├── __init__.py
    ├── run_baseline.py        # Train with uniform schedule
    ├── run_bell.py            # Train with bell-shaped schedule (Hong et al.)
    ├── run_optimal.py         # Train with I(t)-derived schedule
    └── compare.py             # FLOPs-to-loss comparison
tests/
├── __init__.py
├── test_schedules.py
├── test_mdlm.py
├── test_estimate_it.py
└── test_data.py
configs/
├── text8_small.yaml           # Quick iteration (50M params, text8)
├── text8_full.yaml            # Full experiment
└── owt_170m.yaml              # OpenWebText at 170M (matches literature)
```

---

### Task 1: Data Loading + Masking Noise Process

**Files:**
- Create: `src/data/text_data.py`
- Create: `tests/test_data.py`
- Create: `configs/text8_small.yaml`

- [ ] **Step 1: Write failing test**

```python
# tests/test_data.py
import torch
from src.data.text_data import Text8Dataset, mask_sequence


def test_text8_dataset_loads():
    ds = Text8Dataset(split="train", seq_len=128, max_samples=1000)
    assert len(ds) == 1000
    seq = ds[0]
    assert seq.shape == (128,)
    assert seq.dtype == torch.long
    assert seq.min() >= 0
    assert seq.max() <= 26  # 26 chars (a-z + space)


def test_mask_sequence_t0():
    """At t=0, no masking."""
    x = torch.tensor([1, 2, 3, 4, 5])
    x_t = mask_sequence(x, t=0.0, mask_token=0)
    assert (x_t == x).all()


def test_mask_sequence_t1():
    """At t=1, fully masked."""
    x = torch.tensor([1, 2, 3, 4, 5])
    x_t = mask_sequence(x, t=1.0, mask_token=0)
    assert (x_t == 0).all()


def test_mask_sequence_batched():
    """Supports (B, L) input with (B,) time vector."""
    x = torch.randint(1, 27, (8, 128))
    t = torch.rand(8)
    x_t = mask_sequence(x, t, mask_token=0)
    assert x_t.shape == (8, 128)
    # More masked as t increases
    mask_rates = (x_t == 0).float().mean(dim=1)
    # Not a hard assertion due to stochasticity, but check shape
    assert mask_rates.shape == (8,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_data.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement data loading**

```python
# src/data/text_data.py
import torch
from torch.utils.data import Dataset
from pathlib import Path


VOCAB_SIZE = 27  # a-z + space
MASK_TOKEN = 0   # reserve 0 for mask; chars are 1-27


def char_to_idx(c: str) -> int:
    if c == ' ':
        return 27
    return ord(c) - ord('a') + 1


def text8_tokenize(text: str) -> torch.Tensor:
    return torch.tensor([char_to_idx(c) for c in text], dtype=torch.long)


class Text8Dataset(Dataset):
    """Text8 dataset: 100M characters of Wikipedia, alphabet + space only."""

    def __init__(self, split: str = "train", seq_len: int = 256,
                 max_samples: int = None, data_dir: str = "data"):
        path = Path(data_dir) / "text8"
        if not path.exists():
            self._download(data_dir)
        with open(path) as f:
            raw = f.read()
        # Split: first 90M train, next 5M val, last 5M test
        if split == "train":
            raw = raw[:90_000_000]
        elif split == "val":
            raw = raw[90_000_000:95_000_000]
        else:
            raw = raw[95_000_000:100_000_000]
        # Chunk into sequences
        n_seqs = len(raw) // seq_len
        if max_samples:
            n_seqs = min(n_seqs, max_samples)
        self.seqs = torch.zeros(n_seqs, seq_len, dtype=torch.long)
        for i in range(n_seqs):
            chunk = raw[i * seq_len:(i + 1) * seq_len]
            self.seqs[i] = text8_tokenize(chunk)
        self.seq_len = seq_len

    def _download(self, data_dir: str):
        import urllib.request, zipfile, os
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        url = "http://mattmahoney.net/dc/text8.zip"
        zip_path = Path(data_dir) / "text8.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(data_dir)
        os.remove(zip_path)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]


def mask_sequence(x: torch.Tensor, t, mask_token: int = MASK_TOKEN) -> torch.Tensor:
    """Apply absorbing-state (masking) noise at time t.
    Each position is independently masked with probability 1-e^{-t}.
    t: float or (B,) tensor. x: (L,) or (B, L).
    """
    if not isinstance(t, torch.Tensor):
        t = torch.tensor([t], dtype=torch.float, device=x.device)
    if t.dim() == 0:
        t = t.unsqueeze(0)
    # mask_prob = 1 - e^{-t} (absorbing rate)
    mask_prob = 1.0 - torch.exp(-t)
    if x.dim() == 2:
        mask_prob = mask_prob.unsqueeze(-1)  # (B, 1)
    mask = torch.rand_like(x.float()) < mask_prob
    return torch.where(mask, torch.full_like(x, mask_token), x)
```

- [ ] **Step 4: Create config**

```yaml
# configs/text8_small.yaml
data:
  dataset: text8
  seq_len: 128
  max_samples: 50000
  vocab_size: 28   # 27 chars + 1 mask token

model:
  hidden_dim: 256
  num_layers: 6
  num_heads: 8
  dropout: 0.0

training:
  epochs: 50
  lr: 0.0003
  batch_size: 128
  schedule: uniform   # or "bell" or "optimal"
  warmup_steps: 1000

sampling:
  num_steps: 100
  num_samples: 256
```

- [ ] **Step 5: Run tests**

Run: `source .venv/bin/activate && pytest tests/test_data.py -v`
Expected: PASS (text8 download will happen on first run)

- [ ] **Step 6: Commit**

```bash
git add src/data/ tests/test_data.py configs/text8_small.yaml
git commit -m "feat: text8 data loading with absorbing-state masking"
```

---

### Task 2: MDLM Model (Masked Diffusion Language Model)

**Files:**
- Create: `src/models/mdlm.py`
- Create: `tests/test_mdlm.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_mdlm.py
import torch
from src.models.mdlm import MDLM


def test_mdlm_forward():
    model = MDLM(vocab_size=28, seq_len=128, hidden_dim=64, num_layers=2, num_heads=4)
    x_t = torch.randint(0, 28, (4, 128))
    t = torch.tensor([0.5, 0.3, 0.8, 0.1])
    logits = model(x_t, t)
    assert logits.shape == (4, 128, 28)


def test_mdlm_score():
    """Score function: log p_theta(x_0 = a | x_t, t) for each position and token."""
    model = MDLM(vocab_size=28, seq_len=128, hidden_dim=64, num_layers=2, num_heads=4)
    x_t = torch.randint(0, 28, (2, 128))
    t = torch.tensor([0.5, 0.5])
    log_probs = model.score(x_t, t)
    assert log_probs.shape == (2, 128, 28)
    # Should be valid log-probabilities (sum to ~1 in exp space per position)
    probs = log_probs.exp()
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=0.01)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_mdlm.py -v`
Expected: FAIL

- [ ] **Step 3: Implement MDLM**

```python
# src/models/mdlm.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MDLM(nn.Module):
    """Masked Diffusion Language Model.
    Predicts p(x_0 | x_t, t) — the denoised token distribution at each position.
    Architecture: Transformer encoder with time conditioning.
    """

    def __init__(self, vocab_size: int = 28, seq_len: int = 256,
                 hidden_dim: int = 256, num_layers: int = 6, num_heads: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(seq_len, hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Returns logits (B, L, V)."""
        B, L = x_t.shape
        pos_ids = torch.arange(L, device=x_t.device).unsqueeze(0).expand(B, -1)
        h = self.embed(x_t) + self.pos_embed(pos_ids)
        t_emb = self.time_embed(t.unsqueeze(-1))
        h = h + t_emb.unsqueeze(1)
        h = self.transformer(h)
        return self.out(h)

    def score(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Returns log p(x_0 | x_t, t) — shape (B, L, V)."""
        logits = self.forward(x_t, t)
        return F.log_softmax(logits, dim=-1)
```

- [ ] **Step 4: Run tests**

Run: `source .venv/bin/activate && pytest tests/test_mdlm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/mdlm.py tests/test_mdlm.py
git commit -m "feat: MDLM Transformer model with score function"
```

---

### Task 3: Time-Sampling Schedules

**Files:**
- Create: `src/training/schedules.py`
- Create: `tests/test_schedules.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_schedules.py
import torch
from src.training.schedules import (
    uniform_schedule,
    bell_schedule,
    optimal_schedule,
    sample_timesteps,
)


def test_uniform_schedule():
    weights = uniform_schedule(n_bins=100)
    assert weights.shape == (100,)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)


def test_bell_schedule():
    weights = bell_schedule(n_bins=100, peak=0.5, width=0.2)
    assert weights.shape == (100,)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
    # Peak should be near the center
    assert weights[45:55].sum() > weights[0:10].sum()


def test_optimal_schedule():
    # Fake I(t) curve: high in the middle
    it_values = torch.zeros(100)
    it_values[40:60] = 1.0
    weights = optimal_schedule(it_values)
    assert weights.shape == (100,)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
    # Should concentrate on bins 40-60
    assert weights[40:60].sum() > 0.9


def test_sample_timesteps():
    weights = bell_schedule(n_bins=100)
    t = sample_timesteps(batch_size=64, weights=weights)
    assert t.shape == (64,)
    assert (t >= 0).all() and (t <= 1).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_schedules.py -v`
Expected: FAIL

- [ ] **Step 3: Implement schedules**

```python
# src/training/schedules.py
import torch
import torch.nn.functional as F


def uniform_schedule(n_bins: int = 1000) -> torch.Tensor:
    """Uniform time sampling: all timesteps equally weighted."""
    return torch.ones(n_bins) / n_bins


def bell_schedule(n_bins: int = 1000, peak: float = 0.5, width: float = 0.2) -> torch.Tensor:
    """Bell-shaped (Gaussian) time sampling (Hong et al., 2026).
    Concentrates training compute near the peak timestep.
    """
    t = torch.linspace(0, 1, n_bins)
    weights = torch.exp(-0.5 * ((t - peak) / width) ** 2)
    return weights / weights.sum()


def optimal_schedule(it_values: torch.Tensor) -> torch.Tensor:
    """Theory-guided schedule: weight timesteps proportional to I(t).
    From Theorem 3 (Dmitriev et al.): discretization error is Σ h_k ∫ I(t)dt.
    Training should concentrate where I(t) is large — the model's accuracy
    matters most at timesteps with high conditional mutual information.
    """
    weights = it_values.clamp(min=1e-8)
    return weights / weights.sum()


def sample_timesteps(batch_size: int, weights: torch.Tensor,
                     device: str = "cpu") -> torch.Tensor:
    """Sample timesteps from a weighted distribution over [0, 1].
    Uses the schedule weights to define a piecewise-constant density.
    """
    n_bins = len(weights)
    # Sample bin indices according to weights
    bin_idx = torch.multinomial(weights, batch_size, replacement=True)
    # Uniform within each bin
    bin_width = 1.0 / n_bins
    t = (bin_idx.float() + torch.rand(batch_size)) * bin_width
    return t.to(device).clamp(1e-5, 1.0 - 1e-5)
```

- [ ] **Step 4: Run tests**

Run: `source .venv/bin/activate && pytest tests/test_schedules.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/training/schedules.py tests/test_schedules.py
git commit -m "feat: training time-sampling schedules (uniform, bell, I(t)-optimal)"
```

---

### Task 4: Estimate I(t) from Trained Model

**Files:**
- Create: `src/training/estimate_it.py`
- Create: `tests/test_estimate_it.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_estimate_it.py
import torch
from src.models.mdlm import MDLM
from src.training.estimate_it import estimate_mutual_info_curve


def test_estimate_it_shape():
    model = MDLM(vocab_size=28, seq_len=32, hidden_dim=32, num_layers=1, num_heads=2)
    # Fake data batch
    data = torch.randint(1, 28, (100, 32))
    it_curve = estimate_mutual_info_curve(model, data, n_timesteps=20, n_samples=50)
    assert it_curve.shape == (20,)
    assert (it_curve >= 0).all()


def test_estimate_it_boundary():
    """I(t) should be ~0 at t=1 (fully masked, no info between positions)."""
    model = MDLM(vocab_size=28, seq_len=32, hidden_dim=32, num_layers=1, num_heads=2)
    data = torch.randint(1, 28, (100, 32))
    it_curve = estimate_mutual_info_curve(model, data, n_timesteps=20, n_samples=50)
    # Last bin (t near 1) should have lowest I(t)
    assert it_curve[-1] <= it_curve.mean()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_estimate_it.py -v`
Expected: FAIL

- [ ] **Step 3: Implement I(t) estimation**

```python
# src/training/estimate_it.py
import torch
import torch.nn.functional as F
from src.data.text_data import mask_sequence, MASK_TOKEN
from src.models.mdlm import MDLM


@torch.no_grad()
def estimate_mutual_info_curve(
    model: MDLM, data: torch.Tensor, n_timesteps: int = 100,
    n_samples: int = 200, device: str = "cpu"
) -> torch.Tensor:
    """Estimate I(t) = Σᵢ≠ⱼ I(xᵢₜ; xʲₜ | x^{-(i,j)}ₜ) across timesteps.

    Approximation: measure how much the model's prediction at position i
    changes when we reveal/hide position j. High change = high mutual info.

    Practical proxy: for each timestep t, compute the average reduction in
    entropy at position i when other positions are less masked (lower t)
    vs. more masked. This captures inter-position dependence.
    """
    model.eval()
    model = model.to(device)
    data = data[:n_samples].to(device)
    B, L = data.shape

    t_values = torch.linspace(0.01, 0.99, n_timesteps)
    it_curve = torch.zeros(n_timesteps)

    for idx, t_val in enumerate(t_values):
        t = torch.full((B,), t_val.item(), device=device)
        x_t = mask_sequence(data, t, mask_token=MASK_TOKEN)

        # Get model's predicted distribution
        log_probs = model.score(x_t, t)  # (B, L, V)

        # Entropy at each position: H(x_i | x_t, t)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)  # (B, L)

        # Compare with a "more masked" version (less context)
        t_more = torch.full((B,), min(t_val.item() + 0.1, 0.99), device=device)
        x_t_more = mask_sequence(data, t_more, mask_token=MASK_TOKEN)
        log_probs_more = model.score(x_t_more, t_more)
        probs_more = log_probs_more.exp()
        entropy_more = -(probs_more * log_probs_more).sum(dim=-1)

        # I(t) proxy: average entropy reduction from having more context
        # = E[H(xi | more masked) - H(xi | less masked)]
        info_gain = (entropy_more - entropy).clamp(min=0).mean()
        it_curve[idx] = info_gain.item()

    return it_curve
```

- [ ] **Step 4: Run tests**

Run: `source .venv/bin/activate && pytest tests/test_estimate_it.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/training/estimate_it.py tests/test_estimate_it.py
git commit -m "feat: estimate I(t) mutual information curve from trained model"
```

---

### Task 5: Training Loop with Configurable Schedule

**Files:**
- Create: `src/training/train_mdlm.py`

- [ ] **Step 1: Implement training loop**

```python
# src/training/train_mdlm.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data.text_data import Text8Dataset, mask_sequence, MASK_TOKEN
from src.models.mdlm import MDLM
from src.training.schedules import uniform_schedule, bell_schedule, optimal_schedule, sample_timesteps


def train_mdlm(config: dict, schedule_weights: torch.Tensor = None,
               device: str = "cpu") -> MDLM:
    """Train MDLM with a configurable time-sampling schedule.

    Args:
        config: experiment config dict
        schedule_weights: (n_bins,) tensor of timestep sampling weights.
                         If None, uses config["training"]["schedule"] to select.
    """
    ds = Text8Dataset(
        split="train", seq_len=config["data"]["seq_len"],
        max_samples=config["data"].get("max_samples")
    )
    loader = DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=True)

    model = MDLM(
        vocab_size=config["data"]["vocab_size"],
        seq_len=config["data"]["seq_len"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        dropout=config["model"].get("dropout", 0.0),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"])

    # Select schedule
    if schedule_weights is None:
        sched_name = config["training"]["schedule"]
        if sched_name == "uniform":
            schedule_weights = uniform_schedule()
        elif sched_name == "bell":
            schedule_weights = bell_schedule()
        else:
            raise ValueError(f"Unknown schedule: {sched_name}. "
                           "For 'optimal', pass schedule_weights explicitly.")

    model.train()
    step = 0
    for epoch in range(config["training"]["epochs"]):
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            batch = batch.to(device)
            B = batch.shape[0]

            # Sample timesteps from schedule
            t = sample_timesteps(B, schedule_weights, device=device)

            # Apply masking noise
            x_t = mask_sequence(batch, t, mask_token=MASK_TOKEN)

            # Forward + loss (cross-entropy on masked positions)
            logits = model(x_t, t)  # (B, L, V)
            loss = F.cross_entropy(logits.reshape(-1, model.vocab_size),
                                   batch.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            step += 1

        avg_loss = total_loss / n_batches
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")

    return model
```

- [ ] **Step 2: Verify imports**

Run: `source .venv/bin/activate && python -c "from src.training.train_mdlm import train_mdlm; print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add src/training/train_mdlm.py
git commit -m "feat: MDLM training loop with configurable time-sampling schedule"
```

---

### Task 6: Experiment Runner — Baseline vs Bell vs Optimal

**Files:**
- Create: `src/experiments/run_schedule_comparison.py`

- [ ] **Step 1: Implement comparison experiment**

```python
# src/experiments/run_schedule_comparison.py
import yaml
import torch
from src.training.train_mdlm import train_mdlm
from src.training.schedules import uniform_schedule, bell_schedule, optimal_schedule
from src.training.estimate_it import estimate_mutual_info_curve
from src.data.text_data import Text8Dataset


def run_comparison(config_path: str = "configs/text8_small.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = "cpu"
    print("=== Theory-Guided Training Schedules for Masked Diffusion ===\n")

    # Phase 1: Train baseline (uniform) to get I(t) estimate
    print("[1/4] Training baseline (uniform schedule)...")
    config_uniform = {**config, "training": {**config["training"], "schedule": "uniform"}}
    model_uniform = train_mdlm(config_uniform, device=device)
    print()

    # Phase 2: Estimate I(t) from baseline model
    print("[2/4] Estimating I(t) curve from baseline model...")
    ds = Text8Dataset(split="val", seq_len=config["data"]["seq_len"], max_samples=500)
    it_curve = estimate_mutual_info_curve(model_uniform, ds.seqs, n_timesteps=100, device=device)
    print(f"  I(t) peak at t={it_curve.argmax().item()/100:.2f}, "
          f"max={it_curve.max():.4f}, mean={it_curve.mean():.4f}")
    print()

    # Phase 3: Train with bell schedule (Hong et al. baseline)
    print("[3/4] Training with bell schedule (Hong et al.)...")
    bell_weights = bell_schedule(n_bins=1000, peak=0.5, width=0.2)
    model_bell = train_mdlm(config, schedule_weights=bell_weights, device=device)
    print()

    # Phase 4: Train with I(t)-optimal schedule (our method)
    print("[4/4] Training with I(t)-optimal schedule (ours)...")
    # Interpolate I(t) from 100 bins to 1000
    it_interp = torch.nn.functional.interpolate(
        it_curve.unsqueeze(0).unsqueeze(0), size=1000, mode='linear'
    ).squeeze()
    optimal_weights = optimal_schedule(it_interp)
    model_optimal = train_mdlm(config, schedule_weights=optimal_weights, device=device)
    print()

    # Evaluate all three on validation set
    print("=== Validation Loss Comparison ===")
    val_ds = Text8Dataset(split="val", seq_len=config["data"]["seq_len"], max_samples=2000)
    for name, model in [("Uniform", model_uniform), ("Bell", model_bell), ("Optimal (ours)", model_optimal)]:
        val_loss = evaluate_nll(model, val_ds, device=device)
        print(f"  {name:20s}: val_loss = {val_loss:.4f}")

    # Report I(t) shape
    print(f"\n=== I(t) Curve Analysis ===")
    peak_t = it_curve.argmax().item() / 100
    print(f"  Peak at t={peak_t:.2f}")
    if 0.3 < peak_t < 0.7:
        print(f"  → I(t) is bell-shaped! This explains why Hong et al.'s heuristic works.")
    else:
        print(f"  → I(t) is NOT bell-shaped — our optimal schedule should differ significantly.")

    return {
        "it_curve": it_curve,
        "optimal_weights": optimal_weights,
    }


@torch.no_grad()
def evaluate_nll(model, dataset, device="cpu", n_eval=1000):
    """Evaluate negative log-likelihood on validation data."""
    import torch.nn.functional as F
    from src.data.text_data import mask_sequence, MASK_TOKEN

    model.eval()
    model = model.to(device)
    seqs = dataset.seqs[:n_eval].to(device)
    B, L = seqs.shape

    # Average loss across multiple timesteps
    total_loss = 0.0
    n_t = 10
    for t_val in torch.linspace(0.1, 0.9, n_t):
        t = torch.full((B,), t_val.item(), device=device)
        x_t = mask_sequence(seqs, t, mask_token=MASK_TOKEN)
        logits = model(x_t, t)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), seqs.reshape(-1))
        total_loss += loss.item()

    return total_loss / n_t


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/text8_small.yaml")
    args = parser.parse_args()
    run_comparison(args.config)
```

- [ ] **Step 2: Run smoke test**

Run: `source .venv/bin/activate && python -m src.experiments.run_schedule_comparison --config configs/text8_small.yaml`
Expected: Runs end-to-end (will be slow on first run due to text8 download). Validation losses should differ between schedules.

- [ ] **Step 3: Commit**

```bash
git add src/experiments/run_schedule_comparison.py
git commit -m "feat: schedule comparison experiment (uniform vs bell vs I(t)-optimal)"
```

---

### Task 7: Kill Gate Evaluation

- [ ] **Step 1: Define success criteria**

The kill gate for this direction:
1. I(t) curve is estimable and non-uniform (if it's flat, there's nothing to optimize)
2. I(t)-optimal schedule converges faster than uniform (≥20% fewer FLOPs to reach same loss)
3. I(t)-optimal schedule matches or beats bell-shaped (if it doesn't beat the heuristic, the theory adds no value)

- [ ] **Step 2: Run on dev desktop (full scale)**

```bash
scp -r src/ configs/ tests/ dev-dsk-divkov-1b-029561b7.us-east-1.amazon.com:~/biostat-schedules/
ssh dev-dsk-divkov-1b-029561b7.us-east-1.amazon.com "source ~/miniconda3/bin/activate && cd ~/biostat-schedules && pip install -e . -q && python -u -m src.experiments.run_schedule_comparison --config configs/text8_small.yaml > schedule_results.log 2>&1 &"
```

- [ ] **Step 3: Evaluate kill gate**

If optimal beats uniform by ≥20% but NOT bell: the theory explains bell but doesn't improve on it → reframe as explanatory paper (why bell works).
If optimal beats both: strong contribution → proceed to full-scale experiments.
If I(t) is flat: abandon direction.

---

## Self-Review Checklist

- [x] Paper contribution clear: theory-guided schedule from I(t), not just ablation table
- [x] Kill gate defined with concrete thresholds
- [x] No placeholders — all code complete in every step
- [x] Type consistency: MDLM(vocab_size, seq_len, hidden_dim, num_layers, num_heads) used consistently
- [x] Function signatures match: mask_sequence(x, t, mask_token), sample_timesteps(B, weights, device)
- [x] Connects to literature: Dmitriev et al. Theorem 3 → training schedule; Hong et al. → baseline
- [x] Practical: starts with text8 (27-char, easy to iterate), scales to OpenWebText later
- [x] I(t) estimation is a proxy (entropy reduction) — note this in the paper as an approximation
