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
    Training should concentrate where I(t) is large.
    """
    weights = it_values.clamp(min=1e-8)
    return weights / weights.sum()


def sample_timesteps(batch_size: int, weights: torch.Tensor,
                     device: str = "cpu") -> torch.Tensor:
    """Sample timesteps from a weighted distribution over [0, 1].
    Uses the schedule weights to define a piecewise-constant density.
    """
    n_bins = len(weights)
    bin_idx = torch.multinomial(weights, batch_size, replacement=True)
    bin_width = 1.0 / n_bins
    t = (bin_idx.float() + torch.rand(batch_size)) * bin_width
    return t.to(device).clamp(1e-5, 1.0 - 1e-5)
