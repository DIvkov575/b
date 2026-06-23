import torch
import torch.nn.functional as F
from src.data.text_data import mask_sequence, MASK_TOKEN
from src.models.mdlm import MDLM


@torch.no_grad()
def estimate_mutual_info_curve(
    model: MDLM, data: torch.Tensor, n_timesteps: int = 100,
    n_samples: int = 200, device: str = "cpu"
) -> torch.Tensor:
    """Estimate I(t) = Sum_{i!=j} I(x^i_t; x^j_t | x^{-(i,j)}_t) across timesteps.

    Approximation: for each timestep t, measure how much the model's prediction
    entropy decreases when context is less masked (more positions revealed).
    High decrease = high inter-position mutual information at that timestep.
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

        # Entropy at each position: H(x_i | x_t, t)
        log_probs = model.score(x_t, t)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)  # (B, L)

        # Compare with more-masked version (less context available)
        t_more = torch.full((B,), min(t_val.item() + 0.1, 0.99), device=device)
        x_t_more = mask_sequence(data, t_more, mask_token=MASK_TOKEN)
        log_probs_more = model.score(x_t_more, t_more)
        probs_more = log_probs_more.exp()
        entropy_more = -(probs_more * log_probs_more).sum(dim=-1)

        # I(t) proxy: average entropy reduction from having more context
        info_gain = (entropy_more - entropy).clamp(min=0).mean()
        it_curve[idx] = info_gain.item()

    return it_curve
