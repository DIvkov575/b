import torch
from src.models.mdlm import MDLM
from src.training.estimate_it import estimate_mutual_info_curve


def test_estimate_it_shape():
    model = MDLM(vocab_size=28, seq_len=32, hidden_dim=32, num_layers=1, num_heads=2)
    data = torch.randint(1, 28, (100, 32))
    it_curve = estimate_mutual_info_curve(model, data, n_timesteps=20, n_samples=50)
    assert it_curve.shape == (20,)
    assert (it_curve >= 0).all()


def test_estimate_it_boundary():
    """I(t) should be ~0 at t near 1 (fully masked, no info between positions)."""
    model = MDLM(vocab_size=28, seq_len=32, hidden_dim=32, num_layers=1, num_heads=2)
    data = torch.randint(1, 28, (100, 32))
    it_curve = estimate_mutual_info_curve(model, data, n_timesteps=20, n_samples=50)
    # Last bin (t near 1) should have low I(t) since both versions are heavily masked
    assert it_curve[-1] <= it_curve.mean() + 0.1
