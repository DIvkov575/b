import torch
from src.models.mdlm import MDLM


def test_mdlm_forward():
    model = MDLM(vocab_size=28, seq_len=128, hidden_dim=64, num_layers=2, num_heads=4)
    x_t = torch.randint(0, 28, (4, 128))
    t = torch.tensor([0.5, 0.3, 0.8, 0.1])
    logits = model(x_t, t)
    assert logits.shape == (4, 128, 28)


def test_mdlm_score():
    model = MDLM(vocab_size=28, seq_len=128, hidden_dim=64, num_layers=2, num_heads=4)
    x_t = torch.randint(0, 28, (2, 128))
    t = torch.tensor([0.5, 0.5])
    log_probs = model.score(x_t, t)
    assert log_probs.shape == (2, 128, 28)
    probs = log_probs.exp()
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=0.01)
