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
    model = ConditionalProbPathFlow(K=8, L=32, n_conditions=3, hidden_dim=64, num_layers=2)
    x_t = torch.randint(0, 8, (4, 32))
    t = torch.tensor([0.5, 0.3, 0.8, 0.1])
    null_cond = torch.zeros(4, 3)
    logits = model(x_t, t, null_cond)
    assert logits.shape == (4, 32, 8)


def test_conditional_flow_cfg_dropout():
    model = ConditionalProbPathFlow(K=8, L=32, n_conditions=3, hidden_dim=64, num_layers=2)
    x_t = torch.randint(0, 8, (2, 32))
    t = torch.tensor([0.5, 0.5])
    cond = torch.ones(2, 3)
    model.train()
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
