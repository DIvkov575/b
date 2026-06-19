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
