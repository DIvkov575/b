import torch

from src.training.losses import classification_loss, dpp_margin_loss


def test_classification_loss():
    torch.manual_seed(0)
    logits = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 1])
    loss = classification_loss(logits, labels)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_dpp_margin_loss_rewards_positive_margin():
    quality_scores = torch.tensor([2.0, 1.5, -1.0, -2.0])
    actual_margins = torch.tensor([2.0, 1.5, -1.0, -2.0])

    good_loss = dpp_margin_loss(quality_scores, actual_margins)

    bad_actual = torch.tensor([-2.0, -1.5, 1.0, 2.0])
    bad_loss = dpp_margin_loss(quality_scores, bad_actual)

    assert good_loss.item() < bad_loss.item()
