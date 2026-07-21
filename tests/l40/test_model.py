import torch

from src.l40.model import ProteinBERT, create_model


def test_forward_returns_logits_of_expected_shape():
    model = ProteinBERT(vocab_size=24, d_model=16, n_layers=2, n_heads=2, d_ff=32, max_length=20)
    input_ids = torch.randint(1, 24, (3, 20))
    attention_mask = torch.ones(3, 20)

    out = model(input_ids, attention_mask)

    assert out['logits'].shape == (3, 20, 24)
    assert out['loss'] is None


def test_forward_with_labels_computes_finite_loss():
    model = ProteinBERT(vocab_size=24, d_model=16, n_layers=2, n_heads=2, d_ff=32, max_length=20)
    input_ids = torch.randint(1, 24, (3, 20))
    attention_mask = torch.ones(3, 20)
    labels = torch.full((3, 20), -100, dtype=torch.long)
    labels[:, 0] = 5

    out = model(input_ids, attention_mask, labels)

    assert torch.isfinite(out['loss'])


def test_create_model_returns_proteinbert_instance():
    model = create_model(vocab_size=24, d_model=16, n_layers=2, n_heads=2, d_ff=32, max_length=20)
    assert isinstance(model, ProteinBERT)
