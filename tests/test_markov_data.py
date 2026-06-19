import torch
from src.data.synthetic import (
    MarkovSequenceDataset,
    property_starts_with_0,
    property_contains_pattern,
    property_no_adjacent_repeats,
)


def test_markov_dataset_shape():
    ds = MarkovSequenceDataset(n=500, K=8, L=32)
    seq, labels = ds[0]
    assert seq.shape == (32,)
    assert isinstance(labels, dict)
    assert "starts_with_0" in labels
    assert "contains_pattern" in labels
    assert "no_repeats" in labels


def test_markov_dataset_structure():
    """Markov sequences should NOT be uniform — they have transition structure."""
    ds = MarkovSequenceDataset(n=5000, K=8, L=32)
    seqs = ds.seqs
    same_as_prev = (seqs[:, 1:] == seqs[:, :-1]).float().mean().item()
    assert same_as_prev > 0.15 or same_as_prev < 0.10


def test_markov_property_rates():
    """Non-trivial rates for properties compatible with self-biased chain.
    `no_repeats` is excluded — the transition matrix has dominant self-bias by
    design, so 32 consecutive distinct draws is vanishingly rare.
    """
    ds = MarkovSequenceDataset(n=5000, K=8, L=32)
    for name in ("starts_with_0", "contains_pattern"):
        rate = ds.labels[name].float().mean().item()
        assert 0.01 < rate < 0.99, f"{name} rate {rate} is too extreme"


def test_markov_conditional_labels():
    ds = MarkovSequenceDataset(n=100, K=8, L=32)
    seq, labels = ds[0]
    assert all(isinstance(v, bool) for v in labels.values())
