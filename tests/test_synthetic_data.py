import torch
from src.data.synthetic import (
    generate_uniform_sequences,
    property_starts_with_0,
    property_contains_pattern,
    property_no_adjacent_repeats,
    SyntheticSequenceDataset,
)


def test_generate_uniform_sequences():
    seqs = generate_uniform_sequences(n=100, K=8, L=32)
    assert seqs.shape == (100, 32)
    assert seqs.dtype == torch.long
    assert seqs.min() >= 0
    assert seqs.max() <= 7


def test_property_starts_with_0():
    seqs = torch.tensor([[0, 1, 2], [1, 2, 3], [0, 0, 0]])
    labels = property_starts_with_0(seqs)
    assert labels.tolist() == [True, False, True]


def test_property_contains_pattern():
    seqs = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 0, 1, 2, 3, 0, 0, 0],
        [7, 7, 7, 7, 7, 7, 7, 7],
    ])
    labels = property_contains_pattern(seqs, pattern=[1, 2, 3])
    assert labels.tolist() == [True, True, False]


def test_property_no_adjacent_repeats():
    seqs = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 0, 1, 2, 3, 4, 5, 6],
        [1, 2, 1, 2, 1, 2, 1, 2],
    ])
    labels = property_no_adjacent_repeats(seqs)
    assert labels.tolist() == [True, False, True]


def test_synthetic_dataset():
    ds = SyntheticSequenceDataset(n=500, K=8, L=32)
    seq, props = ds[0]
    assert seq.shape == (32,)
    assert len(props) == 3
    assert all(isinstance(v, bool) for v in props.values())
