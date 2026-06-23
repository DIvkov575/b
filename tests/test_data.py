import torch
from src.data.text_data import Text8Dataset, mask_sequence


def test_text8_dataset_loads():
    ds = Text8Dataset(split="train", seq_len=128, max_samples=1000)
    assert len(ds) == 1000
    seq = ds[0]
    assert seq.shape == (128,)
    assert seq.dtype == torch.long
    assert seq.min() >= 0
    assert seq.max() <= 27


def test_mask_sequence_t0():
    x = torch.tensor([1, 2, 3, 4, 5])
    x_t = mask_sequence(x, t=0.0, mask_token=0)
    assert (x_t == x).all()


def test_mask_sequence_t1():
    x = torch.tensor([1, 2, 3, 4, 5])
    x_t = mask_sequence(x, t=10.0, mask_token=0)  # large t -> fully masked
    assert (x_t == 0).all()


def test_mask_sequence_batched():
    x = torch.randint(1, 27, (8, 128))
    t = torch.rand(8)
    x_t = mask_sequence(x, t, mask_token=0)
    assert x_t.shape == (8, 128)
    mask_rates = (x_t == 0).float().mean(dim=1)
    assert mask_rates.shape == (8,)
