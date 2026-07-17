"""torch_scatter is a C++ extension that fails to build on this machine (pybind11/Clang
header conflict against torch 2.13 on Apple Silicon -- same class of toolchain wall as
L37's pyEMMA/Clang-21 problem). DiffCSP's default config (edge_style='fc') only calls
torch_scatter.scatter(..., reduce='mean'); this module provides a native-torch
compat shim for exactly that call shape, verified against torch_scatter's own
documented semantics, not re-derived.
"""
import torch

from src.l35.torch_scatter_compat import scatter


def test_scatter_mean_matches_manual_groupby_mean():
    torch.manual_seed(0)
    src = torch.randn(10, 4)
    index = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 3, 3])
    dim_size = 4

    expected = torch.zeros(dim_size, 4)
    counts = torch.zeros(dim_size)
    for j in range(10):
        expected[index[j]] += src[j]
        counts[index[j]] += 1
    expected = expected / counts[:, None]

    out = scatter(src, index, dim=0, reduce="mean", dim_size=dim_size)

    assert torch.allclose(out, expected, atol=1e-6)


def test_scatter_mean_empty_group_is_zero_not_nan():
    # index never touches group 3 -- torch_scatter leaves untouched groups at 0,
    # not NaN (which a naive sum/count division would produce).
    src = torch.randn(6, 2)
    index = torch.tensor([0, 0, 1, 1, 2, 2])
    dim_size = 4

    out = scatter(src, index, dim=0, reduce="mean", dim_size=dim_size)

    assert torch.equal(out[3], torch.zeros(2))
    assert not torch.isnan(out).any()


def test_scatter_mean_runs_on_mps():
    if not torch.backends.mps.is_available():
        return
    device = torch.device("mps")
    src = torch.randn(10, 4, device=device)
    index = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 3, 3], device=device)

    out = scatter(src, index, dim=0, reduce="mean", dim_size=4)

    assert out.device.type == "mps"
    assert not torch.isnan(out).any()


def test_scatter_infers_dim_size_from_index_when_not_given():
    src = torch.randn(6, 2)
    index = torch.tensor([0, 0, 1, 1, 2, 2])

    out = scatter(src, index, dim=0, reduce="mean")

    assert out.shape == (3, 2)
