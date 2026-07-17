"""Native-torch replacement for the one torch_scatter call DiffCSP's default
(edge_style='fc') code path uses: scatter(src, index, dim=0, reduce='mean').
torch_scatter itself fails to build on this machine (pybind11 header conflict
against torch 2.13 on Apple Silicon); torch.Tensor.scatter_reduce_ covers this
exact call shape natively, including torch_scatter's untouched-group-is-zero
(not NaN) semantics via include_self=False.
"""
import torch


def scatter(src, index, dim=0, reduce="mean", dim_size=None):
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0

    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)

    index_shape = [1] * src.dim()
    index_shape[dim] = index.shape[0]
    expand_shape = list(src.shape)
    expand_shape[dim] = index.shape[0]
    index_expanded = index.view(index_shape).expand(expand_shape)

    return out.scatter_reduce(dim, index_expanded, src, reduce=reduce, include_self=False)
