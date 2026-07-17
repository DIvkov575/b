"""Regression test for the numpy 2.x list-indexing fix in third_party/mdgen's
batched_gather (mdgen/tensor_utils.py). Modern numpy (2.x) rejects advanced indexing
with a plain list of index arrays/slices -- it must be a tuple. MDGen was pinned to
numpy==1.21.2, where the implicit list-to-tuple coercion still worked; the vendored
copy in third_party/mdgen/mdgen/tensor_utils.py is patched (`data[tuple(ranges)]`
instead of `data[ranges]`) to work on the numpy actually installed in this repo's venv.
"""
import sys

import numpy as np
import pytest

MDGEN_PATH = "/Users/divkov/workplace/biostat/third_party/mdgen"


@pytest.fixture(autouse=True)
def _mdgen_on_path():
    if MDGEN_PATH not in sys.path:
        sys.path.insert(0, MDGEN_PATH)
    yield


def test_batched_gather_with_list_indexing_matches_tuple_indexing_reference():
    import torch

    from mdgen.tensor_utils import batched_gather

    data = torch.arange(2 * 3 * 4).reshape(2, 3, 4).float()
    inds = torch.tensor([[0, 2, 1], [1, 0, 2]])  # shape (2, 3), gather along dim=1

    # reference: what the un-patched code intended (list-of-index-objects indexing),
    # computed here via the tuple form so the reference itself doesn't hit the bug.
    ranges = [torch.arange(2).view(2, 1)]
    ranges.append(inds)
    expected = data[tuple(ranges)]

    got = batched_gather(data, inds, dim=1, no_batch_dims=1)
    assert torch.equal(got, expected)


def test_batched_gather_does_not_raise_on_modern_numpy_arrays():
    # the exact failure mode hit against the real MDGen data path: gathering atom14
    # coordinates into atom37 layout for a numpy float array, batched over residues.
    from mdgen.tensor_utils import batched_gather
    import torch

    atom14 = np.random.randn(1, 4, 14, 3).astype(np.float32)
    aatype_to_atom37 = torch.randint(0, 14, (20, 37))  # stand-in restype table
    aatype = torch.tensor([[0, 1, 2, 3]])  # (batch=1, residues=4)

    inds = aatype_to_atom37[aatype]  # shape (1, 4, 37)
    # must not raise "setting an array element with a sequence" (the numpy list-index bug)
    out = batched_gather(atom14, inds, dim=-2, no_batch_dims=len(atom14.shape[:-2]))
    assert out.shape == (1, 4, 37, 3)
