"""Tests for src/l35/train_distill.py's batch-building helper: converts a real
DiffCSP preprocess() result list into the dict shape consistency_distillation_loss
expects (num_atoms, node2graph, atom_types, frac_coords, lattices), matching the
same PyG Batch.from_data_list construction already validated in
test_real_data_slice.py / test_train_distill_integration.py.
"""
import os
import sys

import pytest


@pytest.fixture(autouse=True)
def _diffcsp_on_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    diffcsp_path = os.path.join(repo_root, "third_party", "diffcsp")
    if diffcsp_path not in sys.path:
        sys.path.insert(0, diffcsp_path)
    os.environ.setdefault("PROJECT_ROOT", diffcsp_path)
    original_cwd = os.getcwd()
    yield
    os.chdir(original_cwd)


def test_build_batch_from_real_preprocessed_rows():
    import src.l35.torch_scatter_compat_shim  # noqa: F401

    from src.l35.train_distill import build_batch_from_preprocess_results
    from tests.l35.test_real_data_slice import _load_n_real_rows

    results = _load_n_real_rows(n=3)
    batch = build_batch_from_preprocess_results(results, device="cpu")

    assert set(batch.keys()) == {"num_atoms", "node2graph", "atom_types", "frac_coords", "lattices"}
    assert batch["num_atoms"].shape == (3,)
    assert batch["lattices"].shape == (3, 3, 3)
    total_atoms = int(batch["num_atoms"].sum())
    assert batch["frac_coords"].shape == (total_atoms, 3)
    assert batch["atom_types"].shape == (total_atoms,)
    assert batch["node2graph"].shape == (total_atoms,)
    assert batch["node2graph"].max().item() == 2  # 3 graphs, 0-indexed


def test_build_batch_atom_types_are_1_indexed():
    # CSPNet does atom_types - 1 internally (cspnet.py forward()); real
    # atomic numbers from process_one() must stay 1-indexed here, not
    # be pre-decremented.
    import src.l35.torch_scatter_compat_shim  # noqa: F401

    from src.l35.train_distill import build_batch_from_preprocess_results
    from tests.l35.test_real_data_slice import _load_n_real_rows

    results = _load_n_real_rows(n=3)
    batch = build_batch_from_preprocess_results(results, device="cpu")

    assert batch["atom_types"].min().item() >= 1


def test_build_batch_moves_tensors_to_requested_device():
    import src.l35.torch_scatter_compat_shim  # noqa: F401
    import torch

    from src.l35.train_distill import build_batch_from_preprocess_results
    from tests.l35.test_real_data_slice import _load_n_real_rows

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")

    results = _load_n_real_rows(n=2)
    batch = build_batch_from_preprocess_results(results, device="cuda")

    assert batch["frac_coords"].device.type == "cuda"
    assert batch["lattices"].device.type == "cuda"
