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


def test_preprocess_with_cache_writes_cache_file_on_first_call(tmp_path):
    # Mirrors diffcsp.pl_data.dataset.CrystDataset.preprocess()'s own
    # os.path.exists(save_path) -> torch.load / else preprocess+torch.save
    # convention -- real cache file, real torch.save, not a mock.
    import src.l35.torch_scatter_compat_shim  # noqa: F401

    from src.l35.train_distill import preprocess_with_cache

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    csv_path = os.path.join(repo_root, "third_party", "diffcsp", "data", "mp_20", "test.csv")
    cache_path = tmp_path / "cache.pt"

    assert not cache_path.exists()
    results = preprocess_with_cache(csv_path, num_structures=3, cache_path=str(cache_path))

    assert cache_path.exists()
    assert len(results) == 3


def test_preprocess_with_cache_second_call_skips_reprocessing(tmp_path, monkeypatch):
    import src.l35.torch_scatter_compat_shim  # noqa: F401

    from src.l35.train_distill import preprocess_with_cache
    import src.l35.train_distill as train_distill_module

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    csv_path = os.path.join(repo_root, "third_party", "diffcsp", "data", "mp_20", "test.csv")
    cache_path = tmp_path / "cache.pt"

    first = preprocess_with_cache(csv_path, num_structures=3, cache_path=str(cache_path))

    def _boom(*args, **kwargs):
        raise AssertionError("process_one must not be called on a cache hit")

    monkeypatch.setattr(train_distill_module, "process_one", _boom)

    second = preprocess_with_cache(csv_path, num_structures=3, cache_path=str(cache_path))

    assert len(second) == len(first) == 3
    for a, b in zip(first, second):
        assert a["mp_id"] == b["mp_id"]


def test_preprocess_with_cache_preserves_csv_row_order_under_parallelism(tmp_path):
    # p_umap (used for real parallel speedup on multi-core boxes -- a plain
    # sequential loop was a real regression caught mid-full-run: it left 3
    # of this g5.xlarge's 4 vCPUs idle) returns results in COMPLETION order,
    # not submission order. preprocess_with_cache must remap back to the
    # original CSV row order (matching diffcsp.common.data_utils.preprocess()'s
    # own mpid_to_results dict-remap pattern), not just return p_umap's raw
    # (possibly-shuffled) output list.
    import src.l35.torch_scatter_compat_shim  # noqa: F401
    import pandas as pd

    from src.l35.train_distill import preprocess_with_cache

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    csv_path = os.path.join(repo_root, "third_party", "diffcsp", "data", "mp_20", "test.csv")
    cache_path = tmp_path / "cache.pt"

    results = preprocess_with_cache(csv_path, num_structures=5, cache_path=str(cache_path))

    df = pd.read_csv(csv_path).iloc[:5]
    expected_order = list(df["material_id"])
    actual_order = [r["mp_id"] for r in results]

    assert actual_order == expected_order


def test_preprocess_with_cache_uses_multiple_worker_processes(tmp_path, monkeypatch):
    # Regression test for the exact bug caught mid-full-run: an earlier
    # version used a plain Python list comprehension (single process, single
    # core) instead of p_umap -- confirmed via `ps`/`top` on the real EC2
    # instance (one python3 process at 100% CPU, load average 1.00/4 vCPUs,
    # zero child worker processes). Verifies p_umap is genuinely called with
    # num_cpus > 1 available, not that parallelism helps on any given box.
    import src.l35.torch_scatter_compat_shim  # noqa: F401
    import src.l35.train_distill as train_distill_module

    captured = {}
    real_p_umap = train_distill_module.p_umap

    def _spy_p_umap(*args, **kwargs):
        captured["num_cpus"] = kwargs.get("num_cpus")
        return real_p_umap(*args, **kwargs)

    monkeypatch.setattr(train_distill_module, "p_umap", _spy_p_umap)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    csv_path = os.path.join(repo_root, "third_party", "diffcsp", "data", "mp_20", "test.csv")
    cache_path = tmp_path / "cache.pt"

    train_distill_module.preprocess_with_cache(csv_path, num_structures=3, cache_path=str(cache_path))

    assert "num_cpus" in captured, "preprocess_with_cache must call p_umap, not a sequential loop"
    assert captured["num_cpus"] is None or captured["num_cpus"] > 1


def test_preprocess_with_cache_different_num_structures_is_a_cache_miss(tmp_path):
    # Same cache_path, different num_structures must NOT silently reuse a
    # cache built for a different slice size.
    import src.l35.torch_scatter_compat_shim  # noqa: F401

    from src.l35.train_distill import preprocess_with_cache

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    csv_path = os.path.join(repo_root, "third_party", "diffcsp", "data", "mp_20", "test.csv")
    cache_path = tmp_path / "cache.pt"

    small = preprocess_with_cache(csv_path, num_structures=2, cache_path=str(cache_path))
    large = preprocess_with_cache(csv_path, num_structures=4, cache_path=str(cache_path))

    assert len(small) == 2
    assert len(large) == 4
