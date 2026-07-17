"""MP-20 is bundled directly in third_party/diffcsp/data/mp_20/*.csv (no external
download needed). This loads a handful of REAL rows from the actual test split
through DiffCSP's own process_one() -- real CIF parsing via pymatgen, real graph
construction -- to get real Data objects for mechanics validation, without
preprocessing the full 45k-structure dataset.
"""
import os
import sys

import pandas as pd
import pytest

# Anchored to this file's own location, not cwd: diffcsp.common.utils does
# os.chdir(PROJECT_ROOT) as an import side effect, and callers reached via a
# fixture that already imported diffcsp modules (e.g. from a different test
# file's fixture) would have a different, already-changed cwd -- os.path.abspath
# on a relative string is cwd-relative and silently doubles the path in that case.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture(autouse=True)
def _diffcsp_on_path():
    diffcsp_path = os.path.join(_REPO_ROOT, "third_party", "diffcsp")
    if diffcsp_path not in sys.path:
        sys.path.insert(0, diffcsp_path)
    os.environ.setdefault("PROJECT_ROOT", diffcsp_path)

    original_cwd = os.getcwd()
    yield
    os.chdir(original_cwd)


def _load_n_real_rows(n=4):
    import src.l35.torch_scatter_compat_shim  # noqa: F401
    from diffcsp.common.data_utils import process_one

    csv_path = os.path.join(_REPO_ROOT, "third_party", "diffcsp", "data", "mp_20", "test.csv")
    df = pd.read_csv(csv_path)
    rows = [df.iloc[idx] for idx in range(n)]

    results = [
        process_one(
            row,
            niggli=True,
            primitive=True,
            graph_method="crystalnn",
            prop_list=["formation_energy_per_atom"],
            use_space_group=False,
            tol=0.01,
        )
        for row in rows
    ]
    return results


def test_loads_n_real_mp20_structures_with_graph_arrays():
    results = _load_n_real_rows(n=4)

    assert len(results) == 4
    for result in results:
        assert "graph_arrays" in result
        frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms = result[
            "graph_arrays"
        ]
        assert num_atoms > 0
        assert frac_coords.shape == (num_atoms, 3)
        assert atom_types.shape == (num_atoms,)
        assert lengths.shape == (3,)
        assert angles.shape == (3,)


def test_real_structures_convert_to_pyg_data_batch():
    import src.l35.torch_scatter_compat_shim  # noqa: F401
    import torch
    from torch_geometric.data import Data, Batch

    results = _load_n_real_rows(n=4)

    data_list = []
    for result in results:
        frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms = result[
            "graph_arrays"
        ]
        data_list.append(
            Data(
                frac_coords=torch.Tensor(frac_coords),
                atom_types=torch.LongTensor(atom_types),
                lengths=torch.Tensor(lengths).view(1, -1),
                angles=torch.Tensor(angles).view(1, -1),
                num_atoms=num_atoms,
                num_nodes=num_atoms,
            )
        )

    batch = Batch.from_data_list(data_list)

    assert batch.num_graphs == 4
    assert batch.frac_coords.shape[1] == 3
    assert batch.atom_types.min() >= 1  # 1-indexed atomic numbers, per CSPNet's `- 1` offset
