"""Tests for evaluating against DiffCSP's OWN, real, published sampler
(CSPDiffusion.sample() -- the stochastic annealed-Langevin predictor-
corrector in diffusion.py) rather than only this project's homebrew
deterministic DDIM/PF-ODE reimplementation (sample.py's few_step_sample).

The roast that prompted this: the paper's teacher@1000 "reference ceiling"
(0.425 match rate) was computed entirely through few_step_sample, never
through DiffCSP's real sample() -- and is ~9 points below DiffCSP's own
paper's reported 51.49% match rate on the identical checkpoint/task. This
file establishes that the real sample() can actually be invoked end-to-end
against this project's real teacher checkpoint and real MP-20 data, at its
native NFE (1000) -- see load_teacher_module's docstring for why arbitrary
NFE truncation of the real sampler is invalid (its schedule is baked for a
fixed timestep count; truncating just the loop bound, not respacing the
schedule, means an early stop is fed inputs the schedule believes are
already >99% denoised).
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


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CKPT_PATH = os.path.join(REPO_ROOT, "third_party", "diffcsp_checkpoints", "mp_csp", "last.ckpt")
HPARAMS_PATH = os.path.join(REPO_ROOT, "third_party", "diffcsp_checkpoints", "mp_csp", "hparams.yaml")


def _real_pyg_batch(n, device):
    import torch
    from torch_geometric.data import Batch, Data

    from tests.l35.test_real_data_slice import _load_n_real_rows

    results = _load_n_real_rows(n=n)
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
    return Batch.from_data_list(data_list).to(device), results


def test_load_teacher_module_returns_full_module_not_just_decoder():
    import src.l35.torch_scatter_compat_shim  # noqa: F401

    from src.l35.train_distill import load_teacher_module

    model = load_teacher_module(CKPT_PATH, HPARAMS_PATH, device="cpu")

    assert hasattr(model, "sample"), "must be the full CSPDiffusion module, not just the decoder"
    assert hasattr(model, "decoder")
    assert hasattr(model, "beta_scheduler")
    assert hasattr(model, "sigma_scheduler")


def test_load_teacher_module_is_frozen_and_in_eval_mode():
    import src.l35.torch_scatter_compat_shim  # noqa: F401

    from src.l35.train_distill import load_teacher_module

    model = load_teacher_module(CKPT_PATH, HPARAMS_PATH, device="cpu")

    assert model.training is False
    assert all(not p.requires_grad for p in model.parameters())


def test_real_sample_runs_end_to_end_on_real_teacher_and_real_data():
    # The actual, published DiffCSP sampler -- not a reimplementation --
    # invoked against a real batch built from real MP-20 rows. This is the
    # sampler the paper's teacher@1000 "ceiling" should have used and did
    # not.
    import src.l35.torch_scatter_compat_shim  # noqa: F401
    import torch

    from src.l35.train_distill import load_teacher_module

    model = load_teacher_module(CKPT_PATH, HPARAMS_PATH, device="cpu")
    batch, _ = _real_pyg_batch(n=2, device="cpu")

    torch.manual_seed(0)
    out, _traj = model.sample(batch, step_lr=1e-5)

    assert torch.isfinite(out["frac_coords"]).all()
    assert torch.isfinite(out["lattices"]).all()
    assert out["frac_coords"].shape == (batch.num_nodes, 3)
    assert out["lattices"].shape == (batch.num_graphs, 3, 3)


def test_real_sample_is_reproducible_under_a_fixed_torch_seed():
    # model.sample() takes no generator argument -- it draws from the global
    # torch RNG directly (torch.randn_like/torch.rand inside diffusion.py).
    # A caller sweeping many ground-truth structures needs torch.manual_seed
    # reseeded per-structure for the same "same starting noise regardless of
    # sweep order" guarantee run_eval already provides for the homebrew
    # sampler.
    import src.l35.torch_scatter_compat_shim  # noqa: F401
    import torch

    from src.l35.train_distill import load_teacher_module

    model = load_teacher_module(CKPT_PATH, HPARAMS_PATH, device="cpu")
    batch, _ = _real_pyg_batch(n=2, device="cpu")

    torch.manual_seed(42)
    out_a, _ = model.sample(batch, step_lr=1e-5)
    torch.manual_seed(42)
    out_b, _ = model.sample(batch, step_lr=1e-5)

    assert torch.equal(out_a["frac_coords"], out_b["frac_coords"])
    assert torch.equal(out_a["lattices"], out_b["lattices"])


def test_run_real_sampler_eval_matches_run_eval_metric_shape():
    # run_real_sampler_eval must return the same {"match_rate", "rms_dist",
    # ...validity keys} shape run_eval does, so results can be reported
    # side-by-side in the same table without special-casing downstream.
    import src.l35.torch_scatter_compat_shim  # noqa: F401

    from src.l35.evaluate import run_real_sampler_eval
    from src.l35.train_distill import load_teacher_module
    from tests.l35.test_real_data_slice import _load_n_real_rows

    model = load_teacher_module(CKPT_PATH, HPARAMS_PATH, device="cpu")
    results = _load_n_real_rows(n=2)

    metrics = run_real_sampler_eval(model, results, device="cpu", seed=0)

    assert set(metrics.keys()) >= {
        "match_rate", "rms_dist", "constructed", "comp_valid", "struct_valid", "valid",
    }


def test_run_real_sampler_eval_is_reproducible_given_the_same_seed():
    import src.l35.torch_scatter_compat_shim  # noqa: F401

    from src.l35.evaluate import run_real_sampler_eval
    from src.l35.train_distill import load_teacher_module
    from tests.l35.test_real_data_slice import _load_n_real_rows

    model = load_teacher_module(CKPT_PATH, HPARAMS_PATH, device="cpu")
    results = _load_n_real_rows(n=2)

    first = run_real_sampler_eval(model, results, device="cpu", seed=0)
    second = run_real_sampler_eval(model, results, device="cpu", seed=0)

    assert first["match_rate"] == second["match_rate"]
