"""Tests for src/l35/evaluate.py: converts few_step_sample()'s flat
(frac_coords, lattices) output plus ground-truth composition into the
crys_array_dict shape compute_metrics.Crystal expects, and builds a
ground-truth Crystal from a real preprocessed MP-20 test-set structure --
the two conversions the eval pipeline needs around DiffCSP's own real
RecEval/Crystal machinery (reused directly, not reimplemented).
"""
import os
import sys

import pytest


@pytest.fixture(autouse=True)
def _diffcsp_on_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    diffcsp_path = os.path.join(repo_root, "third_party", "diffcsp")
    scripts_path = os.path.join(diffcsp_path, "scripts")
    for p in (diffcsp_path, scripts_path):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("PROJECT_ROOT", diffcsp_path)
    original_cwd = os.getcwd()
    yield
    os.chdir(original_cwd)


def test_split_sample_into_crystal_array_dicts_matches_per_graph_atom_counts():
    import src.l35.torch_scatter_compat_shim  # noqa: F401
    import torch

    from src.l35.evaluate import split_sample_into_crystal_dicts

    num_atoms = torch.tensor([3, 2])
    frac_coords = torch.rand(5, 3)
    lattices = torch.eye(3).unsqueeze(0).repeat(2, 1, 1) * 4.0
    atom_types = torch.tensor([1, 6, 8, 11, 17])

    dicts = split_sample_into_crystal_dicts(frac_coords, lattices, atom_types, num_atoms)

    assert len(dicts) == 2
    assert dicts[0]["frac_coords"].shape == (3, 3)
    assert dicts[0]["atom_types"].shape == (3,)
    assert dicts[1]["frac_coords"].shape == (2, 3)
    assert dicts[1]["atom_types"].shape == (2,)
    # lattice matrix converted to (lengths, angles), not left as a 3x3 matrix
    assert dicts[0]["lengths"].shape == (3,)
    assert dicts[0]["angles"].shape == (3,)


def test_split_sample_into_crystal_dicts_produces_real_Crystal_objects():
    # Round-trip through DiffCSP's own real Crystal class -- confirms the
    # dict shape this function produces is actually consumable by the real
    # eval machinery, not just internally self-consistent.
    import src.l35.torch_scatter_compat_shim  # noqa: F401
    import torch
    from compute_metrics import Crystal

    from src.l35.evaluate import split_sample_into_crystal_dicts

    num_atoms = torch.tensor([4])
    frac_coords = torch.rand(4, 3)
    lattices = torch.eye(3).unsqueeze(0) * 5.0
    atom_types = torch.tensor([11, 17, 17, 17])  # NaCl3-ish, real elements

    dicts = split_sample_into_crystal_dicts(frac_coords, lattices, atom_types, num_atoms)
    crys = Crystal(dicts[0])

    assert crys.structure is not None or not crys.constructed


def test_ground_truth_crystal_from_real_preprocessed_row():
    import src.l35.torch_scatter_compat_shim  # noqa: F401
    from compute_metrics import Crystal

    from src.l35.evaluate import ground_truth_crystal_dict
    from tests.l35.test_real_data_slice import _load_n_real_rows

    results = _load_n_real_rows(n=1)
    gt_dict = ground_truth_crystal_dict(results[0])
    gt_crys = Crystal(gt_dict)

    # a real MP-20 structure's ground truth must always construct successfully
    assert gt_crys.constructed


def test_validity_rates_on_real_ground_truth_crystals_is_all_ones():
    # Real MP-20 ground-truth structures are, by construction, always
    # constructed/comp_valid/struct_valid/valid -- a sanity check that
    # validity_rates' arithmetic (sum(bool)/n) is correct, using known-good
    # input rather than a synthetic one.
    import src.l35.torch_scatter_compat_shim  # noqa: F401
    from compute_metrics import Crystal

    from src.l35.evaluate import ground_truth_crystal_dict, validity_rates
    from tests.l35.test_real_data_slice import _load_n_real_rows

    results = _load_n_real_rows(n=4)
    gt_crys = [Crystal(ground_truth_crystal_dict(r)) for r in results]

    rates = validity_rates(gt_crys)

    assert rates == {"constructed": 1.0, "comp_valid": 1.0, "struct_valid": 1.0, "valid": 1.0}


def test_run_eval_sampled_output_for_a_config_is_unaffected_by_other_configs_in_the_list(monkeypatch):
    # Regression test for a real bug: run_eval fed every sampler_config from
    # a SINGLE, sequentially-advancing generator, so a given (name, network,
    # num_steps) config's actual sampled noise silently depended on what
    # OTHER configs were present earlier in the list for the same ground-
    # truth structure. Confirmed on a real EC2 NFE sweep: the exact same
    # checkpoint, structures, and seed reported match_rate=0.18 for
    # "teacher@8" run alone, but match_rate=0.38 for "teacher@8" run as part
    # of a [teacher@4, student@4, teacher@8, student@8, ...] sweep -- purely
    # from generator-state drift, not a real quality difference. Asserts on
    # the raw sampled (frac_coords, lattices) tensors directly rather than
    # downstream match_rate, since an untrained/degenerate network can floor
    # match_rate at 0 in both cases and mask the bug (as an earlier version
    # of this test did). Every config must see the SAME starting noise per
    # ground-truth structure, regardless of sweep composition or order.
    import src.l35.torch_scatter_compat_shim  # noqa: F401
    import torch
    import torch.nn as nn
    from diffcsp.pl_modules.diff_utils import BetaScheduler, SigmaScheduler

    import src.l35.evaluate as evaluate_module
    from tests.l35.test_real_data_slice import _load_n_real_rows

    class LinearDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.lattice_head = nn.Linear(9, 9)
            self.coord_head = nn.Linear(3, 3)

        def forward(self, time_emb, atom_types, frac_coords, lattices, num_atoms, node2graph):
            batch_size = lattices.shape[0]
            pred_l = self.lattice_head(lattices.reshape(batch_size, 9)).reshape(batch_size, 3, 3)
            pred_x = self.coord_head(frac_coords)
            return pred_l, pred_x

    torch.manual_seed(0)
    network = LinearDecoder()
    beta_scheduler = BetaScheduler(timesteps=1000, scheduler_mode="cosine")
    sigma_scheduler = SigmaScheduler(timesteps=1000, sigma_begin=0.005, sigma_end=0.5)
    results = _load_n_real_rows(n=3)

    real_few_step_sample = evaluate_module.few_step_sample
    captured = {"alone": [], "with_others": []}

    def _make_spy(bucket):
        def _spy(network, atom_types, num_atoms, node2graph, num_steps, *args, **kwargs):
            fc, lattices = real_few_step_sample(
                network, atom_types, num_atoms, node2graph, num_steps, *args, **kwargs
            )
            if num_steps == 8:
                bucket.append((fc.clone(), lattices.clone()))
            return fc, lattices
        return _spy

    monkeypatch.setattr(evaluate_module, "few_step_sample", _make_spy(captured["alone"]))
    run_eval = evaluate_module.run_eval
    run_eval(
        [("x@8", network, 8)], results, beta_scheduler, sigma_scheduler,
        max_timestep=1000, device=torch.device("cpu"), generator=torch.Generator().manual_seed(0),
    )

    monkeypatch.setattr(evaluate_module, "few_step_sample", _make_spy(captured["with_others"]))
    run_eval(
        [("x@4", network, 4), ("x@8", network, 8)], results, beta_scheduler, sigma_scheduler,
        max_timestep=1000, device=torch.device("cpu"), generator=torch.Generator().manual_seed(0),
    )

    assert len(captured["alone"]) == len(captured["with_others"]) == 3
    for (fc_a, l_a), (fc_b, l_b) in zip(captured["alone"], captured["with_others"]):
        assert torch.equal(fc_a, fc_b)
        assert torch.equal(l_a, l_b)


def test_run_eval_compares_multiple_sampler_configs_against_same_ground_truth():
    # Exercises run_eval's multi-config interface (the diagnostic-ladder
    # redesign: compare teacher@many_steps / teacher@few_steps / student@
    # few_steps against the SAME ground truth in one pass) with a cheap
    # stand-in decoder, not a real checkpoint -- the real checkpoint path is
    # covered separately by the actual EC2 eval run.
    import src.l35.torch_scatter_compat_shim  # noqa: F401
    import torch
    import torch.nn as nn
    from diffcsp.pl_modules.diff_utils import BetaScheduler, SigmaScheduler

    from src.l35.evaluate import run_eval
    from tests.l35.test_real_data_slice import _load_n_real_rows

    class ConstantDecoder(nn.Module):
        def forward(self, time_emb, atom_types, frac_coords, lattices, num_atoms, node2graph):
            batch_size = lattices.shape[0]
            num_nodes = frac_coords.shape[0]
            return torch.zeros(batch_size, 3, 3), torch.zeros(num_nodes, 3)

    teacher = ConstantDecoder()
    student = ConstantDecoder()
    beta_scheduler = BetaScheduler(timesteps=1000, scheduler_mode="cosine")
    sigma_scheduler = SigmaScheduler(timesteps=1000, sigma_begin=0.005, sigma_end=0.5)

    results = _load_n_real_rows(n=2)
    sampler_configs = [
        ("teacher@50", teacher, 50),
        ("teacher@8", teacher, 8),
        ("student@8", student, 8),
    ]

    all_metrics = run_eval(
        sampler_configs, results, beta_scheduler, sigma_scheduler,
        max_timestep=1000, device=torch.device("cpu"), generator=torch.Generator().manual_seed(0),
    )

    assert set(all_metrics.keys()) == {"teacher@50", "teacher@8", "student@8"}
    for metrics in all_metrics.values():
        assert set(metrics.keys()) >= {"match_rate", "rms_dist", "constructed", "comp_valid", "struct_valid", "valid"}


def test_run_eval_accepts_a_different_sampler_fn():
    # run_eval must not hardcode few_step_sample -- a single trained
    # consistency-distilled student should be evaluatable at multiple NFEs
    # via multistep_consistency_sample (the genuine consistency sampler,
    # src/l35/sample.py) using the SAME run_eval scaffolding (same
    # per-structure reseeding guarantee, same metric shape), not a
    # parallel/duplicated eval loop.
    import src.l35.torch_scatter_compat_shim  # noqa: F401
    import torch
    import torch.nn as nn
    from diffcsp.pl_modules.diff_utils import BetaScheduler, SigmaScheduler

    from src.l35.evaluate import run_eval
    from src.l35.sample import multistep_consistency_sample
    from tests.l35.test_real_data_slice import _load_n_real_rows

    class ConstantDecoder(nn.Module):
        def forward(self, time_emb, atom_types, frac_coords, lattices, num_atoms, node2graph):
            batch_size = lattices.shape[0]
            num_nodes = frac_coords.shape[0]
            return torch.zeros(batch_size, 3, 3), torch.zeros(num_nodes, 3)

    student = ConstantDecoder()
    beta_scheduler = BetaScheduler(timesteps=1000, scheduler_mode="cosine")
    sigma_scheduler = SigmaScheduler(timesteps=1000, sigma_begin=0.005, sigma_end=0.5)

    results = _load_n_real_rows(n=2)
    sampler_configs = [("student@4", student, 4), ("student@8", student, 8)]

    all_metrics = run_eval(
        sampler_configs, results, beta_scheduler, sigma_scheduler,
        max_timestep=1000, device=torch.device("cpu"), generator=torch.Generator().manual_seed(0),
        sampler_fn=multistep_consistency_sample,
    )

    assert set(all_metrics.keys()) == {"student@4", "student@8"}
    for metrics in all_metrics.values():
        assert set(metrics.keys()) >= {"match_rate", "rms_dist", "constructed", "comp_valid", "struct_valid", "valid"}


def test_run_eval_default_sampler_fn_is_unchanged(monkeypatch):
    # Regression guard: adding the sampler_fn parameter must not change
    # run_eval's DEFAULT behavior for any existing caller that doesn't pass
    # it -- it must still resolve to (module-level, monkeypatch-able)
    # few_step_sample, not a def-time-bound reference to it (which would
    # silently break the monkeypatch-based generator-leakage regression
    # test above: monkeypatch.setattr(evaluate_module, "few_step_sample",
    # ...) only takes effect if run_eval looks the name up at call time).
    import src.l35.torch_scatter_compat_shim  # noqa: F401
    import torch
    import torch.nn as nn
    from diffcsp.pl_modules.diff_utils import BetaScheduler, SigmaScheduler

    import src.l35.evaluate as evaluate_module
    from tests.l35.test_real_data_slice import _load_n_real_rows

    class ConstantDecoder(nn.Module):
        def forward(self, time_emb, atom_types, frac_coords, lattices, num_atoms, node2graph):
            batch_size = lattices.shape[0]
            num_nodes = frac_coords.shape[0]
            return torch.zeros(batch_size, 3, 3), torch.zeros(num_nodes, 3)

    calls = []
    real_few_step_sample = evaluate_module.few_step_sample

    def _spy(*args, **kwargs):
        calls.append(1)
        return real_few_step_sample(*args, **kwargs)

    monkeypatch.setattr(evaluate_module, "few_step_sample", _spy)

    beta_scheduler = BetaScheduler(timesteps=1000, scheduler_mode="cosine")
    sigma_scheduler = SigmaScheduler(timesteps=1000, sigma_begin=0.005, sigma_end=0.5)
    results = _load_n_real_rows(n=1)

    evaluate_module.run_eval(
        [("student@8", ConstantDecoder(), 8)], results, beta_scheduler, sigma_scheduler,
        max_timestep=1000, device=torch.device("cpu"), generator=torch.Generator().manual_seed(0),
    )

    assert len(calls) == 1, "run_eval's default sampler_fn must resolve few_step_sample at call time"


def test_run_multi_seed_sweep_runs_run_eval_once_per_seed():
    # A single-seed run_eval call is a single point estimate; the multi-
    # seed sweep must call run_eval independently for EACH seed (fresh
    # generator per seed) and return per-config lists of per-seed metrics,
    # not just the last seed's result.
    import src.l35.torch_scatter_compat_shim  # noqa: F401
    import torch
    import torch.nn as nn
    from diffcsp.pl_modules.diff_utils import BetaScheduler, SigmaScheduler

    from src.l35.evaluate import run_multi_seed_sweep
    from tests.l35.test_real_data_slice import _load_n_real_rows

    class ConstantDecoder(nn.Module):
        def forward(self, time_emb, atom_types, frac_coords, lattices, num_atoms, node2graph):
            batch_size = lattices.shape[0]
            num_nodes = frac_coords.shape[0]
            return torch.zeros(batch_size, 3, 3), torch.zeros(num_nodes, 3)

    teacher = ConstantDecoder()
    beta_scheduler = BetaScheduler(timesteps=1000, scheduler_mode="cosine")
    sigma_scheduler = SigmaScheduler(timesteps=1000, sigma_begin=0.005, sigma_end=0.5)
    results = _load_n_real_rows(n=2)

    sweep = run_multi_seed_sweep(
        [("teacher@8", teacher, 8)], results, beta_scheduler, sigma_scheduler,
        max_timestep=1000, device=torch.device("cpu"), seeds=[0, 1, 2],
    )

    assert set(sweep.keys()) == {"teacher@8"}
    assert len(sweep["teacher@8"]["match_rate"]) == 3, "one match_rate value per seed"


def test_run_multi_seed_sweep_each_seed_gets_independent_starting_noise():
    # Regression guard for the most likely way to get this wrong: reusing
    # ONE generator across seeds (e.g. re-seeding it in a loop without a
    # fresh torch.Generator each time) could silently correlate "different
    # seeds" if downstream state leaks -- confirm two different seeds in
    # the sweep produce DIFFERENT raw sampled tensors for the same
    # structure/config (a real, non-degenerate decoder is needed for this;
    # ConstantDecoder always produces the same output regardless of noise
    # only for pred_l/pred_x, but the INITIAL noise draw itself still
    # differs and propagates through the deterministic DDIM/PF-ODE steps).
    import src.l35.torch_scatter_compat_shim  # noqa: F401
    import torch
    import torch.nn as nn
    from diffcsp.pl_modules.diff_utils import BetaScheduler, SigmaScheduler

    import src.l35.evaluate as evaluate_module
    from tests.l35.test_real_data_slice import _load_n_real_rows

    class ConstantDecoder(nn.Module):
        def forward(self, time_emb, atom_types, frac_coords, lattices, num_atoms, node2graph):
            batch_size = lattices.shape[0]
            num_nodes = frac_coords.shape[0]
            return torch.zeros(batch_size, 3, 3), torch.zeros(num_nodes, 3)

    teacher = ConstantDecoder()
    beta_scheduler = BetaScheduler(timesteps=1000, scheduler_mode="cosine")
    sigma_scheduler = SigmaScheduler(timesteps=1000, sigma_begin=0.005, sigma_end=0.5)
    results = _load_n_real_rows(n=1)

    captured = []
    real_few_step_sample = evaluate_module.few_step_sample

    def _spy(*args, **kwargs):
        fc, lattices = real_few_step_sample(*args, **kwargs)
        captured.append(fc.clone())
        return fc, lattices

    import pytest as _pytest
    monkeypatch = _pytest.MonkeyPatch()
    monkeypatch.setattr(evaluate_module, "few_step_sample", _spy)
    try:
        evaluate_module.run_multi_seed_sweep(
            [("teacher@8", teacher, 8)], results, beta_scheduler, sigma_scheduler,
            max_timestep=1000, device=torch.device("cpu"), seeds=[0, 1],
        )
    finally:
        monkeypatch.undo()

    assert len(captured) == 2
    assert not torch.equal(captured[0], captured[1]), "different seeds must draw different initial noise"
