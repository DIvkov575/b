"""Tests for src/l35/sample.py: a deterministic few-step (DDIM/VE-PFODE)
sampler for the CSP task -- given real ground-truth composition (atom_types,
num_atoms) from a test-set structure, denoise frac_coords/lattice from noise
in exactly num_steps steps. Complements DiffCSP's own many-step sample()
(diffusion.py), which we reuse directly for the teacher side of the eval
rather than reimplementing.
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


def _real_schedulers(timesteps=1000):
    import src.l35.torch_scatter_compat_shim  # noqa: F401
    from diffcsp.pl_modules.diff_utils import BetaScheduler, SigmaScheduler

    beta_scheduler = BetaScheduler(timesteps=timesteps, scheduler_mode="cosine")
    sigma_scheduler = SigmaScheduler(timesteps=timesteps, sigma_begin=0.005, sigma_end=0.5)
    return beta_scheduler, sigma_scheduler


def test_few_step_sample_returns_valid_shaped_structure():
    import torch

    from src.l35.sample import few_step_sample

    beta_scheduler, sigma_scheduler = _real_schedulers()

    class ConstantDecoder(torch.nn.Module):
        def forward(self, time_emb, atom_types, frac_coords, lattices, num_atoms, node2graph):
            batch_size = lattices.shape[0]
            num_nodes = frac_coords.shape[0]
            return torch.zeros(batch_size, 3, 3), torch.zeros(num_nodes, 3)

    decoder = ConstantDecoder()
    num_atoms = torch.tensor([3, 4])
    node2graph = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    atom_types = torch.tensor([1, 6, 8, 3, 12, 20, 26])

    frac_coords, lattices = few_step_sample(
        decoder, atom_types, num_atoms, node2graph, num_steps=4,
        beta_scheduler=beta_scheduler, sigma_scheduler=sigma_scheduler,
        max_timestep=1000, generator=torch.Generator().manual_seed(0),
    )

    assert frac_coords.shape == (7, 3)
    assert lattices.shape == (2, 3, 3)
    assert (frac_coords >= 0).all() and (frac_coords < 1).all()
    assert torch.isfinite(frac_coords).all()
    assert torch.isfinite(lattices).all()


def test_few_step_sample_is_deterministic_given_a_seeded_generator():
    import torch

    from src.l35.sample import few_step_sample

    beta_scheduler, sigma_scheduler = _real_schedulers()

    class TinyDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lattice_head = torch.nn.Linear(9, 9)
            self.coord_head = torch.nn.Linear(3, 3)

        def forward(self, time_emb, atom_types, frac_coords, lattices, num_atoms, node2graph):
            batch_size = lattices.shape[0]
            pred_l = self.lattice_head(lattices.reshape(batch_size, 9)).reshape(batch_size, 3, 3)
            pred_x = self.coord_head(frac_coords)
            return pred_l, pred_x

    torch.manual_seed(0)
    decoder = TinyDecoder()
    num_atoms = torch.tensor([3])
    node2graph = torch.tensor([0, 0, 0])
    atom_types = torch.tensor([1, 6, 8])

    fc1, l1 = few_step_sample(
        decoder, atom_types, num_atoms, node2graph, num_steps=4,
        beta_scheduler=beta_scheduler, sigma_scheduler=sigma_scheduler,
        max_timestep=1000, generator=torch.Generator().manual_seed(42),
    )
    fc2, l2 = few_step_sample(
        decoder, atom_types, num_atoms, node2graph, num_steps=4,
        beta_scheduler=beta_scheduler, sigma_scheduler=sigma_scheduler,
        max_timestep=1000, generator=torch.Generator().manual_seed(42),
    )

    assert torch.equal(fc1, fc2)
    assert torch.equal(l1, l2)


def test_few_step_sample_num_steps_one_runs_a_single_denoising_step():
    # num_steps=1 should still run end-to-end without error -- the extreme
    # few-step case this whole distillation project targets.
    import torch

    from src.l35.sample import few_step_sample

    beta_scheduler, sigma_scheduler = _real_schedulers()

    class ConstantDecoder(torch.nn.Module):
        def forward(self, time_emb, atom_types, frac_coords, lattices, num_atoms, node2graph):
            batch_size = lattices.shape[0]
            num_nodes = frac_coords.shape[0]
            return torch.zeros(batch_size, 3, 3), torch.zeros(num_nodes, 3)

    decoder = ConstantDecoder()
    num_atoms = torch.tensor([3])
    node2graph = torch.tensor([0, 0, 0])
    atom_types = torch.tensor([1, 6, 8])

    frac_coords, lattices = few_step_sample(
        decoder, atom_types, num_atoms, node2graph, num_steps=1,
        beta_scheduler=beta_scheduler, sigma_scheduler=sigma_scheduler,
        max_timestep=1000, generator=torch.Generator().manual_seed(0),
    )

    assert torch.isfinite(frac_coords).all()
    assert torch.isfinite(lattices).all()
