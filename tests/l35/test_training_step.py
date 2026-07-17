"""Tests for the L35 consistency-distillation training step, composing the math
core (src/l35/consistency_distill.py) with real DiffCSP model calling conventions:
decoder pred_l IS the epsilon prediction directly (diffusion.py forward():
loss_lattice = mse(pred_l, rand_l)); decoder pred_x is a NORMALIZED score and
must be rescaled by sqrt(sigma_norm) before use as the true score (diffusion.py
sample(): `pred_x = pred_x * torch.sqrt(sigma_norm)`), per the teacher's own
un-normalization convention -- not re-derived, lifted from the real sample() method.

Uses REAL diffcsp.pl_modules.diff_utils.BetaScheduler/SigmaScheduler instances,
not a re-derived approximation: sigma_norm has no closed form (it's a 10,000-
sample Monte Carlo estimate, diff_utils.sigma_norm) and a naive sigma**2
stand-in was caught differing from the real value by ~5 orders of magnitude
before this test file was written -- see docs/L35_PIPELINE_SPEC.md.

Uses small stand-in nn.Modules (real autograd, not mocks) mirroring L37's
test_training_step.py pattern, since a full CSPNet forward pass needs a real
PyG batch -- that's covered separately against the real teacher checkpoint
in test_train_distill_integration.py.
"""
import copy
import os
import sys

import pytest
import torch
import torch.nn as nn


@pytest.fixture(autouse=True)
def _diffcsp_on_path():
    diffcsp_path = os.path.abspath("third_party/diffcsp")
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


from src.l35.training_step import consistency_distillation_loss  # noqa: E402


class ConstantDecoder(nn.Module):
    """Stand-in decoder: returns fixed (pred_l, pred_x) regardless of input,
    matching CSPNet's (lattice_out, coord_out) return signature.
    """

    def __init__(self, pred_l, pred_x):
        super().__init__()
        self.pred_l = pred_l
        self.pred_x = pred_x

    def forward(self, time_emb, atom_types, frac_coords, lattices, num_atoms, node2graph):
        batch_size = lattices.shape[0]
        num_nodes = frac_coords.shape[0]
        return self.pred_l.expand(batch_size, 3, 3), self.pred_x.expand(num_nodes, 3)


class LinearDecoder(nn.Module):
    """Trainable stand-in decoder with real params on both output heads."""

    def __init__(self):
        super().__init__()
        self.lattice_head = nn.Linear(9, 9)
        self.coord_head = nn.Linear(3, 3)

    def forward(self, time_emb, atom_types, frac_coords, lattices, num_atoms, node2graph):
        batch_size = lattices.shape[0]
        pred_l = self.lattice_head(lattices.reshape(batch_size, 9)).reshape(batch_size, 3, 3)
        pred_x = self.coord_head(frac_coords)
        return pred_l, pred_x


def _toy_batch(batch_size=2, num_atoms_per=3):
    num_atoms = torch.full((batch_size,), num_atoms_per)
    node2graph = torch.repeat_interleave(torch.arange(batch_size), num_atoms)
    total_atoms = int(num_atoms.sum())
    atom_types = torch.randint(1, 10, (total_atoms,))
    frac_coords = torch.rand(total_atoms, 3)
    lattices = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
    return dict(
        num_atoms=num_atoms,
        node2graph=node2graph,
        atom_types=atom_types,
        frac_coords=frac_coords,
        lattices=lattices,
    )


def test_teacher_receives_no_gradient_student_does():
    torch.manual_seed(0)
    batch = _toy_batch()
    beta_scheduler, sigma_scheduler = _real_schedulers()

    teacher = LinearDecoder()
    student = LinearDecoder()

    loss, _ = consistency_distillation_loss(
        teacher=teacher,
        student=student,
        batch=batch,
        num_steps=8,
        beta_scheduler=beta_scheduler,
        sigma_scheduler=sigma_scheduler,
        generator=torch.Generator().manual_seed(1),
    )
    loss.backward()

    for p in teacher.parameters():
        assert p.grad is None, "teacher must be frozen (stop-gradient target)"
    for p in student.parameters():
        assert p.grad is not None, "student must receive gradients"


def test_deepcopied_student_still_receives_gradient():
    # Same deepcopy-inherits-requires_grad_False regression covered in L37's
    # training_step tests -- verify the same fix applies here.
    torch.manual_seed(0)
    batch = _toy_batch()
    beta_scheduler, sigma_scheduler = _real_schedulers()

    teacher = LinearDecoder()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = copy.deepcopy(teacher)
    for p in student.parameters():
        p.requires_grad_(True)

    loss, _ = consistency_distillation_loss(
        teacher=teacher,
        student=student,
        batch=batch,
        num_steps=8,
        beta_scheduler=beta_scheduler,
        sigma_scheduler=sigma_scheduler,
        generator=torch.Generator().manual_seed(1),
    )
    loss.backward()

    assert all(p.grad is not None for p in student.parameters())


def test_loss_is_finite_and_positive_for_disagreeing_teacher_and_student():
    torch.manual_seed(0)
    batch = _toy_batch()
    beta_scheduler, sigma_scheduler = _real_schedulers()

    teacher = ConstantDecoder(pred_l=torch.randn(3, 3), pred_x=torch.randn(3))
    student = ConstantDecoder(pred_l=torch.zeros(3, 3), pred_x=torch.zeros(3))

    loss, aux = consistency_distillation_loss(
        teacher=teacher,
        student=student,
        batch=batch,
        num_steps=8,
        beta_scheduler=beta_scheduler,
        sigma_scheduler=sigma_scheduler,
        generator=torch.Generator().manual_seed(1),
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert aux["t_n"].shape == (2,)
    assert aux["t_next"].shape == (2,)


def test_aux_timesteps_are_within_scheduler_bounds():
    torch.manual_seed(0)
    batch = _toy_batch(batch_size=5)
    beta_scheduler, sigma_scheduler = _real_schedulers()
    teacher = ConstantDecoder(pred_l=torch.zeros(3, 3), pred_x=torch.zeros(3))
    student = ConstantDecoder(pred_l=torch.zeros(3, 3), pred_x=torch.zeros(3))

    _, aux = consistency_distillation_loss(
        teacher=teacher,
        student=student,
        batch=batch,
        num_steps=8,
        beta_scheduler=beta_scheduler,
        sigma_scheduler=sigma_scheduler,
        generator=torch.Generator().manual_seed(1),
    )

    assert (aux["t_n"] >= 0).all() and (aux["t_n"] <= 1000).all()
    assert (aux["t_next"] >= 0).all() and (aux["t_next"] <= 1000).all()
    assert (aux["t_next"] > aux["t_n"]).all()


def test_loss_uses_real_sigma_norm_not_a_sigma_squared_approximation():
    # Direct comparison: run the same training step against the real
    # SigmaScheduler vs. a scheduler whose sigmas_norm is the wrong
    # sigma**2 approximation (the exact bug caught while writing this
    # module -- real sigma_norm[500]~=401 vs naive sigma[500]**2~=0.0025,
    # ~5 orders of magnitude apart). Same teacher/student/batch/indices in
    # both runs, isolating sigma_norm as the only difference; if the
    # implementation used sigma**2 internally, the two losses would match.
    torch.manual_seed(0)
    beta_scheduler, real_sigma_scheduler = _real_schedulers()

    class NaiveSigmaScheduler:
        timesteps = real_sigma_scheduler.timesteps
        sigmas = real_sigma_scheduler.sigmas
        sigmas_norm = real_sigma_scheduler.sigmas**2  # the wrong approximation

    naive_sigma_scheduler = NaiveSigmaScheduler()

    batch = _toy_batch(batch_size=1, num_atoms_per=3)
    teacher = ConstantDecoder(pred_l=torch.zeros(3, 3), pred_x=torch.ones(3))
    student = ConstantDecoder(pred_l=torch.zeros(3, 3), pred_x=torch.zeros(3))

    class FixedIndexPair:
        @staticmethod
        def __call__(*args, **kwargs):
            # A realistic distillation gap (num_steps=8 over timesteps=1000
            # jumps ~125 steps), away from idx=0: at the exact boundary
            # sigma_n=0 forces pred_x0=x_n unconditionally regardless of
            # sigma_norm (an earlier version of this test used idx_n=0 and
            # the mid-range sigma_norm difference this test targets never
            # actually got exercised).
            return torch.tensor([250]), torch.tensor([375])

    import src.l35.training_step as training_step_module

    original = training_step_module.sample_index_pair
    training_step_module.sample_index_pair = FixedIndexPair()
    try:
        real_loss, _ = consistency_distillation_loss(
            teacher=teacher,
            student=student,
            batch=batch,
            num_steps=8,
            beta_scheduler=beta_scheduler,
            sigma_scheduler=real_sigma_scheduler,
        )
        naive_loss, _ = consistency_distillation_loss(
            teacher=teacher,
            student=student,
            batch=batch,
            num_steps=8,
            beta_scheduler=beta_scheduler,
            sigma_scheduler=naive_sigma_scheduler,
        )
    finally:
        training_step_module.sample_index_pair = original

    assert real_loss.item() > 0.0
    assert naive_loss.item() > 0.0
    ratio = real_loss.item() / naive_loss.item()
    assert ratio > 100 or ratio < 0.01, (
        f"loss barely changed ({ratio=:.4g}) between real and naive sigma_norm -- "
        "implementation may not actually be using sigma_scheduler.sigmas_norm"
    )
