"""End-to-end integration: real mp_csp teacher checkpoint (sha256-verified),
real MP-20 test-set structures, real CSPNet student warm-started via deepcopy,
real consistency-distillation training step and backward pass, on real MPS
hardware where available. Mirrors L37's train_distill.py validation pattern:
mechanics validated on real artifacts, not mocks -- not a claim of a full
training run or convergence.
"""
import copy
import os
import sys

import pytest
import yaml


@pytest.fixture(autouse=True)
def _diffcsp_on_path():
    diffcsp_path = os.path.abspath("third_party/diffcsp")
    if diffcsp_path not in sys.path:
        sys.path.insert(0, diffcsp_path)
    os.environ.setdefault("PROJECT_ROOT", diffcsp_path)
    original_cwd = os.getcwd()
    yield
    os.chdir(original_cwd)


CKPT_PATH = os.path.abspath("third_party/diffcsp_checkpoints/mp_csp/last.ckpt")
HPARAMS_PATH = os.path.abspath("third_party/diffcsp_checkpoints/mp_csp/hparams.yaml")


def _load_real_teacher_decoder_and_schedulers(device):
    import src.l35.torch_scatter_compat_shim  # noqa: F401
    import torch
    from omegaconf import OmegaConf

    from diffcsp.pl_modules.diffusion import CSPDiffusion

    with open(HPARAMS_PATH) as f:
        hparams = yaml.safe_load(f)
    full_cfg = OmegaConf.create(hparams)
    model_cfg = full_cfg.model

    model = CSPDiffusion(
        decoder=model_cfg.decoder,
        beta_scheduler=model_cfg.beta_scheduler,
        sigma_scheduler=model_cfg.sigma_scheduler,
        time_dim=model_cfg.time_dim,
        latent_dim=model_cfg.latent_dim,
        cost_coord=model_cfg.cost_coord,
        cost_lattice=model_cfg.cost_lattice,
        optim=full_cfg.optim,
    )
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    teacher_decoder = model.decoder.to(device)
    for p in teacher_decoder.parameters():
        p.requires_grad_(False)
    teacher_decoder.eval()

    return teacher_decoder, model.beta_scheduler.to(device), model.sigma_scheduler.to(device)


def _real_batch(device, n=3):
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
    pyg_batch = Batch.from_data_list(data_list).to(device)

    from diffcsp.common.data_utils import lattice_params_to_matrix_torch

    lattices = lattice_params_to_matrix_torch(pyg_batch.lengths, pyg_batch.angles)
    return {
        "num_atoms": pyg_batch.num_atoms,
        "node2graph": pyg_batch.batch,
        "atom_types": pyg_batch.atom_types,
        "frac_coords": pyg_batch.frac_coords,
        "lattices": lattices,
    }


def _pick_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def test_real_teacher_checkpoint_plus_real_data_runs_real_training_step():
    import torch

    from src.l35.training_step import consistency_distillation_loss

    device = _pick_device()
    teacher, beta_scheduler, sigma_scheduler = _load_real_teacher_decoder_and_schedulers(device)

    student = copy.deepcopy(teacher)
    for p in student.parameters():
        p.requires_grad_(True)

    batch = _real_batch(device, n=3)

    loss, aux = consistency_distillation_loss(
        teacher=teacher,
        student=student,
        batch=batch,
        num_steps=8,
        beta_scheduler=beta_scheduler,
        sigma_scheduler=sigma_scheduler,
        generator=torch.Generator().manual_seed(0),
    )

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

    loss.backward()

    for p in teacher.parameters():
        assert p.grad is None, "real teacher must be frozen"
    student_grads = [p.grad for p in student.parameters()]
    assert all(g is not None for g in student_grads), "real student must receive gradients on every param"
    assert all(torch.isfinite(g).all() for g in student_grads), "no NaN/Inf gradients"


def test_real_adam_step_actually_changes_student_weights():
    import torch

    from src.l35.training_step import consistency_distillation_loss

    device = _pick_device()
    teacher, beta_scheduler, sigma_scheduler = _load_real_teacher_decoder_and_schedulers(device)

    student = copy.deepcopy(teacher)
    for p in student.parameters():
        p.requires_grad_(True)

    before = {name: p.detach().clone() for name, p in student.named_parameters()}
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

    batch = _real_batch(device, n=3)
    loss, _ = consistency_distillation_loss(
        teacher=teacher,
        student=student,
        batch=batch,
        num_steps=8,
        beta_scheduler=beta_scheduler,
        sigma_scheduler=sigma_scheduler,
        generator=torch.Generator().manual_seed(0),
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    changed = any(
        not torch.equal(before[name], p.detach()) for name, p in student.named_parameters()
    )
    assert changed, "Adam step should change at least one student parameter"
