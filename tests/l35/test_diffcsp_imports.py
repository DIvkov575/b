"""DiffCSP pins torch_scatter, a C++ extension that fails to build on this machine.
Installing src.l35.torch_scatter_compat_shim before importing vendored DiffCSP code
lets the unmodified third_party/diffcsp source import cleanly on modern torch+MPS.

diffcsp.common.utils requires a PROJECT_ROOT env var (per the repo's own
.env.template) and calls os.chdir(PROJECT_ROOT) as an import side effect --
the fixture sets the env var and restores cwd after each test.
"""
import os
import sys

import pytest


@pytest.fixture(autouse=True)
def _diffcsp_on_path():
    diffcsp_path = os.path.abspath("third_party/diffcsp")
    if diffcsp_path not in sys.path:
        sys.path.insert(0, diffcsp_path)
    os.environ.setdefault("PROJECT_ROOT", diffcsp_path)

    original_cwd = os.getcwd()
    yield
    os.chdir(original_cwd)


def test_install_shim_before_import_lets_cspnet_import():
    import src.l35.torch_scatter_compat_shim  # noqa: F401  (installs sys.modules entries)

    from diffcsp.pl_modules.cspnet import CSPNet

    assert CSPNet is not None


def test_cspnet_forward_runs_on_real_small_batch():
    import src.l35.torch_scatter_compat_shim  # noqa: F401
    import torch

    from diffcsp.pl_modules.cspnet import CSPNet

    torch.manual_seed(0)
    # two tiny "crystals": 3 atoms and 2 atoms, default fully-connected edge_style
    num_atoms = torch.tensor([3, 2])
    node2graph = torch.tensor([0, 0, 0, 1, 1])
    atom_types = torch.tensor([1, 6, 8, 1, 6])  # 1-indexed atomic numbers
    frac_coords = torch.rand(5, 3)
    lattices = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    t_emb = torch.randn(2, 256)

    net = CSPNet(hidden_dim=32, latent_dim=256, num_layers=2, edge_style="fc")
    lattice_out, coord_out = net(t_emb, atom_types, frac_coords, lattices, num_atoms, node2graph)

    assert lattice_out.shape == (2, 3, 3)
    assert coord_out.shape == (5, 3)
    assert not torch.isnan(lattice_out).any()
    assert not torch.isnan(coord_out).any()


def test_cspdiffusion_module_instantiates_from_real_config():
    import src.l35.torch_scatter_compat_shim  # noqa: F401
    import hydra
    from omegaconf import OmegaConf

    from diffcsp.pl_modules.diffusion import CSPDiffusion

    decoder_cfg = OmegaConf.create({
        "_target_": "diffcsp.pl_modules.cspnet.CSPNet",
        "hidden_dim": 32,
        "latent_dim": 256,
        "num_layers": 2,
        "max_atoms": 100,
        "act_fn": "silu",
        "dis_emb": "sin",
        "num_freqs": 16,
        "edge_style": "fc",
        "cutoff": 7.0,
        "max_neighbors": 20,
        "ln": True,
        "ip": True,
    })
    beta_cfg = OmegaConf.create({
        "_target_": "diffcsp.pl_modules.diff_utils.BetaScheduler",
        "timesteps": 50,
        "scheduler_mode": "cosine",
    })
    sigma_cfg = OmegaConf.create({
        "_target_": "diffcsp.pl_modules.diff_utils.SigmaScheduler",
        "timesteps": 50,
        "sigma_begin": 0.01,
        "sigma_end": 1.0,
    })
    optim_cfg = OmegaConf.create({
        "optim": {
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-4},
            "use_lr_scheduler": False,
        }
    })

    model = CSPDiffusion(
        decoder=decoder_cfg,
        beta_scheduler=beta_cfg,
        sigma_scheduler=sigma_cfg,
        time_dim=256,
        latent_dim=0,
        cost_lattice=1.0,
        cost_coord=1.0,
        optim=optim_cfg["optim"],
    )

    assert model.beta_scheduler.timesteps == 50
    assert model.decoder.num_layers == 2
