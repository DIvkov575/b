"""Real pretrained DiffCSP CSP-task checkpoint (mp_csp, from the paper's own
Google Drive release, sha256 c06719217c940718823f7c67b58c74a0fcfba4c3e4312b23cdd22b7a15ea6b5f),
vendored at third_party/diffcsp_checkpoints/mp_csp/. Confirms the real teacher
weights load into the real (patched-import) CSPDiffusion module with an exact
state_dict match -- the actual teacher for L35 distillation, not a fresh/random init.
"""
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


# Absolute: diffcsp.common.utils does os.chdir(PROJECT_ROOT) as an import side
# effect (see fixture above), so a relative path here would resolve against
# the wrong cwd once diffcsp modules are imported.
CKPT_PATH = os.path.abspath("third_party/diffcsp_checkpoints/mp_csp/last.ckpt")
HPARAMS_PATH = os.path.abspath("third_party/diffcsp_checkpoints/mp_csp/hparams.yaml")


def _build_model_from_real_hparams():
    import src.l35.torch_scatter_compat_shim  # noqa: F401
    from omegaconf import OmegaConf

    from diffcsp.pl_modules.diffusion import CSPDiffusion

    with open(HPARAMS_PATH) as f:
        hparams = yaml.safe_load(f)

    # Keep decoder/beta_scheduler/sigma_scheduler nested under the full `model`
    # node (not extracted in isolation) so ${model.max_neighbors}-style
    # interpolations inside decoder can still resolve against their siblings.
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
    return model


def test_real_checkpoint_file_matches_recorded_hash():
    import hashlib

    sha256 = hashlib.sha256()
    with open(CKPT_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha256.update(chunk)

    assert sha256.hexdigest() == "c06719217c940718823f7c67b58c74a0fcfba4c3e4312b23cdd22b7a15ea6b5f"


def test_real_checkpoint_loads_into_real_cspdiffusion_with_exact_match():
    import torch

    model = _build_model_from_real_hparams()
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    # strict=True: raises on any missing/unexpected key -- proves the real
    # checkpoint's parameter names and shapes exactly match this session's
    # from-source CSPDiffusion/CSPNet construction, not a loose/partial load.
    result = model.load_state_dict(ckpt["state_dict"], strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_loaded_teacher_produces_finite_noise_prediction_on_real_batch():
    import torch

    from tests.l35.test_real_data_slice import _load_n_real_rows

    model = _build_model_from_real_hparams()
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

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

    with torch.no_grad():
        output = model(batch)

    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["loss_lattice"])
    assert torch.isfinite(output["loss_coord"])
