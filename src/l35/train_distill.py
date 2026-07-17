"""L35 consistency-distillation training script for DiffCSP's mp_csp teacher.

Requires third_party/diffcsp on sys.path (added below) and the
torch_scatter_compat_shim installed before any diffcsp import, since real
torch_scatter fails to build on this machine's toolchain (see
docs/L35_PIPELINE_SPEC.md). Loads the real mp_csp checkpoint (sha256
c06719217c940718823f7c67b58c74a0fcfba4c3e4312b23cdd22b7a15ea6b5f), warm-starts
a student via deepcopy, and runs consistency distillation on a real slice of
MP-20 preprocessed via DiffCSP's own process_one().
"""
import argparse
import copy
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DIFFCSP_ROOT = os.path.join(REPO_ROOT, "third_party", "diffcsp")
if DIFFCSP_ROOT not in sys.path:
    sys.path.insert(0, DIFFCSP_ROOT)
os.environ.setdefault("PROJECT_ROOT", DIFFCSP_ROOT)

import src.l35.torch_scatter_compat_shim  # noqa: E402  (must precede diffcsp imports)

import torch  # noqa: E402
import yaml  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.l35.training_step import consistency_distillation_loss  # noqa: E402


def build_batch_from_preprocess_results(results, device="cpu"):
    """Convert a list of diffcsp.common.data_utils.process_one() result dicts
    into the flat-tensor batch dict consistency_distillation_loss expects.
    Mirrors the PyG Batch.from_data_list construction validated in
    tests/l35/test_real_data_slice.py and test_train_distill_integration.py,
    without needing torch_geometric.data.Batch itself.
    """
    from diffcsp.common.data_utils import lattice_params_to_matrix_torch

    num_atoms_list = []
    atom_types_list = []
    frac_coords_list = []
    lengths_list = []
    angles_list = []

    for result in results:
        frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms = result[
            "graph_arrays"
        ]
        num_atoms_list.append(num_atoms)
        atom_types_list.append(torch.LongTensor(atom_types))
        frac_coords_list.append(torch.Tensor(frac_coords))
        lengths_list.append(torch.Tensor(lengths))
        angles_list.append(torch.Tensor(angles))

    num_atoms_t = torch.tensor(num_atoms_list, dtype=torch.long)
    node2graph = torch.repeat_interleave(torch.arange(len(results)), num_atoms_t)
    lattices = lattice_params_to_matrix_torch(
        torch.stack(lengths_list), torch.stack(angles_list)
    )

    batch = {
        "num_atoms": num_atoms_t.to(device),
        "node2graph": node2graph.to(device),
        "atom_types": torch.cat(atom_types_list).to(device),
        "frac_coords": torch.cat(frac_coords_list).to(device),
        "lattices": lattices.to(device),
    }
    return batch


def load_teacher_and_schedulers(ckpt_path, hparams_path, device):
    from diffcsp.pl_modules.diffusion import CSPDiffusion

    with open(hparams_path) as f:
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
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    teacher_decoder = model.decoder.to(device)
    for p in teacher_decoder.parameters():
        p.requires_grad_(False)
    teacher_decoder.eval()

    return teacher_decoder, model.beta_scheduler.to(device), model.sigma_scheduler.to(device)


def load_pilot_batches(data_csv, num_structures, batch_size, device):
    """Preprocess the first num_structures rows of a real MP-20 CSV (real CIF
    parsing via pymatgen, real graph construction via process_one()), grouped
    into batches of batch_size. Slices the CSV before calling preprocess() --
    DiffCSP's own preprocess() has no row-limit parameter.
    """
    import pandas as pd
    from diffcsp.common.data_utils import process_one

    df = pd.read_csv(data_csv).iloc[:num_structures]
    results = [
        process_one(
            df.iloc[idx],
            niggli=True,
            primitive=True,
            graph_method="crystalnn",
            prop_list=["formation_energy_per_atom"],
            use_space_group=False,
            tol=0.01,
        )
        for idx in range(len(df))
    ]

    batches = []
    for start in range(0, len(results), batch_size):
        chunk = results[start : start + batch_size]
        batches.append(build_batch_from_preprocess_results(chunk, device=device))
    return batches


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=os.path.join(REPO_ROOT, "third_party", "diffcsp_checkpoints", "mp_csp", "last.ckpt"),
    )
    parser.add_argument(
        "--hparams_path",
        type=str,
        default=os.path.join(REPO_ROOT, "third_party", "diffcsp_checkpoints", "mp_csp", "hparams.yaml"),
    )
    parser.add_argument(
        "--data_csv",
        type=str,
        default=os.path.join(DIFFCSP_ROOT, "data", "mp_20", "train.csv"),
    )
    parser.add_argument("--num_structures", type=int, default=256, help="real MP-20 rows for this pilot")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=8, help="few-step student NFE target")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=1.0,
        help="max gradient norm; caps update size regardless of the sampled "
        "timestep's loss magnitude (high-t x0-estimates are 1/sqrt(alphas_cumprod)-"
        "amplified, see docs/L35_PIPELINE_SPEC.md -- an earlier unclipped 27k-"
        "structure/15-epoch run diverged from loss~1.2K at epoch 0 to ~1.2B by "
        "epoch 14 without this)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_out", type=str, default=None)
    args = parser.parse_args()

    device = pick_device()
    print(f"device: {device}")

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    print(f"loading teacher from {args.ckpt_path}")
    teacher, beta_scheduler, sigma_scheduler = load_teacher_and_schedulers(
        args.ckpt_path, args.hparams_path, device
    )

    student = copy.deepcopy(teacher)
    for p in student.parameters():
        p.requires_grad_(True)  # deepcopy inherits requires_grad=False from the frozen teacher
    student.train()

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    print(f"preprocessing {args.num_structures} real MP-20 structures from {args.data_csv}")
    batches = load_pilot_batches(args.data_csv, args.num_structures, args.batch_size, device)
    print(f"{len(batches)} batches of up to {args.batch_size} structures each")

    step = 0
    for epoch in range(args.num_epochs):
        for batch in batches:
            optimizer.zero_grad()
            loss, aux = consistency_distillation_loss(
                teacher=teacher,
                student=student,
                batch=batch,
                num_steps=args.num_steps,
                beta_scheduler=beta_scheduler,
                sigma_scheduler=sigma_scheduler,
                generator=generator,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip_norm)
            optimizer.step()
            print(
                f"[epoch {epoch} step {step}] loss={loss.item():.6f} "
                f"pre_clip_grad_norm={grad_norm.item():.3f} "  # clip_grad_norm_ returns the norm BEFORE clipping (docs); actual applied update is capped at --grad_clip_norm
                f"t_n_mean={aux['t_n'].float().mean().item():.1f} "
                f"t_next_mean={aux['t_next'].float().mean().item():.1f}"
            )
            step += 1

    if args.checkpoint_out:
        torch.save({"student_state_dict": student.state_dict(), "args": vars(args)}, args.checkpoint_out)
        print(f"saved student checkpoint to {args.checkpoint_out}")


if __name__ == "__main__":
    main()
