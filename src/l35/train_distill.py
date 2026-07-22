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
from p_tqdm import p_umap  # noqa: E402

from src.l35.consistency_distill import ema_update  # noqa: E402
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


def _build_and_load_csp_diffusion(ckpt_path, hparams_path):
    """Constructs the real, full CSPDiffusion LightningModule (decoder +
    schedulers + the real sample()/forward() methods) from the mp_csp
    checkpoint. Shared by load_teacher_and_schedulers (decoder-only callers,
    the consistency-distillation training/eval path) and
    load_teacher_module (callers that need the whole module, i.e. anyone
    calling the REAL DiffCSP sample() as a baseline rather than the
    dual-track DDIM/PF-ODE reimplementation in sample.py).
    """
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
    return model


def load_teacher_and_schedulers(ckpt_path, hparams_path, device):
    model = _build_and_load_csp_diffusion(ckpt_path, hparams_path)

    teacher_decoder = model.decoder.to(device)
    for p in teacher_decoder.parameters():
        p.requires_grad_(False)
    teacher_decoder.eval()

    return teacher_decoder, model.beta_scheduler.to(device), model.sigma_scheduler.to(device)


def load_teacher_module(ckpt_path, hparams_path, device):
    """Like load_teacher_and_schedulers, but returns the full CSPDiffusion
    module (not just its decoder) so callers can invoke the module's own
    real sample() -- DiffCSP's published stochastic annealed-Langevin
    predictor-corrector sampler -- rather than the deterministic DDIM/PF-ODE
    reimplementation in sample.py. Frozen (no_grad-equivalent via
    requires_grad_(False) on every parameter), eval mode, same guarantees as
    load_teacher_and_schedulers's decoder.
    """
    model = _build_and_load_csp_diffusion(ckpt_path, hparams_path)
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


def process_one(*args, **kwargs):
    """Thin re-export so tests can monkeypatch this module's own reference
    (preprocess_with_cache calls the name bound here, not diffcsp's directly)
    to assert a cache hit skips reprocessing entirely.
    """
    from diffcsp.common.data_utils import process_one as _process_one

    return _process_one(*args, **kwargs)


def preprocess_with_cache(data_csv, num_structures, cache_path, num_workers=None, sample_seed=None):
    """Preprocess num_structures rows of a real MP-20 CSV (real CIF parsing
    via pymatgen, real graph construction), caching the result to cache_path.
    Mirrors diffcsp.pl_data.dataset.CrystDataset.preprocess()'s own
    os.path.exists(save_path) -> torch.load / else preprocess+torch.save
    convention. The cache is keyed by (data_csv, num_structures, sample_seed)
    stored alongside the results, so a different num_structures OR a
    different sample_seed against the same cache_path is correctly treated
    as a miss rather than silently reusing a cache built for a different
    slice.

    sample_seed=None (default) takes the first num_structures rows in file
    order -- the training pilot's own use (an arbitrary but fixed slice of
    train.csv is fine for a pilot). sample_seed=<int> instead draws a
    reproducible random sample of num_structures rows from the FULL csv via
    numpy's default_rng, sorted back to original row order. This matters for
    eval sets specifically: an early version of the eval pipeline always
    took df.iloc[:200] of test.csv as the "held-out 200 structures" --
    confirmed on the real MP-20 test.csv that this first-200 slice's mean
    spacegroup number differs measurably from the remaining ~8,846 rows
    (material_id ordering is not randomized upstream), so a fixed-seed
    random draw is needed for the eval set to be representative rather than
    an accidental non-random stratum of the same 200 rows every time.

    Uses p_umap for real multi-core parallelism (num_workers=None lets
    p_umap use all available cores), matching diffcsp.common.data_utils.
    preprocess()'s own parallelization -- a plain sequential loop was a
    real regression here: confirmed via `ps`/`top` on a live 4-vCPU EC2
    instance mid-run (one process at 100% CPU, load average ~1.0/4, zero
    worker child processes) leaving 3 of 4 cores idle. p_umap returns
    results in COMPLETION order, not submission order, so results are
    remapped back to the CSV's original row order via material_id, exactly
    matching preprocess()'s own mpid_to_results dict-remap.
    """
    import numpy as np
    import pandas as pd

    if os.path.exists(cache_path):
        cached = torch.load(cache_path, weights_only=False)
        if (
            cached.get("data_csv") == data_csv
            and cached.get("num_structures") == num_structures
            and cached.get("sample_seed") == sample_seed
        ):
            return cached["results"]

    full_df = pd.read_csv(data_csv)
    if sample_seed is None:
        df = full_df.iloc[:num_structures]
    else:
        rng = np.random.default_rng(sample_seed)
        chosen_idx = rng.choice(len(full_df), size=num_structures, replace=False)
        df = full_df.iloc[np.sort(chosen_idx)]

    unordered_results = p_umap(
        process_one,
        [df.iloc[idx] for idx in range(len(df))],
        [True] * len(df),
        [True] * len(df),
        ["crystalnn"] * len(df),
        [["formation_energy_per_atom"]] * len(df),
        [False] * len(df),
        [0.01] * len(df),
        num_cpus=num_workers,
    )
    mpid_to_results = {result["mp_id"]: result for result in unordered_results}
    results = [mpid_to_results[df.iloc[idx]["material_id"]] for idx in range(len(df))]

    torch.save(
        {
            "data_csv": data_csv,
            "num_structures": num_structures,
            "sample_seed": sample_seed,
            "results": results,
        },
        cache_path,
    )
    return results


def default_cache_path(data_csv, num_structures):
    """Deterministic cache location next to the source CSV, keyed by
    (csv filename, num_structures) so different pilots/full-runs against
    the same split don't collide.
    """
    csv_dir = os.path.dirname(os.path.abspath(data_csv))
    csv_name = os.path.splitext(os.path.basename(data_csv))[0]
    return os.path.join(csv_dir, f".preprocess_cache_{csv_name}_{num_structures}.pt")


def load_pilot_batches(data_csv, num_structures, batch_size, device, cache_path=None):
    """Preprocess (cached) real MP-20 structures, grouped into batches of
    batch_size. cache_path defaults to a deterministic location next to
    data_csv if not given.
    """
    if cache_path is None:
        cache_path = default_cache_path(data_csv, num_structures)
    results = preprocess_with_cache(data_csv, num_structures, cache_path)

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
    parser.add_argument(
        "--num_steps", type=int, default=8,
        help="training-time discretization N: the number of adjacent-index pairs "
        "sampled from DiffCSP's full {0..timesteps} grid per Song et al. 2023 "
        "Algorithm 2. This is NOT an inference-time NFE target -- it is independent "
        "of how many steps the trained student is later SAMPLED at (see "
        "src/l35/sample.py's multistep_consistency_sample, which can run the same "
        "trained student at any NFE via consistency_sampling_grid). Training one "
        "separate student per target NFE, as this project originally did, defeats "
        "the point of consistency distillation; train once with a reasonably fine N "
        "and vary NFE only at sampling time.",
    )
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
    parser.add_argument(
        "--log_every", type=int, default=1,
        help="print loss/grad_norm/weight_norm every N steps (weight_norm walks all "
        "params each time it's logged -- real but bounded overhead)",
    )
    parser.add_argument(
        "--ema_mu", type=float, default=0.999,
        help="EMA decay for the target network (Song et al. 2023 Eq. 8): "
        "target <- mu*target + (1-mu)*student after every optimizer step. "
        "Using the live student as its own target (mu effectively N/A, no EMA "
        "at all) was confirmed to destabilize training at full-dataset scale "
        "-- see training_step.py's module docstring.",
    )
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

    target_network = copy.deepcopy(student)
    for p in target_network.parameters():
        p.requires_grad_(False)
    target_network.eval()

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
                target_network=target_network,
                batch=batch,
                num_steps=args.num_steps,
                beta_scheduler=beta_scheduler,
                sigma_scheduler=sigma_scheduler,
                generator=generator,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip_norm)
            optimizer.step()
            ema_update(target_network, student, mu=args.ema_mu)
            if step % args.log_every == 0:
                with torch.no_grad():
                    weight_norm = torch.sqrt(
                        sum(p.detach().float().pow(2).sum() for p in student.parameters())
                    )
                print(
                    f"[epoch {epoch} step {step}] loss={loss.item():.6f} "
                    f"pre_clip_grad_norm={grad_norm.item():.3f} "  # clip_grad_norm_ returns the norm BEFORE clipping (docs); actual applied update is capped at --grad_clip_norm
                    f"weight_norm={weight_norm.item():.3f} "
                    f"t_n_mean={aux['t_n'].float().mean().item():.1f} "
                    f"t_next_mean={aux['t_next'].float().mean().item():.1f}"
                )
            step += 1

    if args.checkpoint_out:
        torch.save({"student_state_dict": student.state_dict(), "args": vars(args)}, args.checkpoint_out)
        print(f"saved student checkpoint to {args.checkpoint_out}")


if __name__ == "__main__":
    main()
