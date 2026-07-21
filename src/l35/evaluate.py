"""L35 evaluation: compares a few-step distilled student against the real
many-step teacher on DiffCSP's own CSP-task metric (match rate + RMSD
against ground truth, RecEval in third_party/diffcsp/scripts/compute_metrics.py),
reusing that real eval machinery directly rather than reimplementing it.

Requires third_party/diffcsp/scripts on sys.path in addition to
third_party/diffcsp itself (train_distill.py only adds the latter).
"""
import argparse
import copy
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DIFFCSP_ROOT = os.path.join(REPO_ROOT, "third_party", "diffcsp")
DIFFCSP_SCRIPTS = os.path.join(DIFFCSP_ROOT, "scripts")
for _p in (DIFFCSP_ROOT, DIFFCSP_SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("PROJECT_ROOT", DIFFCSP_ROOT)

import src.l35.torch_scatter_compat_shim  # noqa: E402  (must precede diffcsp imports)

import torch  # noqa: E402

from eval_utils import lattices_to_params_shape  # noqa: E402

import src.l35.smact_validity_none_oxidation_states_shim  # noqa: E402  (must follow eval_utils import, patches its smact_validity)
from src.l35.sample import few_step_sample  # noqa: E402
from src.l35.train_distill import load_teacher_and_schedulers, preprocess_with_cache  # noqa: E402


def split_sample_into_crystal_dicts(frac_coords, lattices, atom_types, num_atoms):
    """Split a flat (frac_coords, lattices, atom_types) sample -- the shape
    few_step_sample()/CSPDiffusion.sample() return -- into a list of
    per-graph crys_array_dict, converting each graph's 3x3 lattice matrix
    into (lengths, angles) via DiffCSP's own lattices_to_params_shape,
    matching the shape compute_metrics.Crystal expects.
    """
    lengths, angles = lattices_to_params_shape(lattices)

    dicts = []
    start = 0
    for i, n in enumerate(num_atoms.tolist()):
        dicts.append(
            {
                "frac_coords": frac_coords[start : start + n].detach().cpu().numpy(),
                "atom_types": atom_types[start : start + n].detach().cpu().numpy(),
                "lengths": lengths[i].detach().cpu().numpy(),
                "angles": angles[i].detach().cpu().numpy(),
            }
        )
        start += n
    return dicts


def ground_truth_crystal_dict(preprocess_result):
    """Build a crys_array_dict from one diffcsp.common.data_utils.process_one()
    result -- the real ground-truth structure this sample is being compared
    against.
    """
    frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms = (
        preprocess_result["graph_arrays"]
    )
    return {
        "frac_coords": frac_coords,
        "atom_types": atom_types,
        "lengths": lengths,
        "angles": angles,
    }


def validity_rates(crys_list):
    """Fraction of a Crystal list that is constructed (valid pymatgen
    Structure at all), comp_valid (smact charge-balance), struct_valid (no
    collapsed/too-close atoms), and valid (both) -- a diagnostic ladder for
    WHY a sampler's match_rate is low: raw-geometry collapse vs.
    chemically/physically implausible vs. merely not-matching-ground-truth.
    """
    n = len(crys_list)
    return {
        "constructed": sum(c.constructed for c in crys_list) / n,
        "comp_valid": sum(c.comp_valid for c in crys_list) / n,
        "struct_valid": sum(c.struct_valid for c in crys_list) / n,
        "valid": sum(c.valid for c in crys_list) / n,
    }


def run_eval(
    sampler_configs, results, beta_scheduler, sigma_scheduler, max_timestep, device, generator=None,
):
    """For each real ground-truth structure in results, and for each
    (name, network, num_steps) in sampler_configs: sample via few_step_sample
    from the real ground-truth composition, then run RecEval + validity_rates
    against ground truth.

    sampler_configs lets a single eval run directly compare e.g.
    ("teacher@1000", teacher, 1000), ("teacher@8", teacher, 8), and
    ("student@8", student, 8) against the SAME ground-truth structures with
    the SAME sampler code path -- isolating whether a low few-step match
    rate is caused by few-step sampling itself (teacher@8 also collapses) or
    by the distillation training specifically (student@8 is worse than
    teacher@8).

    Every config sees the SAME starting noise for a given structure,
    regardless of sweep order or composition: few_step_sample's only
    randomness is its initial (l_t, x_t) draw (the deterministic DDIM/PF-ODE
    step loop draws no further noise), so the generator is reseeded from
    (generator.initial_seed(), structure_index) before each config's call.
    A version that instead let one generator advance sequentially across
    every (structure, config) pair was caught giving a DIFFERENT match_rate
    for the identical checkpoint/structures/seed depending on which other
    configs were swept alongside it (0.18 alone vs. 0.38 as part of a
    [4,8,16]-step sweep) -- pure generator-state drift, not a real
    quality difference.

    Returns a dict {name: {"match_rate":..., "rms_dist":..., **validity_rates}}.
    """
    from compute_metrics import Crystal, RecEval

    base_seed = generator.initial_seed() if generator is not None else None

    gt_dicts = []
    sampled_dicts = {name: [] for name, _, _ in sampler_configs}

    for i, result in enumerate(results):
        gt_dict = ground_truth_crystal_dict(result)
        gt_dicts.append(gt_dict)

        atom_types = torch.LongTensor(gt_dict["atom_types"]).to(device)
        num_atoms = torch.tensor([len(gt_dict["atom_types"])], device=device)
        node2graph = torch.zeros(len(gt_dict["atom_types"]), dtype=torch.long, device=device)

        for name, network, num_steps in sampler_configs:
            if generator is not None:
                generator.manual_seed(base_seed + i)
            fc, lattices = few_step_sample(
                network, atom_types, num_atoms, node2graph, num_steps,
                beta_scheduler, sigma_scheduler, max_timestep, generator=generator,
            )
            sampled_dicts[name].extend(split_sample_into_crystal_dicts(fc, lattices, atom_types, num_atoms))

    gt_crys = [Crystal(d) for d in gt_dicts]

    all_metrics = {}
    for name, _, _ in sampler_configs:
        crys = [Crystal(d) for d in sampled_dicts[name]]
        metrics = RecEval(crys, gt_crys).get_metrics()
        metrics.update(validity_rates(crys))
        all_metrics[name] = metrics
    return all_metrics


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
    parser.add_argument("--student_ckpt", type=str, required=True)
    parser.add_argument(
        "--data_csv",
        type=str,
        default=os.path.join(DIFFCSP_ROOT, "data", "mp_20", "test.csv"),
    )
    parser.add_argument("--num_eval_structures", type=int, default=100)
    parser.add_argument(
        "--student_num_steps", type=str, default="8",
        help="comma-separated NFE values to sweep, e.g. '4,8,16'",
    )
    parser.add_argument("--teacher_num_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    teacher, beta_scheduler, sigma_scheduler = load_teacher_and_schedulers(
        args.ckpt_path, args.hparams_path, device
    )

    student = copy.deepcopy(teacher)
    student_ckpt = torch.load(args.student_ckpt, map_location=device, weights_only=False)
    student.load_state_dict(student_ckpt["student_state_dict"])
    student.eval()

    cache_path = os.path.join(
        os.path.dirname(args.data_csv), f".preprocess_cache_eval_{args.num_eval_structures}.pt"
    )
    results = preprocess_with_cache(args.data_csv, args.num_eval_structures, cache_path)

    # Three-way comparison isolates WHY a low few-step match rate happens:
    # teacher@few_steps vs teacher@many_steps separates "few-step sampling
    # itself collapses" from "distillation training specifically hurts"
    # (student@few_steps vs teacher@few_steps, same step count, same sampler).
    # Swept across multiple NFE values in one pass (same ground truth, same
    # generator state per structure) for an efficiency-quality curve rather
    # than a single operating point.
    student_num_steps_list = [int(s) for s in args.student_num_steps.split(",")]
    sampler_configs = [(f"teacher@{args.teacher_num_steps}", teacher, args.teacher_num_steps)]
    for n in student_num_steps_list:
        sampler_configs.append((f"teacher@{n}", teacher, n))
        sampler_configs.append((f"student@{n}", student, n))
    all_metrics = run_eval(
        sampler_configs, results, beta_scheduler, sigma_scheduler,
        max_timestep=beta_scheduler.timesteps, device=device, generator=generator,
    )

    for name, metrics in all_metrics.items():
        print(f"{name}: {metrics}")


if __name__ == "__main__":
    main()
