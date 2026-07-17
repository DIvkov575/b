"""L37 consistency-distillation training script for MDGen's forward-simulation teacher.

Validated end-to-end this session against the real released checkpoint (HF bjing-mit/mdgen,
forward_sim.ckpt) and real downloaded tetrapeptide trajectory data (HF bjing-mit/tetrapeptide-
sims, explicit solvent -- matches the split forward_sim.ckpt was trained/tested on, per its
saved hparams: path_type=GVP, prediction=velocity, sim_condition=True).

Requires third_party/mdgen on sys.path (added below) and its dependencies EXCEPT pyemma
(not needed for training/distillation -- only mdgen/analysis.py and the eval scripts import
it). See docs/L37_PIPELINE_SPEC.md for the full pipeline this script implements Stage C of.
"""
import argparse
import copy
import os
import sys
from functools import partial

MDGEN_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "mdgen")
sys.path.insert(0, os.path.abspath(MDGEN_ROOT))

import numpy as np
import torch

torch.serialization.add_safe_globals([argparse.Namespace])

from mdgen.geometry import atom14_to_frames, atom14_to_atom37, atom37_to_torsions
from mdgen.residue_constants import restype_order
from mdgen.wrapper import NewMDGenWrapper

from src.l37.training_step import consistency_distillation_loss


def load_teacher(ckpt_path, device="cpu", inference_steps=50):
    """Load the frozen MDGen teacher, patched with the (checkpoint-predates-this-flag)
    inference_steps default this session added at mdgen/parsing.py + mdgen/wrapper.py."""
    model = NewMDGenWrapper.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.args.inference_steps = inference_steps
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_student(teacher_model):
    """Student starts as a copy of the teacher's architecture and weights (a warm-started
    few-step student, not a from-scratch architecture -- the cheapest first distillation
    attempt given Stage C is not yet locked to a specific smaller architecture)."""
    student = copy.deepcopy(teacher_model)
    student.train()
    for p in student.parameters():
        p.requires_grad_(True)  # deepcopy inherits requires_grad=False from the frozen teacher
    return student


def load_one_frame_batch(data_dir, name, seqres, suffix="_i100", num_frames=4):
    """Build one training batch from a preprocessed MDGen .npy trajectory file, following
    sim_inference.py's get_batch() convention: a single real frame, expanded across the
    T=num_frames dimension the way rollout() conditions a forward-simulation step."""
    arr = np.lib.format.open_memmap(f"{data_dir}/{name}{suffix}.npy", "r")
    arr = np.copy(arr[0:1]).astype(np.float32)

    frames = atom14_to_frames(torch.from_numpy(arr))
    seqres_t = torch.tensor([restype_order[c] for c in seqres])
    atom37 = torch.from_numpy(atom14_to_atom37(arr, seqres_t[None])).float()
    L = len(seqres_t)
    torsions, torsion_mask = atom37_to_torsions(atom37, seqres_t[None])

    item = {
        "torsions": torsions,
        "torsion_mask": torsion_mask[0],
        "trans": frames._trans,
        "rots": frames._rots._rot_mats,
        "seqres": seqres_t,
        "mask": torch.ones(L),
    }
    batch = torch.utils.data.default_collate([item])

    return {
        "torsions": batch["torsions"].expand(-1, num_frames, -1, -1, -1),
        "torsion_mask": batch["torsion_mask"],
        "trans": batch["trans"].expand(-1, num_frames, -1, -1),
        "rots": batch["rots"].expand(-1, num_frames, -1, -1, -1),
        "seqres": batch["seqres"],
        "mask": batch["mask"],
    }


def training_step_for_peptide(teacher_model, student_model, data_dir, name, seqres,
                               num_steps, generator=None):
    """One consistency-distillation loss + backward for one peptide's batch."""
    raw_batch = load_one_frame_batch(data_dir, name, seqres)
    prep = teacher_model.prep_batch(raw_batch)
    x1, loss_mask, model_kwargs = prep["latents"], prep["loss_mask"], prep["model_kwargs"]

    teacher_fn = partial(teacher_model.model.forward_inference, **model_kwargs)
    student_fn = partial(student_model.model.forward_inference, **model_kwargs)

    x_n = torch.randn_like(x1)
    loss, aux = consistency_distillation_loss(
        teacher=teacher_fn,
        student=student_fn,
        x_n=x_n,
        num_steps=num_steps,
        mask=loss_mask,
        generator=generator,
    )
    return loss, aux


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, help="CSV with name,seqres columns")
    parser.add_argument("--num_steps", type=int, default=4, help="few-step student NFE target")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_pilot_peptides", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import pandas as pd

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    teacher = load_teacher(args.ckpt_path)
    student = build_student(teacher)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    df = pd.read_csv(args.split, index_col="name")
    names = list(df.index[: args.num_pilot_peptides])

    for step, name in enumerate(names):
        seqres = df.seqres[name]
        optimizer.zero_grad()
        loss, aux = training_step_for_peptide(
            teacher, student, args.data_dir, name, seqres, args.num_steps, generator=generator
        )
        loss.backward()
        optimizer.step()
        print(f"[{step}] {name} loss={loss.item():.6f} t_n={aux['t_n'].item():.3f} "
              f"t_next={aux['t_next'].item():.3f}")


if __name__ == "__main__":
    main()
