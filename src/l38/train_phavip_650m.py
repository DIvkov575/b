"""L39 v3 -- fine-tune ESM-2-650M on the real PhaVIP/ESM-PVP-lineage phage
protein corpus (79K sequences, real RefSeq-derived data, not the earlier
UniProt-taxonomy approximation).

Scale-up rationale: ESM-PVP's own paper uses ESM-2-650M (18x larger than
L39's original 35M) -- matching that scale directly attacks the "not a fair
comparison" gap flagged in docs/L39_PHAGE_ESM_FINETUNE.md's Refinement
section, using real compute budget rather than staying at the fast/cheap
35M scale.

Checkpointing: with ~3.4hr/epoch, a save only at the very end risks losing
the entire run to any mid-run failure. Saves a resumable checkpoint every
CHECKPOINT_EVERY_STEPS optimizer steps (~15-20 min apart), plus after every
epoch. Early stopping: tracks held-out eval loss after each epoch and stops
if it fails to improve by more than EARLY_STOP_MIN_DELTA for
EARLY_STOP_PATIENCE consecutive epochs -- avoids burning the full 3-epoch,
~10hr budget once returns diminish.

Run: .venv-l38/bin/python -m src.l38.train_phavip_650m [--resume PATH]
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.l38.phavip_real_data import load_mlm_corpus
from src.l38.train_phage_esm import (
    MLMDataset,
    collate_and_mask,
    compute_pseudo_perplexity,
    set_train_seed,
)
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
OUT_DIR = Path(__file__).resolve().parent / "phavip_650m_finetune_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 700  # covers the 90th percentile (618) of non-PVP sequence lengths in this dataset
BATCH_SIZE = 2  # 650M model @ len=700 OOMs on a 23GB A10G even at batch=8 in fp32 (confirmed via smoke test)
GRAD_ACCUM_STEPS = 8  # effective batch size 16, matching the 35M run's BATCH_SIZE
N_EPOCHS = 3
LR = 1e-5  # lower than the 35M run's 2e-5 -- larger models are more prone to destabilizing at higher LR
MASK_PROB = 0.15
SEED = 0
EVAL_FRAC = 0.05  # 79K sequences is large enough that 5% (~3,958) is still a solid eval slice
USE_BF16 = True  # A10G (Ampere) supports bf16; halves activation memory vs fp32, no loss-scaling needed unlike fp16

CHECKPOINT_EVERY_STEPS = 2500  # micro-batches, not optimizer steps -- ~15min at the observed ~0.35s/step rate
EARLY_STOP_PATIENCE = 1  # consecutive epochs without sufficient improvement before stopping
EARLY_STOP_MIN_DELTA = 0.01  # minimum eval-loss improvement (absolute) to NOT count as a plateau


def save_checkpoint(model, optimizer, epoch, micro_step, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    torch.save(
        {"optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "micro_step": micro_step},
        path / "training_state.pt",
    )
    with open(path / "checkpoint_meta.json", "w") as f:
        json.dump({"epoch": epoch, "micro_step": micro_step}, f)


def load_checkpoint(model, optimizer, path: Path):
    with open(path / "checkpoint_meta.json") as f:
        meta = json.load(f)
    state = torch.load(path / "training_state.pt", map_location="cpu")
    optimizer.load_state_dict(state["optimizer_state_dict"])
    return meta["epoch"], meta["micro_step"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="path to a checkpoint dir to resume model+optimizer state from")
    args = parser.parse_args()

    set_train_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, model: {MODEL_NAME}", flush=True)

    train_seqs, eval_seqs = load_mlm_corpus(eval_frac=EVAL_FRAC, seed=SEED, max_len=MAX_LENGTH)
    print(f"train={len(train_seqs)} eval={len(eval_seqs)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_source = args.resume if args.resume else MODEL_NAME
    model = AutoModelForMaskedLM.from_pretrained(model_source).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}, loaded from: {model_source}", flush=True)

    autocast_dtype = torch.bfloat16 if USE_BF16 else None

    print("\n=== BASELINE (before fine-tuning) ===", flush=True)
    base_loss, base_ppl = compute_pseudo_perplexity(model, tokenizer, eval_seqs, device, max_length=MAX_LENGTH, batch_size=2)
    print(f"base: loss={base_loss:.4f} pseudo-ppl={base_ppl:.4f}", flush=True)

    dataset = MLMDataset(train_seqs, tokenizer, max_length=MAX_LENGTH)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_and_mask(b, tokenizer, mask_prob=MASK_PROB),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    start_epoch = 0
    if args.resume:
        start_epoch, resumed_micro_step = load_checkpoint(model, optimizer, Path(args.resume))
        print(f"resumed from epoch {start_epoch}, micro_step {resumed_micro_step} (restarting this epoch from step 0 -- "
              f"DataLoader shuffle order isn't checkpointed, so full-epoch replay is simpler and cheap relative to epoch length)", flush=True)

    model.train()
    n_steps_per_epoch = -(-len(loader) // GRAD_ACCUM_STEPS)  # ceil division
    print(f"\n=== FINE-TUNING ({N_EPOCHS} epochs, {len(loader)} micro-batches / "
          f"{n_steps_per_epoch} optimizer steps per epoch, grad_accum={GRAD_ACCUM_STEPS}) ===", flush=True)

    eval_losses = []
    t0 = time.time()
    for epoch in range(start_epoch, N_EPOCHS):
        epoch_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=USE_BF16):
                outputs = model(**batch)
                loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()
            epoch_loss += outputs.loss.item()

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(loader):
                optimizer.step()
                optimizer.zero_grad()

            if step % 200 == 0:
                elapsed = time.time() - t0
                print(f"epoch {epoch+1}/{N_EPOCHS} micro-step {step}/{len(loader)} "
                      f"loss={outputs.loss.item():.4f} elapsed={elapsed:.0f}s", flush=True)

            if step > 0 and step % CHECKPOINT_EVERY_STEPS == 0:
                ckpt_path = CHECKPOINT_DIR / "latest"
                save_checkpoint(model, optimizer, epoch, step, ckpt_path)
                print(f"  [checkpoint saved: epoch={epoch} micro_step={step} -> {ckpt_path}]", flush=True)

        avg_loss = epoch_loss / len(loader)
        print(f"epoch {epoch+1} done, avg_loss={avg_loss:.4f}", flush=True)

        ft_loss, ft_ppl = compute_pseudo_perplexity(model, tokenizer, eval_seqs, device, max_length=MAX_LENGTH, batch_size=2)
        print(f"  [eval after epoch {epoch+1}] loss={ft_loss:.4f} ppl={ft_ppl:.4f}", flush=True)
        eval_losses.append(ft_loss)
        model.train()

        # Reuse the same "latest" slot rather than keeping one 7.3GB checkpoint
        # per epoch -- 3 epochs x 7.3GB + the mid-epoch "latest" saves would
        # approach this instance's 31GB free disk. A single rolling
        # checkpoint is enough to resume from the most recent good state.
        save_checkpoint(model, optimizer, epoch + 1, 0, CHECKPOINT_DIR / "latest")
        print(f"  [end-of-epoch checkpoint saved -> {CHECKPOINT_DIR / 'latest'}]", flush=True)

        if len(eval_losses) > EARLY_STOP_PATIENCE:
            recent = eval_losses[-(EARLY_STOP_PATIENCE + 1):]
            best_before = min(recent[:-1])
            if recent[-1] > best_before - EARLY_STOP_MIN_DELTA:
                print(f"\n=== EARLY STOP: eval loss failed to improve by >{EARLY_STOP_MIN_DELTA} "
                      f"over the last {EARLY_STOP_PATIENCE} epoch(s) (history: {eval_losses}) ===", flush=True)
                break

    print("\n=== FINAL RESULTS ===", flush=True)
    final_loss, final_ppl = compute_pseudo_perplexity(model, tokenizer, eval_seqs, device, max_length=MAX_LENGTH, batch_size=2)

    results = {
        "model": MODEL_NAME,
        "n_params": n_params,
        "base_loss": base_loss, "base_ppl": base_ppl,
        "final_loss": final_loss, "final_ppl": final_ppl,
        "ppl_improvement_pct": 100 * (base_ppl - final_ppl) / base_ppl,
        "n_train": len(train_seqs), "n_eval": len(eval_seqs),
        "n_epochs_completed": len(eval_losses), "n_epochs_planned": N_EPOCHS,
        "eval_loss_by_epoch": eval_losses,
        "stopped_early": len(eval_losses) < N_EPOCHS,
        "batch_size": BATCH_SIZE, "grad_accum_steps": GRAD_ACCUM_STEPS,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
        "lr": LR, "max_length": MAX_LENGTH, "bf16": USE_BF16,
    }
    print(json.dumps(results, indent=2), flush=True)

    model.save_pretrained(OUT_DIR / "final_model")
    tokenizer.save_pretrained(OUT_DIR / "final_model")
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved model + results to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
