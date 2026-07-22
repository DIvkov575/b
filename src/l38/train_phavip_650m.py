"""L39 v3 -- fine-tune ESM-2-650M on the real PhaVIP/ESM-PVP-lineage phage
protein corpus (79K sequences, real RefSeq-derived data, not the earlier
UniProt-taxonomy approximation).

Scale-up rationale: ESM-PVP's own paper uses ESM-2-650M (18x larger than
L39's original 35M) -- matching that scale directly attacks the "not a fair
comparison" gap flagged in docs/L39_PHAGE_ESM_FINETUNE.md's Refinement
section, using real compute budget rather than staying at the fast/cheap
35M scale.

Run: .venv-l38/bin/python -m src.l38.train_phavip_650m
"""
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

MAX_LENGTH = 700  # covers the 90th percentile (618) of non-PVP sequence lengths in this dataset
BATCH_SIZE = 2  # 650M model @ len=700 OOMs on a 23GB A10G even at batch=8 in fp32 (confirmed via smoke test);
GRAD_ACCUM_STEPS = 8  # effective batch size 16, matching the 35M run's BATCH_SIZE
N_EPOCHS = 3
LR = 1e-5  # lower than the 35M run's 2e-5 -- larger models are more prone to destabilizing at higher LR
MASK_PROB = 0.15
SEED = 0
EVAL_FRAC = 0.05  # 79K sequences is large enough that 5% (~3,958) is still a solid eval slice
USE_BF16 = True  # A10G (Ampere) supports bf16; halves activation memory vs fp32, no loss-scaling needed unlike fp16


def main():
    set_train_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, model: {MODEL_NAME}", flush=True)

    train_seqs, eval_seqs = load_mlm_corpus(eval_frac=EVAL_FRAC, seed=SEED, max_len=MAX_LENGTH)
    print(f"train={len(train_seqs)} eval={len(eval_seqs)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}", flush=True)

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

    model.train()
    n_steps_per_epoch = -(-len(loader) // GRAD_ACCUM_STEPS)  # ceil division
    print(f"\n=== FINE-TUNING ({N_EPOCHS} epochs, {len(loader)} micro-batches / "
          f"{n_steps_per_epoch} optimizer steps per epoch, grad_accum={GRAD_ACCUM_STEPS}) ===", flush=True)
    t0 = time.time()
    for epoch in range(N_EPOCHS):
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

        avg_loss = epoch_loss / len(loader)
        print(f"epoch {epoch+1} done, avg_loss={avg_loss:.4f}", flush=True)

        ft_loss, ft_ppl = compute_pseudo_perplexity(model, tokenizer, eval_seqs, device, max_length=MAX_LENGTH, batch_size=2)
        print(f"  [eval after epoch {epoch+1}] loss={ft_loss:.4f} ppl={ft_ppl:.4f}", flush=True)
        model.train()

    print("\n=== FINAL RESULTS ===", flush=True)
    final_loss, final_ppl = compute_pseudo_perplexity(model, tokenizer, eval_seqs, device, max_length=MAX_LENGTH, batch_size=2)

    results = {
        "model": MODEL_NAME,
        "n_params": n_params,
        "base_loss": base_loss, "base_ppl": base_ppl,
        "final_loss": final_loss, "final_ppl": final_ppl,
        "ppl_improvement_pct": 100 * (base_ppl - final_ppl) / base_ppl,
        "n_train": len(train_seqs), "n_eval": len(eval_seqs),
        "n_epochs": N_EPOCHS, "batch_size": BATCH_SIZE, "grad_accum_steps": GRAD_ACCUM_STEPS,
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
