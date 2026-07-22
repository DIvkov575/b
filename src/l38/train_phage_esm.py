"""L39 -- fine-tune ESM-2 (small) on phage/viral protein sequences.

Hypothesis under test (Sawhney et al. 2025, PeerJ, DOI 10.7717/peerj.19919):
ESM-2's UniRef training corpus underrepresents viral/phage sequences, so a
cheap masked-LM fine-tune on phage-specific data should measurably close a
pre-existing perplexity gap between phage and general-protein sequences --
without needing anything beyond plain `transformers` (no openfold/ESMFold,
no exotic installs).

Run: .venv-l38/bin/python -m src.l38.train_phage_esm [--train-seed N] [--out-suffix STR]

--train-seed controls torch's RNG (masking pattern selection, DataLoader
shuffle order, model-side dropout if any) -- the DATA split (SEED, below)
stays fixed across seed variants so multi-seed runs train/eval on identical
data and isolate training-randomness variance specifically (per L39's
refinement: resolving whether v1 vs v2's ~0.4pt gap on the virion-
classification downstream eval is real or seed noise).
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.l38.phage_data import clean_sequences, parse_fasta, train_eval_split

MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
DATA_DIR = Path(__file__).resolve().parent / "data_cache" / "phage"

MAX_LENGTH = 512
BATCH_SIZE = 16
N_EPOCHS = 3
LR = 2e-5
MASK_PROB = 0.15
SEED = 0  # data-split seed; NOT varied across multi-seed sweeps


def set_train_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLMDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_length=MAX_LENGTH):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        enc = self.tokenizer(
            seq, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def collate_and_mask(batch, tokenizer, mask_prob=MASK_PROB):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = input_ids.clone()

    special_ids = set(tokenizer.all_special_ids)
    probability_matrix = torch.full(input_ids.shape, mask_prob)
    special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in special_ids:
        special_tokens_mask |= (input_ids == sid)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    labels[~masked_indices] = -100
    input_ids = input_ids.clone()
    input_ids[masked_indices] = tokenizer.mask_token_id

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@torch.no_grad()
def compute_pseudo_perplexity(model, tokenizer, sequences, device, max_length=MAX_LENGTH, batch_size=8):
    """Mean per-token MLM loss (exp of it = pseudo-perplexity) on held-out
    sequences, using a FIXED random mask per sequence (seeded) so repeated
    calls on the same sequences are directly comparable across checkpoints."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    gen = torch.Generator().manual_seed(SEED)

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        enc = tokenizer(
            batch_seqs, truncation=True, max_length=max_length,
            padding=True, return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = input_ids.clone()

        special_ids = set(tokenizer.all_special_ids)
        probability_matrix = torch.full(input_ids.shape, MASK_PROB)
        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for sid in special_ids:
            special_tokens_mask |= (input_ids == sid)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix, generator=gen).bool()

        labels[~masked_indices] = -100
        input_ids = input_ids.clone()
        input_ids[masked_indices] = tokenizer.mask_token_id

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        n_masked = (labels != -100).sum().item()
        if n_masked == 0:
            continue
        total_loss += outputs.loss.item() * n_masked
        total_tokens += n_masked

    mean_loss = total_loss / total_tokens if total_tokens else float("nan")
    return mean_loss, float(torch.exp(torch.tensor(mean_loss)))


def load_all_data():
    phage_raw = parse_fasta(DATA_DIR / "caudoviricetes_20k.fasta")
    phage_clean = clean_sequences(phage_raw)
    phage_train, phage_eval = train_eval_split(phage_clean, eval_frac=0.1, seed=SEED)

    general_raw = parse_fasta(DATA_DIR / "general_reviewed_5k.fasta")
    general_clean = clean_sequences(general_raw)
    _, general_eval = train_eval_split(general_clean, eval_frac=0.4, seed=SEED)  # just need a held-out eval slice

    return phage_train, phage_eval, general_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seed", type=int, default=0, help="seeds torch/numpy/random for masking+shuffle order; data split is NOT affected")
    parser.add_argument("--out-suffix", type=str, default="", help="appended to the output dir name, e.g. '_seed1'")
    args = parser.parse_args()

    set_train_seed(args.train_seed)
    out_dir = Path(__file__).resolve().parent / f"phage_finetune_out{args.out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, train_seed: {args.train_seed}, out_dir: {out_dir}", flush=True)

    phage_train, phage_eval, general_eval = load_all_data()
    print(f"phage_train={len(phage_train)} phage_eval={len(phage_eval)} general_eval={len(general_eval)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)

    print("\n=== BASELINE (before fine-tuning) ===", flush=True)
    base_phage_loss, base_phage_ppl = compute_pseudo_perplexity(model, tokenizer, phage_eval, device)
    base_general_loss, base_general_ppl = compute_pseudo_perplexity(model, tokenizer, general_eval, device)
    print(f"base phage:   loss={base_phage_loss:.4f} pseudo-ppl={base_phage_ppl:.4f}", flush=True)
    print(f"base general: loss={base_general_loss:.4f} pseudo-ppl={base_general_ppl:.4f}", flush=True)
    print(f"base GAP (phage - general loss): {base_phage_loss - base_general_loss:.4f}", flush=True)

    dataset = MLMDataset(phage_train, tokenizer)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_and_mask(b, tokenizer),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    print(f"\n=== FINE-TUNING ({N_EPOCHS} epochs, {len(loader)} steps/epoch) ===", flush=True)
    t0 = time.time()
    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

            if step % 100 == 0:
                elapsed = time.time() - t0
                print(f"epoch {epoch+1}/{N_EPOCHS} step {step}/{len(loader)} loss={loss.item():.4f} elapsed={elapsed:.0f}s", flush=True)

        avg_loss = epoch_loss / len(loader)
        print(f"epoch {epoch+1} done, avg_loss={avg_loss:.4f}", flush=True)

        ft_phage_loss, ft_phage_ppl = compute_pseudo_perplexity(model, tokenizer, phage_eval, device)
        ft_general_loss, ft_general_ppl = compute_pseudo_perplexity(model, tokenizer, general_eval, device)
        print(f"  [eval after epoch {epoch+1}] phage: loss={ft_phage_loss:.4f} ppl={ft_phage_ppl:.4f}  "
              f"general: loss={ft_general_loss:.4f} ppl={ft_general_ppl:.4f}", flush=True)
        model.train()

    print("\n=== FINAL RESULTS ===", flush=True)
    final_phage_loss, final_phage_ppl = compute_pseudo_perplexity(model, tokenizer, phage_eval, device)
    final_general_loss, final_general_ppl = compute_pseudo_perplexity(model, tokenizer, general_eval, device)

    results = {
        "base_phage_loss": base_phage_loss, "base_phage_ppl": base_phage_ppl,
        "base_general_loss": base_general_loss, "base_general_ppl": base_general_ppl,
        "final_phage_loss": final_phage_loss, "final_phage_ppl": final_phage_ppl,
        "final_general_loss": final_general_loss, "final_general_ppl": final_general_ppl,
        "phage_loss_improvement": base_phage_loss - final_phage_loss,
        "phage_ppl_improvement_pct": 100 * (base_phage_ppl - final_phage_ppl) / base_phage_ppl,
        "general_loss_change": base_general_loss - final_general_loss,
        "n_train": len(phage_train), "n_phage_eval": len(phage_eval), "n_general_eval": len(general_eval),
        "n_epochs": N_EPOCHS, "model": MODEL_NAME, "train_seed": args.train_seed,
    }
    print(json.dumps(results, indent=2), flush=True)

    model.save_pretrained(out_dir / "final_model")
    tokenizer.save_pretrained(out_dir / "final_model")
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved model + results to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
