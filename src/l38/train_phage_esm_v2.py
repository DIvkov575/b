"""L39 v2 -- phage ESM-2 fine-tune with real BERT-style masking corruption.

Change from v1 (train_phage_esm.py): v1 always replaced a "masked" position
with the literal [MASK] token. That trains the encoder to rely on seeing
[MASK] as a strong signal, which specifically hurts embedding quality for a
downstream frozen-embedding probe (virion_eval.py) where no [MASK] token is
ever present at inference time. This version uses the standard BERT
corruption recipe (Devlin et al. 2019): of the positions selected for
"masking", 80% -> [MASK], 10% -> random amino acid, 10% -> left unchanged
(but still included in the loss) -- so the encoder learns to produce good
representations for UNMASKED, possibly-corrupted tokens too, closer to what
it sees at real inference/probing time.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.l38.phage_data import clean_sequences, parse_fasta, train_eval_split
from src.l38.train_phage_esm import MODEL_NAME, compute_pseudo_perplexity, load_all_data, set_train_seed

MAX_LENGTH = 512
BATCH_SIZE = 16
N_EPOCHS = 3
LR = 2e-5
MASK_PROB = 0.15
SEED = 0

STANDARD_AA_TOKENS = list("ACDEFGHIKLMNPQRSTVWY")


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


def bert_style_mask(input_ids, tokenizer, mask_prob=MASK_PROB, seed_generator=None):
    """80% [MASK] / 10% random amino acid / 10% unchanged, all three
    contributing to the loss (standard BERT/Devlin et al. 2019 recipe)."""
    labels = input_ids.clone()

    special_ids = set(tokenizer.all_special_ids)
    probability_matrix = torch.full(input_ids.shape, mask_prob)
    special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in special_ids:
        special_tokens_mask |= (input_ids == sid)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    kwargs = {"generator": seed_generator} if seed_generator is not None else {}
    selected = torch.bernoulli(probability_matrix, **kwargs).bool()
    labels[~selected] = -100

    input_ids = input_ids.clone()

    # of selected positions: 80% -> [MASK], 10% -> random AA, 10% -> unchanged
    rand = torch.rand(input_ids.shape, generator=seed_generator) if seed_generator is not None else torch.rand(input_ids.shape)
    mask_positions = selected & (rand < 0.8)
    random_positions = selected & (rand >= 0.8) & (rand < 0.9)
    # remaining 10% (rand >= 0.9) stay unchanged but are still in `labels`

    input_ids[mask_positions] = tokenizer.mask_token_id

    n_random = random_positions.sum().item()
    if n_random > 0:
        random_aa_ids = torch.tensor(
            [tokenizer.convert_tokens_to_ids(STANDARD_AA_TOKENS[i % len(STANDARD_AA_TOKENS)])
             for i in range(n_random)]
        )
        # shuffle to avoid any positional pattern, then scatter into place
        perm = torch.randperm(n_random)
        input_ids[random_positions] = random_aa_ids[perm]

    return input_ids, labels


def collate_and_mask(batch, tokenizer, mask_prob=MASK_PROB):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    masked_input_ids, labels = bert_style_mask(input_ids, tokenizer, mask_prob)
    return {"input_ids": masked_input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seed", type=int, default=0, help="seeds torch/numpy/random for masking+shuffle order; data split is NOT affected")
    parser.add_argument("--out-suffix", type=str, default="", help="appended to the output dir name, e.g. '_seed1'")
    args = parser.parse_args()

    set_train_seed(args.train_seed)
    out_dir = Path(__file__).resolve().parent / f"phage_finetune_v2_out{args.out_suffix}"
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

    dataset = MLMDataset(phage_train, tokenizer)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_and_mask(b, tokenizer),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    print(f"\n=== FINE-TUNING v2 (BERT-style 80/10/10 masking, {N_EPOCHS} epochs, {len(loader)} steps/epoch) ===", flush=True)
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

    print("\n=== FINAL RESULTS (v2) ===", flush=True)
    final_phage_loss, final_phage_ppl = compute_pseudo_perplexity(model, tokenizer, phage_eval, device)
    final_general_loss, final_general_ppl = compute_pseudo_perplexity(model, tokenizer, general_eval, device)

    results = {
        "version": "v2_bert_style_masking",
        "base_phage_loss": base_phage_loss, "base_phage_ppl": base_phage_ppl,
        "base_general_loss": base_general_loss, "base_general_ppl": base_general_ppl,
        "final_phage_loss": final_phage_loss, "final_phage_ppl": final_phage_ppl,
        "final_general_loss": final_general_loss, "final_general_ppl": final_general_ppl,
        "phage_loss_improvement": base_phage_loss - final_phage_loss,
        "phage_ppl_improvement_pct": 100 * (base_phage_ppl - final_phage_ppl) / base_phage_ppl,
        "n_train": len(phage_train), "n_phage_eval": len(phage_eval), "n_general_eval": len(general_eval),
        "n_epochs": N_EPOCHS, "model": MODEL_NAME, "train_seed": args.train_seed,
    }
    print(json.dumps(results, indent=2), flush=True)

    model.save_pretrained(out_dir / "final_model")
    tokenizer.save_pretrained(out_dir / "final_model")
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved v2 model + results to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
