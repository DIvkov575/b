"""Trains ProteinBERT on either the Boltz-MSA-augmented data or the RCSB
single-sequence baseline, with identical hyperparameters, for the L40 ablation.

Run:
    .venv-l38/bin/python -m src.l40.train_pilot --variant boltz \
        --data-path /path/to/rcsb_processed_msa --max-files 2500 --epochs 5
    .venv-l38/bin/python -m src.l40.train_pilot --variant baseline \
        --data-path src/l40/data_cache/rcsb_baseline.jsonl --max-files 2500 --epochs 5
"""
import argparse
import json
import os
from pathlib import Path

# nn.TransformerEncoder's padding-mask fast path calls an op MPS doesn't
# implement (aten::_nested_tensor_from_mask_left_aligned); this falls back to
# CPU for that one op instead of crashing. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.optim as optim

from src.l40.baseline_data import create_baseline_dataloaders
from src.l40.model import create_model
from src.l40.msa_data import create_dataloaders

OUT_DIR = Path(__file__).resolve().parent / "pilot_out"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            out = model(input_ids, attention_mask, labels)
            total_loss += out['loss'].item()
            predictions = torch.argmax(out['logits'], dim=-1)
            mask = labels != -100
            correct += ((predictions == labels) & mask).sum().item()
            total += mask.sum().item()
    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0.0
    model.train()
    return avg_loss, accuracy


def run_training(variant: str, data_path: str, out_path: str, max_files: int,
                  max_length: int = 512, batch_size: int = 32, epochs: int = 5,
                  mask_prob: float = 0.15, sequences_per_file: int = 5,
                  lr: float = 1e-4, d_model: int = 256, n_layers: int = 6,
                  n_heads: int = 8, d_ff: int = 1024, seed: int = 0) -> dict:
    device = get_device()
    print(f"[{variant}] device: {device}", flush=True)

    if variant == "boltz":
        train_loader, val_loader, _ = create_dataloaders(
            data_path, batch_size=batch_size, max_length=max_length,
            max_files=max_files, mask_prob=mask_prob, sequences_per_file=sequences_per_file,
        )
    elif variant == "baseline":
        train_loader, val_loader, _ = create_baseline_dataloaders(
            data_path, batch_size=batch_size, max_length=max_length,
            max_files=max_files, mask_prob=mask_prob,
        )
    else:
        raise ValueError(f"unknown variant: {variant}")

    # Same seed for both variants: both models start from identical initial
    # weights, so any difference in the final metric is attributable to the
    # data, not to which variant happened to draw a luckier init.
    torch.manual_seed(seed)
    model = create_model(
        vocab_size=24, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        d_ff=d_ff, max_length=max_length, dropout=0.1,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    result = {"variant": variant, "max_files": max_files, "epochs": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            out = model(input_ids, attention_mask, labels)
            loss = out['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches
        val_loss, val_accuracy = evaluate(model, val_loader, device)

        print(f"[{variant}] epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_accuracy:.4f}", flush=True)

        result["epochs"].append({
            "epoch": epoch + 1, "train_loss": train_loss,
            "val_loss": val_loss, "val_accuracy": val_accuracy,
        })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["boltz", "baseline"])
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--out-path", default=None)
    parser.add_argument("--max-files", type=int, default=2500)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--sequences-per-file", type=int, default=5,
                         help="Boltz variant only: MSA homolog sequences sampled per structure")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0,
                         help="Use the same value for both variants for a controlled comparison")
    args = parser.parse_args()

    out_path = args.out_path or str(OUT_DIR / f"{args.variant}_results.json")

    run_training(
        variant=args.variant, data_path=args.data_path, out_path=out_path,
        max_files=args.max_files, max_length=args.max_length, batch_size=args.batch_size,
        epochs=args.epochs, sequences_per_file=args.sequences_per_file, lr=args.lr,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, d_ff=args.d_ff,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
