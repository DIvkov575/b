"""Trains MSAAwareProteinBERT with independently switchable MSA features
(deletion, profile, cross-homolog attention) for the full L40 ablation.

Run (one call per arm; --out-path defaults per-arm if omitted):
    .venv-l38/bin/python -m src.l40.train_msa_ablation \
        --data-path /path/to/rcsb_processed_msa --max-files 2344 --epochs 3 \
        --use-deletion-features --use-profile --use-msa-module
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

from src.l40.msa_aware_data import create_msa_aware_dataloaders
from src.l40.msa_model import MSAAwareProteinBERT

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
            out = _forward(model, batch, device)
            total_loss += out['loss'].item()
            predictions = torch.argmax(out['logits'], dim=-1)
            labels = batch['labels'].to(device)
            mask = labels != -100
            correct += ((predictions == labels) & mask).sum().item()
            total += mask.sum().item()
    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0.0
    model.train()
    return avg_loss, accuracy


def _forward(model, batch, device):
    return model(
        msa_tokens=batch['msa_tokens'].to(device),
        has_deletion=batch['has_deletion'].to(device),
        deletion_value=batch['deletion_value'].to(device),
        profile=batch['profile'].to(device),
        deletion_mean=batch['deletion_mean'].to(device),
        attention_mask=batch['attention_mask'].to(device),
        labels=batch['labels'].to(device),
    )


def run_msa_ablation_training(data_path: str, out_path: str, max_files: int,
                                max_length: int = 512, batch_size: int = 32, epochs: int = 5,
                                mask_prob: float = 0.15, msa_depth: int = 8,
                                lr: float = 1e-4, d_model: int = 256, n_layers: int = 6,
                                n_heads: int = 8, d_ff: int = 1024,
                                msa_s: int = 64, token_z: int = 32, msa_blocks: int = 2,
                                use_deletion_features: bool = True, use_profile: bool = True,
                                use_msa_module: bool = True, seed: int = 0) -> dict:
    device = get_device()
    arm_name = f"del{use_deletion_features}_prof{use_profile}_msa{use_msa_module}"
    print(f"[{arm_name}] device: {device}", flush=True)

    train_loader, val_loader, _ = create_msa_aware_dataloaders(
        data_path, batch_size=batch_size, max_length=max_length,
        max_files=max_files, msa_depth=msa_depth, mask_prob=mask_prob,
    )

    torch.manual_seed(seed)
    model = MSAAwareProteinBERT(
        vocab_size=24, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        d_ff=d_ff, max_length=max_length, dropout=0.1,
        msa_s=msa_s, token_z=token_z, msa_blocks=msa_blocks,
        use_deletion_features=use_deletion_features, use_profile=use_profile,
        use_msa_module=use_msa_module,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    result = {
        "arm": arm_name, "max_files": max_files,
        "use_deletion_features": use_deletion_features,
        "use_profile": use_profile, "use_msa_module": use_msa_module,
        "epochs": [],
    }

    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = _forward(model, batch, device)
            loss = out['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches
        val_loss, val_accuracy = evaluate(model, val_loader, device)

        print(f"[{arm_name}] epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  "
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
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--out-path", default=None)
    parser.add_argument("--max-files", type=int, default=2344)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--msa-depth", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--msa-s", type=int, default=64)
    parser.add_argument("--token-z", type=int, default=32)
    parser.add_argument("--msa-blocks", type=int, default=2)
    parser.add_argument("--use-deletion-features", action="store_true")
    parser.add_argument("--use-profile", action="store_true")
    parser.add_argument("--use-msa-module", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    arm_name = f"del{args.use_deletion_features}_prof{args.use_profile}_msa{args.use_msa_module}"
    out_path = args.out_path or str(OUT_DIR / f"msa_ablation_{arm_name}.json")

    run_msa_ablation_training(
        data_path=args.data_path, out_path=out_path, max_files=args.max_files,
        max_length=args.max_length, batch_size=args.batch_size, epochs=args.epochs,
        msa_depth=args.msa_depth, lr=args.lr, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff, msa_s=args.msa_s, token_z=args.token_z,
        msa_blocks=args.msa_blocks, use_deletion_features=args.use_deletion_features,
        use_profile=args.use_profile, use_msa_module=args.use_msa_module, seed=args.seed,
    )


if __name__ == "__main__":
    main()
