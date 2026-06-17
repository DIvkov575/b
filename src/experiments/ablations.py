"""Ablation runner: trains DPP, ESAN-uniform, ESAN-full, and centrality on ZINC."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.baselines.centrality_select import CentralitySelect
from src.baselines.esan_full import ESANFull
from src.baselines.esan_uniform import ESANUniform
from src.data.datasets import get_zinc_loaders
from src.models.dpp_subgraph_gnn import DPPSubgraphGNN
from src.training.trainer import Trainer


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_models(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, budget_k: int):
    return {
        "dpp": DPPSubgraphGNN(in_dim, hidden_dim, out_dim, num_layers, budget_k),
        "uniform": ESANUniform(in_dim, hidden_dim, out_dim, num_layers, budget_k),
        "centrality": CentralitySelect(in_dim, hidden_dim, out_dim, num_layers, budget_k),
        "full": ESANFull(in_dim, hidden_dim, out_dim, num_layers),
    }


def _train_one(name, model, train_loader, val_loader, test_loader, epochs, lr, margin_weight, warmup_epochs, task="classification"):
    trainer = Trainer(model, lr=lr, margin_weight=margin_weight, warmup_epochs=warmup_epochs, task=task)
    history = []
    best_val = None
    best_test = None
    start = time.time()
    for epoch in range(epochs):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)
        test_metrics = trainer.evaluate(test_loader)
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
            }
        )
        if task == "regression":
            val_score = val_metrics.get("mae", float("inf"))
            is_better = best_val is None or val_score < best_val
        else:
            val_score = val_metrics.get("accuracy", 0.0)
            is_better = best_val is None or val_score > best_val
        if is_better:
            best_val = val_score
            best_test = test_metrics
    elapsed = time.time() - start
    return {
        "model": name,
        "best_val": best_val,
        "best_test": best_test,
        "history": history,
        "wall_time_sec": elapsed,
    }


def run_ablation(budget_k: int = 5, epochs: int = 100, seed: int = 0):
    _set_seed(seed)

    train_loader, val_loader, test_loader = get_zinc_loaders(batch_size=1, root="data/zinc")

    sample = next(iter(train_loader))
    in_dim = int(sample.x.shape[-1]) if sample.x is not None else 1
    hidden_dim = 64
    num_layers = 4
    out_dim = 1

    models = _build_models(in_dim, hidden_dim, out_dim, num_layers, budget_k)

    results = {}
    for name, model in models.items():
        results[name] = _train_one(
            name=name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=1e-3,
            margin_weight=0.1,
            warmup_epochs=min(10, max(1, epochs // 10)),
            task="regression",
        )

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ablation_k{budget_k}_seed{seed}.json"
    payload = {
        "budget_k": budget_k,
        "epochs": epochs,
        "seed": seed,
        "results": results,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=float)

    return payload


def run_single_model(model_name, budget_k=5, epochs=100, seed=0):
    """Train a single model and save results to its own file."""
    _set_seed(seed)
    train_loader, val_loader, test_loader = get_zinc_loaders(batch_size=1, root="data/")

    sample = next(iter(train_loader))
    in_dim = sample.x.shape[1] if sample.x is not None else 1
    hidden_dim = 64
    out_dim = 1
    num_layers = 4

    models = _build_models(in_dim, hidden_dim, out_dim, num_layers, budget_k)
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    model = models[model_name]
    result = _train_one(
        name=model_name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=epochs,
        lr=1e-3,
        margin_weight=0.1,
        warmup_epochs=min(10, max(1, epochs // 10)),
        task="regression",
    )

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ablation_{model_name}_k{budget_k}_seed{seed}.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"{model_name}: MAE={result['best_test']['mae']:.4f}, time={result['wall_time_sec']:.0f}s")
    return result


def _parse_args():
    parser = argparse.ArgumentParser(description="Run DPP vs baselines ablation on ZINC.")
    parser.add_argument("--budget-k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default=None, help="Run single model (dpp/uniform/centrality/full)")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.model:
        run_single_model(args.model, budget_k=args.budget_k, epochs=args.epochs, seed=args.seed)
    else:
        run_ablation(budget_k=args.budget_k, epochs=args.epochs, seed=args.seed)


if __name__ == "__main__":
    main()
