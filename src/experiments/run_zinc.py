import argparse
import os
from pathlib import Path

import torch
import yaml

from src.data.datasets import get_zinc_loaders
from src.training.trainer import Trainer


def build_model(model_type: str, config: dict, in_dim: int, out_dim: int):
    hidden_dim = config["model"]["hidden_dim"]
    num_layers = config["model"]["num_layers"]
    budget_k = config["selection"]["budget_k"]

    if model_type == "dpp":
        from src.models.dpp_subgraph_gnn import DPPSubgraphGNN

        return DPPSubgraphGNN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            budget_k=budget_k,
            policy=config["selection"]["policy"],
            aggregation="weighted_sum",
        )
    if model_type == "uniform":
        from src.baselines.esan_uniform import ESANUniform

        return ESANUniform(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            budget_k=budget_k,
        )
    if model_type == "full":
        from src.baselines.esan_full import ESANFull

        return ESANFull(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
        )
    raise ValueError(f"Unknown model type: {model_type}")


def run_experiment(model_type: str, config_path: str, epochs_override: int = None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    epochs = epochs_override if epochs_override is not None else config["training"]["epochs"]
    lr = config["training"]["lr"]
    margin_warmup = config["training"]["margin_warmup_epochs"]

    train_loader, val_loader, test_loader = get_zinc_loaders(batch_size=1, root="data/")

    in_dim = 1
    out_dim = 1

    model = build_model(model_type, config, in_dim=in_dim, out_dim=out_dim)
    trainer = Trainer(model, lr=lr, warmup_epochs=margin_warmup, task="regression")

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    best_path = ckpt_dir / f"zinc_{model_type}_best.pt"

    best_val = float("inf")
    for epoch in range(epochs):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)

        val_score = val_metrics.get("mae", 1.0 - val_metrics.get("accuracy", 0.0))
        if val_score < best_val:
            best_val = val_score
            torch.save(model.state_dict(), best_path)

        if epoch % 10 == 0:
            print(
                f"epoch={epoch} cls_loss={train_metrics['cls_loss']:.4f} "
                f"margin_loss={train_metrics['margin_loss']:.4f} val={val_metrics}"
            )

    if best_path.exists():
        model.load_state_dict(torch.load(best_path))
    test_metrics = trainer.evaluate(test_loader)
    print(f"test={test_metrics}")
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="ZINC experiment runner")
    parser.add_argument("--model", choices=["dpp", "uniform", "full"], required=True)
    parser.add_argument("--config", default="configs/zinc.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    run_experiment(args.model, args.config, args.epochs)


if __name__ == "__main__":
    main()
