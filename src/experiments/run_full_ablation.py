"""Full ablation runner with pre-cached subgraphs and mega-batched encoding."""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import numpy as np
from torch_geometric.data import Batch

from src.data.datasets import get_zinc_loaders
from src.data.subgraph_policies import node_deletion_subgraphs, strip_subgraph
from src.models.base_gnn import GINEncoder
from src.models.bag_aggregator import BagAggregator
from src.models.dpp_selector import DPPSelector
from src.models.margin_scorer import MarginScorer
from src.training.losses import classification_loss, dpp_margin_loss
from src.training.margin_utils import compute_margin


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def precache_subgraphs(loader, budget_k):
    """Generate and cache subgraphs for all graphs in loader."""
    cached = []
    labels = []
    for g in loader:
        subs = [strip_subgraph(s) for s in node_deletion_subgraphs(g)]
        cached.append(subs)
        y = g.y
        if y.dim() == 0:
            y = y.unsqueeze(0)
        labels.append(y)
    return cached, labels


class CachedModel:
    """Wrapper for training with pre-cached subgraphs."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, budget_k, method, aggregation="weighted_sum"):
        self.method = method
        self.budget_k = budget_k
        self.hidden_dim = hidden_dim
        self._training = True

        self.encoder = GINEncoder(in_dim, hidden_dim, hidden_dim, num_layers=num_layers)
        self.aggregator = BagAggregator(hidden_dim, aggregation)
        self.classifier = torch.nn.Linear(hidden_dim, out_dim)

        if method == "dpp":
            self.margin_scorer = MarginScorer(hidden_dim)
            self.dpp_selector = DPPSelector(hidden_dim, budget_k)
        else:
            self.margin_scorer = None
            self.dpp_selector = None

    def parameters(self):
        params = list(self.encoder.parameters()) + list(self.aggregator.parameters()) + list(self.classifier.parameters())
        if self.margin_scorer:
            params += list(self.margin_scorer.parameters())
        return params

    def train(self):
        self.encoder.train()
        self.aggregator.train()
        self.classifier.train()
        if self.margin_scorer:
            self.margin_scorer.train()
        self._training = True

    def eval(self):
        self.encoder.eval()
        self.aggregator.eval()
        self.classifier.eval()
        if self.margin_scorer:
            self.margin_scorer.eval()
        self._training = False

    def select_subgraphs(self, all_subs):
        """Select k subgraphs per graph based on method."""
        selected = []
        for subs in all_subs:
            k = min(self.budget_k, len(subs))
            if k == 0:
                selected.append([])
                continue

            if self.method == "uniform":
                selected.append(random.sample(subs, k))
            elif self.method == "centrality":
                # Degree centrality: pick subgraphs where the DELETED node had highest degree
                # Subgraph with FEWER edges = deleted a high-degree node = more informative
                scores = []
                for s in subs:
                    n_edges = s.edge_index.shape[1] if s.edge_index.numel() > 0 else 0
                    scores.append(n_edges)
                topk = sorted(range(len(subs)), key=lambda i: scores[i])[:k]
                selected.append([subs[i] for i in topk])
            elif self.method in ("full", "dpp"):
                selected.append(subs[:k] if self.method == "full" else subs)
            else:
                selected.append(subs[:k])
        return selected

    def forward_batch(self, all_subs, labels_tensor):
        """Mega-batched forward pass."""
        if self.method == "full":
            per_graph_subs = all_subs
        elif self.method == "dpp":
            per_graph_subs = all_subs  # DPP selects after encoding
        else:
            per_graph_subs = self.select_subgraphs(all_subs)

        # Mega-batch all subgraphs
        mega = []
        boundaries = []
        offset = 0
        for subs in per_graph_subs:
            k = min(self.budget_k, len(subs)) if self.method != "full" else len(subs)
            batch_subs = subs[:k] if self.method != "dpp" else subs
            mega.extend(batch_subs)
            boundaries.append((offset, offset + len(batch_subs)))
            offset += len(batch_subs)

        if not mega:
            return None, None

        batch = Batch.from_data_list(mega)
        all_embs = self.encoder(batch)

        # Per-graph aggregation
        graph_reprs = []
        quality_scores_list = []
        for i, (start, end) in enumerate(boundaries):
            embs = all_embs[start:end]

            if embs.shape[0] == 0:
                graph_reprs.append(torch.zeros(self.hidden_dim))
                quality_scores_list.append(torch.zeros(0))
                continue

            if self.method == "dpp":
                graph_context = embs.mean(dim=0)
                qs = self.margin_scorer(embs, graph_context)
                quality_scores_list.append(qs)

                if self._training:
                    marginals = self.dpp_selector.soft_select(embs, qs)
                    w = marginals / marginals.sum().clamp(min=1e-8)
                    graph_reprs.append(self.aggregator(embs, w))
                else:
                    selected = self.dpp_selector(embs.detach(), qs.detach())
                    sel_embs = embs[torch.tensor(selected, dtype=torch.long)]
                    w = torch.ones(len(selected)) / max(len(selected), 1)
                    graph_reprs.append(self.aggregator(sel_embs, w))
            else:
                w = torch.ones(embs.shape[0]) / max(embs.shape[0], 1)
                graph_reprs.append(self.aggregator(embs, w))
                quality_scores_list.append(torch.zeros(0))

        logits = self.classifier(torch.stack(graph_reprs))
        return logits, quality_scores_list


def _eval_mae(model, cache, labels, budget_k, batch_size=32):
    """Evaluate MAE in batches to avoid OOM."""
    total_ae = 0.0
    total_n = 0
    with torch.no_grad():
        for i in range(0, len(cache), batch_size):
            batch_subs = cache[i:i + batch_size]
            batch_labels = torch.cat(labels[i:i + batch_size])
            logits, _ = model.forward_batch(batch_subs, batch_labels)
            if logits is not None:
                total_ae += torch.abs(logits.view(-1) - batch_labels.view(-1).float()).sum().item()
                total_n += batch_labels.numel()
    return total_ae / max(total_n, 1)


def train_one_model(method, in_dim, hidden_dim, out_dim, num_layers, budget_k,
                    train_cache, train_labels, val_cache, val_labels,
                    test_cache, test_labels, epochs, lr, margin_weight, warmup_epochs):
    """Train a single model with pre-cached subgraphs."""
    model = CachedModel(in_dim, hidden_dim, out_dim, num_layers, budget_k, method)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    batch_size = 32
    n_train = len(train_cache)
    best_val = float("inf")
    best_test = None

    for epoch in range(epochs):
        model.train()
        indices = list(range(n_train))
        random.shuffle(indices)

        cls_total = 0.0
        margin_total = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_subs = [train_cache[j] for j in batch_idx]
            batch_labels = torch.cat([train_labels[j] for j in batch_idx])

            optimizer.zero_grad()
            logits, qs_list = model.forward_batch(batch_subs, batch_labels)
            if logits is None:
                continue

            cls_loss = classification_loss(logits, batch_labels, task="regression")
            loss = cls_loss

            if epoch >= warmup_epochs and method == "dpp":
                margins = compute_margin(logits.detach(), batch_labels)
                m_losses = []
                for j, qs in enumerate(qs_list):
                    if qs.numel() > 0:
                        m_broadcast = margins[j].expand(qs.shape[0])
                        m_losses.append(dpp_margin_loss(qs, m_broadcast))
                if m_losses:
                    m_loss = torch.stack(m_losses).mean()
                    loss = loss + margin_weight * m_loss
                    margin_total += m_loss.item()

            loss.backward()
            optimizer.step()
            cls_total += cls_loss.item()
            n_batches += 1

        # Evaluate (batched to avoid OOM)
        model.eval()
        val_mae = _eval_mae(model, val_cache, val_labels, budget_k, batch_size)
        if val_mae < best_val:
            best_val = val_mae
            test_mae = _eval_mae(model, test_cache, test_labels, budget_k, batch_size)
            best_test = {"mae": test_mae, "epoch": epoch}

        if epoch % 10 == 0:
            print(f"  [{method}] epoch={epoch} train_loss={cls_total/max(n_batches,1):.4f} val_mae={val_mae:.4f} best_test_mae={best_test['mae']:.4f}")

    return best_test


def run_full_ablation(budget_k=5, epochs=100, seed=0):
    set_seed(seed)
    print(f"=== Ablation: k={budget_k}, epochs={epochs}, seed={seed} ===")

    # Load data
    train_loader, val_loader, test_loader = get_zinc_loaders(batch_size=1, root="data/")

    # Pre-cache subgraphs
    t0 = time.time()
    print("Pre-caching subgraphs...")
    train_cache, train_labels = precache_subgraphs(train_loader, budget_k)
    val_cache, val_labels = precache_subgraphs(val_loader, budget_k)
    test_cache, test_labels = precache_subgraphs(test_loader, budget_k)
    print(f"  Cached in {time.time()-t0:.1f}s (train={len(train_cache)}, val={len(val_cache)}, test={len(test_cache)})")

    # Train all models
    results = {}
    for method in ["dpp", "uniform", "centrality", "full"]:
        t0 = time.time()
        print(f"\nTraining: {method}")
        best = train_one_model(
            method=method,
            in_dim=1, hidden_dim=64, out_dim=1, num_layers=4,
            budget_k=budget_k,
            train_cache=train_cache, train_labels=train_labels,
            val_cache=val_cache, val_labels=val_labels,
            test_cache=test_cache, test_labels=test_labels,
            epochs=epochs, lr=1e-3, margin_weight=0.1,
            warmup_epochs=min(10, max(1, epochs // 10)),
        )
        elapsed = time.time() - t0
        results[method] = {**best, "wall_time_sec": elapsed}
        print(f"  Done: MAE={best['mae']:.4f} @ epoch {best['epoch']}, time={elapsed:.0f}s")

    # Save
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ablation_batched_k{budget_k}_seed{seed}.json"
    with out_path.open("w") as f:
        json.dump({"budget_k": budget_k, "epochs": epochs, "seed": seed, "results": results}, f, indent=2)

    print(f"\n=== Final Results (k={budget_k}, seed={seed}) ===")
    print(f"{'Method':<12} {'MAE':<10} {'Time':<8}")
    print("-" * 30)
    for method, r in results.items():
        print(f"{method:<12} {r['mae']:<10.4f} {r['wall_time_sec']:<8.0f}s")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_full_ablation(args.budget_k, args.epochs, args.seed)


if __name__ == "__main__":
    main()
