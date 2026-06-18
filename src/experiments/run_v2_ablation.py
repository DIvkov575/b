"""V2 ablation: DPP with hard selection + LOO supervision vs baselines."""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Batch

from src.data.datasets import get_zinc_loaders
from src.data.subgraph_policies import node_deletion_subgraphs, strip_subgraph
from src.models.base_gnn import GINEncoder
from src.models.bag_aggregator import BagAggregator
from src.models.dpp_selector import DPPSelector
from src.models.margin_scorer import MarginScorer
from src.training.losses import classification_loss
from src.training.margin_utils import compute_margin


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def precache_subgraphs(loader):
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


class DPPv2Model:
    """Hard selection at train+eval. LOO-supervised quality scorer."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, budget_k):
        self.budget_k = budget_k
        self.hidden_dim = hidden_dim
        self._training = True

        self.encoder = GINEncoder(in_dim, hidden_dim, hidden_dim, num_layers=num_layers)
        self.margin_scorer = MarginScorer(hidden_dim)
        self.dpp_selector = DPPSelector(hidden_dim, budget_k)
        self.aggregator = BagAggregator(hidden_dim, "weighted_sum")
        self.classifier = torch.nn.Linear(hidden_dim, out_dim)

    def parameters(self):
        params = (list(self.encoder.parameters()) + list(self.margin_scorer.parameters()) +
                  list(self.aggregator.parameters()) + list(self.classifier.parameters()))
        return params

    def train(self):
        self.encoder.train()
        self.margin_scorer.train()
        self.aggregator.train()
        self.classifier.train()
        self._training = True

    def eval(self):
        self.encoder.eval()
        self.margin_scorer.eval()
        self.aggregator.eval()
        self.classifier.eval()
        self._training = False

    def forward_batch(self, all_subs):
        """Mega-batch encode, per-graph hard DPP select, aggregate."""
        mega = []
        boundaries = []
        offset = 0
        for subs in all_subs:
            mega.extend(subs)
            boundaries.append((offset, offset + len(subs)))
            offset += len(subs)

        if not mega:
            return None, None, None

        batch = Batch.from_data_list(mega)
        all_embs = self.encoder(batch)

        graph_reprs = []
        all_quality_scores = []
        all_loo_scores = []

        for start, end in boundaries:
            embs = all_embs[start:end]
            n = embs.shape[0]

            if n == 0:
                graph_reprs.append(torch.zeros(self.hidden_dim))
                all_quality_scores.append(torch.zeros(0))
                all_loo_scores.append(torch.zeros(0))
                continue

            graph_context = embs.mean(dim=0)
            qs = self.margin_scorer(embs, graph_context)
            all_quality_scores.append(qs)

            # Hard DPP selection (same at train and eval)
            k = min(self.budget_k, n)
            selected = self.dpp_selector(embs.detach(), qs.detach())

            # Exploration: during training, replace 1 selected subgraph with a random non-selected one (20% of the time)
            if self._training and n > k and random.random() < 0.2:
                non_selected = [i for i in range(n) if i not in selected]
                if non_selected:
                    replace_idx = random.randint(0, len(selected) - 1)
                    selected[replace_idx] = random.choice(non_selected)

            sel_idx = torch.tensor(selected, dtype=torch.long)
            sel_embs = embs[sel_idx]

            # Aggregate with uniform normalized weights
            w = torch.ones(len(selected)) / len(selected)
            graph_emb = self.aggregator(sel_embs, w)
            graph_reprs.append(graph_emb)

            # LOO: importance of each selected subgraph (only for selected ones)
            if self._training and len(selected) > 1:
                loo = torch.zeros(len(selected))
                full_pred = self.classifier(graph_emb).detach()
                for j_local in range(len(selected)):
                    mask = torch.ones(len(selected), dtype=torch.bool)
                    mask[j_local] = False
                    rem = sel_embs[mask]
                    w_rem = torch.ones(rem.shape[0]) / rem.shape[0]
                    agg_rem = self.aggregator(rem, w_rem)
                    pred_without = self.classifier(agg_rem).detach()
                    loo[j_local] = (full_pred - pred_without).abs().sum()
                all_loo_scores.append((selected, loo))
            else:
                all_loo_scores.append(None)

        logits = self.classifier(torch.stack(graph_reprs))
        return logits, all_quality_scores, all_loo_scores


class UniformModel:
    """Uniform random selection baseline (same structure for fair comparison)."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, budget_k):
        self.budget_k = budget_k
        self.hidden_dim = hidden_dim

        self.encoder = GINEncoder(in_dim, hidden_dim, hidden_dim, num_layers=num_layers)
        self.aggregator = BagAggregator(hidden_dim, "weighted_sum")
        self.classifier = torch.nn.Linear(hidden_dim, out_dim)

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.aggregator.parameters()) + list(self.classifier.parameters())

    def train(self):
        self.encoder.train()
        self.aggregator.train()
        self.classifier.train()

    def eval(self):
        self.encoder.eval()
        self.aggregator.eval()
        self.classifier.eval()

    def forward_batch(self, all_subs):
        mega = []
        boundaries = []
        offset = 0
        for subs in all_subs:
            k = min(self.budget_k, len(subs))
            sampled = random.sample(subs, k) if len(subs) > k else subs
            mega.extend(sampled)
            boundaries.append((offset, offset + len(sampled)))
            offset += len(sampled)

        if not mega:
            return None, None, None

        batch = Batch.from_data_list(mega)
        all_embs = self.encoder(batch)

        graph_reprs = []
        for start, end in boundaries:
            embs = all_embs[start:end]
            if embs.shape[0] == 0:
                graph_reprs.append(torch.zeros(self.hidden_dim))
            else:
                w = torch.ones(embs.shape[0]) / embs.shape[0]
                graph_reprs.append(self.aggregator(embs, w))

        logits = self.classifier(torch.stack(graph_reprs))
        return logits, None, None


class CentralityModel:
    """Centrality selection baseline."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, budget_k):
        self.budget_k = budget_k
        self.hidden_dim = hidden_dim

        self.encoder = GINEncoder(in_dim, hidden_dim, hidden_dim, num_layers=num_layers)
        self.aggregator = BagAggregator(hidden_dim, "weighted_sum")
        self.classifier = torch.nn.Linear(hidden_dim, out_dim)

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.aggregator.parameters()) + list(self.classifier.parameters())

    def train(self):
        self.encoder.train()
        self.aggregator.train()
        self.classifier.train()

    def eval(self):
        self.encoder.eval()
        self.aggregator.eval()
        self.classifier.eval()

    def forward_batch(self, all_subs):
        mega = []
        boundaries = []
        offset = 0
        for subs in all_subs:
            k = min(self.budget_k, len(subs))
            # Select subgraphs with fewest edges (= deleted high-degree node)
            scores = [s.edge_index.shape[1] if s.edge_index.numel() > 0 else 0 for s in subs]
            topk = sorted(range(len(subs)), key=lambda i: scores[i])[:k]
            selected = [subs[i] for i in topk]
            mega.extend(selected)
            boundaries.append((offset, offset + len(selected)))
            offset += len(selected)

        if not mega:
            return None, None, None

        batch = Batch.from_data_list(mega)
        all_embs = self.encoder(batch)

        graph_reprs = []
        for start, end in boundaries:
            embs = all_embs[start:end]
            if embs.shape[0] == 0:
                graph_reprs.append(torch.zeros(self.hidden_dim))
            else:
                w = torch.ones(embs.shape[0]) / embs.shape[0]
                graph_reprs.append(self.aggregator(embs, w))

        logits = self.classifier(torch.stack(graph_reprs))
        return logits, None, None


def train_model(model, train_cache, train_labels, val_cache, val_labels,
                test_cache, test_labels, epochs, lr, margin_weight, warmup_epochs,
                is_dpp=False):
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
        loo_total = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_subs = [train_cache[j] for j in batch_idx]
            batch_labels = torch.cat([train_labels[j] for j in batch_idx])

            optimizer.zero_grad()
            logits, qs_list, loo_list = model.forward_batch(batch_subs)
            if logits is None:
                continue

            cls_loss = F.l1_loss(logits.view(-1), batch_labels.view(-1).float())
            loss = cls_loss

            # LOO-supervised quality score loss (DPP only)
            # Only train scorer on SELECTED subgraphs — don't push non-selected to 0
            if is_dpp and epoch >= warmup_epochs and qs_list is not None:
                loo_losses = []
                for qs, loo_info in zip(qs_list, loo_list):
                    if loo_info is None or qs.numel() == 0:
                        continue
                    selected_idx, loo_scores = loo_info
                    if loo_scores.sum() <= 0:
                        continue
                    # Only supervise quality scores of selected subgraphs
                    sel_qs = qs[torch.tensor(selected_idx, dtype=torch.long)]
                    loo_norm = loo_scores / loo_scores.max().clamp(min=1e-8)
                    qs_pred = torch.sigmoid(sel_qs)
                    loo_losses.append(F.mse_loss(qs_pred, loo_norm))
                if loo_losses:
                    loo_loss = torch.stack(loo_losses).mean()
                    loss = loss + margin_weight * loo_loss
                    loo_total += loo_loss.item()

            loss.backward()
            optimizer.step()
            cls_total += cls_loss.item()
            n_batches += 1

        # Eval
        model.eval()
        val_mae = _eval_mae(model, val_cache, val_labels, batch_size)
        if val_mae < best_val:
            best_val = val_mae
            test_mae = _eval_mae(model, test_cache, test_labels, batch_size)
            best_test = {"mae": test_mae, "epoch": epoch}

        if epoch % 10 == 0:
            print(f"  epoch={epoch} loss={cls_total/max(n_batches,1):.4f} "
                  f"loo_loss={loo_total/max(n_batches,1):.4f} "
                  f"val_mae={val_mae:.4f} best_test={best_test['mae']:.4f}")

    return best_test


def _eval_mae(model, cache, labels, batch_size=32):
    total_ae = 0.0
    total_n = 0
    with torch.no_grad():
        for i in range(0, len(cache), batch_size):
            batch_subs = cache[i:i + batch_size]
            batch_labels = torch.cat(labels[i:i + batch_size])
            logits, _, _ = model.forward_batch(batch_subs)
            if logits is not None:
                total_ae += torch.abs(logits.view(-1) - batch_labels.view(-1).float()).sum().item()
                total_n += batch_labels.numel()
    return total_ae / max(total_n, 1)


def run_v2_ablation(budget_k=5, epochs=100, seed=0):
    set_seed(seed)
    print(f"=== V2 Ablation: k={budget_k}, epochs={epochs}, seed={seed} ===")

    train_loader, val_loader, test_loader = get_zinc_loaders(batch_size=1, root="data/")

    t0 = time.time()
    print("Pre-caching...")
    train_cache, train_labels = precache_subgraphs(train_loader)
    val_cache, val_labels = precache_subgraphs(val_loader)
    test_cache, test_labels = precache_subgraphs(test_loader)
    print(f"  Cached in {time.time()-t0:.1f}s")

    results = {}
    for method in ["dpp_v2", "uniform", "centrality"]:
        t0 = time.time()
        print(f"\nTraining: {method}")

        if method == "dpp_v2":
            model = DPPv2Model(1, 64, 1, 4, budget_k)
            is_dpp = True
        elif method == "uniform":
            model = UniformModel(1, 64, 1, 4, budget_k)
            is_dpp = False
        else:
            model = CentralityModel(1, 64, 1, 4, budget_k)
            is_dpp = False

        best = train_model(
            model, train_cache, train_labels, val_cache, val_labels,
            test_cache, test_labels, epochs=epochs, lr=1e-3,
            margin_weight=0.1, warmup_epochs=min(10, max(1, epochs // 10)),
            is_dpp=is_dpp,
        )
        elapsed = time.time() - t0
        results[method] = {**best, "wall_time_sec": elapsed}
        print(f"  Done: MAE={best['mae']:.4f} @ epoch {best['epoch']}, time={elapsed:.0f}s")

    # Save
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"v2_ablation_k{budget_k}_seed{seed}.json"
    with out_path.open("w") as f:
        json.dump({"budget_k": budget_k, "epochs": epochs, "seed": seed, "results": results}, f, indent=2)

    print(f"\n=== V2 Results (k={budget_k}, seed={seed}) ===")
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
    run_v2_ablation(args.budget_k, args.epochs, args.seed)


if __name__ == "__main__":
    main()
