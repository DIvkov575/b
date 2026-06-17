"""Batched trainer: encodes subgraphs from multiple graphs in a single forward pass."""
import torch
from torch_geometric.data import Batch

from src.data.subgraph_policies import node_deletion_subgraphs, strip_subgraph
from src.models.base_gnn import GINEncoder
from src.models.dpp_selector import DPPSelector
from src.models.margin_scorer import MarginScorer
from src.models.bag_aggregator import BagAggregator
from src.training.losses import classification_loss, dpp_margin_loss
from src.training.margin_utils import compute_margin


class BatchedDPPSubgraphGNN(torch.nn.Module):
    """DPP subgraph GNN optimized for batched encoding."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=4, budget_k=10,
                 aggregation="weighted_sum"):
        super().__init__()
        self.budget_k = budget_k
        self.encoder = GINEncoder(in_dim, hidden_dim, hidden_dim, num_layers=num_layers)
        self.margin_scorer = MarginScorer(hidden_dim)
        self.dpp_selector = DPPSelector(hidden_dim, budget_k)
        self.aggregator = BagAggregator(hidden_dim, aggregation)
        self.classifier = torch.nn.Linear(hidden_dim, out_dim)

    def forward_batch(self, graphs: list, return_margin_info=False):
        """Process a batch of graphs with mega-batched encoding."""
        # Step 1: Generate all subgraphs (CPU, per-graph)
        all_subs = []
        graph_boundaries = []  # (start_idx, end_idx) per graph
        offset = 0
        for g in graphs:
            subs = node_deletion_subgraphs(g)
            clean = [strip_subgraph(s) for s in subs]
            all_subs.extend(clean)
            graph_boundaries.append((offset, offset + len(clean)))
            offset += len(clean)

        if not all_subs:
            return None

        # Step 2: Mega-batch encode ALL subgraphs at once
        mega_batch = Batch.from_data_list(all_subs)
        all_embeddings = self.encoder(mega_batch)  # (total_subs, hidden_dim)

        # Step 3: Per-graph DPP selection + aggregation
        all_logits = []
        all_info = []
        for i, (start, end) in enumerate(graph_boundaries):
            sub_embs = all_embeddings[start:end]  # (n_i, hidden_dim)
            qs = torch.zeros(0)

            if sub_embs.shape[0] == 0:
                graph_emb = torch.zeros(all_embeddings.shape[1], device=all_embeddings.device)
            else:
                graph_context = sub_embs.mean(dim=0)
                qs = self.margin_scorer(sub_embs, graph_context)

                if self.training:
                    marginals = self.dpp_selector.soft_select(sub_embs, qs)
                    weights = marginals / marginals.sum().clamp(min=1e-8)
                    graph_emb = self.aggregator(sub_embs, weights)
                else:
                    selected = self.dpp_selector(sub_embs.detach(), qs.detach())
                    sel_idx = torch.tensor(selected, dtype=torch.long)
                    sel_embs = sub_embs[sel_idx]
                    weights = torch.ones(len(selected)) / max(len(selected), 1)
                    graph_emb = self.aggregator(sel_embs, weights)

            logits = self.classifier(graph_emb)
            all_logits.append(logits)

            if return_margin_info:
                all_info.append({"quality_scores": qs})

        logits_batch = torch.stack(all_logits)  # (batch, out_dim)

        if return_margin_info:
            return logits_batch, all_info
        return logits_batch


class BatchedTrainer:
    """Trainer that processes multiple graphs per forward pass for efficiency."""

    def __init__(self, model: BatchedDPPSubgraphGNN, lr=1e-3, margin_weight=0.1,
                 warmup_epochs=10, task="classification", batch_size=16):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.margin_weight = margin_weight
        self.warmup_epochs = warmup_epochs
        self.task = task
        self.batch_size = batch_size
        self.epoch = 0

    def _collect_batch(self, loader_iter, n):
        """Collect n graphs from the loader iterator."""
        graphs = []
        labels = []
        for _ in range(n):
            try:
                g = next(loader_iter)
                graphs.append(g)
                y = g.y if hasattr(g, "y") else g["y"]
                if y.dim() == 0:
                    y = y.unsqueeze(0)
                labels.append(y)
            except StopIteration:
                break
        return graphs, labels

    def train_epoch(self, loader):
        self.model.train()
        cls_total = 0.0
        margin_total = 0.0
        n_batches = 0

        loader_iter = iter(loader)
        while True:
            graphs, labels = self._collect_batch(loader_iter, self.batch_size)
            if not graphs:
                break

            self.optimizer.zero_grad()
            labels_tensor = torch.cat(labels)

            result = self.model.forward_batch(graphs, return_margin_info=True)
            if result is None:
                continue
            logits_batch, infos = result

            cls_loss = classification_loss(logits_batch, labels_tensor, task=self.task)
            loss = cls_loss

            margin_value = 0.0
            if self.epoch >= self.warmup_epochs:
                margins = compute_margin(logits_batch.detach(), labels_tensor)
                # Per-graph margin loss on quality scores
                m_losses = []
                for i, info in enumerate(infos):
                    qs = info["quality_scores"]
                    if qs.numel() > 0:
                        m_broadcast = margins[i].expand(qs.shape[0])
                        m_losses.append(dpp_margin_loss(qs, m_broadcast))
                if m_losses:
                    m_loss = torch.stack(m_losses).mean()
                    loss = loss + self.margin_weight * m_loss
                    margin_value = m_loss.item()

            loss.backward()
            self.optimizer.step()

            cls_total += cls_loss.item()
            margin_total += margin_value
            n_batches += 1

        self.epoch += 1
        denom = max(n_batches, 1)
        return {"cls_loss": cls_total / denom, "margin_loss": margin_total / denom}

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_mae = 0.0
        correct = 0
        total = 0
        margin_sum = 0.0

        loader_iter = iter(loader)
        while True:
            graphs, labels = self._collect_batch(loader_iter, self.batch_size)
            if not graphs:
                break

            labels_tensor = torch.cat(labels)
            logits_batch = self.model.forward_batch(graphs)
            if logits_batch is None:
                continue

            if self.task == "regression":
                total_mae += torch.abs(logits_batch.view(-1) - labels_tensor.view(-1).float()).sum().item()
            else:
                preds = logits_batch.argmax(dim=-1)
                correct += (preds == labels_tensor).sum().item()
            total += labels_tensor.numel()

            margins = compute_margin(logits_batch, labels_tensor)
            margin_sum += margins.sum().item()

        metrics = {"avg_margin": margin_sum / max(total, 1)}
        if self.task == "regression":
            metrics["mae"] = total_mae / max(total, 1)
        else:
            metrics["accuracy"] = correct / max(total, 1)
        return metrics
