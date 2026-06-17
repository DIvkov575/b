import torch

from src.training.losses import classification_loss, dpp_margin_loss
from src.training.margin_utils import compute_margin


class Trainer:
    def __init__(self, model, lr=1e-3, margin_weight=0.1, warmup_epochs=10):
        self.model = model
        self.margin_weight = margin_weight
        self.warmup_epochs = warmup_epochs
        self.epoch = 0
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_epoch(self, loader):
        self.model.train()
        cls_total = 0.0
        margin_total = 0.0
        n = 0

        for graph in loader:
            self.optimizer.zero_grad()
            logits, info = self.model(graph, return_margin_info=True)
            label = graph.y if hasattr(graph, "y") else graph["y"]
            if label.dim() == 0:
                label = label.unsqueeze(0)

            cls_loss = classification_loss(logits, label)
            loss = cls_loss

            margin_value = 0.0
            if self.epoch >= self.warmup_epochs:
                actual_margin = compute_margin(logits.detach(), label)
                quality_scores = info["quality_scores"]
                margin_broadcast = actual_margin.expand_as(quality_scores)
                m_loss = dpp_margin_loss(quality_scores, margin_broadcast)
                loss = loss + self.margin_weight * m_loss
                margin_value = m_loss.item()

            loss.backward()
            self.optimizer.step()

            cls_total += cls_loss.item()
            margin_total += margin_value
            n += 1

        self.epoch += 1
        denom = max(n, 1)
        return {"cls_loss": cls_total / denom, "margin_loss": margin_total / denom}

    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        margin_sum = 0.0
        margin_count = 0

        with torch.no_grad():
            for graph in loader:
                logits = self.model(graph)
                if isinstance(logits, tuple):
                    logits = logits[0]
                label = graph.y if hasattr(graph, "y") else graph["y"]
                if label.dim() == 0:
                    label = label.unsqueeze(0)

                preds = logits.argmax(dim=-1)
                correct += (preds == label).sum().item()
                total += label.numel()

                margin = compute_margin(logits, label)
                margin_sum += margin.sum().item()
                margin_count += margin.numel()

        accuracy = correct / total if total > 0 else 0.0
        avg_margin = margin_sum / margin_count if margin_count > 0 else 0.0
        return {"accuracy": accuracy, "avg_margin": avg_margin}
