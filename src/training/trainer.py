import torch

from src.training.losses import classification_loss, dpp_margin_loss
from src.training.margin_utils import compute_margin


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS disabled: PyG 2.6.1 GINConv.propagate() has device mismatch bugs on MPS
    return torch.device("cpu")


class Trainer:
    def __init__(self, model, lr=1e-3, margin_weight=0.1, warmup_epochs=10, task="classification", device=None):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.margin_weight = margin_weight
        self.warmup_epochs = warmup_epochs
        self.task = task
        self.epoch = 0
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_epoch(self, loader):
        self.model.train()
        cls_total = 0.0
        margin_total = 0.0
        n = 0

        for graph in loader:
            graph = graph.to(self.device)
            self.optimizer.zero_grad()
            logits, info = self.model(graph, return_margin_info=True)
            label = graph.y if hasattr(graph, "y") else graph["y"]
            if label.dim() == 0:
                label = label.unsqueeze(0)

            logits_2d = logits.unsqueeze(0) if logits.dim() == 1 else logits
            cls_loss = classification_loss(logits_2d, label, task=self.task)
            loss = cls_loss

            margin_value = 0.0
            if self.epoch >= self.warmup_epochs and info["quality_scores"].numel() > 0:
                actual_margin = compute_margin(logits_2d.detach(), label)
                quality_scores = info["quality_scores"]
                margin_broadcast = actual_margin.expand(quality_scores.shape[0])
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
        total_mae = 0.0
        margin_sum = 0.0
        margin_count = 0

        with torch.no_grad():
            for graph in loader:
                graph = graph.to(self.device)
                logits = self.model(graph)
                if isinstance(logits, tuple):
                    logits = logits[0]
                label = graph.y if hasattr(graph, "y") else graph["y"]
                if label.dim() == 0:
                    label = label.unsqueeze(0)

                logits_2d = logits.unsqueeze(0) if logits.dim() == 1 else logits

                if self.task == "regression":
                    total_mae += torch.abs(logits_2d.view(-1) - label.view(-1).float()).sum().item()
                else:
                    preds = logits_2d.argmax(dim=-1)
                    correct += (preds == label).sum().item()
                total += label.numel()

                margin = compute_margin(logits_2d, label)
                margin_sum += margin.sum().item()
                margin_count += margin.numel()

        metrics = {"avg_margin": margin_sum / max(margin_count, 1)}
        if self.task == "regression":
            metrics["mae"] = total_mae / max(total, 1)
        else:
            metrics["accuracy"] = correct / max(total, 1)
        return metrics
