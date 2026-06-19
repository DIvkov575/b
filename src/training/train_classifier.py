import torch
import torch.nn.functional as F
from src.data.synthetic import SyntheticSequenceDataset
from src.models.classifier import TimeConditionalClassifier
from src.models.ctmc_flow import noise_sequence


def train_classifier(
    property_name: str, config: dict, device: str = "cpu"
) -> TimeConditionalClassifier:
    K = config["data"]["K"]
    L = config["data"]["L"]
    ds = SyntheticSequenceDataset(n=config["data"]["n_train"], K=K, L=L)
    labels = ds.labels[property_name].float().to(device)
    seqs = ds.seqs.to(device)

    model = TimeConditionalClassifier(
        K=K, L=L,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    batch_size = config["training"]["batch_size"]

    model.train()
    for epoch in range(config["training"]["epochs"]):
        perm = torch.randperm(len(seqs), device=device)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(seqs), batch_size):
            idx = perm[i:i + batch_size]
            batch_seqs = seqs[idx]
            batch_labels = labels[idx]
            t = torch.rand(len(idx), device=device)
            x_t = noise_sequence(batch_seqs, t, K)
            logits = model(x_t, t).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 20 == 0:
            print(f"  Classifier[{property_name}] epoch {epoch+1}: "
                  f"loss={total_loss / n_batches:.4f}")
    return model
