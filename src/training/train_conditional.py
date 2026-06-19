import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data.synthetic import MarkovSequenceDataset
from src.models.conditional_flow import ConditionalProbPathFlow
from src.models.prob_path_flow import sample_from_categorical


def train_conditional_flow(config: dict, device: str = "cpu") -> ConditionalProbPathFlow:
    K = config["data"]["K"]
    L = config["data"]["L"]
    n_conditions = len(config["properties"])
    cfg_dropout = config["training"]["cfg_dropout_prob"]

    model = ConditionalProbPathFlow(
        K=K, L=L, n_conditions=n_conditions,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
    ).to(device)

    ds = MarkovSequenceDataset(n=config["data"]["n_train"], K=K, L=L)
    loader = DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    property_names = list(ds.labels.keys())

    model.train()
    for epoch in range(config["training"]["epochs"]):
        total_loss = 0.0
        for seqs, props_batch in loader:
            seqs = seqs.to(device)
            B = seqs.shape[0]

            cond = torch.zeros(B, n_conditions, device=device)
            for i, name in enumerate(property_names):
                for j in range(B):
                    cond[j, i] = float(props_batch[name][j])

            t = torch.rand(B, device=device)
            x_t = sample_from_categorical(seqs, t, K)
            logits = model(x_t, t, cond, cfg_dropout_prob=cfg_dropout)
            loss = F.cross_entropy(logits.reshape(-1, K), seqs.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"  CFG flow epoch {epoch+1}: loss={total_loss / len(loader):.4f}")

    return model
