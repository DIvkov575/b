import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data.synthetic import SyntheticSequenceDataset
from src.models.ctmc_flow import CTMCDenoiser, noise_sequence
from src.models.prob_path_flow import ProbPathDenoiser, sample_from_categorical


def train_ctmc_flow(config: dict, device: str = "cpu") -> CTMCDenoiser:
    K = config["data"]["K"]
    L = config["data"]["L"]
    model = CTMCDenoiser(
        K=K, L=L,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
    ).to(device)
    ds = SyntheticSequenceDataset(n=config["data"]["n_train"], K=K, L=L)
    loader = DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    model.train()
    for epoch in range(config["training"]["epochs"]):
        total_loss = 0.0
        for seqs, _ in loader:
            seqs = seqs.to(device)
            B = seqs.shape[0]
            t = torch.rand(B, device=device)
            x_t = noise_sequence(seqs, t, K)
            logits = model(x_t, t)
            loss = F.cross_entropy(logits.reshape(-1, K), seqs.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"  CTMC epoch {epoch+1}: loss={total_loss / len(loader):.4f}")
    return model


def train_prob_path_flow(config: dict, device: str = "cpu") -> ProbPathDenoiser:
    K = config["data"]["K"]
    L = config["data"]["L"]
    model = ProbPathDenoiser(
        K=K, L=L,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
    ).to(device)
    ds = SyntheticSequenceDataset(n=config["data"]["n_train"], K=K, L=L)
    loader = DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    model.train()
    for epoch in range(config["training"]["epochs"]):
        total_loss = 0.0
        for seqs, _ in loader:
            seqs = seqs.to(device)
            B = seqs.shape[0]
            t = torch.rand(B, device=device)
            x_t = sample_from_categorical(seqs, t, K)
            logits = model(x_t, t)
            loss = F.cross_entropy(logits.reshape(-1, K), seqs.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"  ProbPath epoch {epoch+1}: loss={total_loss / len(loader):.4f}")
    return model
