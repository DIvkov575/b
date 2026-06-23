import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data.text_data import Text8Dataset, mask_sequence, MASK_TOKEN
from src.models.mdlm import MDLM
from src.training.schedules import uniform_schedule, bell_schedule, sample_timesteps


def train_mdlm(config: dict, schedule_weights: torch.Tensor = None,
               device: str = "cpu") -> MDLM:
    """Train MDLM with a configurable time-sampling schedule."""
    ds = Text8Dataset(
        split="train", seq_len=config["data"]["seq_len"],
        max_samples=config["data"].get("max_samples")
    )
    loader = DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=True)

    model = MDLM(
        vocab_size=config["data"]["vocab_size"],
        seq_len=config["data"]["seq_len"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        dropout=config["model"].get("dropout", 0.0),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"])

    if schedule_weights is None:
        sched_name = config["training"]["schedule"]
        if sched_name == "uniform":
            schedule_weights = uniform_schedule()
        elif sched_name == "bell":
            schedule_weights = bell_schedule()
        else:
            raise ValueError(f"Unknown schedule: {sched_name}. "
                           "For 'optimal', pass schedule_weights explicitly.")

    model.train()
    for epoch in range(config["training"]["epochs"]):
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            batch = batch.to(device)
            B = batch.shape[0]

            t = sample_timesteps(B, schedule_weights, device=device)
            x_t = mask_sequence(batch, t, mask_token=MASK_TOKEN)

            logits = model(x_t, t)
            loss = F.cross_entropy(logits.reshape(-1, model.vocab_size),
                                   batch.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")

    return model
