import yaml
import torch
import torch.nn.functional as F
from src.training.train_mdlm import train_mdlm
from src.training.schedules import uniform_schedule, bell_schedule, optimal_schedule
from src.training.estimate_it import estimate_mutual_info_curve
from src.data.text_data import Text8Dataset, mask_sequence, MASK_TOKEN


@torch.no_grad()
def evaluate_nll(model, dataset, device="cpu", n_eval=1000):
    """Evaluate average cross-entropy on validation data across timesteps."""
    model.eval()
    model = model.to(device)
    seqs = dataset.seqs[:n_eval].to(device)
    B, L = seqs.shape

    total_loss = 0.0
    n_t = 10
    for t_val in torch.linspace(0.1, 0.9, n_t):
        t = torch.full((B,), t_val.item(), device=device)
        x_t = mask_sequence(seqs, t, mask_token=MASK_TOKEN)
        logits = model(x_t, t)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), seqs.reshape(-1))
        total_loss += loss.item()

    return total_loss / n_t


def run_comparison(config_path: str = "configs/text8_small.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = "cpu"
    print("=== Theory-Guided Training Schedules for Masked Diffusion ===\n")

    # Phase 1: Train baseline (uniform)
    print("[1/4] Training baseline (uniform schedule)...")
    model_uniform = train_mdlm(config, schedule_weights=uniform_schedule(), device=device)
    print()

    # Phase 2: Estimate I(t) from baseline model
    print("[2/4] Estimating I(t) curve from baseline model...")
    val_ds = Text8Dataset(split="val", seq_len=config["data"]["seq_len"], max_samples=500)
    it_curve = estimate_mutual_info_curve(model_uniform, val_ds.seqs, n_timesteps=100, device=device)
    print(f"  I(t) peak at t={it_curve.argmax().item()/100:.2f}, "
          f"max={it_curve.max():.4f}, mean={it_curve.mean():.4f}")
    print()

    # Phase 3: Train with bell schedule (Hong et al. baseline)
    print("[3/4] Training with bell schedule (Hong et al.)...")
    bell_weights = bell_schedule(n_bins=1000, peak=0.5, width=0.2)
    model_bell = train_mdlm(config, schedule_weights=bell_weights, device=device)
    print()

    # Phase 4: Train with I(t)-optimal schedule (our method)
    print("[4/4] Training with I(t)-optimal schedule (ours)...")
    it_interp = torch.nn.functional.interpolate(
        it_curve.unsqueeze(0).unsqueeze(0), size=1000, mode='linear'
    ).squeeze()
    optimal_weights = optimal_schedule(it_interp)
    model_optimal = train_mdlm(config, schedule_weights=optimal_weights, device=device)
    print()

    # Evaluate all three
    print("=== Validation Loss Comparison ===")
    for name, model in [("Uniform", model_uniform), ("Bell", model_bell), ("Optimal (ours)", model_optimal)]:
        val_loss = evaluate_nll(model, val_ds, device=device)
        print(f"  {name:20s}: val_loss = {val_loss:.4f}")

    # I(t) shape analysis
    print(f"\n=== I(t) Curve Analysis ===")
    peak_t = it_curve.argmax().item() / 100
    print(f"  Peak at t={peak_t:.2f}")
    if 0.3 < peak_t < 0.7:
        print(f"  -> I(t) is bell-shaped — explains Hong et al.'s heuristic")
    else:
        print(f"  -> I(t) is NOT bell-shaped — optimal schedule differs from bell")

    # Kill gate
    val_uniform = evaluate_nll(model_uniform, val_ds, device=device)
    val_bell = evaluate_nll(model_bell, val_ds, device=device)
    val_optimal = evaluate_nll(model_optimal, val_ds, device=device)

    improvement_vs_uniform = (val_uniform - val_optimal) / val_uniform * 100
    improvement_vs_bell = (val_bell - val_optimal) / val_bell * 100

    print(f"\n=== KILL GATE ===")
    print(f"  Optimal vs Uniform: {improvement_vs_uniform:+.1f}% (target: >20%)")
    print(f"  Optimal vs Bell:    {improvement_vs_bell:+.1f}% (target: >0%)")
    if improvement_vs_uniform > 20 and improvement_vs_bell > 0:
        print("  PASS — theory-guided schedule outperforms both baselines")
    elif improvement_vs_uniform > 10:
        print("  PARTIAL — beats uniform but not bell; need more epochs or tuning")
    else:
        print("  FAIL — I(t)-guided schedule not effective")

    return {"it_curve": it_curve, "optimal_weights": optimal_weights,
            "val_uniform": val_uniform, "val_bell": val_bell, "val_optimal": val_optimal}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/text8_small.yaml")
    args = parser.parse_args()
    run_comparison(args.config)
