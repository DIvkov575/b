import torch
import torch.nn.functional as F
from src.models.ctmc_flow import CTMCDenoiser, compute_rate_from_denoiser
from src.models.prob_path_flow import ProbPathDenoiser, sample_euler_step
from src.models.classifier import TimeConditionalClassifier
from src.guidance.ctmc_guidance import guided_rates_ctmc, sample_tau_leaping
from src.guidance.prob_path_guidance import guided_posterior_prob_path


@torch.no_grad()
def generate_unconditional(
    model, n: int, K: int, L: int, num_steps: int, mode: str
) -> torch.Tensor:
    device = next(model.parameters()).device
    x_t = torch.randint(0, K, (n, L), device=device)
    dt = 1.0 / num_steps

    model.eval()
    for step in range(num_steps):
        t_val = 1.0 - (step + 1) * dt
        t = torch.full((n,), t_val, device=device)
        if mode == "ctmc":
            rates = compute_rate_from_denoiser(model, x_t, t, K)
            x_t = sample_tau_leaping(rates, x_t, dt, K)
        else:
            logits = model(x_t, t)
            posterior = F.softmax(logits, dim=-1)
            x_t = sample_euler_step(x_t, posterior, dt, K)
    return x_t


@torch.no_grad()
def generate_guided_single(
    model, classifier: TimeConditionalClassifier,
    n: int, K: int, L: int, num_steps: int, gamma: float, mode: str
) -> torch.Tensor:
    device = next(model.parameters()).device
    x_t = torch.randint(0, K, (n, L), device=device)
    dt = 1.0 / num_steps

    model.eval()
    classifier.eval()
    for step in range(num_steps):
        t_val = 1.0 - (step + 1) * dt
        t = torch.full((n,), t_val, device=device)
        if mode == "ctmc":
            rates = compute_rate_from_denoiser(model, x_t, t, K)
            rates = guided_rates_ctmc(rates, x_t, t, [classifier], [gamma])
            x_t = sample_tau_leaping(rates, x_t, dt, K)
        else:
            logits = model(x_t, t)
            logits = guided_posterior_prob_path(logits, x_t, t, [classifier], [gamma])
            posterior = F.softmax(logits, dim=-1)
            x_t = sample_euler_step(x_t, posterior, dt, K)
    return x_t


@torch.no_grad()
def generate_composed_and(
    model, classifiers: list,
    n: int, K: int, L: int, num_steps: int, gammas: list, mode: str
) -> torch.Tensor:
    device = next(model.parameters()).device
    x_t = torch.randint(0, K, (n, L), device=device)
    dt = 1.0 / num_steps

    model.eval()
    for clf in classifiers:
        clf.eval()
    for step in range(num_steps):
        t_val = 1.0 - (step + 1) * dt
        t = torch.full((n,), t_val, device=device)
        if mode == "ctmc":
            rates = compute_rate_from_denoiser(model, x_t, t, K)
            rates = guided_rates_ctmc(rates, x_t, t, classifiers, gammas)
            x_t = sample_tau_leaping(rates, x_t, dt, K)
        else:
            logits = model(x_t, t)
            logits = guided_posterior_prob_path(logits, x_t, t, classifiers, gammas)
            posterior = F.softmax(logits, dim=-1)
            x_t = sample_euler_step(x_t, posterior, dt, K)
    return x_t


@torch.no_grad()
def generate_composed_not(
    model, clf_keep: TimeConditionalClassifier, clf_avoid: TimeConditionalClassifier,
    n: int, K: int, L: int, num_steps: int, gamma_a: float, gamma_b: float, mode: str
) -> torch.Tensor:
    device = next(model.parameters()).device
    x_t = torch.randint(0, K, (n, L), device=device)
    dt = 1.0 / num_steps

    model.eval()
    clf_keep.eval()
    clf_avoid.eval()
    for step in range(num_steps):
        t_val = 1.0 - (step + 1) * dt
        t = torch.full((n,), t_val, device=device)
        if mode == "ctmc":
            rates = compute_rate_from_denoiser(model, x_t, t, K)
            rates = guided_rates_ctmc(
                rates, x_t, t, [clf_keep], [gamma_a],
                avoid_classifiers=[clf_avoid], avoid_gammas=[gamma_b]
            )
            x_t = sample_tau_leaping(rates, x_t, dt, K)
        else:
            logits = model(x_t, t)
            logits = guided_posterior_prob_path(
                logits, x_t, t, [clf_keep], [gamma_a],
                avoid_classifiers=[clf_avoid], avoid_gammas=[gamma_b]
            )
            posterior = F.softmax(logits, dim=-1)
            x_t = sample_euler_step(x_t, posterior, dt, K)
    return x_t


def evaluate_boolean_accuracy(
    samples: torch.Tensor, property_fns: list, mode: str = "and"
) -> float:
    """Evaluate % of samples satisfying the Boolean condition."""
    results = [fn(samples) for fn in property_fns]
    if mode == "and":
        combined = results[0]
        for r in results[1:]:
            combined = combined & r
    elif mode == "or":
        combined = results[0]
        for r in results[1:]:
            combined = combined | r
    elif mode == "not":
        combined = results[0] & (~results[1])
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return combined.float().mean().item()


def run_full_experiment(config_path: str = "configs/synthetic.yaml"):
    import yaml
    from src.training.train_flow import train_ctmc_flow, train_prob_path_flow
    from src.training.train_classifier import train_classifier
    from src.data.synthetic import PROPERTIES

    with open(config_path) as f:
        config = yaml.safe_load(f)

    K = config["data"]["K"]
    L = config["data"]["L"]
    num_steps = config["guidance"]["num_steps"]
    gamma = config["guidance"]["gamma"]
    n_samples = config["guidance"]["num_samples"]
    mode = config["training"]["flow_type"]

    print(f"=== Composable Discrete Flows — Synthetic Validation ===")
    print(f"Mode: {mode}, K={K}, L={L}, steps={num_steps}, gamma={gamma}")
    print()

    # Step 1: Train base flow
    print("[1/3] Training base flow model...")
    if mode == "ctmc":
        flow_model = train_ctmc_flow(config)
    else:
        flow_model = train_prob_path_flow(config)
    print()

    # Step 2: Train classifiers
    print("[2/3] Training classifiers...")
    classifiers = {}
    for prop_name in PROPERTIES:
        print(f"  Training classifier: {prop_name}")
        classifiers[prop_name] = train_classifier(prop_name, config)
    print()

    # Step 3: Evaluate composition
    print("[3/3] Evaluating composed guidance...")
    prop_fns = {
        "starts_with_0": PROPERTIES["starts_with_0"],
        "contains_pattern": PROPERTIES["contains_pattern"],
        "no_repeats": PROPERTIES["no_repeats"],
    }

    # Baseline: unconditional
    uncond_samples = generate_unconditional(flow_model, n_samples, K, L, num_steps, mode)
    print(f"\n--- Unconditional baseline ---")
    for name, fn in prop_fns.items():
        rate = fn(uncond_samples).float().mean().item()
        print(f"  {name}: {rate*100:.1f}%")

    # Single-condition guidance
    print(f"\n--- Single-condition guidance (gamma={gamma}) ---")
    for name in PROPERTIES:
        samples = generate_guided_single(
            flow_model, classifiers[name], n_samples, K, L, num_steps, gamma, mode
        )
        rate = prop_fns[name](samples).float().mean().item()
        print(f"  {name}: {rate*100:.1f}%")

    # AND composition: starts_with_0 AND contains_pattern
    print(f"\n--- AND(starts_with_0, contains_pattern) ---")
    samples_and = generate_composed_and(
        flow_model,
        [classifiers["starts_with_0"], classifiers["contains_pattern"]],
        n_samples, K, L, num_steps, [gamma, gamma], mode
    )
    rate_a = prop_fns["starts_with_0"](samples_and).float().mean().item()
    rate_b = prop_fns["contains_pattern"](samples_and).float().mean().item()
    acc_and = evaluate_boolean_accuracy(
        samples_and, [prop_fns["starts_with_0"], prop_fns["contains_pattern"]], mode="and"
    )
    print(f"  starts_with_0: {rate_a*100:.1f}%")
    print(f"  contains_pattern: {rate_b*100:.1f}%")
    print(f"  BOTH (AND accuracy): {acc_and*100:.1f}%")

    # NOT composition: starts_with_0 AND NOT no_repeats
    print(f"\n--- NOT(starts_with_0, no_repeats) = starts_with_0 AND NOT no_repeats ---")
    samples_not = generate_composed_not(
        flow_model,
        classifiers["starts_with_0"], classifiers["no_repeats"],
        n_samples, K, L, num_steps, gamma, gamma, mode
    )
    rate_a = prop_fns["starts_with_0"](samples_not).float().mean().item()
    rate_b = prop_fns["no_repeats"](samples_not).float().mean().item()
    acc_not = evaluate_boolean_accuracy(
        samples_not, [prop_fns["starts_with_0"], prop_fns["no_repeats"]], mode="not"
    )
    print(f"  starts_with_0: {rate_a*100:.1f}%")
    print(f"  no_repeats (should be LOW): {rate_b*100:.1f}%")
    print(f"  A AND NOT B accuracy: {acc_not*100:.1f}%")

    # Kill gate
    print(f"\n=== KILL GATE ===")
    print(f"  AND accuracy: {acc_and*100:.1f}% (target: >80%)")
    print(f"  NOT accuracy: {acc_not*100:.1f}% (target: >80%)")
    if acc_and >= 0.8 and acc_not >= 0.8:
        print("  PASS — proceed to Phase 2")
    elif acc_and >= 0.6 or acc_not >= 0.6:
        print("  PARTIAL — diagnose weak operator, tune gamma")
    else:
        print("  FAIL — operators broken, investigate math vs implementation")

    return {"and_accuracy": acc_and, "not_accuracy": acc_not, "mode": mode}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic.yaml")
    args = parser.parse_args()
    run_full_experiment(args.config)
