import yaml
import torch
from src.data.synthetic import MarkovSequenceDataset, PROPERTIES
from src.training.train_conditional import train_conditional_flow
from src.guidance.cfg_compose import cfg_sample, cfg_sample_not


def evaluate_boolean_accuracy(samples: torch.Tensor, property_fns: list, mode: str = "and") -> float:
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


def run_cfg_experiment(config_path: str = "configs/cfg_synthetic.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    K = config["data"]["K"]
    L = config["data"]["L"]
    w = config["guidance"]["w"]
    num_steps = config["guidance"]["num_steps"]
    n_samples = config["guidance"]["num_samples"]
    n_conditions = len(config["properties"])

    print("=== Composable Discrete Flows — CFG Composition ===")
    print(f"K={K}, L={L}, w={w}, steps={num_steps}")
    print()

    # Step 1: Train conditional flow with CFG dropout
    print("[1/2] Training conditional flow with CFG dropout...")
    model = train_conditional_flow(config)
    print()

    # Step 2: Evaluate
    print("[2/2] Evaluating CFG composition...")
    prop_fns = {
        "starts_with_0": PROPERTIES["starts_with_0"],
        "contains_pattern": PROPERTIES["contains_pattern"],
        "no_repeats": PROPERTIES["no_repeats"],
    }
    prop_list = list(prop_fns.keys())

    # Baseline: unconditional (w=0)
    print("\n--- Unconditional (w=0) ---")
    uncond = cfg_sample(model, n_samples, K, L, n_conditions,
                        condition_idxs=[0], ws=[0.0], num_steps=num_steps, mode="single")
    for name, fn in prop_fns.items():
        rate = fn(uncond).float().mean().item()
        print(f"  {name}: {rate*100:.1f}%")

    # Single-condition CFG
    print(f"\n--- Single-condition CFG (w={w}) ---")
    for i, name in enumerate(prop_list):
        samples = cfg_sample(model, n_samples, K, L, n_conditions,
                             condition_idxs=[i], ws=[w], num_steps=num_steps, mode="single")
        rate = prop_fns[name](samples).float().mean().item()
        print(f"  {name}: {rate*100:.1f}%")

    # AND composition
    print(f"\n--- AND(starts_with_0, contains_pattern) w={w} ---")
    samples_and = cfg_sample(model, n_samples, K, L, n_conditions,
                             condition_idxs=[0, 1], ws=[w, w],
                             num_steps=num_steps, mode="and")
    rate_a = prop_fns["starts_with_0"](samples_and).float().mean().item()
    rate_b = prop_fns["contains_pattern"](samples_and).float().mean().item()
    acc_and = evaluate_boolean_accuracy(
        samples_and, [prop_fns["starts_with_0"], prop_fns["contains_pattern"]], mode="and"
    )
    print(f"  starts_with_0: {rate_a*100:.1f}%")
    print(f"  contains_pattern: {rate_b*100:.1f}%")
    print(f"  BOTH (AND accuracy): {acc_and*100:.1f}%")

    # NOT composition
    print(f"\n--- NOT: starts_with_0 AND NOT no_repeats ---")
    samples_not = cfg_sample_not(model, n_samples, K, L, n_conditions,
                                 keep_idx=0, avoid_idx=2,
                                 w_keep=w, w_avoid=w, num_steps=num_steps)
    rate_a = prop_fns["starts_with_0"](samples_not).float().mean().item()
    rate_b = prop_fns["no_repeats"](samples_not).float().mean().item()
    acc_not = evaluate_boolean_accuracy(
        samples_not, [prop_fns["starts_with_0"], prop_fns["no_repeats"]], mode="not"
    )
    print(f"  starts_with_0: {rate_a*100:.1f}%")
    print(f"  no_repeats (should be LOW): {rate_b*100:.1f}%")
    print(f"  A AND NOT B accuracy: {acc_not*100:.1f}%")

    # OR composition
    print(f"\n--- OR(starts_with_0, contains_pattern) w={w} ---")
    samples_or = cfg_sample(model, n_samples, K, L, n_conditions,
                            condition_idxs=[0, 1], ws=[w, w],
                            num_steps=num_steps, mode="or")
    acc_or = evaluate_boolean_accuracy(
        samples_or, [prop_fns["starts_with_0"], prop_fns["contains_pattern"]], mode="or"
    )
    print(f"  A OR B accuracy: {acc_or*100:.1f}%")

    # Kill gate
    print(f"\n=== KILL GATE ===")
    print(f"  AND accuracy: {acc_and*100:.1f}% (target: >80%)")
    print(f"  NOT accuracy: {acc_not*100:.1f}% (target: >80%)")
    if acc_and >= 0.8 and acc_not >= 0.8:
        print("  PASS — proceed to Phase 2")
    elif acc_and >= 0.5 or acc_not >= 0.5:
        print("  PARTIAL — increase w, more epochs, or tune dropout")
    else:
        print("  FAIL — fundamental issue")

    return {"and": acc_and, "not": acc_not, "or": acc_or}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cfg_synthetic.yaml")
    args = parser.parse_args()
    run_cfg_experiment(args.config)
