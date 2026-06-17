"""Run all 4 models in parallel processes for faster ablation."""
import argparse
import subprocess
import sys
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_single_model(model_name, budget_k, epochs, seed):
    """Run one model in a subprocess and return results."""
    cmd = [
        sys.executable, "-m", "src.experiments.ablations",
        "--budget-k", str(budget_k),
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--model", model_name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {"model": model_name, "error": result.stderr[-500:]}
    out_path = Path(f"results/ablation_{model_name}_k{budget_k}_seed{seed}.json")
    if out_path.exists():
        with open(out_path) as f:
            return json.load(f)
    return {"model": model_name, "error": "no output file"}


def main():
    parser = argparse.ArgumentParser(description="Parallel ablation runner")
    parser.add_argument("--budget-k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=["dpp", "uniform", "centrality", "full"])
    args = parser.parse_args()

    print(f"Running {len(args.models)} models in parallel: {args.models}")
    print(f"Budget k={args.budget_k}, epochs={args.epochs}, seed={args.seed}")
    print()

    with ProcessPoolExecutor(max_workers=len(args.models)) as executor:
        futures = {
            executor.submit(run_single_model, m, args.budget_k, args.epochs, args.seed): m
            for m in args.models
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                if "error" in result:
                    print(f"  {name}: FAILED — {result['error'][:100]}")
                else:
                    mae = result.get("best_test", {}).get("mae", "?")
                    t = result.get("wall_time_sec", "?")
                    print(f"  {name}: MAE={mae:.4f}, time={t:.0f}s" if isinstance(mae, float) else f"  {name}: {result}")
            except Exception as e:
                print(f"  {name}: EXCEPTION — {e}")


if __name__ == "__main__":
    main()
