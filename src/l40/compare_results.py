"""Compares the two L40 pilot training runs and reports the final-epoch delta.

Usage:
    .venv-l38/bin/python -m src.l40.compare_results \
        --boltz src/l40/pilot_out/boltz_results.json \
        --baseline src/l40/pilot_out/baseline_results.json
"""
import argparse
import json


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compare(boltz: dict, baseline: dict) -> dict:
    boltz_final = boltz["epochs"][-1]
    baseline_final = baseline["epochs"][-1]
    return {
        "boltz_final_train_loss": boltz_final["train_loss"],
        "baseline_final_train_loss": baseline_final["train_loss"],
        "boltz_final_val_loss": boltz_final["val_loss"],
        "baseline_final_val_loss": baseline_final["val_loss"],
        "val_loss_delta": boltz_final["val_loss"] - baseline_final["val_loss"],
        "boltz_final_val_accuracy": boltz_final["val_accuracy"],
        "baseline_final_val_accuracy": baseline_final["val_accuracy"],
        "val_accuracy_delta": boltz_final["val_accuracy"] - baseline_final["val_accuracy"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boltz", required=True)
    parser.add_argument("--baseline", required=True)
    args = parser.parse_args()

    boltz = load_results(args.boltz)
    baseline = load_results(args.baseline)
    result = compare(boltz, baseline)

    print(json.dumps(result, indent=2))
    print()
    print(f"val_accuracy: boltz={result['boltz_final_val_accuracy']:.4f}  "
          f"baseline={result['baseline_final_val_accuracy']:.4f}  "
          f"delta={result['val_accuracy_delta']:+.4f}")
    print(f"val_loss:     boltz={result['boltz_final_val_loss']:.4f}  "
          f"baseline={result['baseline_final_val_loss']:.4f}  "
          f"delta={result['val_loss_delta']:+.4f}  (negative = boltz better)")


if __name__ == "__main__":
    main()
