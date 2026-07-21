import json

from src.l40.compare_results import compare, load_results


def test_load_results_reads_json(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"variant": "boltz", "epochs": [{"val_accuracy": 0.5}]}))
    result = load_results(str(path))
    assert result["variant"] == "boltz"


def test_compare_reports_final_epoch_delta():
    boltz = {"variant": "boltz", "epochs": [
        {"epoch": 1, "train_loss": 3.0, "val_loss": 2.9, "val_accuracy": 0.10},
        {"epoch": 2, "train_loss": 2.5, "val_loss": 2.4, "val_accuracy": 0.15},
    ]}
    baseline = {"variant": "baseline", "epochs": [
        {"epoch": 1, "train_loss": 3.1, "val_loss": 3.0, "val_accuracy": 0.08},
        {"epoch": 2, "train_loss": 2.7, "val_loss": 2.6, "val_accuracy": 0.11},
    ]}

    result = compare(boltz, baseline)

    assert result["boltz_final_val_accuracy"] == 0.15
    assert result["baseline_final_val_accuracy"] == 0.11
    assert abs(result["val_accuracy_delta"] - 0.04) < 1e-9
    assert abs(result["val_loss_delta"] - (2.4 - 2.6)) < 1e-9
