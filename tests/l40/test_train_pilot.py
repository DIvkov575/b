import json

import numpy as np

from src.l40.train_pilot import run_training


def _write_npz(path, seq_lengths):
    starts, ends, cursor = [], [], 0
    for length in seq_lengths:
        starts.append(cursor)
        cursor += length
        ends.append(cursor)
    sequences = np.array(list(zip(starts, ends)), dtype=[('res_start', 'i4'), ('res_end', 'i4')])
    res_types = np.arange(cursor, dtype='i4') % 20 + 1
    residues = np.array(list(zip(res_types)), dtype=[('res_type', 'i4')])
    np.savez(path, sequences=sequences, residues=residues)


def test_run_training_boltz_variant_produces_finite_metrics(tmp_path):
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    for i in range(12):
        _write_npz(npz_dir / f"s{i}_a.npz", seq_lengths=[40, 40, 40])  # 3 "MSA hits" each

    out_path = tmp_path / "boltz_results.json"
    result = run_training(
        variant="boltz", data_path=str(npz_dir), out_path=str(out_path),
        max_files=12, max_length=48, batch_size=4, epochs=1,
        d_model=16, n_layers=1, n_heads=2, d_ff=32,
    )

    assert np.isfinite(result["epochs"][0]["train_loss"])
    assert np.isfinite(result["epochs"][0]["val_loss"])
    assert 0.0 <= result["epochs"][0]["val_accuracy"] <= 1.0
    assert json.loads(out_path.read_text()) == result


def test_run_training_baseline_variant_produces_finite_metrics(tmp_path):
    jsonl_path = tmp_path / "baseline.jsonl"
    sequence = "ARNDCQEGHILKMFPSTWYV" * 3
    with open(jsonl_path, "w") as f:
        for i in range(12):
            f.write(json.dumps({"structure_id": f"s{i}_a", "sequence": sequence}) + "\n")

    out_path = tmp_path / "baseline_results.json"
    result = run_training(
        variant="baseline", data_path=str(jsonl_path), out_path=str(out_path),
        max_files=12, max_length=48, batch_size=4, epochs=1,
        d_model=16, n_layers=1, n_heads=2, d_ff=32,
    )

    assert np.isfinite(result["epochs"][0]["train_loss"])
    assert np.isfinite(result["epochs"][0]["val_loss"])
