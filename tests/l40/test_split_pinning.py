import json

from src.l40.baseline_data import load_baseline_protein_data
from src.l40.msa_data import create_diverse_splits, load_protein_data


def test_boltz_and_baseline_produce_identical_file_level_split(tmp_path):
    # Simulate a Boltz npz directory listing.
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    filenames = [f"struct{i}_a.npz" for i in range(20)]
    for name in filenames:
        (npz_dir / name).write_bytes(b"")

    # Baseline JSONL written in the SAME sorted order fetch_rcsb_structures.py would use.
    jsonl_path = tmp_path / "baseline.jsonl"
    with open(jsonl_path, "w") as f:
        for name in sorted(filenames):
            structure_id = name[:-4]
            f.write(json.dumps({"structure_id": structure_id, "sequence": "A" * 40}) + "\n")

    boltz_file_sequences = load_protein_data(str(npz_dir), max_files=20, sequences_per_file=3)
    baseline_file_sequences = load_baseline_protein_data(str(jsonl_path), max_files=20)

    boltz_train, boltz_val, boltz_test = create_diverse_splits(boltz_file_sequences)
    baseline_train, baseline_val, baseline_test = create_diverse_splits(
        baseline_file_sequences, max_seqs_per_file_train=1, max_seqs_per_file_val=1,
    )

    boltz_train_files = {s['filename'] for s in boltz_train}
    baseline_train_files = {s['structure_id'] for s in baseline_train}
    boltz_val_files = {s['filename'] for s in boltz_val}
    baseline_val_files = {s['structure_id'] for s in baseline_val}
    boltz_test_files = {s['filename'] for s in boltz_test}
    baseline_test_files = {s['structure_id'] for s in baseline_test}

    # boltz_*_files are filenames with the .npz suffix (from load_protein_data);
    # baseline_*_files are bare structure_ids (from load_baseline_protein_data) —
    # add the suffix back on for a like-for-like comparison.
    assert boltz_train_files == {f"{sid}.npz" for sid in baseline_train_files}
    assert boltz_val_files == {f"{sid}.npz" for sid in baseline_val_files}
    assert boltz_test_files == {f"{sid}.npz" for sid in baseline_test_files}
