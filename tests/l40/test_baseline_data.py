import json

import torch

from src.l40.baseline_data import (
    RCSBBaselineDataset,
    create_baseline_dataloaders,
    load_baseline_protein_data,
)
from src.l40.vocab import PAD_TOKEN


def _write_jsonl(path, structure_ids_and_seqs):
    with open(path, "w") as f:
        for sid, seq in structure_ids_and_seqs:
            f.write(json.dumps({"structure_id": sid, "sequence": seq}) + "\n")


class TestLoadBaselineProteinData:
    def test_builds_one_record_per_structure_in_file_order(self, tmp_path):
        jsonl_path = tmp_path / "baseline.jsonl"
        _write_jsonl(jsonl_path, [("b_a", "AAAA"), ("a_a", "CCCC")])

        file_sequences = load_baseline_protein_data(str(jsonl_path), max_files=10)

        assert list(file_sequences.keys()) == ["b_a", "a_a"]  # preserves JSONL order, NOT sorted
        assert len(file_sequences["b_a"]) == 1

    def test_respects_max_files(self, tmp_path):
        jsonl_path = tmp_path / "baseline.jsonl"
        _write_jsonl(jsonl_path, [("a", "AAAA"), ("b", "CCCC"), ("c", "GGGG")])

        file_sequences = load_baseline_protein_data(str(jsonl_path), max_files=2)

        assert len(file_sequences) == 2


class TestRCSBBaselineDataset:
    def test_getitem_produces_msadataset_compatible_dict(self, tmp_path):
        jsonl_path = tmp_path / "baseline.jsonl"
        sequence = "ARNDCQEGHILKMFPSTWYV" * 3  # 60 residues, all standard AAs
        _write_jsonl(jsonl_path, [("s1", sequence)])
        record = {"jsonl_path": str(jsonl_path), "structure_id": "s1"}
        dataset = RCSBBaselineDataset(records=[record], max_length=64, fixed_seed=0)

        item = dataset[0]

        assert set(item.keys()) == {"input_ids", "attention_mask", "labels", "source_file", "seq_length"}
        assert item["input_ids"].shape == (64,)
        assert item["seq_length"] == 60
        assert (item["input_ids"][60:] == PAD_TOKEN).all()

    def test_short_sequence_returns_zero_loss_dummy(self, tmp_path):
        jsonl_path = tmp_path / "baseline.jsonl"
        _write_jsonl(jsonl_path, [("s1", "ARN")])  # length 3 < min 20
        record = {"jsonl_path": str(jsonl_path), "structure_id": "s1"}
        dataset = RCSBBaselineDataset(records=[record], max_length=32)

        item = dataset[0]

        assert item["seq_length"] == 0
        assert (item["labels"] == -100).all()

    def test_fixed_seed_gives_deterministic_masking(self, tmp_path):
        jsonl_path = tmp_path / "baseline.jsonl"
        sequence = "ARNDCQEGHILKMFPSTWYV" * 3
        _write_jsonl(jsonl_path, [("s1", sequence)])
        record = {"jsonl_path": str(jsonl_path), "structure_id": "s1"}

        ds1 = RCSBBaselineDataset(records=[record], max_length=64, fixed_seed=42)
        ds2 = RCSBBaselineDataset(records=[record], max_length=64, fixed_seed=42)

        assert torch.equal(ds1[0]["input_ids"], ds2[0]["input_ids"])


class TestCreateBaselineDataloaders:
    def test_returns_loaders_with_expected_batch_shape(self, tmp_path):
        jsonl_path = tmp_path / "baseline.jsonl"
        sequence = "ARNDCQEGHILKMFPSTWYV" * 3
        _write_jsonl(jsonl_path, [(f"s{i}", sequence) for i in range(10)])

        train_loader, val_loader, test_loader = create_baseline_dataloaders(
            str(jsonl_path), batch_size=4, max_length=64, max_files=10,
        )

        batch = next(iter(train_loader))
        assert batch["input_ids"].shape[1] == 64
