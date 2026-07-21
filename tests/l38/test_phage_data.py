from pathlib import Path

import pytest

from src.l38.phage_data import clean_sequences, parse_fasta, train_eval_split


def test_parse_fasta_extracts_sequences(tmp_path):
    fasta = tmp_path / "test.fasta"
    fasta.write_text(">seq1 desc\nMKRL\nRPSD\n>seq2 desc\nAACC\n")

    sequences = parse_fasta(fasta)

    assert sequences == ["MKRLRPSD", "AACC"]


def test_parse_fasta_empty_file(tmp_path):
    fasta = tmp_path / "empty.fasta"
    fasta.write_text("")

    assert parse_fasta(fasta) == []


def test_clean_sequences_drops_out_of_range_length():
    sequences = ["A" * 10, "A" * 100, "A" * 1000]
    cleaned = clean_sequences(sequences, min_len=20, max_len=512)
    assert cleaned == ["A" * 100]


def test_clean_sequences_drops_non_standard_residues():
    sequences = ["MKRLRPSDKFFELLGYKPHHVQLAIHRSTAKRRVACLGRQ", "MKRLXRPSDKFFELLGYKPHHVQLAIHRSTAKRRVACLGRQ"]
    cleaned = clean_sequences(sequences, min_len=20, max_len=512)
    assert cleaned == [sequences[0]]


def test_train_eval_split_is_deterministic_given_seed():
    sequences = [f"SEQ{i}" for i in range(100)]
    train1, eval1 = train_eval_split(sequences, eval_frac=0.2, seed=42)
    train2, eval2 = train_eval_split(sequences, eval_frac=0.2, seed=42)
    assert train1 == train2
    assert eval1 == eval2


def test_train_eval_split_sizes_and_no_overlap():
    sequences = [f"SEQ{i}" for i in range(100)]
    train, eva = train_eval_split(sequences, eval_frac=0.1, seed=0)
    assert len(eva) == 10
    assert len(train) == 90
    assert set(train).isdisjoint(set(eva))


def test_train_eval_split_rejects_invalid_frac():
    with pytest.raises(ValueError):
        train_eval_split(["A", "B"], eval_frac=1.5)
    with pytest.raises(ValueError):
        train_eval_split(["A", "B"], eval_frac=0.0)
