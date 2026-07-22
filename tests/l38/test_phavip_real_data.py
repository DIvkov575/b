"""Tests for the real PhaVIP/ESM-PVP-lineage dataset loader. Uses synthetic
FASTA fixtures written to a temp dir monkeypatched over DATA_DIR -- avoids
depending on the real 41MB downloaded files for unit testing."""
from pathlib import Path

import pytest

from src.l38 import phavip_real_data


@pytest.fixture
def fake_data_dir(tmp_path, monkeypatch):
    (tmp_path / "pvp.fa").write_text(
        ">p1\nMKRLRPSDKFFELLGYKPHHVQLAIHRSTAKRRVACLGRQ\n"
        ">p2\nMAAGFKTVEPLEYYRRFLKENCRPDGRELGEFRTTTVNIGS\n"
        ">p3\nMTKFSSFSLFFLIVGAYMTHVCFNMEIIGGKEVSPHSRPFM\n"
    )
    (tmp_path / "non_pvp.fa").write_text(
        ">n1\nMEPAFGEVNQLGGVFVNGRPLPNAIRLRIVELAQLGIRPCD\n"
        ">n2\nMPRSFLVRKPSDPNRKPNYSELQDSNPEFTFQQPYDQAHLL\n"
        ">n3\nMTKFSSFSLFFLIVGAYMTHVCFNMEIIGGKEVSPHSRPFM\n"  # exact dup of p3 -- should be dropped from both
    )
    monkeypatch.setattr(phavip_real_data, "DATA_DIR", tmp_path)
    return tmp_path


def test_load_pvp_labeled_dataset_assigns_correct_labels(fake_data_dir):
    sequences, labels = phavip_real_data.load_pvp_labeled_dataset(min_len=1, max_len=1000)

    n_pvp = sum(1 for l in labels if l == 1)
    n_non_pvp = sum(1 for l in labels if l == 0)
    # p3/n3 are an exact duplicate across files -> dropped from both
    assert n_pvp == 2
    assert n_non_pvp == 2
    assert len(sequences) == len(labels) == 4


def test_load_pvp_labeled_dataset_drops_cross_file_duplicates(fake_data_dir):
    sequences, labels = phavip_real_data.load_pvp_labeled_dataset(min_len=1, max_len=1000)

    dup_seq = "MTKFSSFSLFFLIVGAYMTHVCFNMEIIGGKEVSPHSRPFM"
    assert dup_seq not in sequences


def test_load_pvp_labeled_dataset_respects_length_filters(fake_data_dir):
    sequences, labels = phavip_real_data.load_pvp_labeled_dataset(min_len=1000, max_len=2000)
    assert sequences == []
    assert labels == []


def test_load_mlm_corpus_train_eval_disjoint(fake_data_dir):
    train, eva = phavip_real_data.load_mlm_corpus(eval_frac=0.5, seed=0, min_len=1, max_len=1000)

    assert set(train).isdisjoint(set(eva))
    assert len(train) + len(eva) == 4  # same dedup as above


def test_load_mlm_corpus_is_deterministic_given_seed(fake_data_dir):
    train1, eval1 = phavip_real_data.load_mlm_corpus(eval_frac=0.5, seed=42, min_len=1, max_len=1000)
    train2, eval2 = phavip_real_data.load_mlm_corpus(eval_frac=0.5, seed=42, min_len=1, max_len=1000)

    assert train1 == train2
    assert eval1 == eval2
