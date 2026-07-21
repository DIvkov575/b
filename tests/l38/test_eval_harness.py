"""L38 Gate 0/1 harness: low-MSA-depth slice selection + bootstrap stats.

Matches ProteinGym's own aggregation convention (performance_DMS_benchmarks.py):
one score per assay/protein already, bootstrap over assays (not over individual
mutants). See docs/L38_PROTOCOL.md for the locked protocol this harness serves.
"""
import numpy as np
import pandas as pd
import pytest

from src.l38.eval_harness import (
    bootstrap_mean_se,
    load_low_msa_depth_assay_ids,
    per_assay_scores_for_models,
)


def test_bootstrap_mean_se_matches_naive_mean():
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    mean, se, n = bootstrap_mean_se(values, n_boot=2000, seed=0)
    assert n == 5
    assert mean == pytest.approx(np.mean(values), abs=1e-9)
    assert se > 0.0


def test_bootstrap_mean_se_is_deterministic_given_seed():
    values = np.random.RandomState(1).normal(0.4, 0.1, size=40)
    mean1, se1, _ = bootstrap_mean_se(values, n_boot=1000, seed=42)
    mean2, se2, _ = bootstrap_mean_se(values, n_boot=1000, seed=42)
    assert mean1 == mean2
    assert se1 == se2


def test_bootstrap_mean_se_drops_nans():
    values = [0.1, 0.2, np.nan, 0.4]
    mean, se, n = bootstrap_mean_se(values, n_boot=500, seed=0)
    assert n == 3
    assert not np.isnan(mean)


def test_bootstrap_mean_se_larger_n_gives_smaller_se():
    rng = np.random.RandomState(7)
    small = rng.normal(0.4, 0.15, size=10)
    large = rng.normal(0.4, 0.15, size=200)
    _, se_small, _ = bootstrap_mean_se(small, n_boot=5000, seed=0)
    _, se_large, _ = bootstrap_mean_se(large, n_boot=5000, seed=0)
    assert se_large < se_small


def test_load_low_msa_depth_assay_ids_selects_only_low_category(tmp_path):
    ref_csv = tmp_path / "ref.csv"
    pd.DataFrame(
        {
            "DMS_id": ["a1", "a2", "a3", "a4"],
            "MSA_Neff_L_category": ["Low", "Medium", "High", "Low"],
        }
    ).to_csv(ref_csv, index=False)

    ids = load_low_msa_depth_assay_ids(ref_csv)

    assert ids == {"a1", "a4"}


def test_per_assay_scores_for_models_filters_to_requested_ids_and_models(tmp_path):
    dms_csv = tmp_path / "dms.csv"
    pd.DataFrame(
        {
            "DMS ID": ["a1", "a2", "a3"],
            "ModelX": [0.1, 0.2, 0.3],
            "ModelY": [0.4, 0.5, 0.6],
        }
    ).to_csv(dms_csv, index=False)

    result = per_assay_scores_for_models(dms_csv, assay_ids={"a1", "a3"}, model_names=["ModelX"])

    assert set(result["DMS ID"]) == {"a1", "a3"}
    assert list(result.columns) == ["DMS ID", "ModelX"]


def test_per_assay_scores_for_models_raises_on_missing_model(tmp_path):
    dms_csv = tmp_path / "dms.csv"
    pd.DataFrame({"DMS ID": ["a1"], "ModelX": [0.1]}).to_csv(dms_csv, index=False)

    with pytest.raises(KeyError):
        per_assay_scores_for_models(dms_csv, assay_ids={"a1"}, model_names=["Nonexistent"])
