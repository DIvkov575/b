"""Gate 0 (docs/L38_PROTOCOL.md): reproduce ProteinGym's own published
Low_MSA_depth numbers from the raw per-assay CSV, using our own bootstrap
harness. If this doesn't match within noise, the harness has a bug -- fix
before trusting any Gate 1 result.

Requires the real ProteinGym CSVs cached under src/l38/data_cache/ (pulled
from github.com/OATML-Markslab/ProteinGym). Skips if not present, since this
is a data-dependent sanity check, not a unit test of the harness logic
(covered separately in test_eval_harness.py).
"""
from pathlib import Path

import pytest

from src.l38.eval_harness import (
    bootstrap_mean_se,
    load_low_msa_depth_assay_ids,
    per_assay_scores_for_models,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "src" / "l38" / "data_cache"
REFERENCE_CSV = DATA_DIR / "DMS_substitutions.csv"
DMS_LEVEL_CSV = DATA_DIR / "DMS_substitutions_Spearman_DMS_level.csv"

pytestmark = pytest.mark.skipif(
    not (REFERENCE_CSV.exists() and DMS_LEVEL_CSV.exists()),
    reason="ProteinGym reference CSVs not cached locally under src/l38/data_cache/",
)

# Published Summary_performance_DMS_substitutions_Spearman.csv Low_MSA_depth
# column (rounded to 3dp), pulled 2026-07-20 from OATML-Markslab/ProteinGym main.
PUBLISHED_LOW_MSA_DEPTH = {
    "AIDO Protein-RAG (16B)": 0.498,
    "VenusREM": 0.495,
    "ProteinMPNN": 0.187,
}


def test_low_msa_depth_slice_has_36_assays():
    ids = load_low_msa_depth_assay_ids(REFERENCE_CSV)
    assert len(ids) == 36


def test_reproduced_means_match_published_within_bootstrap_noise():
    ids = load_low_msa_depth_assay_ids(REFERENCE_CSV)
    models = list(PUBLISHED_LOW_MSA_DEPTH)
    scores = per_assay_scores_for_models(DMS_LEVEL_CSV, assay_ids=ids, model_names=models)

    for model, published in PUBLISHED_LOW_MSA_DEPTH.items():
        mean, se, n = bootstrap_mean_se(scores[model].values, n_boot=10000, seed=0)
        assert n == 36
        # Published summary numbers are computed on a slightly different (but
        # documented-equivalent) protein-level aggregation; require our
        # from-scratch bootstrap mean to fall within +/- 3 SE of theirs. This
        # is the harness-correctness check, not a claim about the model.
        assert abs(mean - published) <= max(3 * se, 0.02), (
            f"{model}: reproduced mean {mean:.3f} vs published {published:.3f}, "
            f"SE {se:.4f} -- harness may have an aggregation bug"
        )
