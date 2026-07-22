"""Tests for virion_eval.py's bootstrap CI logic (pure numpy, no GPU/model
needed -- the embedding/probe-fitting pieces require a real ESM-2 checkpoint
and are exercised only on the remote GPU host, not unit-tested here)."""
import numpy as np

from src.l38.virion_eval import (
    bootstrap_metric_ci,
    paired_bootstrap_metric_diff,
    _accuracy_metric,
    _auc_metric,
    _f1_metric,
)


def test_bootstrap_metric_ci_point_estimate_matches_direct_computation():
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 1, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.4, 0.2, 0.3, 0.6, 0.7, 0.1])

    result = bootstrap_metric_ci(y_true, y_pred, y_prob, _accuracy_metric, n_boot=500, seed=0)

    from sklearn.metrics import accuracy_score
    assert result["point_estimate"] == accuracy_score(y_true, y_pred)


def test_bootstrap_metric_ci_bounds_contain_point_estimate():
    rng = np.random.RandomState(0)
    n = 200
    y_true = rng.randint(0, 2, size=n)
    y_prob = np.clip(y_true * 0.7 + rng.normal(0, 0.2, size=n) + 0.15, 0.01, 0.99)
    y_pred = (y_prob > 0.5).astype(int)

    result = bootstrap_metric_ci(y_true, y_pred, y_prob, _auc_metric, n_boot=2000, seed=0)

    assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]


def test_bootstrap_metric_ci_is_deterministic_given_seed():
    rng = np.random.RandomState(1)
    n = 100
    y_true = rng.randint(0, 2, size=n)
    y_pred = rng.randint(0, 2, size=n)
    y_prob = rng.uniform(size=n)

    r1 = bootstrap_metric_ci(y_true, y_pred, y_prob, _f1_metric, n_boot=1000, seed=42)
    r2 = bootstrap_metric_ci(y_true, y_pred, y_prob, _f1_metric, n_boot=1000, seed=42)

    assert r1 == r2


def test_bootstrap_metric_ci_narrower_for_larger_n():
    rng = np.random.RandomState(0)

    def make_data(n):
        y_true = rng.randint(0, 2, size=n)
        y_prob = np.clip(y_true * 0.6 + rng.normal(0, 0.25, size=n) + 0.2, 0.01, 0.99)
        y_pred = (y_prob > 0.5).astype(int)
        return y_true, y_pred, y_prob

    small = bootstrap_metric_ci(*make_data(30), _accuracy_metric, n_boot=3000, seed=0)
    large = bootstrap_metric_ci(*make_data(1000), _accuracy_metric, n_boot=3000, seed=0)

    small_width = small["ci_upper"] - small["ci_lower"]
    large_width = large["ci_upper"] - large["ci_lower"]
    assert large_width < small_width


def test_two_non_overlapping_cis_indicate_a_real_gap():
    """Sanity check on the CI mechanics themselves: if two models' accuracy
    CIs don't overlap at all, that's a strong signal the gap is real, not
    split-luck -- exactly the check L39's refinement needs to make on the
    real base-vs-finetuned comparison."""
    rng = np.random.RandomState(0)
    n = 500

    y_true = rng.randint(0, 2, size=n)
    # "weak" model: barely better than chance
    weak_prob = np.clip(y_true * 0.1 + rng.normal(0, 0.3, size=n) + 0.45, 0.01, 0.99)
    weak_pred = (weak_prob > 0.5).astype(int)
    # "strong" model: much better separation
    strong_prob = np.clip(y_true * 0.8 + rng.normal(0, 0.1, size=n) + 0.1, 0.01, 0.99)
    strong_pred = (strong_prob > 0.5).astype(int)

    weak_ci = bootstrap_metric_ci(y_true, weak_pred, weak_prob, _accuracy_metric, n_boot=3000, seed=0)
    strong_ci = bootstrap_metric_ci(y_true, strong_pred, strong_prob, _accuracy_metric, n_boot=3000, seed=0)

    assert weak_ci["ci_upper"] < strong_ci["ci_lower"]


def test_paired_bootstrap_detects_a_real_paired_improvement():
    """The paired test should be MORE sensitive than comparing independent
    CIs, since it cancels out shared per-example noise. Construct a case
    where model B beats model A on the SAME examples by a small, consistent
    margin -- a case where independent CIs would likely overlap but the
    paired difference should still cleanly exclude zero."""
    rng = np.random.RandomState(0)
    n = 500
    y_true = rng.randint(0, 2, size=n)

    # Both models share the same underlying noise realization (same rng draw),
    # but model B's probabilities are shifted slightly closer to the truth --
    # a small, CONSISTENT per-example improvement, which is exactly the kind
    # of effect a paired test is powered to detect and an unpaired CI-overlap
    # check tends to miss.
    shared_noise = rng.normal(0, 0.35, size=n)
    prob_a = np.clip(y_true * 0.5 + shared_noise + 0.25, 0.01, 0.99)
    prob_b = np.clip(y_true * 0.58 + shared_noise + 0.21, 0.01, 0.99)  # slightly better separation
    pred_a = (prob_a > 0.5).astype(int)
    pred_b = (prob_b > 0.5).astype(int)

    result = paired_bootstrap_metric_diff(y_true, pred_a, prob_a, pred_b, prob_b, _auc_metric, n_boot=5000, seed=0)

    assert result["point_estimate_diff"] > 0
    assert result["significant_at_95pct"] is True
    assert result["ci_lower"] > 0


def test_paired_bootstrap_zero_diff_when_models_identical():
    rng = np.random.RandomState(0)
    n = 200
    y_true = rng.randint(0, 2, size=n)
    prob = np.clip(y_true * 0.6 + rng.normal(0, 0.2, size=n) + 0.2, 0.01, 0.99)
    pred = (prob > 0.5).astype(int)

    result = paired_bootstrap_metric_diff(y_true, pred, prob, pred, prob, _accuracy_metric, n_boot=1000, seed=0)

    assert result["point_estimate_diff"] == 0.0
    assert result["significant_at_95pct"] is False


def test_paired_bootstrap_not_significant_for_noise_level_differences():
    rng = np.random.RandomState(0)
    n = 100
    y_true = rng.randint(0, 2, size=n)
    prob_a = rng.uniform(size=n)
    prob_b = rng.uniform(size=n)  # independent random noise, no real relationship to prob_a
    pred_a = (prob_a > 0.5).astype(int)
    pred_b = (prob_b > 0.5).astype(int)

    result = paired_bootstrap_metric_diff(y_true, pred_a, prob_a, pred_b, prob_b, _accuracy_metric, n_boot=3000, seed=0)

    assert result["significant_at_95pct"] is False
