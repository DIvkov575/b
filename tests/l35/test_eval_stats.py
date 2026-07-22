"""Tests for src/l35/eval_stats.py: turns a list of per-seed point estimates
(e.g. match_rate from N independent eval runs with different starting-noise
seeds) into a mean + confidence interval, instead of reporting a single
n=200/one-seed point estimate as if it were exact.

Motivated directly by the roast: the original paper reported single-seed
match_rate numbers with no repeats, despite evaluate.py's own footnote
admitting a prior generator bug caused a 20-point match-rate swing on an
identical checkpoint/seed -- a single point estimate from a measurement
pipeline known to be that noisy is not trustworthy on its own.
"""
import math

import pytest


def test_aggregate_across_seeds_reports_mean():
    from src.l35.eval_stats import aggregate_across_seeds

    result = aggregate_across_seeds([0.20, 0.22, 0.18, 0.24, 0.21])

    assert result["mean"] == pytest.approx(0.21, abs=1e-9)
    assert result["n_seeds"] == 5


def test_aggregate_across_seeds_ci_widens_with_higher_variance():
    from src.l35.eval_stats import aggregate_across_seeds

    tight = aggregate_across_seeds([0.20, 0.21, 0.20, 0.21, 0.20])
    wide = aggregate_across_seeds([0.05, 0.35, 0.10, 0.30, 0.20])

    tight_width = tight["ci_high"] - tight["ci_low"]
    wide_width = wide["ci_high"] - wide["ci_low"]
    assert wide_width > tight_width


def test_aggregate_across_seeds_ci_contains_the_mean():
    from src.l35.eval_stats import aggregate_across_seeds

    result = aggregate_across_seeds([0.165, 0.20, 0.18, 0.19, 0.21])

    assert result["ci_low"] <= result["mean"] <= result["ci_high"]


def test_aggregate_across_seeds_requires_at_least_two_seeds():
    # A confidence interval from a single point estimate is undefined --
    # this must fail loudly rather than silently report a zero-width CI
    # that looks precise but isn't (the exact failure mode this function
    # exists to prevent).
    from src.l35.eval_stats import aggregate_across_seeds

    with pytest.raises(ValueError):
        aggregate_across_seeds([0.20])


def test_aggregate_across_seeds_single_seed_variance_still_flagged():
    # All-identical values across seeds (zero variance) must not produce a
    # zero-width / NaN CI that looks artificially precise.
    from src.l35.eval_stats import aggregate_across_seeds

    result = aggregate_across_seeds([0.20, 0.20, 0.20])

    assert result["mean"] == pytest.approx(0.20)
    assert not math.isnan(result["ci_low"])
    assert not math.isnan(result["ci_high"])


def test_paired_sign_test_detects_a_real_consistent_difference():
    # Structure-level paired comparison: same ground-truth structures,
    # config A matches strictly more often than config B on every seed.
    # Should report a small p-value -- a real, consistent effect, not the
    # roast's NFE=4 case where a single unpaired point estimate gave
    # p~0.06 on noise.
    from src.l35.eval_stats import paired_sign_test

    # 8 seeds, config A's match_rate beats config B's on all 8.
    a = [0.30, 0.28, 0.32, 0.29, 0.31, 0.27, 0.33, 0.30]
    b = [0.20, 0.19, 0.22, 0.18, 0.21, 0.17, 0.23, 0.20]

    result = paired_sign_test(a, b)

    assert result["p_value"] < 0.05
    assert result["n_wins_a"] == 8


def test_paired_sign_test_reports_no_significance_for_a_coin_flip_pattern():
    from src.l35.eval_stats import paired_sign_test

    # Alternating wins -- exactly the "noise, not a real difference" case.
    a = [0.30, 0.20, 0.31, 0.19, 0.29, 0.21, 0.28, 0.22]
    b = [0.20, 0.30, 0.19, 0.31, 0.21, 0.29, 0.22, 0.28]

    result = paired_sign_test(a, b)

    assert result["p_value"] > 0.05
