"""Tests for the L18 H1 correlation gate — the PASS/KILL adjudicator.

The gate reduces to: given per-backbone (distance, designable) observations for a
coordinate distance (d_rmsd) and the candidate feature distance (d_feat), decide
PASS / KILL-A / KILL-B / WEAK per the pre-registered rule in docs/L18_PROTOCOL.md.

Sign convention: distances are SMALL for good (designable-like) backbones, so we
expect a NEGATIVE raw correlation with the binary `designable`. The gate works on
the absolute value |rho|, so tests construct data with that in mind.
"""
import numpy as np

from src.l18.analysis import spearman_abs, bootstrap_ci, adjudicate


def _make(n, rho_target, seed):
    """Make a distance array correlated with a random binary designability label.

    Returns (distance, designable) where larger distance => less designable, with
    correlation strength tuned by mixing signal and noise.
    """
    rng = np.random.default_rng(seed)
    designable = (rng.random(n) < 0.5).astype(int)
    # base distance: designable ones smaller. noise controls |rho|.
    signal = 1.0 - designable  # 0 for designable, 1 for not
    noise = rng.normal(0, 1, n)
    dist = rho_target * signal + (1 - rho_target) * noise
    return dist, designable


def test_spearman_abs_is_absolute_value():
    # perfectly anti-correlated: distance increases as designable goes 1->0
    designable = np.array([1, 1, 0, 0])
    dist = np.array([0.1, 0.2, 0.9, 1.0])
    rho = spearman_abs(dist, designable)
    # binary label => ties cap perfect monotonic association below 1.0 (~0.894)
    assert 0.85 <= rho <= 1.0


def test_spearman_abs_near_zero_for_random():
    rng = np.random.default_rng(0)
    dist = rng.normal(0, 1, 2000)
    designable = (rng.random(2000) < 0.5).astype(int)
    rho = spearman_abs(dist, designable)
    assert rho < 0.15


def test_bootstrap_ci_brackets_point_estimate():
    dist, designable = _make(500, 0.6, seed=1)
    point = spearman_abs(dist, designable)
    lo, hi = bootstrap_ci(dist, designable, n_boot=1000, seed=7)
    assert lo <= point <= hi
    assert lo < hi


def test_adjudicate_kill_a_when_rmsd_already_strong():
    # d_rmsd alone predicts designability > 0.6 => nothing to add => KILL-A
    verdict = adjudicate(rho_feat=0.5, rho_rmsd=0.65)
    assert verdict.status == "KILL-A"


def test_adjudicate_kill_b_when_feat_no_signal():
    # feature distance < 0.3 => no signal => KILL-B (rmsd not strong enough for KILL-A)
    verdict = adjudicate(rho_feat=0.2, rho_rmsd=0.4)
    assert verdict.status == "KILL-B"


def test_adjudicate_pass_when_feat_beats_rmsd_by_margin():
    # feat > 0.3 AND feat - rmsd >= 0.15, rmsd not >0.6
    verdict = adjudicate(rho_feat=0.55, rho_rmsd=0.35)
    assert verdict.status == "PASS"


def test_adjudicate_weak_when_feat_ok_but_margin_small():
    # feat > 0.3, positive margin but < 0.15 => WEAK
    verdict = adjudicate(rho_feat=0.45, rho_rmsd=0.40)
    assert verdict.status == "WEAK"


def test_adjudicate_kill_a_takes_precedence_over_pass():
    # even if feat looks great, if rmsd > 0.6 the gate is KILL-A (nothing to add)
    verdict = adjudicate(rho_feat=0.9, rho_rmsd=0.7)
    assert verdict.status == "KILL-A"
