"""Gate 1 (docs/L38_PROTOCOL.md): decision-rule logic tests.

Unit tests use synthetic fixtures (no network/data-cache dependency) for the
pure decision-rule function; the data-dependent degradation/stacking
functions are exercised against the real cached CSVs in a separate
data-dependent test module (skipped if the cache is absent), mirroring the
Gate 0 split between harness-logic tests and data-reproduction tests.
"""
from src.l38.gate1_analysis import evaluate_deliverable


def test_evaluate_deliverable_true_when_margin_clears_2x_se():
    # sota=0.498 (module default), need mean - 0.498 >= 2*se
    assert evaluate_deliverable(mean=0.55, se=0.02, sota=0.498) is True


def test_evaluate_deliverable_false_when_within_noise_band():
    assert evaluate_deliverable(mean=0.51, se=0.02, sota=0.498) is False


def test_evaluate_deliverable_false_when_below_sota():
    assert evaluate_deliverable(mean=0.40, se=0.02, sota=0.498) is False


def test_evaluate_deliverable_boundary_exactly_2x_se_passes():
    # mean - sota == 2*se exactly should pass (>=, not >)
    assert evaluate_deliverable(mean=0.498 + 0.04, se=0.02, sota=0.498) is True
