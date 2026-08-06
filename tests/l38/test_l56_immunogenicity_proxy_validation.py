import numpy as np
import pytest

from src.l38.l56_immunogenicity_proxy_validation import (
    _core_window_scores,
    aromatic_density,
    composition_vector,
    kd_hydropathy,
    motif_core_best,
    motif_core_mean,
    motif_strong_window_density,
    p1_anchor_density,
    partial_correlation,
)


def test_core_window_scores_max_for_perfect_anchor_register():
    # F at P1, R at P4, S at P6, A at P9, with E (in none of the 4 anchor
    # sets) filling every other position -- true full-strength register.
    window = "F" + "EE" + "R" + "E" + "S" + "EE" + "A"
    assert len(window) == 9
    scores = _core_window_scores(window)
    assert scores == [pytest.approx(2.5)]  # 1.0 (P1) + 0.5*3 (P4/P6/P9)


def test_core_window_scores_zero_when_no_anchors_present():
    # E/H/P are each in none of P1/P4/P6/P9's residue sets.
    scores = _core_window_scores("EHP" * 3)
    assert scores == [pytest.approx(0.0)]


def test_core_window_scores_slides_across_multiple_registers():
    assert len(_core_window_scores("A" * 12)) == 12 - 8


def test_core_window_scores_empty_for_short_sequence():
    assert _core_window_scores("SHORT") == []


def test_motif_core_mean_zero_for_no_windows():
    assert motif_core_mean("SHORT") == pytest.approx(0.0)


def test_motif_core_best_picks_max_scoring_window():
    # a strong P1 register followed by a weak one -- best should find the strong one.
    strong = "F" + "E" * 8  # P1=F hit, E hits nothing else
    weak = "E" * 9  # hits nothing anywhere
    combined = strong + weak
    assert motif_core_best(combined) >= motif_core_best(weak)
    assert motif_core_best(strong) == pytest.approx(1.0)
    assert motif_core_best(weak) == pytest.approx(0.0)


def test_motif_strong_window_density_counts_only_above_threshold():
    # 9-char windows (positions 0,3,5,8 are P1/P4/P6/P9; E hits none of them).
    below_threshold = "FEEREEEEE"  # P1=F, P4=R hit; P6, P9 miss -> 1.5, < 2.0
    at_threshold = "FEERESEEA"  # P1=F, P4=R, P6=S, P9=A all hit -> 2.5, >= 2.0
    assert motif_strong_window_density(below_threshold) == pytest.approx(0.0)
    assert motif_strong_window_density(at_threshold) == pytest.approx(1.0)


def test_p1_anchor_density_full_for_pure_anchor_sequence():
    assert p1_anchor_density("F" * 10) == pytest.approx(1.0)


def test_p1_anchor_density_zero_for_no_anchors():
    assert p1_anchor_density("D" * 10) == pytest.approx(0.0)


def test_aromatic_density_counts_fwy_only():
    assert aromatic_density("FWYAAA") == pytest.approx(3 / 6)


def test_kd_hydropathy_matches_known_scale_value_for_homopolymer():
    assert kd_hydropathy("I" * 10) == pytest.approx(4.5)


def test_kd_hydropathy_skips_unknown_residues():
    assert kd_hydropathy("IXI") == pytest.approx(4.5)


def test_composition_vector_sums_to_one():
    vec = composition_vector("ACDEFGHIKLMNPQRSTVWY")
    assert np.sum(vec) == pytest.approx(1.0)


def test_composition_vector_length_matches_alphabet_size():
    assert len(composition_vector("A")) == 20


def test_partial_correlation_zero_when_relationship_fully_explained_by_control():
    # y is an exact linear function of the control -> once length is
    # partialled out, x (independent noise) should show ~zero partial r.
    rng = np.random.RandomState(0)
    control = rng.normal(size=200)
    x = rng.normal(size=200)
    y = 2.0 * control + 1.0
    r = partial_correlation(x, y, control)
    assert abs(r) < 0.15


def test_partial_correlation_recovers_real_relationship_independent_of_control():
    rng = np.random.RandomState(0)
    control = rng.normal(size=200)
    x = rng.normal(size=200)
    y = 3.0 * x + rng.normal(scale=0.1, size=200)  # y driven by x, not control
    r = partial_correlation(x, y, control)
    assert r > 0.9
