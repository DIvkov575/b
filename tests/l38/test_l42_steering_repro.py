import numpy as np
import pytest

from src.l38.l42_steering_repro import (
    difference_of_means_vector,
    dose_response_is_monotonic_then_collapsing,
    is_degenerate_sequence,
    ivywrel_fraction,
    paired_bootstrap_mean_diff,
    renormalize_to_original_norm,
    split_by_percentile,
)


def test_split_by_percentile_separates_low_and_high_groups():
    sequences = [f"seq{i}" for i in range(10)]
    scores = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

    low, high = split_by_percentile(sequences, scores, low_pct=20.0, high_pct=80.0)

    assert "seq0" in low or "seq1" in low
    assert "seq9" in high or "seq8" in high
    assert set(low).isdisjoint(set(high))


def test_split_by_percentile_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        split_by_percentile(["a", "b"], np.array([1.0]))


def test_split_by_percentile_rejects_invalid_percentiles():
    with pytest.raises(ValueError):
        split_by_percentile(["a"], np.array([1.0]), low_pct=80.0, high_pct=20.0)


def test_difference_of_means_vector_zero_when_groups_identical():
    activations = np.ones((10, 4))
    vector = difference_of_means_vector(activations, activations)
    np.testing.assert_allclose(vector, np.zeros(4))


def test_difference_of_means_vector_points_toward_high_group():
    low = np.zeros((5, 3))
    high = np.ones((5, 3)) * 10.0
    vector = difference_of_means_vector(low, high)
    np.testing.assert_allclose(vector, [10.0, 10.0, 10.0])


def test_difference_of_means_vector_rejects_mismatched_dims():
    with pytest.raises(ValueError):
        difference_of_means_vector(np.zeros((5, 3)), np.zeros((5, 4)))


def test_renormalize_preserves_original_norm():
    original = np.array([[3.0, 4.0]])  # norm 5
    perturbed = np.array([[30.0, 40.0]])  # norm 50, same direction

    renormalized = renormalize_to_original_norm(perturbed, original)

    np.testing.assert_allclose(np.linalg.norm(renormalized, axis=-1), [5.0])


def test_renormalize_preserves_direction_not_just_magnitude():
    original = np.array([[3.0, 4.0]])
    perturbed = np.array([[6.0, 8.0]])  # same direction, double magnitude

    renormalized = renormalize_to_original_norm(perturbed, original)

    # direction unchanged (still [3,4] normalized), magnitude now matches original's norm (5)
    expected = np.array([[3.0, 4.0]])  # already norm 5
    np.testing.assert_allclose(renormalized, expected, atol=1e-6)


def test_renormalize_handles_near_zero_perturbed_norm_without_dividing_by_zero():
    original = np.array([[3.0, 4.0]])
    perturbed = np.array([[1e-10, 1e-10]])

    renormalized = renormalize_to_original_norm(perturbed, original)

    assert np.all(np.isfinite(renormalized))


def test_dose_response_detects_increasing_pattern():
    alphas = [0.0, 5.0, 10.0]
    effects = [0.0, 0.5, 1.0]
    assert dose_response_is_monotonic_then_collapsing(alphas, effects) is True


def test_dose_response_rejects_flat_noise():
    alphas = [0.0, 5.0, 10.0]
    effects = [0.01, -0.01, 0.005]  # flat/noisy, no real trend
    assert dose_response_is_monotonic_then_collapsing(alphas, effects, collapse_tolerance=0.02) is False


def test_dose_response_requires_matching_lengths():
    with pytest.raises(ValueError):
        dose_response_is_monotonic_then_collapsing([0.0, 1.0], [0.0])


def test_dose_response_false_for_single_point():
    assert dose_response_is_monotonic_then_collapsing([0.0], [1.0]) is False


def test_is_degenerate_sequence_flags_poly_leucine_collapse():
    # Real observed collapse artifact from the L42 run (docs/L42_STEERING_REPRO.md):
    # sustained runs of a single residue well past what real proteins show.
    poly_leucine = "MAQTLPIAEQMALLNNSLDTLFAADLSLRLLNATCPARLQNSVDQRKILRSFLDLLLSL"
    assert is_degenerate_sequence(poly_leucine) is True


def test_is_degenerate_sequence_does_not_flag_healthy_baseline():
    # Real unsteered-baseline generated sequence from the same run -- max
    # single-residue fraction 0.227, below the 0.25 threshold.
    baseline = "MNTEELKELIQKSVALLEQTEELHELLQEEPEEVERIVSLPEEERLERLKEEVIRLIQEVPQMLEELHQLLEEAGLLEYVSPILEEVEGLFMAPPKELNEETGLAALMDELFLAERLLEEVNDEYIMRVGDPMIPFDMTMLHEIVHSLIGEPYANELEQVLMIATLGLFGLELLYEKNDLLLLFMDKKLNDLLIELLQRLLEMSTQMGLDSLLQFN"
    assert is_degenerate_sequence(baseline) is False


def test_is_degenerate_sequence_threshold_is_configurable():
    seq = "AABB"  # max single-char fraction = 0.5
    assert is_degenerate_sequence(seq, max_single_aa_fraction=0.6) is False
    assert is_degenerate_sequence(seq, max_single_aa_fraction=0.4) is True


def test_is_degenerate_sequence_rejects_empty_sequence():
    with pytest.raises(ValueError):
        is_degenerate_sequence("")


def test_paired_bootstrap_mean_diff_detects_clear_positive_shift():
    rng = np.random.RandomState(0)
    scores_a = rng.normal(loc=0.0, scale=0.1, size=200)
    scores_b = scores_a + 1.0  # every paired item shifts by exactly 1.0
    result = paired_bootstrap_mean_diff(scores_a, scores_b, n_boot=2000, seed=1)
    assert result["point_estimate"] == pytest.approx(1.0, abs=0.05)
    assert result["ci_lower"] > 0.0
    assert result["significant_at_95pct"] is True


def test_paired_bootstrap_mean_diff_null_when_no_real_difference():
    # seed=1 (data) confirmed to draw a near-zero mean diff here; a generic
    # seed would occasionally land in the ~5% tail by chance and flake.
    rng = np.random.RandomState(1)
    scores_a = rng.normal(loc=0.0, scale=1.0, size=200)
    scores_b = rng.normal(loc=0.0, scale=1.0, size=200)  # independent noise, no shift
    result = paired_bootstrap_mean_diff(scores_a, scores_b, n_boot=2000, seed=1)
    assert result["significant_at_95pct"] is False


def test_paired_bootstrap_mean_diff_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap_mean_diff(np.array([1.0, 2.0]), np.array([1.0]))


def test_paired_bootstrap_mean_diff_rejects_empty_input():
    with pytest.raises(ValueError):
        paired_bootstrap_mean_diff(np.array([]), np.array([]))


def test_ivywrel_fraction_all_ivywrel_residues():
    assert ivywrel_fraction("IVYWREL") == pytest.approx(1.0)


def test_ivywrel_fraction_no_ivywrel_residues():
    assert ivywrel_fraction("ACDGHKMNPQSTF") == pytest.approx(0.0)


def test_ivywrel_fraction_mixed_sequence():
    # "LL" (2 IVYWREL residues) + "AA" (0) = 2/4
    assert ivywrel_fraction("LLAA") == pytest.approx(0.5)


def test_ivywrel_fraction_respects_custom_residue_set():
    assert ivywrel_fraction("LLAA", residues=frozenset("A")) == pytest.approx(0.5)


def test_ivywrel_fraction_rejects_empty_sequence():
    with pytest.raises(ValueError):
        ivywrel_fraction("")
