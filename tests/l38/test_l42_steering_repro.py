import numpy as np
import pytest

from src.l38.l42_steering_repro import (
    difference_of_means_vector,
    dose_response_is_monotonic_then_collapsing,
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
