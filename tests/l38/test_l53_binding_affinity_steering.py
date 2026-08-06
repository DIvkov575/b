import numpy as np
import pytest

from src.l38.l53_binding_affinity_steering import (
    binding_affinity_proxy,
    binding_affinity_proxy_excluding,
    mutational_sensitivity_weights,
    parse_mutant_positions,
    unweighted_identity,
    weighted_wildtype_preservation,
)


def test_parse_mutant_positions_single_and_multi():
    assert parse_mutant_positions("A11C") == [10]
    assert parse_mutant_positions("A11C:D38C") == [10, 37]


def test_parse_mutant_positions_skips_unparseable_tokens():
    assert parse_mutant_positions("A11C:WT:xx") == [10]


def test_weights_concentrate_on_binding_reducing_positions():
    # Position 1 (index 0) mutations tank the score; position 5 (index 4) are neutral.
    mutants = ["A1C", "A1D", "E5C", "E5D"]
    scores = [-3.0, -3.0, 0.0, 0.0]
    weights = mutational_sensitivity_weights(mutants, scores, reference_length=10)
    assert weights[0] > weights[4]
    assert weights.sum() == pytest.approx(1.0)


def test_weights_reject_all_neutral_labels():
    with pytest.raises(ValueError):
        mutational_sensitivity_weights(["A1C", "A1D"], [1.0, 1.0], reference_length=5)


def test_weights_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        mutational_sensitivity_weights(["A1C"], [1.0, 2.0], reference_length=5)


def test_weighted_preservation_is_one_for_exact_wildtype():
    ref = "ACDEF"
    weights = np.array([0.5, 0.2, 0.2, 0.1, 0.0])
    assert weighted_wildtype_preservation(ref, ref, weights) == pytest.approx(1.0)


def test_weighted_preservation_drops_by_the_mutated_positions_weight():
    ref = "ACDEF"
    weights = np.array([0.5, 0.2, 0.2, 0.1, 0.0])
    # Mutating index 0 (weight 0.5) costs more than mutating index 3 (weight 0.1).
    assert weighted_wildtype_preservation("WCDEF", ref, weights) == pytest.approx(0.5)
    assert weighted_wildtype_preservation("ACDWF", ref, weights) == pytest.approx(0.9)


def test_proxy_prefers_mutating_an_insensitive_position():
    ref = "ACDEF"
    weights = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    # Both variants carry exactly one mutation, so unweighted identity is equal;
    # only WHERE the mutation lands differs.
    hits_sensitive = binding_affinity_proxy("WCDEF", ref, weights)
    hits_neutral = binding_affinity_proxy("ACDEW", ref, weights)
    assert hits_neutral > hits_sensitive


def test_proxy_is_blind_to_fidelity_alone():
    """The confound the normalization exists to remove: with weight spread
    uniformly (no position-specific knowledge), the proxy must not reward
    higher overall identity."""
    ref = "ACDEFGHIKL"
    weights = np.full(10, 0.1)
    high_fidelity = binding_affinity_proxy("ACDEFGHIKW", ref, weights)
    low_fidelity = binding_affinity_proxy("WWWWFGHIKW", ref, weights)
    assert high_fidelity == pytest.approx(low_fidelity, abs=1e-12)


def test_proxy_is_zero_for_exact_wildtype():
    ref = "ACDEF"
    weights = np.array([0.5, 0.2, 0.2, 0.1, 0.0])
    assert binding_affinity_proxy(ref, ref, weights) == pytest.approx(0.0)


def test_unweighted_identity_counts_aligned_matches():
    assert unweighted_identity("ACDEF", "ACDEF") == pytest.approx(1.0)
    assert unweighted_identity("ACDEW", "ACDEF") == pytest.approx(0.8)


def test_proxy_rejects_empty_sequence():
    with pytest.raises(ValueError):
        binding_affinity_proxy("", "ACDEF", np.ones(5))


def test_proxy_rejects_weights_shorter_than_aligned_region():
    with pytest.raises(ValueError):
        binding_affinity_proxy("ACDEF", "ACDEF", np.ones(3))


def test_proxy_excluding_masks_positions_not_characters():
    """Regression guard for the alignment bug the exclusion variant exists to
    avoid: excluding a residue must not shift downstream positions."""
    ref = "ACDEF"
    weights = np.array([0.0, 0.25, 0.25, 0.25, 0.25])
    # Sequence matches ref except index 0 carries the excluded residue W.
    # Masking index 0 leaves indices 1-4, all matching -> weighted 1.0,
    # identity 1.0 -> proxy 0.0. A character-deleting implementation would
    # instead compare "CDEF" against "ACDE" and score near -1.0.
    assert binding_affinity_proxy_excluding("WCDEF", ref, weights, frozenset("W")) == pytest.approx(0.0)


def test_proxy_excluding_rejects_all_positions_excluded():
    with pytest.raises(ValueError):
        binding_affinity_proxy_excluding("WWW", "ACD", np.ones(3), frozenset("W"))


def test_proxy_excluding_matches_plain_proxy_when_nothing_is_excluded():
    ref = "ACDEF"
    weights = np.array([0.4, 0.3, 0.2, 0.1, 0.0])
    seq = "ACDWF"
    assert binding_affinity_proxy_excluding(seq, ref, weights, frozenset()) == pytest.approx(
        binding_affinity_proxy(seq, ref, weights)
    )
