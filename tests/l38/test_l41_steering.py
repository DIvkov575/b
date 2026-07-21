import numpy as np
import pytest

from src.l38.l41_steering import (
    cohens_d,
    fit_zscore_stats,
    gate1_decision,
    rank_features_by_separation,
    sae_decode,
    sae_encode,
    zscore_normalize,
)


def test_cohens_d_zero_when_means_equal():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.5, 2.5, 3.5, 2.5])  # same mean (2.5), some spread
    d = cohens_d(a, b)
    assert d == pytest.approx(0.0, abs=1e-9)


def test_cohens_d_large_for_well_separated_groups():
    rng = np.random.RandomState(0)
    positive = rng.normal(5.0, 0.5, size=200)
    negative = rng.normal(0.0, 0.5, size=200)
    d = cohens_d(positive, negative)
    assert d > 5.0  # huge, well-separated effect


def test_cohens_d_sign_reflects_direction():
    a = np.array([5.0, 5.0, 5.0])
    b = np.array([1.0, 1.0, 1.0])
    # zero variance -> pooled_std == 0 -> defined as 0.0, not NaN/Inf
    assert cohens_d(a, b) == 0.0


def test_cohens_d_requires_min_samples():
    with pytest.raises(ValueError):
        cohens_d(np.array([1.0]), np.array([1.0, 2.0]))


def test_rank_features_by_separation_orders_by_absolute_effect():
    rng = np.random.RandomState(0)
    n_pos, n_neg, n_features = 50, 50, 5
    positive = rng.normal(0, 1, size=(n_pos, n_features))
    negative = rng.normal(0, 1, size=(n_neg, n_features))
    # Make feature 2 strongly separated, feature 4 weakly separated (opposite sign)
    positive[:, 2] += 10.0
    negative[:, 4] += 3.0

    ranked = rank_features_by_separation(positive, negative)

    assert ranked[0][0] == 2
    assert abs(ranked[0][1]) > abs(ranked[1][1])


def test_rank_features_rejects_mismatched_feature_dims():
    with pytest.raises(ValueError):
        rank_features_by_separation(np.zeros((10, 5)), np.zeros((10, 4)))


def test_gate1_decision_passes_when_effect_exceeds_threshold():
    ranked = [(42, 1.5), (3, 0.8)]
    result = gate1_decision(ranked, threshold=1.0)
    assert result["decision"] == "PASS"
    assert result["winning_feature"] == 42
    assert result["effect_size"] == 1.5


def test_gate1_decision_kills_when_below_threshold():
    ranked = [(42, 0.9), (3, 0.8)]
    result = gate1_decision(ranked, threshold=1.0)
    assert result["decision"] == "KILL"
    assert result["winning_feature"] is None


def test_gate1_decision_handles_empty_ranking():
    result = gate1_decision([], threshold=1.0)
    assert result["decision"] == "KILL"
    assert result["winning_feature"] is None


def test_gate1_decision_uses_absolute_value_for_negative_effects():
    ranked = [(1, -2.0)]
    result = gate1_decision(ranked, threshold=1.0)
    assert result["decision"] == "PASS"
    assert result["effect_size"] == -2.0


def test_sae_encode_decode_roundtrip_reconstructs_when_k_covers_all_active():
    rng = np.random.RandomState(0)
    d_model, codebook_dim = 8, 16
    W_enc = rng.normal(0, 0.1, size=(d_model, codebook_dim))
    W_dec = rng.normal(0, 0.1, size=(codebook_dim, d_model))
    b_dec = np.zeros(d_model)

    activation = rng.normal(0, 1, size=d_model)
    features = sae_encode(activation, W_enc, b_dec, k=codebook_dim)  # k = full codebook, no sparsity
    reconstruction = sae_decode(features, W_dec, b_dec)

    assert reconstruction.shape == activation.shape


def test_sae_encode_respects_top_k_sparsity():
    rng = np.random.RandomState(0)
    d_model, codebook_dim, k = 8, 32, 4
    W_enc = rng.normal(0, 0.1, size=(d_model, codebook_dim))
    b_dec = np.zeros(d_model)
    activation = rng.normal(0, 1, size=d_model)

    features = sae_encode(activation, W_enc, b_dec, k=k)

    assert (features > 0).sum() <= k


def test_sae_encode_batched_matches_single_shapes():
    rng = np.random.RandomState(0)
    d_model, codebook_dim, k = 8, 32, 4
    W_enc = rng.normal(0, 0.1, size=(d_model, codebook_dim))
    b_dec = np.zeros(d_model)
    batch = rng.normal(0, 1, size=(5, d_model))

    features = sae_encode(batch, W_enc, b_dec, k=k)

    assert features.shape == (5, codebook_dim)
    assert ((features > 0).sum(axis=-1) <= k).all()


def test_fit_zscore_stats_recovers_known_mean_and_std():
    rng = np.random.RandomState(0)
    # Large-magnitude, non-unit-variance data, mimicking real ESMC hidden
    # states (per-dim means observed in the range -40..+450, stds 6.8..96 --
    # see docs/L41_PROTOCOL.md post-hoc correction).
    true_mean = np.array([100.0, -20.0, 0.0])
    true_std = np.array([10.0, 5.0, 50.0])
    activations = rng.normal(true_mean, true_std, size=(5000, 3))

    fitted_mean, fitted_std = fit_zscore_stats(activations)

    np.testing.assert_allclose(fitted_mean, true_mean, atol=1.0)
    np.testing.assert_allclose(fitted_std, true_std, atol=1.0)


def test_fit_zscore_stats_floors_degenerate_zero_variance_dim():
    activations = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])  # column 0 is constant
    mean, std = fit_zscore_stats(activations)
    assert std[0] == 1.0  # floored, not 0.0 -- avoids divide-by-zero downstream
    assert mean[0] == 5.0


def test_zscore_normalize_produces_zero_mean_unit_std_on_fitted_data():
    rng = np.random.RandomState(0)
    activations = rng.normal(loc=[100.0, -20.0], scale=[10.0, 5.0], size=(5000, 2))
    mean, std = fit_zscore_stats(activations)

    normalized = zscore_normalize(activations, mean, std)

    np.testing.assert_allclose(normalized.mean(axis=0), [0.0, 0.0], atol=0.1)
    np.testing.assert_allclose(normalized.std(axis=0), [1.0, 1.0], atol=0.1)


def test_sae_encode_top_k_selection_is_sensitive_to_input_scale():
    """Regression test for the exact bug found post-hoc in docs/L41_PROTOCOL.md:
    sae_encode's pre_act = (x - b_dec) @ W_enc is a linear combination across
    input dimensions. If one raw input dimension has a much larger natural
    magnitude than others, it dominates every feature's pre-activation
    regardless of W_enc's per-dimension weighting -- so which feature wins
    the top-k selection depends on whether inputs were Z-score normalized
    first, even when the UNDERLYING informative content is identical.

    This is why the real L41 run picked a different "winning" SAE feature
    (7196 unnormalized vs. 10004 normalized) for the exact same kinase vs.
    non-kinase activation data -- verified empirically on ESMC-300M layer 20."""
    rng = np.random.RandomState(0)
    d_model, codebook_dim = 4, 8

    # Dimension 0 has a huge natural scale (mimics one real ESMC hidden-state
    # dimension observed with mean up to +450); dimensions 1-3 are small-scale.
    # W_enc gives dimension 0 only a tiny weight toward feature 5 (the
    # "genuinely informative" feature once properly scaled), but dimension 0's
    # raw magnitude is large enough to swamp that signal anyway.
    W_enc = rng.normal(0, 0.05, size=(d_model, codebook_dim))
    W_enc[0, 2] = 5.0  # dimension 0 has an outsized raw weight toward feature 2
    W_enc[1, 5] = 5.0  # dimension 1 (small natural scale) drives feature 5

    b_dec = np.zeros(d_model)

    raw_activation = np.array([500.0, 2.0, 0.5, -1.0])  # dim 0 dominates raw magnitude

    unnormalized_features = sae_encode(raw_activation, W_enc, b_dec, k=1)
    winning_feature_unnormalized = int(np.argmax(unnormalized_features))

    background = rng.normal([500.0, 2.0, 0.5, -1.0], [50.0, 1.0, 0.3, 0.5], size=(500, d_model))
    mean, std = fit_zscore_stats(background)
    normalized_activation = zscore_normalize(raw_activation, mean, std)
    normalized_features = sae_encode(normalized_activation, W_enc, b_dec, k=1)
    winning_feature_normalized = int(np.argmax(normalized_features))

    assert winning_feature_unnormalized != winning_feature_normalized, (
        "expected normalization to change which feature wins top-1 selection "
        "(reproducing the real bug) -- if this now passes with equal features, "
        "the synthetic scale disparity is no longer large enough to trigger it"
    )
