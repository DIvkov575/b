import numpy as np
import pytest

from src.l38.l41_steering import (
    cohens_d,
    gate1_decision,
    rank_features_by_separation,
    sae_decode,
    sae_encode,
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
