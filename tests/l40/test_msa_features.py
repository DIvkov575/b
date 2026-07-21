import numpy as np

from src.l40.msa_features import compute_deletion_features, compute_profile


def _make_seqs_and_deletions():
    # 2 sequences, each 3 residues long.
    # seq 0 (query): no deletions anywhere.
    # seq 1 (homolog): deletion count 3 at res_idx=1, deletion count 0 elsewhere (absent from array).
    sequences = np.array(
        [(0, -1, 0, 3, 0, 0), (1, 555, 3, 6, 0, 1)],
        dtype=[('seq_idx', 'i2'), ('taxonomy', 'i4'), ('res_start', 'i4'),
               ('res_end', 'i4'), ('del_start', 'i4'), ('del_end', 'i4')],
    )
    deletions = np.array(
        [(1, 3)],
        dtype=[('res_idx', 'i2'), ('deletion', 'i2')],
    )
    residues = np.array(
        [(5,), (8,), (12,), (5,), (9,), (12,)],
        dtype=[('res_type', 'i1')],
    )
    return sequences, deletions, residues


class TestComputeDeletionFeatures:
    def test_returns_zero_features_for_sequence_with_no_deletions(self):
        sequences, deletions, residues = _make_seqs_and_deletions()
        has_deletion, deletion_value = compute_deletion_features(sequences[0], deletions, seq_length=3)
        assert np.array_equal(has_deletion, [False, False, False])
        assert np.allclose(deletion_value, [0.0, 0.0, 0.0])

    def test_applies_arctan_transform_at_the_correct_position(self):
        sequences, deletions, residues = _make_seqs_and_deletions()
        has_deletion, deletion_value = compute_deletion_features(sequences[1], deletions, seq_length=3)
        assert np.array_equal(has_deletion, [False, True, False])
        expected = np.pi / 2 * np.arctan(3 / 3)
        assert np.isclose(deletion_value[1], expected)
        assert deletion_value[0] == 0.0
        assert deletion_value[2] == 0.0

    def test_has_deletion_dtype_is_bool(self):
        sequences, deletions, residues = _make_seqs_and_deletions()
        has_deletion, _ = compute_deletion_features(sequences[1], deletions, seq_length=3)
        assert has_deletion.dtype == np.bool_


class TestComputeProfile:
    def test_profile_sums_to_one_per_position(self):
        sequences, deletions, residues = _make_seqs_and_deletions()
        profile = compute_profile(sequences, residues, seq_length=3, vocab_size=24)
        assert profile.shape == (3, 24)
        assert np.allclose(profile.sum(axis=1), 1.0)

    def test_profile_matches_manual_count_for_identical_column(self):
        # Position 0: seq0 has res_type 5, seq1 has res_type 5 -> profile[0, 5] should be 1.0
        sequences, deletions, residues = _make_seqs_and_deletions()
        profile = compute_profile(sequences, residues, seq_length=3, vocab_size=24)
        assert np.isclose(profile[0, 5], 1.0)

    def test_profile_splits_evenly_for_differing_column(self):
        # Position 1: seq0 has res_type 8, seq1 has res_type 9 -> both get 0.5
        sequences, deletions, residues = _make_seqs_and_deletions()
        profile = compute_profile(sequences, residues, seq_length=3, vocab_size=24)
        assert np.isclose(profile[1, 8], 0.5)
        assert np.isclose(profile[1, 9], 0.5)

    def test_single_sequence_profile_is_one_hot(self):
        sequences, deletions, residues = _make_seqs_and_deletions()
        profile = compute_profile(sequences[:1], residues, seq_length=3, vocab_size=24)
        assert np.isclose(profile[0, 5], 1.0)
        assert np.isclose(profile[1, 8], 1.0)
        assert np.isclose(profile[2, 12], 1.0)
