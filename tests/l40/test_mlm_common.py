import numpy as np

from src.l40.mlm_common import apply_mlm_masking, pad_or_truncate


class TestApplyMlmMasking:
    def test_masked_prob_zero_leaves_sequence_unchanged_and_no_labels(self):
        sequence = np.arange(1, 21, dtype=np.int32)
        rng = np.random.RandomState(0)
        masked, labels = apply_mlm_masking(sequence, mask_prob=0.0, mask_token=23, rng=rng)
        assert np.array_equal(masked, sequence)
        assert (labels == -100).all()

    def test_masked_prob_one_labels_every_position(self):
        sequence = np.arange(1, 21, dtype=np.int32)
        rng = np.random.RandomState(0)
        masked, labels = apply_mlm_masking(sequence, mask_prob=1.0, mask_token=23, rng=rng)
        assert (labels != -100).all()
        assert np.array_equal(labels, sequence)

    def test_deterministic_given_same_rng_state(self):
        sequence = np.arange(1, 21, dtype=np.int32)
        masked1, labels1 = apply_mlm_masking(sequence, 0.5, 23, np.random.RandomState(42))
        masked2, labels2 = apply_mlm_masking(sequence, 0.5, 23, np.random.RandomState(42))
        assert np.array_equal(masked1, masked2)
        assert np.array_equal(labels1, labels2)


class TestPadOrTruncate:
    def test_pads_short_sequence_with_pad_value(self):
        sequence = np.array([1, 2, 3], dtype=np.int32)
        result = pad_or_truncate(sequence, max_length=6, pad_value=0)
        assert np.array_equal(result, [1, 2, 3, 0, 0, 0])

    def test_truncates_long_sequence(self):
        sequence = np.arange(10, dtype=np.int32)
        result = pad_or_truncate(sequence, max_length=4, pad_value=0)
        assert np.array_equal(result, [0, 1, 2, 3])

    def test_exact_length_is_unchanged(self):
        sequence = np.array([5, 6, 7], dtype=np.int32)
        result = pad_or_truncate(sequence, max_length=3, pad_value=0)
        assert np.array_equal(result, sequence)
