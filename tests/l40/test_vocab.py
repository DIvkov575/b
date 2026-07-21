import numpy as np

from src.l40.vocab import AA_VOCAB, sequence_to_ids


def test_sequence_to_ids_maps_standard_residues():
    result = sequence_to_ids("ARN")
    assert list(result) == [AA_VOCAB['A'], AA_VOCAB['R'], AA_VOCAB['N']]


def test_sequence_to_ids_returns_int32_array():
    result = sequence_to_ids("ARN")
    assert result.dtype == np.int32


def test_sequence_to_ids_maps_unknown_letters_to_X():
    result = sequence_to_ids("AUZ")  # U, Z are non-standard/ambiguous codes
    assert list(result) == [AA_VOCAB['A'], AA_VOCAB['X'], AA_VOCAB['X']]


def test_sequence_to_ids_is_case_insensitive():
    assert list(sequence_to_ids("arn")) == list(sequence_to_ids("ARN"))


def test_sequence_to_ids_empty_string_returns_empty_array():
    result = sequence_to_ids("")
    assert len(result) == 0
