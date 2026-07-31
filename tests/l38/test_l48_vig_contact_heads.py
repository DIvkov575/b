from pathlib import Path

import numpy as np
import pytest

from src.l38.l48_vig_contact_heads import (
    contact_background_rate,
    extract_sequence_and_contact_map,
    head_contact_enrichment,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def test_extract_sequence_and_contact_map_on_real_ubiquitin():
    # Real fixture PDB, real known sequence -- confirms the parser recovers
    # the actual ubiquitin sequence exactly, not a garbled/truncated one.
    sequence, contact_map = extract_sequence_and_contact_map(FIXTURE_DIR / "ubiquitin.pdb")
    expected = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    assert sequence == expected
    assert contact_map.shape == (len(sequence), len(sequence))
    assert contact_map.dtype == bool


def test_contact_map_is_symmetric():
    _, contact_map = extract_sequence_and_contact_map(FIXTURE_DIR / "ubiquitin.pdb")
    assert np.array_equal(contact_map, contact_map.T)


def test_contact_map_excludes_near_diagonal_pairs():
    # MIN_SEQUENCE_SEPARATION=6 -- adjacent/near residues should never be
    # marked as "contacts" regardless of real 3D distance, since they're
    # trivially close due to chain connectivity, not structure.
    _, contact_map = extract_sequence_and_contact_map(FIXTURE_DIR / "ubiquitin.pdb")
    n = contact_map.shape[0]
    for i in range(n):
        for j in range(max(0, i - 5), min(n, i + 6)):
            assert not contact_map[i, j], f"({i},{j}) within min separation should be excluded"


def test_contact_background_rate_is_between_zero_and_one():
    _, contact_map = extract_sequence_and_contact_map(FIXTURE_DIR / "ubiquitin.pdb")
    rate = contact_background_rate(contact_map)
    assert 0.0 < rate < 1.0


def test_contact_background_rate_rejects_too_short_sequence():
    tiny_map = np.zeros((3, 3), dtype=bool)
    with pytest.raises(ValueError):
        contact_background_rate(tiny_map)


def test_head_contact_enrichment_perfect_alignment_gives_high_enrichment():
    # Synthetic: attention EXACTLY matches the contact map (uniform over
    # contacts, zero elsewhere) -- should give fraction=1.0 and enrichment
    # = 1/background (the maximum possible enrichment for this contact map).
    contact_map = np.zeros((10, 10), dtype=bool)
    contact_map[0, 8] = contact_map[8, 0] = True
    contact_map[1, 9] = contact_map[9, 1] = True

    attention = np.zeros((10, 10))
    attention[0, 8] = attention[8, 0] = 0.5
    attention[1, 9] = attention[9, 1] = 0.5

    result = head_contact_enrichment(attention, contact_map)
    assert result["attention_on_contacts_fraction"] == pytest.approx(1.0)
    assert result["enrichment_ratio"] > 1.0


def test_head_contact_enrichment_uniform_attention_gives_enrichment_near_one():
    # Uniform attention over all eligible pairs should hit contacts at
    # roughly the background rate -- enrichment ratio near 1.0 (no
    # meaningful alignment, positive or negative).
    n = 10
    contact_map = np.zeros((n, n), dtype=bool)
    contact_map[0, 8] = contact_map[8, 0] = True

    attention = np.ones((n, n)) / (n * n)
    result = head_contact_enrichment(attention, contact_map)
    assert result["enrichment_ratio"] == pytest.approx(1.0, rel=0.3)


def test_head_contact_enrichment_rejects_shape_mismatch():
    contact_map = np.zeros((5, 5), dtype=bool)
    attention = np.zeros((6, 6))
    with pytest.raises(ValueError):
        head_contact_enrichment(attention, contact_map)
