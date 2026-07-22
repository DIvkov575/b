import pytest

from src.l38.l42_steering_repro import instability_index


def test_instability_index_known_stable_protein_lysozyme():
    # Hen egg-white lysozyme -- a canonical stable protein, commonly reported
    # instability index well under the 40 "stable" threshold in the literature.
    lysozyme = (
        "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSR"
        "NLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
    )
    result = instability_index(lysozyme)
    assert result < 40.0


def test_instability_index_high_for_proline_rich_sequence():
    # Proline strongly destabilizes most dipeptide contexts in the DIWV table.
    result = instability_index("P" * 50)
    assert result > 40.0


def test_instability_index_requires_min_length():
    with pytest.raises(ValueError):
        instability_index("A")


def test_instability_index_handles_unknown_residues_without_crashing():
    # 'X' (unknown residue, can appear in degenerate mask-fill output) should
    # not raise -- DIWV convention treats unknown dipeptides as neutral (1.0).
    result = instability_index("AXAXA")
    assert isinstance(result, float)


def test_instability_index_scales_with_length_normalization():
    # Repeating a stable-scoring dipeptide pattern should give a roughly
    # similar per-residue instability index regardless of total sequence
    # length (the formula normalizes by 10/length) -- some drift is expected
    # since a short repeat has proportionally more boundary-dipeptide effect.
    short = instability_index("AG" * 5)
    long = instability_index("AG" * 50)
    assert abs(short - long) < 10.0
