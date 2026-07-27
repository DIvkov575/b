import pytest

from src.l38.l43_solubility_steering import (
    gravy_score,
    solubility_proxy,
    solubility_proxy_excluding,
)


def test_gravy_score_known_soluble_protein_ubiquitin():
    # Human ubiquitin -- canonical small, highly soluble protein.
    ubiquitin = (
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    )
    result = gravy_score(ubiquitin)
    assert result < 0.0  # negative GRAVY = hydrophilic-leaning = solubility-consistent


def test_gravy_score_high_for_hydrophobic_homopolymer():
    result = gravy_score("L" * 50)
    assert result == pytest.approx(3.8)


def test_gravy_score_low_for_hydrophilic_homopolymer():
    result = gravy_score("D" * 50)
    assert result == pytest.approx(-3.5)


def test_gravy_score_rejects_empty_sequence():
    with pytest.raises(ValueError):
        gravy_score("")


def test_gravy_score_handles_unknown_residues_without_crashing():
    result = gravy_score("AXAXA")
    assert isinstance(result, float)


def test_solubility_proxy_is_negated_gravy():
    seq = "DDDDD"
    assert solubility_proxy(seq) == pytest.approx(-gravy_score(seq))


def test_solubility_proxy_higher_for_hydrophilic_sequence():
    hydrophilic = solubility_proxy("D" * 50)
    hydrophobic = solubility_proxy("L" * 50)
    assert hydrophilic > hydrophobic


def test_solubility_proxy_excluding_removes_specified_residues():
    # "LLLDDD" with L excluded should score identically to "DDD" alone.
    assert solubility_proxy_excluding("LLLDDD", frozenset("L")) == pytest.approx(
        solubility_proxy("DDD")
    )


def test_solubility_proxy_excluding_rejects_all_residues_excluded():
    with pytest.raises(ValueError):
        solubility_proxy_excluding("LLL", frozenset("L"))
