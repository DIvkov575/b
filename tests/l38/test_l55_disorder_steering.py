import pytest

from src.l38.l55_disorder_steering import (
    disorder_proxy,
    disorder_proxy_excluding,
    top_idp_score,
)


def test_top_idp_score_high_for_proline_homopolymer():
    # P is the single most disorder-promoting residue on the published scale.
    assert top_idp_score("P" * 10) == pytest.approx(0.987)


def test_top_idp_score_low_for_tryptophan_homopolymer():
    # W is the single most order-promoting residue on the published scale.
    assert top_idp_score("W" * 10) == pytest.approx(-0.884)


def test_top_idp_score_averages_mixed_sequence():
    # PW -> mean of the two known per-residue values.
    assert top_idp_score("PW") == pytest.approx((0.987 - 0.884) / 2)


def test_top_idp_score_skips_noncanonical_residues():
    # X has no defined TOP-IDP value; should be skipped, not crash or zero it out.
    assert top_idp_score("PXP") == pytest.approx(0.987)


def test_top_idp_score_rejects_all_noncanonical_sequence():
    with pytest.raises(ValueError):
        top_idp_score("XXX")


def test_disorder_proxy_is_top_idp_score_unchanged():
    seq = "PWSK"
    assert disorder_proxy(seq) == pytest.approx(top_idp_score(seq))


def test_disorder_proxy_higher_for_disorder_promoting_sequence():
    disordered = disorder_proxy("P" * 20)
    ordered = disorder_proxy("W" * 20)
    assert disordered > ordered


def test_disorder_proxy_excluding_removes_specified_residues():
    assert disorder_proxy_excluding("PPPWWW", frozenset("W")) == pytest.approx(
        disorder_proxy("PPP")
    )


def test_disorder_proxy_excluding_rejects_all_residues_excluded():
    with pytest.raises(ValueError):
        disorder_proxy_excluding("PPP", frozenset("P"))
