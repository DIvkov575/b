import pytest

from src.l38.l54_catalytic_activity_steering import (
    catalytic_activity_proxy,
    catalytic_activity_proxy_excluding,
    gly_minus_arg,
)


def test_gly_minus_arg_pure_glycine():
    assert gly_minus_arg("G" * 10) == pytest.approx(1.0)


def test_gly_minus_arg_pure_arginine():
    assert gly_minus_arg("R" * 10) == pytest.approx(-1.0)


def test_gly_minus_arg_balanced_is_zero():
    assert gly_minus_arg("GRAA") == pytest.approx(0.0)


def test_gly_minus_arg_rejects_empty_sequence():
    with pytest.raises(ValueError):
        gly_minus_arg("")


def test_catalytic_activity_proxy_is_gly_minus_arg_unchanged():
    seq = "GGRRAA"
    assert catalytic_activity_proxy(seq) == pytest.approx(gly_minus_arg(seq))


def test_catalytic_activity_proxy_higher_for_glycine_rich_sequence():
    flexible = catalytic_activity_proxy("G" * 20)
    rigid = catalytic_activity_proxy("R" * 20)
    assert flexible > rigid


def test_catalytic_activity_proxy_excluding_removes_specified_residues():
    # excluding one of the proxy's own two terms is the strictest form of
    # this check -- see docs/L54_CATALYTIC_STEERING.md's robustness section.
    assert catalytic_activity_proxy_excluding("GGGRRR", frozenset("R")) == pytest.approx(
        catalytic_activity_proxy("GGG")
    )


def test_catalytic_activity_proxy_excluding_rejects_all_residues_excluded():
    with pytest.raises(ValueError):
        catalytic_activity_proxy_excluding("GGG", frozenset("G"))
