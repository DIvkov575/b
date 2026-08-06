import pytest

from src.l38.l57_expression_yield_steering import (
    absolute_charge_average,
    expression_yield_proxy,
    expression_yield_proxy_excluding,
)


def test_absolute_charge_average_zero_for_neutral_sequence():
    assert absolute_charge_average("AAAA") == pytest.approx(0.0)


def test_absolute_charge_average_positive_for_pure_positive_charge():
    assert absolute_charge_average("K" * 10) == pytest.approx(1.0)


def test_absolute_charge_average_positive_for_pure_negative_charge():
    # this proxy asks "how far from neutral," so pure D is scored the same
    # magnitude as pure K -- unlike L51's signed net_charge.
    assert absolute_charge_average("D" * 10) == pytest.approx(1.0)


def test_absolute_charge_average_cancels_balanced_charge():
    assert absolute_charge_average("KD") == pytest.approx(0.0)


def test_absolute_charge_average_rejects_empty_sequence():
    with pytest.raises(ValueError):
        absolute_charge_average("")


def test_expression_yield_proxy_is_absolute_charge_average_unchanged():
    seq = "KDRE"
    assert expression_yield_proxy(seq) == pytest.approx(absolute_charge_average(seq))


def test_expression_yield_proxy_higher_for_charge_extreme_sequence():
    charged = expression_yield_proxy("K" * 20)
    neutral = expression_yield_proxy("A" * 20)
    assert charged > neutral


def test_expression_yield_proxy_excluding_removes_specified_residues():
    assert expression_yield_proxy_excluding("KKKAAA", frozenset("K")) == pytest.approx(
        expression_yield_proxy("AAA")
    )


def test_expression_yield_proxy_excluding_rejects_all_residues_excluded():
    with pytest.raises(ValueError):
        expression_yield_proxy_excluding("KKK", frozenset("K"))
