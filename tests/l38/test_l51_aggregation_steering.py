import pytest

from src.l38.l51_aggregation_steering import (
    aggregation_resistance_proxy,
    aggregation_resistance_proxy_excluding,
    net_charge,
)


def test_net_charge_negative_for_acidic_homopolymer():
    # D/E-rich -> (neg - pos)/n is positive by this module's sign convention.
    assert net_charge("D" * 10) == pytest.approx(1.0)


def test_net_charge_negative_for_basic_homopolymer():
    assert net_charge("K" * 10) == pytest.approx(-1.0)


def test_net_charge_zero_for_neutral_sequence():
    assert net_charge("AAAA") == pytest.approx(0.0)


def test_net_charge_rejects_empty_sequence():
    with pytest.raises(ValueError):
        net_charge("")


def test_aggregation_resistance_proxy_is_net_charge_unchanged():
    seq = "DEKRAA"
    assert aggregation_resistance_proxy(seq) == pytest.approx(net_charge(seq))


def test_aggregation_resistance_proxy_higher_for_acidic_sequence():
    acidic = aggregation_resistance_proxy("D" * 20)
    basic = aggregation_resistance_proxy("K" * 20)
    assert acidic > basic


def test_aggregation_resistance_proxy_excluding_removes_specified_residues():
    assert aggregation_resistance_proxy_excluding("DDDKKK", frozenset("K")) == pytest.approx(
        aggregation_resistance_proxy("DDD")
    )


def test_aggregation_resistance_proxy_excluding_rejects_all_residues_excluded():
    with pytest.raises(ValueError):
        aggregation_resistance_proxy_excluding("DDD", frozenset("D"))
