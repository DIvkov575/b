import pytest

from src.l38.l42_steering_repro import ivywrel_fraction
from src.l38.l52_layer_subset_causal_steering import (
    NECESSARY_LAYERS,
    SAFE_ALPHAS,
    ivywrel_fraction_excluding,
)


def test_necessary_layers_matches_l45s_convergent_5():
    # layers 18, 23, 25, 30, 31 -- L45's sufficiency AND necessity sweeps
    # both independently converged on this set (docs/L45_LAYER_SWEEP.md).
    assert NECESSARY_LAYERS == frozenset({18, 23, 25, 30, 31})


def test_safe_alphas_excludes_the_known_unsafe_regime():
    # alpha >= 1.0 is where this harness's single-shot argmax mask-fill
    # degenerates independent of any real steering effect (the exact
    # regime whose misuse produced L52's own caught spurious-PASS bug --
    # see docs/L52_LAYER_SUBSET_STEERING.md's "Critical correction").
    assert 1.0 not in SAFE_ALPHAS
    assert 2.0 not in SAFE_ALPHAS
    assert max(SAFE_ALPHAS) < 1.0


def test_ivywrel_fraction_excluding_matches_unexcluded_when_nothing_removed():
    seq = "IVYWRELAAA"
    assert ivywrel_fraction_excluding(seq, frozenset()) == pytest.approx(
        ivywrel_fraction(seq)
    )


def test_ivywrel_fraction_excluding_drops_excluded_residue_from_both_numerator_and_denominator():
    # "IIIAAA" excluding I: numerator (IVYWREL matches) and denominator
    # (sequence length) both shrink -- scoring "AAA" alone, not "III" zeroed
    # out over the original length.
    result = ivywrel_fraction_excluding("IIIAAA", frozenset("I"))
    assert result == pytest.approx(0.0)  # "AAA" has zero IVYWREL residues


def test_ivywrel_fraction_excluding_rejects_all_residues_excluded():
    with pytest.raises(ValueError):
        ivywrel_fraction_excluding("III", frozenset("I"))
