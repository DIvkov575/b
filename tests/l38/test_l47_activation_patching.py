import pytest

from src.l38.l47_activation_patching import PatchTarget


def test_patch_target_accepts_valid_components():
    for component in ["attention_output", "mlp_output", "residual_stream"]:
        target = PatchTarget(layer=5, component=component)
        assert target.component == component


def test_patch_target_rejects_unknown_component():
    with pytest.raises(ValueError):
        PatchTarget(layer=0, component="not_a_real_component")


def test_patch_target_accepts_head_index_for_attention_output():
    target = PatchTarget(layer=5, component="attention_output", head=3)
    assert target.head == 3


def test_patch_target_rejects_head_index_for_non_attention_component():
    with pytest.raises(ValueError):
        PatchTarget(layer=5, component="mlp_output", head=3)


def test_patch_target_head_defaults_to_none():
    target = PatchTarget(layer=0, component="residual_stream")
    assert target.head is None
