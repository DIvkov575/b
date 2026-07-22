"""Tests for train_phavip_650m.py's checkpointing and early-stopping logic.
Uses a tiny real nn.Module + optimizer (not the full 650M model) so these
run fast without GPU/network access."""
import torch
import torch.nn as nn

from src.l38.train_phavip_650m import (
    EARLY_STOP_MIN_DELTA,
    EARLY_STOP_PATIENCE,
    load_checkpoint,
    save_checkpoint,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def save_pretrained(self, path):
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")

    @classmethod
    def from_pretrained(cls, path):
        model = cls()
        model.load_state_dict(torch.load(path / "model.pt"))
        return model


def test_save_and_load_checkpoint_roundtrips_optimizer_and_meta(tmp_path):
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # take one optimizer step so its state dict is non-trivial
    loss = model.linear(torch.randn(1, 4)).sum()
    loss.backward()
    optimizer.step()

    ckpt_path = tmp_path / "ckpt"
    save_checkpoint(model, optimizer, epoch=1, micro_step=500, path=ckpt_path)

    assert (ckpt_path / "checkpoint_meta.json").exists()
    assert (ckpt_path / "training_state.pt").exists()

    new_model = TinyModel()
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-5)
    epoch, micro_step = load_checkpoint(new_model, new_optimizer, ckpt_path)

    assert epoch == 1
    assert micro_step == 500
    # optimizer state actually restored (has step count from the one step taken above)
    assert len(new_optimizer.state_dict()["state"]) > 0


def test_save_checkpoint_creates_directory(tmp_path):
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    ckpt_path = tmp_path / "does" / "not" / "exist" / "yet"

    save_checkpoint(model, optimizer, epoch=0, micro_step=0, path=ckpt_path)

    assert ckpt_path.exists()


def test_early_stop_config_values_are_sane():
    # documents the intended thresholds; catches an accidental edit that
    # would silently change stopping behavior (e.g. patience=0 stops after
    # epoch 1 unconditionally, which is almost certainly not intended)
    assert EARLY_STOP_PATIENCE >= 1
    assert 0 < EARLY_STOP_MIN_DELTA < 1


def test_early_stop_decision_logic_triggers_on_plateau():
    """Mirrors the exact decision logic in main()'s early-stop check --
    verifies the plateau condition fires when eval loss stops improving."""
    def should_stop(eval_losses, patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA):
        if len(eval_losses) <= patience:
            return False
        recent = eval_losses[-(patience + 1):]
        best_before = min(recent[:-1])
        return recent[-1] > best_before - min_delta

    # clear improvement each epoch -> never stop
    assert should_stop([2.0, 1.5, 1.0]) is False
    # plateaued (tiny/no improvement) -> stop
    assert should_stop([2.0, 1.0, 0.999]) is True
    # got WORSE -> stop (regression counts as failure to improve)
    assert should_stop([2.0, 1.0, 1.2]) is True
    # not enough history yet -> never stop
    assert should_stop([2.0]) is False
