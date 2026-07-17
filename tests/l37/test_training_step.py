"""Tests for the L37 consistency-distillation training step (composes consistency_distill
math with a teacher velocity model and a student raw-output model).

Uses small linear "models" (real nn.Module instances, not mocks) so the test exercises
real autograd through the actual loss function, per the anti-mock testing guidance.
"""
import torch
import torch.nn as nn

from src.l37.training_step import consistency_distillation_loss


class ConstantVelocityTeacher(nn.Module):
    """Teacher whose velocity field is a fixed constant vector, independent of (x, t)."""

    def __init__(self, velocity):
        super().__init__()
        self.velocity = velocity

    def forward(self, x, t):
        return self.velocity.expand_as(x)


class LinearStudent(nn.Module):
    """Student raw-output model: a single linear layer applied per-feature, with t ignored
    except as a broadcast bias -- enough to exercise real gradients without a real MDGen model.
    """

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x, t):
        return self.linear(x)


def test_loss_is_zero_when_student_reproduces_teacher_step_exactly():
    # Build a student whose raw output, once passed through consistency_output at t_n,
    # exactly reproduces the teacher-stepped target's consistency_output at t_next.
    # Simplest way to force this deterministically: use num_steps=1 (t_n=0, t_next=1),
    # where consistency_output at t_next=1 collapses to the identity (boundary condition),
    # so the "student self-consistency" target is just x_next itself, and the loss is zero
    # iff consistency_output(student_raw(x_n, t_n=0), x_n, t_n=0) == x_next exactly.
    # At t_n=0, consistency_output(raw, x, 0) == raw (noise-endpoint identity, tested
    # separately in test_consistency_distill.py), so we need student_raw(x_n, 0) == x_next.
    torch.manual_seed(0)
    batch, dim = 4, 3
    x_n = torch.randn(batch, dim)
    velocity = torch.randn(dim)
    teacher = ConstantVelocityTeacher(velocity)

    class PerfectStudent(nn.Module):
        def forward(self, x, t):
            # x_next = euler_step(x_n, t_n=0, t_next=1, velocity) = x_n + 1*velocity
            return x + velocity

    mask = torch.ones(batch, dim)
    loss, _ = consistency_distillation_loss(
        teacher=teacher,
        student=PerfectStudent(),
        x_n=x_n,
        num_steps=1,
        mask=mask,
        generator=torch.Generator().manual_seed(1),
    )
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)


def test_loss_is_positive_when_student_disagrees_with_teacher():
    torch.manual_seed(0)
    batch, dim = 4, 3
    x_n = torch.randn(batch, dim)
    velocity = torch.randn(dim)
    teacher = ConstantVelocityTeacher(velocity)
    student = LinearStudent(dim)
    mask = torch.ones(batch, dim)

    loss, _ = consistency_distillation_loss(
        teacher=teacher,
        student=student,
        x_n=x_n,
        num_steps=4,
        mask=mask,
        generator=torch.Generator().manual_seed(1),
    )
    assert loss.item() > 0.0


def test_teacher_parameters_receive_no_gradient():
    torch.manual_seed(0)
    batch, dim = 4, 3
    x_n = torch.randn(batch, dim, requires_grad=False)
    teacher = LinearStudent(dim)  # reuse as a "teacher" with real params, to check no-grad
    student = LinearStudent(dim)
    mask = torch.ones(batch, dim)

    loss, _ = consistency_distillation_loss(
        teacher=teacher,
        student=student,
        x_n=x_n,
        num_steps=4,
        mask=mask,
        generator=torch.Generator().manual_seed(1),
    )
    loss.backward()

    for p in teacher.parameters():
        assert p.grad is None, "teacher must be frozen (stop-gradient target)"
    for p in student.parameters():
        assert p.grad is not None, "student must receive gradients"


def test_student_deepcopied_from_frozen_teacher_still_receives_gradient():
    # Regression test: a student built by `copy.deepcopy(frozen_teacher)` inherits
    # requires_grad=False from the source tensors. Callers MUST re-enable requires_grad
    # on every student parameter after the copy, or the student silently trains as a
    # no-op (loss computes fine, .backward() raises "does not require grad"). Caught via
    # a real MDGen checkpoint smoke test; reproduced here without the checkpoint.
    import copy

    torch.manual_seed(0)
    batch, dim = 4, 3
    x_n = torch.randn(batch, dim)
    mask = torch.ones(batch, dim)

    teacher = LinearStudent(dim)
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = copy.deepcopy(teacher)
    for p in student.parameters():
        p.requires_grad_(True)  # the fix under test: must re-enable after deepcopy

    loss, _ = consistency_distillation_loss(
        teacher=teacher,
        student=student,
        x_n=x_n,
        num_steps=4,
        mask=mask,
        generator=torch.Generator().manual_seed(1),
    )
    loss.backward()

    assert all(p.grad is not None for p in student.parameters())


def test_loss_respects_mask_shape_matching_x():
    torch.manual_seed(0)
    batch, res, dim = 2, 5, 3
    x_n = torch.randn(batch, res, dim)
    teacher = ConstantVelocityTeacher(torch.randn(dim))

    class ShapePreservingStudent(nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)

    mask = torch.ones(batch, res, dim)
    mask[:, -1, :] = 0.0  # mask out the last residue

    loss, aux = consistency_distillation_loss(
        teacher=teacher,
        student=ShapePreservingStudent(),
        x_n=x_n,
        num_steps=4,
        mask=mask,
        generator=torch.Generator().manual_seed(1),
    )
    assert torch.isfinite(loss)
    assert aux["t_n"].shape == (batch,)
    assert aux["t_next"].shape == (batch,)
