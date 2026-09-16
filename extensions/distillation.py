"""BDD-HFL bidirectional decoupled response distillation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class DecoupledLosses:
    """DREL losses for the communicated and private models."""

    global_kd: torch.Tensor
    private_kd: torch.Tensor
    global_target_class: torch.Tensor
    global_non_target_class: torch.Tensor
    private_target_class: torch.Tensor
    private_non_target_class: torch.Tensor


def _ground_truth_mask(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    labels = targets.reshape(-1, 1).long()
    return torch.zeros_like(logits, dtype=torch.bool).scatter_(1, labels, True)


def _other_class_mask(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return ~_ground_truth_mask(logits, targets)


def _collapse_target_and_other(
    probabilities: torch.Tensor,
    target_mask: torch.Tensor,
    other_mask: torch.Tensor,
) -> torch.Tensor:
    target_probability = (probabilities * target_mask).sum(dim=1, keepdim=True)
    other_probability = (probabilities * other_mask).sum(dim=1, keepdim=True)
    return torch.cat((target_probability, other_probability), dim=1)


def _one_way_decoupled_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float,
    target_class_weight: float,
    non_target_class_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    student_target = _ground_truth_mask(student_logits, targets)
    student_other = _other_class_mask(student_logits, targets)
    teacher_target = _ground_truth_mask(teacher_logits, targets)
    teacher_other = _other_class_mask(teacher_logits, targets)

    student_probabilities = F.softmax(student_logits / temperature, dim=1)
    teacher_probabilities = F.softmax(teacher_logits / temperature, dim=1)
    collapsed_student = _collapse_target_and_other(
        student_probabilities,
        student_target,
        student_other,
    )
    collapsed_teacher = _collapse_target_and_other(
        teacher_probabilities,
        teacher_target,
        teacher_other,
    )
    target_class_loss = F.kl_div(
        torch.log(collapsed_student.clamp_min(torch.finfo(collapsed_student.dtype).tiny)),
        collapsed_teacher.detach(),
        reduction="batchmean",
    ) * (temperature**2)

    target_penalty = 1000.0
    student_non_target = F.log_softmax(
        student_logits / temperature - target_penalty * student_target,
        dim=1,
    )
    teacher_non_target = F.softmax(
        teacher_logits / temperature - target_penalty * teacher_target,
        dim=1,
    )
    non_target_class_loss = F.kl_div(
        student_non_target,
        teacher_non_target.detach(),
        reduction="batchmean",
    ) * (temperature**2)
    total = (
        target_class_weight * target_class_loss
        + non_target_class_weight * non_target_class_loss
    )
    return total, target_class_loss, non_target_class_loss


def bidirectional_decoupled_losses(
    global_logits: torch.Tensor,
    private_logits: torch.Tensor,
    targets: torch.Tensor,
    config,
) -> DecoupledLosses:
    """Compute BDD-HFL Equations 1-4 in both teacher/student directions."""
    global_kd, global_target, global_non_target = _one_way_decoupled_loss(
        global_logits,
        private_logits,
        targets,
        temperature=float(config.temperature),
        target_class_weight=float(config.target_class_weight),
        non_target_class_weight=float(config.non_target_class_weight),
    )
    private_kd, private_target, private_non_target = _one_way_decoupled_loss(
        private_logits,
        global_logits,
        targets,
        temperature=float(config.temperature),
        target_class_weight=float(config.target_class_weight),
        non_target_class_weight=float(config.non_target_class_weight),
    )
    return DecoupledLosses(
        global_kd=global_kd,
        private_kd=private_kd,
        global_target_class=global_target,
        global_non_target_class=global_non_target,
        private_target_class=private_target,
        private_non_target_class=private_non_target,
    )
