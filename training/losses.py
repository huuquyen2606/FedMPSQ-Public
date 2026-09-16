"""Loss helpers for FedAvg and FedProx local training."""

from __future__ import annotations

import torch
from torch import nn


def clone_trainable_parameters(model: nn.Module) -> list[torch.Tensor]:
    """Clone current trainable model parameters as the global reference."""
    return [param.detach().clone() for param in model.parameters() if param.requires_grad]


def fedprox_objective(
    ce_loss: torch.Tensor,
    model: nn.Module,
    global_parameters: list[torch.Tensor],
    mu: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CE + mu/2 * ||w_local - w_global||^2 and proximal term."""
    proximal_term = torch.zeros((), device=ce_loss.device)
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if len(trainable_parameters) != len(global_parameters):
        raise ValueError("Global parameter snapshot does not match model parameters")
    for local_param, global_param in zip(trainable_parameters, global_parameters, strict=True):
        proximal_term = proximal_term + torch.sum(
            (local_param - global_param.to(local_param.device)) ** 2
        )
    return ce_loss + (mu / 2.0) * proximal_term, proximal_term.detach()

