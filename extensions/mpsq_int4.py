"""Fixed-INT4 MPS refinements: scheduled sparsity and online task saliency.

The saliency estimator is an explicit alternative to the original full-dataset
gradient estimator. Both retain the class-aware objective and EMA across rounds.
"""
from __future__ import annotations

import math
import torch


def round_sparsity(target: float, initial: float, warmup_rounds: int, server_round: int) -> float:
    """Deterministic linear ramp; quantization remains INT4 in every round."""
    if not all(math.isfinite(x) and 0 <= x < 1 for x in (target, initial)):
        raise ValueError("Sparsities must be finite and in [0, 1)")
    if initial > target or warmup_rounds < 0 or server_round < 1:
        raise ValueError("Invalid sparsity schedule")
    if warmup_rounds <= 1:
        return target
    progress = min((server_round - 1) / (warmup_rounds - 1), 1.0)
    return initial + (target - initial) * progress


class OnlineTaskSaliency:
    """Sample-weighted |w * grad(task)| measured before each SGD update.

    Since objective = task + mu/2 * ||w-global||^2, remove the proximal gradient
    analytically. This excludes optimizer weight decay too (applied by SGD only
    at step). No second forward/backward or retained autograd graph is needed.
    """
    def __init__(self, model: torch.nn.Module):
        self.sums = {name: torch.zeros_like(parameter, dtype=torch.float32)
                     for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.examples = 0

    @torch.no_grad()
    def observe(self, model, global_parameters, *, proximal_mu: float, batch_size: int):
        if batch_size <= 0:
            raise ValueError("Saliency batch size must be positive")
        for name, parameter in model.named_parameters():
            if name not in self.sums or parameter.grad is None:
                continue
            task_gradient = parameter.grad.detach() - proximal_mu * (parameter.detach() - global_parameters[name])
            importance = (parameter.detach() * task_gradient).abs().float()
            # Do not synchronize CUDA once per parameter on every minibatch.
            # Nonnegative sums preserve NaN/Inf, so validate once at finish,
            # before the saliency can be used for compression/aggregation.
            self.sums[name].add_(importance, alpha=batch_size)
        self.examples += batch_size

    @torch.no_grad()
    def finish(self, model, previous, *, eta: float):
        if not 0 <= eta < 1 or self.examples == 0:
            raise ValueError("Online saliency requires observed examples and eta in [0,1)")
        result = {}
        for name, tensor in model.state_dict().items():
            if not torch.is_floating_point(tensor):
                continue
            current = self.sums.get(name)
            current = (current / self.examples).cpu() if current is not None else torch.zeros_like(tensor, dtype=torch.float32, device="cpu")
            if not bool(torch.isfinite(current).all()):
                raise FloatingPointError(f"Nonfinite online saliency: {name}")
            old = previous.get(name, torch.zeros_like(current)).detach().cpu().float()
            if old.shape != current.shape:
                raise ValueError(f"Saliency state shape mismatch: {name}")
            if not bool(torch.isfinite(old).all()):
                raise FloatingPointError(f"Nonfinite previous saliency: {name}")
            result[name] = eta * old + (1 - eta) * current
        return result


def deterministic_topk_indices(score: torch.Tensor, keep: int) -> torch.Tensor:
    """Linear-time threshold selection with exactly the old stable tie rule.

    Each threshold subset is in coordinate order. Sparse reconstruction is
    identical to stable descending argsort, including ties on zero saliency.
    """
    if score.ndim != 1 or not 0 <= keep <= score.numel():
        raise ValueError("Invalid Top-K input")
    if not bool(torch.isfinite(score).all()):
        raise FloatingPointError("Nonfinite Top-K scores")
    if keep == 0:
        return torch.empty(0, dtype=torch.long, device=score.device)
    if keep == score.numel():
        return torch.arange(keep, device=score.device)
    threshold = torch.kthvalue(score, score.numel() - keep + 1).values
    greater = torch.nonzero(score > threshold, as_tuple=False).flatten()
    ties = torch.nonzero(score == threshold, as_tuple=False).flatten()[:keep - greater.numel()]
    return torch.cat((greater, ties))
