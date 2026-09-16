"""Federated Adaptive Pruning (FAP) active-sample selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from training.losses import clone_trainable_parameters, fedprox_objective


class IndexedSubset(Dataset):
    """View immutable source data through a persistent active-index list."""

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = [int(index) for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        source_index = self.indices[item]
        features, target = self.dataset[source_index]
        return features, target, source_index


@dataclass(frozen=True)
class FAPTrainResult:
    """Local metrics and the active subset for the next federated round."""

    metrics: dict[str, float]
    active_indices: list[int]


def _build_optimizer(model: nn.Module, algorithm) -> torch.optim.Optimizer:
    optimizer_name = algorithm.optimizer.lower()
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=algorithm.learning_rate,
            momentum=algorithm.momentum,
            weight_decay=algorithm.weight_decay,
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=algorithm.learning_rate,
            weight_decay=algorithm.weight_decay,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=algorithm.learning_rate,
            weight_decay=algorithm.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {algorithm.optimizer}")


def fisher_information_score(probabilities: np.ndarray) -> float:
    """Reproduce the official FAP score-function variance approximation."""
    safe_probabilities = np.clip(
        np.asarray(probabilities, dtype=np.float64),
        np.finfo(np.float64).tiny,
        1.0,
    )
    score_function = np.gradient(np.log(safe_probabilities))
    return float(np.var(score_function))


def make_fap_loader(
    dataset: Dataset,
    active_indices: Sequence[int],
    *,
    batch_size: int,
    shuffle_seed: int,
) -> DataLoader:
    """Build a deterministic loader without modifying the source partition."""
    if not active_indices:
        raise ValueError("FAP active subset must not be empty")
    return DataLoader(
        IndexedSubset(dataset, active_indices),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(shuffle_seed),
    )


def _laplace_noise(
    shape: torch.Size,
    *,
    scale: float,
    generator: np.random.Generator,
    parameter: torch.Tensor,
) -> torch.Tensor:
    noise = generator.laplace(0.0, scale, size=tuple(shape))
    return torch.as_tensor(noise, dtype=parameter.dtype, device=parameter.device)


def train_fap_model(
    model: nn.Module,
    train_loader: DataLoader,
    active_indices: Sequence[int],
    algorithm,
    pruning,
    device: torch.device,
    *,
    server_round: int,
    original_num_examples: int,
    noise_seed: int,
) -> FAPTrainResult:
    """Train with FAP Fisher/confidence pruning and Laplace gradient DP."""
    if len(train_loader) == 0:
        raise ValueError("train_loader has no batches")
    optimizer = _build_optimizer(model, algorithm)
    criterion = nn.CrossEntropyLoss()
    use_fedprox = algorithm.name == "fedprox" and algorithm.proximal_mu > 0.0
    global_parameters = clone_trainable_parameters(model) if use_fedprox else []
    noise_generator = np.random.default_rng(noise_seed)
    active_before = [int(index) for index in active_indices]
    if original_num_examples <= 0:
        raise ValueError("FAP original_num_examples must be positive")
    if len(active_before) > original_num_examples:
        raise ValueError("FAP active subset cannot exceed the original client split")
    effective_target = max(
        pruning.target_samples,
        int(math.ceil(original_num_examples * pruning.target_fraction)),
    )
    active_set = set(active_before)
    removal_candidates: list[int] = []
    seen_candidates: set[int] = set()
    minimum_active = max(
        1,
        int(math.ceil(effective_target * pruning.minimum_target_fraction)),
    )
    pruning_active = server_round > pruning.warmup_rounds

    model.train()
    total_examples = 0
    total_loss = 0.0
    total_ce_loss = 0.0
    total_proximal = 0.0
    total_sensitivity = 0.0
    total_noise_scale = 0.0
    total_batches = 0
    total_fisher = 0.0
    scored_examples = 0

    for _ in range(algorithm.local_epochs):
        for features, targets, source_indices in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            ce_loss = criterion(logits, targets)
            proximal_term = torch.zeros((), device=device)
            if use_fedprox:
                loss, proximal_term = fedprox_objective(
                    ce_loss,
                    model,
                    global_parameters,
                    algorithm.proximal_mu,
                )
            else:
                loss = ce_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=pruning.gradient_clip_norm,
            )
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            sensitivity = max(
                (float(gradient.norm(2).detach().cpu()) for gradient in gradients),
                default=0.0,
            )
            noise_scale = (
                sensitivity
                / float(pruning.dp_epsilon)
                * (len(active_before) / float(effective_target))
            )
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.add_(
                        _laplace_noise(
                            parameter.grad.shape,
                            scale=noise_scale,
                            generator=noise_generator,
                            parameter=parameter.grad,
                        )
                    )
            optimizer.step()

            probabilities = torch.softmax(logits.detach(), dim=1).cpu().numpy()
            max_probabilities = probabilities.max(axis=1)
            for row, source_index in enumerate(source_indices.tolist()):
                fisher_score = fisher_information_score(probabilities[row])
                total_fisher += fisher_score
                scored_examples += 1
                if (
                    pruning_active
                    and len(active_before) > minimum_active
                    and max_probabilities[row] > pruning.confidence_threshold
                    and fisher_score > pruning.fisher_threshold
                    and source_index in active_set
                    and source_index not in seen_candidates
                ):
                    removal_candidates.append(int(source_index))
                    seen_candidates.add(int(source_index))

            batch_size = int(targets.shape[0])
            total_examples += batch_size
            total_loss += float(loss.detach().cpu()) * batch_size
            total_ce_loss += float(ce_loss.detach().cpu()) * batch_size
            total_proximal += float(proximal_term.detach().cpu()) * batch_size
            total_sensitivity += sensitivity
            total_noise_scale += noise_scale
            total_batches += 1

    floor_limited_removals = max(0, len(active_before) - minimum_active)
    round_limited_removals = int(
        math.floor(len(active_before) * pruning.max_pruning_fraction_per_round)
    )
    maximum_removals = (
        min(floor_limited_removals, round_limited_removals)
        if pruning_active
        else 0
    )
    removed = set(removal_candidates[:maximum_removals])
    active_after = [index for index in active_before if index not in removed]
    denominator = max(total_examples, 1)
    original_count = len(active_before)
    return FAPTrainResult(
        metrics={
            "train_loss": total_loss / denominator,
            "train_ce_loss": total_ce_loss / denominator,
            "train_proximal_term": total_proximal / denominator,
            "fap_active_examples_before": float(original_count),
            "fap_active_examples_after": float(len(active_after)),
            "fap_pruned_examples": float(len(removed)),
            "fap_round_pruning_ratio": len(removed) / max(original_count, 1),
            "fap_mean_fisher_score": total_fisher / max(scored_examples, 1),
            "fap_mean_gradient_sensitivity": total_sensitivity / max(total_batches, 1),
            "fap_mean_laplace_scale": total_noise_scale / max(total_batches, 1),
            "fap_original_examples": float(original_num_examples),
            "fap_effective_target_samples": float(effective_target),
            "fap_minimum_active_examples": float(minimum_active),
            "fap_warmup_active": float(not pruning_active),
            "fap_max_pruning_fraction_per_round": float(
                pruning.max_pruning_fraction_per_round
            ),
            "fap_dp_epsilon": float(pruning.dp_epsilon),
            "fap_confidence_threshold": float(pruning.confidence_threshold),
            "fap_fisher_threshold": float(pruning.fisher_threshold),
        },
        active_indices=active_after,
    )
