"""Local PyTorch training and evaluation."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from extensions.distillation import bidirectional_decoupled_losses
from training.losses import clone_trainable_parameters, fedprox_objective
from training.metrics import (
    classification_details_from_confusion,
    classification_metrics,
    classification_metrics_from_confusion,
    confusion_matrix_counts,
)


def build_optimizer(model: nn.Module, algorithm) -> torch.optim.Optimizer:
    """Create the configured local optimizer."""
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


def train_local_model(
    model: nn.Module,
    train_loader: DataLoader,
    algorithm,
    device: torch.device,
    *,
    learning_rate: float | None = None,
) -> dict[str, float]:
    """Train one client locally with FedAvg or FedProx objective."""
    if len(train_loader) == 0:
        raise ValueError("train_loader has no batches")
    criterion = nn.CrossEntropyLoss()
    if learning_rate is not None:
        algorithm = type(algorithm)(**{**algorithm.__dict__, "learning_rate": learning_rate})
    optimizer = build_optimizer(model, algorithm)
    use_fedprox = algorithm.name == "fedprox" and algorithm.proximal_mu > 0.0
    global_parameters = clone_trainable_parameters(model) if use_fedprox else []

    model.train()
    total_examples = 0
    total_loss = 0.0
    total_ce_loss = 0.0
    total_proximal = 0.0
    for _ in range(algorithm.local_epochs):
        for features, targets in train_loader:
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
            optimizer.step()

            batch_size = int(targets.shape[0])
            total_examples += batch_size
            total_loss += float(loss.detach().cpu()) * batch_size
            total_ce_loss += float(ce_loss.detach().cpu()) * batch_size
            total_proximal += float(proximal_term.detach().cpu()) * batch_size

    denominator = max(total_examples, 1)
    return {
        "train_loss": total_loss / denominator,
        "train_ce_loss": total_ce_loss / denominator,
        "train_proximal_term": total_proximal / denominator,
    }


def train_bdd_hfl_models(
    global_model: nn.Module,
    private_model: nn.Module,
    train_loader: DataLoader,
    algorithm,
    distillation,
    device: torch.device,
    *,
    learning_rate: float | None = None,
) -> dict[str, float]:
    """Train persistent private and communicated models with BDD-HFL DREL."""
    if len(train_loader) == 0:
        raise ValueError("train_loader has no batches")
    if learning_rate is not None:
        algorithm = type(algorithm)(**{**algorithm.__dict__, "learning_rate": learning_rate})
    global_optimizer = build_optimizer(global_model, algorithm)
    private_optimizer = build_optimizer(private_model, algorithm)
    global_scheduler = torch.optim.lr_scheduler.StepLR(
        global_optimizer,
        step_size=distillation.scheduler_step_size,
        gamma=distillation.scheduler_gamma,
    )
    private_scheduler = torch.optim.lr_scheduler.StepLR(
        private_optimizer,
        step_size=distillation.scheduler_step_size,
        gamma=distillation.scheduler_gamma,
    )
    criterion = nn.CrossEntropyLoss()
    use_fedprox = algorithm.name == "fedprox" and algorithm.proximal_mu > 0.0
    global_parameters = (
        clone_trainable_parameters(global_model) if use_fedprox else []
    )

    global_model.train()
    private_model.train()
    totals = {
        "examples": 0,
        "global_loss": 0.0,
        "private_loss": 0.0,
        "global_ce": 0.0,
        "private_ce": 0.0,
        "global_kd": 0.0,
        "private_kd": 0.0,
        "global_target_kd": 0.0,
        "global_non_target_kd": 0.0,
        "private_target_kd": 0.0,
        "private_non_target_kd": 0.0,
        "proximal": 0.0,
    }
    for _ in range(algorithm.local_epochs):
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            global_optimizer.zero_grad(set_to_none=True)
            private_optimizer.zero_grad(set_to_none=True)

            global_logits = global_model(features)
            private_logits = private_model(features)
            losses = bidirectional_decoupled_losses(
                global_logits,
                private_logits,
                targets,
                distillation,
            )
            global_ce = criterion(global_logits, targets)
            private_ce = criterion(private_logits, targets)
            proximal_term = torch.zeros((), device=device)
            if use_fedprox:
                global_task, proximal_term = fedprox_objective(
                    global_ce,
                    global_model,
                    global_parameters,
                    algorithm.proximal_mu,
                )
            else:
                global_task = global_ce
            global_loss = global_task + distillation.global_kd_weight * losses.global_kd
            private_loss = private_ce + distillation.private_kd_weight * losses.private_kd

            private_loss.backward()
            if distillation.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    private_model.parameters(),
                    max_norm=distillation.gradient_clip_norm,
                )
            private_optimizer.step()

            global_loss.backward()
            if distillation.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    global_model.parameters(),
                    max_norm=distillation.gradient_clip_norm,
                )
            global_optimizer.step()

            batch_size = int(targets.shape[0])
            totals["examples"] += batch_size
            for key, value in {
                "global_loss": global_loss,
                "private_loss": private_loss,
                "global_ce": global_ce,
                "private_ce": private_ce,
                "global_kd": losses.global_kd,
                "private_kd": losses.private_kd,
                "global_target_kd": losses.global_target_class,
                "global_non_target_kd": losses.global_non_target_class,
                "private_target_kd": losses.private_target_class,
                "private_non_target_kd": losses.private_non_target_class,
                "proximal": proximal_term,
            }.items():
                totals[key] += float(value.detach().cpu()) * batch_size
        private_scheduler.step()
        global_scheduler.step()

    denominator = max(int(totals["examples"]), 1)
    return {
        "train_loss": totals["global_loss"] / denominator,
        "train_ce_loss": totals["global_ce"] / denominator,
        "train_proximal_term": totals["proximal"] / denominator,
        "bdd_private_loss": totals["private_loss"] / denominator,
        "bdd_private_ce_loss": totals["private_ce"] / denominator,
        "bdd_global_kd_loss": totals["global_kd"] / denominator,
        "bdd_private_kd_loss": totals["private_kd"] / denominator,
        "bdd_global_target_kd_loss": totals["global_target_kd"] / denominator,
        "bdd_global_non_target_kd_loss": totals["global_non_target_kd"] / denominator,
        "bdd_private_target_kd_loss": totals["private_target_kd"] / denominator,
        "bdd_private_non_target_kd_loss": totals["private_non_target_kd"] / denominator,
        "bdd_temperature": float(distillation.temperature),
        "bdd_global_kd_weight": float(distillation.global_kd_weight),
        "bdd_private_kd_weight": float(distillation.private_kd_weight),
    }


@torch.no_grad()
def evaluate_cross_entropy_loss(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """Return mean local CE before training, as required by DAdaQuant."""
    if len(data_loader) == 0:
        raise ValueError("data_loader has no batches")
    was_training = model.training
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_examples = 0
    for features, targets in data_loader:
        features = features.to(device)
        targets = targets.to(device)
        total_loss += float(criterion(model(features), targets).detach().cpu())
        total_examples += int(targets.shape[0])
    model.train(was_training)
    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    *,
    num_classes: int,
) -> dict[str, float]:
    """Evaluate model and return scalar metrics."""
    if len(data_loader) == 0:
        raise ValueError("data_loader has no batches")
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    total_loss = 0.0
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    for features, targets in data_loader:
        features = features.to(device)
        targets = targets.to(device)
        logits = model(features)
        total_loss += float(criterion(logits, targets).detach().cpu())
        predictions = torch.argmax(logits, dim=1)
        all_targets.append(targets.detach().cpu().numpy())
        all_predictions.append(predictions.detach().cpu().numpy())
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)
    metrics = classification_metrics(y_true, y_pred, num_classes=num_classes)
    metrics["loss"] = total_loss / max(len(y_true), 1)
    return metrics


@torch.no_grad()
def evaluate_model_detailed(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    *,
    num_classes: int,
) -> tuple[dict[str, float], dict[str, object]]:
    """Evaluate with streaming confusion state and no retained probabilities."""
    if len(data_loader) == 0:
        raise ValueError("data_loader has no batches")
    criterion = nn.CrossEntropyLoss(reduction="sum")
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss = 0.0
    total_examples = 0
    was_training = model.training
    model.eval()
    try:
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            total_loss += float(criterion(logits, targets).detach().cpu())
            predictions = torch.argmax(logits, dim=1)
            confusion += confusion_matrix_counts(
                targets.detach().cpu().numpy(),
                predictions.detach().cpu().numpy(),
                num_classes=num_classes,
            )
            total_examples += int(targets.shape[0])
    finally:
        model.train(was_training)
    metrics = classification_metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / total_examples
    return metrics, classification_details_from_confusion(confusion)
