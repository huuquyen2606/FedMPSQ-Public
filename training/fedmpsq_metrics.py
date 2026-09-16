"""Detailed detection metrics required by the FedMPSQ protocol."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from training.metrics import (
    CLASSIFICATION_METRIC_KEYS,
    benign_class_id as resolve_benign_class_id,
    classification_metrics_v4_from_confusion,
    classification_metrics_from_confusion,
    confusion_matrix_counts,
    per_class_rates_from_confusion,
)

REQUIRED_QUALITY_KEYS = (
    "accuracy", "macro_precision", "micro_precision", "weighted_precision",
    "macro_recall", "micro_recall", "weighted_recall", "macro_f1", "micro_f1", "weighted_f1",
)
SCALAR_METRIC_KEYS = tuple(dict.fromkeys(("loss", *CLASSIFICATION_METRIC_KEYS, *REQUIRED_QUALITY_KEYS)))


def _specificity(confusion: np.ndarray, class_id: int) -> float:
    """Return the one-versus-rest specificity of a single class."""
    matrix = np.asarray(confusion, dtype=np.float64)
    total = float(matrix.sum())
    true_positive = float(matrix[class_id, class_id])
    false_positive = float(matrix[:, class_id].sum()) - true_positive
    false_negative = float(matrix[class_id, :].sum()) - true_positive
    true_negative = total - true_positive - false_positive - false_negative
    negative = true_negative + false_positive
    return true_negative / negative if negative > 0 else 0.0


def detailed_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    *,
    class_names: tuple[str, ...] | list[str],
    minority_class_ids: tuple[int, ...] | list[int],
    benign_id: int | None = None,
) -> dict[str, Any]:
    """Compute scalar, per-class, ranking, and confusion-matrix metrics.

    ``benign_id`` defaults to the benign class of the supplied label space, so
    the binary benign-versus-attack block appears automatically for the two
    real tasks and is omitted for synthetic fixtures that have no benign class.
    """
    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    predictions = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64)
    num_classes = len(class_names)
    if truth.size == 0:
        raise ValueError("Cannot evaluate an empty target set")
    if predictions.shape != truth.shape:
        raise ValueError("Predictions and targets must have identical shapes")
    if scores.shape != (truth.size, num_classes):
        raise ValueError("Probability matrix has an incompatible shape")
    if (
        truth.min() < 0
        or truth.max() >= num_classes
        or predictions.min() < 0
        or predictions.max() >= num_classes
    ):
        raise ValueError("Targets or predictions fall outside the class space")
    if not bool(np.isfinite(scores).all()):
        raise FloatingPointError("Probability matrix contains NaN or Inf")

    confusion = confusion_matrix_counts(
        truth,
        predictions,
        num_classes=num_classes,
    ).astype(np.float64)
    support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    precision, recall, f1 = per_class_rates_from_confusion(confusion)

    if benign_id is None:
        benign_id = resolve_benign_class_id(class_names)

    per_class_pr_auc: list[float | None] = []
    per_class_roc_auc: list[float | None] = []
    for class_id in range(num_classes):
        binary_truth = (truth == class_id).astype(np.int64)
        positives = int(binary_truth.sum())
        if positives == 0:
            per_class_pr_auc.append(None)
            per_class_roc_auc.append(None)
            continue
        per_class_pr_auc.append(
            float(average_precision_score(binary_truth, scores[:, class_id]))
        )
        # ROC-AUC is undefined when one of the two groups is empty, which
        # happens for a class that covers the entire evaluation set.
        if positives == binary_truth.size:
            per_class_roc_auc.append(None)
        else:
            per_class_roc_auc.append(
                float(roc_auc_score(binary_truth, scores[:, class_id]))
            )
    supported_pr_auc = [value for value in per_class_pr_auc if value is not None]
    supported_roc_auc = [value for value in per_class_roc_auc if value is not None]
    minority_ids = [int(value) for value in minority_class_ids]
    if not minority_ids:
        raise ValueError("minority_class_ids must not be empty")
    if min(minority_ids) < 0 or max(minority_ids) >= num_classes:
        raise ValueError("minority class ID outside the target label space")

    per_class = []
    for class_id, class_name in enumerate(class_names):
        per_class.append(
            {
                "class_id": class_id,
                "class_name": str(class_name),
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
                "support": int(support[class_id]),
                "predicted_support": int(predicted_support[class_id]),
                "pr_auc": per_class_pr_auc[class_id],
                "roc_auc": per_class_roc_auc[class_id],
                "specificity": float(
                    _specificity(confusion, class_id)
                ),
                "is_frozen_minority": class_id in minority_ids,
                "is_benign": benign_id is not None and class_id == benign_id,
            }
        )

    ranking = {
        "macro_pr_auc": float(np.mean(supported_pr_auc)) if supported_pr_auc else 0.0,
        "macro_roc_auc": float(np.mean(supported_roc_auc)) if supported_roc_auc else 0.0,
    }
    if benign_id is not None:
        attack_truth = (truth != benign_id).astype(np.int64)
        # One minus the benign probability is the model's total attack mass, so
        # it is the natural score for the collapsed binary problem.
        attack_score = 1.0 - scores[:, benign_id]
        positives = int(attack_truth.sum())
        ranking["binary_roc_auc"] = float(
            roc_auc_score(attack_truth, attack_score)
            if 0 < positives < attack_truth.size else 0.0
        )
        ranking["binary_pr_auc"] = float(
            average_precision_score(attack_truth, attack_score)
            if positives > 0 else 0.0
        )

    return {
        **classification_metrics_v4_from_confusion(
            confusion,
            minority_class_ids=minority_ids,
            benign_id=benign_id,
        ),
        **ranking,
        **classification_metrics_from_confusion(confusion),
        "macro_pr_auc_supported_classes": len(supported_pr_auc),
        "macro_roc_auc_supported_classes": len(supported_roc_auc),
        "macro_pr_auc_absent_class_policy": "exclude_absent_truth_classes_and_log_count",
        "benign_class_id": benign_id,
        "num_examples": int(truth.size),
        "confusion_matrix": confusion.astype(np.int64).tolist(),
        "per_class": per_class,
    }


@torch.no_grad()
def evaluate_model_detailed(
    model: nn.Module,
    loaders: DataLoader | Iterable[DataLoader],
    device: torch.device,
    *,
    class_names: tuple[str, ...] | list[str],
    minority_class_ids: tuple[int, ...] | list[int],
    benign_id: int | None = None,
) -> dict[str, Any]:
    """Evaluate one model on one loader or the ordered union of loaders."""
    loader_iterable = (loaders,) if isinstance(loaders, DataLoader) else loaders
    criterion = nn.CrossEntropyLoss(reduction="sum")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    all_truth: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    saw_loader = False
    for loader in loader_iterable:
        saw_loader = True
        if len(loader) == 0:
            continue
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            total_loss += float(criterion(logits, targets).detach().cpu())
            probabilities = torch.softmax(logits, dim=1)
            all_truth.append(targets.detach().cpu().numpy())
            all_predictions.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
            all_probabilities.append(probabilities.detach().cpu().numpy())
    model.train(was_training)
    if not saw_loader:
        raise ValueError("At least one evaluation loader is required")
    if not all_truth:
        raise ValueError("Evaluation loaders contain no examples")
    truth = np.concatenate(all_truth)
    metrics = detailed_classification_metrics(
        truth,
        np.concatenate(all_predictions),
        np.concatenate(all_probabilities),
        class_names=class_names,
        minority_class_ids=minority_class_ids,
        benign_id=benign_id,
    )
    metrics["loss"] = total_loss / max(len(truth), 1)
    return metrics


def scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Return the scalar figures actually present in one evaluation.

    The seven binary keys are absent for label spaces with no benign class, so
    the projection skips missing keys rather than failing on a fixture that was
    never meant to have them.
    """
    return {
        key: float(metrics[key])
        for key in SCALAR_METRIC_KEYS
        if key in metrics
    }
