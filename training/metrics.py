"""Classification metrics for multi-class IDS experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def confusion_matrix_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    """Return an additive true-class x predicted-class count matrix."""
    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    predictions = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if truth.shape != predictions.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    if truth.size == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    if (
        np.any(truth < 0)
        or np.any(truth >= num_classes)
        or np.any(predictions < 0)
        or np.any(predictions >= num_classes)
    ):
        raise ValueError("Class IDs must be within [0, num_classes)")
    flat = truth * num_classes + predictions
    return np.bincount(
        flat,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def classification_metrics_from_confusion(
    confusion: np.ndarray,
) -> dict[str, float]:
    """Compute classification metrics from an additive confusion matrix."""
    matrix = np.asarray(confusion, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion must be a square matrix")
    if np.any(matrix < 0):
        raise ValueError("confusion counts must be non-negative")
    total_support = float(matrix.sum())
    if total_support <= 0:
        raise ValueError("Cannot compute metrics from an empty confusion matrix")
    true_positive = np.diag(matrix)
    supports = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted > 0,
    )
    recall = np.divide(
        true_positive,
        supports,
        out=np.zeros_like(true_positive),
        where=supports > 0,
    )
    denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    micro_true_positive = float(true_positive.sum())
    micro_false_positive = float((predicted - true_positive).sum())
    micro_false_negative = float((supports - true_positive).sum())
    micro_precision = micro_true_positive / (
        micro_true_positive + micro_false_positive
    )
    micro_recall = micro_true_positive / (
        micro_true_positive + micro_false_negative
    )
    micro_f1 = (
        2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall > 0
        else 0.0
    )
    return {
        "accuracy": float(micro_true_positive / total_support),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "weighted_precision": float(np.sum(precision * supports) / total_support),
        "weighted_recall": float(np.sum(recall * supports) / total_support),
        "weighted_f1": float(np.sum(f1 * supports) / total_support),
    }


def classification_details_from_confusion(
    confusion: np.ndarray,
) -> dict[str, object]:
    """Return confusion and per-class metrics from streaming count state."""
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion must be a square matrix")
    if np.any(matrix < 0):
        raise ValueError("confusion counts must be non-negative")
    supports = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    true_positive = np.diag(matrix).astype(np.float64)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0
    )
    recall = np.divide(
        true_positive, supports, out=np.zeros_like(true_positive), where=supports > 0
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) > 0,
    )
    return {
        "confusion_matrix": matrix.tolist(),
        "per_class": [
            {
                "class_id": class_id,
                "support": int(supports[class_id]),
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
            }
            for class_id in range(matrix.shape[0])
        ],
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int,
) -> dict[str, float]:
    """Compute accuracy and macro, micro, and weighted precision/recall/F1."""
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on an empty target array")
    confusion = confusion_matrix_counts(
        y_true,
        y_pred,
        num_classes=num_classes,
    )
    return classification_metrics_from_confusion(confusion)


# ---------------------------------------------------------------------------
# Classification metric schema v4
# ---------------------------------------------------------------------------
# Schema v2 reported accuracy plus macro/micro/weighted precision, recall and
# F1.  In single-label multi-class evaluation the three micro figures are all
# equal to accuracy, so v2 spent four of its ten slots on one number.
#
# v4 keeps the ten independent metrics of v3 and adds the two families an
# intrusion-detection paper is normally read against:
#
#   * per-class error rates the macro averages hide - specificity, false
#     positive rate, and the geometric mean of sensitivity and specificity;
#   * the binary benign-versus-attack view almost every published CICIoT2023
#     result reports, where the positive class is "any attack" and the
#     headline numbers are detection rate and false alarm rate.
#
# Prevalence-weighted precision, recall and F1 return from v2 because they are
# the standard companion to the macro averages on a skewed label distribution.
CLASSIFICATION_METRIC_SCHEMA_VERSION = 4

# The ten independent metrics carried over unchanged from v3.
CORE_METRIC_KEYS = (
    "accuracy",
    "macro_precision",
    "balanced_accuracy",
    "macro_f1",
    "multiclass_mcc",
    "cohen_kappa",
    "macro_pr_auc",
    "minority_recall",
    "minority_f1",
    "worst_class_recall",
)
# Prevalence-weighted averages over the same per-class rates.
WEIGHTED_METRIC_KEYS = (
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
)
# Error-rate view of the multi-class problem.
ERROR_RATE_METRIC_KEYS = (
    "macro_specificity",
    "macro_false_positive_rate",
    "g_mean",
    "macro_roc_auc",
)
# Benign versus any-attack. Absent when the label space has no benign class.
BINARY_METRIC_KEYS = (
    "binary_accuracy",
    "binary_precision",
    "binary_detection_rate",
    "binary_f1",
    "binary_false_alarm_rate",
    "binary_roc_auc",
    "binary_pr_auc",
)
CLASSIFICATION_METRIC_KEYS = (
    *CORE_METRIC_KEYS,
    *WEIGHTED_METRIC_KEYS,
    *ERROR_RATE_METRIC_KEYS,
    *BINARY_METRIC_KEYS,
)
# Ranking metrics need retained predicted scores; a streaming confusion matrix
# cannot produce them, so they only appear where probabilities are kept.
SCORE_DEPENDENT_METRIC_KEYS = (
    "macro_pr_auc",
    "macro_roc_auc",
    "binary_roc_auc",
    "binary_pr_auc",
)
CONFUSION_DERIVED_METRIC_KEYS = tuple(
    key for key in CLASSIFICATION_METRIC_KEYS
    if key not in SCORE_DEPENDENT_METRIC_KEYS
)
# Metrics whose theoretical range includes negative agreement.
SIGNED_METRIC_KEYS = ("multiclass_mcc", "cohen_kappa")

BINARY_POSITIVE_DEFINITION = (
    "positive = any class other than the benign class; a true attack predicted "
    "as a different attack still counts as detected"
)
G_MEAN_DEFINITION = (
    "sqrt(balanced_accuracy * macro_specificity); the geometric mean of "
    "per-class recalls is not used because a single unlearned class would "
    "force it to exactly zero"
)


def metric_keys_for(*, has_benign_class: bool, scored: bool) -> tuple[str, ...]:
    """Return the keys reportable for one label space and evaluation mode."""
    keys = [
        key for key in CLASSIFICATION_METRIC_KEYS
        if has_benign_class or key not in BINARY_METRIC_KEYS
    ]
    if not scored:
        keys = [key for key in keys if key not in SCORE_DEPENDENT_METRIC_KEYS]
    return tuple(keys)


def benign_class_id(class_names: Sequence[str]) -> int | None:
    """Locate the benign class so the binary view has a defined negative.

    The project's two real label spaces name it ``BenignTraffic`` (34-class
    CICIoT2023) and ``Benign`` (the eight-group task). Synthetic fixtures have
    no benign class, and then the binary metrics are simply not reported
    rather than being computed against an arbitrary class.
    """
    names = [str(name) for name in class_names]
    for candidate in ("BenignTraffic", "Benign"):
        if candidate in names:
            return names.index(candidate)
    return None


def per_class_rates_from_confusion(
    confusion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-class precision, recall, and F1 from a confusion matrix."""
    matrix = np.asarray(confusion, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion must be a square matrix")
    if np.any(matrix < 0):
        raise ValueError("confusion counts must be non-negative")
    num_classes = matrix.shape[0]
    true_positive = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    precision = np.divide(
        true_positive,
        predicted_support,
        out=np.zeros(num_classes, dtype=np.float64),
        where=predicted_support > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(num_classes, dtype=np.float64),
        where=support > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    return precision, recall, f1


def binary_counts_from_confusion(
    confusion: np.ndarray,
    *,
    benign_id: int,
) -> tuple[float, float, float, float]:
    """Collapse a multi-class confusion matrix to benign-versus-attack counts.

    Returns ``(true_positive, false_positive, false_negative, true_negative)``
    where positive means attack. An attack predicted as a *different* attack is
    a true positive: the operator was still alerted.
    """
    matrix = np.asarray(confusion, dtype=np.float64)
    num_classes = matrix.shape[0]
    if not 0 <= int(benign_id) < num_classes:
        raise ValueError("benign class ID outside the target label space")
    benign = int(benign_id)
    total = float(matrix.sum())
    true_negative = float(matrix[benign, benign])
    false_positive = float(matrix[benign, :].sum()) - true_negative
    false_negative = float(matrix[:, benign].sum()) - true_negative
    true_positive = total - true_negative - false_positive - false_negative
    return true_positive, false_positive, false_negative, true_negative


def classification_metrics_v4_from_confusion(
    confusion: np.ndarray,
    *,
    minority_class_ids: Sequence[int],
    benign_id: int | None = None,
) -> dict[str, float]:
    """Compute every confusion-derived metric of schema v4.

    The four ranking metrics are deliberately absent: they require retained
    predicted scores and are added by the evaluator that keeps them. The seven
    binary metrics are absent when ``benign_id`` is None, which is the case for
    label spaces that have no benign class.
    """
    matrix = np.asarray(confusion, dtype=np.float64)
    total = float(matrix.sum())
    if total <= 0:
        raise ValueError("Cannot compute metrics from an empty confusion matrix")
    num_classes = matrix.shape[0]
    minority_ids = [int(value) for value in minority_class_ids]
    if not minority_ids:
        raise ValueError("minority_class_ids must not be empty")
    if min(minority_ids) < 0 or max(minority_ids) >= num_classes:
        raise ValueError("minority class ID outside the target label space")
    precision, recall, f1 = per_class_rates_from_confusion(matrix)
    support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    correct = float(np.diag(matrix).sum())
    marginal_product = float(np.sum(support * predicted_support))
    mcc_denominator = np.sqrt(
        (total**2 - float(np.sum(predicted_support**2)))
        * (total**2 - float(np.sum(support**2)))
    )
    multiclass_mcc = (
        (correct * total - marginal_product) / mcc_denominator
        if mcc_denominator > 0
        else 0.0
    )
    accuracy = correct / total
    expected_agreement = marginal_product / (total**2)
    cohen_kappa = (
        (accuracy - expected_agreement) / (1.0 - expected_agreement)
        if expected_agreement < 1.0
        else 0.0
    )
    # Per-class specificity needs the true negatives the macro averages hide.
    true_positive = np.diag(matrix)
    false_positive = predicted_support - true_positive
    false_negative = support - true_positive
    true_negative = total - true_positive - false_positive - false_negative
    negative = true_negative + false_positive
    specificity = np.divide(
        true_negative,
        negative,
        out=np.zeros(num_classes, dtype=np.float64),
        where=negative > 0,
    )
    balanced_accuracy = float(recall.mean())
    macro_specificity = float(specificity.mean())

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(precision.mean()),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": float(f1.mean()),
        "multiclass_mcc": float(multiclass_mcc),
        "cohen_kappa": float(cohen_kappa),
        "minority_recall": float(np.mean(recall[minority_ids])),
        "minority_f1": float(np.mean(f1[minority_ids])),
        "worst_class_recall": float(np.min(recall)),
        "weighted_precision": float(np.sum(precision * support) / total),
        "weighted_recall": float(np.sum(recall * support) / total),
        "weighted_f1": float(np.sum(f1 * support) / total),
        "macro_specificity": macro_specificity,
        "macro_false_positive_rate": float(1.0 - macro_specificity),
        "g_mean": float(np.sqrt(max(balanced_accuracy * macro_specificity, 0.0))),
    }

    if benign_id is not None:
        binary_tp, binary_fp, binary_fn, binary_tn = binary_counts_from_confusion(
            matrix,
            benign_id=benign_id,
        )
        predicted_attack = binary_tp + binary_fp
        actual_attack = binary_tp + binary_fn
        actual_benign = binary_tn + binary_fp
        binary_precision = binary_tp / predicted_attack if predicted_attack > 0 else 0.0
        detection_rate = binary_tp / actual_attack if actual_attack > 0 else 0.0
        denominator = binary_precision + detection_rate
        metrics.update({
            "binary_accuracy": float((binary_tp + binary_tn) / total),
            "binary_precision": float(binary_precision),
            "binary_detection_rate": float(detection_rate),
            "binary_f1": float(
                2.0 * binary_precision * detection_rate / denominator
                if denominator > 0 else 0.0
            ),
            "binary_false_alarm_rate": float(
                binary_fp / actual_benign if actual_benign > 0 else 0.0
            ),
        })
    return metrics
