"""Reproducible client-by-class distribution statistics for FL partitions."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from data.integrity import sha256_file
from data.integrity import sha256_indices
from data.splits import deterministic_split_indices


STATISTICS_SCHEMA_VERSION = 1
MATRIX_FILENAME = "client_train_class_counts.csv"
SUMMARY_FILENAME = "client_train_distribution_statistics.json"
TRAINING_POPULATION_SCOPE = "deterministic_local_training_subsets"


def _entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0.0]
    if len(positive) == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive)))


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = 0.5 * (left + right)

    def kl_divergence(source: np.ndarray) -> float:
        positive = source > 0.0
        return float(np.sum(source[positive] * np.log(source[positive] / midpoint[positive])))

    return 0.5 * kl_divergence(left) + 0.5 * kl_divergence(right)


def compute_client_distribution_statistics(
    client_labels: Iterable[np.ndarray],
    class_names: list[str],
    *,
    population_scope: str = TRAINING_POPULATION_SCOPE,
) -> dict[str, Any]:
    """Compute class-count matrix and distribution-skew metrics.

    Entropy and Jensen-Shannon divergence use natural logarithms. The
    normalized variants are divided by their theoretical maxima, ``log(C)``
    and ``log(2)`` respectively. The finite imbalance ratio uses only classes
    observed at that client; missing classes are reported separately.
    """
    if not class_names:
        raise ValueError("At least one class name is required")
    if not population_scope:
        raise ValueError("population_scope must be explicit")
    if len(set(class_names)) != len(class_names):
        raise ValueError("Class names must be unique")
    num_classes = len(class_names)
    rows: list[np.ndarray] = []
    for client_id, labels in enumerate(client_labels):
        values = np.asarray(labels)
        if values.ndim != 1:
            raise ValueError(f"Client {client_id} labels must be one-dimensional")
        if len(values) == 0:
            raise ValueError(f"Client {client_id} partition is empty")
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"Client {client_id} labels must be integers")
        if int(values.min()) < 0 or int(values.max()) >= num_classes:
            raise ValueError(f"Client {client_id} contains a label outside [0, {num_classes})")
        rows.append(np.bincount(values.astype(np.int64), minlength=num_classes))
    if not rows:
        raise ValueError("At least one client partition is required")

    return compute_client_distribution_statistics_from_counts(
        np.stack(rows),
        class_names,
        population_scope=population_scope,
    )


def compute_client_distribution_statistics_from_counts(
    client_class_counts: np.ndarray,
    class_names: list[str],
    *,
    population_scope: str = TRAINING_POPULATION_SCOPE,
) -> dict[str, Any]:
    """Compute distribution-skew metrics from a client-by-class count matrix."""
    if not class_names:
        raise ValueError("At least one class name is required")
    if not population_scope:
        raise ValueError("population_scope must be explicit")
    if len(set(class_names)) != len(class_names):
        raise ValueError("Class names must be unique")
    matrix = np.asarray(client_class_counts)
    if matrix.ndim != 2:
        raise ValueError("client_class_counts must be a two-dimensional matrix")
    if matrix.shape[0] == 0:
        raise ValueError("At least one client partition is required")
    if matrix.shape[1] != len(class_names):
        raise ValueError("Count-matrix width differs from class_names")
    if not np.issubdtype(matrix.dtype, np.integer):
        raise ValueError("Client class counts must be integers")
    if np.any(matrix < 0):
        raise ValueError("Client class counts must be non-negative")
    if np.any(matrix.sum(axis=1) <= 0):
        raise ValueError("Every client partition must contain at least one example")

    matrix = matrix.astype(np.int64, copy=False)
    num_classes = len(class_names)
    pooled_counts = matrix.sum(axis=0)
    pooled_probabilities = pooled_counts / pooled_counts.sum()
    max_entropy = math.log(num_classes) if num_classes > 1 else 1.0
    max_js = math.log(2.0)
    clients: list[dict[str, Any]] = []
    for client_id, counts in enumerate(matrix):
        total = int(counts.sum())
        probabilities = counts / total
        present = counts[counts > 0]
        entropy_nats = _entropy(probabilities)
        js_nats = _js_divergence(probabilities, pooled_probabilities)
        clients.append(
            {
                "client_id": client_id,
                "total_examples": total,
                "present_classes": int(len(present)),
                "missing_classes": int(num_classes - len(present)),
                "missing_class_rate": float((num_classes - len(present)) / num_classes),
                "entropy_nats": entropy_nats,
                "normalized_entropy": float(entropy_nats / max_entropy),
                "observed_class_imbalance_ratio": float(present.max() / present.min()),
                "js_divergence_to_pooled_nats": js_nats,
                "normalized_js_divergence_to_pooled": float(js_nats / max_js),
                "class_counts": {
                    name: int(count)
                    for name, count in zip(class_names, counts, strict=True)
                },
            }
        )

    return {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "num_clients": int(matrix.shape[0]),
        "num_classes": num_classes,
        "class_names": class_names,
        "population_scope": population_scope,
        "definitions": {
            "entropy_nats": "Shannon entropy -sum(p_c * ln(p_c)).",
            "normalized_entropy": "entropy_nats / ln(num_classes).",
            "observed_class_imbalance_ratio": (
                "largest non-zero class count / smallest non-zero class count; "
                "missing classes are excluded and reported by missing_class_rate."
            ),
            "missing_class_rate": "number of zero-count classes / num_classes.",
            "js_divergence_to_pooled_nats": (
                "Jensen-Shannon divergence from the pooled client distribution, "
                "using natural logarithms."
            ),
            "normalized_js_divergence_to_pooled": (
                "js_divergence_to_pooled_nats / ln(2)."
            ),
            "pooled_distribution_scope": (
                f"{population_scope}; client validation and global test excluded."
            ),
        },
        "pooled_client_class_counts": {
            name: int(count)
            for name, count in zip(class_names, pooled_counts, strict=True)
        },
        "clients": clients,
    }


def write_client_distribution_statistics(
    output_dir: str | Path,
    client_labels: Iterable[np.ndarray],
    class_names: list[str],
    *,
    population_scope: str = TRAINING_POPULATION_SCOPE,
) -> dict[str, Any]:
    """Write the client×class matrix CSV and a definitions-rich JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary = compute_client_distribution_statistics(
        client_labels,
        class_names,
        population_scope=population_scope,
    )
    metric_fields = [
        "population_scope",
        "client_id",
        "total_examples",
        "present_classes",
        "missing_classes",
        "missing_class_rate",
        "entropy_nats",
        "normalized_entropy",
        "observed_class_imbalance_ratio",
        "js_divergence_to_pooled_nats",
        "normalized_js_divergence_to_pooled",
    ]
    matrix_path = output_path / MATRIX_FILENAME
    with matrix_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields + class_names)
        writer.writeheader()
        for client in summary["clients"]:
            row = {"population_scope": population_scope}
            row.update(
                {
                    field: client[field]
                    for field in metric_fields
                    if field != "population_scope"
                }
            )
            row.update(client["class_counts"])
            writer.writerow(row)

    summary_path = output_path / SUMMARY_FILENAME
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "population_scope": population_scope,
        "matrix_file": matrix_path.name,
        "matrix_sha256": sha256_file(matrix_path),
        "summary_file": summary_path.name,
        "summary_sha256": sha256_file(summary_path),
        "metric_definitions": summary["definitions"],
    }


def export_partition_client_statistics(
    partitions_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load prepared NPZ clients and export their distribution statistics."""
    root = Path(partitions_dir)
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Partition metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    num_clients = int(metadata["num_clients"])
    num_classes = int(metadata["num_classes"])
    class_names = list(metadata.get("labels") or [])
    if not class_names:
        class_names = [f"class_{class_id}" for class_id in range(num_classes)]
    if len(class_names) != num_classes:
        raise ValueError("metadata labels length differs from num_classes")
    labels: list[np.ndarray] = []
    preprocessing = metadata.get("preprocessing", {})
    if preprocessing.get("fit_scope") != "deterministic_local_training_subsets_only":
        raise ValueError("metadata does not identify leakage-free local-training subsets")
    val_ratio = float(preprocessing["client_val_ratio"])
    seed = int(preprocessing["seed"])
    recorded_hashes = preprocessing.get("client_train_indices_sha256")
    if not isinstance(recorded_hashes, list) or len(recorded_hashes) != num_clients:
        raise ValueError("metadata lacks exact per-client training-index hashes")
    for client_id in range(num_clients):
        path = root / f"client_{client_id:03d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Client partition not found: {path}")
        with np.load(path) as payload:
            values = payload["y"]
            train_indices, _ = deterministic_split_indices(
                len(values),
                val_ratio,
                seed + client_id,
            )
            if sha256_indices(train_indices) != recorded_hashes[client_id]:
                raise ValueError(f"Training-index provenance differs for client {client_id}")
            labels.append(values[np.asarray(train_indices, dtype=np.int64)].copy())
    destination = Path(output_dir) if output_dir is not None else root
    artifacts = write_client_distribution_statistics(
        destination,
        labels,
        class_names,
        population_scope=TRAINING_POPULATION_SCOPE,
    )
    if destination.resolve() == root.resolve():
        metadata["client_statistics"] = artifacts
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return artifacts
