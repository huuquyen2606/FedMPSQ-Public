"""Frozen CICIoT2023 34-to-8 task mapping for FedMPSQ."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from data.ciciot2023 import CICIOT2023_LABELS
from data.dataset import FlowDataset, client_partition_path, load_metadata
from data.integrity import sha256_indices
from data.splits import deterministic_split_indices


FEDMPSQ_CLASS_NAMES = (
    "DDoS",
    "DoS",
    "Mirai",
    "Recon",
    "Spoofing",
    "Benign",
    "Web",
    "BruteForce",
)

SOURCE_TO_GROUP = np.asarray(
    [0] * 12
    + [1] * 4
    + [2] * 3
    + [3] * 5
    + [4] * 2
    + [5]
    + [6] * 6
    + [7],
    dtype=np.int64,
)

# The 34-class scale study keeps the untouched CICIoT2023 label space as the
# task itself, so the mapping is the identity and no grouping decision enters
# the contract.
IDENTITY_34_LABEL_SPACE = "ciciot2023_identity_34"
GROUPED_LABEL_SPACES = ("ciciot2023_34", "ciciot2023_groups_8")
# Synthetic fixtures (smoke tests, integration tests) train on a label space the
# project does not name.  They still need a well-formed evaluation contract, so
# a generic width-K space is recognised and given positional class names.
GENERIC_IDENTITY_PATTERN = re.compile(r"identity_(\d+)")


def generic_identity_label_space(num_classes: int) -> str:
    """Return the generic width-K label-space identifier."""
    width = int(num_classes)
    if width <= 0:
        raise ValueError("num_classes must be positive")
    return f"identity_{width}"


def _generic_identity_width(source_label_space: str) -> int | None:
    match = GENERIC_IDENTITY_PATTERN.fullmatch(source_label_space)
    return int(match.group(1)) if match else None


def target_class_names(source_label_space: str) -> tuple[str, ...]:
    """Return the ordered target class names for one declared label space."""
    if source_label_space in GROUPED_LABEL_SPACES:
        return FEDMPSQ_CLASS_NAMES
    if source_label_space == IDENTITY_34_LABEL_SPACE:
        return tuple(CICIOT2023_LABELS)
    width = _generic_identity_width(source_label_space)
    if width is not None:
        return tuple(f"class_{class_id}" for class_id in range(width))
    raise ValueError(f"Unsupported source label space: {source_label_space}")


def task_name(source_label_space: str) -> str:
    """Return the auditable task identifier written into the contract."""
    if source_label_space in GROUPED_LABEL_SPACES:
        return "CICIoT2023_Benign_plus_7_attack_groups"
    if source_label_space == IDENTITY_34_LABEL_SPACE:
        return "CICIoT2023_34_source_classes"
    width = _generic_identity_width(source_label_space)
    if width is not None:
        return f"generic_{width}_class_identity_task"
    raise ValueError(f"Unsupported source label space: {source_label_space}")


def minority_class_ids_from_support(
    pooled_support: np.ndarray,
    *,
    minority_fraction: float,
) -> list[int]:
    """Rank classes by pooled training support and return the rarest fraction.

    The rule is shared by every campaign so that the eight baseline runs and FedMPSQ score the
    same classes: take ``ceil(num_classes * minority_fraction)`` classes with
    the smallest pooled local-training support, breaking ties by ascending
    class id.
    """
    support = np.asarray(pooled_support, dtype=np.int64).reshape(-1)
    num_classes = int(support.size)
    if num_classes == 0:
        raise ValueError("pooled_support must not be empty")
    if not 0.0 < float(minority_fraction) <= 1.0:
        raise ValueError("minority_fraction must be in (0, 1]")
    count = max(1, int(math.ceil(num_classes * float(minority_fraction))))
    ranked = sorted(
        range(num_classes),
        key=lambda class_id: (int(support[class_id]), class_id),
    )
    return ranked[:count]


# Method options added after the rare-v3 plan was frozen. The resolved
# config is what gets hashed, so a new field would otherwise change the
# scientific hash of every arm recorded before it existed and make a
# rerun look like a different experiment. Dropping each one only while
# it holds its default keeps historical hashes bit-identical, while an
# arm that actually sets one still hashes differently from one that
# does not.
ADDITIVE_METHOD_DEFAULTS = {
    "uplink_budget_bytes": None,
    "scale_codec": "fp32",
    "incoherent_rotation": False,
    "quantizer": "integer",
}


def scientific_config_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the config as it is hashed: no resume, no defaulted extras."""
    payload = json.loads(json.dumps(config))
    if not isinstance(payload.get("results"), dict):
        raise ValueError("resolved config misses results.resume")
    payload["results"]["resume"] = None
    method = payload.get("method")
    if isinstance(method, dict):
        for key, default in ADDITIVE_METHOD_DEFAULTS.items():
            if key in method and method[key] == default:
                del method[key]
    return payload


def canonical_sha256(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def remap_ciciot2023_labels(
    labels: np.ndarray,
    *,
    source_label_space: str,
) -> np.ndarray:
    """Return stable IDs in the locked eight-group task."""
    values = np.asarray(labels, dtype=np.int64)
    if source_label_space == "ciciot2023_groups_8":
        if values.size and (values.min() < 0 or values.max() >= 8):
            raise ValueError("Eight-class source labels must be within [0, 8)")
        return values.copy()
    if source_label_space == IDENTITY_34_LABEL_SPACE:
        if values.size and (
            values.min() < 0 or values.max() >= len(CICIOT2023_LABELS)
        ):
            raise ValueError("CICIoT2023 source labels must be within [0, 34)")
        return values.copy()
    width = _generic_identity_width(source_label_space)
    if width is not None:
        if values.size and (values.min() < 0 or values.max() >= width):
            raise ValueError(f"Source labels must be within [0, {width})")
        return values.copy()
    if source_label_space != "ciciot2023_34":
        raise ValueError(f"Unsupported source label space: {source_label_space}")
    if values.size and (values.min() < 0 or values.max() >= len(SOURCE_TO_GROUP)):
        raise ValueError("CICIoT2023 source labels must be within [0, 34)")
    return SOURCE_TO_GROUP[values]


def load_grouped_npz_dataset(
    path: str | Path,
    *,
    source_label_space: str,
) -> FlowDataset:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Partition file not found: {source}")
    with np.load(source) as payload:
        features = np.asarray(payload["x"], dtype=np.float32)
        labels = remap_ciciot2023_labels(
            payload["y"],
            source_label_space=source_label_space,
        )
    return FlowDataset(features, labels)


def mapping_records(source_label_space: str) -> list[dict[str, Any]]:
    if _generic_identity_width(source_label_space) is not None:
        return [
            {
                "source_id": class_id,
                "source_name": class_name,
                "target_id": class_id,
                "target_name": class_name,
            }
            for class_id, class_name in enumerate(
                target_class_names(source_label_space)
            )
        ]
    if source_label_space == IDENTITY_34_LABEL_SPACE:
        return [
            {
                "source_id": class_id,
                "source_name": class_name,
                "target_id": class_id,
                "target_name": class_name,
            }
            for class_id, class_name in enumerate(CICIOT2023_LABELS)
        ]
    if source_label_space == "ciciot2023_groups_8":
        return [
            {
                "source_id": class_id,
                "source_name": class_name,
                "target_id": class_id,
                "target_name": class_name,
            }
            for class_id, class_name in enumerate(FEDMPSQ_CLASS_NAMES)
        ]
    if source_label_space != "ciciot2023_34":
        raise ValueError(f"Unsupported source label space: {source_label_space}")
    return [
        {
            "source_id": source_id,
            "source_name": CICIOT2023_LABELS[source_id],
            "target_id": int(SOURCE_TO_GROUP[source_id]),
            "target_name": FEDMPSQ_CLASS_NAMES[int(SOURCE_TO_GROUP[source_id])],
        }
        for source_id in range(len(CICIOT2023_LABELS))
    ]


def source_label_schema_contract(settings: Any) -> dict[str, Any]:
    """Require the NPZ integer IDs to have the exact declared semantics."""
    metadata = load_metadata(settings.data.partitions_dir)
    expected_labels = (
        list(FEDMPSQ_CLASS_NAMES)
        if settings.data.source_label_space == "ciciot2023_groups_8"
        else list(CICIOT2023_LABELS)
    )
    expected_mapping = {
        label: class_id
        for class_id, label in enumerate(expected_labels)
    }
    labels = metadata.get("labels")
    label_to_id = metadata.get("label_to_id")
    if labels != expected_labels or label_to_id != expected_mapping:
        raise ValueError(
            "metadata labels/label_to_id do not prove the integer-label "
            f"semantics for {settings.data.source_label_space}"
        )
    schema = {
        "source_label_space": settings.data.source_label_space,
        "labels": expected_labels,
        "label_to_id": expected_mapping,
    }
    return {
        **schema,
        "sha256": canonical_sha256(schema),
    }


def _distribution_statistics(
    client_support: list[list[int]],
) -> dict[str, Any]:
    matrix = np.asarray(client_support, dtype=np.float64)
    num_classes = int(matrix.shape[1])
    sizes = matrix.sum(axis=1)
    pooled = matrix.sum(axis=0)
    pooled_distribution = pooled / max(float(pooled.sum()), 1.0)
    client_records = []
    js_values = []
    for client_id, (counts, size) in enumerate(zip(matrix, sizes, strict=True)):
        distribution = counts / max(float(size), 1.0)
        positive = distribution > 0
        entropy = float(-np.sum(distribution[positive] * np.log(distribution[positive])))
        midpoint = 0.5 * (distribution + pooled_distribution)
        client_positive = distribution > 0
        pooled_positive = pooled_distribution > 0
        client_kl = float(
            np.sum(
                distribution[client_positive]
                * np.log(
                    distribution[client_positive]
                    / midpoint[client_positive]
                )
            )
        )
        pooled_kl = float(
            np.sum(
                pooled_distribution[pooled_positive]
                * np.log(
                    pooled_distribution[pooled_positive]
                    / midpoint[pooled_positive]
                )
            )
        )
        js_divergence = 0.5 * (client_kl + pooled_kl)
        js_values.append(js_divergence)
        present_counts = counts[counts > 0]
        client_records.append(
            {
                "client_id": client_id,
                "num_training_examples": int(size),
                "present_classes": int(np.count_nonzero(counts)),
                "missing_classes": int(np.count_nonzero(counts == 0)),
                "missing_class_rate": float(np.mean(counts == 0)),
                "label_entropy_nats": entropy,
                "normalized_label_entropy": float(
                    entropy / math.log(num_classes)
                ),
                "imbalance_ratio_max_to_min_present": float(
                    present_counts.max() / present_counts.min()
                ),
                "js_divergence_to_pooled_nats": js_divergence,
            }
        )
    size_mean = float(sizes.mean())
    return {
        "clients": client_records,
        "client_size_mean": size_mean,
        "client_size_sd_population": float(sizes.std(ddof=0)),
        "client_size_cv": float(sizes.std(ddof=0) / max(size_mean, 1.0)),
        "mean_missing_class_rate": float(
            np.mean([item["missing_class_rate"] for item in client_records])
        ),
        "mean_js_divergence_to_pooled_nats": float(np.mean(js_values)),
        "max_js_divergence_to_pooled_nats": float(np.max(js_values)),
    }


def _feature_audit_record(
    path: Path,
    *,
    expected_input_dim: int,
) -> dict[str, Any]:
    with np.load(path) as payload:
        features = np.asarray(payload["x"])
        labels = np.asarray(payload["y"])
    if features.ndim != 2 or features.shape[1] != expected_input_dim:
        raise ValueError(f"Unexpected feature shape in {path}: {features.shape}")
    if labels.ndim != 1 or len(labels) != len(features):
        raise ValueError(f"Feature/label row count differs in {path}")
    if features.dtype != np.float32:
        raise ValueError(f"FedMPSQ prepared features must be float32 in {path}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"FedMPSQ prepared labels must be integer typed in {path}")
    finite = np.isfinite(features)
    if not bool(finite.all()):
        raise FloatingPointError(f"Prepared features contain NaN or Inf in {path}")
    feature_min = features.min(axis=0)
    feature_max = features.max(axis=0)
    constant = feature_min == feature_max
    return {
        "file": path.name,
        "num_examples": int(len(labels)),
        "input_dim": int(features.shape[1]),
        "x_dtype": str(features.dtype),
        "y_dtype": str(labels.dtype),
        "nan_count": int(np.isnan(features).sum()),
        "positive_inf_count": int(np.isposinf(features).sum()),
        "negative_inf_count": int(np.isneginf(features).sum()),
        "feature_min": [float(value) for value in feature_min],
        "feature_max": [float(value) for value in feature_max],
        "constant_feature_count": int(constant.sum()),
        "constant_feature_rate": float(constant.mean()),
    }


def pooled_training_support(
    partitions_dir: str | Path,
    *,
    num_clients: int,
    num_classes: int,
    source_label_space: str,
    client_val_ratio: float,
    split_seed: int,
) -> np.ndarray:
    """Return pooled local-training class counts for one prepared partition.

    Only the label arrays are read, so this stays cheap enough to call at the
    start of every run.  Both campaign trainers use it, which is what keeps the
    frozen minority-class definition identical across the eight baseline runs and FedMPSQ.
    """
    pooled = np.zeros(int(num_classes), dtype=np.int64)
    for client_id in range(int(num_clients)):
        path = client_partition_path(partitions_dir, client_id)
        with np.load(path) as payload:
            labels = remap_ciciot2023_labels(
                payload["y"],
                source_label_space=source_label_space,
            )
        train_indices, _ = deterministic_split_indices(
            len(labels),
            client_val_ratio,
            int(split_seed) + client_id,
        )
        train_labels = labels[np.asarray(train_indices, dtype=np.int64)]
        pooled += np.bincount(train_labels, minlength=int(num_classes)).astype(
            np.int64
        )
    return pooled


def build_fedmpsq_task_contract(
    settings: Any,
    *,
    partition_hash: str,
) -> dict[str, Any]:
    """Freeze mapping, pooled train support, and minority IDs before training."""
    class_names = target_class_names(settings.data.source_label_space)
    num_targets = int(settings.data.target_num_classes)
    if len(class_names) != num_targets:
        raise ValueError(
            "target_num_classes disagrees with the declared label space: "
            f"{num_targets} versus {len(class_names)}"
        )
    pooled_support = np.zeros(num_targets, dtype=np.int64)
    client_support: list[list[int]] = []
    split_hash_material: list[dict[str, Any]] = []
    metadata = load_metadata(settings.data.partitions_dir)
    expected_input_dim = int(metadata["input_dim"])
    feature_names = metadata.get("feature_columns")
    if (
        not isinstance(feature_names, list)
        or len(feature_names) != expected_input_dim
        or len(set(str(name) for name in feature_names)) != expected_input_dim
    ):
        raise ValueError(
            "metadata.feature_columns must identify every input feature exactly once"
        )
    feature_records: list[dict[str, Any]] = []
    for client_id in range(settings.data.num_clients):
        client_path = client_partition_path(
            settings.data.partitions_dir,
            client_id,
        )
        with np.load(client_path) as payload:
            labels = remap_ciciot2023_labels(
                payload["y"],
                source_label_space=settings.data.source_label_space,
            )
        feature_records.append(
            {
                "role": "client_train_and_validation",
                "client_id": client_id,
                **_feature_audit_record(
                    client_path,
                    expected_input_dim=expected_input_dim,
                ),
            }
        )
        train_indices, validation_indices = deterministic_split_indices(
            len(labels),
            settings.data.client_val_ratio,
            settings.data.split_seed + client_id,
        )
        train_labels = labels[np.asarray(train_indices, dtype=np.int64)]
        counts = np.bincount(train_labels, minlength=num_targets).astype(
            np.int64
        )
        pooled_support += counts
        client_support.append([int(value) for value in counts])
        split_hash_material.append(
            {
                "client_id": client_id,
                "num_examples": len(labels),
                "num_training_examples": len(train_indices),
                "num_validation_examples": len(validation_indices),
                "train_indices_sha256": sha256_indices(train_indices),
                "validation_indices_sha256": sha256_indices(validation_indices),
            }
        )

    minority_ids = minority_class_ids_from_support(
        pooled_support,
        minority_fraction=settings.data.minority_fraction,
    )
    records = mapping_records(settings.data.source_label_space)
    source_schema = source_label_schema_contract(settings)
    support_payload = {
        "pooled_training_support": [int(value) for value in pooled_support],
        "client_training_support": client_support,
    }
    distribution_statistics = _distribution_statistics(client_support)
    global_test_path = Path(settings.data.partitions_dir) / "global_test.npz"
    feature_audit = {
        "policy": (
            "prepared NPZ arrays must be float32, finite, shape-consistent; "
            "constant features are reported rather than silently removed"
        ),
        "expected_input_dim": expected_input_dim,
        "feature_names": feature_names,
        "partitions": [
            *feature_records,
            {
                "role": "global_test_evaluation_only",
                **_feature_audit_record(
                    global_test_path,
                    expected_input_dim=expected_input_dim,
                ),
            },
        ],
    }
    return {
        "schema_version": 1,
        "protocol_version": settings.protocol_version,
        "task": task_name(settings.data.source_label_space),
        "source_label_space": settings.data.source_label_space,
        "target_class_names": list(class_names),
        "target_num_classes": num_targets,
        "source_label_schema": source_schema,
        "source_label_schema_sha256": source_schema["sha256"],
        "mapping": records,
        "mapping_sha256": canonical_sha256(records),
        "partition_hash": partition_hash,
        "split_seed": settings.data.split_seed,
        "client_val_ratio": settings.data.client_val_ratio,
        "client_split_assignments": split_hash_material,
        "split_assignment_sha256": canonical_sha256(split_hash_material),
        **support_payload,
        "support_sha256": canonical_sha256(support_payload),
        "distribution_statistics": distribution_statistics,
        "distribution_statistics_sha256": canonical_sha256(
            distribution_statistics
        ),
        "feature_audit": feature_audit,
        "feature_audit_sha256": canonical_sha256(feature_audit),
        "minority_rule": {
            "name": "bottom_pooled_training_support_fraction",
            "fraction": settings.data.minority_fraction,
            "tie_break": "ascending_target_class_id",
            "guide_status": "auditable_engineering_definition_not_numerically_specified",
        },
        "minority_class_ids": minority_ids,
        "minority_class_names": [class_names[item] for item in minority_ids],
    }


def enforce_fedmpsq_task_contract(
    results_root: str | Path,
    contract: dict[str, Any],
) -> Path:
    """Create once, then reject any mapping/support/split drift."""
    protocol_dir = Path(results_root) / "_protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    path = protocol_dir / "fedmpsq_task_contract.json"
    serialized = json.dumps(contract, indent=2, sort_keys=True, ensure_ascii=False)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError(
                "FedMPSQ task mapping, train support, minority IDs, or split "
                f"differs from the frozen contract at {path}"
            )
    return path
