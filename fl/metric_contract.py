"""Frozen evaluation contract shared by every scenario of one campaign.

Both trainers in this repository must score the *same* classes as minority and
report the *same* metrics under the *same* benign-class definition, otherwise a
baseline and the proposed method are not comparable.  This module derives that contract from the prepared partition
alone and then refuses any later drift, mirroring the fail-closed behaviour of
``data.fedmpsq_labels.enforce_fedmpsq_task_contract``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.fedmpsq_labels import (
    canonical_sha256,
    minority_class_ids_from_support,
    pooled_training_support,
    target_class_names,
)
from training.metrics import (
    BINARY_POSITIVE_DEFINITION,
    CLASSIFICATION_METRIC_SCHEMA_VERSION,
    G_MEAN_DEFINITION,
    SCORE_DEPENDENT_METRIC_KEYS,
    benign_class_id,
    metric_keys_for,
)

METRIC_CONTRACT_FILENAME = "classification_metric_contract.json"

MINORITY_RULE = {
    "name": "bottom_pooled_training_support_fraction",
    "tie_break": "ascending_target_class_id",
    "support_scope": (
        "deterministic_local_training_subsets; client validation and global "
        "test excluded"
    ),
}


def build_metric_contract(
    *,
    partitions_dir: str | Path,
    num_clients: int,
    num_classes: int,
    source_label_space: str,
    client_val_ratio: float,
    split_seed: int,
    minority_fraction: float,
    partition_hash: str,
) -> dict[str, Any]:
    """Freeze class names, pooled support, minority classes, and the benign class.

    The benign class is part of the contract because it defines the negative
    side of the binary benign-versus-attack view: two runs that disagreed about
    it would report incomparable detection and false-alarm rates.
    """
    class_names = target_class_names(source_label_space)
    if len(class_names) != int(num_classes):
        raise ValueError(
            "num_classes disagrees with the declared label space: "
            f"{num_classes} versus {len(class_names)}"
        )
    support = pooled_training_support(
        partitions_dir,
        num_clients=num_clients,
        num_classes=num_classes,
        source_label_space=source_label_space,
        client_val_ratio=client_val_ratio,
        split_seed=split_seed,
    )
    minority_ids = minority_class_ids_from_support(
        support,
        minority_fraction=minority_fraction,
    )
    benign_id = benign_class_id(class_names)
    scored_keys = metric_keys_for(has_benign_class=benign_id is not None, scored=True)
    streaming_keys = metric_keys_for(
        has_benign_class=benign_id is not None,
        scored=False,
    )
    payload = {
        "schema_version": 2,
        "classification_metric_schema_version": CLASSIFICATION_METRIC_SCHEMA_VERSION,
        "classification_metric_keys": list(scored_keys),
        "confusion_derived_metric_keys": list(streaming_keys),
        "score_dependent_metric_keys": [
            key for key in scored_keys if key in SCORE_DEPENDENT_METRIC_KEYS
        ],
        "benign_class_id": benign_id,
        "benign_class_name": class_names[benign_id] if benign_id is not None else None,
        "binary_positive_definition": (
            BINARY_POSITIVE_DEFINITION if benign_id is not None else None
        ),
        "g_mean_definition": G_MEAN_DEFINITION,
        "partition_hash": partition_hash,
        "num_clients": int(num_clients),
        "num_classes": int(num_classes),
        "source_label_space": source_label_space,
        "target_class_names": list(class_names),
        "client_val_ratio": float(client_val_ratio),
        "split_seed": int(split_seed),
        "pooled_training_support": [int(value) for value in support],
        "minority_rule": {**MINORITY_RULE, "fraction": float(minority_fraction)},
        "minority_class_ids": list(minority_ids),
        "minority_class_names": [class_names[item] for item in minority_ids],
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def enforce_metric_contract(
    results_root: str | Path,
    contract: dict[str, Any],
) -> Path:
    """Create the contract once, then reject any scenario that disagrees."""
    protocol_dir = Path(results_root) / "_protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    path = protocol_dir / METRIC_CONTRACT_FILENAME
    serialized = json.dumps(contract, indent=2, sort_keys=True, ensure_ascii=False)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError(
                "Minority classes, pooled training support, or the metric "
                f"schema differ from the frozen contract at {path}. Runs under "
                "this results root are not comparable."
            ) from None
    return path
