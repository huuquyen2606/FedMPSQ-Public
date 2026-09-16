"""Immutable partition manifest creation and verification."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from data.dataset import client_partition_path
from data.integrity import sha256_array, sha256_file, sha256_indices
from data.splits import deterministic_split_indices


MANIFEST_SCHEMA_VERSION = 2
# Every campaign in this repository is frozen to one of these client counts.
# Keeping an explicit allowlist preserves the fail-closed behaviour of the
# original single-value contract while admitting the 100-client scale study.
SUPPORTED_CLIENT_COUNTS = (10, 100)


def require_supported_client_count(num_clients: int) -> int:
    """Fail closed on any client count outside the frozen campaign set."""
    count = int(num_clients)
    if count not in SUPPORTED_CLIENT_COUNTS:
        raise ValueError(
            "num_clients must be one of "
            f"{SUPPORTED_CLIENT_COUNTS}; got {num_clients}"
        )
    return count


def _npz_record(
    path: Path,
    *,
    client_id: int | None = None,
    val_ratio: float | None = None,
    split_seed: int | None = None,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Partition file not found: {path}")
    with np.load(path) as payload:
        x = payload["x"]
        y = payload["y"]
        record: dict[str, Any] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "x_sha256": sha256_array(x),
            "y_sha256": sha256_array(y),
            "num_examples": int(len(y)),
            "x_shape": list(x.shape),
            "y_shape": list(y.shape),
            "x_dtype": str(x.dtype),
            "y_dtype": str(y.dtype),
        }
    if client_id is not None:
        if val_ratio is None or split_seed is None:
            raise ValueError("Client records require val_ratio and split_seed")
        train_indices, val_indices = deterministic_split_indices(
            record["num_examples"],
            val_ratio,
            split_seed,
        )
        record.update(
            {
                "client_id": client_id,
                "train_examples": len(train_indices),
                "validation_examples": len(val_indices),
                "train_indices_sha256": sha256_indices(train_indices),
                "validation_indices_sha256": sha256_indices(val_indices),
            }
        )
    return record


def _global_test_record(root: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Build and cross-check the immutable evaluation-set contract."""
    record = _npz_record(root / "global_test.npz")
    record.update({"role": "evaluation_only", "immutable": True})
    provenance = metadata.get("global_test_provenance")
    if not isinstance(provenance, dict) or provenance.get("status") != "complete":
        raise ValueError(
            "metadata.json lacks complete global_test_provenance; regenerate the "
            "prepared partitions before freezing or running experiments"
        )
    if provenance.get("role") != "evaluation_only" or provenance.get("immutable") is not True:
        raise ValueError("Global-test provenance must declare immutable evaluation-only data")
    if provenance.get("preprocessing_fit_includes_global_test") is not False:
        raise ValueError("Global test must be excluded from preprocessing fit")
    if not provenance.get("source_kind"):
        raise ValueError("Global-test provenance must identify its source kind")
    expected_hashes = {
        "prepared_npz_sha256": record["sha256"],
        "prepared_x_sha256": record["x_sha256"],
        "prepared_y_sha256": record["y_sha256"],
    }
    for key, actual_digest in expected_hashes.items():
        recorded_digest = provenance.get(key)
        if recorded_digest != actual_digest:
            raise ValueError(
                f"metadata global-test provenance {key} differs from global_test.npz"
            )
    record["provenance"] = provenance
    return record


def _validate_preprocessing_contract(
    metadata: dict[str, Any],
    clients: list[dict[str, Any]],
    *,
    client_val_ratio: float,
    seed: int,
) -> str:
    """Fail closed unless preparation used the exact frozen local-train split."""
    preprocessing = metadata.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError(
            "metadata.json lacks preprocessing provenance; regenerate the prepared "
            "partitions to exclude validation/test leakage"
        )
    fit_scope = preprocessing.get("fit_scope")
    if fit_scope != "deterministic_local_training_subsets_only":
        raise ValueError("Preprocessing was not fit only on deterministic local-training rows")
    excluded = set(preprocessing.get("excluded_from_fit", []))
    if not {"client_validation", "global_test"}.issubset(excluded):
        raise ValueError("Preprocessing provenance does not exclude validation and global test")
    if float(preprocessing.get("client_val_ratio")) != float(client_val_ratio):
        raise ValueError("Preprocessing validation ratio differs from the frozen manifest")
    if int(preprocessing.get("seed")) != int(seed):
        raise ValueError("Preprocessing split seed differs from the frozen manifest")
    train_hashes = preprocessing.get("client_train_indices_sha256")
    validation_hashes = preprocessing.get("client_validation_indices_sha256")
    if not isinstance(train_hashes, list) or not isinstance(validation_hashes, list):
        raise ValueError("Preprocessing provenance lacks per-client split hashes")
    if len(train_hashes) != len(clients) or len(validation_hashes) != len(clients):
        raise ValueError("Preprocessing split-hash count differs from client count")
    for client_id, client in enumerate(clients):
        if train_hashes[client_id] != client["train_indices_sha256"]:
            raise ValueError(f"Preprocessing train indices differ for client {client_id}")
        if validation_hashes[client_id] != client["validation_indices_sha256"]:
            raise ValueError(f"Preprocessing validation indices differ for client {client_id}")
    return fit_scope


def _derived_artifact_records(
    root: Path,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Require and hash deterministic train-only distribution reports."""
    records: list[dict[str, Any]] = []
    statistics = metadata.get("client_statistics")
    if not isinstance(statistics, dict):
        raise ValueError(
            "metadata.json lacks mandatory train-only client-distribution artifacts"
        )
    if statistics.get("population_scope") != "deterministic_local_training_subsets":
        raise ValueError("Client statistics must describe exact local-training subsets")
    for file_key, digest_key, role in (
        ("matrix_file", "matrix_sha256", "client_training_class_count_matrix"),
        (
            "summary_file",
            "summary_sha256",
            "client_training_distribution_statistics",
        ),
    ):
        filename = statistics.get(file_key)
        if not filename:
            raise ValueError(f"Client statistics metadata lacks mandatory {file_key}")
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f"Derived partition artifact not found: {path}")
        digest = sha256_file(path)
        if statistics.get(digest_key) != digest:
            raise ValueError(f"metadata digest differs from derived artifact: {path}")
        if role == "client_training_class_count_matrix":
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != int(metadata["num_clients"]):
                raise ValueError("Train-only client matrix row count differs from clients")
            if any(
                row.get("population_scope")
                != "deterministic_local_training_subsets"
                for row in rows
            ):
                raise ValueError("Client matrix does not explicitly identify train-only scope")
        else:
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("population_scope") != "deterministic_local_training_subsets":
                raise ValueError("Client statistics JSON does not identify train-only scope")
            if int(summary.get("num_clients", -1)) != int(metadata["num_clients"]):
                raise ValueError("Client statistics JSON count differs from clients")
        records.append({"file": filename, "role": role, "sha256": digest})
    return records


def create_partition_manifest(
    partitions_dir: str | Path,
    partition_file: str | Path,
    *,
    dataset: str,
    num_clients: int,
    client_val_ratio: float,
    seed: int,
) -> dict[str, Any]:
    """Write one immutable manifest for the already-prepared partition files."""
    manifest = create_partition_manifest_payload(
        partitions_dir,
        dataset=dataset,
        num_clients=num_clients,
        client_val_ratio=client_val_ratio,
        seed=seed,
    )

    output_path = Path(partition_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise FileExistsError(
                f"Refusing to replace an existing partition manifest: {output_path}"
            )
        return existing
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def verify_partition_manifest(settings) -> str:
    """Fail if any dataset, client split, file, or sample assignment changed."""
    data = settings.data
    require_supported_client_count(data.num_clients)
    if data.dataset != "existing_dataset":
        raise ValueError("data.dataset must remain 'existing_dataset'")
    if data.regenerate_partition:
        raise ValueError("data.regenerate_partition must be false")

    manifest_path = Path(data.partition_file)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Frozen partition manifest not found: {manifest_path}. "
            "Run scripts/freeze_partition.py once against the existing prepared split."
        )
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported partition manifest schema; regenerate/freeze the data with "
            f"schema version {MANIFEST_SCHEMA_VERSION} before running experiments"
        )
    if expected.get("dataset") != data.dataset:
        raise ValueError("Config dataset differs from the frozen partition manifest")
    if expected.get("num_clients") != data.num_clients:
        raise ValueError("Config client count differs from the frozen partition manifest")
    policy = expected.get("partitioning", {})
    if policy.get("regenerate_partition") is not False:
        raise ValueError("Frozen manifest permits partition regeneration")
    if float(policy.get("client_val_ratio")) != float(data.client_val_ratio):
        raise ValueError("Validation split ratio differs from the frozen manifest")
    if int(policy.get("seed")) != int(data.seed):
        raise ValueError("Partition seed differs from the frozen manifest")

    root = Path(data.partitions_dir)
    actual = create_partition_manifest_payload(
        root,
        dataset=data.dataset,
        num_clients=data.num_clients,
        client_val_ratio=data.client_val_ratio,
        seed=data.seed,
    )
    if actual != expected:
        raise ValueError(
            "Partition integrity check failed: client IDs, samples, split indices, "
            "metadata, or global test data changed"
        )
    return str(expected["partition_hash"])


def create_partition_manifest_payload(
    partitions_dir: str | Path,
    *,
    dataset: str,
    num_clients: int,
    client_val_ratio: float,
    seed: int,
) -> dict[str, Any]:
    """Build a manifest payload without writing it."""
    require_supported_client_count(num_clients)
    root = Path(partitions_dir)
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Partition metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("num_clients", num_clients)) != num_clients:
        raise ValueError(
            "metadata.json num_clients does not match the frozen client contract"
        )
    clients = [
        _npz_record(
            client_partition_path(root, client_id),
            client_id=client_id,
            val_ratio=client_val_ratio,
            split_seed=seed + client_id,
        )
        for client_id in range(num_clients)
    ]
    preprocessing_fit_scope = _validate_preprocessing_contract(
        metadata,
        clients,
        client_val_ratio=client_val_ratio,
        seed=seed,
    )
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": dataset,
        "num_clients": num_clients,
        "partitioning": {
            "regenerate_partition": False,
            "client_val_ratio": client_val_ratio,
            "seed": seed,
            "validation_split": "torch.randperm deterministic ordered indices",
            "client_seed_rule": "seed + client_id",
        },
        "data_contract": {
            "global_test_role": "evaluation_only",
            "global_test_immutable": True,
            "preprocessing_fit_scope": preprocessing_fit_scope,
        },
        "metadata": {
            "file": metadata_path.name,
            "sha256": sha256_file(metadata_path),
            "content": metadata,
        },
        "clients": clients,
        "global_test": _global_test_record(root, metadata),
        "derived_artifacts": _derived_artifact_records(root, metadata),
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["partition_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return manifest
