#!/usr/bin/env python
"""Train one FedMPSQ experiment on a frozen client partition."""

from __future__ import annotations

import argparse
import dataclasses
import copy
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import (
    client_partition_path,
    load_metadata,
    split_client_dataset,
)
from data.fedmpsq_labels import (
    build_fedmpsq_task_contract,
    canonical_sha256,
    scientific_config_payload,
    enforce_fedmpsq_task_contract,
    load_grouped_npz_dataset,
    target_class_names,
)
from data.partition_manifest import verify_partition_manifest
from extensions.flat_uplink import unflatten_state, wire_layout
from extensions.fedmpsq import (
    CODEC_FLAGS,
    address_payload,
    aggregate_decoded_payloads,
    clone_tensor_state,
    decode_payload,
    require_finite_state,
    serialize_dense_state,
    serialize_sparse_state,
    zeros_like_floating_state,
)
from fl.config import seed_everything, select_client_devices
from fl.fedmpsq_config import (
    SUPPORTED_SOURCE_LABEL_SPACES,
    TARGET_NUM_CLASSES_BY_LABEL_SPACE,
    FedMPSQConfig,
    load_fedmpsq_config,
    write_config_snapshot,
)
from fl.metric_contract import build_metric_contract, enforce_metric_contract
from fl.metrics_logger import RoundMetricsLogger
from models import build_model
from training.fedmpsq import (
    FedMPSQBatchRequest,
    FedMPSQClientResult,
    FedMPSQClientSpec,
    initialize_fedmpsq_worker,
    train_fedmpsq_batch,
    train_fedmpsq_batch_local,
)
from training.fedmpsq_metrics import evaluate_model_detailed, scalar_metrics
from training.fedmpsq_metrics import REQUIRED_QUALITY_KEYS
from training.costs import benchmark_inference


CHECKPOINT_POLICY = {
    "eligible_rounds": "completed_training_rounds_1_to_R",
    "metric": "global_validation_macro_f1",
    "mode": "max",
    "tie_break": "earliest_round",
    "test_usage": "best_and_final_only_after_validation_selection",
}
COMPONENT_KEYS = (
    "header_bytes",
    "metadata_bytes",
    "layout_bytes",
    "index_bytes",
    "value_bytes",
    "scale_bytes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--partitions-dir")
    parser.add_argument("--partition-file")
    parser.add_argument("--results-dir")
    parser.add_argument("--run-name")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--local-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--parallel-clients", type=int)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--alpha-s", type=float)
    parser.add_argument("--sparsity", type=float)
    parser.add_argument(
        "--loss", choices=["ce", "bounded_cb", "fedlc", "bounded_cb_lc"]
    )
    parser.add_argument("--class-prior", choices=["local", "global"])
    parser.add_argument("--aggregation", choices=["sample", "uniform"])
    parser.add_argument(
        "--source-label-space",
        choices=list(SUPPORTED_SOURCE_LABEL_SPACES),
    )
    parser.add_argument("--num-clients", type=int)
    parser.add_argument("--target-macro-f1", type=float)
    parser.add_argument("--resume")
    parser.add_argument("--launcher-command")
    parser.add_argument(
        "--no-save-round-checkpoints",
        action="store_true",
        help="Save only the best and final full-state checkpoints.",
    )
    parser.add_argument(
        "--skip-test-evaluation",
        action="store_true",
        help=(
            "Train and select checkpoints on validation only; do not load or "
            "evaluate the global test set."
        ),
    )
    return parser.parse_args()


def settings_from_args(args: argparse.Namespace) -> FedMPSQConfig:
    overrides: dict[str, Any] = {}
    mapping = {
        "partitions_dir": "data.partitions_dir",
        "partition_file": "data.partition_file",
        "results_dir": "results.dir",
        "run_name": "results.run_name",
        "rounds": "algorithm.num_server_rounds",
        "local_epochs": "algorithm.local_epochs",
        "batch_size": "algorithm.batch_size",
        "device": "runtime.device",
        "parallel_clients": "runtime.parallel_clients",
        "seed": "runtime.seed",
        "alpha_s": "method.alpha_s",
        "sparsity": "method.sparsity",
        "loss": "method.loss",
        "source_label_space": "data.source_label_space",
        "target_macro_f1": "results.target_macro_f1",
        "resume": "results.resume",
    }
    for argument, path in mapping.items():
        value = getattr(args, argument)
        if value is not None:
            overrides[path] = value
    if args.num_clients is not None:
        overrides["algorithm.num_clients"] = args.num_clients
        overrides["data.num_clients"] = args.num_clients
    if args.source_label_space is not None:
        # The task width follows the declared label space so a config cannot
        # silently keep an eight-class head on the 34-class task.
        target_classes = TARGET_NUM_CLASSES_BY_LABEL_SPACE[args.source_label_space]
        overrides["data.target_num_classes"] = target_classes
        overrides["model.num_classes"] = target_classes
    if args.no_save_round_checkpoints:
        overrides["results.save_round_checkpoints"] = False
    return load_fedmpsq_config(args.config_path, overrides)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def write_records(path_base: Path, records: list[dict[str, Any]]) -> None:
    json_path = path_base.parent / f"{path_base.name}.json"
    csv_path = path_base.parent / f"{path_base.name}.csv"
    write_json(json_path, records)
    if not records:
        return
    keys: list[str] = []
    for record in records:
        for key, value in record.items():
            if isinstance(value, (dict, list, tuple)):
                continue
            if key not in keys:
                keys.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: record.get(key)
                    for key in keys
                }
            )


def cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in state.items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def scientific_config_sha256(settings: FedMPSQConfig) -> str:
    return canonical_sha256(scientific_config_payload(settings.config_dict))


def resolve_input_dim(settings: FedMPSQConfig) -> int:
    metadata = load_metadata(settings.data.partitions_dir)
    input_dim = metadata.get("input_dim", settings.model.input_dim)
    if input_dim is None:
        raise ValueError("input_dim is absent from metadata and config")
    declared_classes = metadata.get("num_classes")
    expected = 8 if settings.data.source_label_space == "ciciot2023_groups_8" else 34
    if declared_classes is not None and int(declared_classes) != expected:
        raise ValueError(
            "metadata.num_classes disagrees with data.source_label_space: "
            f"{declared_classes} versus {expected}"
        )
    return int(input_dim)


def manifest_record(settings: FedMPSQConfig, partition_hash: str) -> dict[str, Any]:
    path = Path(settings.data.partition_file).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("partition_hash") != partition_hash:
        raise ValueError("Verified partition hash differs from manifest JSON")
    return payload


def enforce_target_contract(
    results_root: Path,
    target_macro_f1: float | None,
) -> tuple[Path | None, dict[str, Any]]:
    """Freeze one validation target across every directly compared main run."""
    contract = {
        "schema_version": 1,
        "metric": "global_validation_macro_f1",
        "target_macro_f1": target_macro_f1,
        "round_rule": "first_completed_round_at_or_above_target",
        "traffic_outputs": [
            "uplink_bytes_to_target",
            "total_bytes_to_target",
        ],
        "test_set_used_to_choose_target": False,
    }
    if target_macro_f1 is None:
        return None, contract
    protocol_dir = results_root / "_protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    path = protocol_dir / "fedmpsq_target_contract.json"
    serialized = json.dumps(
        contract,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError(
                "Target Macro-F1 differs from the frozen comparison contract "
                f"at {path}"
            )
    return path, contract


def validation_loaders(settings: FedMPSQConfig) -> Iterable[DataLoader]:
    for client_id in range(settings.data.num_clients):
        dataset = load_grouped_npz_dataset(
            client_partition_path(settings.data.partitions_dir, client_id),
            source_label_space=settings.data.source_label_space,
        )
        _, validation = split_client_dataset(
            dataset,
            val_ratio=settings.data.client_val_ratio,
            seed=settings.data.split_seed + client_id,
        )
        yield DataLoader(
            validation,
            batch_size=settings.algorithm.batch_size,
            shuffle=False,
            num_workers=settings.runtime.num_workers,
        )


def global_test_loader(settings: FedMPSQConfig) -> DataLoader:
    dataset = load_grouped_npz_dataset(
        Path(settings.data.partitions_dir) / "global_test.npz",
        source_label_space=settings.data.source_label_space,
    )
    return DataLoader(
        dataset,
        batch_size=settings.algorithm.batch_size,
        shuffle=False,
        num_workers=settings.runtime.num_workers,
    )


def select_clients(settings: FedMPSQConfig, server_round: int) -> list[int]:
    count = max(
        1,
        int(math.ceil(settings.algorithm.fraction_train * settings.data.num_clients)),
    )
    if count >= settings.data.num_clients:
        return list(range(settings.data.num_clients))
    rng = np.random.default_rng(settings.runtime.seed + server_round)
    return sorted(
        int(value)
        for value in rng.choice(settings.data.num_clients, size=count, replace=False)
    )


def create_executors(devices: list[torch.device]) -> list[ProcessPoolExecutor]:
    if len(devices) <= 1:
        return []
    context = mp.get_context("spawn")
    return [
        ProcessPoolExecutor(
            max_workers=1,
            mp_context=context,
            initializer=initialize_fedmpsq_worker,
            initargs=(str(device),),
        )
        for device in devices
    ]


def train_clients(
    *,
    settings: FedMPSQConfig,
    server_round: int,
    global_state: dict[str, torch.Tensor],
    input_dim: int,
    client_ids: list[int],
    saliency_states: dict[int, dict[str, torch.Tensor]],
    residual_states: dict[int, dict[str, torch.Tensor]],
    device: torch.device,
    executors: list[ProcessPoolExecutor],
    pooled_counts: tuple[int, ...] | None = None,
) -> list[FedMPSQClientResult]:
    specs = [
        FedMPSQClientSpec(
            client_id=client_id,
            saliency_state=saliency_states[client_id],
            residual_state=residual_states[client_id],
        )
        for client_id in client_ids
    ]
    if not executors:
        return train_fedmpsq_batch_local(
            FedMPSQBatchRequest(
                server_round=server_round,
                global_state=global_state,
                settings=settings,
                input_dim=input_dim,
                clients=tuple(specs),
                pooled_counts=pooled_counts,
            ),
            device,
        )
    buckets: list[list[FedMPSQClientSpec]] = [[] for _ in executors]
    for index, spec in enumerate(specs):
        buckets[index % len(buckets)].append(spec)
    futures = []
    for executor, bucket in zip(executors, buckets, strict=True):
        if bucket:
            futures.append(
                executor.submit(
                    train_fedmpsq_batch,
                    FedMPSQBatchRequest(
                        server_round=server_round,
                        global_state=global_state,
                        settings=settings,
                        input_dim=input_dim,
                        clients=tuple(bucket),
                        pooled_counts=pooled_counts,
                    ),
                )
            )
    by_client = {
        result.client_id: result
        for future in futures
        for result in future.result()
    }
    return [by_client[client_id] for client_id in client_ids]


def weighted_mean(results: list[FedMPSQClientResult], key: str) -> float:
    denominator = sum(result.num_examples for result in results)
    return sum(
        result.metrics[key] * result.num_examples
        for result in results
    ) / max(denominator, 1)


def distribution_metrics(
    results: list[FedMPSQClientResult],
    key: str,
) -> dict[str, float]:
    values = np.asarray([result.metrics[key] for result in results], dtype=np.float64)
    return {
        f"{key}_sample_weighted_mean": weighted_mean(results, key),
        f"{key}_median": float(np.median(values)),
        f"{key}_p95": float(np.percentile(values, 95)),
        f"{key}_max": float(np.max(values)),
    }


def summarize_system_cost(
    client_records: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    summary = {}
    for key in (
        "train_seconds",
        "saliency_seconds",
        "compression_seconds",
        "server_decode_seconds",
        "client_downlink_decode_seconds",
        "peak_memory_bytes",
    ):
        values = np.asarray(
            [float(record[key]) for record in client_records],
            dtype=np.float64,
        )
        summary[key] = {
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }
    return summary


def empty_traffic() -> dict[str, int]:
    payload = {
        "uplink_bytes": 0,
        "downlink_bytes": 0,
        "total_bytes": 0,
        "theoretical_uplink_bytes": 0,
        "theoretical_ideal_index_uplink_bytes": 0,
        "dense_reference_serialized_bytes": 0,
        "dense_reference_raw_bytes": 0,
    }
    for direction in ("uplink", "downlink"):
        for component in COMPONENT_KEYS:
            payload[f"{direction}_{component}"] = 0
    return payload


def add_traffic(target: dict[str, int], incoming: dict[str, int]) -> None:
    for key, value in incoming.items():
        target[key] = int(target.get(key, 0)) + int(value)


def validate_traffic(traffic: dict[str, int]) -> None:
    expected_keys = set(empty_traffic())
    if set(traffic) != expected_keys:
        missing = sorted(expected_keys - set(traffic))
        extra = sorted(set(traffic) - expected_keys)
        raise ValueError(
            f"Traffic counter schema mismatch; missing={missing}, extra={extra}"
        )
    for key, value in traffic.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"Traffic counter {key} must be an integer")
        if int(value) < 0:
            raise ValueError(f"Traffic counter {key} must be non-negative")
    for direction in ("uplink", "downlink"):
        component_sum = sum(
            traffic[f"{direction}_{component}"]
            for component in COMPONENT_KEYS
        )
        if component_sum != traffic[f"{direction}_bytes"]:
            raise RuntimeError(f"{direction} byte components do not sum to total")
    if traffic["uplink_bytes"] + traffic["downlink_bytes"] != traffic["total_bytes"]:
        raise RuntimeError("Uplink plus downlink does not equal total traffic")


def checkpoint_payload(
    *,
    settings: FedMPSQConfig,
    server_round: int,
    global_state: dict[str, torch.Tensor],
    saliency_states: dict[int, dict[str, torch.Tensor]],
    residual_states: dict[int, dict[str, torch.Tensor]],
    cumulative_traffic: dict[str, int],
    client_cumulative_traffic: dict[int, dict[str, int]],
    best_round: int | None,
    best_macro_f1: float | None,
    best_model_state: dict[str, torch.Tensor] | None,
    best_validation: dict[str, Any] | None,
    round_records: list[dict[str, Any]],
    client_records: list[dict[str, Any]],
    partition_hash: str,
    task_contract_hash: str,
    target_contract_hash: str,
    config_hash: str,
    target_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_version": settings.protocol_version,
        "round": server_round,
        "global_state": clone_tensor_state(global_state),
        "global_state_sha256": state_sha256(global_state),
        "saliency_ema_by_client": {
            client_id: clone_tensor_state(value)
            for client_id, value in saliency_states.items()
        },
        "error_feedback_residual_by_client": {
            client_id: clone_tensor_state(value)
            for client_id, value in residual_states.items()
        },
        "cumulative_traffic": dict(cumulative_traffic),
        "client_cumulative_traffic": copy.deepcopy(client_cumulative_traffic),
        "best_round": best_round,
        "best_validation_macro_f1": best_macro_f1,
        "best_model_state": (
            clone_tensor_state(best_model_state)
            if best_model_state is not None
            else None
        ),
        "best_validation": copy.deepcopy(best_validation),
        "round_records": copy.deepcopy(round_records),
        "client_records": copy.deepcopy(client_records),
        "partition_hash": partition_hash,
        "task_contract_sha256": task_contract_hash,
        "target_contract_sha256": target_contract_hash,
        "scientific_config_sha256": config_hash,
        "training_seed": settings.runtime.seed,
        "data_split_seed": settings.data.split_seed,
        "target_state": copy.deepcopy(target_state),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }


def _validate_tensor_state(
    candidate: Any,
    reference: dict[str, torch.Tensor],
    *,
    label: str,
) -> None:
    if not isinstance(candidate, dict) or candidate.keys() != reference.keys():
        raise ValueError(f"{label} tensor layout does not match the current model")
    for name, expected in reference.items():
        tensor = candidate[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{label} entry {name!r} is not a tensor")
        if tuple(tensor.shape) != tuple(expected.shape):
            raise ValueError(f"{label} tensor shape differs for {name}")
        if tensor.dtype != expected.dtype:
            raise ValueError(f"{label} tensor dtype differs for {name}")
    require_finite_state(candidate, label=label)


def validate_resume_checkpoint(
    checkpoint: Any,
    *,
    settings: FedMPSQConfig,
    reference_global_state: dict[str, torch.Tensor],
    partition_hash: str,
    task_contract_hash: str,
    target_contract_hash: str,
    config_hash: str,
) -> None:
    """Fail closed unless a checkpoint can reproduce the next exact round."""
    if not isinstance(checkpoint, dict):
        raise TypeError("Resume checkpoint must be a mapping")
    required = {
        "schema_version",
        "protocol_version",
        "round",
        "global_state",
        "global_state_sha256",
        "saliency_ema_by_client",
        "error_feedback_residual_by_client",
        "cumulative_traffic",
        "client_cumulative_traffic",
        "best_round",
        "best_validation_macro_f1",
        "best_model_state",
        "best_validation",
        "round_records",
        "client_records",
        "partition_hash",
        "task_contract_sha256",
        "target_contract_sha256",
        "scientific_config_sha256",
        "training_seed",
        "data_split_seed",
        "target_state",
        "rng_state",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Resume checkpoint misses required fields: {missing}")
    expected_scalars = {
        "schema_version": 1,
        "protocol_version": settings.protocol_version,
        "partition_hash": partition_hash,
        "task_contract_sha256": task_contract_hash,
        "target_contract_sha256": target_contract_hash,
        "scientific_config_sha256": config_hash,
        "training_seed": settings.runtime.seed,
        "data_split_seed": settings.data.split_seed,
    }
    for key, value in expected_scalars.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Resume checkpoint {key} does not match this run")

    server_round = checkpoint["round"]
    if (
        isinstance(server_round, bool)
        or not isinstance(server_round, int)
        or not 1 <= server_round <= settings.algorithm.num_server_rounds
    ):
        raise ValueError("Resume checkpoint round is outside the configured budget")
    _validate_tensor_state(
        checkpoint["global_state"],
        reference_global_state,
        label="Checkpoint global state",
    )
    if state_sha256(checkpoint["global_state"]) != checkpoint["global_state_sha256"]:
        raise ValueError("Resume checkpoint global-state hash is invalid")

    floating_reference = zeros_like_floating_state(reference_global_state)
    expected_clients = set(range(settings.data.num_clients))
    for field, label in (
        ("saliency_ema_by_client", "Checkpoint saliency"),
        ("error_feedback_residual_by_client", "Checkpoint residual"),
    ):
        states = checkpoint[field]
        if not isinstance(states, dict) or set(states) != expected_clients:
            raise ValueError(f"{label} does not contain every configured client")
        for client_id in sorted(expected_clients):
            _validate_tensor_state(
                states[client_id],
                floating_reference,
                label=f"{label} client {client_id}",
            )

    validate_traffic(checkpoint["cumulative_traffic"])
    client_traffic = checkpoint["client_cumulative_traffic"]
    if not isinstance(client_traffic, dict) or set(client_traffic) != expected_clients:
        raise ValueError("Checkpoint traffic does not contain every client")
    summed_traffic = empty_traffic()
    for client_id in sorted(expected_clients):
        validate_traffic(client_traffic[client_id])
        add_traffic(summed_traffic, client_traffic[client_id])
    if summed_traffic != checkpoint["cumulative_traffic"]:
        raise ValueError("Checkpoint client traffic does not sum to global traffic")

    best_round = checkpoint["best_round"]
    best_score = checkpoint["best_validation_macro_f1"]
    if (
        isinstance(best_round, bool)
        or not isinstance(best_round, int)
        or not 1 <= best_round <= server_round
        or not isinstance(best_score, (int, float))
        or not math.isfinite(float(best_score))
    ):
        raise ValueError("Checkpoint best-round state is invalid")
    _validate_tensor_state(
        checkpoint["best_model_state"],
        reference_global_state,
        label="Checkpoint best model",
    )
    if not isinstance(checkpoint["best_validation"], dict):
        raise TypeError("Checkpoint best validation metrics are missing")

    round_records = checkpoint["round_records"]
    if not isinstance(round_records, list):
        raise TypeError("Checkpoint round records must be a list")
    observed_rounds = [
        record.get("round")
        for record in round_records
        if isinstance(record, dict)
    ]
    if observed_rounds != list(range(server_round + 1)):
        raise ValueError("Checkpoint round-record sequence is incomplete or duplicated")
    client_records = checkpoint["client_records"]
    if not isinstance(client_records, list) or any(
        not isinstance(record, dict)
        or not 1 <= int(record.get("round", -1)) <= server_round
        or int(record.get("client_id", -1)) not in expected_clients
        for record in client_records
    ):
        raise ValueError("Checkpoint client-record sequence is invalid")

    rng = checkpoint["rng_state"]
    rng_keys = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not isinstance(rng, dict) or set(rng) != rng_keys:
        raise ValueError("Checkpoint RNG state is incomplete")
    if not isinstance(rng["torch_cpu"], torch.Tensor):
        raise TypeError("Checkpoint CPU RNG state is invalid")
    if not isinstance(rng["torch_cuda"], list):
        raise TypeError("Checkpoint CUDA RNG state is invalid")
    if not isinstance(checkpoint["target_state"], dict):
        raise TypeError("Checkpoint target state is invalid")


def restore_rng(payload: dict[str, Any]) -> None:
    rng = payload["rng_state"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.random.set_rng_state(rng["torch_cpu"])
    if torch.cuda.is_available() and rng["torch_cuda"]:
        if len(rng["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA RNG device count differs")
        torch.cuda.set_rng_state_all(rng["torch_cuda"])


def save_full_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def evaluate_and_save(
    model: torch.nn.Module,
    loaders: DataLoader | Iterable[DataLoader],
    device: torch.device,
    *,
    class_names: tuple[str, ...],
    minority_ids: list[int],
    benign_id: int | None,
    path: Path,
) -> dict[str, Any]:
    metrics = evaluate_model_detailed(
        model,
        loaders,
        device,
        class_names=class_names,
        minority_class_ids=minority_ids,
        benign_id=benign_id,
    )
    write_json(path, metrics)
    return metrics


def run_experiment(
    settings: FedMPSQConfig,
    *,
    launcher_command: str | None,
    skip_test_evaluation: bool = False,
) -> None:
    results_dir = Path(settings.results.dir).expanduser().resolve()
    resume_path = Path(settings.results.resume).expanduser().resolve() if settings.results.resume else None
    if resume_path is None and results_dir.exists() and any(results_dir.iterdir()):
        raise FileExistsError(
            f"FedMPSQ results already exist: {results_dir}. "
            "Use a new directory or provide --resume."
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    status_path = results_dir / f"{settings.results.run_name}_status.json"
    write_json(
        status_path,
        {
            "status": "running",
            "started_at_utc": utc_now(),
            "run_name": settings.results.run_name,
        },
    )

    partition_hash = verify_partition_manifest(settings)
    manifest = manifest_record(settings, partition_hash)
    input_dim = resolve_input_dim(settings)
    if settings.runtime.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
    seed_everything(settings.runtime.seed)
    client_devices = select_client_devices(
        settings.runtime.device,
        settings.runtime.parallel_clients,
    )
    device = client_devices[0]
    print(
        "FEDMPSQ_RUNTIME "
        f"device={device} clients={settings.data.num_clients} "
        f"parallel_clients={len(client_devices)}",
        flush=True,
    )

    task_contract = build_fedmpsq_task_contract(
        settings,
        partition_hash=partition_hash,
    )
    task_contract_path = enforce_fedmpsq_task_contract(
        results_dir.parent,
        task_contract,
    )
    task_contract_hash = canonical_sha256(task_contract)
    target_contract_path, target_contract = enforce_target_contract(
        results_dir.parent,
        settings.results.target_macro_f1,
    )
    target_contract_hash = canonical_sha256(target_contract)
    config_hash = scientific_config_sha256(settings)
    minority_ids = [int(value) for value in task_contract["minority_class_ids"]]
    class_names = target_class_names(settings.data.source_label_space)
    if list(class_names) != list(task_contract["target_class_names"]):
        raise ValueError("Task contract class names differ from the label space")
    # The shared evaluation contract is written by both trainers so that a
    # baseline run and a proposed run under one results root cannot disagree
    # about which classes are minority or which ten metrics are reported.
    metric_contract = build_metric_contract(
        partitions_dir=settings.data.partitions_dir,
        num_clients=settings.data.num_clients,
        num_classes=settings.data.target_num_classes,
        source_label_space=settings.data.source_label_space,
        client_val_ratio=settings.data.client_val_ratio,
        split_seed=settings.data.split_seed,
        minority_fraction=settings.data.minority_fraction,
        partition_hash=partition_hash,
    )
    # Already frozen in the contract both trainers write, so the global class
    # prior adds no communication and no information the protocol did not fix.
    pooled_counts = tuple(int(v) for v in metric_contract["pooled_training_support"])
    if list(metric_contract["minority_class_ids"]) != list(minority_ids):
        raise ValueError(
            "Shared metric contract and FedMPSQ task contract disagree about "
            "the frozen minority classes"
        )
    benign_id = metric_contract["benign_class_id"]
    metric_contract_path = enforce_metric_contract(
        results_dir.parent,
        metric_contract,
    )
    print(
        "METRIC_CONTRACT "
        f"schema_version={metric_contract['classification_metric_schema_version']} "
        f"num_classes={settings.data.target_num_classes} "
        f"metrics={len(metric_contract['classification_metric_keys'])} "
        f"benign_class={metric_contract['benign_class_name']} "
        f"minority_classes={metric_contract['minority_class_names']}",
        flush=True,
    )

    protocol_logger = RoundMetricsLogger(results_dir, settings.results.run_name)
    write_config_snapshot(
        settings,
        results_dir / f"{settings.results.run_name}_config.json",
    )
    fixed_test_path = protocol_logger.enforce_fixed_test_contract(
        manifest["global_test"]
    )
    protocol_logger.save_provenance(
        settings,
        project_root=PROJECT_ROOT,
        partition_record={
            "partition_hash": partition_hash,
            "partitioning": manifest.get("partitioning", {}),
            "clients": manifest.get("clients", []),
            "metadata": manifest.get("metadata", {}),
            "global_test": manifest["global_test"],
            "task_contract_path": str(task_contract_path),
            "task_contract_sha256": task_contract_hash,
            "target_contract_path": (
                str(target_contract_path)
                if target_contract_path is not None
                else None
            ),
            "target_contract": target_contract,
            "target_contract_sha256": target_contract_hash,
            "fixed_test_contract_path": str(fixed_test_path),
            "metric_contract_path": str(metric_contract_path),
            "metric_contract_sha256": metric_contract["sha256"],
            "classification_metric_schema_version": metric_contract[
                "classification_metric_schema_version"
            ],
            "classification_metric_keys": metric_contract[
                "classification_metric_keys"
            ],
            "minority_class_ids": metric_contract["minority_class_ids"],
            "minority_class_names": metric_contract["minority_class_names"],
            "evaluation_mode": (
                "validation_only"
                if skip_test_evaluation
                else "post_selection_global_test"
            ),
        },
        launcher_command=launcher_command,
    )

    model = build_model(
        settings.model,
        input_dim=input_dim,
        num_classes=settings.data.target_num_classes,
    ).to(device)
    initial_state = cpu_state(model)
    saliency_states = {
        client_id: zeros_like_floating_state(initial_state)
        for client_id in range(settings.data.num_clients)
    }
    residual_states = {
        client_id: zeros_like_floating_state(initial_state)
        for client_id in range(settings.data.num_clients)
    }
    cumulative_traffic = empty_traffic()
    client_cumulative_traffic = {
        client_id: empty_traffic()
        for client_id in range(settings.data.num_clients)
    }
    round_records: list[dict[str, Any]] = []
    client_records: list[dict[str, Any]] = []
    best_round: int | None = None
    best_macro_f1: float | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    best_validation: dict[str, Any] | None = None
    target_state = {
        "target_macro_f1": settings.results.target_macro_f1,
        "status": (
            "not_configured"
            if settings.results.target_macro_f1 is None
            else "not_reached_yet"
        ),
        "reached": False,
        "round_to_target": None,
        "uplink_bytes_to_target": None,
        "total_bytes_to_target": None,
    }
    start_round = 1

    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_resume_checkpoint(
            checkpoint,
            settings=settings,
            reference_global_state=cpu_state(model),
            partition_hash=partition_hash,
            task_contract_hash=task_contract_hash,
            target_contract_hash=target_contract_hash,
            config_hash=config_hash,
        )
        model.load_state_dict(checkpoint["global_state"], strict=True)
        saliency_states = checkpoint["saliency_ema_by_client"]
        residual_states = checkpoint["error_feedback_residual_by_client"]
        cumulative_traffic = checkpoint["cumulative_traffic"]
        client_cumulative_traffic = checkpoint["client_cumulative_traffic"]
        best_round = checkpoint["best_round"]
        best_macro_f1 = checkpoint["best_validation_macro_f1"]
        best_model_state = checkpoint["best_model_state"]
        best_validation = checkpoint["best_validation"]
        round_records = checkpoint.get("round_records", [])
        client_records = checkpoint.get("client_records", [])
        target_state = checkpoint.get("target_state", target_state)
        start_round = int(checkpoint["round"]) + 1
        restore_rng(checkpoint)
        print(f"FEDMPSQ_RESUME round={checkpoint['round']}", flush=True)
    else:
        initial_validation = evaluate_and_save(
            model,
            validation_loaders(settings),
            device,
            class_names=class_names,
            minority_ids=minority_ids,
            benign_id=benign_id,
            path=results_dir / "validation" / "round_000.json",
        )
        initial_record = {
            "timestamp": utc_now(),
            "round": 0,
            **{
                f"validation_{key}": value
                for key, value in scalar_metrics(initial_validation).items()
            },
            **{
                f"communication_round_{key}": float(value)
                for key, value in empty_traffic().items()
            },
            **{
                f"communication_cumulative_{key}": float(value)
                for key, value in empty_traffic().items()
            },
            "checkpoint_eligible": 0.0,
        }
        round_records.append(initial_record)
        write_records(
            results_dir / f"{settings.results.run_name}_round_metrics",
            round_records,
        )

    executors = create_executors(client_devices)
    try:
        for server_round in range(start_round, settings.algorithm.num_server_rounds + 1):
            round_started = time.perf_counter()
            print(
                f"FEDMPSQ_ROUND {server_round}/{settings.algorithm.num_server_rounds}",
                flush=True,
            )
            global_state = cpu_state(model)
            client_ids = select_clients(settings, server_round)
            round_traffic = empty_traffic()
            downlinks = {}
            downlink_started = time.perf_counter()
            for client_id in client_ids:
                if downlinks:
                    downlinks[client_id] = address_payload(downlinks[client_ids[0]], client_id)
                    continue
                downlink_options = {}
                serializer = serialize_dense_state
                if settings.method.downlink_bits != 32:
                    serializer = serialize_sparse_state
                    downlink_options = dict(
                        quantized=True, declared_sparsity=0.0, alpha_s=0.0,
                        quant_bits=settings.method.downlink_bits,
                        group_size=settings.method.quant_group_size,
                        clipping=settings.method.quant_clipping,
                        index_codec=settings.method.index_codec,
                        compact_layout=settings.method.compact_layout,
                        raw_names=frozenset(name for name, value in global_state.items()
                                            if not torch.is_floating_point(value)),
                    )
                downlinks[client_id] = serializer(
                    global_state,
                    client_id=client_id,
                    server_round=server_round,
                    num_examples=0,
                    flags=2,
                    **downlink_options,
                )
            downlink_seconds = time.perf_counter() - downlink_started
            downlink_decode_seconds = {}
            decoded_downlinks = {}
            downlink_decode_started = time.perf_counter()
            for client_id in client_ids:
                client_decode_started = time.perf_counter()
                decoded_downlink = decode_payload(downlinks[client_id].data)
                if (
                    decoded_downlink.client_id != client_id
                    or decoded_downlink.server_round != server_round
                    or decoded_downlink.num_examples != 0
                    or decoded_downlink.flags != 2
                ):
                    raise RuntimeError("Decoded downlink metadata differs from request")
                decoded_downlinks[client_id] = decoded_downlink.state
                downlink_decode_seconds[client_id] = (
                    time.perf_counter() - client_decode_started
                )
            downlink_decode_total_seconds = (
                time.perf_counter() - downlink_decode_started
            )
            transmitted_global_state = decoded_downlinks[client_ids[0]]
            if settings.method.downlink_bits == 32 and state_sha256(transmitted_global_state) != state_sha256(global_state):
                raise RuntimeError("Decoded downlink differs from global model state")
            if any(state_sha256(state) != state_sha256(transmitted_global_state)
                   for state in decoded_downlinks.values()):
                raise RuntimeError("All clients must receive the same decoded base")
            client_results = train_clients(
                settings=settings,
                server_round=server_round,
                global_state=transmitted_global_state,
                input_dim=input_dim,
                client_ids=client_ids,
                saliency_states=saliency_states,
                residual_states=residual_states,
                device=device,
                executors=executors,
                pooled_counts=pooled_counts,
            )
            for result in client_results:
                saliency_states[result.client_id] = result.saliency_state
                residual_states[result.client_id] = result.residual_state

            decoded_payloads = []
            per_client_decode_seconds = {}
            decode_started = time.perf_counter()
            flat_uplink = bool(settings.method.flat_uplink)
            server_wire_layout = wire_layout(transmitted_global_state) if flat_uplink else None
            for result in client_results:
                client_decode_started = time.perf_counter()
                decoded = decode_payload(result.payload)
                if (
                    decoded.client_id != result.client_id
                    or decoded.server_round != server_round
                    or decoded.num_examples != result.num_examples
                    or decoded.flags & ~CODEC_FLAGS != 1
                ):
                    raise RuntimeError("Decoded uplink metadata differs from client result")
                if flat_uplink:
                    # Restore the named tensors the aggregator expects. The
                    # layout came from the model the server already holds, so
                    # this reconstruction costs no bytes on the wire.
                    decoded = dataclasses.replace(
                        decoded, state=unflatten_state(decoded.state, server_wire_layout))
                decoded_payloads.append(decoded)
                per_client_decode_seconds[result.client_id] = (
                    time.perf_counter() - client_decode_started
                )
            decode_seconds = time.perf_counter() - decode_started
            aggregation_started = time.perf_counter()
            if settings.algorithm.aggregation == "uniform":
                # Equal client weights are a separate objective ablation.
                aggregation_weights = [1.0] * len(decoded_payloads)
            else:
                aggregation_weights = None
            aggregated_state = aggregate_decoded_payloads(
                transmitted_global_state,
                decoded_payloads,
                aggregation_weights,
            )
            require_finite_state(aggregated_state, label="Aggregated global state")
            aggregation_seconds = time.perf_counter() - aggregation_started
            if [item.client_id for item in decoded_payloads] != client_ids:
                raise RuntimeError("Decoded client order differs from selected client order")
            model.load_state_dict(aggregated_state, strict=True)

            for result, decoded in zip(
                client_results,
                decoded_payloads,
                strict=True,
            ):
                downlink = downlinks[result.client_id]
                uplink_components = {
                    key: int(result.metrics[f"uplink_{key}"])
                    for key in COMPONENT_KEYS
                }
                downlink_components = {
                    key: int(downlink.byte_breakdown.as_dict()[key])
                    for key in COMPONENT_KEYS
                }
                client_traffic = {
                    "uplink_bytes": len(result.payload),
                    "downlink_bytes": downlink.payload_bytes,
                    "total_bytes": len(result.payload) + downlink.payload_bytes,
                    "theoretical_uplink_bytes": int(
                        result.metrics["theoretical_uplink_bytes"]
                    ),
                    "theoretical_ideal_index_uplink_bytes": int(
                        result.metrics["theoretical_ideal_index_uplink_bytes"]
                    ),
                    "dense_reference_serialized_bytes": int(
                        result.metrics["dense_reference_serialized_bytes"]
                    ),
                    "dense_reference_raw_bytes": int(
                        result.metrics["dense_reference_raw_bytes"]
                    ),
                    **{
                        f"uplink_{key}": value
                        for key, value in uplink_components.items()
                    },
                    **{
                        f"downlink_{key}": value
                        for key, value in downlink_components.items()
                    },
                }
                validate_traffic(client_traffic)
                add_traffic(round_traffic, client_traffic)
                add_traffic(
                    client_cumulative_traffic[result.client_id],
                    client_traffic,
                )
                record = {
                    "timestamp": utc_now(),
                    "round": server_round,
                    "client_id": result.client_id,
                    "num_examples": result.num_examples,
                    "partition_hash": partition_hash,
                    "task_contract_sha256": task_contract_hash,
                    "class_counts": list(result.class_counts),
                    "peak_memory_kind": result.peak_memory_kind,
                    "server_decode_seconds": per_client_decode_seconds[
                        result.client_id
                    ],
                    "client_downlink_decode_seconds": downlink_decode_seconds[
                        result.client_id
                    ],
                    **result.metrics,
                    **{
                        f"communication_{key}": float(value)
                        for key, value in client_traffic.items()
                    },
                    **{
                        f"communication_cumulative_{key}": float(value)
                        for key, value in client_cumulative_traffic[
                            result.client_id
                        ].items()
                    },
                }
                client_records.append(record)
                print(
                    "FEDMPSQ_CLIENT "
                    f"round={server_round} id={result.client_id} "
                    f"examples={result.num_examples} "
                    f"uplink_bytes={len(result.payload)}",
                    flush=True,
                )
            validate_traffic(round_traffic)
            add_traffic(cumulative_traffic, round_traffic)
            validate_traffic(cumulative_traffic)

            validation = evaluate_and_save(
                model,
                validation_loaders(settings),
                device,
                class_names=class_names,
                minority_ids=minority_ids,
                benign_id=benign_id,
                path=results_dir / "validation" / f"round_{server_round:03d}.json",
            )
            macro_f1 = float(validation["macro_f1"])
            is_best = (
                best_macro_f1 is None
                or macro_f1 > best_macro_f1
            )
            if is_best:
                best_round = server_round
                best_macro_f1 = macro_f1
                best_model_state = cpu_state(model)
                best_validation = copy.deepcopy(validation)

            if (
                settings.results.target_macro_f1 is not None
                and not target_state["reached"]
                and macro_f1 >= settings.results.target_macro_f1
            ):
                target_state = {
                    "target_macro_f1": settings.results.target_macro_f1,
                    "status": "reached",
                    "reached": True,
                    "round_to_target": server_round,
                    "uplink_bytes_to_target": cumulative_traffic["uplink_bytes"],
                    "total_bytes_to_target": cumulative_traffic["total_bytes"],
                }

            round_wall_seconds = time.perf_counter() - round_started
            record = {
                "timestamp": utc_now(),
                "round": server_round,
                **{
                    f"validation_{key}": value
                    for key, value in scalar_metrics(validation).items()
                },
                **{
                    f"communication_round_{key}": float(value)
                    for key, value in round_traffic.items()
                },
                **{
                    f"communication_cumulative_{key}": float(value)
                    for key, value in cumulative_traffic.items()
                },
                "selected_clients": float(len(client_ids)),
                "round_wall_seconds": float(round_wall_seconds),
                "downlink_serialization_seconds": float(downlink_seconds),
                "downlink_decode_seconds": float(downlink_decode_total_seconds),
                "server_decode_seconds": float(decode_seconds),
                "server_aggregation_seconds": float(aggregation_seconds),
                "checkpoint_eligible": 1.0,
                "is_new_best": float(is_best),
                "best_round_so_far": float(best_round),
                "target_configured": float(settings.results.target_macro_f1 is not None),
                "target_reached": float(target_state["reached"]),
                "round_to_target": float(
                    target_state["round_to_target"]
                    if target_state["round_to_target"] is not None
                    else -1
                ),
            }
            for key in (
                "train_seconds",
                "saliency_seconds",
                "compression_seconds",
                "peak_memory_bytes",
                "update_sparsity",
                "payload_sparsity",
                "topk_selected_values",
                "transmitted_values",
                "sparsification_l2_error",
                "sparsification_relative_error",
                "sparsification_relative_l2_error",
                "quantization_l2_error",
                "quantization_relative_error",
                "quantization_relative_l2_error",
                "residual_l2_norm",
            ):
                record.update(distribution_metrics(client_results, key))
            round_records.append(record)
            write_records(
                results_dir / f"{settings.results.run_name}_round_metrics",
                round_records,
            )
            write_records(
                results_dir / f"{settings.results.run_name}_client_metrics",
                client_records,
            )

            full = checkpoint_payload(
                settings=settings,
                server_round=server_round,
                global_state=cpu_state(model),
                saliency_states=saliency_states,
                residual_states=residual_states,
                cumulative_traffic=cumulative_traffic,
                client_cumulative_traffic=client_cumulative_traffic,
                best_round=best_round,
                best_macro_f1=best_macro_f1,
                best_model_state=best_model_state,
                best_validation=best_validation,
                round_records=round_records,
                client_records=client_records,
                partition_hash=partition_hash,
                task_contract_hash=task_contract_hash,
                target_contract_hash=target_contract_hash,
                config_hash=config_hash,
                target_state=target_state,
            )
            checkpoint_dir = results_dir / "checkpoints"
            if settings.results.save_round_checkpoints:
                save_full_checkpoint(
                    checkpoint_dir
                    / f"{settings.results.run_name}_round_{server_round:03d}.pt",
                    full,
                )
            if is_best:
                save_full_checkpoint(
                    checkpoint_dir / f"{settings.results.run_name}_best_full.pt",
                    full,
                )
    finally:
        for executor in executors:
            executor.shutdown(wait=True)

    if settings.results.target_macro_f1 is not None and not target_state["reached"]:
        target_state = {
            **target_state,
            "status": "not_reached_censored",
            "censored_at_round": settings.algorithm.num_server_rounds,
            "censored_uplink_bytes": cumulative_traffic["uplink_bytes"],
            "censored_total_bytes": cumulative_traffic["total_bytes"],
        }

    if best_model_state is None or best_round is None or best_validation is None:
        raise RuntimeError("No completed round was eligible for checkpoint selection")
    final_model_state = cpu_state(model)
    final_validation = json.loads(
        (
            results_dir
            / "validation"
            / f"round_{settings.algorithm.num_server_rounds:03d}.json"
        ).read_text(encoding="utf-8")
    )
    best_test: dict[str, Any] | None = None
    final_test: dict[str, Any] | None = None
    if not skip_test_evaluation:
        final_test = evaluate_and_save(
            model,
            global_test_loader(settings),
            device,
            class_names=class_names,
            minority_ids=minority_ids,
            benign_id=benign_id,
            path=results_dir / "test" / "final.json",
        )
        model.load_state_dict(best_model_state, strict=True)
        best_test = evaluate_and_save(
            model,
            global_test_loader(settings),
            device,
            class_names=class_names,
            minority_ids=minority_ids,
            benign_id=benign_id,
            path=results_dir / "test" / "best.json",
        )
        model.load_state_dict(final_model_state, strict=True)

    torch.save(
        best_model_state,
        results_dir / f"{settings.results.run_name}_best_model.pt",
    )
    torch.save(
        final_model_state,
        results_dir / f"{settings.results.run_name}_final_model.pt",
    )
    final_full = checkpoint_payload(
        settings=settings,
        server_round=settings.algorithm.num_server_rounds,
        global_state=final_model_state,
        saliency_states=saliency_states,
        residual_states=residual_states,
        cumulative_traffic=cumulative_traffic,
        client_cumulative_traffic=client_cumulative_traffic,
        best_round=best_round,
        best_macro_f1=best_macro_f1,
        best_model_state=best_model_state,
        best_validation=best_validation,
        round_records=round_records,
        client_records=client_records,
        partition_hash=partition_hash,
        task_contract_hash=task_contract_hash,
        target_contract_hash=target_contract_hash,
        config_hash=config_hash,
        target_state=target_state,
    )
    save_full_checkpoint(
        results_dir
        / "checkpoints"
        / f"{settings.results.run_name}_final_full.pt",
        final_full,
    )
    checkpoint_policy = dict(CHECKPOINT_POLICY)
    if skip_test_evaluation:
        checkpoint_policy["test_usage"] = "not_evaluated_validation_only"
    else:
        test_evaluations = {
            "selection_policy": checkpoint_policy,
            "global_test_sha256": manifest["global_test"]["sha256"],
            "best": {
                "round": best_round,
                "validation": best_validation,
                "test": best_test,
            },
            "final": {
                "round": settings.algorithm.num_server_rounds,
                "validation": final_validation,
                "test": final_test,
            },
        }
        write_json(
            results_dir / f"{settings.results.run_name}_test_evaluations.json",
            test_evaluations,
        )
    model.load_state_dict(final_model_state, strict=True)
    timing_features = next(iter(next(iter(validation_loaders(settings)))))[0]
    inference_cost = {
        "checkpoint": "final",
        "single_example": benchmark_inference(model, timing_features[:1], device),
        "batch": benchmark_inference(model, timing_features, device),
    }
    summary = {
        "status": "completed",
        "experiment_id": settings.experiment_id,
        "run_name": settings.results.run_name,
        "ablation": settings.method.ablation,
        "loss": settings.method.loss,
        "proximal_mu": settings.algorithm.proximal_mu,
        "alpha_s": settings.method.alpha_s,
        "sparsity": settings.method.sparsity,
        "quant_bits": settings.method.quant_bits,
        "index_codec": settings.method.index_codec,
        "stochastic_rounding": settings.method.stochastic_rounding,
        "error_feedback": settings.method.error_feedback,
        # Which support Eq. (3) was scored on, and whether the FedLC offset was
        # applied, decide what the objective actually was; a summary that names
        # only the loss cannot distinguish the arms of the quality study.
        "class_prior": settings.method.class_prior,
        "fedlc_tau": settings.method.fedlc_tau,
        "model_norm": settings.model.norm,
        "model_name": settings.model.name,
        "fidelity_classification": settings.fidelity_classification,
        "protocol_version": settings.protocol_version,
        "training_budget": {
            "num_server_rounds": settings.algorithm.num_server_rounds,
            "num_clients": settings.algorithm.num_clients,
            "fraction_train": settings.algorithm.fraction_train,
            "local_epochs": settings.algorithm.local_epochs,
            "batch_size": settings.algorithm.batch_size,
            "optimizer": settings.algorithm.optimizer,
            "learning_rate": settings.algorithm.learning_rate,
            "momentum": settings.algorithm.momentum,
            "weight_decay": settings.algorithm.weight_decay,
        },
        "training_seed": settings.runtime.seed,
        "data_split_seed": settings.data.split_seed,
        "partition_hash": partition_hash,
        "num_clients": settings.data.num_clients,
        "num_classes": settings.data.target_num_classes,
        "source_label_space": settings.data.source_label_space,
        "task_contract_sha256": task_contract_hash,
        "classification_metric_schema_version": metric_contract[
            "classification_metric_schema_version"
        ],
        "classification_metric_keys": metric_contract["classification_metric_keys"],
        "metric_contract_sha256": metric_contract["sha256"],
        "minority_class_ids": metric_contract["minority_class_ids"],
        "minority_class_names": metric_contract["minority_class_names"],
        "target_contract": target_contract,
        "target_contract_sha256": target_contract_hash,
        "scientific_config_sha256": config_hash,
        "best_round": best_round,
        "best_validation_macro_f1": best_macro_f1,
        "best_validation_metrics": best_validation,
        "final_validation_metrics": final_validation,
        "evaluation_mode": (
            "validation_only"
            if skip_test_evaluation
            else "post_selection_global_test"
        ),
        "test_evaluation_performed": not skip_test_evaluation,
        "system_cost": summarize_system_cost(client_records),
        "required_quality_metric_keys": list(REQUIRED_QUALITY_KEYS),
        "quantization_scope": "communication_only_fp32_training_and_inference",
        "lowbit_options": {
            "block_size": settings.method.block_size,
            "minority_head_reserve": settings.method.minority_head_reserve,
            "index_codec": settings.method.index_codec,
            "compact_layout": settings.method.compact_layout,
            "group_size": settings.method.quant_group_size,
            "clipping": settings.method.quant_clipping,
            "protect_small_tensors": settings.method.protect_small_tensors,
            "adaptive_quant_error": settings.method.adaptive_quant_error,
            "downlink_bits": settings.method.downlink_bits,
        },
        "loss_options": {
            "loss": settings.method.loss,
            "logit_calibration_enabled": settings.method.loss in {"fedlc", "bounded_cb_lc"},
            "fedlc_tau": settings.method.fedlc_tau,
            "class_prior": settings.method.class_prior,
        },
        "total_local_train_seconds": sum(float(row["train_seconds"]) for row in client_records),
        "total_round_wall_seconds": sum(float(row.get("round_wall_seconds", 0)) for row in round_records),
        "resident_model_bytes": sum(v.numel() * v.element_size() for v in final_model_state.values()),
        "inference_latency": inference_cost,
        "cumulative_traffic": cumulative_traffic,
        "client_cumulative_traffic": client_cumulative_traffic,
        "target": target_state,
        "checkpoint_selection": checkpoint_policy,
    }
    if not skip_test_evaluation:
        if best_test is None or final_test is None:
            raise AssertionError("Global-test metrics were not produced")
        summary.update(
            {
                "best_test_scalar_metrics": scalar_metrics(best_test),
                "final_test_scalar_metrics": scalar_metrics(final_test),
                "best_test_metrics": best_test,
                "final_test_metrics": final_test,
            }
        )
    write_json(
        results_dir / f"{settings.results.run_name}_summary.json",
        summary,
    )
    write_json(
        status_path,
        {
            "status": "completed",
            "completed_at_utc": utc_now(),
            "run_name": settings.results.run_name,
            "best_round": best_round,
        },
    )


def main() -> None:
    args = parse_args()
    if args.torch_threads is not None:
        if args.torch_threads < 1:
            raise ValueError("torch-threads must be positive")
        torch.set_num_threads(args.torch_threads)
    settings: FedMPSQConfig | None = None
    results_dir = (
        Path(args.results_dir).expanduser().resolve()
        if args.results_dir is not None
        else None
    )
    run_name = args.run_name or "fedmpsq_configuration_failure"
    try:
        settings = settings_from_args(args)
        results_dir = Path(settings.results.dir).expanduser().resolve()
        run_name = settings.results.run_name
        run_experiment(
            settings,
            launcher_command=args.launcher_command,
            skip_test_evaluation=args.skip_test_evaluation,
        )
    except BaseException as error:
        if results_dir is not None:
            results_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                results_dir / f"{run_name}_status.json",
                {
                    "status": "failed",
                    "failed_at_utc": utc_now(),
                    "run_name": run_name,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        raise


if __name__ == "__main__":
    main()
