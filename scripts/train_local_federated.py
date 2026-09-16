#!/usr/bin/env python
"""Run the audited federated experiment loop with optional multi-GPU clients."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import multiprocessing as mp
import sys
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.ciciot2023 import CICIOT2023_LABELS
from data.dataset import (
    client_partition_path,
    load_global_test_loader,
    load_task_npz_dataset,
    split_client_dataset,
)
from data.fedmpsq_labels import (
    IDENTITY_34_LABEL_SPACE,
    generic_identity_label_space,
)
from data.partition_manifest import verify_partition_manifest
from extensions.quantization import (
    DAdaQuantController,
    QuantizedStateUpdate,
    quantize_state_update,
    serialize_dense_state,
)
from fl.config import (
    load_settings,
    resolve_dimensions,
    seed_everything,
    select_client_devices,
)
from fl.metric_contract import build_metric_contract, enforce_metric_contract
from fl.metrics_logger import BEST_CHECKPOINT_POLICY, RoundMetricsLogger
from models import build_model
from training.fedmpsq_metrics import (
    evaluate_model_detailed as evaluate_model_detailed_with_scores,
)
from training.metrics import (
    BINARY_METRIC_KEYS,
    CLASSIFICATION_METRIC_KEYS,
    SCORE_DEPENDENT_METRIC_KEYS,
    classification_metrics_from_confusion,
    classification_metrics_v4_from_confusion,
    confusion_matrix_counts,
)
from training.client_worker import (
    ClientBatchRequest,
    ClientTrainResult,
    ClientTrainSpec,
    initialize_client_worker,
    train_client_batch,
    train_client_batch_local,
)

RESUME_CHECKPOINT_SCHEMA_VERSION = 1
PAYLOAD_COMPONENT_NAMES = (
    "header_bytes",
    "norm_bytes",
    "layout_bytes",
    "value_bytes",
    "index_bytes",
    "auxiliary_bytes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", default=None, help="YAML or TOML config")
    parser.add_argument("--partitions-dir", default=None, help="Prepared NPZ directory")
    parser.add_argument("--partition-file", default=None, help="Frozen manifest path")
    parser.add_argument("--algorithm", choices=["fedavg", "fedprox"], default=None)
    parser.add_argument("--rounds", type=int, default=None, help="Global training rounds")
    parser.add_argument("--fraction-train", type=float, default=None)
    parser.add_argument("--min-train-nodes", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default=None)
    parser.add_argument("--proximal-mu", type=float, default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--parallel-clients", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument(
        "--model-name",
        choices=["dcnn_bilstm"],
        default=None,
        help="DCNN-BiLSTM backbone used by all reported arms",
    )
    parser.add_argument(
        "--launcher-command",
        default=None,
        help="Exact outer runner command to preserve in provenance",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Full-state checkpoint produced after a completed federated round",
    )
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--save-round-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


def _settings_from_args(args: argparse.Namespace):
    run_config: dict[str, Any] = {}
    if args.partitions_dir is not None:
        run_config["data.partitions-dir"] = args.partitions_dir
    if args.partition_file is not None:
        run_config["data.partition-file"] = args.partition_file
    legacy_defaults: dict[str, Any] = {
        "algorithm.name": "fedavg",
        "algorithm.num-server-rounds": 50,
        "algorithm.fraction-train": 1.0,
        "algorithm.min-train-nodes": 10,
        "algorithm.local-epochs": 2,
        "algorithm.batch-size": 256,
        "algorithm.learning-rate": 0.001,
        "algorithm.optimizer": "sgd",
        "algorithm.proximal-mu": 0.0,
        "results.dir": "results/fedavg",
        "results.run-name": "fedavg-baseline",
        "results.save-model": True,
        "results.save-round-checkpoints": True,
        "runtime.device": "auto",
        "runtime.parallel-clients": 1,
        "runtime.seed": 42,
        "data.seed": 42,
    }
    optional_overrides: dict[str, Any] = {
        "algorithm.name": args.algorithm,
        "algorithm.num-server-rounds": args.rounds,
        "algorithm.fraction-train": args.fraction_train,
        "algorithm.min-train-nodes": args.min_train_nodes,
        "algorithm.local-epochs": args.local_epochs,
        "algorithm.batch-size": args.batch_size,
        "algorithm.learning-rate": args.learning_rate,
        "algorithm.optimizer": args.optimizer,
        "algorithm.proximal-mu": args.proximal_mu,
        "results.dir": args.results_dir,
        "results.run-name": args.run_name,
        "results.save-model": args.save_model,
        "results.save-round-checkpoints": args.save_round_checkpoints,
        "runtime.device": args.device,
        "runtime.parallel-clients": args.parallel_clients,
        "runtime.seed": args.seed,
        # The data split seed is deliberately independent of the training RNG seed.
        # Config profiles freeze it at 42 for fair 42/43/44 repetitions.
        "data.seed": None,
    }
    # ``getattr`` keeps callers that build a narrower Namespace working; the
    # protocol tests construct one directly.
    num_clients = getattr(args, "num_clients", None)
    if num_clients is not None:
        run_config["data.num-clients"] = num_clients
        run_config["algorithm.min-train-nodes"] = num_clients
        run_config["algorithm.min-evaluate-nodes"] = num_clients
        run_config["algorithm.min-available-nodes"] = num_clients
    num_classes = getattr(args, "num_classes", None)
    if num_classes is not None:
        run_config["data.num-classes"] = num_classes
        run_config["model.num-classes"] = num_classes
    model_name = getattr(args, "model_name", None)
    if model_name is not None:
        run_config["model.name"] = model_name
    for key, value in optional_overrides.items():
        if value is not None:
            run_config[key] = value
        elif args.config_path is None:
            run_config[key] = legacy_defaults[key]
    if args.config_path:
        run_config["config-path"] = args.config_path
    return load_settings(run_config)


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _clone_state_dict(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return an independent CPU copy of a model state dictionary.

    Client private states are updated in-place in the round loop.  A plain
    ``dict.copy`` would therefore make a purported best checkpoint alias the
    live (final-round) tensors.  Keeping this helper explicit makes the
    checkpoint boundary auditable and works for both CPU and accelerator
    tensors.
    """
    return {
        key: value.detach().cpu().clone()
        for key, value in state.items()
    }


def _clone_private_states(
    private_states: dict[int, dict[str, torch.Tensor]],
) -> dict[int, dict[str, torch.Tensor]]:
    """Deep-copy all persistent BDD-HFL private client states."""
    return {
        int(client_id): _clone_state_dict(state)
        for client_id, state in private_states.items()
    }

#: Model fields introduced together with the tabular backbone, after the resume
#: contract had already been frozen by running arms. A checkpoint written before
#: they existed hashed a payload that did not contain them, so it disagrees with
#: the current hash even though nothing about the model changed. When all three
#: hold the values the old code implied, the network built now is byte-identical
#: to the one the checkpoint stores and the run must still be resumable; any
#: other value is a real architecture change and the strict hash refuses it.
LEGACY_CONTRACT_MODEL_DEFAULTS = {
    "hidden_size": 320,
    "num_blocks": 1,
    "norm": "batch",
}


def _resume_contract_payload(settings) -> dict[str, Any]:
    """Scientific settings only, with output and runtime relocation allowed."""
    payload = asdict(settings)
    payload.pop("results", None)
    payload["runtime"] = {"seed": settings.runtime.seed}
    payload["data"].pop("partitions_dir", None)
    payload["data"].pop("partition_file", None)
    return payload


def _hash_contract_payload(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _resume_contract_sha256(settings) -> str:
    """Hash scientific settings while allowing output/runtime relocation."""
    return _hash_contract_payload(_resume_contract_payload(settings))


def _legacy_resume_contract_sha256(settings) -> str | None:
    """Hash as the contract did before the tabular backbone was added.

    Returns ``None`` when this run's model is not the one that contract could
    have described, so a genuine architecture change can never be waved through
    as a legacy checkpoint.
    """
    payload = _resume_contract_payload(settings)
    model = payload.get("model")
    if not isinstance(model, dict) or model.get("name") != "dcnn_bilstm":
        return None
    for field, default in LEGACY_CONTRACT_MODEL_DEFAULTS.items():
        if field not in model or model[field] != default:
            return None
        model.pop(field)
    return _hash_contract_payload(payload)

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
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            raise ValueError(f"{label} contains non-finite values in {name}")

def _quant_controller_checkpoint_state(
    controller: DAdaQuantController | None,
) -> dict[str, Any] | None:
    if controller is None:
        return None
    return {
        "base_level": int(controller.base_level),
        "moving_losses": [float(value) for value in controller.moving_losses],
        "last_increase_index": int(controller.last_increase_index),
    }

def _restore_quant_controller(
    controller: DAdaQuantController | None,
    state: Any,
) -> None:
    if controller is None:
        if state is not None:
            raise ValueError("Checkpoint has DAdaQuant state for a non-DAdaQuant run")
        return
    if not isinstance(state, dict) or set(state) != {
        "base_level",
        "moving_losses",
        "last_increase_index",
    }:
        raise ValueError("Checkpoint DAdaQuant controller state is invalid")
    base_level = state["base_level"]
    moving_losses = state["moving_losses"]
    last_increase_index = state["last_increase_index"]
    if (
        isinstance(base_level, bool)
        or not isinstance(base_level, int)
        or not controller.config.min_level <= base_level <= controller.config.max_level
    ):
        raise ValueError("Checkpoint DAdaQuant base level is invalid")
    if not isinstance(moving_losses, list) or any(
        not isinstance(value, (int, float)) or not math.isfinite(float(value))
        for value in moving_losses
    ):
        raise ValueError("Checkpoint DAdaQuant moving losses are invalid")
    if (
        isinstance(last_increase_index, bool)
        or not isinstance(last_increase_index, int)
        or not 0 <= last_increase_index <= len(moving_losses)
    ):
        raise ValueError("Checkpoint DAdaQuant increase index is invalid")
    controller.base_level = base_level
    controller.moving_losses = [float(value) for value in moving_losses]
    controller.last_increase_index = last_increase_index

def _resume_checkpoint_payload(
    *,
    settings,
    server_round: int,
    partition_hash: str,
    global_state: dict[str, torch.Tensor],
    private_states: dict[int, dict[str, torch.Tensor]],
    fap_original_num_examples: dict[int, int],
    fap_active_indices: dict[int, list[int]],
    quant_controller: DAdaQuantController | None,
    cumulative_upload_bytes: int,
    cumulative_payload_components: dict[str, int],
    best_round: int,
    best_validation_macro_f1: float,
    best_validation_metrics: dict[str, float],
    best_state: dict[str, torch.Tensor],
    best_private_states: dict[int, dict[str, torch.Tensor]] | None,
    final_round_metrics: dict[str, float],
    round_records: list[dict[str, Any]],
    client_history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": RESUME_CHECKPOINT_SCHEMA_VERSION,
        "round": int(server_round),
        "experiment_id": settings.experiment_id,
        "run_name": settings.results.run_name,
        "resume_contract_sha256": _resume_contract_sha256(settings),
        "partition_hash": partition_hash,
        "training_seed": int(settings.runtime.seed),
        "data_split_seed": int(settings.data.seed),
        "global_state": _clone_state_dict(global_state),
        "private_states": _clone_private_states(private_states),
        "fap_original_num_examples": {
            int(client_id): int(value)
            for client_id, value in fap_original_num_examples.items()
        },
        "fap_active_indices": {
            int(client_id): [int(index) for index in indices]
            for client_id, indices in fap_active_indices.items()
        },
        "quant_controller": _quant_controller_checkpoint_state(quant_controller),
        "cumulative_upload_bytes": int(cumulative_upload_bytes),
        "cumulative_payload_components": {
            name: int(value)
            for name, value in cumulative_payload_components.items()
        },
        "best_round": int(best_round),
        "best_validation_macro_f1": float(best_validation_macro_f1),
        "best_validation_metrics": copy.deepcopy(best_validation_metrics),
        "best_state": _clone_state_dict(best_state),
        "best_private_states": (
            _clone_private_states(best_private_states)
            if best_private_states is not None
            else None
        ),
        "final_round_metrics": copy.deepcopy(final_round_metrics),
        "round_records": copy.deepcopy(round_records),
        "client_history": copy.deepcopy(client_history),
    }

def _save_resume_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

def _validate_resume_checkpoint(
    checkpoint: Any,
    *,
    settings,
    partition_hash: str,
    reference_global_state: dict[str, torch.Tensor],
    reference_private_states: dict[int, dict[str, torch.Tensor]],
    reference_fap_original_num_examples: dict[int, int],
    reference_fap_active_indices: dict[int, list[int]],
    quant_controller: DAdaQuantController | None,
) -> None:
    """Fail closed unless the checkpoint can reproduce the next exact round."""
    if not isinstance(checkpoint, dict):
        raise TypeError("Resume checkpoint must be a mapping")
    required = {
        "schema_version",
        "round",
        "experiment_id",
        "run_name",
        "resume_contract_sha256",
        "partition_hash",
        "training_seed",
        "data_split_seed",
        "global_state",
        "private_states",
        "fap_original_num_examples",
        "fap_active_indices",
        "quant_controller",
        "cumulative_upload_bytes",
        "cumulative_payload_components",
        "best_round",
        "best_validation_macro_f1",
        "best_validation_metrics",
        "best_state",
        "best_private_states",
        "final_round_metrics",
        "round_records",
        "client_history",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Resume checkpoint misses required fields: {missing}")
    expected_scalars = {
        "schema_version": RESUME_CHECKPOINT_SCHEMA_VERSION,
        "experiment_id": settings.experiment_id,
        "run_name": settings.results.run_name,
        "resume_contract_sha256": _resume_contract_sha256(settings),
        "partition_hash": partition_hash,
        "training_seed": settings.runtime.seed,
        "data_split_seed": settings.data.seed,
    }
    # The contract hash is compared separately so a checkpoint predating the
    # tabular backbone stays resumable. Everything the hash protects that the
    # legacy form omitted is still checked: the three model fields must hold
    # their pre-existing defaults for the legacy hash to be offered at all, and
    # the global-state layout below is verified against the model built now.
    contract = expected_scalars.pop("resume_contract_sha256")
    for key, expected in expected_scalars.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"Resume checkpoint {key} does not match this run")
    if checkpoint.get("resume_contract_sha256") != contract:
        legacy = _legacy_resume_contract_sha256(settings)
        if legacy is None or checkpoint.get("resume_contract_sha256") != legacy:
            raise ValueError(
                "Resume checkpoint resume_contract_sha256 does not match this run"
            )
        print(
            "RESUME_CONTRACT legacy=accepted "
            f"checkpoint={checkpoint['resume_contract_sha256']} current={contract} "
            "reason=model_fields_added_after_checkpoint_at_default_values",
            flush=True,
        )

    completed_round = checkpoint["round"]
    if (
        isinstance(completed_round, bool)
        or not isinstance(completed_round, int)
        or not 1 <= completed_round <= settings.algorithm.num_server_rounds
    ):
        raise ValueError("Resume checkpoint round is outside the configured budget")
    _validate_tensor_state(
        checkpoint["global_state"],
        reference_global_state,
        label="Checkpoint global state",
    )

    expected_clients = set(range(settings.data.num_clients))
    private_states = checkpoint["private_states"]
    if settings.technique.distillation.enabled:
        if not isinstance(private_states, dict) or set(private_states) != expected_clients:
            raise ValueError("Checkpoint private state does not contain every client")
        reference_private = next(iter(reference_private_states.values()))
        for client_id in sorted(expected_clients):
            _validate_tensor_state(
                private_states[client_id],
                reference_private,
                label=f"Checkpoint private state client {client_id}",
            )
    elif private_states != {}:
        raise ValueError("Checkpoint private state is unexpected for this run")

    original_counts = checkpoint["fap_original_num_examples"]
    active_indices = checkpoint["fap_active_indices"]
    if settings.technique.pruning.enabled:
        if original_counts != reference_fap_original_num_examples:
            raise ValueError("Checkpoint FAP original sample counts do not match")
        if not isinstance(active_indices, dict) or set(active_indices) != expected_clients:
            raise ValueError("Checkpoint FAP state does not contain every client")
        for client_id in sorted(expected_clients):
            indices = active_indices[client_id]
            reference_indices = reference_fap_active_indices[client_id]
            active_set = set(indices) if isinstance(indices, list) else set()
            if (
                not isinstance(indices, list)
                or not indices
                or any(
                    isinstance(index, bool) or not isinstance(index, int)
                    for index in indices
                )
                or len(indices) != len(active_set)
                or indices
                != [index for index in reference_indices if index in active_set]
            ):
                raise ValueError(
                    f"Checkpoint FAP active indices are invalid for client {client_id}"
                )
    elif original_counts != {} or active_indices != {}:
        raise ValueError("Checkpoint FAP state is unexpected for this run")

    _restore_quant_controller(quant_controller, checkpoint["quant_controller"])

    cumulative_upload_bytes = checkpoint["cumulative_upload_bytes"]
    components = checkpoint["cumulative_payload_components"]
    if (
        isinstance(cumulative_upload_bytes, bool)
        or not isinstance(cumulative_upload_bytes, int)
        or cumulative_upload_bytes < 0
        or not isinstance(components, dict)
        or set(components) != set(PAYLOAD_COMPONENT_NAMES)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in components.values()
        )
        or sum(components.values()) != cumulative_upload_bytes
    ):
        raise ValueError("Checkpoint communication counters are invalid")

    best_round = checkpoint["best_round"]
    best_score = checkpoint["best_validation_macro_f1"]
    best_metrics = checkpoint["best_validation_metrics"]
    final_metrics = checkpoint["final_round_metrics"]
    if (
        isinstance(best_round, bool)
        or not isinstance(best_round, int)
        or not 1 <= best_round <= completed_round
        or not isinstance(best_score, (int, float))
        or not math.isfinite(float(best_score))
        or not isinstance(best_metrics, dict)
        or best_metrics.get("validation_macro_f1") != best_score
        or not isinstance(final_metrics, dict)
        or final_metrics.get("communication_cumulative_bytes")
        != float(cumulative_upload_bytes)
    ):
        raise ValueError("Checkpoint best/final metric state is invalid")
    _validate_tensor_state(
        checkpoint["best_state"],
        reference_global_state,
        label="Checkpoint best state",
    )
    best_private_states = checkpoint["best_private_states"]
    if settings.technique.distillation.enabled:
        if (
            not isinstance(best_private_states, dict)
            or set(best_private_states) != expected_clients
        ):
            raise ValueError("Checkpoint best private state does not contain every client")
        reference_private = next(iter(reference_private_states.values()))
        for client_id in sorted(expected_clients):
            _validate_tensor_state(
                best_private_states[client_id],
                reference_private,
                label=f"Checkpoint best private state client {client_id}",
            )
    elif best_private_states is not None:
        raise ValueError("Checkpoint best private state is unexpected for this run")

    round_records = checkpoint["round_records"]
    if not isinstance(round_records, list) or [
        record.get("round") if isinstance(record, dict) else None
        for record in round_records
    ] != list(range(completed_round + 1)):
        raise ValueError("Checkpoint round metric sequence is incomplete or duplicated")
    client_history = checkpoint["client_history"]
    if not isinstance(client_history, list) or any(
        not isinstance(record, dict)
        or not 1 <= int(record.get("round", -1)) <= completed_round
        or int(record.get("client_id", -1)) not in expected_clients
        for record in client_history
    ):
        raise ValueError("Checkpoint client history is invalid")
    for server_round in range(1, completed_round + 1):
        expected_ids = _select_train_clients(settings, server_round)
        observed_ids = [
            record["client_id"]
            for record in client_history
            if record["round"] == server_round
        ]
        if observed_ids != expected_ids:
            raise ValueError(
                f"Checkpoint client history is incomplete for round {server_round}"
            )


def _state_size_bytes(state: dict[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in state.values())


def _weighted_average_state_dicts(
    client_states: list[tuple[dict[str, torch.Tensor], int]],
) -> dict[str, torch.Tensor]:
    """FedAvg Equation 1: sample-count weighted model aggregation."""
    if not client_states:
        raise ValueError("No client model states to aggregate")
    total_examples = sum(num_examples for _, num_examples in client_states)
    if total_examples <= 0:
        raise ValueError("Cannot aggregate client states with zero examples")
    first_state = client_states[0][0]
    averaged: dict[str, torch.Tensor] = {}
    for key, first_tensor in first_state.items():
        if torch.is_floating_point(first_tensor):
            accumulator = torch.zeros_like(first_tensor, dtype=torch.float64)
            for state, num_examples in client_states:
                accumulator += state[key].to(dtype=torch.float64) * float(num_examples)
            averaged[key] = (accumulator / float(total_examples)).to(first_tensor.dtype)
        else:
            averaged[key] = first_tensor.clone()
    return averaged


def _aggregate_quantized_updates(
    global_state: dict[str, torch.Tensor],
    updates: list[tuple[QuantizedStateUpdate, int]],
) -> dict[str, torch.Tensor]:
    """Decode then aggregate quantized client updates with FedAvg weights."""
    if not updates:
        raise ValueError("No quantized client updates to aggregate")
    total_examples = sum(num_examples for _, num_examples in updates)
    if total_examples <= 0:
        raise ValueError("Cannot aggregate quantized updates with zero examples")
    raw_names = {spec.name for spec in updates[0][0].raw_tensor_layout}
    if any(
        {spec.name for spec in update.raw_tensor_layout} != raw_names
        for update, _ in updates[1:]
    ):
        raise ValueError("Quantized client payloads use different raw tensor layouts")
    aggregated: dict[str, torch.Tensor] = {}
    for name, global_tensor in global_state.items():
        if torch.is_floating_point(global_tensor) and name not in raw_names:
            accumulator = torch.zeros_like(global_tensor, dtype=torch.float64)
            for update, num_examples in updates:
                accumulator += update.state[name].to(torch.float64) * float(num_examples)
            mean_delta = accumulator / float(total_examples)
            aggregated[name] = (global_tensor.to(torch.float64) + mean_delta).to(
                global_tensor.dtype
            )
        elif torch.is_floating_point(global_tensor):
            accumulator = torch.zeros_like(global_tensor, dtype=torch.float64)
            for update, num_examples in updates:
                accumulator += update.state[name].to(torch.float64) * float(num_examples)
            aggregated[name] = (accumulator / float(total_examples)).to(
                global_tensor.dtype
            )
        else:
            aggregated[name] = updates[0][0].state[name].clone()
    return aggregated


def _global_evaluate(
    model: torch.nn.Module,
    settings,
    device: torch.device,
    num_classes: int,
    *,
    class_names: tuple[str, ...],
    minority_class_ids: list[int],
    benign_id: int | None,
) -> tuple[dict[str, float], dict[str, object]]:
    """Score the frozen global test under both metric schemas in one pass.

    Only this post-selection evaluation retains predicted probabilities, which
    is what ``macro_pr_auc`` needs; the per-round validation pass stays on
    streaming confusion counts.
    """
    test_loader = load_global_test_loader(
        settings.data.partitions_dir,
        batch_size=settings.algorithm.batch_size,
        data_config=settings.data,
    )
    detailed = evaluate_model_detailed_with_scores(
        model,
        test_loader,
        device,
        class_names=class_names,
        minority_class_ids=minority_class_ids,
        benign_id=benign_id,
    )
    confusion = np.asarray(detailed["confusion_matrix"], dtype=np.int64)
    if confusion.shape != (num_classes, num_classes):
        raise ValueError("Global-test confusion matrix has an unexpected shape")
    legacy_metrics = classification_metrics_from_confusion(confusion)
    metrics = {
        **legacy_metrics,
        **{
            key: float(detailed[key])
            for key in ("loss", *CLASSIFICATION_METRIC_KEYS)
            if key in detailed
        },
    }
    details = {
        "confusion_matrix": detailed["confusion_matrix"],
        "per_class": detailed["per_class"],
        "num_examples": int(detailed["num_examples"]),
        "macro_pr_auc_supported_classes": int(
            detailed["macro_pr_auc_supported_classes"]
        ),
        "macro_roc_auc_supported_classes": int(
            detailed["macro_roc_auc_supported_classes"]
        ),
        "macro_pr_auc_absent_class_policy": detailed[
            "macro_pr_auc_absent_class_policy"
        ],
        "benign_class_id": detailed["benign_class_id"],
    }
    return metrics, details


def _private_validation_evaluate(
    private_states: dict[int, dict[str, torch.Tensor]],
    settings,
    device: torch.device,
    *,
    input_dim: int,
    num_classes: int,
    minority_class_ids: list[int],
    benign_id: int | None,
) -> dict[str, Any]:
    """Evaluate each persistent BDD-HFL private model on its own holdout.

    The holdout is the same deterministic client-validation split used for
    global-union checkpoint selection.  Each client's metrics are retained,
    followed by an arithmetic client mean, client standard deviation, and a
    sample-count-weighted mean.  The latter weights *per-client metrics* by
    the number of held-out examples; it is deliberately kept separate from
    the global-model metrics and from checkpoint selection.

    ``private_states`` must represent one coherent training round.  Callers
    therefore pass an immutable best-round snapshot or an immutable
    final-round snapshot, never the live mapping that continues to change
    during training.
    """
    expected_client_ids = set(range(settings.data.num_clients))
    # ``torch.load`` normally preserves integer keys, but normalizing here
    # also makes archived checkpoints serialized through JSON-compatible
    # tooling unambiguous.
    normalized_states = {
        int(client_id): state for client_id, state in private_states.items()
    }
    actual_client_ids = set(normalized_states)
    missing = sorted(expected_client_ids - actual_client_ids)
    if missing:
        raise ValueError(
            "Private-state checkpoint is missing clients: "
            f"{missing}"
        )
    unexpected = sorted(actual_client_ids - expected_client_ids)
    if unexpected:
        raise ValueError(
            "Private-state checkpoint contains unknown clients: "
            f"{unexpected}"
        )

    model = build_model(
        settings.model,
        input_dim=input_dim,
        num_classes=num_classes,
    )
    model.to(device)
    # BDD-HFL's contribution is the personalised private model, so it has to be
    # scored with the same schema as everything else; reporting it on a smaller
    # metric set than the global model would understate the method.
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    metric_names = (
        *[
            key
            for key in CLASSIFICATION_METRIC_KEYS
            if key not in SCORE_DEPENDENT_METRIC_KEYS
            and (benign_id is not None or key not in BINARY_METRIC_KEYS)
        ],
        "loss",
    )
    records: list[dict[str, Any]] = []
    try:
        for client_id in sorted(expected_client_ids):
            # States are persisted on CPU so that this evaluation remains
            # independent of the worker process/device used for training.
            model.load_state_dict(normalized_states[client_id], strict=True)
            dataset = load_task_npz_dataset(
                client_partition_path(settings.data.partitions_dir, client_id),
                settings.data,
            )
            _, validation_dataset = split_client_dataset(
                dataset,
                val_ratio=settings.data.client_val_ratio,
                seed=settings.data.seed + client_id,
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=settings.algorithm.batch_size,
                shuffle=False,
            )
            confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
            total_loss = 0.0
            total_examples = 0
            was_training = model.training
            model.eval()
            try:
                with torch.no_grad():
                    for features, targets in validation_loader:
                        features = features.to(device)
                        targets = targets.to(device)
                        logits = model(features)
                        total_loss += float(criterion(logits, targets).detach().cpu())
                        confusion += confusion_matrix_counts(
                            targets.detach().cpu().numpy(),
                            torch.argmax(logits, dim=1).detach().cpu().numpy(),
                            num_classes=num_classes,
                        )
                        total_examples += int(targets.shape[0])
            finally:
                model.train(was_training)
            metrics = {
                **classification_metrics_v4_from_confusion(
                    confusion,
                    minority_class_ids=minority_class_ids,
                    benign_id=benign_id,
                ),
                "loss": total_loss / max(total_examples, 1),
            }
            records.append(
                {
                    "client_id": int(client_id),
                    "split_seed": int(settings.data.seed + client_id),
                    "num_examples": int(len(validation_dataset)),
                    "validation_examples": int(len(validation_dataset)),
                    **{
                        name: float(metrics[name])
                        for name in metric_names
                    },
                }
            )
            del validation_loader, validation_dataset, dataset
    finally:
        del model

    if not records:
        raise ValueError("No private client validation records were produced")
    values = {
        name: np.asarray([record[name] for record in records], dtype=np.float64)
        for name in metric_names
    }
    weights = np.asarray(
        [record["num_examples"] for record in records],
        dtype=np.float64,
    )
    if np.any(weights <= 0.0) or not np.isfinite(weights).all():
        raise ValueError("Private validation splits must have positive sample counts")
    return {
        "split": "client_validation",
        "split_seed": int(settings.data.seed),
        "client_val_ratio": float(settings.data.client_val_ratio),
        "num_clients": int(len(records)),
        "per_client": records,
        "mean": {
            name: float(values[name].mean())
            for name in metric_names
        },
        # Population SD describes dispersion across the complete set of
        # participating clients (all ten clients are required by this suite).
        "std": {
            name: float(values[name].std(ddof=0))
            for name in metric_names
        },
        "sample_weighted_mean": {
            name: float(np.average(values[name], weights=weights))
            for name in metric_names
        },
        "total_examples": int(weights.sum()),
    }


def _global_validation_evaluate(
    model: torch.nn.Module,
    settings,
    device: torch.device,
    num_classes: int,
    *,
    minority_class_ids: list[int],
    benign_id: int | None,
) -> dict[str, float]:
    """Evaluate the global model on the frozen union of local validation splits."""
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_examples = 0
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for client_id in range(settings.data.num_clients):
                dataset = load_task_npz_dataset(
                    client_partition_path(settings.data.partitions_dir, client_id),
                    settings.data,
                )
                _, validation_dataset = split_client_dataset(
                    dataset,
                    val_ratio=settings.data.client_val_ratio,
                    seed=settings.data.seed + client_id,
                )
                validation_loader = DataLoader(
                    validation_dataset,
                    batch_size=settings.algorithm.batch_size,
                    shuffle=False,
                )
                for features, targets in validation_loader:
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
                del validation_loader, validation_dataset, dataset
    finally:
        model.train(was_training)
    union_metrics = {
        **classification_metrics_from_confusion(confusion),
        **classification_metrics_v4_from_confusion(
            confusion,
            minority_class_ids=minority_class_ids,
            benign_id=benign_id,
        ),
    }
    return {
        "validation_loss": total_loss / max(total_examples, 1),
        **{
            f"validation_{name}": value
            for name, value in union_metrics.items()
        },
        "validation_examples": float(total_examples),
    }


def _partition_provenance(settings, partition_hash: str) -> dict[str, Any]:
    """Load the already-verified manifest fields needed to audit a run."""
    manifest_path = Path(settings.data.partition_file).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("partition_hash") != partition_hash:
        raise ValueError("Verified partition hash differs from the manifest payload")
    return {
        "manifest_path": str(manifest_path),
        "partition_hash": partition_hash,
        "partitioning": manifest.get("partitioning", {}),
        "metadata_sha256": manifest.get("metadata", {}).get("sha256"),
        "clients": manifest.get("clients", []),
        "global_test": manifest["global_test"],
    }


def _validation_is_better(
    candidate_score: float,
    candidate_round: int,
    best_score: float | None,
    best_round: int | None,
) -> bool:
    """Apply the fixed max-Macro-F1, earliest-round tie-breaking policy."""
    if not math.isfinite(candidate_score):
        raise ValueError("validation_macro_f1 must be finite")
    if best_score is None or best_round is None:
        return True
    if candidate_score > best_score:
        return True
    return candidate_score == best_score and candidate_round < best_round


def _select_train_clients(settings, server_round: int) -> list[int]:
    num_clients = settings.data.num_clients
    requested = max(1, ceil(num_clients * settings.algorithm.fraction_train))
    min_train_nodes = min(settings.algorithm.min_train_nodes, num_clients)
    num_selected = min(num_clients, max(requested, min_train_nodes))
    if num_selected == num_clients:
        return list(range(num_clients))
    generator = torch.Generator().manual_seed(settings.runtime.seed + server_round)
    return torch.randperm(num_clients, generator=generator)[:num_selected].tolist()



def _print_metrics(scope: str, server_round: int, metrics: dict[str, float]) -> None:
    metric_text = " ".join(
        f"{key}={value:.6f}" for key, value in sorted(metrics.items())
    )
    print(f"{scope} round={server_round} {metric_text}", flush=True)


def _write_client_history(
    history: list[dict[str, Any]],
    results_dir: Path,
    run_name: str,
) -> None:
    json_path = results_dir / f"{run_name}_client_history.json"
    csv_path = results_dir / f"{run_name}_client_history.csv"
    json_path.write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")
    if not history:
        return
    keys: list[str] = []
    for record in history:
        for key in record:
            if key not in keys:
                keys.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def _write_summary(
    settings,
    logger: RoundMetricsLogger,
    *,
    partition_record: dict[str, Any],
    model_size_bytes: int,
    communication_bytes: int,
    communication_components: dict[str, int],
    best_round: int,
    best_validation_metrics: dict[str, float],
    final_round_metrics: dict[str, float],
    best_test_metrics: dict[str, float],
    final_test_metrics: dict[str, float],
    metric_contract: dict[str, Any],
    best_private_validation_metrics: dict[str, Any] | None = None,
    final_private_validation_metrics: dict[str, Any] | None = None,
) -> None:
    quantization = settings.technique.quantization
    if sum(communication_components.values()) != communication_bytes:
        raise ValueError("Summary communication components do not sum to total bytes")
    summary = {
        "experiment_id": settings.experiment_id,
        "run_name": settings.results.run_name,
        "algorithm": settings.algorithm.name,
        "technique": settings.technique.name,
        "fidelity": settings.fidelity.classification,
        "status": settings.fidelity.status,
        "num_clients": settings.data.num_clients,
        "num_classes": settings.data.num_classes,
        "classification_metric_schema_version": metric_contract[
            "classification_metric_schema_version"
        ],
        "classification_metric_keys": metric_contract["classification_metric_keys"],
        "metric_contract_sha256": metric_contract["sha256"],
        "minority_class_ids": metric_contract["minority_class_ids"],
        "minority_class_names": metric_contract["minority_class_names"],
        "seed": settings.runtime.seed,
        "training_seed": settings.runtime.seed,
        "data_split_seed": settings.data.seed,
        "partition_hash": partition_record["partition_hash"],
        "global_test_sha256": partition_record["global_test"]["sha256"],
        "global_test_x_sha256": partition_record["global_test"]["x_sha256"],
        "global_test_y_sha256": partition_record["global_test"]["y_sha256"],
        "model_size_bytes": model_size_bytes,
        "communication_upload_bytes": communication_bytes,
        "communication_accounting": {
            "scope": "serialized_client_uplink_only",
            "downlink_measured": False,
            "includes_message_metadata": True,
            "components_bytes": communication_components,
        },
        "sparsity": float(final_round_metrics.get("sparsity", 0.0)),
        "quantization_method": quantization.method,
        "quantization_level": quantization.levels if quantization.enabled else None,
        "bit_width": (
            math.ceil(math.log2(quantization.levels + 1)) + 1
            if quantization.enabled
            else 32
        ),
        "final_round_effective_bits_per_float_value": final_round_metrics.get(
            "bit_width"
        ),
        "zero_point": None,
        "checkpoint_selection": BEST_CHECKPOINT_POLICY,
        "best_round": best_round,
        "final_round": settings.algorithm.num_server_rounds,
        "best_validation_metrics": best_validation_metrics,
        "final_round_metrics": final_round_metrics,
        "best_test_metrics": best_test_metrics,
        "final_test_metrics": final_test_metrics,
        "final_metrics": final_test_metrics,
    }
    # Only BDD-HFL variants provide private models. Keep these fields separate from the
    # global test metrics so downstream reports cannot mistake personalized
    # client-validation results for centralized global-test results.
    if best_private_validation_metrics is not None:
        summary["best_private_validation_metrics"] = best_private_validation_metrics
        summary["best_private_state_round"] = best_round
    if final_private_validation_metrics is not None:
        summary["final_private_validation_metrics"] = final_private_validation_metrics
        summary["final_private_state_round"] = settings.algorithm.num_server_rounds
    path = logger.results_dir / f"{settings.results.run_name}_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _create_client_executors(
    client_devices: list[torch.device],
) -> list[ProcessPoolExecutor]:
    if len(client_devices) <= 1:
        return []
    context = mp.get_context("spawn")
    return [
        ProcessPoolExecutor(
            max_workers=1,
            mp_context=context,
            initializer=initialize_client_worker,
            initargs=(str(device),),
        )
        for device in client_devices
    ]


def _train_round_clients(
    *,
    settings,
    server_round: int,
    global_state: dict[str, torch.Tensor],
    input_dim: int,
    num_classes: int,
    train_client_ids: list[int],
    private_states: dict[int, dict[str, torch.Tensor]],
    fap_active_indices: dict[int, list[int]],
    fap_original_num_examples: dict[int, int],
    device: torch.device,
    executors: list[ProcessPoolExecutor],
) -> list[ClientTrainResult]:
    specs = [
        ClientTrainSpec(
            client_id=client_id,
            private_state=private_states.get(client_id),
            active_indices=(
                tuple(fap_active_indices[client_id])
                if settings.technique.pruning.enabled
                else None
            ),
            original_num_examples=(
                fap_original_num_examples[client_id]
                if settings.technique.pruning.enabled
                else None
            ),
        )
        for client_id in train_client_ids
    ]
    if not executors:
        request = ClientBatchRequest(
            server_round=server_round,
            global_state=global_state,
            settings=settings,
            input_dim=input_dim,
            num_classes=num_classes,
            clients=tuple(specs),
        )
        return train_client_batch_local(request, device)

    client_buckets: list[list[ClientTrainSpec]] = [
        [] for _ in range(len(executors))
    ]
    for index, spec in enumerate(specs):
        client_buckets[index % len(executors)].append(spec)

    futures = []
    for executor, bucket in zip(executors, client_buckets, strict=True):
        if not bucket:
            continue
        request = ClientBatchRequest(
            server_round=server_round,
            global_state=global_state,
            settings=settings,
            input_dim=input_dim,
            num_classes=num_classes,
            clients=tuple(bucket),
        )
        futures.append(executor.submit(train_client_batch, request))

    results_by_client = {
        result.client_id: result
        for future in futures
        for result in future.result()
    }
    return [results_by_client[client_id] for client_id in train_client_ids]


def _metric_label_space(settings, num_classes: int) -> str:
    """Map a baseline data config onto the shared label-space vocabulary.

    ``data.source_label_space`` describes what the loader does to the stored
    NPZ ids.  The metric contract instead needs the *task* those ids end up in,
    which is what decides the class names and therefore the minority set.
    """
    if settings.data.source_label_space == "ciciot2023_34":
        return "ciciot2023_34"
    if int(num_classes) == len(CICIOT2023_LABELS):
        return IDENTITY_34_LABEL_SPACE
    # Synthetic fixtures train on widths the project does not name; they still
    # get a well-formed contract with positional class names.
    return generic_identity_label_space(num_classes)


def main() -> None:
    args = parse_args()
    settings = _settings_from_args(args)
    partition_hash = verify_partition_manifest(settings)
    partition_record = _partition_provenance(settings, partition_hash)
    seed_everything(settings.runtime.seed)
    input_dim, num_classes = resolve_dimensions(settings)
    metric_contract = build_metric_contract(
        partitions_dir=settings.data.partitions_dir,
        num_clients=settings.data.num_clients,
        num_classes=num_classes,
        source_label_space=_metric_label_space(settings, num_classes),
        client_val_ratio=settings.data.client_val_ratio,
        split_seed=settings.data.seed,
        minority_fraction=settings.data.minority_fraction,
        partition_hash=partition_hash,
    )
    class_names = tuple(metric_contract["target_class_names"])
    minority_class_ids = [int(value) for value in metric_contract["minority_class_ids"]]
    benign_id = metric_contract["benign_class_id"]
    print(
        "METRIC_CONTRACT "
        f"schema_version={metric_contract['classification_metric_schema_version']} "
        f"num_classes={num_classes} "
        f"metrics={len(metric_contract['classification_metric_keys'])} "
        f"benign_class={metric_contract['benign_class_name']} "
        f"minority_classes={metric_contract['minority_class_names']}",
        flush=True,
    )
    client_devices = select_client_devices(
        settings.runtime.device,
        settings.runtime.parallel_clients,
    )
    device = client_devices[0]
    print(
        "RUNTIME "
        f"evaluation_device={device} "
        f"parallel_clients={len(client_devices)} "
        f"client_devices={','.join(str(item) for item in client_devices)}",
        flush=True,
    )

    logger = RoundMetricsLogger(settings.results.dir, settings.results.run_name)
    logger.save_config(settings)
    test_contract_path = logger.enforce_fixed_test_contract(
        partition_record["global_test"]
    )
    metric_contract_path = enforce_metric_contract(
        logger.results_dir.parent,
        metric_contract,
    )
    logger.save_provenance(
        settings,
        project_root=PROJECT_ROOT,
        partition_record={
            **partition_record,
            "fixed_test_contract_path": str(test_contract_path),
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
        },
        launcher_command=args.launcher_command,
    )
    global_model = build_model(settings.model, input_dim=input_dim, num_classes=num_classes)
    global_model.to(device)
    quantized_parameter_names = {
        name for name, _ in global_model.named_parameters()
    }
    model_size_bytes = _state_size_bytes(_cpu_state_dict(global_model))
    client_history: list[dict[str, Any]] = []
    cumulative_upload_bytes = 0
    cumulative_payload_components = {
        name: 0 for name in PAYLOAD_COMPONENT_NAMES
    }

    private_states: dict[int, dict[str, torch.Tensor]] = {}
    if settings.technique.distillation.enabled:
        private_template = build_model(
            settings.model,
            input_dim=input_dim,
            num_classes=num_classes,
        )
        template_state = _cpu_state_dict(private_template)
        private_states = {
            client_id: {name: value.clone() for name, value in template_state.items()}
            for client_id in range(settings.data.num_clients)
        }

    quant_controller = (
        DAdaQuantController(settings.technique.quantization)
        if settings.technique.quantization.method == "dadaquant"
        else None
    )
    fap_original_num_examples: dict[int, int] = {}
    fap_active_indices: dict[int, list[int]] = {}
    if settings.technique.pruning.enabled:
        for client_id in range(settings.data.num_clients):
            dataset = load_task_npz_dataset(
                client_partition_path(settings.data.partitions_dir, client_id),
                settings.data,
            )
            train_dataset, _ = split_client_dataset(
                dataset,
                val_ratio=settings.data.client_val_ratio,
                seed=settings.data.seed + client_id,
            )
            indices = [
                int(index)
                for index in getattr(train_dataset, "indices", range(len(train_dataset)))
            ]
            fap_original_num_examples[client_id] = len(indices)
            fap_active_indices[client_id] = indices
        del dataset, train_dataset

    resume_path = (
        Path(args.resume).expanduser().resolve()
        if args.resume is not None
        else None
    )
    final_round_metrics: dict[str, float]
    best_round: int | None = None
    best_validation_macro_f1: float | None = None
    best_validation_metrics: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    # BDD-HFL has one persistent private state per client.  These snapshots
    # are intentionally independent from ``private_states`` so a later round
    # can never mutate the state paired with the validation-selected global
    # checkpoint.
    best_private_states: dict[int, dict[str, torch.Tensor]] | None = None
    start_round = 1

    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        checkpoint = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )
        _validate_resume_checkpoint(
            checkpoint,
            settings=settings,
            partition_hash=partition_hash,
            reference_global_state=_cpu_state_dict(global_model),
            reference_private_states=private_states,
            reference_fap_original_num_examples=fap_original_num_examples,
            reference_fap_active_indices=fap_active_indices,
            quant_controller=quant_controller,
        )
        global_model.load_state_dict(checkpoint["global_state"], strict=True)
        private_states = _clone_private_states(checkpoint["private_states"])
        fap_original_num_examples = {
            int(client_id): int(value)
            for client_id, value in checkpoint["fap_original_num_examples"].items()
        }
        fap_active_indices = {
            int(client_id): [int(index) for index in indices]
            for client_id, indices in checkpoint["fap_active_indices"].items()
        }
        cumulative_upload_bytes = int(checkpoint["cumulative_upload_bytes"])
        cumulative_payload_components = {
            name: int(value)
            for name, value in checkpoint["cumulative_payload_components"].items()
        }
        best_round = int(checkpoint["best_round"])
        best_validation_macro_f1 = float(
            checkpoint["best_validation_macro_f1"]
        )
        best_validation_metrics = copy.deepcopy(
            checkpoint["best_validation_metrics"]
        )
        best_state = _clone_state_dict(checkpoint["best_state"])
        best_private_states = (
            _clone_private_states(checkpoint["best_private_states"])
            if checkpoint["best_private_states"] is not None
            else None
        )
        final_round_metrics = copy.deepcopy(checkpoint["final_round_metrics"])
        client_history = copy.deepcopy(checkpoint["client_history"])
        logger.restore_records(checkpoint["round_records"])
        completed_round = int(checkpoint["round"])
        start_round = completed_round + 1
        print(
            "RESUME "
            f"checkpoint={resume_path} "
            f"completed_round={completed_round} "
            f"next_round={start_round}",
            flush=True,
        )
    else:
        initial_metrics = _global_validation_evaluate(
            global_model,
            settings,
            device,
            num_classes,
            minority_class_ids=minority_class_ids,
            benign_id=benign_id,
        )
        initial_metrics.update(
            {
                "model_size_bytes": float(model_size_bytes),
                "communication_upload_bytes": 0.0,
                "communication_cumulative_bytes": 0.0,
                **{
                    f"communication_upload_{name}": 0.0
                    for name in PAYLOAD_COMPONENT_NAMES
                },
                **{
                    f"communication_cumulative_{name}": 0.0
                    for name in PAYLOAD_COMPONENT_NAMES
                },
                "sparsity": 0.0,
                "bit_width": 32.0,
            }
        )
        logger.log_round(0, initial_metrics)
        _print_metrics("GLOBAL_VALIDATION", 0, initial_metrics)
        final_round_metrics = initial_metrics

    client_executors = _create_client_executors(client_devices)
    for server_round in range(start_round, settings.algorithm.num_server_rounds + 1):
        print(f"ROUND {server_round}/{settings.algorithm.num_server_rounds}", flush=True)
        global_state = _cpu_state_dict(global_model)
        client_states: list[tuple[dict[str, torch.Tensor], int]] = []
        quantized_updates: list[tuple[QuantizedStateUpdate, int]] = []
        round_upload_bytes = 0
        round_payload_components = {
            name: 0 for name in PAYLOAD_COMPONENT_NAMES
        }
        round_quantized_values = 0
        round_quantized_bits = 0
        round_quantization_level_weighted = 0.0
        round_quantization_scale_weighted = 0.0
        round_pretrain_loss_weighted = 0.0
        round_examples = 0
        round_fap_original_examples = 0
        round_fap_active_before = 0
        round_fap_active_after = 0
        round_fap_pruned_examples = 0

        train_client_ids = _select_train_clients(settings, server_round)
        client_results = _train_round_clients(
            settings=settings,
            server_round=server_round,
            global_state=global_state,
            input_dim=input_dim,
            num_classes=num_classes,
            train_client_ids=train_client_ids,
            private_states=private_states,
            fap_active_indices=fap_active_indices,
            fap_original_num_examples=fap_original_num_examples,
            device=device,
            executors=client_executors,
        )
        example_counts = {
            result.client_id: result.num_examples for result in client_results
        }
        client_levels: dict[int, int] = {}
        if settings.technique.quantization.method == "fedpaq":
            client_levels = {
                client_id: settings.technique.quantization.levels
                for client_id in train_client_ids
            }
        elif quant_controller is not None:
            base_level = quant_controller.level_for_next_round()
            client_levels = quant_controller.client_levels(
                base_level,
                example_counts,
                min_level=settings.technique.quantization.min_level,
                max_level=settings.technique.quantization.max_level,
            )

        for result in client_results:
            client_id = result.client_id
            local_state = result.local_state
            metrics = result.metrics
            num_examples = result.num_examples
            pretrain_loss = result.pretrain_loss
            if pretrain_loss is not None:
                round_pretrain_loss_weighted += pretrain_loss * num_examples
            if settings.technique.distillation.enabled:
                if result.private_state is None:
                    raise RuntimeError("Distillation worker did not return private state")
                private_states[client_id] = result.private_state
            elif settings.technique.pruning.enabled:
                active_before = list(fap_active_indices[client_id])
                if result.active_indices is None:
                    raise RuntimeError("FAP worker did not return active indices")
                fap_active_indices[client_id] = list(result.active_indices)
                round_fap_original_examples += fap_original_num_examples[client_id]
                round_fap_active_before += len(active_before)
                round_fap_active_after += len(result.active_indices)
                round_fap_pruned_examples += len(active_before) - len(
                    result.active_indices
                )

            record: dict[str, Any] = {
                "round": server_round,
                "client_id": client_id,
                "num_examples": num_examples,
                "partition_hash": partition_hash,
            }
            record.update(metrics)
            if pretrain_loss is not None:
                record["dadaquant_pretrain_loss"] = pretrain_loss

            if settings.technique.quantization.enabled:
                level = client_levels[client_id]
                quantized = quantize_state_update(
                    local_state,
                    global_state,
                    levels=level,
                    seed=settings.runtime.seed + server_round * 10_000 + client_id,
                    lossless_qsgd_encoding=(
                        settings.technique.quantization.zero_run_length_encoding
                        and settings.technique.quantization.elias_omega_encoding
                    ),
                    client_id=client_id,
                    server_round=server_round,
                    num_examples=num_examples,
                    quantized_names=quantized_parameter_names,
                )
                quantized_updates.append((quantized, num_examples))
                serialized_upload = quantized
                upload_bytes = serialized_upload.payload_bytes
                round_upload_bytes += upload_bytes
                round_quantized_values += quantized.num_values
                round_quantized_bits += upload_bytes * 8
                round_quantization_level_weighted += level * num_examples
                round_quantization_scale_weighted += quantized.scale * num_examples
                record.update(
                    {
                        "upload_bytes": upload_bytes,
                        "quantization_level": level,
                        "quantization_scale": quantized.scale,
                        "quantization_zero_point": "not_applicable_qsgd",
                        "quantization_effective_bits_per_value": quantized.effective_bits_per_value,
                        "quantization_nonzero_values": quantized.nonzero_values,
                        "quantization_squared_error": quantized.squared_error,
                        "quantization_squared_norm": quantized.squared_norm,
                        "quantization_encoding": quantized.encoding,
                        "quantization_magnitude_bit_width": quantized.magnitude_width,
                    }
                )
            else:
                serialized_upload = serialize_dense_state(
                    local_state,
                    client_id=client_id,
                    server_round=server_round,
                    num_examples=num_examples,
                )
                # Aggregate the decoded state to ensure the measured wire
                # representation, rather than an in-memory shortcut, is used.
                client_states.append((serialized_upload.state, num_examples))
                upload_bytes = serialized_upload.payload_bytes
                round_upload_bytes += upload_bytes
                record.update({"upload_bytes": upload_bytes, "bit_width": 32})

            component_counts = serialized_upload.byte_counts
            exclusive_components = {
                "header_bytes": component_counts.header_bytes,
                "norm_bytes": component_counts.norm_bytes,
                "layout_bytes": component_counts.layout_bytes,
                "value_bytes": component_counts.quantized_value_bytes,
                "index_bytes": component_counts.index_bytes,
                "auxiliary_bytes": component_counts.auxiliary_value_bytes,
            }
            if sum(exclusive_components.values()) != upload_bytes:
                raise RuntimeError("Serialized upload component counts do not sum to bytes")
            for name, count in exclusive_components.items():
                round_payload_components[name] += count
                record[f"payload_{name}"] = count

            round_examples += num_examples
            client_history.append(record)
            print(
                "CLIENT "
                f"round={server_round} id={client_id} "
                f"num_examples={num_examples} train_loss={metrics['train_loss']:.6f} "
                f"upload_bytes={upload_bytes}",
                flush=True,
            )

        if settings.technique.quantization.enabled:
            aggregated_state = _aggregate_quantized_updates(global_state, quantized_updates)
        else:
            aggregated_state = _weighted_average_state_dicts(client_states)
        global_model.load_state_dict(aggregated_state, strict=True)

        if quant_controller is not None:
            quant_controller.observe_weighted_loss(
                round_pretrain_loss_weighted / max(round_examples, 1)
            )
        if sum(round_payload_components.values()) != round_upload_bytes:
            raise RuntimeError("Round payload component counts do not sum to upload bytes")
        cumulative_upload_bytes += round_upload_bytes
        for name, count in round_payload_components.items():
            cumulative_payload_components[name] += count
        if sum(cumulative_payload_components.values()) != cumulative_upload_bytes:
            raise RuntimeError(
                "Cumulative payload component counts do not sum to upload bytes"
            )
        round_metrics = _global_validation_evaluate(
            global_model,
            settings,
            device,
            num_classes,
            minority_class_ids=minority_class_ids,
            benign_id=benign_id,
        )
        if settings.technique.quantization.enabled:
            effective_bits = round_quantized_bits / max(round_quantized_values, 1)
            mean_level = round_quantization_level_weighted / max(round_examples, 1)
            mean_scale = round_quantization_scale_weighted / max(round_examples, 1)
        else:
            effective_bits = 32.0
            mean_level = 0.0
            mean_scale = 0.0
        if settings.technique.pruning.enabled:
            fap_data_sparsity = 1.0 - (
                round_fap_active_after / max(round_fap_original_examples, 1)
            )
        else:
            fap_data_sparsity = 0.0
        round_metrics.update(
            {
                "model_size_bytes": float(model_size_bytes),
                "communication_upload_bytes": float(round_upload_bytes),
                "communication_cumulative_bytes": float(cumulative_upload_bytes),
                **{
                    f"communication_upload_{name}": float(count)
                    for name, count in round_payload_components.items()
                },
                **{
                    f"communication_cumulative_{name}": float(count)
                    for name, count in cumulative_payload_components.items()
                },
                "sparsity": float(fap_data_sparsity),
                "bit_width": float(effective_bits),
                "quantization_level": float(mean_level),
                "quantization_scale": float(mean_scale),
                "fap_original_train_examples": float(round_fap_original_examples),
                "fap_active_examples_before": float(round_fap_active_before),
                "fap_active_examples_after": float(round_fap_active_after),
                "fap_pruned_examples": float(round_fap_pruned_examples),
                "fap_data_sparsity": float(fap_data_sparsity),
            }
        )
        logger.log_round(server_round, round_metrics)
        _print_metrics("GLOBAL_VALIDATION", server_round, round_metrics)
        final_round_metrics = round_metrics

        validation_macro_f1 = round_metrics["validation_macro_f1"]
        is_new_best = _validation_is_better(
            validation_macro_f1,
            server_round,
            best_validation_macro_f1,
            best_round,
        )
        if is_new_best:
            best_round = server_round
            best_validation_macro_f1 = validation_macro_f1
            best_validation_metrics = dict(round_metrics)
            best_state = _cpu_state_dict(global_model)
            if settings.technique.distillation.enabled:
                best_private_states = _clone_private_states(private_states)
            print(
                "BEST_CHECKPOINT "
                f"round={best_round} "
                f"validation_macro_f1={best_validation_macro_f1:.6f}",
                flush=True,
            )

        if settings.results.save_round_checkpoints:
            checkpoint_dir = logger.results_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                global_model.state_dict(),
                checkpoint_dir / f"{settings.results.run_name}_round_{server_round:03d}.pt",
            )
            if (
                best_round is None
                or best_validation_macro_f1 is None
                or best_validation_metrics is None
                or best_state is None
            ):
                raise RuntimeError("Resume checkpoint is missing best-round state")
            resume_checkpoint = (
                checkpoint_dir / f"{settings.results.run_name}_resume.pt"
            )
            _save_resume_checkpoint(
                resume_checkpoint,
                _resume_checkpoint_payload(
                    settings=settings,
                    server_round=server_round,
                    partition_hash=partition_hash,
                    global_state=_cpu_state_dict(global_model),
                    private_states=private_states,
                    fap_original_num_examples=fap_original_num_examples,
                    fap_active_indices=fap_active_indices,
                    quant_controller=quant_controller,
                    cumulative_upload_bytes=cumulative_upload_bytes,
                    cumulative_payload_components=cumulative_payload_components,
                    best_round=best_round,
                    best_validation_macro_f1=best_validation_macro_f1,
                    best_validation_metrics=best_validation_metrics,
                    best_state=best_state,
                    best_private_states=best_private_states,
                    final_round_metrics=final_round_metrics,
                    round_records=logger.snapshot_records(),
                    client_history=client_history,
                ),
            )
            print(
                f"RESUME_CHECKPOINT round={server_round} path={resume_checkpoint}",
                flush=True,
            )

    for executor in client_executors:
        executor.shutdown(wait=True)

    if best_round is None or best_validation_metrics is None or best_state is None:
        raise RuntimeError("No trained round was eligible for checkpoint selection")

    # Capture the live private mapping before any post-selection model loading.
    # This is the only valid final-round private checkpoint; the best snapshot
    # above was captured at the exact round that selected ``best_state``.
    final_private_states: dict[int, dict[str, torch.Tensor]] | None = None
    if settings.technique.distillation.enabled:
        final_private_states = _clone_private_states(private_states)
        if best_private_states is None:
            raise RuntimeError(
                "BDD-HFL checkpoint selection did not capture private states"
            )
        if settings.results.save_round_checkpoints:
            # Write exactly the private snapshot paired with the selected
            # global round.  Deferring this until selection is complete avoids
            # leaving ambiguous stale private-round files when a later round
            # supersedes an earlier validation best.
            checkpoint_dir = logger.results_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                best_private_states,
                checkpoint_dir
                / f"{settings.results.run_name}_private_round_{best_round:03d}.pt",
            )

    final_state = _cpu_state_dict(global_model)
    final_test_metrics, final_test_details = _global_evaluate(
        global_model,
        settings,
        device,
        num_classes,
        class_names=class_names,
        minority_class_ids=minority_class_ids,
        benign_id=benign_id,
    )
    _print_metrics(
        "GLOBAL_TEST_FINAL",
        settings.algorithm.num_server_rounds,
        final_test_metrics,
    )
    if best_round == settings.algorithm.num_server_rounds:
        best_test_metrics = dict(final_test_metrics)
        best_test_details = dict(final_test_details)
    else:
        global_model.load_state_dict(best_state, strict=True)
        best_test_metrics, best_test_details = _global_evaluate(
            global_model,
            settings,
            device,
            num_classes,
            class_names=class_names,
            minority_class_ids=minority_class_ids,
            benign_id=benign_id,
        )
        _print_metrics("GLOBAL_TEST_BEST", best_round, best_test_metrics)
        global_model.load_state_dict(final_state, strict=True)

    best_private_validation_metrics: dict[str, Any] | None = None
    final_private_validation_metrics: dict[str, Any] | None = None
    if best_private_states is not None and final_private_states is not None:
        # Evaluate the two immutable snapshots independently.  In particular,
        # never evaluate ``private_states`` after restoring ``best_state``:
        # that mapping is always the final-round state.
        best_private_validation_metrics = _private_validation_evaluate(
            best_private_states,
            settings,
            device,
            input_dim=input_dim,
            num_classes=num_classes,
            minority_class_ids=minority_class_ids,
            benign_id=benign_id,
        )
        final_private_validation_metrics = _private_validation_evaluate(
            final_private_states,
            settings,
            device,
            input_dim=input_dim,
            num_classes=num_classes,
            minority_class_ids=minority_class_ids,
            benign_id=benign_id,
        )
        _print_metrics(
            "PRIVATE_VALIDATION_BEST",
            best_round,
            best_private_validation_metrics["sample_weighted_mean"],
        )
        _print_metrics(
            "PRIVATE_VALIDATION_FINAL",
            settings.algorithm.num_server_rounds,
            final_private_validation_metrics["sample_weighted_mean"],
        )

    test_evaluations = {
        "global_test_sha256": partition_record["global_test"]["sha256"],
        "selection_policy": BEST_CHECKPOINT_POLICY,
        "best": {
            "round": best_round,
            "validation_metrics": best_validation_metrics,
            "test_metrics": best_test_metrics,
            "details": best_test_details,
        },
        "final": {
            "round": settings.algorithm.num_server_rounds,
            "validation_metrics": final_round_metrics,
            "test_metrics": final_test_metrics,
            "details": final_test_details,
        },
    }
    if best_private_validation_metrics is not None:
        test_evaluations["best"]["private_state_round"] = best_round
        test_evaluations["best"][
            "private_validation_metrics"
        ] = best_private_validation_metrics
    if final_private_validation_metrics is not None:
        test_evaluations["final"][
            "private_state_round"
        ] = settings.algorithm.num_server_rounds
        test_evaluations["final"][
            "private_validation_metrics"
        ] = final_private_validation_metrics
    logger.save_test_evaluations(test_evaluations)

    _write_client_history(client_history, logger.results_dir, settings.results.run_name)
    if settings.results.save_model:
        torch.save(
            final_state,
            logger.results_dir / f"{settings.results.run_name}_final_model.pt",
        )
        torch.save(
            best_state,
            logger.results_dir / f"{settings.results.run_name}_best_model.pt",
        )
        if best_private_states is not None and final_private_states is not None:
            torch.save(
                best_private_states,
                logger.results_dir
                / f"{settings.results.run_name}_best_private_models.pt",
            )
            torch.save(
                final_private_states,
                logger.results_dir
                / f"{settings.results.run_name}_final_private_models.pt",
            )
    _write_summary(
        settings,
        logger,
        partition_record=partition_record,
        model_size_bytes=model_size_bytes,
        communication_bytes=cumulative_upload_bytes,
        communication_components=dict(cumulative_payload_components),
        best_round=best_round,
        best_validation_metrics=best_validation_metrics,
        final_round_metrics=final_round_metrics,
        best_test_metrics=best_test_metrics,
        final_test_metrics=final_test_metrics,
        metric_contract=metric_contract,
        best_private_validation_metrics=best_private_validation_metrics,
        final_private_validation_metrics=final_private_validation_metrics,
    )


if __name__ == "__main__":
    main()
