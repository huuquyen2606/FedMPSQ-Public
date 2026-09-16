"""Isolated client training for sequential and multi-GPU federated runs."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import (
    FlowDataset,
    client_partition_path,
    load_task_npz_dataset,
    split_client_dataset,
)
from extensions.pruning import make_fap_loader, train_fap_model
from fl.config import ExperimentSettings
from models import build_model
from training.trainer import (
    evaluate_cross_entropy_loss,
    train_bdd_hfl_models,
    train_local_model,
)


@dataclass(frozen=True)
class ClientTrainSpec:
    """Per-client state required for one federated round."""

    client_id: int
    private_state: dict[str, torch.Tensor] | None = None
    active_indices: tuple[int, ...] | None = None
    original_num_examples: int | None = None


@dataclass(frozen=True)
class ClientBatchRequest:
    """A group of clients trained sequentially on one device."""

    server_round: int
    global_state: dict[str, torch.Tensor]
    settings: ExperimentSettings
    input_dim: int
    num_classes: int
    clients: tuple[ClientTrainSpec, ...]


@dataclass(frozen=True)
class ClientTrainResult:
    """Client output returned to the central aggregation process."""

    client_id: int
    local_state: dict[str, torch.Tensor]
    metrics: dict[str, float]
    num_examples: int
    pretrain_loss: float | None = None
    private_state: dict[str, torch.Tensor] | None = None
    active_indices: tuple[int, ...] | None = None


_WORKER_DEVICE: torch.device | None = None
_DATASET_CACHE: dict[tuple[str, int, str, int], FlowDataset] = {}


def initialize_client_worker(device_name: str) -> None:
    """Bind a persistent worker process to exactly one CUDA device."""
    global _WORKER_DEVICE
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    torch.set_num_threads(1)
    _WORKER_DEVICE = device


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _client_seed(settings: ExperimentSettings, server_round: int, client_id: int) -> int:
    return settings.runtime.seed + server_round * 100_000 + client_id


def _seed_client(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)


def _load_client_dataset(settings: ExperimentSettings, client_id: int) -> FlowDataset:
    cache_key = (
        str(Path(settings.data.partitions_dir).resolve()),
        client_id,
        settings.data.source_label_space,
        settings.data.num_classes,
    )
    dataset = _DATASET_CACHE.get(cache_key)
    if dataset is None:
        dataset = load_task_npz_dataset(
            client_partition_path(settings.data.partitions_dir, client_id),
            settings.data,
        )
        _DATASET_CACHE[cache_key] = dataset
    return dataset


def _build_train_loader(
    request: ClientBatchRequest,
    spec: ClientTrainSpec,
) -> DataLoader:
    settings = request.settings
    dataset = _load_client_dataset(settings, spec.client_id)
    shuffle_seed = _client_seed(settings, request.server_round, spec.client_id)
    if settings.technique.pruning.enabled:
        if spec.active_indices is None:
            raise ValueError("FAP client request requires active_indices")
        return make_fap_loader(
            dataset,
            spec.active_indices,
            batch_size=settings.algorithm.batch_size,
            shuffle_seed=shuffle_seed,
        )

    train_dataset, _ = split_client_dataset(
        dataset,
        val_ratio=settings.data.client_val_ratio,
        seed=settings.data.seed + spec.client_id,
    )
    return DataLoader(
        train_dataset,
        batch_size=settings.algorithm.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(shuffle_seed),
    )


def _train_client(
    request: ClientBatchRequest,
    spec: ClientTrainSpec,
    device: torch.device,
) -> ClientTrainResult:
    settings = request.settings
    client_seed = _client_seed(settings, request.server_round, spec.client_id)
    _seed_client(client_seed, device)
    train_loader = _build_train_loader(request, spec)
    num_examples = len(train_loader.dataset)

    local_model = build_model(
        settings.model,
        input_dim=request.input_dim,
        num_classes=request.num_classes,
    )
    local_model.load_state_dict(request.global_state, strict=True)
    local_model.to(device)

    pretrain_loss = None
    if settings.technique.quantization.method == "dadaquant":
        loss_loader = DataLoader(
            train_loader.dataset,
            batch_size=settings.algorithm.batch_size,
            shuffle=False,
        )
        pretrain_loss = evaluate_cross_entropy_loss(local_model, loss_loader, device)

    private_state = None
    active_indices = None
    if settings.technique.distillation.enabled:
        if spec.private_state is None:
            raise ValueError("Distillation client request requires private_state")
        private_model = build_model(
            settings.model,
            input_dim=request.input_dim,
            num_classes=request.num_classes,
        )
        private_model.load_state_dict(spec.private_state, strict=True)
        private_model.to(device)
        metrics = train_bdd_hfl_models(
            local_model,
            private_model,
            train_loader,
            settings.algorithm,
            settings.technique.distillation,
            device,
        )
        private_state = _cpu_state_dict(private_model)
        del private_model
    elif settings.technique.pruning.enabled:
        if spec.active_indices is None:
            raise ValueError("FAP client request requires active_indices")
        if spec.original_num_examples is None:
            raise ValueError("FAP client request requires original_num_examples")
        fap_result = train_fap_model(
            local_model,
            train_loader,
            spec.active_indices,
            settings.algorithm,
            settings.technique.pruning,
            device,
            server_round=request.server_round,
            original_num_examples=spec.original_num_examples,
            noise_seed=(
                settings.runtime.seed
                + request.server_round * 10_000
                + spec.client_id
            ),
        )
        metrics = fap_result.metrics
        active_indices = tuple(fap_result.active_indices)
    else:
        metrics = train_local_model(
            local_model,
            train_loader,
            settings.algorithm,
            device,
        )

    local_state = _cpu_state_dict(local_model)
    del local_model
    return ClientTrainResult(
        client_id=spec.client_id,
        local_state=local_state,
        metrics=metrics,
        num_examples=num_examples,
        pretrain_loss=pretrain_loss,
        private_state=private_state,
        active_indices=active_indices,
    )


def train_client_batch(request: ClientBatchRequest) -> list[ClientTrainResult]:
    """Train one client batch inside an initialized worker process."""
    if _WORKER_DEVICE is None:
        raise RuntimeError("Client worker was not initialized with a device")
    return [
        _train_client(request, spec, _WORKER_DEVICE)
        for spec in request.clients
    ]


def train_client_batch_local(
    request: ClientBatchRequest,
    device: torch.device,
) -> list[ClientTrainResult]:
    """Train one client batch in the central process for single-device runs."""
    return [_train_client(request, spec, device) for spec in request.clients]
