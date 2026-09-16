"""Isolated local worker for the FedMPSQ MVP."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from training.costs import RSSMonitor
from extensions.mpsq_int4 import OnlineTaskSaliency, round_sparsity
from extensions.rare_budget import minority_head_groups
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

from data.dataset import FlowDataset, client_partition_path, split_client_dataset
from data.fedmpsq_labels import load_grouped_npz_dataset
from extensions.flat_uplink import (
    FLAT_KEY,
    check_flat_uplink_is_sound,
    flatten_state,
    wire_layout,
)
from extensions.fedmpsq import (
    clone_tensor_state,
    compress_update,
    floating_state_delta,
    zeros_like_floating_state,
)
from fl.fedmpsq_config import FedMPSQConfig
from models import build_model
from training.trainer import build_optimizer


@dataclass(frozen=True)
class FedMPSQClientSpec:
    client_id: int
    saliency_state: dict[str, torch.Tensor]
    residual_state: dict[str, torch.Tensor]


@dataclass(frozen=True)
class FedMPSQBatchRequest:
    server_round: int
    global_state: dict[str, torch.Tensor]
    settings: FedMPSQConfig
    input_dim: int
    clients: tuple[FedMPSQClientSpec, ...]
    # Pooled local-training class counts, already frozen in the shared metric
    # contract at run start, so method.class_prior == "global" costs no extra
    # communication and reveals nothing the protocol did not already fix.
    pooled_counts: tuple[int, ...] | None = None


@dataclass(frozen=True)
class FedMPSQClientResult:
    client_id: int
    num_examples: int
    payload: bytes
    saliency_state: dict[str, torch.Tensor]
    residual_state: dict[str, torch.Tensor]
    class_counts: tuple[int, ...]
    metrics: dict[str, float]
    peak_memory_kind: str


_WORKER_DEVICE: torch.device | None = None
_DATASET_CACHE: dict[tuple[str, str, int], FlowDataset] = {}


def initialize_fedmpsq_worker(device_name: str) -> None:
    global _WORKER_DEVICE
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    torch.set_num_threads(1)
    _WORKER_DEVICE = device


def _client_seed(settings: FedMPSQConfig, server_round: int, client_id: int) -> int:
    return settings.runtime.seed + server_round * 100_000 + client_id


def _seed_client(seed: int, device: torch.device, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def _load_dataset(settings: FedMPSQConfig, client_id: int) -> FlowDataset:
    key = (
        str(Path(settings.data.partitions_dir).resolve()),
        settings.data.source_label_space,
        client_id,
    )
    dataset = _DATASET_CACHE.get(key)
    if dataset is None:
        dataset = load_grouped_npz_dataset(
            client_partition_path(settings.data.partitions_dir, client_id),
            source_label_space=settings.data.source_label_space,
        )
        _DATASET_CACHE[key] = dataset
    return dataset


def _train_subset_and_counts(
    settings: FedMPSQConfig,
    client_id: int,
) -> tuple[Subset | FlowDataset, np.ndarray]:
    dataset = _load_dataset(settings, client_id)
    train_dataset, _ = split_client_dataset(
        dataset,
        val_ratio=settings.data.client_val_ratio,
        seed=settings.data.split_seed + client_id,
    )
    if isinstance(train_dataset, Subset):
        indices = torch.as_tensor(train_dataset.indices, dtype=torch.long)
        labels = dataset.y[indices].numpy()
    else:
        labels = dataset.y.numpy()
    counts = np.bincount(labels, minlength=settings.data.target_num_classes)
    return train_dataset, counts.astype(np.int64)


def bounded_effective_number_weights(
    counts: np.ndarray | torch.Tensor,
    *,
    beta: float,
    w_max: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Eq. (3), normalized on the local sample distribution.

    The reference scale is the support-weighted mean of the effective-number
    weights, not their unweighted mean over present classes. Under CICIoT2023
    the local imbalance ratio reaches 1e6, so the unweighted mean is set by the
    rarest class; every majority class then normalizes far below ``1 / w_max``
    and is clamped to the floor. Because those classes carry essentially all of
    the samples, the batch mean of the class-aware loss collapses to about
    ``1 / w_max`` of plain cross-entropy, which silently divides the gradient --
    and therefore the effective learning rate -- by ``w_max``. Normalizing on
    the sample distribution keeps the objective on the cross-entropy scale, so
    the same learning rate, weight decay, and proximal_mu mean the same thing
    here as they do in the the eight baseline runs baselines, while the bounded reweighting
    ratio of at most ``w_max ** 2`` between two classes is unchanged.
    """
    values = torch.as_tensor(counts, dtype=torch.float64)
    present = values > 0
    weights = torch.zeros_like(values)
    if bool(torch.any(present)):
        support = values[present]
        if beta == 0.0:
            effective = torch.ones_like(support)
        else:
            effective = (1.0 - beta) / (1.0 - torch.pow(beta, support))
        reference = torch.sum(support * effective) / torch.sum(support)
        normalized = effective / reference
        weights[present] = torch.clamp(normalized, 1.0 / w_max, w_max)
    return weights.to(dtype=torch.float32, device=device)


def client_class_weights(
    local_counts: np.ndarray | torch.Tensor,
    pooled_counts: np.ndarray | torch.Tensor | None,
    *,
    prior: str,
    beta: float,
    w_max: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Eq. (3) weights, from either the local or the pooled class distribution.

    The local distribution is the wrong reference for a globally rare class
    that happens to dominate the one client holding it. Mirai-greeth_flood is
    2.1 percent of the pooled training set but 28 percent of its only owner's
    rows, so a local effective-number weight treats it as a majority class at
    the exact client that has to teach it. The pooled counts are already frozen
    in the metric contract, so scoring the weights against them costs nothing
    and only changes which classes the reweighting considers rare.

    Absent classes still receive zero weight, and the surviving weights are
    rescaled on the local support so the objective keeps the cross-entropy
    scale whichever prior is used.
    """
    local = torch.as_tensor(local_counts, dtype=torch.float64)
    if prior == "local":
        return bounded_effective_number_weights(
            local, beta=beta, w_max=w_max, device=device
        )
    if prior != "global":
        raise ValueError(f"Unsupported class prior: {prior}")
    if pooled_counts is None:
        raise ValueError("The global class prior requires pooled class counts")
    pooled = torch.as_tensor(pooled_counts, dtype=torch.float64)
    if pooled.shape != local.shape:
        raise ValueError("Pooled and local class counts have different lengths")
    weights = bounded_effective_number_weights(
        pooled, beta=beta, w_max=w_max
    ).to(torch.float64)
    weights = torch.where(local > 0, weights, torch.zeros_like(weights))
    support = local.sum()
    reference = torch.sum(local * weights) / support if support > 0 else None
    if reference is not None and float(reference) > 0.0:
        weights = weights / reference
    return weights.to(dtype=torch.float32, device=device)


def fedlc_offsets(
    counts: np.ndarray | torch.Tensor,
    *,
    tau: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """FedLC local logit offsets tau times n_c to the power minus one quarter."""
    values = torch.as_tensor(counts, dtype=torch.float32, device=device)
    return tau * torch.clamp(values, min=1.0).pow(-0.25)


def class_aware_per_example_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_name: str,
    class_weights: torch.Tensor,
    calibration_offsets: torch.Tensor,
) -> torch.Tensor:
    if loss_name == "ce":
        return F.cross_entropy(logits, targets, reduction="none")
    if loss_name == "bounded_cb":
        losses = F.cross_entropy(logits, targets, reduction="none")
        return losses * class_weights[targets]
    if loss_name == "fedlc":
        return F.cross_entropy(
            logits - calibration_offsets.unsqueeze(0),
            targets,
            reduction="none",
        )
    if loss_name == "bounded_cb_lc":
        # Eq. (3) reweighting answers "how often is this class seen here"; the
        # FedLC offset answers "how hard did the local softmax push this class
        # down". They are different failures and both are present in this
        # partition: a client that holds none of a class still drives that
        # class's logit toward minus infinity through its own denominator, and
        # under sample-count aggregation the clients that never saw the class
        # can outweigh the one that did. Reweighting alone cannot repair that,
        # because the absent class carries weight zero at exactly the clients
        # doing the suppressing.
        losses = F.cross_entropy(
            logits - calibration_offsets.unsqueeze(0),
            targets,
            reduction="none",
        )
        return losses * class_weights[targets]
    raise ValueError(f"Unsupported FedMPSQ loss: {loss_name}")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_saliency(
    model: nn.Module,
    loader: DataLoader,
    *,
    settings: FedMPSQConfig,
    class_weights: torch.Tensor,
    calibration_offsets: torch.Tensor,
    previous: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Evaluate Eq. (6) on the deterministic complete local training subset."""
    original_module_modes = {
        module: module.training
        for module in model.modules()
    }
    model.train()
    deterministic_eval_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
        nn.FeatureAlphaDropout,
    )
    for module in model.modules():
        if isinstance(module, deterministic_eval_types):
            module.eval()
    model.zero_grad(set_to_none=True)
    num_examples = len(loader.dataset)
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        logits = model(features)
        loss_sum = class_aware_per_example_loss(
            logits,
            targets,
            loss_name=settings.method.loss,
            class_weights=class_weights,
            calibration_offsets=calibration_offsets,
        ).sum()
        (loss_sum / max(num_examples, 1)).backward()

    named_parameters = dict(model.named_parameters())
    current: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if not torch.is_floating_point(tensor):
            continue
        parameter = named_parameters.get(name)
        if parameter is None or parameter.grad is None:
            importance = torch.zeros_like(tensor, dtype=torch.float32, device="cpu")
        else:
            importance = (
                parameter.detach() * parameter.grad.detach()
            ).abs().to(torch.float32).cpu()
        old = previous.get(name)
        if old is None:
            old = torch.zeros_like(importance)
        if old.shape != importance.shape:
            raise ValueError(f"Saliency shape differs for tensor {name}")
        current[name] = (
            settings.method.eta_s * old.detach().cpu().to(torch.float32)
            + (1.0 - settings.method.eta_s) * importance
        )
    model.zero_grad(set_to_none=True)
    for module, was_training in original_module_modes.items():
        module.training = was_training
    return current


def _train_one_client(
    request: FedMPSQBatchRequest,
    spec: FedMPSQClientSpec,
    device: torch.device,
) -> FedMPSQClientResult:
    monitor = RSSMonitor()
    monitor.start()
    try:
        return _train_one_client_impl(request, spec, device, monitor)
    finally:
        monitor.stop()


def _train_one_client_impl(
    request: FedMPSQBatchRequest, spec: FedMPSQClientSpec,
    device: torch.device, rss_monitor: RSSMonitor,
) -> FedMPSQClientResult:
    settings = request.settings
    seed = _client_seed(settings, request.server_round, spec.client_id)
    _seed_client(seed, device, settings.runtime.deterministic)
    train_dataset, counts = _train_subset_and_counts(settings, spec.client_id)
    shuffle_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.algorithm.batch_size,
        shuffle=True,
        generator=shuffle_generator,
        num_workers=settings.runtime.num_workers,
    )
    saliency_loader = DataLoader(
        train_dataset,
        batch_size=settings.algorithm.batch_size,
        shuffle=False,
        num_workers=settings.runtime.num_workers,
    )
    if len(train_loader) == 0:
        raise ValueError(f"Client {spec.client_id} has no local training batches")

    model = build_model(
        settings.model,
        input_dim=request.input_dim,
        num_classes=settings.data.target_num_classes,
    )
    model.load_state_dict(request.global_state, strict=True)
    model.to(device)
    class_weights = client_class_weights(
        counts,
        request.pooled_counts,
        prior=settings.method.class_prior,
        beta=settings.method.beta,
        w_max=settings.method.w_max,
        device=device,
    )
    calibration_offsets = fedlc_offsets(
        counts,
        tau=settings.method.fedlc_tau,
        device=device,
    )
    optimizer = build_optimizer(model, settings.algorithm)
    global_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        peak_memory_kind = "torch_cuda_max_memory_allocated"
    else:
        peak_memory_kind = "process_rss_sampled_10ms_includes_native_tensors"
    _synchronize(device)
    train_started = time.perf_counter()
    totals = {
        "examples": 0,
        "objective": 0.0,
        "task": 0.0,
        "proximal": 0.0,
    }
    effective_saliency = settings.uses_saliency and settings.method.alpha_s > 0.0
    online_saliency = OnlineTaskSaliency(model) if effective_saliency and settings.method.saliency_mode == "online_task" else None
    online_saliency_seconds = 0.0
    model.train()
    for _ in range(settings.algorithm.local_epochs):
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            task_loss = class_aware_per_example_loss(
                logits,
                targets,
                loss_name=settings.method.loss,
                class_weights=class_weights,
                calibration_offsets=calibration_offsets,
            ).mean()
            proximal = torch.zeros((), device=device)
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    proximal = proximal + torch.sum(
                        (parameter - global_parameters[name]) ** 2
                    )
            objective = task_loss + (settings.algorithm.proximal_mu / 2.0) * proximal
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError(
                    "Non-finite local objective for "
                    f"client={spec.client_id}, round={request.server_round}"
                )
            objective.backward()
            if online_saliency is not None:
                _synchronize(device)
                saliency_step_started = time.perf_counter()
                online_saliency.observe(model, global_parameters,
                                        proximal_mu=settings.algorithm.proximal_mu,
                                        batch_size=int(targets.shape[0]))
                _synchronize(device)
                online_saliency_seconds += time.perf_counter() - saliency_step_started
            optimizer.step()
            batch_size = int(targets.shape[0])
            totals["examples"] += batch_size
            totals["objective"] += float(objective.detach().cpu()) * batch_size
            totals["task"] += float(task_loss.detach().cpu()) * batch_size
            totals["proximal"] += float(proximal.detach().cpu()) * batch_size
    _synchronize(device)
    train_seconds = time.perf_counter() - train_started

    saliency_started = time.perf_counter()
    if online_saliency is not None:
        saliency_state = online_saliency.finish(model, spec.saliency_state, eta=settings.method.eta_s)
    elif effective_saliency:
        saliency_state = _measure_saliency(
            model,
            saliency_loader,
            settings=settings,
            class_weights=class_weights,
            calibration_offsets=calibration_offsets,
            previous=spec.saliency_state,
            device=device,
        )
    else:
        saliency_state = {
            name: spec.saliency_state.get(name, torch.zeros_like(value)).detach().cpu().clone()
            for name, value in model.state_dict().items()
            if torch.is_floating_point(value)
        }
    _synchronize(device)
    saliency_seconds = time.perf_counter() - saliency_started

    local_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    # Buffers such as the BatchNorm running statistics are not gradients, and
    # their round-to-round deltas are far larger than any weight delta, so they
    # are transmitted densely instead of competing in the Top-K score.
    trainable_names = {name for name, _ in model.named_parameters()}
    dense_names = frozenset(
        name
        for name, value in local_state.items()
        if torch.is_floating_point(value) and name not in trainable_names
    )
    update = floating_state_delta(local_state, request.global_state)
    residual = spec.residual_state or zeros_like_floating_state(local_state)
    protected_groups = minority_head_groups(
        model, counts, request.pooled_counts, settings.data.minority_fraction
    ) if settings.method.minority_head_reserve > 0 else ()
    tensor_bits = ({name: 8 for name, value in update.items() if value.ndim <= 1}
                   if settings.method.protect_small_tensors else None)
    if settings.method.flat_uplink:
        # One flat vector on the wire. The per-tensor layout and the per-tensor
        # scale headers are static description that both endpoints already hold,
        # so re-sending them per message only spends budget that could carry
        # coordinates. Nothing about selection or quantization changes.
        check_flat_uplink_is_sound(dense_names=dense_names,
                                   protected_groups=protected_groups,
                                   tensor_bits=tensor_bits)
        wire = wire_layout(local_state)
        update = flatten_state(update, wire)
        residual = residual if FLAT_KEY in residual else flatten_state(residual, wire)
        saliency_state = flatten_state(saliency_state, wire) if saliency_state else saliency_state
        dense_names = frozenset()
    compression_started = time.perf_counter()
    compression = compress_update(
        update,
        saliency_state,
        residual,
        uses_sparse=settings.uses_sparse,
        uses_saliency=effective_saliency,
        uses_int8=settings.uses_int8,
        uses_error_feedback=settings.uses_error_feedback,
        sparsity=round_sparsity(settings.method.sparsity, settings.method.sparsity_initial,
                                settings.method.sparsity_warmup_rounds, request.server_round),
        alpha_s=settings.method.alpha_s,
        epsilon=settings.method.epsilon,
        client_id=spec.client_id,
        server_round=request.server_round,
        num_examples=len(train_dataset),
        dense_names=dense_names,
        quant_bits=settings.method.quant_bits,
        index_codec=settings.method.index_codec,
        stochastic_rounding=settings.method.stochastic_rounding,
        group_size=settings.method.quant_group_size,
        clipping=settings.method.quant_clipping,
        tensor_bits=tensor_bits,
        adaptive_error=settings.method.adaptive_quant_error,
        rounding_seed=settings.runtime.seed,
        compact_layout=settings.method.compact_layout,
        block_size=settings.method.block_size,
        protected_groups=protected_groups,
        reserve_fraction=settings.method.minority_head_reserve,
        rotate=settings.method.incoherent_rotation,
        scale_codec=settings.method.scale_codec,
        quantizer=settings.method.quantizer,
        uplink_budget_bytes=settings.method.uplink_budget_bytes,
    )
    compression_seconds = time.perf_counter() - compression_started
    process_peak = rss_monitor.stop()
    if device.type == "cuda":
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory_bytes = process_peak

    denominator = max(int(totals["examples"]), 1)
    metrics = {
        "logit_calibration_enabled": float(settings.method.loss in {"fedlc", "bounded_cb_lc"}),
        "fedlc_offset_max": float(calibration_offsets.max()),
        "minority_protected_class_count": float(len(protected_groups)),
        "minority_head_reserve_fraction": settings.method.minority_head_reserve,
        "sparsification_block_size": float(settings.method.block_size),
        "train_objective": totals["objective"] / denominator,
        "train_class_aware_loss": totals["task"] / denominator,
        "train_proximal_squared_norm": totals["proximal"] / denominator,
        "train_seconds": float(train_seconds),
        "saliency_seconds": float(saliency_seconds),
        "online_saliency_seconds_included_in_train": float(online_saliency_seconds),
        "effective_sparsity": round_sparsity(settings.method.sparsity, settings.method.sparsity_initial,
                                             settings.method.sparsity_warmup_rounds, request.server_round),
        "compression_seconds": float(compression_seconds),
        "peak_memory_bytes": float(peak_memory_bytes),
        "process_peak_rss_bytes": float(process_peak),
        "process_rss_before_training_bytes": float(rss_monitor.start_bytes),
        "resident_model_bytes": float(sum(v.numel() * v.element_size() for v in local_state.values())),
        "residual_state_bytes": float(sum(v.numel() * v.element_size() for v in compression.residual_state.values())),
        "saliency_state_bytes": float(sum(v.numel() * v.element_size() for v in saliency_state.values())),
        "class_weight_min_present": float(class_weights[class_weights > 0].min())
        if bool(torch.any(class_weights > 0))
        else 0.0,
        "class_weight_max_present": float(class_weights.max()),
        **compression.metrics,
    }
    del model
    return FedMPSQClientResult(
        client_id=spec.client_id,
        num_examples=len(train_dataset),
        payload=compression.payload.data,
        saliency_state=clone_tensor_state(saliency_state),
        residual_state=clone_tensor_state(compression.residual_state),
        class_counts=tuple(int(value) for value in counts),
        metrics=metrics,
        peak_memory_kind=peak_memory_kind,
    )


def train_fedmpsq_batch(
    request: FedMPSQBatchRequest,
) -> list[FedMPSQClientResult]:
    if _WORKER_DEVICE is None:
        raise RuntimeError("FedMPSQ worker was not initialized")
    return [
        _train_one_client(request, spec, _WORKER_DEVICE)
        for spec in request.clients
    ]


def train_fedmpsq_batch_local(
    request: FedMPSQBatchRequest,
    device: torch.device,
) -> list[FedMPSQClientResult]:
    return [_train_one_client(request, spec, device) for spec in request.clients]
