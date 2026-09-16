"""Experiment configuration loading, normalization, and fidelity guards."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from data.dataset import load_metadata
from data.partition_manifest import require_supported_client_count
from models import SUPPORTED_MODELS
from models.dcnn_bilstm import SUPPORTED_NORMS

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


FIXED_DATA_SPLIT_SEED = 42


@dataclass(frozen=True)
class AlgorithmConfig:
    """Federated optimizer and local training settings."""

    name: str = "fedavg"
    num_server_rounds: int = 20
    fraction_train: float = 1.0
    fraction_evaluate: float = 1.0
    min_train_nodes: int = 10
    min_evaluate_nodes: int = 10
    min_available_nodes: int = 10
    local_epochs: int = 1
    batch_size: int = 256
    learning_rate: float = 0.001
    optimizer: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 0.0001
    proximal_mu: float = 0.0


@dataclass(frozen=True)
class DataConfig:
    """Immutable prepared-partition settings."""

    partitions_dir: str = "data/partitions/ciciot2023-redpacket"
    num_clients: int = 10
    num_classes: int = 34
    client_val_ratio: float = 0.2
    seed: int = 42
    dataset: str = "existing_dataset"
    partition_file: str = "data_partitions/client_partition.json"
    regenerate_partition: bool = False
    source_label_space: str = "identity"
    # Fraction of the rarest classes, by pooled local-training support, that the
    # frozen evaluation contract scores as minority. Shared with FedMPSQ.
    minority_fraction: float = 0.25


@dataclass(frozen=True)
class ModelConfig:
    """DCNN-BiLSTM model settings."""

    name: str = "dcnn_bilstm"
    hidden_size: int = 320
    num_blocks: int = 1
    input_dim: int | None = 46
    num_classes: int = 34
    conv_channels: tuple[int, ...] = (64, 128, 128)
    kernel_size: int = 3
    lstm_hidden_size: int = 128
    lstm_layers: int = 1
    dropout: float = 0.2
    # Normalisation inside the convolution blocks. "batch" keeps running
    # statistics that federated averaging must aggregate; "group" and "layer"
    # carry none, which removes that failure mode entirely.
    norm: str = "batch"


@dataclass(frozen=True)
class DistillationConfig:
    """BDD-HFL bidirectional decoupled distillation settings."""

    enabled: bool = False
    method: str = "none"
    temperature: float = 4.0
    target_class_weight: float = 1.0
    non_target_class_weight: float = 8.0
    private_kd_weight: float = 0.5
    global_kd_weight: float = 0.5
    private_model: str = "same_architecture"
    gradient_clip_norm: float | None = None
    scheduler_step_size: int = 1
    scheduler_gamma: float = 1.0


@dataclass(frozen=True)
class PruningConfig:
    """FAP adaptive data-pruning and differential-privacy settings."""

    enabled: bool = False
    method: str = "none"
    status: str = "disabled"
    blocked_reason: str = ""
    confidence_threshold: float = 0.95
    fisher_threshold: float = 10.0
    target_samples: int = 500
    target_fraction: float = 0.0
    minimum_target_fraction: float = 0.1
    warmup_rounds: int = 0
    max_pruning_fraction_per_round: float = 1.0
    gradient_clip_norm: float = 1.0
    dp_enabled: bool = True
    dp_mechanism: str = "laplace"
    dp_epsilon: float = 100.0


@dataclass(frozen=True)
class QuantizationConfig:
    """FedPAQ or DAdaQuant update-quantization settings."""

    enabled: bool = False
    method: str = "none"
    levels: int = 1
    stochastic: bool = True
    difference_coding: bool = True
    zero_run_length_encoding: bool = False
    elias_omega_encoding: bool = False
    time_adaptive: bool = False
    client_adaptive: bool = False
    min_level: int = 1
    max_level: int = 1
    moving_average: float = 0.9
    convergence_interval: int = 2


@dataclass(frozen=True)
class TechniqueConfig:
    """Exactly one optional paper technique per experiment."""

    name: str = "none"
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)


@dataclass(frozen=True)
class FidelityConfig:
    """Declared paper-fidelity class and execution state."""

    classification: str = "B"
    status: str = "ready"
    paper: str = "FedAvg"
    official_source: str = ""
    notes: str = ""


@dataclass(frozen=True)
class ResultsConfig:
    """Result output settings."""

    dir: str = "results/fedavg"
    run_name: str = "fedavg-redpacket"
    save_model: bool = True
    save_round_checkpoints: bool = True


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime settings."""

    device: str = "auto"
    parallel_clients: int = 1
    seed: int = 42


@dataclass(frozen=True)
class ExperimentSettings:
    """Complete experiment settings."""

    experiment_id: str = "FedAvg"
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    technique: TechniqueConfig = field(default_factory=TechniqueConfig)
    fidelity: FidelityConfig = field(default_factory=FidelityConfig)
    results: ResultsConfig = field(default_factory=ResultsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_id": "FedAvg",
    "algorithm": {
        "name": "fedavg",
        "num_server_rounds": 20,
        "fraction_train": 1.0,
        "fraction_evaluate": 1.0,
        "min_train_nodes": 10,
        "min_evaluate_nodes": 10,
        "min_available_nodes": 10,
        "local_epochs": 1,
        "batch_size": 256,
        "learning_rate": 0.001,
        "optimizer": "sgd",
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "proximal_mu": 0.0,
    },
    "data": {
        "partitions_dir": "data/partitions/ciciot2023-redpacket",
        "num_clients": 10,
        "num_classes": 34,
        "client_val_ratio": 0.2,
        "seed": 42,
        "dataset": "existing_dataset",
        "partition_file": "data_partitions/client_partition.json",
        "regenerate_partition": False,
        "source_label_space": "identity",
        "minority_fraction": 0.25,
    },
    "model": {
        "name": "dcnn_bilstm",
        "hidden_size": 320,
        "num_blocks": 1,
        "input_dim": 46,
        "num_classes": 34,
        "conv_channels": "64,128,128",
        "kernel_size": 3,
        "lstm_hidden_size": 128,
        "lstm_layers": 1,
        "dropout": 0.2,
        "norm": "batch",
    },
    "technique": {
        "name": "none",
        "distillation": {
            "enabled": False,
            "method": "none",
            "temperature": 4.0,
            "target_class_weight": 1.0,
            "non_target_class_weight": 8.0,
            "private_kd_weight": 0.5,
            "global_kd_weight": 0.5,
            "private_model": "same_architecture",
            "gradient_clip_norm": None,
            "scheduler_step_size": 1,
            "scheduler_gamma": 1.0,
        },
        "pruning": {
            "enabled": False,
            "method": "none",
            "status": "disabled",
            "blocked_reason": "",
            "confidence_threshold": 0.95,
            "fisher_threshold": 10.0,
            "target_samples": 500,
            "target_fraction": 0.0,
            "minimum_target_fraction": 0.1,
            "warmup_rounds": 0,
            "max_pruning_fraction_per_round": 1.0,
            "gradient_clip_norm": 1.0,
            "dp_enabled": True,
            "dp_mechanism": "laplace",
            "dp_epsilon": 100.0,
        },
        "quantization": {
            "enabled": False,
            "method": "none",
            "levels": 1,
            "stochastic": True,
            "difference_coding": True,
            "zero_run_length_encoding": False,
            "elias_omega_encoding": False,
            "time_adaptive": False,
            "client_adaptive": False,
            "min_level": 1,
            "max_level": 1,
            "moving_average": 0.9,
            "convergence_interval": 2,
        },
    },
    "fidelity": {
        "classification": "B",
        "status": "ready",
        "paper": "FedAvg",
        "official_source": "",
        "notes": "",
    },
    "results": {
        "dir": "results/fedavg",
        "run_name": "fedavg-redpacket",
        "save_model": True,
        "save_round_checkpoints": True,
    },
    "runtime": {"device": "auto", "parallel_clients": 1, "seed": 42},
}


FLAT_KEY_MAP: dict[str, tuple[str, ...]] = {
    "algorithm": ("algorithm", "name"),
    "num-server-rounds": ("algorithm", "num_server_rounds"),
    "fraction-train": ("algorithm", "fraction_train"),
    "fraction-evaluate": ("algorithm", "fraction_evaluate"),
    "min-train-nodes": ("algorithm", "min_train_nodes"),
    "min-evaluate-nodes": ("algorithm", "min_evaluate_nodes"),
    "min-available-nodes": ("algorithm", "min_available_nodes"),
    "local-epochs": ("algorithm", "local_epochs"),
    "batch-size": ("algorithm", "batch_size"),
    "learning-rate": ("algorithm", "learning_rate"),
    "optimizer": ("algorithm", "optimizer"),
    "momentum": ("algorithm", "momentum"),
    "weight-decay": ("algorithm", "weight_decay"),
    "proximal-mu": ("algorithm", "proximal_mu"),
    "mu": ("algorithm", "proximal_mu"),
    "data-dir": ("data", "partitions_dir"),
    "partitions-dir": ("data", "partitions_dir"),
    "num-clients": ("data", "num_clients"),
    "num-classes": ("model", "num_classes"),
    "client-val-ratio": ("data", "client_val_ratio"),
    "input-dim": ("model", "input_dim"),
    "results-dir": ("results", "dir"),
    "run-name": ("results", "run_name"),
    "save-model": ("results", "save_model"),
    "save-round-checkpoints": ("results", "save_round_checkpoints"),
    "device": ("runtime", "device"),
    "parallel-clients": ("runtime", "parallel_clients"),
    "seed": ("runtime", "seed"),
}

ALIASES = {
    "mu": "proximal_mu",
    "lr": "learning_rate",
    "learning-rate": "learning_rate",
    "input-dim": "input_dim",
    "num-classes": "num_classes",
    "num-server-rounds": "num_server_rounds",
    "local-epochs": "local_epochs",
    "batch-size": "batch_size",
    "partitions-dir": "partitions_dir",
    "partition-file": "partition_file",
    "regenerate-partition": "regenerate_partition",
    "client-val-ratio": "client_val_ratio",
    "save-model": "save_model",
    "save-round-checkpoints": "save_round_checkpoints",
    "run-name": "run_name",
    "parallel-clients": "parallel_clients",
}


def _deep_update(target: dict[str, Any], incoming: Mapping[str, Any]) -> None:
    for key, value in incoming.items():
        normalized_key = str(key).replace("-", "_")
        if isinstance(value, Mapping) and isinstance(target.get(normalized_key), dict):
            _deep_update(target[normalized_key], value)
        else:
            target[normalized_key] = value


def _set_path(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = config
    for part in path[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[path[-1]] = value


def _normalize_part(part: str) -> str:
    return ALIASES.get(part, part).replace("-", "_")


def _apply_run_overrides(config: dict[str, Any], run_config: Mapping[str, Any]) -> None:
    for raw_key, value in run_config.items():
        key = str(raw_key)
        if key == "config-path":
            continue
        section = key.replace("-", "_")
        if isinstance(value, Mapping) and section in config:
            _deep_update(config[section], value)
            continue
        if key in FLAT_KEY_MAP:
            _set_path(config, FLAT_KEY_MAP[key], value)
            continue
        if "." in key:
            parts = tuple(_normalize_part(part) for part in key.split("."))
            if parts and parts[0] in config:
                _set_path(config, parts, value)


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return dict(payload or {})
    if config_path.suffix.lower() == ".toml":
        with config_path.open("rb") as handle:
            return tomllib.load(handle)
    raise ValueError("Config path must use .yaml, .yml, or .toml")


def _settings_from_dict(config: Mapping[str, Any]) -> ExperimentSettings:
    model = dict(config["model"])
    conv_channels = model["conv_channels"]
    if isinstance(conv_channels, str):
        model["conv_channels"] = tuple(
            int(value.strip()) for value in conv_channels.split(",") if value.strip()
        )
    else:
        model["conv_channels"] = tuple(int(value) for value in conv_channels)
    technique = dict(config["technique"])
    return ExperimentSettings(
        experiment_id=str(config["experiment_id"]),
        algorithm=AlgorithmConfig(**config["algorithm"]),
        data=DataConfig(**config["data"]),
        model=ModelConfig(**model),
        technique=TechniqueConfig(
            name=technique["name"],
            distillation=DistillationConfig(**technique["distillation"]),
            pruning=PruningConfig(**technique["pruning"]),
            quantization=QuantizationConfig(**technique["quantization"]),
        ),
        fidelity=FidelityConfig(**config["fidelity"]),
        results=ResultsConfig(**config["results"]),
        runtime=RuntimeConfig(**config["runtime"]),
    )


def _validate_technique(settings: ExperimentSettings) -> None:
    technique = settings.technique
    enabled = {
        "distillation": technique.distillation.enabled,
        "pruning": technique.pruning.enabled,
        "quantization": technique.quantization.enabled,
    }
    if sum(enabled.values()) > 1:
        raise ValueError("Each experiment may enable only one additional technique")
    expected_name = next((name for name, active in enabled.items() if active), "none")
    if technique.name != expected_name:
        raise ValueError(
            f"technique.name={technique.name!r} does not match enabled technique {expected_name!r}"
        )
    if technique.distillation.enabled:
        cfg = technique.distillation
        if cfg.method != "bdd_hfl":
            raise ValueError("Distillation scenarios must use method 'bdd_hfl'")
        if cfg.temperature <= 0.0:
            raise ValueError("BDD-HFL temperature must be > 0")
        if min(cfg.target_class_weight, cfg.non_target_class_weight) < 0.0:
            raise ValueError("BDD-HFL decoupled loss weights must be non-negative")
        if min(cfg.private_kd_weight, cfg.global_kd_weight) < 0.0:
            raise ValueError("BDD-HFL KD weights must be non-negative")
    if technique.pruning.enabled:
        cfg = technique.pruning
        if cfg.method != "fap":
            raise ValueError("Pruning scenarios must use method 'fap'")
        if cfg.status != "ready":
            raise RuntimeError(cfg.blocked_reason or "FAP config is not ready")
        if not 0.0 <= cfg.confidence_threshold <= 1.0:
            raise ValueError("FAP confidence_threshold must be in [0, 1]")
        if cfg.fisher_threshold < 0.0:
            raise ValueError("FAP fisher_threshold must be non-negative")
        if cfg.target_samples <= 0:
            raise ValueError("FAP target_samples must be positive")
        if not 0.0 <= cfg.target_fraction <= 1.0:
            raise ValueError("FAP target_fraction must be in [0, 1]")
        if not 0.0 < cfg.minimum_target_fraction <= 1.0:
            raise ValueError("FAP minimum_target_fraction must be in (0, 1]")
        if cfg.warmup_rounds < 0:
            raise ValueError("FAP warmup_rounds must be non-negative")
        if not 0.0 < cfg.max_pruning_fraction_per_round <= 1.0:
            raise ValueError(
                "FAP max_pruning_fraction_per_round must be in (0, 1]"
            )
        if cfg.gradient_clip_norm <= 0.0:
            raise ValueError("FAP gradient_clip_norm must be positive")
        if not cfg.dp_enabled or cfg.dp_mechanism != "laplace":
            raise ValueError("Paper-faithful FAP requires Laplace differential privacy")
        if cfg.dp_epsilon <= 0.0:
            raise ValueError("FAP dp_epsilon must be positive")
    if technique.quantization.enabled:
        cfg = technique.quantization
        expected_method = (
            "fedpaq" if settings.algorithm.name == "fedavg" else "dadaquant"
        )
        if cfg.method != expected_method:
            raise ValueError(
                f"{settings.algorithm.name} quantization must use {expected_method}"
            )
        if cfg.levels < 1 or cfg.min_level < 1 or cfg.max_level < cfg.min_level:
            raise ValueError("Quantization levels must satisfy 1 <= min <= max")
        if cfg.method == "fedpaq" and (cfg.time_adaptive or cfg.client_adaptive):
            raise ValueError("FedPAQ config cannot enable DAdaQuant adaptations")
        if cfg.method == "dadaquant" and not (
            cfg.time_adaptive and cfg.client_adaptive
        ):
            raise ValueError("DAdaQuant requires both time and client adaptation")


def load_settings(run_config: Mapping[str, Any] | None = None) -> ExperimentSettings:
    """Load defaults, a YAML/TOML profile, and explicit run overrides."""
    run_config = dict(run_config or {})
    config = copy.deepcopy(DEFAULT_CONFIG)
    config_path = run_config.get("config-path")
    if config_path:
        _deep_update(config, _load_config(config_path))
    _apply_run_overrides(config, run_config)
    settings = _settings_from_dict(config)
    if settings.algorithm.name not in {"fedavg", "fedprox"}:
        raise ValueError("algorithm.name must be either 'fedavg' or 'fedprox'")
    if not 0.0 < settings.algorithm.fraction_train <= 1.0:
        raise ValueError("algorithm.fraction_train must be in (0, 1]")
    if settings.algorithm.min_train_nodes <= 0:
        raise ValueError("algorithm.min_train_nodes must be > 0")
    if settings.algorithm.name == "fedprox" and settings.algorithm.proximal_mu < 0:
        raise ValueError("FedProx requires algorithm.proximal_mu >= 0")
    require_supported_client_count(settings.data.num_clients)
    for name, value in (
        ("min_train_nodes", settings.algorithm.min_train_nodes),
        ("min_evaluate_nodes", settings.algorithm.min_evaluate_nodes),
        ("min_available_nodes", settings.algorithm.min_available_nodes),
    ):
        if int(value) != int(settings.data.num_clients):
            raise ValueError(
                f"algorithm.{name} must equal data.num_clients "
                f"({settings.data.num_clients}); got {value}"
            )
    if settings.data.dataset != "existing_dataset":
        raise ValueError("data.dataset must remain 'existing_dataset'")
    if settings.data.regenerate_partition:
        raise ValueError("data.regenerate_partition must remain false")
    if settings.data.seed != FIXED_DATA_SPLIT_SEED:
        raise ValueError(
            f"data.seed must remain {FIXED_DATA_SPLIT_SEED} so every training seed "
            "uses the exact same train/validation assignment"
        )
    if settings.data.source_label_space not in {"identity", "ciciot2023_34"}:
        raise ValueError("Unsupported data.source_label_space")
    if not 0.0 < settings.data.minority_fraction <= 1.0:
        raise ValueError("data.minority_fraction must be in (0, 1]")
    if settings.model.name not in SUPPORTED_MODELS:
        raise ValueError(f"model.name must be one of {SUPPORTED_MODELS}")
    if settings.model.norm not in SUPPORTED_NORMS:
        raise ValueError(f"model.norm must be one of {SUPPORTED_NORMS}")
    if (
        settings.data.source_label_space == "ciciot2023_34"
        and (settings.data.num_classes != 8 or settings.model.num_classes != 8)
    ):
        raise ValueError(
            "ciciot2023_34 grouped-label tasks require data/model num_classes=8"
        )
    if settings.runtime.parallel_clients <= 0:
        raise ValueError("runtime.parallel_clients must be positive")
    if settings.fidelity.classification not in {"A", "B", "C"}:
        raise ValueError("fidelity.classification must be A, B, or C")
    if settings.fidelity.status == "blocked":
        raise RuntimeError(settings.fidelity.notes or "Experiment is blocked")
    if settings.fidelity.classification == "C":
        raise RuntimeError("Paper-inspired adaptation C requires explicit user approval")
    _validate_technique(settings)
    return settings


def resolve_dimensions(settings: ExperimentSettings) -> tuple[int, int]:
    """Resolve input dimension and number of classes from metadata or config."""
    try:
        metadata = load_metadata(settings.data.partitions_dir)
    except FileNotFoundError:
        metadata = {}
    input_dim = metadata.get("input_dim", settings.model.input_dim)
    # Prepared metadata describes the source NPZ label space. A task-aware
    # mapping owns the model output dimension and must not be overridden by 34.
    num_classes = (
        settings.model.num_classes
        if settings.data.source_label_space != "identity"
        else metadata.get("num_classes", settings.model.num_classes)
    )
    if input_dim is None:
        raise ValueError(
            "model.input_dim is not set and metadata.json is unavailable. "
            "Prepare data first or set model.input_dim."
        )
    return int(input_dim), int(num_classes)


def select_device(device_setting: str) -> torch.device:
    """Select a torch device from config."""
    if device_setting == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_setting)


def select_client_devices(
    device_setting: str,
    parallel_clients: int,
) -> list[torch.device]:
    """Resolve one dedicated CUDA device per parallel client worker."""
    if parallel_clients <= 0:
        raise ValueError("parallel_clients must be positive")
    if parallel_clients == 1:
        return [select_device(device_setting)]
    if device_setting not in {"auto", "cuda", "cuda:0"}:
        raise ValueError(
            "Multi-GPU client training requires --device auto, cuda, or cuda:0"
        )
    device_count = torch.cuda.device_count()
    if device_count < parallel_clients:
        raise RuntimeError(
            f"Requested {parallel_clients} parallel CUDA clients, but only "
            f"{device_count} CUDA device(s) are visible"
        )
    return [torch.device(f"cuda:{index}") for index in range(parallel_clients)]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
