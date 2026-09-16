"""Configuration for the isolated FedMPSQ MVP pipeline.

This module intentionally does not extend fl.config. The latter is the locked
the eight baseline runs protocol and must remain unchanged while the proposal is developed.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from data.partition_manifest import require_supported_client_count
from models import SUPPORTED_MODELS
from models.dcnn_bilstm import SUPPORTED_NORMS


SUPPORTED_SEEDS = (42, 43, 44)
# The MVP froze the eight-group task; the 34-class scale study reuses the
# identical method code on the untouched CICIoT2023 label space.
SUPPORTED_TARGET_NUM_CLASSES = (8, 34)
# Source IDs stored in the prepared NPZ -> the task the model is trained on.
TARGET_NUM_CLASSES_BY_LABEL_SPACE = {
    "ciciot2023_34": 8,
    "ciciot2023_groups_8": 8,
    "ciciot2023_identity_34": 34,
}
SUPPORTED_SOURCE_LABEL_SPACES = tuple(TARGET_NUM_CLASSES_BY_LABEL_SPACE)
# The screening grid of the S03/S05 stages. Those results are frozen, and the
# scripts that reproduce them enumerate this tuple, so it must not grow.
MVP_SPARSITIES = (0.5, 0.75, 0.9)
# What a config may declare, which is a wider set than any one stage sweeps.
# The compression study needs 0.95--0.99: below 0.95 the index dominates the
# payload however the values are coded, so the byte budget only starts moving
# once the mask itself is sparse enough for the gap code to pay for itself.
SUPPORTED_SPARSITIES = MVP_SPARSITIES + (0.95, 0.98, 0.99)
MVP_QUANT_BITS = (2, 4, 8, 32)
MVP_INDEX_CODECS = ("bitmap", "rice", "auto_runs")
MVP_ABLATIONS = ("a0", "a1", "a2", "a3", "a4", "a5")


@dataclass(frozen=True)
class FedMPSQAlgorithmConfig:
    num_server_rounds: int = 20
    num_clients: int = 10
    fraction_train: float = 1.0
    local_epochs: int = 1
    batch_size: int = 256
    learning_rate: float = 0.001
    optimizer: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 0.0001
    proximal_mu: float = 0.01
    aggregation: str = "sample"


@dataclass(frozen=True)
class FedMPSQDataConfig:
    partitions_dir: str = "data/partitions/ciciot2023-redpacket"
    partition_file: str = "data_partitions/client_partition.json"
    dataset: str = "existing_dataset"
    source_label_space: str = "ciciot2023_34"
    target_num_classes: int = 8
    client_val_ratio: float = 0.2
    split_seed: int = 42
    num_clients: int = 10
    minority_fraction: float = 0.25

    @property
    def seed(self) -> int:
        '''Compatibility alias for the immutable partition verifier.'''
        return self.split_seed

    @property
    def regenerate_partition(self) -> bool:
        return False


@dataclass(frozen=True)
class FedMPSQModelConfig:
    name: str = "dcnn_bilstm"
    hidden_size: int = 320
    num_blocks: int = 1
    input_dim: int | None = 46
    num_classes: int = 8
    conv_channels: tuple[int, ...] = (64, 128, 128)
    kernel_size: int = 3
    lstm_hidden_size: int = 128
    lstm_layers: int = 1
    dropout: float = 0.2
    norm: str = "batch"


@dataclass(frozen=True)
class FedMPSQMethodConfig:
    """The method switchboard; only fixed-MVP combinations are accepted."""

    ablation: str = "a5"
    loss: str = "bounded_cb"
    beta: float = 0.9999
    w_max: float = 10.0
    fedlc_tau: float = 1.0
    eta_s: float = 0.9
    alpha_s: float = 0.5
    sparsity: float = 0.75
    normalization: str = "max"
    quant_bits: int = 8
    index_codec: str = "rice"
    stochastic_rounding: bool = False
    error_feedback: bool = True
    epsilon: float = 1.0e-12
    class_prior: str = "local"
    quant_group_size: int = 0
    quant_clipping: str = "max"
    protect_small_tensors: bool = False
    adaptive_quant_error: float | None = None
    downlink_bits: int = 32
    sparsity_warmup_rounds: int = 0
    sparsity_initial: float = 0.0
    saliency_mode: str = "full_gradient"
    compact_layout: bool = False
    # Send the update as one flat vector instead of the named state_dict.
    # The tensor layout is static and both endpoints hold it, so describing
    # it per message only spends budget that could carry coordinates.
    flat_uplink: bool = False
    block_size: int = 1
    minority_head_reserve: float = 0.0
    # A hard ceiling on the complete serialized client message. It is the
    # only setting that makes two arms comparable at equal cost, because
    # a density target does not fix the byte count once the index code,
    # the scale stream and the value width all move.
    uplink_budget_bytes: int | None = None
    scale_codec: str = "fp32"
    incoherent_rotation: bool = False
    quantizer: str = "integer"


@dataclass(frozen=True)
class FedMPSQRuntimeConfig:
    device: str = "auto"
    parallel_clients: int = 1
    seed: int = 42
    deterministic: bool = True
    num_workers: int = 0


@dataclass(frozen=True)
class FedMPSQResultsConfig:
    dir: str = "results/fedmpsq"
    run_name: str = "fedmpsq_a5_seed42"
    save_round_checkpoints: bool = True
    save_model: bool = True
    resume: str | None = None
    target_macro_f1: float | None = None


@dataclass(frozen=True)
class FedMPSQConfig:
    experiment_id: str = "FedMPSQ-MVP"
    algorithm: FedMPSQAlgorithmConfig = field(default_factory=FedMPSQAlgorithmConfig)
    data: FedMPSQDataConfig = field(default_factory=FedMPSQDataConfig)
    model: FedMPSQModelConfig = field(default_factory=FedMPSQModelConfig)
    method: FedMPSQMethodConfig = field(default_factory=FedMPSQMethodConfig)
    runtime: FedMPSQRuntimeConfig = field(default_factory=FedMPSQRuntimeConfig)
    results: FedMPSQResultsConfig = field(default_factory=FedMPSQResultsConfig)
    fidelity_classification: str = "P"
    protocol_version: str = "fedmpsq-mvp-v1"

    @property
    def uses_balance(self) -> bool:
        return self.method.ablation != "a0"

    @property
    def uses_sparse(self) -> bool:
        return self.method.ablation in {"a2", "a3", "a4", "a5"}

    @property
    def uses_saliency(self) -> bool:
        return self.method.ablation in {"a3", "a4", "a5"}

    @property
    def uses_int8(self) -> bool:
        # Historical name: this now means packed low-bit transport.
        return self.method.ablation in {"a4", "a5"} and self.method.quant_bits != 32

    @property
    def uses_error_feedback(self) -> bool:
        return self.method.ablation == "a5"

    @property
    def config_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_json(self) -> str:
        return json.dumps(self.config_dict, sort_keys=True, separators=(",", ":"))


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_id": "FedMPSQ-MVP",
    "algorithm": asdict(FedMPSQAlgorithmConfig()),
    "data": asdict(FedMPSQDataConfig()),
    "model": {
        **asdict(FedMPSQModelConfig()),
        "conv_channels": "64,128,128",
        "hidden_size": 320,
        "num_blocks": 1,
    },
    "method": asdict(FedMPSQMethodConfig()),
    "runtime": asdict(FedMPSQRuntimeConfig()),
    "results": asdict(FedMPSQResultsConfig()),
    "fidelity_classification": "P",
    "protocol_version": "fedmpsq-mvp-v1",
}


def _deep_update(target: dict[str, Any], incoming: Mapping[str, Any]) -> None:
    for raw_key, value in incoming.items():
        key = str(raw_key).replace("-", "_")
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    parts = [part.replace("-", "_") for part in key.split(".")]
    cursor = config
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            raise KeyError(f"Unknown FedMPSQ override path: {key}")
        cursor = cursor[part]
    if parts[-1] not in cursor:
        raise KeyError(f"Unknown FedMPSQ override key: {key}")
    cursor[parts[-1]] = value


def _load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"FedMPSQ config not found: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("FedMPSQ YAML root must be a mapping")
    return dict(payload)


def _coerce(config: dict[str, Any]) -> FedMPSQConfig:
    model = dict(config["model"])
    channels = model.get("conv_channels", (64, 128, 128))
    if isinstance(channels, str):
        channels = tuple(int(item.strip()) for item in channels.split(",") if item.strip())
    else:
        channels = tuple(int(item) for item in channels)
    model["conv_channels"] = channels
    return FedMPSQConfig(
        experiment_id=str(config["experiment_id"]),
        algorithm=FedMPSQAlgorithmConfig(**dict(config["algorithm"])),
        data=FedMPSQDataConfig(**dict(config["data"])),
        model=FedMPSQModelConfig(**model),
        method=FedMPSQMethodConfig(**dict(config["method"])),
        runtime=FedMPSQRuntimeConfig(**dict(config["runtime"])),
        results=FedMPSQResultsConfig(**dict(config["results"])),
        fidelity_classification=str(config.get("fidelity_classification", "P")),
        protocol_version=str(config.get("protocol_version", "fedmpsq-mvp-v1")),
    )


def validate_fedmpsq_config(settings: FedMPSQConfig) -> None:
    """Fail closed on combinations outside the guide's MVP."""
    if settings.fidelity_classification != "P":
        raise ValueError("FedMPSQ must use proposed-method fidelity class P")
    if settings.protocol_version not in {
        "fedmpsq-mvp-v1",
        "fedmpsq-lowbit-v2",
        "fedmpsq-rare-v3",
        # The uplink codec study: block-structured selection, ternary values,
        # byte-coded group scales and incoherence processing, all screened at
        # a fixed serialized byte cap rather than at a fixed density.
        "fedmpsq-uplink-v4",
    }:
        raise ValueError("Unsupported FedMPSQ protocol version")
    if settings.algorithm.num_clients != settings.data.num_clients:
        raise ValueError(
            "algorithm.num_clients and data.num_clients must agree; got "
            f"{settings.algorithm.num_clients} and {settings.data.num_clients}"
        )
    require_supported_client_count(settings.data.num_clients)
    if settings.data.target_num_classes != settings.model.num_classes:
        raise ValueError("data.target_num_classes and model.num_classes must agree")
    if settings.data.target_num_classes not in SUPPORTED_TARGET_NUM_CLASSES:
        raise ValueError(
            "target_num_classes must be one of "
            f"{SUPPORTED_TARGET_NUM_CLASSES}; got {settings.data.target_num_classes}"
        )
    if settings.data.split_seed != 42:
        raise ValueError("The frozen client split seed must remain 42")
    if settings.data.source_label_space not in SUPPORTED_SOURCE_LABEL_SPACES:
        raise ValueError("Unsupported source_label_space")
    expected_targets = TARGET_NUM_CLASSES_BY_LABEL_SPACE[
        settings.data.source_label_space
    ]
    if settings.data.target_num_classes != expected_targets:
        raise ValueError(
            f"source_label_space {settings.data.source_label_space!r} defines a "
            f"{expected_targets}-class task; got "
            f"target_num_classes={settings.data.target_num_classes}"
        )
    if settings.method.ablation not in MVP_ABLATIONS:
        raise ValueError(f"ablation must be one of {MVP_ABLATIONS}")
    if settings.method.loss not in {"ce", "bounded_cb", "fedlc", "bounded_cb_lc"}:
        raise ValueError("loss must be ce, bounded_cb, fedlc, or bounded_cb_lc")
    if settings.method.ablation == "a0" and settings.method.loss != "ce":
        raise ValueError("A0 is CE + proximal")
    # A5 with plain CE is the one documented exception to the ladder: it is the
    # loss control for the main table. the eight baseline runs train on unweighted cross-entropy,
    # so an A5 arm that keeps the whole codec and drops only the class-balanced
    # reweighting is what separates the compressor's contribution from the
    # objective's. Without it a quality-over-uplink margin is attributable to either.
    # Every other ablation keeps the original coupling.
    loss_control = (
        settings.method.ablation == "a5" and settings.method.loss == "ce"
    )
    if (
        settings.method.ablation != "a0"
        and settings.method.loss == "ce"
        and not loss_control
    ):
        raise ValueError("A1--A5 require bounded CB or FedLC loss")
    if (
        settings.method.ablation in {"a2", "a3", "a4", "a5"}
        and settings.method.loss not in {"bounded_cb", "bounded_cb_lc"}
        and not loss_control
    ):
        raise ValueError(
            "A2--A5 use bounded_cb, optionally with the FedLC offset "
            "(bounded_cb_lc); plain FedLC is the separate A1 validation baseline"
        )
    expected_int8 = settings.method.ablation in {"a4", "a5"}
    expected_ef = settings.method.ablation == "a5"
    if settings.method.quant_bits not in MVP_QUANT_BITS:
        raise ValueError(f"quant_bits must be one of {MVP_QUANT_BITS}")
    if settings.method.quant_group_size not in {0, 32, 64, 128, 256, 512, 1024}:
        raise ValueError("Unsupported quant_group_size")
    if settings.method.quant_clipping not in {"max", "mse", "mse_refine"}:
        raise ValueError("quant_clipping must be max, mse or mse_refine")
    if settings.method.quant_clipping == "mse_refine" and settings.method.stochastic_rounding:
        raise ValueError("mse_refine requires deterministic rounding")
    if settings.method.downlink_bits not in {4, 8, 32}:
        raise ValueError("downlink_bits must be 4, 8 or 32")
    if settings.method.saliency_mode not in {"full_gradient", "online_task"}:
        raise ValueError("saliency_mode must be full_gradient or online_task")
    if type(settings.method.compact_layout) is not bool:
        raise ValueError("compact_layout must be a boolean")
    if type(settings.method.flat_uplink) is not bool:
        raise ValueError("flat_uplink must be a boolean")
    if settings.method.flat_uplink:
        if settings.method.minority_head_reserve > 0:
            raise ValueError("flat_uplink needs minority_head_reserve=0.0")
        if settings.method.protect_small_tensors:
            raise ValueError("flat_uplink cannot express per-tensor precision")
        if not settings.uses_sparse:
            raise ValueError("flat_uplink is defined for sparse uplinks")
    if settings.method.scale_codec not in {"fp32", "log8"}:
        raise ValueError("scale_codec must be fp32 or log8")
    if settings.method.quantizer not in {"integer", "gaussian4"}:
        raise ValueError("quantizer must be integer or gaussian4")
    if settings.method.quantizer == "gaussian4" and (
        settings.method.quant_bits != 2 or not settings.method.quant_group_size
    ):
        raise ValueError("The Gaussian codebook is a two-bit grouped quantizer")
    if type(settings.method.incoherent_rotation) is not bool:
        raise ValueError("incoherent_rotation must be a boolean")
    grouped = settings.method.quant_group_size
    if settings.method.incoherent_rotation and not (
        grouped and grouped & (grouped - 1) == 0
    ):
        raise ValueError("incoherent_rotation needs a power-of-two quant_group_size")
    if settings.method.scale_codec == "log8" and not grouped:
        raise ValueError("scale_codec log8 needs a positive quant_group_size")
    if settings.method.scale_codec == "log8" and settings.method.stochastic_rounding:
        raise ValueError("scale_codec log8 requires deterministic rounding")
    if settings.method.uplink_budget_bytes is not None:
        if (type(settings.method.uplink_budget_bytes) is not int
                or settings.method.uplink_budget_bytes <= 0):
            raise ValueError("uplink_budget_bytes must be a positive integer")
        if not settings.uses_sparse:
            raise ValueError("A byte budget requires sparse transport")
    if type(settings.method.block_size) is not int or settings.method.block_size not in {1, 4, 8, 16, 32}:
        raise ValueError("block_size must be 1, 4, 8, 16 or 32")
    if not math.isfinite(settings.method.minority_head_reserve) or not 0 <= settings.method.minority_head_reserve <= .5:
        raise ValueError("minority_head_reserve must be in [0,.5]")
    if (settings.method.block_size != 1 or settings.method.minority_head_reserve > 0) and not settings.uses_sparse:
        raise ValueError("Block/minority selection requires sparse transport")
    if type(settings.method.sparsity_warmup_rounds) is not int or settings.method.sparsity_warmup_rounds < 0:
        raise ValueError("sparsity_warmup_rounds must be a nonnegative integer")
    if not math.isfinite(settings.method.sparsity_initial) or not 0 <= settings.method.sparsity_initial <= settings.method.sparsity:
        raise ValueError("sparsity_initial must be in [0, target sparsity]")
    if settings.method.adaptive_quant_error is not None and (
        not math.isfinite(settings.method.adaptive_quant_error)
        or settings.method.adaptive_quant_error <= 0
    ):
        raise ValueError("adaptive_quant_error must be finite and positive")
    if settings.method.quant_bits != 8 and not expected_int8:
        raise ValueError("4-bit quantization is defined only for the A4/A5 codec")
    if settings.method.index_codec not in MVP_INDEX_CODECS:
        raise ValueError(f"index_codec must be one of {MVP_INDEX_CODECS}")
    if settings.method.stochastic_rounding and not expected_int8:
        raise ValueError("Stochastic rounding applies only to the A4/A5 quantizer")
    if bool(settings.method.error_feedback) != expected_ef:
        raise ValueError("error_feedback must match the selected ablation")
    if settings.method.sparsity not in SUPPORTED_SPARSITIES and not (
        settings.protocol_version == "fedmpsq-lowbit-v2" and settings.method.sparsity == 0.0
    ) and not (
        settings.protocol_version in {"fedmpsq-rare-v3", "fedmpsq-uplink-v4"}
        and math.isfinite(settings.method.sparsity) and 0 <= settings.method.sparsity < 1
    ):
        raise ValueError(f"sparsity must be one of {SUPPORTED_SPARSITIES}")
    if (
        not math.isfinite(settings.method.alpha_s)
        or not 0.0 <= settings.method.alpha_s <= 1.0
    ):
        raise ValueError("alpha_s must be in [0, 1]")
    if (
        not math.isfinite(settings.method.eta_s)
        or not 0.0 <= settings.method.eta_s < 1.0
    ):
        raise ValueError("eta_s must be in [0, 1)")
    if (
        not math.isfinite(settings.method.beta)
        or not 0.0 <= settings.method.beta < 1.0
    ):
        raise ValueError("beta must be in [0, 1)")
    if not math.isfinite(settings.method.w_max) or settings.method.w_max < 1.0:
        raise ValueError("w_max must be >= 1")
    if (
        not math.isfinite(settings.method.fedlc_tau)
        or settings.method.fedlc_tau <= 0.0
    ):
        raise ValueError("fedlc_tau must be positive")
    if settings.method.normalization != "max":
        raise ValueError("The MVP implements only max normalization")
    if settings.method.class_prior not in {"local", "global"}:
        raise ValueError("method.class_prior must be local or global")
    if settings.algorithm.aggregation not in {"sample", "uniform"}:
        raise ValueError("algorithm.aggregation must be sample or uniform")
    if settings.method.class_prior == "global" and settings.method.loss == "ce":
        raise ValueError("The global class prior only applies to a weighted loss")
    if settings.model.name not in SUPPORTED_MODELS:
        raise ValueError(f"model.name must be one of {SUPPORTED_MODELS}")
    if settings.model.norm not in SUPPORTED_NORMS:
        raise ValueError(f"model.norm must be one of {SUPPORTED_NORMS}")
    if settings.model.hidden_size <= 0 or settings.model.num_blocks < 0:
        raise ValueError("model.hidden_size must be positive and num_blocks non-negative")
    if not math.isfinite(settings.method.epsilon) or settings.method.epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if (
        not math.isfinite(settings.algorithm.proximal_mu)
        or settings.algorithm.proximal_mu < 0.0
    ):
        raise ValueError("proximal_mu must be non-negative")
    if settings.runtime.seed not in SUPPORTED_SEEDS:
        raise ValueError(f"seed must be one of {SUPPORTED_SEEDS}")
    if settings.algorithm.num_server_rounds <= 0:
        raise ValueError("num_server_rounds must be positive")
    if settings.algorithm.local_epochs <= 0:
        raise ValueError("local_epochs must be positive")
    if settings.algorithm.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if (
        not math.isfinite(settings.algorithm.learning_rate)
        or settings.algorithm.learning_rate <= 0.0
    ):
        raise ValueError("learning_rate must be finite and positive")
    if (
        not math.isfinite(settings.algorithm.momentum)
        or not 0.0 <= settings.algorithm.momentum < 1.0
    ):
        raise ValueError("momentum must be in [0, 1)")
    if (
        not math.isfinite(settings.algorithm.weight_decay)
        or settings.algorithm.weight_decay < 0.0
    ):
        raise ValueError("weight_decay must be finite and non-negative")
    if settings.runtime.parallel_clients <= 0:
        raise ValueError("parallel_clients must be positive")
    if not 0.0 < settings.algorithm.fraction_train <= 1.0:
        raise ValueError("fraction_train must be in (0, 1]")
    if settings.data.client_val_ratio < 0.0 or settings.data.client_val_ratio >= 1.0:
        raise ValueError("client_val_ratio must be in [0, 1)")
    if settings.data.minority_fraction <= 0.0 or settings.data.minority_fraction > 1.0:
        raise ValueError("minority_fraction must be in (0, 1]")
    if (
        settings.results.target_macro_f1 is not None
        and not 0.0 <= settings.results.target_macro_f1 <= 1.0
    ):
        raise ValueError("target_macro_f1 must be in [0, 1]")
    if not settings.results.save_model:
        raise ValueError("FedMPSQ protocol requires best and final model artifacts")


def load_fedmpsq_config(
    config_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> FedMPSQConfig:
    """Load an isolated YAML profile and explicit dotted overrides."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path is not None:
        _deep_update(config, _load_yaml(config_path))
    for key, value in dict(overrides or {}).items():
        if key in {"config_path", "config-path"}:
            continue
        _set_dotted(config, str(key), value)
    settings = _coerce(config)
    validate_fedmpsq_config(settings)
    return settings


def write_config_snapshot(settings: FedMPSQConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(settings.config_dict, indent=2, sort_keys=True), encoding="utf-8")
