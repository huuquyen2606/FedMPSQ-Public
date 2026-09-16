"""Client-side runtime used by the thin Flower adapter."""

from __future__ import annotations

from typing import Any

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict

from data.dataset import load_client_loaders
from fl.config import load_settings, resolve_dimensions, select_device
from models import build_model
from training.trainer import evaluate_model, train_local_model


def _node_int(context: Context, *keys: str, default: int = 0) -> int:
    for key in keys:
        if key in context.node_config:
            return int(context.node_config[key])
    return default


def _message_config(msg: Message) -> dict[str, Any]:
    if "config" not in msg.content:
        return {}
    return dict(msg.content["config"])


def train_client(msg: Message, context: Context) -> Message:
    """Train one simulated client and return updated model weights."""
    settings = load_settings(context.run_config)
    input_dim, num_classes = resolve_dimensions(settings)
    device = select_device(settings.runtime.device)

    model = build_model(settings.model, input_dim=input_dim, num_classes=num_classes)
    global_state = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(global_state, strict=True)
    model.to(device)

    partition_id = _node_int(context, "partition-id", "partition_id")
    train_loader, _ = load_client_loaders(
        settings.data.partitions_dir,
        partition_id,
        batch_size=settings.algorithm.batch_size,
        val_ratio=settings.data.client_val_ratio,
        seed=settings.data.seed,
        data_config=settings.data,
    )

    train_config = _message_config(msg)
    metrics = train_local_model(
        model,
        train_loader,
        settings.algorithm,
        device,
        learning_rate=float(train_config.get("lr", settings.algorithm.learning_rate)),
    )
    metrics["num-examples"] = len(train_loader.dataset)

    content = RecordDict(
        {
            "arrays": ArrayRecord(model.state_dict()),
            "metrics": MetricRecord(metrics),
        }
    )
    return Message(content=content, reply_to=msg)


def evaluate_client(msg: Message, context: Context) -> Message:
    """Evaluate one simulated client on its local validation split."""
    settings = load_settings(context.run_config)
    input_dim, num_classes = resolve_dimensions(settings)
    device = select_device(settings.runtime.device)

    model = build_model(settings.model, input_dim=input_dim, num_classes=num_classes)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=True)
    model.to(device)

    partition_id = _node_int(context, "partition-id", "partition_id")
    _, val_loader = load_client_loaders(
        settings.data.partitions_dir,
        partition_id,
        batch_size=settings.algorithm.batch_size,
        val_ratio=settings.data.client_val_ratio,
        seed=settings.data.seed,
        data_config=settings.data,
    )
    metrics = evaluate_model(model, val_loader, device, num_classes=num_classes)
    metrics["num-examples"] = len(val_loader.dataset)

    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)
