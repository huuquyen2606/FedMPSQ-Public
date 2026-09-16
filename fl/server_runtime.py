"""Server-side runtime used by the thin Flower adapter."""

from __future__ import annotations

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from data.partition_manifest import verify_partition_manifest
from fl.config import (
    load_settings,
    resolve_dimensions,
    seed_everything,
)
from fl.metrics_logger import RoundMetricsLogger
from models import build_model


def run_server(grid: Grid, context: Context) -> None:
    """Run the configured Flower ServerApp."""
    settings = load_settings(context.run_config)
    verify_partition_manifest(settings)
    if settings.technique.name != "none":
        raise RuntimeError(
            "Paper integrations use the audited single-process runner. "
            "Run python main.py --config-path configs/eX_*.yaml."
        )
    seed_everything(settings.runtime.seed)
    input_dim, num_classes = resolve_dimensions(settings)

    logger = RoundMetricsLogger(settings.results.dir, settings.results.run_name)
    logger.save_config(settings)
    print(
        "FLOWER_DIAGNOSTIC local client validation only; this adapter does not "
        "produce paper-protocol-v3 best/final test artifacts. Use "
        "scripts/run_kaggle.py for paper runs.",
        flush=True,
    )

    global_model = build_model(settings.model, input_dim=input_dim, num_classes=num_classes)
    arrays = ArrayRecord(global_model.state_dict())
    strategy = FedAvg(
        fraction_train=settings.algorithm.fraction_train,
        fraction_evaluate=settings.algorithm.fraction_evaluate,
        min_train_nodes=settings.algorithm.min_train_nodes,
        min_evaluate_nodes=settings.algorithm.min_evaluate_nodes,
        min_available_nodes=settings.algorithm.min_available_nodes,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(
            {
                "lr": settings.algorithm.learning_rate,
                "algorithm": settings.algorithm.name,
                "proximal_mu": settings.algorithm.proximal_mu,
            }
        ),
        num_rounds=settings.algorithm.num_server_rounds,
        # Never expose the frozen global test to a per-round Flower callback.
        # ClientApp evaluation uses each client's fixed local-validation split.
        evaluate_fn=None,
    )
    logger.save_flower_history(result)

    if settings.results.save_model:
        model_path = logger.results_dir / f"{settings.results.run_name}_final_model.pt"
        torch.save(result.arrays.to_torch_state_dict(), model_path)
