"""PyTorch datasets and loaders for prepared CICIoT2023 partitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from data.splits import deterministic_split_indices


class FlowDataset(Dataset):
    """Tensor dataset for tabular network-flow features."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(x) != len(y):
            raise ValueError("x and y must contain the same number of examples")
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def load_metadata(partitions_dir: str | Path) -> dict[str, Any]:
    """Load partition metadata."""
    path = Path(partitions_dir) / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Partition metadata not found: {path}. Run scripts/prepare_ciciot2023.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz_dataset(path: str | Path) -> FlowDataset:
    """Load a compressed NPZ partition."""
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Partition file not found: {npz_path}")
    with np.load(npz_path) as payload:
        return FlowDataset(payload["x"], payload["y"])


def load_task_npz_dataset(path: str | Path, data_config: Any) -> FlowDataset:
    """Load an NPZ partition in the configured target label space."""
    source_label_space = getattr(data_config, "source_label_space", "identity")
    if source_label_space == "identity":
        return load_npz_dataset(path)
    if source_label_space == "ciciot2023_34":
        # Lazy import avoids a module cycle: the frozen mapping module also
        # uses FlowDataset for the separate FedMPSQ campaign.
        from data.fedmpsq_labels import load_grouped_npz_dataset

        return load_grouped_npz_dataset(
            path,
            source_label_space=source_label_space,
        )
    raise ValueError(f"Unsupported source label space: {source_label_space}")


def client_partition_path(partitions_dir: str | Path, partition_id: int) -> Path:
    """Return a client partition path."""
    return Path(partitions_dir) / f"client_{partition_id:03d}.npz"


def split_client_dataset(
    dataset: Dataset,
    val_ratio: float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    """Split a client dataset into local train and validation datasets."""
    train_indices, val_indices = deterministic_split_indices(
        len(dataset),
        val_ratio,
        seed,
    )
    if train_indices == val_indices:
        return dataset, dataset
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def load_client_loaders(
    partitions_dir: str | Path,
    partition_id: int,
    *,
    batch_size: int,
    val_ratio: float,
    seed: int,
    shuffle_seed: int | None = None,
    num_workers: int = 0,
    data_config: Any | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Load train and validation dataloaders for one simulated client."""
    path = client_partition_path(partitions_dir, partition_id)
    dataset = (
        load_npz_dataset(path)
        if data_config is None
        else load_task_npz_dataset(path, data_config)
    )
    train_dataset, val_dataset = split_client_dataset(
        dataset,
        val_ratio=val_ratio,
        seed=seed + partition_id,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(
            seed + partition_id if shuffle_seed is None else shuffle_seed
        ),
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def load_global_test_loader(
    partitions_dir: str | Path,
    *,
    batch_size: int,
    num_workers: int = 0,
    data_config: Any | None = None,
) -> DataLoader:
    """Load the centralized global test split."""
    path = Path(partitions_dir) / "global_test.npz"
    dataset = (
        load_npz_dataset(path)
        if data_config is None
        else load_task_npz_dataset(path, data_config)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
