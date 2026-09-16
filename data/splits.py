"""Shared deterministic split rules for preparation, loading, and manifests."""

from __future__ import annotations

import torch


def deterministic_split_indices(
    num_examples: int,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Return the exact local train/validation assignment used by the runner.

    The ordered indices reproduce ``torch.random_split`` with a dedicated
    generator. A one-example partition cannot provide an independent
    validation set, so both roles necessarily refer to that sole example.
    """
    if num_examples <= 0:
        raise ValueError("Client partition is empty")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("client validation ratio must be in [0, 1)")
    if num_examples < 2 or val_ratio <= 0.0:
        indices = list(range(num_examples))
        return indices, indices
    val_size = int(round(num_examples * val_ratio))
    val_size = max(1, min(val_size, num_examples - 1))
    train_size = num_examples - val_size
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_examples, generator=generator).tolist()
    return permutation[:train_size], permutation[train_size:]
