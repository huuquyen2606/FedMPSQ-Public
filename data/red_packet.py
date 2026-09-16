"""Red Packet non-IID partitioning for label-skewed federated data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RedPacketConfig:
    """Configuration for Red Packet allocation."""

    num_clients: int = 10
    min_label_clients: int = 1
    max_label_clients: int = 8
    seed: int = 42
    lognormal_sigma: float = 1.25


def random_packet_counts(
    total: int,
    receivers: int,
    rng: np.random.Generator,
    *,
    sigma: float,
) -> np.ndarray:
    """Allocate a total count into uneven positive integer packets."""
    if receivers <= 0:
        raise ValueError("receivers must be positive")
    if total < receivers:
        receivers = total
    if receivers == 1:
        return np.array([total], dtype=np.int64)
    weights = rng.lognormal(mean=0.0, sigma=sigma, size=receivers)
    probabilities = weights / weights.sum()
    counts = rng.multinomial(total - receivers, probabilities) + 1
    return counts.astype(np.int64)


def _label_index_groups(y: np.ndarray) -> tuple[list[int], list[np.ndarray]]:
    """Return sorted labels with their ascending index arrays in one pass.

    Equivalent to ``np.flatnonzero(y == label)`` per sorted label, but one
    stable argsort replaces one full scan per class.  Order inside each group
    is ascending, exactly as ``np.flatnonzero`` returns it, so the random
    stream consumed by the caller is unchanged.
    """
    order = np.argsort(y, kind="stable")
    labels, starts = np.unique(y[order], return_index=True)
    bounds = [*starts.tolist(), len(y)]
    return (
        [int(label) for label in labels],
        [
            order[bounds[position] : bounds[position + 1]].astype(np.int64)
            for position in range(len(labels))
        ],
    )


def red_packet_partition(
    y: np.ndarray,
    config: RedPacketConfig,
) -> list[np.ndarray]:
    """Partition labels by assigning each class to a random subset of clients.

    For each label, a random subset of at most ``max_label_clients`` clients
    receives that label. The examples for that label are then distributed with
    a skewed random packet allocation so both label skew and quantity skew
    appear across clients.

    Index bookkeeping stays in NumPy so a 37-million-row pool partitions in
    seconds; the drawn random numbers, and therefore the resulting partition,
    are identical to the element-wise formulation.
    """
    if config.num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if config.min_label_clients < 1:
        raise ValueError("min_label_clients must be at least 1")
    if config.max_label_clients < config.min_label_clients:
        raise ValueError("max_label_clients must be >= min_label_clients")

    labels = np.asarray(y)
    rng = np.random.default_rng(config.seed)
    client_chunks: list[list[np.ndarray]] = [[] for _ in range(config.num_clients)]
    label_clients: dict[int, set[int]] = {}
    for label, label_indices in zip(*_label_index_groups(labels), strict=True):
        rng.shuffle(label_indices)
        if len(label_indices) == 0:
            continue
        max_receivers = min(config.max_label_clients, config.num_clients, len(label_indices))
        min_receivers = min(config.min_label_clients, max_receivers)
        receivers = int(rng.integers(min_receivers, max_receivers + 1))
        selected_clients = rng.choice(config.num_clients, size=receivers, replace=False)
        counts = random_packet_counts(
            total=len(label_indices),
            receivers=receivers,
            rng=rng,
            sigma=config.lognormal_sigma,
        )
        offset = 0
        label_clients[label] = set()
        for client_id, count in zip(selected_clients, counts, strict=True):
            count = int(count)
            client_chunks[int(client_id)].append(label_indices[offset : offset + count])
            label_clients[label].add(int(client_id))
            offset += count

    client_arrays = [
        np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
        for chunks in client_chunks
    ]
    if any(array.size == 0 for array in client_arrays):
        # Only the repair path needs element-wise lists, and it only runs when
        # the draw left a client with nothing at all.
        client_indices = [array.tolist() for array in client_arrays]
        _fill_empty_clients(client_indices, label_clients, labels, config)
        client_arrays = [
            np.asarray(indices, dtype=np.int64) for indices in client_indices
        ]

    partitions: list[np.ndarray] = []
    for array in client_arrays:
        array = np.ascontiguousarray(array, dtype=np.int64)
        rng.shuffle(array)
        partitions.append(array)
    return partitions


def _fill_empty_clients(
    client_indices: list[list[int]],
    label_clients: dict[int, set[int]],
    y: np.ndarray,
    config: RedPacketConfig,
) -> None:
    """Move a few samples so every simulated client owns at least one example."""
    empty_clients = [idx for idx, indices in enumerate(client_indices) if not indices]
    if not empty_clients:
        return
    for empty_client in empty_clients:
        donor_client = _find_donor_client(client_indices)
        if donor_client is None:
            raise ValueError("Unable to fill empty clients from the available data")
        moved_index = _pop_movable_index(
            client_indices[donor_client],
            label_clients,
            y,
            config,
            donor_client=donor_client,
            target_client=empty_client,
        )
        if moved_index is None:
            raise ValueError(
                "Unable to preserve the 1-8 clients-per-label constraint while "
                "filling empty clients"
            )
        label = int(y[moved_index])
        client_indices[empty_client].append(moved_index)
        label_clients[label].add(empty_client)


def _find_donor_client(client_indices: list[list[int]]) -> int | None:
    donor_sizes = sorted(
        ((len(indices), client_id) for client_id, indices in enumerate(client_indices)),
        reverse=True,
    )
    for size, client_id in donor_sizes:
        if size > 1:
            return client_id
    return None


def _pop_movable_index(
    donor_indices: list[int],
    label_clients: dict[int, set[int]],
    y: np.ndarray,
    config: RedPacketConfig,
    *,
    donor_client: int,
    target_client: int,
) -> int | None:
    for offset in range(len(donor_indices) - 1, -1, -1):
        candidate = donor_indices[offset]
        label = int(y[candidate])
        updated_receivers = set(label_clients[label])
        donor_label_count = sum(1 for idx in donor_indices if int(y[idx]) == label)
        if donor_label_count == 1:
            updated_receivers.discard(donor_client)
        updated_receivers.add(target_client)
        if len(updated_receivers) <= min(config.max_label_clients, config.num_clients):
            moved_index = donor_indices.pop(offset)
            label_clients[label] = updated_receivers
            return moved_index
    return None
