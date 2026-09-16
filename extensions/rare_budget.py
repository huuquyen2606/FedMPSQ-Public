"""Budgeted block selection with a local-present minority classifier quota.

The quota reallocates K, never adds coordinates on top of K. Class identities
come from the frozen pooled TRAIN counts, not validation or test performance.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch

from extensions.mpsq_int4 import deterministic_topk_indices


def minority_head_groups(model, local_counts, pooled_counts, fraction):
    """Return flattened coordinate groups for the final Linear's rare rows."""
    if pooled_counts is None:
        raise ValueError("Minority protection requires pooled training counts")
    pooled = torch.as_tensor(pooled_counts)
    local = torch.as_tensor(local_counts)
    if local.shape != pooled.shape or bool((local < 0).any()) or bool((pooled < local).any()):
        raise ValueError("Invalid local/pooled class counts")
    rare = sorted(range(len(pooled)), key=lambda c: (int(pooled[c]), c))[:max(1, math.ceil(len(pooled)*fraction))]
    heads = [(name, layer) for name, layer in model.named_modules()
             if isinstance(layer, torch.nn.Linear) and layer.out_features == len(pooled)]
    if not heads:
        raise ValueError("Cannot identify a final Linear classification head")
    name, head = heads[-1]
    prefix = name + "." if name else ""
    groups = []
    for c in rare:
        if local[c] <= 0:
            continue
        group = {prefix + "weight": torch.arange(c*head.in_features, (c+1)*head.in_features)}
        if head.bias is not None:
            group[prefix + "bias"] = torch.tensor([c])
        groups.append(group)
    return groups


def budgeted_block_indices(score, shapes: Mapping[str, torch.Size], keep, *,
                           block_size=1, protected_groups=(), reserve_fraction=0.0):
    """Choose at most K coordinates, reserving an equal quota per rare row.

    Blocks never cross tensor boundaries. Scores are sum of squared coordinate
    scores; the final partial block uses deterministic scalar Top-K. No hidden
    dense-head exemption and no client-to-server support cache is required.
    """
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if not math.isfinite(reserve_fraction) or not 0 <= reserve_fraction <= 1:
        raise ValueError("reserve_fraction must be in [0,1]")
    if score.ndim != 1 or not 0 < keep <= score.numel() or not bool(torch.isfinite(score).all()):
        raise ValueError("Invalid block selection scores/budget")
    offsets, start = {}, 0
    blocks = []
    for name, shape in shapes.items():
        size = math.prod(shape)
        offsets[name] = (start, size)
        positions = torch.arange(start, start+size)
        positions = torch.nn.functional.pad(positions, (0, (-size) % block_size), value=-1)
        blocks.append(positions.reshape(-1, block_size))
        start += size
    if start != score.numel():
        raise ValueError("Block layout differs from score size")
    selected = torch.zeros_like(score, dtype=torch.bool)
    reserved_budget = int(keep * reserve_fraction)
    for i, group in enumerate(protected_groups):
        coords = []
        for name, ids in group.items():
            if name not in offsets:
                raise ValueError(f"Protected tensor is outside Top-K pool: {name}")
            offset, size = offsets[name]
            ids = torch.as_tensor(ids, dtype=torch.long)
            if ids.ndim != 1 or bool(((ids < 0) | (ids >= size)).any()):
                raise ValueError("Protected coordinates outside tensor")
            coords.append(ids + offset)
        coords = torch.unique(torch.cat(coords), sorted=True)
        coords = coords[~selected[coords]]
        quota = reserved_budget // len(protected_groups) + (i < reserved_budget % len(protected_groups))
        if quota and coords.numel():
            picked = deterministic_topk_indices(score[coords], min(quota, coords.numel()))
            selected[coords[picked]] = True
    reserved_count = int(selected.sum())
    remaining = keep - reserved_count
    if remaining:
        positions = torch.cat(blocks)
        valid = positions >= 0
        valid &= ~selected[positions.clamp_min(0)]
        block_scores = torch.where(valid, score[positions.clamp_min(0)].double().square(), 0).sum(dim=1)
        lengths = valid.sum(dim=1)
        order = torch.argsort(block_scores, descending=True, stable=True)
        order = order[lengths[order] > 0]
        costs = lengths[order].cumsum(0)
        full = order[costs <= remaining]
        selected[positions[full][valid[full]]] = True
        leftover = keep - int(selected.sum())
        if leftover:
            next_block = order[len(full)]
            coords = positions[next_block][valid[next_block]]
            selected[coords[deterministic_topk_indices(score[coords], leftover)]] = True
    return torch.nonzero(selected, as_tuple=False).flatten(), reserved_count


def encode_runs(indices):
    """Unsigned LEB128 (gap, run length) pairs for sorted coordinate runs."""
    positions = indices.tolist()
    runs = []
    for value in positions:
        if runs and value == runs[-1][0] + runs[-1][1]:
            runs[-1][1] += 1
        else:
            runs.append([value, 1])
    result, end = bytearray(), 0
    for start, size in runs:
        for value in (start-end, size):
            while value >= 128:
                result.append((value & 127) | 128)
                value >>= 7
            result.append(value)
        end = start + size
    return bytes(result)


def decode_runs(blob, count, numel):
    import numpy as np
    values, offset, end = [], 0, 0
    while offset < len(blob):
        pair = []
        for _ in range(2):
            value = 0
            for shift in range(0, 70, 7):
                if offset >= len(blob):
                    raise ValueError("Truncated run varint")
                byte = blob[offset]
                offset += 1
                value |= (byte & 127) << shift
                if not byte & 128:
                    break
            else:
                raise ValueError("Run varint exceeds uint64")
            pair.append(value)
        gap, size = pair
        start = end + gap
        if size <= 0 or start + size > numel or len(values) + size > count:
            raise ValueError("Invalid coordinate run")
        values.extend(range(start, start+size))
        end = start + size
    if len(values) != count:
        raise ValueError("Run count differs from descriptor")
    return np.array(values, dtype=np.int64)
