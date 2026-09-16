"""Stable content hashes used by prepared-data provenance and manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file's exact serialized bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Hash an array including dtype and shape, not only its raw bytes."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(json.dumps(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def sha256_indices(indices: list[int] | np.ndarray) -> str:
    """Return the stable array hash for an ordered list of row indices."""
    return sha256_array(np.asarray(indices, dtype=np.int64))
