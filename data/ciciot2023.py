"""CICIoT2023 loading and Red Packet partition preparation."""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data.client_statistics import write_client_distribution_statistics
from data.integrity import sha256_array, sha256_file, sha256_indices
from data.red_packet import RedPacketConfig, red_packet_partition
from data.splits import deterministic_split_indices

CICIOT2023_LABELS = [
    "DDoS-RSTFINFlood",
    "DDoS-PSHACK_Flood",
    "DDoS-SYN_Flood",
    "DDoS-UDP_Flood",
    "DDoS-TCP_Flood",
    "DDoS-ICMP_Flood",
    "DDoS-SynonymousIP_Flood",
    "DDoS-ACK_Fragmentation",
    "DDoS-UDP_Fragmentation",
    "DDoS-ICMP_Fragmentation",
    "DDoS-SlowLoris",
    "DDoS-HTTP_Flood",
    "DoS-UDP_Flood",
    "DoS-SYN_Flood",
    "DoS-TCP_Flood",
    "DoS-HTTP_Flood",
    "Mirai-greeth_flood",
    "Mirai-greip_flood",
    "Mirai-udpplain",
    "Recon-PingSweep",
    "Recon-OSScan",
    "Recon-PortScan",
    "VulnerabilityScan",
    "Recon-HostDiscovery",
    "DNS_Spoofing",
    "MITM-ArpSpoofing",
    "BenignTraffic",
    "BrowserHijacking",
    "Backdoor_Malware",
    "XSS",
    "Uploading_Attack",
    "SqlInjection",
    "CommandInjection",
    "DictionaryBruteForce",
]


def discover_csv_files(raw_data: str | Path) -> list[Path]:
    """Return sorted CSV files from a file or directory path."""
    raw_path = Path(raw_data)
    if raw_path.is_file():
        return [raw_path]
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data path does not exist: {raw_path}")
    files = sorted(raw_path.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under: {raw_path}")
    return files


def read_ciciot2023_csvs(
    csv_files: Iterable[Path],
    label_column: str,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Read CICIoT2023 CSV files into one dataframe."""
    frames: list[pd.DataFrame] = []
    remaining = max_rows
    for csv_file in csv_files:
        nrows = remaining if remaining is not None else None
        if remaining is not None and remaining <= 0:
            break
        frame = pd.read_csv(csv_file, nrows=nrows)
        if label_column not in frame.columns:
            raise ValueError(f"Missing label column '{label_column}' in {csv_file}")
        frames.append(frame)
        if remaining is not None:
            remaining -= len(frame)
    if not frames:
        raise ValueError("No rows were read from the supplied CICIoT2023 CSV files")
    return pd.concat(frames, ignore_index=True)


def encode_labels(labels: pd.Series, strict: bool = True) -> np.ndarray:
    """Map CICIoT2023 label strings to stable integer class ids."""
    label_to_id = {label: idx for idx, label in enumerate(CICIOT2023_LABELS)}
    unknown = sorted(set(labels.astype(str)) - set(label_to_id))
    if unknown and strict:
        raise ValueError(
            "Found labels outside the expected CICIoT2023 34-label set: "
            + ", ".join(unknown[:10])
        )
    return labels.astype(str).map(label_to_id).to_numpy(dtype=np.int64)


def encode_features(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """Encode numeric and categorical feature columns into a numeric table."""
    features = frame.drop(columns=[label_column]).copy()
    features = features.replace([np.inf, -np.inf], np.nan)
    object_columns = [
        column
        for column in features.columns
        if (
            isinstance(features[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(features[column].dtype)
            or pd.api.types.is_string_dtype(features[column].dtype)
        )
    ]
    if object_columns:
        features = pd.get_dummies(features, columns=object_columns, dummy_na=True)
    for column in features.columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    return features


def _feature_columns_from_training_frame(
    frame: pd.DataFrame,
    label_column: str,
) -> list[str]:
    """Return the fixed feature schema learned from training rows only.

    ``get_dummies`` is deliberately called only for the rows supplied here.
    Evaluation rows are later reindexed to this schema by
    :func:`_transform_feature_frame`; categories that occur only in validation
    or test therefore cannot add columns to the fitted schema.
    """
    encoded = encode_features(frame, label_column)
    if encoded.shape[1] == 0:
        raise ValueError("No feature columns were found in local-training rows")
    return list(encoded.columns)


def _transform_feature_frame(
    frame: pd.DataFrame,
    label_column: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Encode rows with an already-fitted training-only feature schema."""
    return encode_features(frame, label_column).reindex(
        columns=feature_columns,
        fill_value=0.0,
    )


def _fit_frame_medians(
    frame: pd.DataFrame,
) -> np.ndarray:
    """Fit per-feature medians on a training-only encoded frame.

    Infinite values have already been converted to NaN by ``encode_features``;
    the explicit replacement here also protects callers that provide an
    already-encoded frame. Features with no finite training value use the
    documented zero fallback.
    """
    clean = frame.replace([np.inf, -np.inf], np.nan).apply(
        pd.to_numeric,
        errors="coerce",
    )
    medians = clean.median(axis=0, skipna=True).replace(
        [np.inf, -np.inf], np.nan
    )
    return medians.fillna(0.0).to_numpy(dtype=np.float32)


def _fill_frame_with_medians(
    frame: pd.DataFrame,
    medians: np.ndarray,
) -> pd.DataFrame:
    """Apply fitted medians (and zero fallback) without refitting anything."""
    if frame.shape[1] != len(medians):
        raise ValueError(
            "Feature frame and median vector have incompatible dimensions"
        )
    fill_values = pd.Series(
        np.asarray(medians, dtype=np.float32),
        index=frame.columns,
    )
    return frame.replace([np.inf, -np.inf], np.nan).fillna(fill_values).fillna(0.0)


def stratified_global_test_split(
    y: np.ndarray,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train and global-test indices using per-class sampling."""
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("global_test_ratio must be between 0 and 1")
    rng = np.random.default_rng(seed)
    train_indices: list[np.ndarray] = []
    test_indices: list[np.ndarray] = []
    for class_id in np.unique(y):
        indices = np.flatnonzero(y == class_id)
        rng.shuffle(indices)
        if len(indices) <= 1:
            train_indices.append(indices)
            continue
        test_count = int(round(len(indices) * test_ratio))
        test_count = max(1, min(test_count, len(indices) - 1))
        test_indices.append(indices[:test_count])
        train_indices.append(indices[test_count:])
    train = np.concatenate(train_indices) if train_indices else np.array([], dtype=np.int64)
    test = np.concatenate(test_indices) if test_indices else np.array([], dtype=np.int64)
    if len(test) == 0:
        raise ValueError("Unable to create a non-empty global test split")
    return train, test


def class_counts(y: np.ndarray) -> dict[str, int]:
    """Return class counts keyed by CICIoT2023 label name."""
    counts = np.bincount(y, minlength=len(CICIOT2023_LABELS))
    return {
        CICIOT2023_LABELS[idx]: int(count)
        for idx, count in enumerate(counts)
        if int(count) > 0
    }


def sorted_client_csv_files(splits_dir: str | Path, pattern: str) -> list[Path]:
    """Return client CSV files sorted by numeric client id when present."""
    files = list(Path(splits_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No client CSV files matching {pattern!r} in {splits_dir}")

    def key(path: Path) -> tuple[int, str]:
        match = re.search(r"client[_-]?(\d+)", path.stem)
        if match:
            return int(match.group(1)), path.name
        return 10**9, path.name

    return sorted(files, key=key)


def _align_feature_frames(
    train_frames: list[pd.DataFrame],
    test_frame: pd.DataFrame,
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    train_columns: list[str] = []
    for frame in train_frames:
        for column in frame.columns:
            if column not in train_columns:
                train_columns.append(column)
    aligned_train = [
        frame.reindex(columns=train_columns, fill_value=0.0) for frame in train_frames
    ]
    aligned_test = test_frame.reindex(columns=train_columns, fill_value=0.0)
    return aligned_train, aligned_test


def _iter_csv_chunks(path: Path, chunk_size: int | None) -> Iterable[pd.DataFrame]:
    """Yield a CSV file as one or more dataframes."""
    if chunk_size is None or chunk_size <= 0:
        yield pd.read_csv(path)
        return
    yield from pd.read_csv(path, chunksize=chunk_size)


def _discover_existing_split_schema(
    train_files: list[Path],
    test_file: Path,
    *,
    label_column: str,
    chunk_size: int | None,
    client_val_ratio: float,
    seed: int,
) -> tuple[list[str], dict[Path, int], list[list[int]], list[list[int]]]:
    """Fit the feature schema only on deterministic local-training rows."""
    row_counts: dict[Path, int] = {}
    for path in train_files + [test_file]:
        row_count = 0
        for chunk in _iter_csv_chunks(path, chunk_size):
            if label_column not in chunk.columns:
                raise ValueError(f"Missing label column '{label_column}' in {path}")
            row_count += len(chunk)
        if row_count == 0:
            raise ValueError(f"No rows were read from {path}")
        row_counts[path] = row_count

    train_indices: list[list[int]] = []
    validation_indices: list[list[int]] = []
    feature_columns: list[str] = []
    for client_id, path in enumerate(train_files):
        local_train, local_validation = deterministic_split_indices(
            row_counts[path],
            client_val_ratio,
            seed + client_id,
        )
        train_indices.append(local_train)
        validation_indices.append(local_validation)
        training_mask = np.zeros(row_counts[path], dtype=bool)
        training_mask[np.asarray(local_train, dtype=np.int64)] = True
        offset = 0
        for chunk in _iter_csv_chunks(path, chunk_size):
            rows = len(chunk)
            selected = chunk.iloc[training_mask[offset : offset + rows]]
            offset += rows
            if len(selected) == 0:
                continue
            features = encode_features(selected, label_column)
            for column in features.columns:
                if column not in feature_columns:
                    feature_columns.append(column)
    if not feature_columns:
        raise ValueError("No feature columns were found in local-training rows")
    return feature_columns, row_counts, train_indices, validation_indices


def _close_memmap(array: np.ndarray | None) -> None:
    """Close a NumPy memmap deterministically so its file can be deleted."""
    if array is None:
        return
    mmap_handle = getattr(array, "_mmap", None)
    if mmap_handle is not None and not mmap_handle.closed:
        mmap_handle.close()


def _materialize_existing_split_csv(
    csv_path: Path,
    *,
    x_path: Path,
    y_path: Path,
    row_count: int,
    feature_columns: list[str],
    label_column: str,
    chunk_size: int | None,
) -> tuple[Path, Path]:
    """Write one CSV split to temporary NPY memmaps."""
    x_memmap: np.memmap | None = None
    y_memmap: np.memmap | None = None
    try:
        x_memmap = np.lib.format.open_memmap(
            x_path,
            mode="w+",
            dtype=np.float32,
            shape=(row_count, len(feature_columns)),
        )
        y_memmap = np.lib.format.open_memmap(
            y_path,
            mode="w+",
            dtype=np.int64,
            shape=(row_count,),
        )
        offset = 0
        for chunk in _iter_csv_chunks(csv_path, chunk_size):
            labels = encode_labels(chunk[label_column], strict=True)
            features = encode_features(chunk, label_column).reindex(
                columns=feature_columns,
                fill_value=0.0,
            )
            rows = len(chunk)
            x_memmap[offset : offset + rows] = features.to_numpy(
                dtype=np.float32,
                copy=True,
            )
            y_memmap[offset : offset + rows] = labels
            offset += rows
        if offset != row_count:
            raise ValueError(f"Expected {row_count} rows from {csv_path}, read {offset}")
        x_memmap.flush()
        y_memmap.flush()
    finally:
        _close_memmap(y_memmap)
        _close_memmap(x_memmap)
    return x_path, y_path


def _fill_missing_with_medians(block: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """Return a dense block with NaNs replaced by training medians."""
    filled = np.array(block, dtype=np.float32, copy=True)
    missing = np.isnan(filled)
    if missing.any():
        filled[missing] = np.take(medians, np.nonzero(missing)[1])
    return filled


def _compute_feature_medians(
    client_x_paths: list[Path],
    input_dim: int,
    client_train_indices: list[list[int]],
) -> np.ndarray:
    """Compute exact medians from local-training rows, excluding validation."""
    medians = np.zeros(input_dim, dtype=np.float32)
    client_arrays = [np.load(path, mmap_mode="r") for path in client_x_paths]
    try:
        for column_idx in range(input_dim):
            values = np.concatenate(
                [
                    array[np.asarray(indices, dtype=np.int64), column_idx]
                    for array, indices in zip(
                        client_arrays,
                        client_train_indices,
                        strict=True,
                    )
                ]
            )
            median = float(np.nanmedian(values))
            medians[column_idx] = median if np.isfinite(median) else 0.0
    finally:
        for client_array in client_arrays:
            _close_memmap(client_array)
    return medians


def _iter_array_blocks(array: np.ndarray, block_size: int | None) -> Iterable[np.ndarray]:
    """Yield row blocks from an array or memmap."""
    if block_size is None or block_size <= 0:
        yield array
        return
    for start in range(0, array.shape[0], block_size):
        yield array[start : start + block_size]


def _fit_scaler_from_memmaps(
    client_x_paths: list[Path],
    client_train_indices: list[list[int]],
    medians: np.ndarray,
    *,
    block_size: int | None,
) -> StandardScaler:
    """Fit StandardScaler incrementally on local-training rows only."""
    scaler = StandardScaler()
    effective_block_size = block_size if block_size is not None and block_size > 0 else None
    for x_path, indices in zip(
        client_x_paths,
        client_train_indices,
        strict=True,
    ):
        x_memmap = np.load(x_path, mmap_mode="r")
        try:
            index_array = np.asarray(indices, dtype=np.int64)
            step = effective_block_size or len(index_array)
            for start in range(0, len(index_array), step):
                block = x_memmap[index_array[start : start + step]]
                scaler.partial_fit(_fill_missing_with_medians(block, medians))
        finally:
            _close_memmap(x_memmap)
    return scaler


def _write_scaled_npz(
    *,
    x_path: Path,
    y_path: Path,
    output_path: Path,
    scaled_x_path: Path,
    medians: np.ndarray,
    scaler: StandardScaler,
    block_size: int | None,
) -> None:
    """Scale one temporary split and save it as the project NPZ format."""
    x_memmap: np.memmap | None = None
    scaled_x: np.memmap | None = None
    y_memmap: np.memmap | None = None
    completed = False
    try:
        x_memmap = np.load(x_path, mmap_mode="r")
        scaled_x = np.lib.format.open_memmap(
            scaled_x_path,
            mode="w+",
            dtype=np.float32,
            shape=x_memmap.shape,
        )
        offset = 0
        for block in _iter_array_blocks(x_memmap, block_size):
            transformed = scaler.transform(
                _fill_missing_with_medians(block, medians)
            ).astype(np.float32)
            rows = transformed.shape[0]
            scaled_x[offset : offset + rows] = transformed
            offset += rows
        scaled_x.flush()
        y_memmap = np.load(y_path, mmap_mode="r")
        np.savez_compressed(output_path, x=scaled_x, y=y_memmap)
        completed = True
    finally:
        _close_memmap(y_memmap)
        _close_memmap(scaled_x)
        _close_memmap(x_memmap)
        scaled_x_path.unlink(missing_ok=True)
        if not completed:
            output_path.unlink(missing_ok=True)


def _count_csv_rows(
    path: Path,
    label_column: str,
    chunk_size: int | None,
) -> int:
    """Count rows of one split while asserting the label column is present."""
    row_count = 0
    for chunk in _iter_csv_chunks(path, chunk_size):
        if label_column not in chunk.columns:
            raise ValueError(f"Missing label column '{label_column}' in {path}")
        row_count += len(chunk)
    if row_count == 0:
        raise ValueError(f"No rows were read from {path}")
    return row_count


def _shared_csv_feature_schema(
    paths: list[Path],
    *,
    label_column: str,
) -> list[str]:
    """Return the one feature schema every split shares, or fail closed.

    Re-partitioning moves rows between clients, so a schema derived from a
    particular client's rows would not be well defined.  The schema is instead
    the shared CSV header, and a categorical column - which would make the
    encoded width depend on which rows were read - is rejected outright.
    """
    reference: list[str] | None = None
    for path in paths:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if header.count(label_column) != 1:
            raise ValueError(
                f"CSV must contain exactly one '{label_column}' column: {path}"
            )
        if header[-1] != label_column:
            raise ValueError(f"'{label_column}' must be the final column: {path}")
        features = header[:-1]
        if len(features) != len(set(features)):
            raise ValueError(f"Duplicate feature names in {path}")
        if reference is None:
            reference = features
        elif features != reference:
            raise ValueError(f"Feature schema differs in {path}")
        probe = pd.read_csv(path, nrows=2048)
        encoded = list(encode_features(probe, label_column).columns)
        if encoded != features:
            raise ValueError(
                "Re-partitioning requires a purely numeric feature schema; "
                f"{path} encodes to {len(encoded)} columns instead of "
                f"{len(features)}"
            )
    if not reference:
        raise ValueError("No feature columns were found in the source splits")
    return reference


def _pooled_source_offsets(row_counts: list[int]) -> np.ndarray:
    """Return exclusive-prefix offsets of the concatenated source splits."""
    offsets = np.zeros(len(row_counts) + 1, dtype=np.int64)
    np.cumsum(np.asarray(row_counts, dtype=np.int64), out=offsets[1:])
    return offsets


def _locate_pooled_indices(
    pooled_indices: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split pooled row ids into (source file id, row id inside that file)."""
    indices = np.asarray(pooled_indices, dtype=np.int64)
    if indices.size and (indices.min() < 0 or indices.max() >= int(offsets[-1])):
        raise ValueError("Pooled index outside the concatenated source range")
    source_ids = np.searchsorted(offsets, indices, side="right") - 1
    return source_ids, indices - offsets[source_ids]


def _gather_pooled_rows(
    source_paths: list[Path],
    pooled_indices: np.ndarray,
    offsets: np.ndarray,
    *,
    dtype: Any,
    width: int | None,
) -> np.ndarray:
    """Collect scattered pooled rows, reading each source file exactly once."""
    indices = np.asarray(pooled_indices, dtype=np.int64)
    shape = (len(indices),) if width is None else (len(indices), width)
    output = np.empty(shape, dtype=dtype)
    source_ids, local_rows = _locate_pooled_indices(indices, offsets)
    for source_id, path in enumerate(source_paths):
        positions = np.flatnonzero(source_ids == source_id)
        if positions.size == 0:
            continue
        rows = local_rows[positions]
        # Ascending reads keep a memory-mapped gather close to sequential.
        order = np.argsort(rows, kind="stable")
        source_array = np.load(path, mmap_mode="r")
        try:
            output[positions[order]] = source_array[rows[order]]
        finally:
            _close_memmap(source_array)
    return output


def prepare_repartitioned_ciciot2023_splits(
    splits_dir: str | Path,
    output_dir: str | Path,
    *,
    label_column: str = "label",
    client_pattern: str = "client_*.csv",
    global_test_file: str = "global_test.csv",
    source_clients: int = 10,
    num_clients: int = 100,
    min_label_clients: int = 1,
    max_label_clients: int | None = None,
    lognormal_sigma: float = 1.25,
    partition_seed: int = 42,
    chunk_size: int | None = 100_000,
    client_val_ratio: float = 0.2,
    seed: int = 42,
    temp_dir: str | Path | None = None,
) -> dict:
    """Re-partition an existing client pool into num_clients Red Packet shards.

    The pooled rows are exactly the concatenation of the source client CSVs in
    client-id order: no row is added, dropped, or moved between the pool and the
    global test.  The immutable global test is re-encoded with the statistics
    fitted on *this* partition's local-training rows, so the prepared directory
    is a self-contained experiment rather than a mixture of two fit scopes.

    max_label_clients defaults to num_clients, which is the rule the frozen
    10-client split obeys: at least one and at most all clients receive any
    given class.
    """
    splits_path = Path(splits_dir)
    client_files = sorted_client_csv_files(splits_path, client_pattern)
    if len(client_files) != source_clients:
        raise ValueError(
            f"Expected {source_clients} source client CSV files, "
            f"found {len(client_files)}"
        )
    global_test_path = splits_path / global_test_file
    if not global_test_path.exists():
        raise FileNotFoundError(f"Global test CSV not found: {global_test_path}")
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    receivers_cap = num_clients if max_label_clients is None else int(max_label_clients)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    feature_columns = _shared_csv_feature_schema(
        [*client_files, global_test_path],
        label_column=label_column,
    )
    input_dim = len(feature_columns)

    # Staging needs roughly one uncompressed copy of the pool, so it can be
    # pointed at a scratch volume instead of the published output directory.
    staging_root = Path(temp_dir) if temp_dir is not None else output_path
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".ids_fed_repartition_",
        dir=staging_root,
    ) as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        source_x_paths: list[Path] = []
        source_y_paths: list[Path] = []
        source_row_counts: list[int] = []
        for source_id, csv_path in enumerate(client_files):
            row_count = _count_csv_rows(csv_path, label_column, chunk_size)
            x_path, y_path = _materialize_existing_split_csv(
                csv_path,
                x_path=tmp_dir / f"source_{source_id:03d}_x.npy",
                y_path=tmp_dir / f"source_{source_id:03d}_y.npy",
                row_count=row_count,
                feature_columns=feature_columns,
                label_column=label_column,
                chunk_size=chunk_size,
            )
            source_x_paths.append(x_path)
            source_y_paths.append(y_path)
            source_row_counts.append(row_count)
        offsets = _pooled_source_offsets(source_row_counts)
        pooled_rows = int(offsets[-1])

        pooled_labels = np.empty(pooled_rows, dtype=np.int64)
        for source_id, y_path in enumerate(source_y_paths):
            labels_memmap = np.load(y_path, mmap_mode="r")
            try:
                start = int(offsets[source_id])
                pooled_labels[start : start + len(labels_memmap)] = labels_memmap
            finally:
                _close_memmap(labels_memmap)

        packet_config = RedPacketConfig(
            num_clients=num_clients,
            min_label_clients=min_label_clients,
            max_label_clients=receivers_cap,
            seed=partition_seed,
            lognormal_sigma=lognormal_sigma,
        )
        client_indices = red_packet_partition(pooled_labels, packet_config)
        assigned = np.concatenate(client_indices)
        if len(assigned) != pooled_rows or len(np.unique(assigned)) != pooled_rows:
            raise RuntimeError(
                "Red Packet partition is not an exact permutation of the pool"
            )
        del assigned

        client_train_indices: list[list[int]] = []
        client_validation_indices: list[list[int]] = []
        for client_id, indices in enumerate(client_indices):
            local_train, local_validation = deterministic_split_indices(
                len(indices),
                client_val_ratio,
                seed + client_id,
            )
            client_train_indices.append(local_train)
            client_validation_indices.append(local_validation)

        fit_pooled = np.concatenate(
            [
                indices[np.asarray(local_train, dtype=np.int64)]
                for indices, local_train in zip(
                    client_indices,
                    client_train_indices,
                    strict=True,
                )
            ]
        )
        fit_source_ids, fit_local_rows = _locate_pooled_indices(fit_pooled, offsets)
        fit_indices_by_source = [
            np.sort(fit_local_rows[fit_source_ids == source_id])
            for source_id in range(len(source_x_paths))
        ]
        del fit_pooled, fit_source_ids, fit_local_rows

        medians_array = _compute_feature_medians(
            source_x_paths,
            input_dim,
            fit_indices_by_source,
        )
        scaler = _fit_scaler_from_memmaps(
            source_x_paths,
            fit_indices_by_source,
            medians_array,
            block_size=chunk_size,
        )

        client_examples: list[int] = []
        client_class_counts: list[dict[str, int]] = []
        client_training_class_counts: list[dict[str, int]] = []
        training_label_arrays: list[np.ndarray] = []
        for client_id, indices in enumerate(client_indices):
            features = _gather_pooled_rows(
                source_x_paths,
                indices,
                offsets,
                dtype=np.float32,
                width=input_dim,
            )
            labels = pooled_labels[indices]
            scaled = scaler.transform(
                _fill_missing_with_medians(features, medians_array)
            ).astype(np.float32)
            np.savez_compressed(
                output_path / f"client_{client_id:03d}.npz",
                x=scaled,
                y=labels,
            )
            del features, scaled
            training_labels = labels[
                np.asarray(client_train_indices[client_id], dtype=np.int64)
            ]
            training_label_arrays.append(training_labels)
            client_examples.append(int(len(labels)))
            client_class_counts.append(class_counts(labels))
            client_training_class_counts.append(class_counts(training_labels))

        test_row_count = _count_csv_rows(global_test_path, label_column, chunk_size)
        test_x_path, test_y_path = _materialize_existing_split_csv(
            global_test_path,
            x_path=tmp_dir / "global_test_x.npy",
            y_path=tmp_dir / "global_test_y.npy",
            row_count=test_row_count,
            feature_columns=feature_columns,
            label_column=label_column,
            chunk_size=chunk_size,
        )
        _write_scaled_npz(
            x_path=test_x_path,
            y_path=test_y_path,
            output_path=output_path / "global_test.npz",
            scaled_x_path=tmp_dir / "global_test_scaled_x.npy",
            medians=medians_array,
            scaler=scaler,
            block_size=chunk_size,
        )
        test_labels_memmap = np.load(test_y_path, mmap_mode="r")
        try:
            global_test_examples = int(len(test_labels_memmap))
            global_test_class_counts = class_counts(np.asarray(test_labels_memmap))
        finally:
            _close_memmap(test_labels_memmap)

        statistics_artifacts = write_client_distribution_statistics(
            output_path,
            training_label_arrays,
            CICIOT2023_LABELS,
        )

        source_file_records = [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in [*client_files, global_test_path]
        ]
        prepared_test_path = output_path / "global_test.npz"
        with np.load(prepared_test_path) as prepared_test:
            prepared_test_x_sha256 = sha256_array(prepared_test["x"])
            prepared_test_y_sha256 = sha256_array(prepared_test["y"])

        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_dir": str(splits_path),
            "source_files": [str(path) for path in client_files]
            + [str(global_test_path)],
            "source_file_records": source_file_records,
            "partition_strategy": "red_packet_non_iid_repartition_of_existing_pool",
            "red_packet": asdict(packet_config),
            "source_partition": {
                "source_clients": source_clients,
                "source_client_files": [path.name for path in client_files],
                "source_client_rows": source_row_counts,
                "pooled_rows": pooled_rows,
                "pooled_row_order": (
                    "concatenation of the source client CSVs in ascending "
                    "client id, each file in its original row order"
                ),
                "rows_added_or_removed": 0,
                "global_test_rows_touched": False,
            },
            "num_clients": num_clients,
            "num_classes": len(CICIOT2023_LABELS),
            "labels": CICIOT2023_LABELS,
            "label_to_id": {label: idx for idx, label in enumerate(CICIOT2023_LABELS)},
            "label_column": label_column,
            "feature_columns": feature_columns,
            "input_dim": input_dim,
            "global_test_examples": global_test_examples,
            "global_test_class_counts": global_test_class_counts,
            "client_examples": client_examples,
            "client_class_counts": client_class_counts,
            "client_class_counts_scope": (
                "all_local_examples_including_client_validation; legacy summary field"
            ),
            "client_training_class_counts": client_training_class_counts,
            "client_training_class_counts_scope": (
                "deterministic_local_training_subsets; validation and global "
                "test excluded"
            ),
            "preprocessing": {
                "fit_scope": "deterministic_local_training_subsets_only",
                "excluded_from_fit": ["client_validation", "global_test"],
                "client_val_ratio": client_val_ratio,
                "seed": seed,
                "client_seed_rule": "seed + client_id",
                "split_algorithm": "torch.randperm deterministic ordered indices",
                "fit_examples": int(sum(map(len, client_train_indices))),
                "client_fit_examples": list(map(len, client_train_indices)),
                "client_validation_examples": list(
                    map(len, client_validation_indices)
                ),
                "transform_scope": [
                    "client_training",
                    "client_validation",
                    "global_test",
                ],
                "transform_statistics_source": (
                    "deterministic_local_training_subsets_only"
                ),
                "client_train_indices_sha256": [
                    sha256_indices(indices) for indices in client_train_indices
                ],
                "client_validation_indices_sha256": [
                    sha256_indices(indices) for indices in client_validation_indices
                ],
                "feature_encoder": {
                    "method": "shared_csv_header_numeric_schema",
                    "schema_fit_scope": (
                        "shared header of every source split; a categorical "
                        "column is rejected before any row is materialised"
                    ),
                    "unseen_category_policy": "not_applicable_numeric_only_schema",
                },
                "missing_value_imputer": {
                    "method": "per-feature median then zero fallback",
                    "fit_scope": "deterministic_local_training_subsets_only",
                },
                "scaler": {
                    "method": "sklearn.preprocessing.StandardScaler",
                    "fit_scope": "deterministic_local_training_subsets_only",
                },
            },
            "global_test_provenance": {
                "status": "complete",
                "role": "evaluation_only",
                "immutable": True,
                "source_kind": "existing_presplit_csv",
                "source_path": str(global_test_path.resolve()),
                "source_sha256": sha256_file(global_test_path),
                "prepared_file": prepared_test_path.name,
                "prepared_npz_sha256": sha256_file(prepared_test_path),
                "prepared_x_sha256": prepared_test_x_sha256,
                "prepared_y_sha256": prepared_test_y_sha256,
                "preprocessing_fit_includes_global_test": False,
            },
            "client_statistics": statistics_artifacts,
            "scaler_mean": scaler.mean_.astype(float).tolist(),
            "scaler_scale": scaler.scale_.astype(float).tolist(),
            "feature_medians": {
                column: float(value)
                for column, value in zip(feature_columns, medians_array, strict=True)
            },
            "chunk_size": chunk_size,
        }
        (output_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return metadata


def prepare_existing_ciciot2023_splits(
    splits_dir: str | Path,
    output_dir: str | Path,
    *,
    label_column: str = "label",
    client_pattern: str = "client_*.csv",
    global_test_file: str = "global_test.csv",
    expected_clients: int = 10,
    chunk_size: int | None = 100_000,
    client_val_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    """Convert pre-split CSVs without fitting preprocessing on evaluation rows."""
    splits_path = Path(splits_dir)
    client_files = sorted_client_csv_files(splits_path, client_pattern)
    if len(client_files) != expected_clients:
        raise ValueError(
            f"Expected {expected_clients} client CSV files, found {len(client_files)}"
        )
    global_test_path = splits_path / global_test_file
    if not global_test_path.exists():
        raise FileNotFoundError(f"Global test CSV not found: {global_test_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    (
        feature_columns,
        row_counts,
        client_train_indices,
        client_validation_indices,
    ) = _discover_existing_split_schema(
        client_files,
        global_test_path,
        label_column=label_column,
        chunk_size=chunk_size,
        client_val_ratio=client_val_ratio,
        seed=seed,
    )
    input_dim = len(feature_columns)

    with tempfile.TemporaryDirectory(
        prefix=".ids_fed_prepare_",
        dir=output_path,
    ) as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        client_x_paths: list[Path] = []
        client_y_paths: list[Path] = []
        for client_id, client_file in enumerate(client_files):
            x_path, y_path = _materialize_existing_split_csv(
                client_file,
                x_path=tmp_dir / f"client_{client_id:03d}_x.npy",
                y_path=tmp_dir / f"client_{client_id:03d}_y.npy",
                row_count=row_counts[client_file],
                feature_columns=feature_columns,
                label_column=label_column,
                chunk_size=chunk_size,
            )
            client_x_paths.append(x_path)
            client_y_paths.append(y_path)

        medians_array = _compute_feature_medians(
            client_x_paths,
            input_dim,
            client_train_indices,
        )
        scaler = _fit_scaler_from_memmaps(
            client_x_paths,
            client_train_indices,
            medians_array,
            block_size=chunk_size,
        )

        for client_id, (x_path, y_path) in enumerate(
            zip(client_x_paths, client_y_paths, strict=True)
        ):
            _write_scaled_npz(
                x_path=x_path,
                y_path=y_path,
                output_path=output_path / f"client_{client_id:03d}.npz",
                scaled_x_path=tmp_dir / f"client_{client_id:03d}_scaled_x.npy",
                medians=medians_array,
                scaler=scaler,
                block_size=chunk_size,
            )
            x_path.unlink()

        test_x_path, test_y_path = _materialize_existing_split_csv(
            global_test_path,
            x_path=tmp_dir / "global_test_x.npy",
            y_path=tmp_dir / "global_test_y.npy",
            row_count=row_counts[global_test_path],
            feature_columns=feature_columns,
            label_column=label_column,
            chunk_size=chunk_size,
        )
        _write_scaled_npz(
            x_path=test_x_path,
            y_path=test_y_path,
            output_path=output_path / "global_test.npz",
            scaled_x_path=tmp_dir / "global_test_scaled_x.npy",
            medians=medians_array,
            scaler=scaler,
            block_size=chunk_size,
        )
        test_x_path.unlink()

        y_clients = [np.load(path, mmap_mode="r") for path in client_y_paths]
        y_test = np.load(test_y_path, mmap_mode="r")
        try:
            statistics_artifacts = write_client_distribution_statistics(
                output_path,
                [
                    labels[np.asarray(indices, dtype=np.int64)]
                    for labels, indices in zip(
                        y_clients,
                        client_train_indices,
                        strict=True,
                    )
                ],
                CICIOT2023_LABELS,
            )
            global_test_examples = int(len(y_test))
            global_test_class_counts = class_counts(y_test)
            client_examples = [int(len(y_client)) for y_client in y_clients]
            client_class_counts = [class_counts(y_client) for y_client in y_clients]
            client_training_class_counts = [
                class_counts(labels[np.asarray(indices, dtype=np.int64)])
                for labels, indices in zip(
                    y_clients,
                    client_train_indices,
                    strict=True,
                )
            ]
        finally:
            for memmap in [*y_clients, y_test]:
                _close_memmap(memmap)
        for y_path in [*client_y_paths, test_y_path]:
            y_path.unlink()

        source_file_records = [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in client_files + [global_test_path]
        ]
        prepared_test_path = output_path / "global_test.npz"
        with np.load(prepared_test_path) as prepared_test:
            prepared_test_x_sha256 = sha256_array(prepared_test["x"])
            prepared_test_y_sha256 = sha256_array(prepared_test["y"])

        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_dir": str(splits_path),
            "source_files": [str(path) for path in client_files]
            + [str(global_test_path)],
            "source_file_records": source_file_records,
            "partition_strategy": "existing_presplit_clients",
            "num_clients": expected_clients,
            "num_classes": len(CICIOT2023_LABELS),
            "labels": CICIOT2023_LABELS,
            "label_to_id": {label: idx for idx, label in enumerate(CICIOT2023_LABELS)},
            "label_column": label_column,
            "feature_columns": feature_columns,
            "input_dim": input_dim,
            "global_test_examples": global_test_examples,
            "global_test_class_counts": global_test_class_counts,
            "client_examples": client_examples,
            "client_class_counts": client_class_counts,
            "client_class_counts_scope": (
                "all_local_examples_including_client_validation; legacy summary field"
            ),
            "client_training_class_counts": client_training_class_counts,
            "client_training_class_counts_scope": (
                "deterministic_local_training_subsets; validation and global test excluded"
            ),
            "preprocessing": {
                "fit_scope": "deterministic_local_training_subsets_only",
                "excluded_from_fit": ["client_validation", "global_test"],
                "client_val_ratio": client_val_ratio,
                "seed": seed,
                "client_seed_rule": "seed + client_id",
                "split_algorithm": "torch.randperm deterministic ordered indices",
                "fit_examples": int(sum(map(len, client_train_indices))),
                "client_fit_examples": list(map(len, client_train_indices)),
                "client_validation_examples": list(
                    map(len, client_validation_indices)
                ),
                "transform_scope": [
                    "client_training",
                    "client_validation",
                    "global_test",
                ],
                "transform_statistics_source": "deterministic_local_training_subsets_only",
                "client_train_indices_sha256": [
                    sha256_indices(indices) for indices in client_train_indices
                ],
                "client_validation_indices_sha256": [
                    sha256_indices(indices) for indices in client_validation_indices
                ],
                "feature_encoder": {
                    "method": "pandas.get_dummies",
                    "schema_fit_scope": "deterministic_local_training_subsets_only",
                    "unseen_category_policy": "all-zero known dummy columns",
                },
                "missing_value_imputer": {
                    "method": "per-feature median then zero fallback",
                    "fit_scope": "deterministic_local_training_subsets_only",
                },
                "scaler": {
                    "method": "sklearn.preprocessing.StandardScaler",
                    "fit_scope": "deterministic_local_training_subsets_only",
                },
            },
            "global_test_provenance": {
                "status": "complete",
                "role": "evaluation_only",
                "immutable": True,
                "source_kind": "existing_presplit_csv",
                "source_path": str(global_test_path.resolve()),
                "source_sha256": sha256_file(global_test_path),
                "prepared_file": prepared_test_path.name,
                "prepared_npz_sha256": sha256_file(prepared_test_path),
                "prepared_x_sha256": prepared_test_x_sha256,
                "prepared_y_sha256": prepared_test_y_sha256,
                "preprocessing_fit_includes_global_test": False,
            },
            "client_statistics": statistics_artifacts,
            "scaler_mean": scaler.mean_.astype(float).tolist(),
            "scaler_scale": scaler.scale_.astype(float).tolist(),
            "feature_medians": {
                column: float(value)
                for column, value in zip(feature_columns, medians_array, strict=True)
            },
            "chunk_size": chunk_size,
        }
        (output_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return metadata


def prepare_ciciot2023_partitions(
    raw_data: str | Path,
    output_dir: str | Path,
    *,
    label_column: str = "label",
    num_clients: int = 10,
    global_test_ratio: float = 0.2,
    seed: int = 42,
    min_label_clients: int = 1,
    max_label_clients: int = 8,
    sample_fraction: float | None = None,
    max_rows: int | None = None,
    client_val_ratio: float = 0.2,
) -> dict:
    """Create Red Packet non-IID client partitions and a global test split."""
    if num_clients != 10:
        raise ValueError("This project is configured for exactly 10 simulated clients")
    csv_files = discover_csv_files(raw_data)
    frame = read_ciciot2023_csvs(csv_files, label_column, max_rows=max_rows)
    if sample_fraction is not None:
        if not 0.0 < sample_fraction <= 1.0:
            raise ValueError("sample_fraction must be in (0, 1]")
        frame = frame.sample(frac=sample_fraction, random_state=seed).reset_index(drop=True)

    y = encode_labels(frame[label_column], strict=True)
    train_idx, test_idx = stratified_global_test_split(y, global_test_ratio, seed)
    y_train = y[train_idx]
    y_test = y[test_idx]

    packet_config = RedPacketConfig(
        num_clients=num_clients,
        min_label_clients=min_label_clients,
        max_label_clients=max_label_clients,
        seed=seed,
    )
    client_indices = red_packet_partition(y_train, packet_config)
    client_train_indices: list[list[int]] = []
    client_validation_indices: list[list[int]] = []
    preprocessing_fit_indices: list[np.ndarray] = []
    for client_id, indices in enumerate(client_indices):
        local_train, local_validation = deterministic_split_indices(
            len(indices),
            client_val_ratio,
            seed + client_id,
        )
        client_train_indices.append(local_train)
        client_validation_indices.append(local_validation)
        preprocessing_fit_indices.append(
            indices[np.asarray(local_train, dtype=np.int64)]
        )
    fit_train_indices = np.concatenate(preprocessing_fit_indices)

    # Phase 1: fit the feature schema, imputer, and scaler on the exact union
    # of deterministic local-training rows.  Validation and global-test rows
    # are not read by any fitting operation below.
    fit_source_rows = train_idx[fit_train_indices]
    x_fit_frame = encode_features(frame.iloc[fit_source_rows], label_column)
    feature_columns = _feature_columns_from_training_frame(
        frame.iloc[fit_source_rows],
        label_column,
    )
    x_fit_frame = x_fit_frame.reindex(columns=feature_columns, fill_value=0.0)
    medians = _fit_frame_medians(x_fit_frame)
    x_fit_frame = _fill_frame_with_medians(x_fit_frame, medians)
    scaler = StandardScaler()
    scaler.fit(x_fit_frame)

    # Phase 2: transform each client's train and validation rows separately
    # with the already-fitted statistics.  The project NPZ contract stores the
    # two roles together in client_<id>.npz; deterministic indices in the
    # manifest/runner recover the roles without allowing validation rows into
    # the fit above.
    x_train = np.empty((len(train_idx), len(feature_columns)), dtype=np.float32)
    transformed_rows = np.zeros(len(train_idx), dtype=bool)
    for client_id, indices in enumerate(client_indices):
        source_rows = train_idx[np.asarray(indices, dtype=np.int64)]
        client_frame = frame.iloc[source_rows]
        client_encoded = _transform_feature_frame(
            client_frame,
            label_column,
            feature_columns,
        )
        local_train = np.asarray(client_train_indices[client_id], dtype=np.int64)
        local_validation = np.asarray(
            client_validation_indices[client_id],
            dtype=np.int64,
        )
        # Transforming the roles in separate calls makes the fit/transform
        # boundary explicit and guards against accidental partial_fit changes.
        train_frame = _fill_frame_with_medians(
            client_encoded.iloc[local_train],
            medians,
        )
        validation_frame = _fill_frame_with_medians(
            client_encoded.iloc[local_validation],
            medians,
        )
        client_positions = np.asarray(indices, dtype=np.int64)
        train_positions = client_positions[local_train]
        validation_positions = client_positions[local_validation]
        x_train[train_positions] = scaler.transform(train_frame).astype(np.float32)
        x_train[validation_positions] = scaler.transform(validation_frame).astype(
            np.float32
        )
        transformed_rows[train_positions] = True
        transformed_rows[validation_positions] = True

    if not transformed_rows.all():
        raise RuntimeError("Some client rows were not transformed after the split")

    # The global test is transformed only after fitting and is never passed to
    # either the median or StandardScaler fit.
    x_test_frame = _transform_feature_frame(
        frame.iloc[test_idx],
        label_column,
        feature_columns,
    )
    x_test = scaler.transform(
        _fill_frame_with_medians(x_test_frame, medians)
    ).astype(np.float32)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for client_id, indices in enumerate(client_indices):
        np.savez_compressed(
            output_path / f"client_{client_id:03d}.npz",
            x=x_train[indices],
            y=y_train[indices],
        )
    np.savez_compressed(output_path / "global_test.npz", x=x_test, y=y_test)
    statistics_artifacts = write_client_distribution_statistics(
        output_path,
        [
            y_train[indices[np.asarray(local_train, dtype=np.int64)]]
            for indices, local_train in zip(
                client_indices,
                client_train_indices,
                strict=True,
            )
        ],
        CICIOT2023_LABELS,
    )
    source_file_records = [
        {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for path in csv_files
    ]
    prepared_test_path = output_path / "global_test.npz"

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_files": [str(path) for path in csv_files],
        "source_file_records": source_file_records,
        "partition_strategy": "red_packet_non_iid",
        "red_packet": asdict(packet_config),
        "num_clients": num_clients,
        "num_classes": len(CICIOT2023_LABELS),
        "labels": CICIOT2023_LABELS,
        "label_to_id": {label: idx for idx, label in enumerate(CICIOT2023_LABELS)},
        "label_column": label_column,
        "feature_columns": feature_columns,
        "input_dim": int(x_train.shape[1]),
        "global_test_ratio": global_test_ratio,
        "seed": seed,
        "sample_fraction": sample_fraction,
        "max_rows": max_rows,
        "train_examples": int(len(y_train)),
        "global_test_examples": int(len(y_test)),
        "global_test_class_counts": class_counts(y_test),
        "client_examples": [int(len(indices)) for indices in client_indices],
        "client_class_counts": [class_counts(y_train[indices]) for indices in client_indices],
        "client_class_counts_scope": (
            "all_local_examples_including_client_validation; legacy summary field"
        ),
        "client_training_class_counts": [
            class_counts(
                y_train[indices[np.asarray(local_train, dtype=np.int64)]]
            )
            for indices, local_train in zip(
                client_indices,
                client_train_indices,
                strict=True,
            )
        ],
        "client_training_class_counts_scope": (
            "deterministic_local_training_subsets; validation and global test excluded"
        ),
        "preprocessing": {
            "fit_scope": "deterministic_local_training_subsets_only",
            "excluded_from_fit": ["client_validation", "global_test"],
            "client_val_ratio": client_val_ratio,
            "seed": seed,
            "client_seed_rule": "seed + client_id",
            "split_algorithm": "torch.randperm deterministic ordered indices",
            "fit_examples": int(len(fit_train_indices)),
            "client_fit_examples": list(map(len, client_train_indices)),
            "client_validation_examples": list(
                map(len, client_validation_indices)
            ),
            "transform_scope": [
                "client_training",
                "client_validation",
                "global_test",
            ],
            "transform_statistics_source": "deterministic_local_training_subsets_only",
            "client_train_indices_sha256": [
                sha256_indices(indices) for indices in client_train_indices
            ],
            "client_validation_indices_sha256": [
                sha256_indices(indices) for indices in client_validation_indices
            ],
            "feature_encoder": {
                "method": "pandas.get_dummies",
                "schema_fit_scope": "deterministic_local_training_subsets_only",
                "unseen_category_policy": "all-zero known dummy columns",
            },
            "missing_value_imputer": {
                "method": "per-feature median then zero fallback",
                "fit_scope": "deterministic_local_training_subsets_only",
            },
            "scaler": {
                "method": "sklearn.preprocessing.StandardScaler",
                "fit_scope": "deterministic_local_training_subsets_only",
            },
        },
        "global_test_provenance": {
            "status": "complete",
            "role": "evaluation_only",
            "immutable": True,
            "source_kind": "stratified_split_from_source_csvs",
            "source_file_records": source_file_records,
            "selection_indices_sha256": sha256_indices(test_idx),
            "selection_indices_coordinate_system": (
                "concatenated rows after optional deterministic sampling"
            ),
            "split_seed": seed,
            "split_ratio": global_test_ratio,
            "prepared_file": prepared_test_path.name,
            "prepared_npz_sha256": sha256_file(prepared_test_path),
            "prepared_x_sha256": sha256_array(x_test),
            "prepared_y_sha256": sha256_array(y_test),
            "preprocessing_fit_includes_global_test": False,
        },
        "client_statistics": statistics_artifacts,
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "feature_medians": {
            column: float(value)
            for column, value in zip(feature_columns, medians, strict=True)
        },
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata
