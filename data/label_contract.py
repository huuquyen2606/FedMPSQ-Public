"""Auditable CICIoT2023 source-label and eight-group task contract."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from data.ciciot2023 import CICIOT2023_LABELS
from data.fedmpsq_labels import (
    FEDMPSQ_CLASS_NAMES,
    canonical_sha256,
    mapping_records,
)

LABEL_CONTRACT_SCHEMA_VERSION = 1
MAPPING_CSV_FILENAME = "ciciot2023_34_to_8_mapping.csv"
MAPPING_JSON_FILENAME = "ciciot2023_34_to_8_mapping.json"
GROUPED_COUNTS_FILENAME = "ciciot2023_existing10_grouped_counts.csv"
AUDIT_JSON_FILENAME = "ciciot2023_label_contract_audit.json"


def explicit_mapping_rows() -> list[dict[str, int | str]]:
    """Return the frozen mapping with unambiguous source and target fields."""
    return [
        {
            "source_label_id": int(record["source_id"]),
            "source_label": str(record["source_name"]),
            "target_label_id": int(record["target_id"]),
            "target_label": str(record["target_name"]),
        }
        for record in mapping_records("ciciot2023_34")
    ]


def validate_mapping_rows(
    rows: list[dict[str, int | str]],
) -> dict[str, Any]:
    """Fail closed when either label space or any mapping ID drifts."""
    if len(CICIOT2023_LABELS) != 34 or len(set(CICIOT2023_LABELS)) != 34:
        raise ValueError("CICIoT2023 source schema must contain 34 unique labels")
    if len(FEDMPSQ_CLASS_NAMES) != 8 or len(set(FEDMPSQ_CLASS_NAMES)) != 8:
        raise ValueError("Target schema must contain eight unique groups")
    if len(rows) != len(CICIOT2023_LABELS):
        raise ValueError("The 34-to-8 mapping must contain exactly 34 rows")

    source_ids = [int(row["source_label_id"]) for row in rows]
    source_labels = [str(row["source_label"]) for row in rows]
    target_ids = [int(row["target_label_id"]) for row in rows]
    target_labels = [str(row["target_label"]) for row in rows]
    if source_ids != list(range(34)):
        raise ValueError("source_label_id must be the ordered range [0, 34)")
    if source_labels != list(CICIOT2023_LABELS):
        raise ValueError("source_label order differs from CICIOT2023_LABELS")
    if min(target_ids) != 0 or max(target_ids) != 7:
        raise ValueError("target_label_id must remain within [0, 8)")
    if set(target_ids) != set(range(8)):
        raise ValueError("Every target_label_id in [0, 8) must be represented")
    for target_id, target_label in zip(target_ids, target_labels, strict=True):
        if target_label != FEDMPSQ_CLASS_NAMES[target_id]:
            raise ValueError("target label name and ID differ")

    return {
        "source_num_classes": 34,
        "target_num_classes": 8,
        "implementation_mapping_sha256": canonical_sha256(
            mapping_records("ciciot2023_34")
        ),
        "explicit_mapping_sha256": canonical_sha256(rows),
    }


def expected_split_paths(
    splits_dir: str | Path,
    *,
    expected_clients: int,
) -> list[Path]:
    root = Path(splits_dir)
    return [
        *(root / f"client_{client_id}.csv" for client_id in range(expected_clients)),
        root / "global_test.csv",
    ]


def audit_split_headers(
    splits_dir: str | Path,
    *,
    expected_clients: int = 10,
    label_column: str = "label",
) -> dict[str, Any]:
    """Require one shared feature schema and one final label column."""
    files = expected_split_paths(splits_dir, expected_clients=expected_clients)
    records: list[dict[str, Any]] = []
    reference_features: list[str] | None = None
    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"Required split CSV not found: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
        if not header:
            raise ValueError(f"CSV has no header: {path}")
        if header.count(label_column) != 1:
            raise ValueError(
                f"CSV must contain exactly one '{label_column}' column: {path}"
            )
        if header[-1] != label_column:
            raise ValueError(f"'{label_column}' must be the final column: {path}")
        features = header[:-1]
        if len(features) != len(set(features)):
            raise ValueError(f"Duplicate feature names in {path}")
        if reference_features is None:
            reference_features = features
        elif features != reference_features:
            raise ValueError(f"Feature schema differs in {path}")
        records.append(
            {
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "num_features": len(features),
                "label_column": label_column,
                "feature_schema_sha256": canonical_sha256(features),
            }
        )
    if reference_features is None:
        raise ValueError("No split CSV files were audited")
    return {
        "expected_clients": expected_clients,
        "files": records,
        "feature_names": reference_features,
        "num_features": len(reference_features),
        "feature_schema_sha256": canonical_sha256(reference_features),
    }


def audit_statistics_file(
    statistics_file: str | Path,
    rows: list[dict[str, int | str]],
) -> tuple[dict[str, Any], list[dict[str, int | str]]]:
    """Validate source-label columns and aggregate counts into eight groups."""
    path = Path(statistics_file)
    if not path.exists():
        raise FileNotFoundError(f"Statistics CSV not found: {path}")
    mapping_by_label = {
        str(row["source_label"]): int(row["target_label_id"])
        for row in rows
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("Statistics CSV has no header")
        partition_column = fieldnames[0]
        count_columns = [
            field
            for field in fieldnames[1:]
            if field and field != "TOTAL_SAMPLES"
        ]
        missing = sorted(set(CICIOT2023_LABELS) - set(count_columns))
        unknown = sorted(set(count_columns) - set(CICIOT2023_LABELS))
        if missing or unknown or len(count_columns) != 34:
            raise ValueError(
                "Statistics label schema differs from the canonical 34 labels: "
                f"missing={missing}, unknown={unknown}"
            )

        grouped_rows: list[dict[str, int | str]] = []
        source_counts_by_partition: dict[str, dict[str, int]] = {}
        for input_row in reader:
            partition = str(input_row.get(partition_column, "")).strip()
            if not partition:
                raise ValueError("Statistics row has an empty partition name")
            if partition in source_counts_by_partition:
                raise ValueError(f"Duplicate statistics partition: {partition}")
            grouped = [0] * len(FEDMPSQ_CLASS_NAMES)
            source_total = 0
            source_counts: dict[str, int] = {}
            for label in CICIOT2023_LABELS:
                raw_value = input_row.get(label)
                try:
                    count = int(raw_value) if raw_value is not None else -1
                except ValueError as error:
                    raise ValueError(
                        f"Invalid count for {partition}/{label}: {raw_value}"
                    ) from error
                if count < 0:
                    raise ValueError(f"Negative count for {partition}/{label}")
                source_counts[label] = count
                grouped[mapping_by_label[label]] += count
                source_total += count
            declared_total = int(input_row.get("TOTAL_SAMPLES", source_total))
            if declared_total != source_total:
                raise ValueError(
                    f"TOTAL_SAMPLES differs from source counts for {partition}"
                )
            output_row: dict[str, int | str] = {"partition": partition}
            output_row.update(
                {
                    target_name: grouped[target_id]
                    for target_id, target_name in enumerate(FEDMPSQ_CLASS_NAMES)
                }
            )
            output_row["total_samples"] = source_total
            if sum(grouped) != source_total:
                raise AssertionError("Grouped counts do not preserve the row total")
            grouped_rows.append(output_row)
            source_counts_by_partition[partition] = source_counts

    if not grouped_rows:
        raise ValueError("Statistics CSV contains no data rows")
    return (
        {
            "source": "statistics_csv",
            "file": str(path.resolve()),
            "partition_column": partition_column,
            "observed_source_labels": count_columns,
            "observed_source_label_count": len(count_columns),
            "row_count": len(grouped_rows),
            "grouped_total_preserved": True,
            "source_counts_by_partition": source_counts_by_partition,
        },
        grouped_rows,
    )


def scan_raw_label_values(
    splits_dir: str | Path,
    *,
    expected_clients: int = 10,
    label_column: str = "label",
    chunk_size: int = 500_000,
) -> dict[str, Any]:
    """Optionally scan the label column of all split CSVs in bounded memory."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    files = expected_split_paths(splits_dir, expected_clients=expected_clients)
    expected = set(CICIOT2023_LABELS)
    observed_union: set[str] = set()
    file_records: list[dict[str, Any]] = []
    for file_index, path in enumerate(files):
        partition = (
            f"Client_{file_index}"
            if file_index < expected_clients
            else "Global_Test"
        )
        counts: Counter[str] = Counter()
        for chunk in pd.read_csv(
            path,
            usecols=[label_column],
            chunksize=chunk_size,
        ):
            labels = chunk[label_column]
            if labels.isna().any():
                raise ValueError(f"Null labels found in {path}")
            counts.update(labels.astype(str).tolist())
        observed = set(counts)
        unknown = sorted(observed - expected)
        if unknown:
            raise ValueError(f"Unknown raw labels in {path}: {unknown}")
        observed_union.update(observed)
        file_records.append(
            {
                "file": path.name,
                "partition": partition,
                "rows_scanned": int(sum(counts.values())),
                "observed_labels": sorted(observed),
                "class_counts": dict(sorted(counts.items())),
            }
        )
    missing_union = sorted(expected - observed_union)
    if missing_union:
        raise ValueError(
            "The union of raw CSV labels does not contain all 34 labels: "
            f"{missing_union}"
        )
    return {
        "performed": True,
        "chunk_size": chunk_size,
        "observed_union": sorted(observed_union),
        "files": file_records,
    }


def derive_statistics_from_raw_label_audit(
    raw_label_audit: dict[str, Any],
    rows: list[dict[str, int | str]],
    *,
    requested_statistics_file: str | Path,
) -> tuple[dict[str, Any], list[dict[str, int | str]]]:
    """Derive grouped counts from a required full raw-label scan."""
    if not raw_label_audit.get("performed"):
        raise ValueError(
            "Missing statistics CSV can only be replaced by a full raw-label scan"
        )
    mapping_by_label = {
        str(row["source_label"]): int(row["target_label_id"])
        for row in rows
    }
    grouped_rows: list[dict[str, int | str]] = []
    source_counts_by_partition: dict[str, dict[str, int]] = {}
    for record in raw_label_audit["files"]:
        partition = str(record["partition"])
        if partition in source_counts_by_partition:
            raise ValueError(f"Duplicate raw-scan partition: {partition}")
        observed_counts = {
            str(label): int(count)
            for label, count in dict(record["class_counts"]).items()
        }
        source_counts = {
            label: int(observed_counts.get(label, 0))
            for label in CICIOT2023_LABELS
        }
        source_total = sum(source_counts.values())
        if source_total != int(record["rows_scanned"]):
            raise ValueError(f"Raw-scan row total differs for {partition}")
        grouped = [0] * len(FEDMPSQ_CLASS_NAMES)
        for label, count in source_counts.items():
            grouped[mapping_by_label[label]] += count
        output_row: dict[str, int | str] = {"partition": partition}
        output_row.update(
            {
                target_name: grouped[target_id]
                for target_id, target_name in enumerate(FEDMPSQ_CLASS_NAMES)
            }
        )
        output_row["total_samples"] = source_total
        if sum(grouped) != source_total:
            raise AssertionError("Grouped raw counts do not preserve the row total")
        grouped_rows.append(output_row)
        source_counts_by_partition[partition] = source_counts

    return (
        {
            "source": "raw_label_scan",
            "file": None,
            "requested_file": str(Path(requested_statistics_file).resolve()),
            "partition_column": "partition",
            "observed_source_labels": list(CICIOT2023_LABELS),
            "observed_source_label_count": len(CICIOT2023_LABELS),
            "row_count": len(grouped_rows),
            "grouped_total_preserved": True,
            "source_counts_by_partition": source_counts_by_partition,
        },
        grouped_rows,
    )


def audit_raw_statistics_consistency(
    raw_label_audit: dict[str, Any],
    statistics_audit: dict[str, Any],
) -> dict[str, Any]:
    """Require raw 34-label counts to match the supplied statistics exactly."""
    if not raw_label_audit.get("performed"):
        return {"performed": False}

    raw_counts_by_partition = {
        str(record["partition"]): {
            str(label): int(count)
            for label, count in dict(record["class_counts"]).items()
        }
        for record in raw_label_audit["files"]
    }
    statistics_counts_by_partition = {
        str(partition): {
            str(label): int(count)
            for label, count in dict(counts).items()
        }
        for partition, counts in dict(
            statistics_audit["source_counts_by_partition"]
        ).items()
    }
    if set(raw_counts_by_partition) != set(statistics_counts_by_partition):
        raise ValueError(
            "Raw split partitions differ from statistics partitions: "
            f"raw={sorted(raw_counts_by_partition)}, "
            f"statistics={sorted(statistics_counts_by_partition)}"
        )

    mismatches: list[str] = []
    total_rows = 0
    for partition in sorted(raw_counts_by_partition):
        raw_counts = raw_counts_by_partition[partition]
        statistics_counts = statistics_counts_by_partition[partition]
        for label in CICIOT2023_LABELS:
            raw_count = int(raw_counts.get(label, 0))
            statistics_count = int(statistics_counts.get(label, 0))
            if raw_count != statistics_count:
                mismatches.append(
                    f"{partition}/{label}: raw={raw_count}, "
                    f"statistics={statistics_count}"
                )
        total_rows += sum(raw_counts.values())
    if mismatches:
        preview = "; ".join(mismatches[:10])
        raise ValueError(f"Raw label counts differ from statistics: {preview}")

    return {
        "performed": True,
        "matched": True,
        "partition_count": len(raw_counts_by_partition),
        "source_label_count": len(CICIOT2023_LABELS),
        "rows_compared": total_rows,
    }


def build_label_contract() -> dict[str, Any]:
    rows = explicit_mapping_rows()
    mapping_audit = validate_mapping_rows(rows)
    source_schema = {
        "name": "ciciot2023_34",
        "num_classes": 34,
        "labels": list(CICIOT2023_LABELS),
        "label_to_id": {
            label: label_id for label_id, label in enumerate(CICIOT2023_LABELS)
        },
    }
    target_schema = {
        "name": "ciciot2023_groups_8",
        "num_classes": 8,
        "labels": list(FEDMPSQ_CLASS_NAMES),
        "label_to_id": {
            label: label_id
            for label_id, label in enumerate(FEDMPSQ_CLASS_NAMES)
        },
    }
    contract = {
        "schema_version": LABEL_CONTRACT_SCHEMA_VERSION,
        "task": "CICIoT2023_Benign_plus_7_attack_groups",
        "encoding_pipeline": [
            "raw label string -> source_label_id in [0, 34)",
            "prepared NPZ y stores source_label_id",
            "task loader remaps source_label_id -> target_label_id in [0, 8)",
            "model output and metrics use target_label_id",
        ],
        "source_label_space": source_schema,
        "target_label_space": target_schema,
        "mapping": rows,
        **mapping_audit,
        "source_schema_sha256": canonical_sha256(source_schema),
        "target_schema_sha256": canonical_sha256(target_schema),
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    return contract


def write_label_contract_artifacts(
    *,
    splits_dir: str | Path,
    statistics_file: str | Path,
    output_dir: str | Path,
    expected_clients: int = 10,
    label_column: str = "label",
    scan_raw_labels: bool = False,
    chunk_size: int = 500_000,
    allow_missing_statistics: bool = False,
) -> dict[str, Any]:
    """Audit current inputs and write small versionable contract artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    contract = build_label_contract()
    rows = contract["mapping"]
    header_audit = audit_split_headers(
        splits_dir,
        expected_clients=expected_clients,
        label_column=label_column,
    )
    statistics_path = Path(statistics_file)
    if not statistics_path.exists() and not allow_missing_statistics:
        raise FileNotFoundError(f"Statistics CSV not found: {statistics_path}")
    if not statistics_path.exists() and not scan_raw_labels:
        raise ValueError(
            "--allow-missing-statistics requires --scan-raw-labels so counts can "
            "be derived from the complete raw splits"
        )
    if statistics_path.exists():
        statistics_audit, grouped_rows = audit_statistics_file(statistics_path, rows)
    raw_label_audit = (
        scan_raw_label_values(
            splits_dir,
            expected_clients=expected_clients,
            label_column=label_column,
            chunk_size=chunk_size,
        )
        if scan_raw_labels
        else {"performed": False}
    )
    if statistics_path.exists():
        raw_statistics_consistency = audit_raw_statistics_consistency(
            raw_label_audit,
            statistics_audit,
        )
    else:
        statistics_audit, grouped_rows = derive_statistics_from_raw_label_audit(
            raw_label_audit,
            rows,
            requested_statistics_file=statistics_path,
        )
        raw_statistics_consistency = {
            "performed": False,
            "matched": None,
            "reason": "statistics_csv_missing_grouped_counts_derived_from_raw_scan",
            "raw_rows_used": sum(
                int(record["rows_scanned"])
                for record in raw_label_audit["files"]
            ),
        }

    mapping_csv = output_path / MAPPING_CSV_FILENAME
    with mapping_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    mapping_json = output_path / MAPPING_JSON_FILENAME
    mapping_json.write_text(
        json.dumps(contract, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    grouped_counts_csv = output_path / GROUPED_COUNTS_FILENAME
    grouped_fields = ["partition", *FEDMPSQ_CLASS_NAMES, "total_samples"]
    with grouped_counts_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=grouped_fields)
        writer.writeheader()
        writer.writerows(grouped_rows)

    audit = {
        "schema_version": LABEL_CONTRACT_SCHEMA_VERSION,
        "status": "passed",
        "contract_sha256": contract["contract_sha256"],
        "implementation_mapping_sha256": contract[
            "implementation_mapping_sha256"
        ],
        "explicit_mapping_sha256": contract["explicit_mapping_sha256"],
        "split_header_audit": header_audit,
        "statistics_audit": statistics_audit,
        "raw_label_scan": raw_label_audit,
        "raw_statistics_consistency": raw_statistics_consistency,
        "artifacts": {
            "mapping_csv": mapping_csv.name,
            "mapping_json": mapping_json.name,
            "grouped_counts_csv": grouped_counts_csv.name,
        },
    }
    audit_json = output_path / AUDIT_JSON_FILENAME
    audit_json.write_text(
        json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return audit
