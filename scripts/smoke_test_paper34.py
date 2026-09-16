#!/usr/bin/env python
"""Run one synthetic round of every paper34 scenario at both client counts.

A full campaign run costs hours of GPU time, so every configuration is first
exercised end to end on a tiny synthetic partition: same code paths, same
frozen contracts, and the full classification-metric schema. A failure here is a
configuration bug caught before a session is spent on it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.ciciot2023 import CICIOT2023_LABELS
from data.client_statistics import write_client_distribution_statistics
from data.integrity import sha256_array, sha256_file, sha256_indices
from data.partition_manifest import (
    SUPPORTED_CLIENT_COUNTS,
    create_partition_manifest,
)
from data.splits import deterministic_split_indices
from scripts.audit_paper34_run import audit_run  # noqa: E402
from scripts.paper_runs import (  # noqa: E402
    PAPER_SCENARIOS,
    RUN_BY_SLUG,
    canonical_run_name,
    scenario_family,
)
from training.metrics import CLASSIFICATION_METRIC_KEYS

NUM_CLASSES = len(CICIOT2023_LABELS)
INPUT_DIM = 8
ROWS_PER_CLIENT = 80


def config_path(scenario: str) -> Path:
    try:
        filename = RUN_BY_SLUG[scenario].config_filename
    except KeyError as exc:
        raise ValueError(f"Unsupported paper scenario: {scenario}") from exc
    return PROJECT_ROOT / "configs" / "paper34_v1" / filename


def trainer_script(scenario: str) -> str:
    family = scenario_family(scenario)
    if family == "baseline":
        return "train_local_federated.py"
    if family == "proposed":
        return "train_fedmpsq.py"
    raise ValueError(f"Unsupported paper scenario: {scenario}")


def _write_synthetic_partition(base_dir: Path, num_clients: int) -> None:
    """Write a miniature prepared partition with all 34 classes represented."""
    base_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    train_hashes: list[str] = []
    validation_hashes: list[str] = []
    client_examples: list[int] = []
    client_training_labels: list[np.ndarray] = []
    for client_id in range(num_clients):
        features = rng.normal(size=(ROWS_PER_CLIENT, INPUT_DIM)).astype(np.float32)
        # Every class appears somewhere, and each client still misses many of
        # them, which is what the metric contract has to cope with.
        labels = np.concatenate(
            [
                np.arange(NUM_CLASSES, dtype=np.int64),
                rng.integers(
                    0,
                    NUM_CLASSES,
                    size=ROWS_PER_CLIENT - NUM_CLASSES,
                    dtype=np.int64,
                ),
            ]
        )
        rng.shuffle(labels)
        np.savez_compressed(
            base_dir / f"client_{client_id:03d}.npz",
            x=features,
            y=labels,
        )
        train_indices, validation_indices = deterministic_split_indices(
            ROWS_PER_CLIENT,
            0.2,
            42 + client_id,
        )
        train_hashes.append(sha256_indices(train_indices))
        validation_hashes.append(sha256_indices(validation_indices))
        client_examples.append(ROWS_PER_CLIENT)
        client_training_labels.append(
            labels[np.asarray(train_indices, dtype=np.int64)]
        )
    test_rows = 4 * NUM_CLASSES
    x_test = rng.normal(size=(test_rows, INPUT_DIM)).astype(np.float32)
    y_test = np.tile(np.arange(NUM_CLASSES, dtype=np.int64), 4)
    global_test_path = base_dir / "global_test.npz"
    np.savez_compressed(global_test_path, x=x_test, y=y_test)
    statistics_artifacts = write_client_distribution_statistics(
        base_dir,
        client_training_labels,
        list(CICIOT2023_LABELS),
    )
    metadata = {
        "input_dim": INPUT_DIM,
        "num_classes": NUM_CLASSES,
        "num_clients": num_clients,
        "labels": list(CICIOT2023_LABELS),
        "label_to_id": {
            label: index for index, label in enumerate(CICIOT2023_LABELS)
        },
        "feature_columns": [f"feature_{index:02d}" for index in range(INPUT_DIM)],
        "client_examples": client_examples,
        "preprocessing": {
            "fit_scope": "deterministic_local_training_subsets_only",
            "excluded_from_fit": ["client_validation", "global_test"],
            "client_val_ratio": 0.2,
            "seed": 42,
            "client_train_indices_sha256": train_hashes,
            "client_validation_indices_sha256": validation_hashes,
        },
        "global_test_provenance": {
            "status": "complete",
            "role": "evaluation_only",
            "immutable": True,
            "source_kind": "synthetic_smoke_fixture",
            "prepared_npz_sha256": sha256_file(global_test_path),
            "prepared_x_sha256": sha256_array(x_test),
            "prepared_y_sha256": sha256_array(y_test),
            "preprocessing_fit_includes_global_test": False,
        },
        "client_statistics": statistics_artifacts,
    }
    (base_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _tiny_model_overrides(payload: dict, *, key: str) -> None:
    # The LSTM width is what makes the buffer share representative, not the
    # parameter count. BatchNorm carries two buffer values per conv channel
    # while the recurrent weights carry none, so at width 4 the buffers are
    # 1.2 percent of the model against 0.18 percent at the campaign's width --
    # enough that the sparse compressor's budget check fires here on a
    # configuration that is comfortable on the real model. Width 16 puts the
    # ratio at 0.20 percent for 4,034 parameters, so the smoke test still
    # finishes in seconds but exercises the arithmetic the campaign will meet.
    payload[key].update(
        {
            "input_dim": INPUT_DIM,
            "num_classes": NUM_CLASSES,
            "conv_channels": "4",
            "lstm_hidden_size": 16,
            "dropout": 0.0,
        }
    )


def _write_smoke_config(
    scenario: str,
    *,
    num_clients: int,
    partitions_dir: Path,
    manifest_path: Path,
    results_dir: Path,
    output_path: Path,
) -> None:
    payload = yaml.safe_load(config_path(scenario).read_text(encoding="utf-8"))
    run_name = canonical_run_name(scenario, num_clients, 42)
    if trainer_script(scenario) == "train_local_federated.py":
        payload["algorithm"].update(
            {
                "num_server_rounds": 1,
                "local_epochs": 1,
                "batch_size": 8,
                "min_train_nodes": num_clients,
                "min_evaluate_nodes": num_clients,
                "min_available_nodes": num_clients,
            }
        )
        payload["data"].update(
            {
                "partitions_dir": str(partitions_dir),
                "partition_file": str(manifest_path),
                "num_clients": num_clients,
                "num_classes": NUM_CLASSES,
            }
        )
        _tiny_model_overrides(payload, key="model")
        payload["results"].update(
            {
                "dir": str(results_dir),
                "run_name": run_name,
                "save_model": False,
                "save_round_checkpoints": False,
            }
        )
        payload["runtime"].update({"device": "cpu", "parallel_clients": 1})
    else:
        payload["algorithm"].update(
            {
                "num_server_rounds": 1,
                "local_epochs": 1,
                "batch_size": 8,
                "num_clients": num_clients,
            }
        )
        payload["data"].update(
            {
                "partitions_dir": str(partitions_dir),
                "partition_file": str(manifest_path),
                "num_clients": num_clients,
            }
        )
        _tiny_model_overrides(payload, key="model")
        payload["results"].update(
            {
                "dir": str(results_dir),
                "run_name": run_name,
                "save_model": True,
                "save_round_checkpoints": False,
            }
        )
        payload["runtime"].update({"device": "cpu", "parallel_clients": 1})
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _baseline_metric_keys(results_dir: Path, run_name: str) -> set[str]:
    evaluations = json.loads(
        (results_dir / f"{run_name}_test_evaluations.json").read_text(encoding="utf-8")
    )
    return set(evaluations["final"]["test_metrics"])


def _proposed_metric_keys(results_dir: Path) -> set[str]:
    return set(json.loads((results_dir / "test" / "final.json").read_text(encoding="utf-8")))


def run_scenario(
    scenario: str,
    *,
    num_clients: int,
    temp_root: Path,
    partitions_dir: Path,
    manifest_path: Path,
) -> dict[str, object]:
    run_name = canonical_run_name(scenario, num_clients, 42)
    results_dir = temp_root / "results" / f"clients_{num_clients:03d}" / scenario
    smoke_config = temp_root / f"{scenario}_{num_clients}.yaml"
    _write_smoke_config(
        scenario,
        num_clients=num_clients,
        partitions_dir=partitions_dir,
        manifest_path=manifest_path,
        results_dir=results_dir,
        output_path=smoke_config,
    )
    if trainer_script(scenario) == "train_local_federated.py":
        command = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "scripts" / trainer_script(scenario)),
            "--config-path", str(smoke_config),
        ]
    else:
        command = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "scripts" / trainer_script(scenario)),
            "--config-path", str(smoke_config),
        ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Smoke run failed for {scenario} at {num_clients} clients:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    keys = (
        _baseline_metric_keys(results_dir, run_name)
        if trainer_script(scenario) == "train_local_federated.py"
        else _proposed_metric_keys(results_dir)
    )
    contract_path = results_dir.parent / "_protocol" / "classification_metric_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected = contract["classification_metric_keys"]
    missing = sorted(set(expected) - keys)
    if missing:
        raise AssertionError(
            f"{scenario} at {num_clients} clients did not report the contracted "
            f"metrics: missing {missing}"
        )
    if not set(expected).issubset(CLASSIFICATION_METRIC_KEYS):
        raise AssertionError("Contract declares metrics this build does not know")
    receipt = audit_run(
        results_dir,
        scenario=scenario,
        num_clients=num_clients,
        expected_rounds=1,
        expected_partition_hash=None,
        expected_global_test_sha256=None,
        seed=42,
    )
    return {
        "scenario": scenario,
        "num_clients": num_clients,
        "status": "passed",
        "audit_status": receipt["status"],
        "num_metrics": len(expected),
        "benign_class_name": contract["benign_class_name"],
        "metric_contract_sha256": contract["sha256"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", action="append", choices=PAPER_SCENARIOS)
    parser.add_argument(
        "--num-clients",
        action="append",
        type=int,
        choices=SUPPORTED_CLIENT_COUNTS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = tuple(args.scenario or PAPER_SCENARIOS)
    client_counts = tuple(args.num_clients or SUPPORTED_CLIENT_COUNTS)
    records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="paper34_smoke_") as tmp:
        temp_root = Path(tmp)
        for num_clients in client_counts:
            partitions_dir = temp_root / f"partitions_{num_clients:03d}"
            _write_synthetic_partition(partitions_dir, num_clients)
            manifest_path = temp_root / f"client_partition_{num_clients:03d}.json"
            create_partition_manifest(
                partitions_dir,
                manifest_path,
                dataset="existing_dataset",
                num_clients=num_clients,
                client_val_ratio=0.2,
                seed=42,
            )
            for scenario in scenarios:
                record = run_scenario(
                    scenario,
                    num_clients=num_clients,
                    temp_root=temp_root,
                    partitions_dir=partitions_dir,
                    manifest_path=manifest_path,
                )
                print(
                    f"SMOKE PASS {scenario} clients={num_clients}",
                    flush=True,
                )
                records.append(record)
    print(
        json.dumps(
            {
                "status": "passed",
                "num_runs": len(records),
                "classification_metric_keys": list(CLASSIFICATION_METRIC_KEYS),
                "proposed_scenarios": ["fedmpsq"],
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
