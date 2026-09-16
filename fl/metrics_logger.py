"""Round-level metrics logging."""

from __future__ import annotations

import copy
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


BEST_CHECKPOINT_POLICY = {
    "eligible_rounds": "completed training rounds 1..R (round 0 is excluded)",
    "metric": "validation_macro_f1",
    "metric_scope": "global union of all frozen client validation splits",
    "mode": "max",
    "tie_break": "earliest round",
    "test_usage": "global test is evaluated only after selection, for best and final",
}


def fixed_test_contract(global_test_record: dict[str, Any]) -> dict[str, Any]:
    """Select the immutable fields that identify the exact global test set."""
    keys = (
        "file",
        "sha256",
        "x_sha256",
        "y_sha256",
        "num_examples",
        "x_shape",
        "y_shape",
        "x_dtype",
        "y_dtype",
    )
    missing = [key for key in keys if key not in global_test_record]
    if missing:
        raise ValueError(f"Global-test manifest record misses fields: {missing}")
    return {
        "schema_version": 1,
        "role": "fixed_global_test_not_used_for_checkpoint_selection",
        "global_test": {key: global_test_record[key] for key in keys},
    }


def _run_git(project_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _installed_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        try:
            name = distribution.metadata.get("Name")
            version = distribution.version
        except (KeyError, OSError, UnicodeError):
            continue
        if name:
            packages[str(name)] = str(version)
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))


def _gpu_inventory() -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for device_id in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(device_id)
        inventory.append(
            {
                "device_id": device_id,
                "name": properties.name,
                "compute_capability": list(properties.major_minor)
                if hasattr(properties, "major_minor")
                else [properties.major, properties.minor],
                "total_memory_bytes": int(properties.total_memory),
            }
        )
    return inventory


class RoundMetricsLogger:
    """Write round metrics to CSV and JSON files."""

    def __init__(self, results_dir: str | Path, run_name: str) -> None:
        self.results_dir = Path(results_dir)
        self.run_name = run_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.results_dir / f"{run_name}_round_metrics.csv"
        self.json_path = self.results_dir / f"{run_name}_round_metrics.json"
        self._records: list[dict[str, Any]] = []

    def save_config(self, settings: Any) -> None:
        """Persist experiment settings."""
        config_path = self.results_dir / f"{self.run_name}_config.json"
        config_path.write_text(
            json.dumps(asdict(settings), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def enforce_fixed_test_contract(
        self,
        global_test_record: dict[str, Any],
    ) -> Path:
        """Create or verify the shared test-set contract across all run folders."""
        contract = fixed_test_contract(global_test_record)
        protocol_dir = self.results_dir.parent / "_protocol"
        protocol_dir.mkdir(parents=True, exist_ok=True)
        contract_path = protocol_dir / "fixed_global_test.json"
        serialized = json.dumps(contract, indent=2, sort_keys=True)
        try:
            with contract_path.open("x", encoding="utf-8") as handle:
                handle.write(serialized)
        except FileExistsError:
            existing = json.loads(contract_path.read_text(encoding="utf-8"))
            if existing != contract:
                raise ValueError(
                    "Global test set differs from the frozen contract at "
                    f"{contract_path}. Use the original test set; do not mix these runs."
                )
        return contract_path

    def save_provenance(
        self,
        settings: Any,
        *,
        project_root: str | Path,
        partition_record: dict[str, Any],
        launcher_command: str | None,
    ) -> None:
        """Persist code, command, environment, hardware, and data identities."""
        root = Path(project_root).resolve()
        git_sha = _run_git(root, "rev-parse", "HEAD")
        git_status = _run_git(root, "status", "--short")
        cudnn_version = torch.backends.cudnn.version()
        provenance = {
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_id": settings.experiment_id,
            "run_name": settings.results.run_name,
            "seed": settings.runtime.seed,
            "training_seed": settings.runtime.seed,
            "data_split_seed": settings.data.seed,
            "commands": {
                "launcher": launcher_command,
                "training_argv": list(sys.argv),
            },
            "working_directory": os.getcwd(),
            "git": {
                "sha": git_sha,
                "dirty": bool(git_status),
                "status_short": git_status or "",
            },
            "python": {
                "executable": sys.executable,
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
            },
            "platform": platform.platform(),
            "torch": {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "cudnn_version": cudnn_version,
                "gpu_count": torch.cuda.device_count(),
                "gpus": _gpu_inventory(),
            },
            "environment": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "installed_packages": _installed_packages(),
            "data": partition_record,
            "checkpoint_selection": BEST_CHECKPOINT_POLICY,
        }
        path = self.results_dir / f"{self.run_name}_provenance.json"
        path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def save_test_evaluations(self, evaluations: dict[str, Any]) -> None:
        """Persist the post-selection global-test evaluations explicitly."""
        path = self.results_dir / f"{self.run_name}_test_evaluations.json"
        path.write_text(
            json.dumps(evaluations, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def log_round(self, server_round: int, metrics: dict[str, float]) -> None:
        """Append one round of metrics."""
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "round": int(server_round),
        }
        record.update({key: float(value) for key, value in metrics.items()})
        self._records.append(record)
        self._write_csv()
        self.json_path.write_text(
            json.dumps(self._records, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def snapshot_records(self) -> list[dict[str, Any]]:
        """Return an independent copy suitable for a resume checkpoint."""
        return copy.deepcopy(self._records)

    def restore_records(self, records: list[dict[str, Any]]) -> None:
        """Restore an audited metric prefix before appending resumed rounds."""
        if not isinstance(records, list) or any(
            not isinstance(record, dict) for record in records
        ):
            raise TypeError("Round metric records must be a list of mappings")
        self._records = copy.deepcopy(records)
        self._write_csv()
        self.json_path.write_text(
            json.dumps(self._records, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def save_flower_history(self, result: Any) -> None:
        """Persist all Flower result metric histories in CSV and JSON."""
        records: list[dict[str, Any]] = []
        records.extend(
            self._metric_history_records(
                result.train_metrics_clientapp,
                source="client_train",
            )
        )
        records.extend(
            self._metric_history_records(
                result.evaluate_metrics_clientapp,
                source="client_evaluate",
            )
        )
        records.extend(
            self._metric_history_records(
                result.evaluate_metrics_serverapp,
                source="server_evaluation",
            )
        )

        json_path = self.results_dir / f"{self.run_name}_flower_history.json"
        csv_path = self.results_dir / f"{self.run_name}_flower_history.csv"
        json_path.write_text(
            json.dumps(records, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if not records:
            return
        keys: list[str] = []
        for record in records:
            for key in record:
                if key not in keys:
                    keys.append(key)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)

    def _write_csv(self) -> None:
        keys: list[str] = []
        for record in self._records:
            for key in record:
                if key not in keys:
                    keys.append(key)
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self._records)

    @staticmethod
    def _metric_history_records(
        history: dict[int, Any],
        *,
        source: str,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for server_round, metrics in sorted(history.items()):
            record: dict[str, Any] = {
                "round": int(server_round),
                "source": source,
            }
            record.update(dict(metrics))
            records.append(record)
        return records
