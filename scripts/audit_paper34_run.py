#!/usr/bin/env python
"""Acceptance audit for one completed run of the paper34 campaign.

Fail-closed checks over the artifacts a finished run leaves behind: the frozen
partition, the shared evaluation contract, a complete round history, and all
finite metrics from the current classification schema on the frozen global
test. A run that does not pass this audit must not enter a table.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.partition_manifest import SUPPORTED_CLIENT_COUNTS
from scripts.paper_runs import (  # noqa: E402
    PAPER_SCENARIOS,
    RUN_BY_SLUG,
    canonical_run_name,
    scenario_family,
)
from scripts.run_paper34_campaign import (  # noqa: E402
    CAMPAIGN_ID,
    NUM_CLASSES,
    SEED,
)
from training.metrics import (
    CLASSIFICATION_METRIC_KEYS,
    CLASSIFICATION_METRIC_SCHEMA_VERSION,
    SIGNED_METRIC_KEYS,
    benign_class_id,
    metric_keys_for,
)

SIGNED_METRICS = frozenset(SIGNED_METRIC_KEYS)


class AuditFailure(AssertionError):
    """Raised when one acceptance check fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def read_json(path: Path) -> Any:
    require(path.is_file() and path.stat().st_size > 0, f"Missing or empty: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_metric_block(
    block: dict[str, Any],
    *,
    label: str,
    expected_keys: tuple[str, ...],
) -> dict[str, float]:
    """Require every contracted metric, finite and inside its theoretical range."""
    values: dict[str, float] = {}
    for key in expected_keys:
        require(key in block, f"{label} does not report '{key}'")
        value = float(block[key])
        require(math.isfinite(value), f"{label}.{key} is not finite: {value}")
        lower = -1.0 if key in SIGNED_METRICS else 0.0
        require(
            lower <= value <= 1.0,
            f"{label}.{key} outside [{lower}, 1]: {value}",
        )
        values[key] = value
    return values


def check_round_history(
    records: list[dict[str, Any]],
    *,
    expected_rounds: int,
    label: str,
) -> None:
    require(bool(records), f"{label} has no round records")
    rounds = sorted({int(record["round"]) for record in records})
    require(
        rounds == list(range(expected_rounds + 1)),
        f"{label} does not cover rounds 0..{expected_rounds}; got {rounds}",
    )


def audit_baseline(
    run_dir: Path,
    run_name: str,
    *,
    expected_rounds: int,
    expected_keys: tuple[str, ...],
) -> dict[str, Any]:
    summary = read_json(run_dir / f"{run_name}_summary.json")
    evaluations = read_json(run_dir / f"{run_name}_test_evaluations.json")
    rounds = read_json(run_dir / f"{run_name}_round_metrics.json")
    read_json(run_dir / f"{run_name}_provenance.json")
    read_json(run_dir / f"{run_name}_config.json")
    read_json(run_dir / f"{run_name}_config.json")

    check_round_history(rounds, expected_rounds=expected_rounds, label="round metrics")
    best = check_metric_block(
        evaluations["best"]["test_metrics"],
        label="best global test",
        expected_keys=expected_keys,
    )
    final = check_metric_block(
        evaluations["final"]["test_metrics"],
        label="final global test",
        expected_keys=expected_keys,
    )
    for checkpoint in ("best", "final"):
        confusion = evaluations[checkpoint]["details"]["confusion_matrix"]
        require(
            len(confusion) == NUM_CLASSES
            and all(len(row) == NUM_CLASSES for row in confusion),
            f"{checkpoint} confusion matrix is not {NUM_CLASSES}x{NUM_CLASSES}",
        )
        require(
            sum(sum(row) for row in confusion)
            == int(evaluations[checkpoint]["details"]["num_examples"]),
            f"{checkpoint} confusion matrix does not sum to the evaluated rows",
        )
    return {
        "summary": summary,
        "best_test_metrics": best,
        "final_test_metrics": final,
        "best_round": int(summary["best_round"]),
        "global_test_examples": int(evaluations["final"]["details"]["num_examples"]),
        "global_test_sha256": evaluations["global_test_sha256"],
    }


def audit_proposed(
    run_dir: Path,
    run_name: str,
    *,
    expected_rounds: int,
    expected_keys: tuple[str, ...],
) -> dict[str, Any]:
    summary = read_json(run_dir / f"{run_name}_summary.json")
    evaluations = read_json(run_dir / f"{run_name}_test_evaluations.json")
    rounds = read_json(run_dir / f"{run_name}_round_metrics.json")
    read_json(run_dir / f"{run_name}_provenance.json")
    best_test = read_json(run_dir / "test" / "best.json")
    final_test = read_json(run_dir / "test" / "final.json")

    check_round_history(rounds, expected_rounds=expected_rounds, label="round metrics")
    best = check_metric_block(
        best_test, label="best global test", expected_keys=expected_keys
    )
    final = check_metric_block(
        final_test, label="final global test", expected_keys=expected_keys
    )
    for checkpoint, block in (("best", best_test), ("final", final_test)):
        confusion = block["confusion_matrix"]
        require(
            len(confusion) == NUM_CLASSES
            and all(len(row) == NUM_CLASSES for row in confusion),
            f"{checkpoint} confusion matrix is not {NUM_CLASSES}x{NUM_CLASSES}",
        )
        require(
            sum(sum(row) for row in confusion) == int(block["num_examples"]),
            f"{checkpoint} confusion matrix does not sum to the evaluated rows",
        )
    require(
        summary.get("status") == "completed",
        f"FedMPSQ run status is {summary.get('status')!r}, expected 'completed'",
    )
    validation_dir = run_dir / "validation"
    require(validation_dir.is_dir(), "FedMPSQ run has no validation directory")
    validation_rounds = sorted(
        int(path.stem.split("_")[-1]) for path in validation_dir.glob("round_*.json")
    )
    require(
        validation_rounds == list(range(expected_rounds + 1)),
        f"validation history does not cover rounds 0..{expected_rounds}",
    )
    return {
        "summary": summary,
        "best_test_metrics": best,
        "final_test_metrics": final,
        "best_round": int(summary["best_round"]),
        "global_test_examples": int(final_test["num_examples"]),
        "global_test_sha256": evaluations["global_test_sha256"],
    }


def audit_run(
    run_dir: Path,
    *,
    scenario: str,
    num_clients: int,
    expected_rounds: int,
    expected_partition_hash: str | None,
    expected_global_test_sha256: str | None,
    seed: int = SEED,
) -> dict[str, Any]:
    require(run_dir.is_dir(), f"Run directory not found: {run_dir}")
    family = scenario_family(scenario)
    name = canonical_run_name(scenario, num_clients, seed)

    contract_path = run_dir.parent / "_protocol" / "classification_metric_contract.json"
    contract = read_json(contract_path)
    require(
        contract["classification_metric_schema_version"]
        == CLASSIFICATION_METRIC_SCHEMA_VERSION,
        "Frozen contract does not declare the current classification metric schema",
    )
    expected_keys = tuple(contract["classification_metric_keys"])
    require(
        set(expected_keys).issubset(CLASSIFICATION_METRIC_KEYS),
        "Frozen contract declares metrics this build does not know",
    )
    require(
        expected_keys
        == metric_keys_for(
            has_benign_class=contract.get("benign_class_id") is not None,
            scored=True,
        ),
        "Frozen contract metric list does not match its own label space",
    )
    require(
        contract.get("benign_class_id")
        == benign_class_id(contract["target_class_names"]),
        "Frozen contract names a benign class the label space does not have",
    )
    require(
        int(contract["num_classes"]) == NUM_CLASSES,
        f"Frozen contract is not the {NUM_CLASSES}-class task",
    )
    require(
        int(contract["num_clients"]) == num_clients,
        "Frozen contract client count differs from the audited run",
    )
    require(
        len(contract["minority_class_ids"]) == math.ceil(NUM_CLASSES * 0.25),
        "Frozen contract does not hold the rarest quarter of the classes",
    )

    audited = (
        audit_proposed(
            run_dir, name, expected_rounds=expected_rounds, expected_keys=expected_keys
        )
        if family == "proposed"
        else audit_baseline(
            run_dir, name, expected_rounds=expected_rounds, expected_keys=expected_keys
        )
    )
    summary = audited["summary"]

    require(
        summary.get("run_name") == name,
        "Run summary name differs from the canonical artifact name",
    )
    require(
        summary.get("experiment_id") == RUN_BY_SLUG[scenario].display_name,
        "Run summary experiment identifier differs from the canonical paper run",
    )
    require(
        summary.get("metric_contract_sha256") == contract["sha256"],
        "Run summary references a different evaluation contract",
    )
    require(
        int(summary.get("num_clients", -1)) == num_clients,
        "Run summary client count differs from the audited client count",
    )
    require(
        int(summary.get("num_classes", -1)) == NUM_CLASSES,
        f"Run summary is not the {NUM_CLASSES}-class task",
    )
    require(
        int(summary.get("training_seed", -1)) == seed,
        f"Run summary training seed is not {seed}",
    )
    if expected_partition_hash is not None:
        require(
            summary["partition_hash"] == expected_partition_hash,
            "Run used a different frozen partition than expected",
        )
        require(
            contract["partition_hash"] == expected_partition_hash,
            "Evaluation contract references a different frozen partition",
        )
    if expected_global_test_sha256 is not None:
        require(
            audited["global_test_sha256"] == expected_global_test_sha256,
            "Run evaluated a different global test set than expected",
        )

    return {
        "campaign_id": CAMPAIGN_ID,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "scenario": scenario,
        "family": family,
        "num_clients": num_clients,
        "num_classes": NUM_CLASSES,
        "seed": seed,
        "run_name": name,
        "run_dir": str(run_dir),
        "rounds_completed": expected_rounds,
        "best_round": audited["best_round"],
        "partition_hash": summary["partition_hash"],
        "global_test_sha256": audited["global_test_sha256"],
        "global_test_examples": audited["global_test_examples"],
        "classification_metric_schema_version": CLASSIFICATION_METRIC_SCHEMA_VERSION,
        "classification_metric_keys": list(expected_keys),
        "benign_class_id": contract["benign_class_id"],
        "benign_class_name": contract["benign_class_name"],
        "metric_contract_sha256": contract["sha256"],
        "minority_class_ids": contract["minority_class_ids"],
        "minority_class_names": contract["minority_class_names"],
        "best_test_metrics": audited["best_test_metrics"],
        "final_test_metrics": audited["final_test_metrics"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--scenario", required=True, choices=PAPER_SCENARIOS)
    parser.add_argument(
        "--num-clients",
        required=True,
        type=int,
        choices=SUPPORTED_CLIENT_COUNTS,
    )
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Training seed this run was produced under; the audit fails closed "
             "if the run summary disagrees",
    )
    parser.add_argument("--expected-partition-hash", default=None)
    parser.add_argument("--expected-global-test-sha256", default=None)
    parser.add_argument(
        "--write-acceptance",
        action="store_true",
        help="Write the receipt under <run-dir>/../_acceptance/",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        receipt = audit_run(
            args.run_dir.expanduser().resolve(),
            scenario=args.scenario,
            num_clients=args.num_clients,
            expected_rounds=args.rounds,
            expected_partition_hash=args.expected_partition_hash,
            expected_global_test_sha256=args.expected_global_test_sha256,
            seed=args.seed,
        )
    except AuditFailure as failure:
        print(json.dumps({"status": "failed", "reason": str(failure)}, indent=2))
        return 1
    if args.write_acceptance:
        acceptance_dir = args.run_dir.expanduser().resolve().parent / "_acceptance"
        acceptance_dir.mkdir(parents=True, exist_ok=True)
        path = acceptance_dir / f"{receipt['run_name']}_acceptance.json"
        path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        receipt["acceptance_file"] = str(path)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    print("PASS")
    return 0


PROPOSED_SCENARIO_NAMES = ("fedmpsq",)


if __name__ == "__main__":
    raise SystemExit(main())
