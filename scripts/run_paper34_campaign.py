#!/usr/bin/env python
"""Plan or execute one of the eight reported baseline runs.

The same eight configurations are used at 10 and 100 clients. The runner
overrides only the client count, input partition, output directory, device,
parallelism, and training seed. One invocation launches exactly one run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.partition_manifest import SUPPORTED_CLIENT_COUNTS
from scripts.paper_runs import BASELINE_RUNS, canonical_run_name
from training.metrics import (
    CLASSIFICATION_METRIC_KEYS,
    CLASSIFICATION_METRIC_SCHEMA_VERSION,
)

CONFIG_DIR = PROJECT_ROOT / "configs" / "paper34_v1"
CAMPAIGN_ID = "fedmpsq-paper-10-100-clients"
NUM_CLASSES = 34
DEFAULT_SEED = 42
SEED = DEFAULT_SEED
REPLICATION_SEEDS = (42, 43, 44)

BASELINE_SCENARIOS = {run.slug: run.config_filename for run in BASELINE_RUNS}

# Compatibility with the synthetic smoke runner. FedMPSQ is launched by the
# separate byte-budget-aware runner in run_uplink_v4.py.
PROPOSED_SCENARIOS: dict[str, str] = {}
SCENARIOS = tuple(BASELINE_SCENARIOS)


def trainer_script(scenario: str) -> str:
    if scenario not in BASELINE_SCENARIOS:
        raise ValueError(f"Unsupported baseline scenario: {scenario}")
    return "train_local_federated.py"


def run_directory_name(scenario: str, seed: int) -> str:
    if scenario not in BASELINE_SCENARIOS:
        raise ValueError(f"Unsupported baseline scenario: {scenario}")
    return scenario if seed == DEFAULT_SEED else f"{scenario}_seed{seed}"


def config_path(scenario: str) -> Path:
    try:
        path = CONFIG_DIR / BASELINE_SCENARIOS[scenario]
    except KeyError as exc:
        raise ValueError(f"Unsupported baseline scenario: {scenario}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def run_name(scenario: str, num_clients: int, seed: int = DEFAULT_SEED) -> str:
    if scenario not in BASELINE_SCENARIOS:
        raise ValueError(f"Unsupported baseline scenario: {scenario}")
    return canonical_run_name(scenario, num_clients, seed)


def build_job(
    output_root: str | Path,
    *,
    scenario: str,
    num_clients: int,
    seed: int = DEFAULT_SEED,
    partitions_dir: str | None = None,
    partition_file: str | None = None,
    device: str = "auto",
    parallel_clients: int = 2,
    resume: str | None = None,
) -> dict:
    if scenario not in BASELINE_SCENARIOS:
        raise ValueError(f"Unsupported baseline scenario: {scenario}")
    if num_clients not in SUPPORTED_CLIENT_COUNTS:
        raise ValueError(
            f"num_clients must be one of {SUPPORTED_CLIENT_COUNTS}; got {num_clients}"
        )
    if seed not in REPLICATION_SEEDS:
        raise ValueError(f"seed must be one of {REPLICATION_SEEDS}; got {seed}")
    if parallel_clients <= 0:
        raise ValueError("parallel_clients must be positive")

    root = Path(output_root).resolve()
    results_dir = root / f"clients_{num_clients:03d}" / run_directory_name(
        scenario, seed
    )
    name = run_name(scenario, num_clients, seed)
    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts" / trainer_script(scenario)),
        "--config-path",
        str(config_path(scenario)),
        "--seed",
        str(seed),
        "--num-clients",
        str(num_clients),
        "--num-classes",
        str(NUM_CLASSES),
        "--results-dir",
        str(results_dir),
        "--run-name",
        name,
        "--device",
        device,
        "--parallel-clients",
        str(parallel_clients),
    ]
    if partitions_dir is not None:
        command.extend(["--partitions-dir", partitions_dir])
    if partition_file is not None:
        command.extend(["--partition-file", partition_file])
    if resume is not None:
        command.extend(["--resume", resume])

    return {
        "campaign_id": CAMPAIGN_ID,
        "scenario": scenario,
        "family": "baseline",
        "trainer": trainer_script(scenario),
        "num_clients": num_clients,
        "num_classes": NUM_CLASSES,
        "seed": seed,
        "classification_metric_schema_version": CLASSIFICATION_METRIC_SCHEMA_VERSION,
        "classification_metric_keys": list(CLASSIFICATION_METRIC_KEYS),
        "config": str(config_path(scenario)),
        "results_dir": str(results_dir),
        "run_name": name,
        "resume": resume,
        "command": command,
    }


def build_matrix(output_root: str | Path, seed: int = DEFAULT_SEED) -> list[dict]:
    return [
        build_job(
            output_root,
            scenario=scenario,
            num_clients=num_clients,
            seed=seed,
        )
        for num_clients in SUPPORTED_CLIENT_COUNTS
        for scenario in BASELINE_SCENARIOS
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="results/local")
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--num-clients", type=int, choices=SUPPORTED_CLIENT_COUNTS)
    parser.add_argument("--seed", type=int, choices=REPLICATION_SEEDS, default=DEFAULT_SEED)
    parser.add_argument("--partitions-dir")
    parser.add_argument("--partition-file")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--parallel-clients", type=int, default=2)
    parser.add_argument("--resume")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the selected job; without this flag only the plan is printed.",
    )
    args = parser.parse_args()
    if args.execute and (args.scenario is None or args.num_clients is None):
        parser.error("--execute requires --scenario and --num-clients")
    return args


def main() -> None:
    args = parse_args()
    if args.scenario is None or args.num_clients is None:
        print(
            json.dumps(
                {
                    "mode": "matrix",
                    "campaign_id": CAMPAIGN_ID,
                    "seed": args.seed,
                    "num_runs": len(BASELINE_SCENARIOS) * len(SUPPORTED_CLIENT_COUNTS),
                    "jobs": build_matrix(args.output_root, seed=args.seed),
                },
                indent=2,
            )
        )
        return

    job = build_job(
        args.output_root,
        scenario=args.scenario,
        num_clients=args.num_clients,
        seed=args.seed,
        partitions_dir=args.partitions_dir,
        partition_file=args.partition_file,
        device=args.device,
        parallel_clients=args.parallel_clients,
        resume=args.resume,
    )
    print(
        json.dumps(
            {"mode": "execute" if args.execute else "plan", "job": job}, indent=2
        )
    )
    if not args.execute:
        return
    if args.partitions_dir is None or args.partition_file is None:
        raise ValueError("--execute requires --partitions-dir and --partition-file")
    results_dir = Path(job["results_dir"])
    if args.resume is None and results_dir.exists() and any(results_dir.iterdir()):
        raise FileExistsError(f"Results already exist: {results_dir}")
    command = list(job["command"])
    command.extend(["--launcher-command", subprocess.list2cmdline(command)])
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
