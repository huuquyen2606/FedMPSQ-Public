"""Prepare or run the FedMPSQ configuration reported in the paper.

The 10-client run uses a 3,334-byte uplink cap and the 100-client run uses a
3,044-byte cap. Planning is the default; ``--run`` requires CUDA. Checkpoint
selection remains validation-only, and ``--evaluate-test`` enables the final
test evaluation after the configuration has been fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.integrity import sha256_file
from fl.fedmpsq_config import load_fedmpsq_config
from scripts.paper_runs import canonical_run_name
from scripts.train_fedmpsq import scientific_config_sha256

CLIENTS = 10
DEFAULT_ROUNDS = 20
CLIENTS_100 = 100
RUN_SLUG = "fedmpsq"

# Four levels on the Lloyd-Max codebook for a standard normal, which is what a
# rotated group looks like, with a one-byte logarithmic group scale.
CODEBOOK = dict(block_size=32, quant_bits=2, quant_group_size=256,
                scale_codec="log8", incoherent_rotation=True, quantizer="gaussian4")

ARMS = {
    "rht_g4_b3334": dict(**CODEBOOK, uplink_budget_bytes=3334),
}

ARMS_100 = {
    "rht_g4_b3044_c100": dict(**CODEBOOK, uplink_budget_bytes=3044),
}

SCALES = {
    CLIENTS: ARMS,
    CLIENTS_100: ARMS_100,
}
ALL_ARMS = {**ARMS, **ARMS_100}


def run_name(clients: int, seed: int) -> str:
    if clients not in SCALES:
        raise ValueError(f"Unsupported client count: {clients}")
    return canonical_run_name(RUN_SLUG, clients, seed)


def verify_exact_npz(
    root: Path, manifest_path: Path, expected_clients: int
) -> dict:
    """Fail if a prepared partition differs from its frozen manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest["partition_hash"]
    payload = {key: value for key, value in manifest.items() if key != "partition_hash"}
    actual_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if actual_hash != expected_hash or int(manifest["num_clients"]) != expected_clients:
        raise ValueError("Invalid frozen NPZ manifest")
    records = [
        manifest["metadata"],
        *manifest["clients"],
        manifest["global_test"],
        *manifest["derived_artifacts"],
    ]
    for record in records:
        filename = record["file"]
        if not filename or Path(filename).name != filename:
            raise ValueError("Manifest file names must be basenames")
        path = root / filename
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Prepared partition differs from manifest: {path}")
    return manifest


def source_hash() -> str:
    """Hash the Python source that determines the run."""
    digest = hashlib.sha256()
    for folder in ("data", "extensions", "fl", "models", "scripts", "training"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare(data_dir, manifest_path, output, seeds, clients=CLIENTS):
    arms = SCALES[clients]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest["num_clients"]) != clients:
        raise ValueError(
            f"Manifest describes {manifest['num_clients']} clients, not {clients}")
    base = ROOT / "configs/paper34_v1/fedmpsq.yaml"
    rows, contents = [], {}
    for name, arm in arms.items():
        method = {key: value for key, value in arm.items() if key != "rounds"}
        rounds = int(arm.get("rounds", DEFAULT_ROUNDS))
        for seed in seeds:
            artifact_name = run_name(clients, seed)
            settings = load_fedmpsq_config(base, {
                **{f"method.{key}": value for key, value in method.items()},
                "algorithm.num_server_rounds": rounds,
                "data.partitions_dir": str(data_dir.resolve()),
                "data.partition_file": str(manifest_path.resolve()),
                "runtime.seed": seed,
                "data.split_seed": seed,
                "algorithm.num_clients": clients,
                "data.num_clients": clients,
                "results.dir": str(output / "runs" / artifact_name),
                "results.run_name": artifact_name,
            })
            config = output / "configs" / f"{artifact_name}.yaml"
            contents[config] = yaml.safe_dump(settings.config_dict, sort_keys=False)
            planned = method["uplink_budget_bytes"] * clients * rounds
            rows.append(dict(
                arm=name, seed=seed, config=str(config),
                payload_cap_per_client_round=method["uplink_budget_bytes"],
                rounds=rounds,
                planned_total_uplink_upper_bound=planned,
                scientific_config_sha256=scientific_config_sha256(settings)))
    plan = dict(
        schema=1, source_code_sha256=source_hash(),
        partition_hash=manifest["partition_hash"], selection="validation_only",
        rounds=DEFAULT_ROUNDS, arms=rows, num_clients=clients)
    contents[output / "plan.json"] = json.dumps(plan, indent=2)
    # Refuse to rewrite a plan/config used by an earlier experiment.
    for path, content in contents.items():
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Existing plan/config differs: {path}; use a new output directory")
    for path, content in contents.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ALL_ARMS, required=True)
    parser.add_argument("--clients", type=int, choices=sorted(SCALES), default=CLIENTS,
                        help="Client count of the attached partition; picks the arm table")
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Evaluate the global test set after validation-only selection")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.seed not in args.seeds:
        parser.error("--seed must be included in --seeds")
    if args.arm not in SCALES[args.clients]:
        parser.error(f"--arm {args.arm} is not part of the {args.clients}-client table")
    if args.run and not torch.cuda.is_available():
        parser.error("Full-data run requires CUDA-enabled PyTorch; this interpreter has CPU-only torch")
    plan = prepare(args.data_dir, args.manifest, args.output.resolve(), args.seeds,
                   clients=args.clients)
    print(f"Prepared {len(plan['arms'])} configs: {args.output.resolve() / 'plan.json'}", flush=True)
    if not args.run:
        return
    verify_exact_npz(args.data_dir, args.manifest, expected_clients=args.clients)
    entry = next(r for r in plan["arms"] if r["arm"] == args.arm and r["seed"] == args.seed)
    settings = load_fedmpsq_config(entry["config"])
    run = Path(settings.results.dir)
    summary = run / f"{settings.results.run_name}_summary.json"
    if summary.exists():
        record = json.loads(summary.read_text(encoding="utf-8"))
        if (record.get("status") != "completed"
                or record.get("scientific_config_sha256") != entry["scientific_config_sha256"]):
            raise ValueError("Existing summary is incomplete or mismatched")
        print("Run already completed")
        return
    command = [sys.executable, str(ROOT / "scripts/train_fedmpsq.py"),
               "--config-path", entry["config"], "--torch-threads", "4"]
    if not args.evaluate_test:
        command.append("--skip-test-evaluation")
    checkpoints = sorted((run / "checkpoints").glob(f"{settings.results.run_name}_round_*.pt"))
    if checkpoints:
        command += ["--resume", str(checkpoints[-1])]
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
