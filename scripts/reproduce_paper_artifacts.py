#!/usr/bin/env python
"""Regenerate the paper's compact result tables and figures from snapshots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_runs import PAPER_RUNS

DEFAULT_RESULTS = ROOT / "results" / "paper"
DEFAULT_OUTPUT = ROOT / "results"
ROUNDS = 20
DENSE_MESSAGE_BYTES = 1_397_020

ARM_STYLES = {
    "fedavg": ("#1b6ca8", "-"),
    "fedprox": ("#ce6b20", "--"),
    "bdd_hfl": ("#8b75af", ":"),
    "bdd_hfl_mu": ("#6c777e", "-."),
    "fap": ("#8c6d3f", (0, (3, 1, 1, 1))),
    "fap_mu": ("#5ba3c9", (0, (5, 2))),
    "fedpaq": ("#319660", "-"),
    "dadaquant": ("#c73e70", "--"),
    "fedmpsq": ("#111111", "-"),
}
ARMS = [
    (run.slug, run.display_name, *ARM_STYLES[run.slug]) for run in PAPER_RUNS
]
MINORITY_ARMS = [
    arm
    for arm in ARMS
    if arm[0] not in {"bdd_hfl", "bdd_hfl_mu"}
]


def validate_manifest(results_root: Path) -> None:
    manifest_path = results_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = manifest.get("runs", [])
    if manifest.get("schema") != 2 or len(runs) != 18:
        raise ValueError(f"Unexpected snapshot manifest: {manifest_path}")

    expected = {
        (clients, scenario)
        for clients in (10, 100)
        for scenario, _, _, _ in ARMS
    }
    observed = set()
    for run in runs:
        key = (int(run["clients"]), str(run["scenario"]))
        observed.add(key)
        for path_key, hash_key in (
            ("round_metrics_file", "round_metrics_sha256"),
            ("config_file", "config_sha256"),
        ):
            candidate = (results_root / run[path_key]).resolve()
            try:
                candidate.relative_to(results_root)
            except ValueError as exc:
                raise ValueError(f"Snapshot path escapes results root: {candidate}") from exc
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if digest != run[hash_key]:
                raise ValueError(f"Snapshot digest mismatch: {candidate}")
    if observed != expected:
        raise ValueError("Snapshot manifest does not cover the 18 reported runs")


def read_rounds(results_root: Path, clients: int, scenario: str) -> list[dict]:
    path = results_root / f"{clients}_clients" / scenario / "round_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(float(row["round"])))
    rounds = [int(float(row["round"])) for row in rows]
    if ROUNDS not in rounds:
        raise ValueError(f"{path} has no round {ROUNDS}")
    return rows


def uplink_column(row: dict) -> str:
    if "communication_cumulative_uplink_bytes" in row:
        return "communication_cumulative_uplink_bytes"
    return "communication_cumulative_bytes"


def collect(results_root: Path) -> list[dict]:
    records = []
    for clients in (10, 100):
        for scenario, method, _, _ in ARMS:
            rows = read_rounds(results_root, clients, scenario)
            final = next(row for row in rows if int(float(row["round"])) == ROUNDS)
            total_uplink = float(final[uplink_column(final)])
            mean_uplink = total_uplink / (clients * ROUNDS)
            accuracy = float(final["validation_accuracy"])
            weighted_recall = float(final["validation_weighted_recall"])
            if abs(accuracy - weighted_recall) > 1e-9:
                raise ValueError(
                    f"Weighted recall identity failed for {clients}/{scenario}"
                )
            records.append(
                {
                    "clients": clients,
                    "scenario": scenario,
                    "method": method,
                    "round": ROUNDS,
                    "accuracy": accuracy,
                    "macro_recall": float(final["validation_macro_recall"]),
                    "weighted_recall": weighted_recall,
                    "macro_f1": float(final["validation_macro_f1"]),
                    "weighted_f1": float(final["validation_weighted_f1"]),
                    "minority_recall": float(final["validation_minority_recall"]),
                    "mean_uplink_bytes_per_client_round": mean_uplink,
                    "total_uplink_bytes": total_uplink,
                    "dense_download_bytes_per_client_round": DENSE_MESSAGE_BYTES,
                    "uplink_compression_ratio": DENSE_MESSAGE_BYTES / mean_uplink,
                    "total_traffic_reduction": (2 * DENSE_MESSAGE_BYTES)
                    / (DENSE_MESSAGE_BYTES + mean_uplink),
                }
            )
    return records


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_communication_table(output_dir: Path, records: list[dict]) -> None:
    fields = [
        "clients",
        "scenario",
        "method",
        "mean_uplink_bytes_per_client_round",
        "total_uplink_bytes",
        "dense_download_bytes_per_client_round",
        "uplink_compression_ratio",
        "total_traffic_reduction",
    ]
    rows = [{field: record[field] for field in fields} for record in records]
    write_csv(output_dir / "tables" / "communication.csv", rows)


def write_summaries(output_dir: Path, records: list[dict]) -> None:
    summary_dir = output_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for clients in (10, 100):
        payload = {
            "clients": clients,
            "round": ROUNDS,
            "runs": [record for record in records if record["clients"] == clients],
        }
        (summary_dir / f"k{clients}_summary.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


def plot_learning_curves(results_root: Path, output_dir: Path) -> None:
    metrics = [
        ("validation_macro_f1", "Macro-F1 (%)"),
        ("validation_accuracy", "Accuracy (%)"),
        ("validation_weighted_f1", "Weighted-F1 (%)"),
    ]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(9.0, 4.8), sharex=True)
    for row_index, clients in enumerate((10, 100)):
        for column_index, (metric, ylabel) in enumerate(metrics):
            axis = axes[row_index][column_index]
            for scenario, method, colour, line_style in ARMS:
                rows = read_rounds(results_root, clients, scenario)
                points = [row for row in rows if int(float(row["round"])) >= 1]
                x = [int(float(row["round"])) for row in points]
                y = [100 * float(row[metric]) for row in points]
                proposed = scenario == "fedmpsq"
                axis.plot(
                    x,
                    y,
                    label=method,
                    color=colour,
                    linestyle=line_style,
                    linewidth=1.8 if proposed else 0.9,
                    marker="o" if proposed else None,
                    markersize=2.8,
                    markevery=[-1] if proposed else None,
                )
            axis.set_title(f"K={clients}: {ylabel}")
            axis.set_xlabel("Communication round")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.18, linewidth=0.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "metrics_rounds.png", dpi=200, bbox_inches="tight")
    figure.savefig(output_dir / "metrics_rounds.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_minority_recall(results_root: Path, output_dir: Path) -> None:
    path = results_root / "minority_recall.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        expected_split = (
            "client_validation"
            if row["scenario"] == "fedmpsq"
            else "global_test"
        )
        if row["evaluation_split"] != expected_split:
            raise ValueError(
                f"Unexpected minority-recall split for {row['scenario']}: "
                f"{row['evaluation_split']}"
            )
    scenarios = [scenario for scenario, _, _, _ in MINORITY_ARMS]
    labels = [method for _, method, _, _ in MINORITY_ARMS]
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
    for axis, clients in zip(axes, (10, 100)):
        scale = [row for row in rows if int(row["clients"]) == clients]
        class_ids = sorted({int(row["class_id"]) for row in scale})
        class_names = {
            int(row["class_id"]): row["class_name"] for row in scale
        }
        lookup = {
            (row["scenario"], int(row["class_id"])): 100 * float(row["recall"])
            for row in scale
        }
        matrix = np.array(
            [[lookup[(scenario, class_id)] for scenario in scenarios] for class_id in class_ids]
        )
        matrix = np.vstack([matrix, matrix.mean(axis=0)])
        ylabels = [class_names[class_id] for class_id in class_ids] + ["Minority mean"]
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=45, aspect="auto")
        axis.set_title(f"K={clients} clients")
        axis.set_xticks(range(len(labels)), labels, rotation=90)
        axis.set_yticks(range(len(ylabels)), ylabels)
        for y_index in range(matrix.shape[0]):
            for x_index in range(matrix.shape[1]):
                value = matrix[y_index, x_index]
                if value > 0:
                    axis.text(
                        x_index,
                        y_index,
                        f"{value:.0f}" if value >= 10 else f"{value:.1f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if value > 26 else "#172331",
                    )
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "minority_recall.png", dpi=200, bbox_inches="tight")
    figure.savefig(output_dir / "minority_recall.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check-only", action="store_true", help="Validate snapshots without plotting."
    )
    args = parser.parse_args()
    results_root = args.results_root.resolve()
    output_dir = args.output_dir.resolve()
    validate_manifest(results_root)
    records = collect(results_root)
    if len(records) != 18:
        raise RuntimeError(f"Expected 18 reported runs; got {len(records)}")
    if args.check_only:
        print(f"Validated {len(records)} runs; no artifacts written")
        return
    write_csv(output_dir / "tables" / "main_results.csv", records)
    write_communication_table(output_dir, records)
    write_summaries(output_dir, records)
    plot_learning_curves(results_root, output_dir / "figures")
    plot_minority_recall(results_root, output_dir / "figures")
    print(f"Validated {len(records)} runs; wrote artifacts to {output_dir}")


if __name__ == "__main__":
    main()
