#!/usr/bin/env python
"""Re-partition the existing client pool into N Red Packet clients.

The source CSV splits are read once, their rows are pooled in client-id order,
and the pool is redistributed with the same Red Packet rule the frozen
10-client split obeys.  The global test CSV is never repartitioned; it is only
re-encoded with the statistics fitted on this partition's local-training rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.ciciot2023 import prepare_repartitioned_ciciot2023_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits-dir",
        required=True,
        help="Directory containing client_0.csv ... client_9.csv and global_test.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for client_*.npz, global_test.npz, metadata.json",
    )
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--client-pattern", default="client_*.csv")
    parser.add_argument("--global-test-file", default="global_test.csv")
    parser.add_argument("--source-clients", type=int, default=10)
    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--min-label-clients", type=int, default=1)
    parser.add_argument(
        "--max-label-clients",
        type=int,
        default=None,
        help="Defaults to --num-clients, the rule the frozen 10-client split obeys",
    )
    parser.add_argument("--lognormal-sigma", type=float, default=1.25)
    parser.add_argument(
        "--partition-seed",
        type=int,
        default=42,
        help="Seed of the Red Packet allocation itself",
    )
    parser.add_argument(
        "--client-val-ratio",
        type=float,
        default=0.2,
        help="Must match data.client_val_ratio used by every experiment.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for the deterministic local train/validation split.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Rows per CSV chunk. Lower this if conversion is memory constrained.",
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help=(
            "Scratch directory for the uncompressed staging copy of the pool. "
            "Defaults to --output-dir; point it at a volume outside the "
            "published output when disk is tight."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare_repartitioned_ciciot2023_splits(
        splits_dir=args.splits_dir,
        output_dir=args.output_dir,
        label_column=args.label_column,
        client_pattern=args.client_pattern,
        global_test_file=args.global_test_file,
        source_clients=args.source_clients,
        num_clients=args.num_clients,
        min_label_clients=args.min_label_clients,
        max_label_clients=args.max_label_clients,
        lognormal_sigma=args.lognormal_sigma,
        partition_seed=args.partition_seed,
        chunk_size=args.chunk_size,
        client_val_ratio=args.client_val_ratio,
        seed=args.seed,
        temp_dir=args.temp_dir,
    )
    client_examples = metadata["client_examples"]
    summary = {
        "output_dir": args.output_dir,
        "num_clients": metadata["num_clients"],
        "input_dim": metadata["input_dim"],
        "partition_strategy": metadata["partition_strategy"],
        "red_packet": metadata["red_packet"],
        "pooled_rows": metadata["source_partition"]["pooled_rows"],
        "client_rows_total": int(sum(client_examples)),
        "client_rows_min": int(min(client_examples)),
        "client_rows_max": int(max(client_examples)),
        "global_test_examples": metadata["global_test_examples"],
        "preprocessing_fit_scope": metadata["preprocessing"]["fit_scope"],
        "global_test_sha256": metadata["global_test_provenance"][
            "prepared_npz_sha256"
        ],
        "client_statistics": metadata["client_statistics"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
