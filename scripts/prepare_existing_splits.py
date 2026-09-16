#!/usr/bin/env python
"""Convert existing client/global-test CSV splits to NPZ partitions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.ciciot2023 import prepare_existing_ciciot2023_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits-dir",
        required=True,
        help="Directory containing client_0.csv ... client_9.csv and global_test.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/partitions/ciciot2023-existing-splits",
        help="Directory for client_*.npz, global_test.npz, metadata.json",
    )
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--client-pattern", default="client_*.csv")
    parser.add_argument("--global-test-file", default="global_test.csv")
    parser.add_argument("--expected-clients", type=int, default=10)
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
        help="Base seed for deterministic local train/validation fit scope.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Rows per CSV chunk. Lower this if conversion is still memory constrained.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare_existing_ciciot2023_splits(
        splits_dir=args.splits_dir,
        output_dir=args.output_dir,
        label_column=args.label_column,
        client_pattern=args.client_pattern,
        global_test_file=args.global_test_file,
        expected_clients=args.expected_clients,
        chunk_size=args.chunk_size,
        client_val_ratio=args.client_val_ratio,
        seed=args.seed,
    )
    summary = {
        "output_dir": args.output_dir,
        "input_dim": metadata["input_dim"],
        "client_examples": metadata["client_examples"],
        "global_test_examples": metadata["global_test_examples"],
        "partition_strategy": metadata["partition_strategy"],
        "preprocessing_fit_scope": metadata["preprocessing"]["fit_scope"],
        "global_test_sha256": metadata["global_test_provenance"][
            "prepared_npz_sha256"
        ],
        "client_statistics": metadata["client_statistics"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
