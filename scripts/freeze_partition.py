#!/usr/bin/env python
"""Freeze an existing 10-client NPZ partition without repartitioning it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.partition_manifest import create_partition_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partitions-dir", required=True)
    parser.add_argument(
        "--partition-file",
        default="data_partitions/client_partition.json",
    )
    parser.add_argument("--dataset", default="existing_dataset")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--client-val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = create_partition_manifest(
        args.partitions_dir,
        args.partition_file,
        dataset=args.dataset,
        num_clients=args.num_clients,
        client_val_ratio=args.client_val_ratio,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "partition_hash": manifest["partition_hash"],
                "manifest_schema_version": manifest["schema_version"],
                "global_test_sha256": manifest["global_test"]["sha256"],
                "global_test_x_sha256": manifest["global_test"]["x_sha256"],
                "global_test_y_sha256": manifest["global_test"]["y_sha256"],
                "preprocessing_fit_scope": manifest["data_contract"][
                    "preprocessing_fit_scope"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
