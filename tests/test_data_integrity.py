from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import data.ciciot2023 as ciciot2023_module
from data.ciciot2023 import (
    CICIOT2023_LABELS,
    encode_labels,
    prepare_ciciot2023_partitions,
    prepare_existing_ciciot2023_splits,
    stratified_global_test_split,
)
from data.client_statistics import (
    MATRIX_FILENAME,
    SUMMARY_FILENAME,
    compute_client_distribution_statistics,
    write_client_distribution_statistics,
)
from data.integrity import sha256_array, sha256_file, sha256_indices
from data.partition_manifest import (
    MANIFEST_SCHEMA_VERSION,
    create_partition_manifest,
    verify_partition_manifest,
)
from data.red_packet import RedPacketConfig, red_packet_partition
from data.splits import deterministic_split_indices


class ExistingSplitPreprocessingTests(unittest.TestCase):
    def test_encoder_imputer_and_scaler_fit_only_local_training_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "prepared"
            source.mkdir()
            val_ratio = 0.2
            seed = 42
            expected_fit_values: list[float] = []
            for client_id in range(2):
                train_indices, validation_indices = deterministic_split_indices(
                    5,
                    val_ratio,
                    seed + client_id,
                )
                values = np.arange(5, dtype=np.float64) + client_id * 10.0
                categories = np.full(5, "training_category", dtype=object)
                values[validation_indices] = 1_000_000.0 + client_id
                categories[validation_indices] = "validation_only_category"
                expected_fit_values.extend(values[train_indices].tolist())
                pd.DataFrame(
                    {
                        "value": values,
                        "kind": categories,
                        "label": [CICIOT2023_LABELS[client_id]] * 5,
                    }
                ).to_csv(source / f"client_{client_id}.csv", index=False)
            pd.DataFrame(
                {
                    "value": [9_000_000.0, 10_000_000.0],
                    "kind": ["test_only_category", "test_only_category"],
                    "label": [CICIOT2023_LABELS[0], CICIOT2023_LABELS[1]],
                }
            ).to_csv(source / "global_test.csv", index=False)

            original_write_scaled_npz = ciciot2023_module._write_scaled_npz
            completed_x_paths: list[Path] = []

            def tracked_write_scaled_npz(**kwargs) -> None:
                for completed_x_path in completed_x_paths:
                    self.assertFalse(completed_x_path.exists())
                if kwargs["x_path"].name.startswith("client_"):
                    self.assertFalse(
                        (kwargs["x_path"].parent / "global_test_x.npy").exists()
                    )
                original_write_scaled_npz(**kwargs)
                self.assertFalse(kwargs["scaled_x_path"].exists())
                completed_x_paths.append(kwargs["x_path"])

            with patch.object(
                ciciot2023_module,
                "_write_scaled_npz",
                side_effect=tracked_write_scaled_npz,
            ):
                metadata = prepare_existing_ciciot2023_splits(
                    source,
                    output,
                    expected_clients=2,
                    chunk_size=2,
                    client_val_ratio=val_ratio,
                    seed=seed,
                )

            self.assertEqual(len(completed_x_paths), 3)

            value_column = metadata["feature_columns"].index("value")
            expected_mean = float(np.mean(expected_fit_values))
            expected_scale = float(np.std(expected_fit_values, ddof=0))
            self.assertAlmostEqual(
                metadata["scaler_mean"][value_column],
                expected_mean,
            )
            self.assertAlmostEqual(
                metadata["scaler_scale"][value_column],
                expected_scale,
            )
            self.assertAlmostEqual(
                metadata["feature_medians"]["value"],
                float(np.median(expected_fit_values)),
            )
            self.assertIn("kind_training_category", metadata["feature_columns"])
            self.assertNotIn("kind_validation_only_category", metadata["feature_columns"])
            self.assertNotIn("kind_test_only_category", metadata["feature_columns"])
            self.assertEqual(
                metadata["preprocessing"]["fit_scope"],
                "deterministic_local_training_subsets_only",
            )
            self.assertFalse(
                metadata["global_test_provenance"][
                    "preprocessing_fit_includes_global_test"
                ]
            )
            self.assertEqual(
                metadata["preprocessing"]["transform_scope"],
                ["client_training", "client_validation", "global_test"],
            )
            self.assertEqual(
                metadata["preprocessing"]["transform_statistics_source"],
                "deterministic_local_training_subsets_only",
            )
            # The held-out rows are transformed with the train-only statistics,
            # rather than changing the fitted mean/scale themselves.
            train_indices, validation_indices = deterministic_split_indices(
                5,
                val_ratio,
                seed,
            )
            with np.load(output / "client_000.npz") as client_payload:
                client_x = client_payload["x"]
            validation_value = client_x[np.asarray(validation_indices)[0], value_column]
            self.assertAlmostEqual(
                validation_value,
                (1_000_000.0 - expected_mean) / expected_scale,
                places=2,
            )
            with np.load(output / "global_test.npz") as test_payload:
                test_value = float(test_payload["x"][0, value_column])
            self.assertAlmostEqual(
                test_value,
                (9_000_000.0 - expected_mean) / expected_scale,
                places=2,
            )
            self.assertEqual(
                metadata["global_test_provenance"]["source_sha256"],
                sha256_file(source / "global_test.csv"),
            )
            self.assertEqual(
                metadata["global_test_provenance"]["prepared_npz_sha256"],
                sha256_file(output / "global_test.npz"),
            )
            self.assertTrue((output / MATRIX_FILENAME).exists())
            self.assertTrue((output / SUMMARY_FILENAME).exists())

    def test_generated_split_preprocessing_excludes_test_and_local_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_path = root / "raw.csv"
            output = root / "prepared"
            labels = np.repeat(CICIOT2023_LABELS[:4], 100)
            y = encode_labels(pd.Series(labels))
            train_indices, test_indices = stratified_global_test_split(y, 0.2, 42)
            client_indices = red_packet_partition(
                y[train_indices],
                RedPacketConfig(seed=42),
            )
            fit_source_indices: list[int] = []
            validation_only_source_indices: list[int] = []
            for client_id, indices in enumerate(client_indices):
                local_train, local_validation = deterministic_split_indices(
                    len(indices),
                    0.2,
                    42 + client_id,
                )
                fit_source_indices.extend(
                    train_indices[indices[np.asarray(local_train, dtype=np.int64)]].tolist()
                )
                validation_only = set(local_validation) - set(local_train)
                validation_only_source_indices.extend(
                    train_indices[
                        indices[np.asarray(sorted(validation_only), dtype=np.int64)]
                    ].tolist()
                )

            values = np.arange(len(labels), dtype=np.float64)
            categories = np.full(len(labels), "training_category", dtype=object)
            values[np.asarray(validation_only_source_indices, dtype=np.int64)] = 1_000_000.0
            categories[np.asarray(validation_only_source_indices, dtype=np.int64)] = (
                "validation_only_category"
            )
            values[test_indices] = 9_000_000.0
            categories[test_indices] = "test_only_category"
            pd.DataFrame(
                {"value": values, "kind": categories, "label": labels}
            ).to_csv(raw_path, index=False)

            metadata = prepare_ciciot2023_partitions(
                raw_path,
                output,
                client_val_ratio=0.2,
                seed=42,
            )
            value_column = metadata["feature_columns"].index("value")
            fit_values = values[np.asarray(fit_source_indices, dtype=np.int64)]
            self.assertAlmostEqual(
                metadata["scaler_mean"][value_column],
                float(np.mean(fit_values)),
            )
            self.assertAlmostEqual(
                metadata["scaler_scale"][value_column],
                float(np.std(fit_values, ddof=0)),
            )
            self.assertAlmostEqual(
                metadata["feature_medians"]["value"],
                float(np.median(fit_values)),
            )
            self.assertEqual(
                metadata["preprocessing"]["transform_scope"],
                ["client_training", "client_validation", "global_test"],
            )
            self.assertNotIn("kind_validation_only_category", metadata["feature_columns"])
            self.assertNotIn("kind_test_only_category", metadata["feature_columns"])
            self.assertEqual(
                metadata["global_test_provenance"]["selection_indices_sha256"],
                sha256_indices(test_indices),
            )
            with np.load(output / "global_test.npz") as test_payload:
                test_value = float(test_payload["x"][0, value_column])
            self.assertAlmostEqual(
                test_value,
                (9_000_000.0 - float(np.mean(fit_values)))
                / float(np.std(fit_values, ddof=0)),
                places=2,
            )


class ClientDistributionStatisticsTests(unittest.TestCase):
    def test_matrix_and_skew_metric_definitions_are_exported(self) -> None:
        labels = [
            np.asarray([0, 0, 1], dtype=np.int64),
            np.asarray([1, 2, 2, 2], dtype=np.int64),
        ]
        summary = compute_client_distribution_statistics(labels, ["A", "B", "C"])
        first = summary["clients"][0]
        self.assertEqual(first["class_counts"], {"A": 2, "B": 1, "C": 0})
        self.assertAlmostEqual(first["missing_class_rate"], 1.0 / 3.0)
        self.assertEqual(first["observed_class_imbalance_ratio"], 2.0)
        self.assertGreater(first["js_divergence_to_pooled_nats"], 0.0)
        self.assertIn("pooled_distribution_scope", summary["definitions"])

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_client_distribution_statistics(
                temp_dir,
                labels,
                ["A", "B", "C"],
            )
            matrix_path = Path(temp_dir) / MATRIX_FILENAME
            with matrix_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["client_id"] for row in rows], ["0", "1"])
            self.assertTrue(
                all(
                    row["population_scope"]
                    == "deterministic_local_training_subsets"
                    for row in rows
                )
            )
            self.assertEqual(rows[0]["C"], "0")
            self.assertEqual(artifacts["matrix_sha256"], sha256_file(matrix_path))


def _write_manifest_ready_partitions(root: Path) -> None:
    root.mkdir()
    train_hashes: list[str] = []
    validation_hashes: list[str] = []
    train_examples: list[int] = []
    validation_examples: list[int] = []
    client_training_labels: list[np.ndarray] = []
    for client_id in range(10):
        x = np.arange(48, dtype=np.float32).reshape(8, 6) + client_id * 1000
        y = (np.arange(8, dtype=np.int64) + client_id) % 4
        np.savez_compressed(root / f"client_{client_id:03d}.npz", x=x, y=y)
        train_indices, val_indices = deterministic_split_indices(8, 0.2, 42 + client_id)
        train_hashes.append(sha256_indices(train_indices))
        validation_hashes.append(sha256_indices(val_indices))
        train_examples.append(len(train_indices))
        validation_examples.append(len(val_indices))
        client_training_labels.append(y[np.asarray(train_indices, dtype=np.int64)])
    x_test = np.arange(72, dtype=np.float32).reshape(12, 6)
    y_test = np.arange(12, dtype=np.int64) % 4
    global_test_path = root / "global_test.npz"
    np.savez_compressed(global_test_path, x=x_test, y=y_test)
    statistics_artifacts = write_client_distribution_statistics(
        root,
        client_training_labels,
        [f"class_{class_id}" for class_id in range(4)],
    )
    metadata = {
        "input_dim": 6,
        "num_classes": 4,
        "num_clients": 10,
        "preprocessing": {
            "fit_scope": "deterministic_local_training_subsets_only",
            "excluded_from_fit": ["client_validation", "global_test"],
            "client_val_ratio": 0.2,
            "seed": 42,
            "client_train_indices_sha256": train_hashes,
            "client_validation_indices_sha256": validation_hashes,
            "client_fit_examples": train_examples,
            "client_validation_examples": validation_examples,
        },
        "global_test_provenance": {
            "status": "complete",
            "role": "evaluation_only",
            "immutable": True,
            "source_kind": "synthetic_test_fixture",
            "prepared_file": global_test_path.name,
            "prepared_npz_sha256": sha256_file(global_test_path),
            "prepared_x_sha256": sha256_array(x_test),
            "prepared_y_sha256": sha256_array(y_test),
            "preprocessing_fit_includes_global_test": False,
        },
        "client_statistics": statistics_artifacts,
    }
    (root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


class ImmutableManifestTests(unittest.TestCase):
    def test_manifest_freezes_global_test_and_rejects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "partitions"
            _write_manifest_ready_partitions(root)
            manifest_path = Path(temp_dir) / "client_partition.json"
            manifest = create_partition_manifest(
                root,
                manifest_path,
                dataset="existing_dataset",
                num_clients=10,
                client_val_ratio=0.2,
                seed=42,
            )
            self.assertEqual(manifest["schema_version"], MANIFEST_SCHEMA_VERSION)
            self.assertTrue(manifest["global_test"]["immutable"])
            self.assertEqual(manifest["global_test"]["role"], "evaluation_only")

            settings = SimpleNamespace(
                data=SimpleNamespace(
                    num_clients=10,
                    dataset="existing_dataset",
                    regenerate_partition=False,
                    partition_file=str(manifest_path),
                    partitions_dir=str(root),
                    client_val_ratio=0.2,
                    seed=42,
                )
            )
            self.assertEqual(
                verify_partition_manifest(settings),
                manifest["partition_hash"],
            )
            with np.load(root / "global_test.npz") as payload:
                x_test = payload["x"].copy()
                y_test = payload["y"].copy()
            x_test[0, 0] += 1.0
            np.savez_compressed(root / "global_test.npz", x=x_test, y=y_test)
            with self.assertRaisesRegex(ValueError, "global-test provenance|integrity"):
                verify_partition_manifest(settings)

    def test_legacy_metadata_without_provenance_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "partitions"
            _write_manifest_ready_partitions(root)
            metadata_path = root / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.pop("global_test_provenance")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "global_test_provenance"):
                create_partition_manifest(
                    root,
                    Path(temp_dir) / "client_partition.json",
                    dataset="existing_dataset",
                    num_clients=10,
                    client_val_ratio=0.2,
                    seed=42,
                )

    def test_missing_train_only_statistics_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "partitions"
            _write_manifest_ready_partitions(root)
            metadata_path = root / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.pop("client_statistics")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "train-only client-distribution"):
                create_partition_manifest(
                    root,
                    Path(temp_dir) / "client_partition.json",
                    dataset="existing_dataset",
                    num_clients=10,
                    client_val_ratio=0.2,
                    seed=42,
                )


if __name__ == "__main__":
    unittest.main()
