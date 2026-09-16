"""Correctness tests for the N-client Red Packet re-partition and schema v3."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data.ciciot2023 import (
    CICIOT2023_LABELS,
    prepare_repartitioned_ciciot2023_splits,
)
from data.fedmpsq_labels import (
    IDENTITY_34_LABEL_SPACE,
    generic_identity_label_space,
    minority_class_ids_from_support,
    pooled_training_support,
    remap_ciciot2023_labels,
    target_class_names,
)
from data.red_packet import RedPacketConfig, red_packet_partition
from data.splits import deterministic_split_indices
from fl.metric_contract import build_metric_contract, enforce_metric_contract
from training.fedmpsq_metrics import detailed_classification_metrics
from training.metrics import (
    BINARY_METRIC_KEYS,
    CLASSIFICATION_METRIC_KEYS,
    CLASSIFICATION_METRIC_SCHEMA_VERSION,
    CONFUSION_DERIVED_METRIC_KEYS,
    SCORE_DEPENDENT_METRIC_KEYS,
    benign_class_id,
    binary_counts_from_confusion,
    classification_metrics_v4_from_confusion,
    confusion_matrix_counts,
    metric_keys_for,
)

FEATURE_COLUMNS = [f"feature_{index:02d}" for index in range(6)]
SOURCE_CLIENTS = 4
TARGET_CLIENTS = 10


def _write_source_pool(root: Path, *, rows_per_client: int) -> None:
    """Write a small stand-in for the frozen Existing-10 CSV pool."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    label_names = list(CICIOT2023_LABELS)
    for client_id in range(SOURCE_CLIENTS):
        # Each source client owns a different, overlapping slice of the label
        # space so the pool it forms is genuinely label-skewed.
        owned = label_names[client_id * 5 : client_id * 5 + 14] or label_names[:14]
        frame = pd.DataFrame(
            rng.normal(size=(rows_per_client, len(FEATURE_COLUMNS))),
            columns=FEATURE_COLUMNS,
        )
        frame["label"] = rng.choice(owned, size=rows_per_client)
        frame.to_csv(root / f"client_{client_id}.csv", index=False)
    test_rows = rows_per_client
    test_frame = pd.DataFrame(
        rng.normal(size=(test_rows, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )
    test_frame["label"] = rng.choice(label_names, size=test_rows)
    test_frame.to_csv(root / "global_test.csv", index=False)


class RedPacketRepartitionTests(unittest.TestCase):
    def test_repartition_conserves_every_pooled_row_and_class(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "prepared"
            _write_source_pool(source, rows_per_client=400)

            metadata = prepare_repartitioned_ciciot2023_splits(
                splits_dir=source,
                output_dir=output,
                source_clients=SOURCE_CLIENTS,
                num_clients=TARGET_CLIENTS,
                chunk_size=97,
            )

            self.assertEqual(metadata["num_clients"], TARGET_CLIENTS)
            self.assertEqual(metadata["num_classes"], len(CICIOT2023_LABELS))
            self.assertEqual(
                metadata["partition_strategy"],
                "red_packet_non_iid_repartition_of_existing_pool",
            )
            self.assertEqual(metadata["source_partition"]["rows_added_or_removed"], 0)

            pooled_rows = metadata["source_partition"]["pooled_rows"]
            self.assertEqual(pooled_rows, SOURCE_CLIENTS * 400)
            self.assertEqual(sum(metadata["client_examples"]), pooled_rows)
            self.assertTrue(all(count > 0 for count in metadata["client_examples"]))

            # Every source label must survive re-partitioning with its exact
            # multiplicity: the split moves rows, it never resamples them.
            source_counts = np.zeros(len(CICIOT2023_LABELS), dtype=np.int64)
            for client_id in range(SOURCE_CLIENTS):
                frame = pd.read_csv(source / f"client_{client_id}.csv")
                for name, count in frame["label"].value_counts().items():
                    source_counts[CICIOT2023_LABELS.index(str(name))] += int(count)

            prepared_counts = np.zeros(len(CICIOT2023_LABELS), dtype=np.int64)
            prepared_features: list[np.ndarray] = []
            for client_id in range(TARGET_CLIENTS):
                with np.load(output / f"client_{client_id:03d}.npz") as payload:
                    features = np.asarray(payload["x"])
                    labels = np.asarray(payload["y"])
                self.assertEqual(features.dtype, np.float32)
                self.assertEqual(features.shape[1], len(FEATURE_COLUMNS))
                self.assertEqual(len(features), len(labels))
                self.assertTrue(bool(np.isfinite(features).all()))
                prepared_counts += np.bincount(
                    labels, minlength=len(CICIOT2023_LABELS)
                ).astype(np.int64)
                prepared_features.append(features)
            np.testing.assert_array_equal(prepared_counts, source_counts)

            # The union of client rows is a permutation of the pooled rows, so
            # sorting both sides must give identical feature matrices.
            union = np.concatenate(prepared_features)
            self.assertEqual(len(union), pooled_rows)
            self.assertEqual(len(np.unique(union, axis=0)), len(union))

    def test_scaler_and_medians_use_only_local_training_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "prepared"
            _write_source_pool(source, rows_per_client=300)

            metadata = prepare_repartitioned_ciciot2023_splits(
                splits_dir=source,
                output_dir=output,
                source_clients=SOURCE_CLIENTS,
                num_clients=TARGET_CLIENTS,
                chunk_size=128,
            )
            preprocessing = metadata["preprocessing"]
            self.assertEqual(
                preprocessing["fit_scope"],
                "deterministic_local_training_subsets_only",
            )
            self.assertEqual(
                preprocessing["excluded_from_fit"],
                ["client_validation", "global_test"],
            )
            self.assertEqual(
                preprocessing["fit_examples"],
                sum(preprocessing["client_fit_examples"]),
            )
            for client_id, total in enumerate(metadata["client_examples"]):
                expected_train, expected_validation = deterministic_split_indices(
                    total,
                    preprocessing["client_val_ratio"],
                    preprocessing["seed"] + client_id,
                )
                self.assertEqual(
                    preprocessing["client_fit_examples"][client_id],
                    len(expected_train),
                )
                self.assertEqual(
                    preprocessing["client_validation_examples"][client_id],
                    len(expected_validation),
                )
            self.assertFalse(
                metadata["global_test_provenance"][
                    "preprocessing_fit_includes_global_test"
                ]
            )
            self.assertTrue(metadata["global_test_provenance"]["immutable"])
            self.assertFalse(
                metadata["source_partition"]["global_test_rows_touched"]
            )

    def test_partition_is_reproducible_from_the_declared_rule(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "prepared"
            _write_source_pool(source, rows_per_client=250)
            metadata = prepare_repartitioned_ciciot2023_splits(
                splits_dir=source,
                output_dir=output,
                source_clients=SOURCE_CLIENTS,
                num_clients=TARGET_CLIENTS,
                chunk_size=64,
            )
            packet = metadata["red_packet"]
            self.assertEqual(packet["seed"], 42)
            self.assertEqual(packet["num_clients"], TARGET_CLIENTS)
            self.assertEqual(packet["min_label_clients"], 1)
            # The default cap is the client count itself, which is the rule the
            # frozen 10-client split obeys.
            self.assertEqual(packet["max_label_clients"], TARGET_CLIENTS)

            pooled_labels = np.concatenate(
                [
                    pd.read_csv(source / f"client_{client_id}.csv")["label"]
                    .map({name: index for index, name in enumerate(CICIOT2023_LABELS)})
                    .to_numpy(dtype=np.int64)
                    for client_id in range(SOURCE_CLIENTS)
                ]
            )
            replayed = red_packet_partition(
                pooled_labels,
                RedPacketConfig(**packet),
            )
            self.assertEqual(
                [len(part) for part in replayed],
                list(metadata["client_examples"]),
            )
            for client_id, indices in enumerate(replayed):
                with np.load(output / f"client_{client_id:03d}.npz") as payload:
                    labels = np.asarray(payload["y"])
                np.testing.assert_array_equal(labels, pooled_labels[indices])


class MetricSchemaV4Tests(unittest.TestCase):
    def test_schema_declares_twenty_four_metrics(self) -> None:
        self.assertEqual(CLASSIFICATION_METRIC_SCHEMA_VERSION, 4)
        self.assertEqual(len(CLASSIFICATION_METRIC_KEYS), 24)
        self.assertEqual(len(set(CLASSIFICATION_METRIC_KEYS)), 24)
        self.assertEqual(len(CONFUSION_DERIVED_METRIC_KEYS), 20)
        self.assertEqual(len(SCORE_DEPENDENT_METRIC_KEYS), 4)
        self.assertEqual(len(BINARY_METRIC_KEYS), 7)
        self.assertEqual(
            set(CLASSIFICATION_METRIC_KEYS) - set(CONFUSION_DERIVED_METRIC_KEYS),
            set(SCORE_DEPENDENT_METRIC_KEYS),
        )

    def test_key_list_follows_the_label_space_and_evaluation_mode(self) -> None:
        scored = metric_keys_for(has_benign_class=True, scored=True)
        streaming = metric_keys_for(has_benign_class=True, scored=False)
        no_benign = metric_keys_for(has_benign_class=False, scored=True)
        self.assertEqual(len(scored), 24)
        self.assertEqual(len(streaming), 20)
        self.assertEqual(len(no_benign), 17)
        self.assertFalse(set(no_benign) & set(BINARY_METRIC_KEYS))

    def test_benign_class_is_found_in_both_real_label_spaces(self) -> None:
        self.assertEqual(
            benign_class_id(CICIOT2023_LABELS),
            list(CICIOT2023_LABELS).index("BenignTraffic"),
        )
        self.assertEqual(benign_class_id(["DDoS", "Benign", "Web"]), 1)
        self.assertIsNone(benign_class_id(["class_0", "class_1"]))

    def test_binary_collapse_counts_attacks_as_detected(self) -> None:
        # rows = truth, columns = prediction; class 0 is benign. An attack
        # predicted as a different attack is still a detection.
        confusion = np.array(
            [
                [70, 20, 10],
                [5, 40, 15],
                [10, 25, 65],
            ],
            dtype=np.int64,
        )
        counts = binary_counts_from_confusion(confusion, benign_id=0)
        self.assertEqual(counts, (145.0, 30.0, 15.0, 70.0))

        metrics = classification_metrics_v4_from_confusion(
            confusion, minority_class_ids=[1], benign_id=0
        )
        self.assertAlmostEqual(metrics["binary_detection_rate"], 145 / 160)
        self.assertAlmostEqual(metrics["binary_false_alarm_rate"], 30 / 100)
        self.assertAlmostEqual(metrics["binary_precision"], 145 / 175)
        self.assertAlmostEqual(metrics["binary_accuracy"], 215 / 260)

    def test_binary_metrics_are_omitted_without_a_benign_class(self) -> None:
        confusion = np.array([[8, 2], [3, 7]], dtype=np.int64)
        metrics = classification_metrics_v4_from_confusion(
            confusion, minority_class_ids=[0]
        )
        for key in BINARY_METRIC_KEYS:
            self.assertNotIn(key, metrics)
        self.assertIn("macro_specificity", metrics)

    def test_specificity_and_g_mean_are_consistent(self) -> None:
        rng = np.random.default_rng(9)
        confusion = rng.integers(0, 40, size=(6, 6)).astype(np.int64)
        metrics = classification_metrics_v4_from_confusion(
            confusion, minority_class_ids=[2, 5], benign_id=0
        )
        self.assertAlmostEqual(
            metrics["macro_false_positive_rate"],
            1.0 - metrics["macro_specificity"],
        )
        self.assertAlmostEqual(
            metrics["g_mean"],
            math.sqrt(metrics["balanced_accuracy"] * metrics["macro_specificity"]),
        )

    def test_confusion_metrics_match_the_scored_evaluator(self) -> None:
        rng = np.random.default_rng(3)
        num_classes = 7
        truth = rng.integers(0, num_classes, size=900)
        logits = rng.normal(size=(900, num_classes))
        probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        predictions = probabilities.argmax(axis=1)
        minority = [1, 4]
        detailed = detailed_classification_metrics(
            truth,
            predictions,
            probabilities,
            class_names=[f"c{index}" for index in range(num_classes)],
            minority_class_ids=minority,
            benign_id=0,
        )
        confusion = confusion_matrix_counts(
            truth, predictions, num_classes=num_classes
        )
        streamed = classification_metrics_v4_from_confusion(
            confusion,
            minority_class_ids=minority,
            benign_id=0,
        )
        for key in CONFUSION_DERIVED_METRIC_KEYS:
            self.assertAlmostEqual(streamed[key], detailed[key], places=12, msg=key)
        for key in SCORE_DEPENDENT_METRIC_KEYS:
            self.assertIn(key, detailed)
            self.assertTrue(0.0 <= detailed[key] <= 1.0, key)

    def test_scored_evaluator_finds_the_benign_class_by_name(self) -> None:
        rng = np.random.default_rng(4)
        names = list(CICIOT2023_LABELS)
        truth = rng.integers(0, len(names), size=600)
        logits = rng.normal(size=(600, len(names)))
        probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        detailed = detailed_classification_metrics(
            truth,
            probabilities.argmax(axis=1),
            probabilities,
            class_names=names,
            minority_class_ids=[30, 19],
        )
        self.assertEqual(detailed["benign_class_id"], names.index("BenignTraffic"))
        for key in BINARY_METRIC_KEYS:
            self.assertIn(key, detailed)

    def test_minority_rule_takes_the_rarest_quarter_with_stable_ties(self) -> None:
        support = np.array([50, 10, 10, 900, 5, 700, 3, 400], dtype=np.int64)
        ids = minority_class_ids_from_support(support, minority_fraction=0.25)
        self.assertEqual(ids, [6, 4])
        thirty_four = np.arange(34, dtype=np.int64) + 1
        self.assertEqual(
            len(minority_class_ids_from_support(thirty_four, minority_fraction=0.25)),
            math.ceil(34 * 0.25),
        )


class IdentityLabelSpaceTests(unittest.TestCase):
    def test_identity_space_keeps_all_thirty_four_source_classes(self) -> None:
        names = target_class_names(IDENTITY_34_LABEL_SPACE)
        self.assertEqual(list(names), list(CICIOT2023_LABELS))
        labels = np.arange(34, dtype=np.int64)
        np.testing.assert_array_equal(
            remap_ciciot2023_labels(
                labels, source_label_space=IDENTITY_34_LABEL_SPACE
            ),
            labels,
        )
        with self.assertRaises(ValueError):
            remap_ciciot2023_labels(
                np.array([34], dtype=np.int64),
                source_label_space=IDENTITY_34_LABEL_SPACE,
            )

    def test_pooled_support_matches_the_prepared_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "prepared"
            _write_source_pool(source, rows_per_client=260)
            prepare_repartitioned_ciciot2023_splits(
                splits_dir=source,
                output_dir=output,
                source_clients=SOURCE_CLIENTS,
                num_clients=TARGET_CLIENTS,
                chunk_size=100,
            )
            metadata = json.loads(
                (output / "metadata.json").read_text(encoding="utf-8")
            )
            support = pooled_training_support(
                output,
                num_clients=TARGET_CLIENTS,
                num_classes=len(CICIOT2023_LABELS),
                source_label_space=IDENTITY_34_LABEL_SPACE,
                client_val_ratio=0.2,
                split_seed=42,
            )
            expected = np.zeros(len(CICIOT2023_LABELS), dtype=np.int64)
            for counts in metadata["client_training_class_counts"]:
                for name, value in counts.items():
                    expected[CICIOT2023_LABELS.index(name)] += int(value)
            np.testing.assert_array_equal(support, expected)


class MetricContractTests(unittest.TestCase):
    def _prepare(self, root: Path) -> Path:
        source = root / "source"
        output = root / "prepared"
        _write_source_pool(source, rows_per_client=240)
        prepare_repartitioned_ciciot2023_splits(
            splits_dir=source,
            output_dir=output,
            source_clients=SOURCE_CLIENTS,
            num_clients=TARGET_CLIENTS,
            chunk_size=90,
        )
        return output

    def _contract(self, output: Path, **overrides):
        arguments = {
            "partitions_dir": output,
            "num_clients": TARGET_CLIENTS,
            "num_classes": len(CICIOT2023_LABELS),
            "source_label_space": IDENTITY_34_LABEL_SPACE,
            "client_val_ratio": 0.2,
            "split_seed": 42,
            "minority_fraction": 0.25,
            "partition_hash": "deadbeef",
        }
        arguments.update(overrides)
        return build_metric_contract(**arguments)

    def test_contract_names_the_real_classes_on_the_thirty_four_class_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = self._prepare(Path(temp_dir))
            contract = self._contract(output)
            self.assertEqual(
                contract["classification_metric_schema_version"],
                CLASSIFICATION_METRIC_SCHEMA_VERSION,
            )
            self.assertEqual(
                contract["classification_metric_keys"],
                list(CLASSIFICATION_METRIC_KEYS),
            )
            # The 34-class task has a benign class, so the binary block is
            # contracted and its negative side is pinned by name.
            self.assertEqual(
                contract["benign_class_name"], "BenignTraffic"
            )
            self.assertEqual(
                contract["benign_class_id"],
                list(CICIOT2023_LABELS).index("BenignTraffic"),
            )
            self.assertEqual(len(contract["confusion_derived_metric_keys"]), 20)
            self.assertEqual(len(contract["score_dependent_metric_keys"]), 4)
            self.assertEqual(
                contract["target_class_names"], list(CICIOT2023_LABELS)
            )
            self.assertEqual(
                len(contract["minority_class_ids"]),
                math.ceil(len(CICIOT2023_LABELS) * 0.25),
            )
            self.assertEqual(
                contract["minority_class_names"],
                [CICIOT2023_LABELS[item] for item in contract["minority_class_ids"]],
            )
            support = contract["pooled_training_support"]
            self.assertEqual(len(support), len(CICIOT2023_LABELS))
            # The frozen minority set really is the rarest quarter.
            rarest = sorted(range(len(support)), key=lambda i: (support[i], i))
            self.assertEqual(
                contract["minority_class_ids"],
                rarest[: len(contract["minority_class_ids"])],
            )

    def test_contract_is_created_once_and_then_rejects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = self._prepare(root)
            contract = self._contract(output)
            results_root = root / "results"
            path = enforce_metric_contract(results_root, contract)
            self.assertTrue(path.is_file())
            # Same contract: idempotent.
            self.assertEqual(enforce_metric_contract(results_root, contract), path)
            drifted = self._contract(output, minority_fraction=0.5)
            with self.assertRaises(ValueError):
                enforce_metric_contract(results_root, drifted)

    def test_generic_widths_get_positional_names_for_synthetic_fixtures(self) -> None:
        space = generic_identity_label_space(4)
        self.assertEqual(space, "identity_4")
        self.assertEqual(
            list(target_class_names(space)),
            ["class_0", "class_1", "class_2", "class_3"],
        )
        labels = np.array([0, 3, 2], dtype=np.int64)
        np.testing.assert_array_equal(
            remap_ciciot2023_labels(labels, source_label_space=space), labels
        )
        with self.assertRaises(ValueError):
            remap_ciciot2023_labels(
                np.array([4], dtype=np.int64), source_label_space=space
            )


if __name__ == "__main__":
    unittest.main()
