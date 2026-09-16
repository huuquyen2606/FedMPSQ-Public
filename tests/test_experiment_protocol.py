from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fl.metrics_logger import RoundMetricsLogger
from scripts.train_local_federated import (
    PROJECT_ROOT,
    _settings_from_args,
    _validation_is_better,
)
from training.metrics import (
    classification_metrics_from_confusion,
    confusion_matrix_counts,
)


def _training_args(*, seed: int) -> argparse.Namespace:
    return argparse.Namespace(
        config_path=str(
            PROJECT_ROOT / "configs" / "paper34_v1" / "fedavg.yaml"
        ),
        partitions_dir=None,
        partition_file=None,
        algorithm=None,
        rounds=None,
        fraction_train=None,
        min_train_nodes=None,
        local_epochs=None,
        batch_size=None,
        learning_rate=None,
        optimizer=None,
        proximal_mu=None,
        results_dir=None,
        run_name=None,
        device=None,
        parallel_clients=None,
        seed=seed,
        launcher_command=None,
        save_model=None,
        save_round_checkpoints=None,
    )


def _global_test_record(sha256: str = "file-hash") -> dict[str, object]:
    return {
        "file": "global_test.npz",
        "sha256": sha256,
        "x_sha256": "x-hash",
        "y_sha256": "y-hash",
        "num_examples": 12,
        "x_shape": [12, 3],
        "y_shape": [12],
        "x_dtype": "float32",
        "y_dtype": "int64",
    }


class ExperimentProtocolTests(unittest.TestCase):
    def test_runtime_seeds_share_the_fixed_data_split_seed(self) -> None:
        for runtime_seed in (42, 43, 44):
            settings = _settings_from_args(_training_args(seed=runtime_seed))
            self.assertEqual(settings.runtime.seed, runtime_seed)
            self.assertEqual(settings.data.seed, 42)

    def test_best_checkpoint_maximizes_validation_macro_f1(self) -> None:
        self.assertTrue(_validation_is_better(0.4, 1, None, None))
        self.assertTrue(_validation_is_better(0.5, 2, 0.4, 1))
        self.assertFalse(_validation_is_better(0.3, 2, 0.4, 1))
        self.assertFalse(_validation_is_better(0.4, 2, 0.4, 1))
        self.assertTrue(_validation_is_better(0.4, 1, 0.4, 2))

    def test_union_macro_f1_comes_from_summed_confusion_counts(self) -> None:
        first = confusion_matrix_counts(
            np.zeros(10, dtype=np.int64),
            np.zeros(10, dtype=np.int64),
            num_classes=2,
        )
        second = confusion_matrix_counts(
            np.ones(10, dtype=np.int64),
            np.zeros(10, dtype=np.int64),
            num_classes=2,
        )
        client_mean = (
            classification_metrics_from_confusion(first)["macro_f1"]
            + classification_metrics_from_confusion(second)["macro_f1"]
        ) / 2.0
        union_macro_f1 = classification_metrics_from_confusion(first + second)[
            "macro_f1"
        ]
        self.assertAlmostEqual(client_mean, 0.25)
        self.assertAlmostEqual(union_macro_f1, 1.0 / 3.0)

    def test_classification_metrics_include_all_ten_requested_scalars(self) -> None:
        confusion = np.asarray(
            [[1, 1, 0], [0, 1, 1], [0, 0, 0]],
            dtype=np.int64,
        )
        metrics = classification_metrics_from_confusion(confusion)
        expected_keys = {
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "micro_precision",
            "micro_recall",
            "micro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
        }
        self.assertEqual(set(metrics), expected_keys)
        self.assertAlmostEqual(metrics["micro_precision"], 0.5)
        self.assertAlmostEqual(metrics["micro_recall"], 0.5)
        self.assertAlmostEqual(metrics["micro_f1"], 0.5)
        self.assertAlmostEqual(metrics["weighted_precision"], 0.75)
        self.assertAlmostEqual(metrics["weighted_recall"], 0.5)

    def test_shared_test_contract_rejects_a_different_test_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = RoundMetricsLogger(root / "run-a", "run-a")
            second = RoundMetricsLogger(root / "run-b", "run-b")
            first.enforce_fixed_test_contract(_global_test_record())
            second.enforce_fixed_test_contract(_global_test_record())
            with self.assertRaises(ValueError):
                second.enforce_fixed_test_contract(
                    _global_test_record(sha256="different-file-hash")
                )


if __name__ == "__main__":
    unittest.main()
