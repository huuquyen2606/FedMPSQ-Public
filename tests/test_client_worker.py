from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from fl.config import load_settings, seed_everything, select_client_devices
from models import build_model
from scripts.train_local_federated import (
    _create_client_executors,
    _train_round_clients,
)
from training.client_worker import (
    ClientBatchRequest,
    ClientTrainSpec,
    train_client_batch_local,
)


def _write_partitions(root: Path) -> None:
    for client_id in range(10):
        features = np.arange(48, dtype=np.float32).reshape(8, 6) + client_id
        targets = (np.arange(8, dtype=np.int64) + client_id) % 4
        np.savez_compressed(
            root / f"client_{client_id:03d}.npz",
            x=features,
            y=targets,
        )
    np.savez_compressed(
        root / "global_test.npz",
        x=np.arange(72, dtype=np.float32).reshape(12, 6),
        y=np.arange(12, dtype=np.int64) % 4,
    )
    (root / "metadata.json").write_text(
        json.dumps({"input_dim": 6, "num_classes": 4, "num_clients": 10}),
        encoding="utf-8",
    )


class ClientWorkerTests(unittest.TestCase):
    def test_spawned_workers_return_client_states(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            partitions_dir = Path(temp_dir)
            _write_partitions(partitions_dir)
            settings = load_settings(
                {
                    "data.partitions-dir": str(partitions_dir),
                    "algorithm.batch-size": 4,
                    "algorithm.local-epochs": 1,
                    "model.input-dim": 6,
                    "model.num-classes": 4,
                    "model.conv-channels": "4,4",
                    "model.lstm-hidden-size": 4,
                    "model.dropout": 0.2,
                }
            )
            seed_everything(settings.runtime.seed)
            model = build_model(settings.model, input_dim=6, num_classes=4)
            global_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            executors = _create_client_executors(
                [torch.device("cpu"), torch.device("cpu")]
            )
            try:
                results = _train_round_clients(
                    settings=settings,
                    server_round=1,
                    global_state=global_state,
                    input_dim=6,
                    num_classes=4,
                    train_client_ids=[0, 1],
                    private_states={},
                    fap_active_indices={},
                    fap_original_num_examples={},
                    device=torch.device("cpu"),
                    executors=executors,
                )
            finally:
                for executor in executors:
                    executor.shutdown(wait=True)

            self.assertEqual([result.client_id for result in results], [0, 1])
            self.assertTrue(all(result.num_examples == 6 for result in results))

    def test_client_results_do_not_depend_on_worker_batching(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            partitions_dir = Path(temp_dir)
            _write_partitions(partitions_dir)
            settings = load_settings(
                {
                    "data.partitions-dir": str(partitions_dir),
                    "algorithm.batch-size": 4,
                    "algorithm.local-epochs": 1,
                    "model.input-dim": 6,
                    "model.num-classes": 4,
                    "model.conv-channels": "4,4",
                    "model.lstm-hidden-size": 4,
                    "model.dropout": 0.2,
                }
            )
            seed_everything(settings.runtime.seed)
            model = build_model(settings.model, input_dim=6, num_classes=4)
            global_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            common = {
                "server_round": 1,
                "global_state": global_state,
                "settings": settings,
                "input_dim": 6,
                "num_classes": 4,
            }
            combined = train_client_batch_local(
                ClientBatchRequest(
                    **common,
                    clients=(ClientTrainSpec(0), ClientTrainSpec(1)),
                ),
                torch.device("cpu"),
            )
            separate = [
                train_client_batch_local(
                    ClientBatchRequest(
                        **common,
                        clients=(ClientTrainSpec(client_id),),
                    ),
                    torch.device("cpu"),
                )[0]
                for client_id in (0, 1)
            ]

            for combined_result, separate_result in zip(
                combined,
                separate,
                strict=True,
            ):
                self.assertEqual(combined_result.client_id, separate_result.client_id)
                self.assertEqual(combined_result.num_examples, separate_result.num_examples)
                for name in combined_result.local_state:
                    self.assertTrue(
                        torch.equal(
                            combined_result.local_state[name],
                            separate_result.local_state[name],
                        )
                    )

    def test_two_parallel_clients_resolve_to_two_cuda_devices(self) -> None:
        with patch("fl.config.torch.cuda.device_count", return_value=2):
            self.assertEqual(
                select_client_devices("auto", 2),
                [torch.device("cuda:0"), torch.device("cuda:1")],
            )

    def test_multi_gpu_rejects_cpu_device(self) -> None:
        with self.assertRaises(ValueError):
            select_client_devices("cpu", 2)


if __name__ == "__main__":
    unittest.main()
