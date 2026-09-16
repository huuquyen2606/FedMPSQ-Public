from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data.partition_manifest import (
    create_partition_manifest,
    deterministic_split_indices,
    verify_partition_manifest,
)
from data.integrity import sha256_array, sha256_file, sha256_indices
from data.client_statistics import write_client_distribution_statistics
from extensions.distillation import bidirectional_decoupled_losses
from extensions.pruning import make_fap_loader, train_fap_model
from extensions.quantization import (
    DAdaQuantController,
    decode_quantized_payload,
    quantize_state_update,
    serialize_dense_state,
)
from fl.config import load_settings
from scripts.train_local_federated import (
    _aggregate_quantized_updates,
    _weighted_average_state_dicts,
)
from training.losses import fedprox_objective
from training.trainer import train_local_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATHS = [
    PROJECT_ROOT / "configs" / "paper34_v1" / "fedavg.yaml",
    PROJECT_ROOT / "configs" / "paper34_v1" / "fedprox.yaml",
    PROJECT_ROOT / "configs" / "paper34_v1" / "bdd_hfl.yaml",
    PROJECT_ROOT / "configs" / "paper34_v1" / "bdd_hfl_mu.yaml",
    PROJECT_ROOT / "configs" / "paper34_v1" / "fap.yaml",
    PROJECT_ROOT / "configs" / "paper34_v1" / "fap_mu.yaml",
    PROJECT_ROOT / "configs" / "paper34_v1" / "fedpaq.yaml",
    PROJECT_ROOT / "configs" / "paper34_v1" / "dadaquant.yaml",
]


def _write_partitions(root: Path) -> None:
    train_hashes = []
    validation_hashes = []
    client_training_labels = []
    for client_id in range(10):
        x = np.arange(48, dtype=np.float32).reshape(8, 6) + client_id * 1000
        y = (np.arange(8, dtype=np.int64) + client_id) % 4
        np.savez_compressed(root / f"client_{client_id:03d}.npz", x=x, y=y)
        train_indices, validation_indices = deterministic_split_indices(
            8,
            0.2,
            42 + client_id,
        )
        train_hashes.append(sha256_indices(train_indices))
        validation_hashes.append(sha256_indices(validation_indices))
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
        },
        "global_test_provenance": {
            "status": "complete",
            "role": "evaluation_only",
            "immutable": True,
            "source_kind": "synthetic_test_fixture",
            "prepared_npz_sha256": sha256_file(global_test_path),
            "prepared_x_sha256": sha256_array(x_test),
            "prepared_y_sha256": sha256_array(y_test),
            "preprocessing_fit_includes_global_test": False,
        },
        "client_statistics": statistics_artifacts,
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


class ConfigContractTests(unittest.TestCase):
    def test_all_configs_share_the_fixed_data_contract(self) -> None:
        contracts = []
        for path in CONFIG_PATHS:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            data = payload["data"]
            contracts.append(
                (
                    data["num_clients"],
                    data["dataset"],
                    data["partition_file"],
                    data["regenerate_partition"],
                    data["partitions_dir"],
                    data["client_val_ratio"],
                    data["seed"],
                )
            )
        self.assertEqual(len(set(contracts)), 1)
        self.assertEqual(contracts[0][0], 10)
        self.assertEqual(contracts[0][1], "existing_dataset")
        self.assertEqual(contracts[0][2], "PLACEHOLDER_PARTITION_FILE")
        self.assertEqual(contracts[0][4], "PLACEHOLDER_PARTITIONS_DIR")
        self.assertFalse(contracts[0][3])

    def test_each_config_enables_only_its_named_technique(self) -> None:
        expected = ["none", "none", "distillation", "distillation", "pruning", "pruning", "quantization", "quantization"]
        for path, expected_name in zip(CONFIG_PATHS, expected, strict=True):
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            technique = payload["technique"]
            self.assertEqual(technique["name"], expected_name)
            enabled = [
                bool(technique.get(name, {}).get("enabled", False))
                for name in ("distillation", "pruning", "quantization")
            ]
            self.assertLessEqual(sum(enabled), 1)

    def test_all_configs_share_protocol_model_and_seed(self) -> None:
        payloads = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in CONFIG_PATHS]
        protocol_keys = [
            "num_server_rounds",
            "fraction_train",
            "fraction_evaluate",
            "min_train_nodes",
            "min_evaluate_nodes",
            "min_available_nodes",
            "local_epochs",
            "batch_size",
            "learning_rate",
            "optimizer",
            "momentum",
            "weight_decay",
        ]
        protocol = [tuple(payload["algorithm"][key] for key in protocol_keys) for payload in payloads]
        self.assertEqual(len(set(protocol)), 1)
        models = [json.dumps(payload["model"], sort_keys=True) for payload in payloads]
        self.assertEqual(len(set(models)), 1)
        seeds = [(payload["data"]["seed"], payload["runtime"]["seed"]) for payload in payloads]
        self.assertEqual(len(set(seeds)), 1)
        self.assertEqual(seeds[0], (42, 42))
        self.assertTrue(all(payload["algorithm"]["fraction_train"] == 1.0 for payload in payloads))

    def test_all_eight_configs_load_as_paper_guided_integrations(self) -> None:
        for path in CONFIG_PATHS:
            settings = load_settings({"config-path": str(path)})
            self.assertEqual(settings.data.num_clients, 10)
            self.assertEqual(settings.fidelity.classification, "B")

    def test_technique_tuning_matches_the_paper_configs(self) -> None:
        distillation_settings = [
            load_settings({"config-path": str(CONFIG_PATHS[index])})
            for index in (2, 3)
        ]
        for settings in distillation_settings:
            self.assertEqual(settings.technique.distillation.private_kd_weight, 0.5)
            self.assertEqual(settings.technique.distillation.global_kd_weight, 0.5)

        pruning_settings = [
            load_settings({"config-path": str(CONFIG_PATHS[index])})
            for index in (4, 5)
        ]
        for settings in pruning_settings:
            self.assertEqual(settings.technique.pruning.target_samples, 500)
            self.assertEqual(settings.technique.pruning.target_fraction, 0.5)
            self.assertEqual(settings.technique.pruning.minimum_target_fraction, 1.0)
            self.assertEqual(settings.technique.pruning.warmup_rounds, 2)
            self.assertEqual(
                settings.technique.pruning.max_pruning_fraction_per_round,
                0.1,
            )


class PartitionContractTests(unittest.TestCase):
    def test_manifest_is_reused_by_all_ready_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "partitions"
            root.mkdir()
            _write_partitions(root)
            manifest_path = Path(tmp) / "client_partition.json"
            manifest = create_partition_manifest(
                root,
                manifest_path,
                dataset="existing_dataset",
                num_clients=10,
                client_val_ratio=0.2,
                seed=42,
            )
            hashes = set()
            for path in CONFIG_PATHS:
                settings = load_settings(
                    {
                        "config-path": str(path),
                        "data.partitions-dir": str(root),
                        "data.partition-file": str(manifest_path),
                    }
                )
                hashes.add(verify_partition_manifest(settings))
            self.assertEqual(hashes, {manifest["partition_hash"]})
            self.assertEqual([entry["client_id"] for entry in manifest["clients"]], list(range(10)))
            self.assertEqual(sum(entry["num_examples"] for entry in manifest["clients"]), 80)

    def test_manifest_detects_moved_or_changed_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "partitions"
            root.mkdir()
            _write_partitions(root)
            manifest_path = Path(tmp) / "client_partition.json"
            create_partition_manifest(
                root,
                manifest_path,
                dataset="existing_dataset",
                num_clients=10,
                client_val_ratio=0.2,
                seed=42,
            )
            settings = load_settings(
                {
                    "config-path": str(CONFIG_PATHS[0]),
                    "data.partitions-dir": str(root),
                    "data.partition-file": str(manifest_path),
                }
            )
            with np.load(root / "client_000.npz") as payload:
                x = payload["x"].copy()
                y = payload["y"].copy()
            x[0, 0] += 1.0
            np.savez_compressed(root / "client_000.npz", x=x, y=y)
            with self.assertRaisesRegex(ValueError, "Partition integrity"):
                verify_partition_manifest(settings)

    def test_split_indices_are_deterministic(self) -> None:
        train_a, val_a = deterministic_split_indices(101, 0.2, 42)
        train_b, val_b = deterministic_split_indices(101, 0.2, 42)
        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertEqual(set(train_a).intersection(val_a), set())
        self.assertEqual(set(train_a).union(val_a), set(range(101)))


class BaselineMathTests(unittest.TestCase):
    def test_fedavg_uses_sample_count_weights(self) -> None:
        states = [
            ({"weight": torch.tensor([1.0]), "counter": torch.tensor(3)}, 1),
            ({"weight": torch.tensor([5.0]), "counter": torch.tensor(7)}, 3),
        ]
        averaged = _weighted_average_state_dicts(states)
        self.assertTrue(torch.equal(averaged["weight"], torch.tensor([4.0])))
        self.assertEqual(averaged["counter"].item(), 3)

    def test_fedprox_formula_and_mu_zero(self) -> None:
        model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[2.0, -1.0]]))
        global_parameters = [torch.tensor([[1.0, 1.0]])]
        ce = torch.tensor(3.0)
        loss, proximal = fedprox_objective(ce, model, global_parameters, mu=0.2)
        self.assertAlmostEqual(proximal.item(), 5.0)
        self.assertAlmostEqual(loss.item(), 3.5)
        zero_loss, _ = fedprox_objective(ce, model, global_parameters, mu=0.0)
        self.assertEqual(zero_loss.item(), ce.item())

    def test_fedprox_mu_zero_trains_exactly_like_fedavg(self) -> None:
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]])
        targets = torch.tensor([0, 1, 1, 0])
        loader = DataLoader(TensorDataset(features, targets), batch_size=2, shuffle=False)
        fedavg = SimpleNamespace(
            name="fedavg", optimizer="sgd", learning_rate=0.1, momentum=0.0,
            weight_decay=0.0, proximal_mu=0.0, local_epochs=1,
        )
        fedprox = SimpleNamespace(**{**fedavg.__dict__, "name": "fedprox"})
        model_avg = nn.Linear(2, 2)
        model_prox = nn.Linear(2, 2)
        model_prox.load_state_dict(model_avg.state_dict())
        train_local_model(model_avg, loader, fedavg, torch.device("cpu"))
        train_local_model(model_prox, loader, fedprox, torch.device("cpu"))
        for avg_param, prox_param in zip(model_avg.parameters(), model_prox.parameters(), strict=True):
            self.assertTrue(torch.equal(avg_param, prox_param))


class TechniqueMathTests(unittest.TestCase):
    def test_bdd_hfl_decoupled_loss_is_zero_for_identical_logits(self) -> None:
        logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.2, -0.4]])
        targets = torch.tensor([0, 1])
        config = SimpleNamespace(
            temperature=4.0,
            target_class_weight=1.0,
            non_target_class_weight=8.0,
        )
        losses = bidirectional_decoupled_losses(logits, logits.clone(), targets, config)
        self.assertAlmostEqual(losses.global_kd.item(), 0.0, places=5)
        self.assertAlmostEqual(losses.private_kd.item(), 0.0, places=5)

    def test_qsgd_range_scale_and_self_describing_payload(self) -> None:
        global_state = {
            "weight": torch.zeros(8, dtype=torch.float32),
            "counter": torch.tensor(1, dtype=torch.int64),
        }
        local_state = {
            "weight": torch.tensor([0.0, 0.0, 1.0, -2.0, 0.0, 3.0, 0.0, 0.0]),
            "counter": torch.tensor(2, dtype=torch.int64),
        }
        update = quantize_state_update(
            local_state,
            global_state,
            levels=4,
            seed=7,
            lossless_qsgd_encoding=True,
            client_id=3,
            server_round=5,
            num_examples=123,
        )
        values = update.state["weight"]
        self.assertLessEqual(float(values.abs().max()), update.norm + 1e-6)
        scaled = values / update.scale
        self.assertTrue(torch.allclose(scaled, scaled.round(), atol=1e-5))
        self.assertEqual(update.state["counter"].item(), 2)
        self.assertEqual(update.payload_bytes, len(update.payload))
        self.assertEqual(update.payload_bytes, update.byte_counts.total_bytes)
        self.assertGreater(update.byte_counts.header_bytes, 0)
        self.assertEqual(update.byte_counts.norm_bytes, 4)
        self.assertGreater(update.byte_counts.layout_bytes, 0)
        self.assertGreater(update.byte_counts.quantized_value_bytes, 0)
        self.assertEqual(update.byte_counts.index_bytes, 0)
        self.assertEqual(update.byte_counts.auxiliary_value_bytes, 8)
        components = update.byte_counts.as_dict()
        total = components.pop("total_bytes")
        self.assertEqual(sum(components.values()), total)
        self.assertEqual(total, len(update.payload))

        decoded = decode_quantized_payload(update.payload)
        self.assertEqual(decoded.levels, 4)
        self.assertEqual(decoded.num_values, 8)
        self.assertEqual(decoded.client_id, 3)
        self.assertEqual(decoded.server_round, 5)
        self.assertEqual(decoded.num_examples, 123)
        self.assertEqual(decoded.encoding, "zero_rle_elias_omega")
        self.assertEqual(decoded.auxiliary_state["counter"].item(), 2)
        np.testing.assert_array_equal(decoded.values, values.numpy())
        with self.assertRaisesRegex(ValueError, "length"):
            decode_quantized_payload(update.payload[:-1])

    def test_fixed_width_qsgd_header_carries_bit_width_and_levels(self) -> None:
        global_state = {"weight": torch.zeros(16, dtype=torch.float32)}
        local_state = {"weight": torch.linspace(-2.0, 2.0, 16)}
        update = quantize_state_update(
            local_state,
            global_state,
            levels=10,
            seed=17,
            lossless_qsgd_encoding=False,
            client_id=1,
            server_round=2,
            num_examples=50,
        )
        decoded = decode_quantized_payload(update.payload)
        self.assertEqual(decoded.encoding, "fixed_width")
        self.assertEqual(decoded.levels, 10)
        self.assertEqual(decoded.magnitude_width, 4)
        self.assertEqual(decoded.byte_counts.index_bytes, 0)
        np.testing.assert_array_equal(decoded.values, update.state["weight"].numpy())

    def test_qsgd_transmits_batch_norm_buffers_without_quantizing_them(self) -> None:
        global_state = {
            "weight": torch.zeros(4, dtype=torch.float32),
            "running_mean": torch.zeros(2, dtype=torch.float32),
            "running_var": torch.ones(2, dtype=torch.float32),
            "num_batches_tracked": torch.tensor(0, dtype=torch.int64),
        }
        local_state = {
            "weight": torch.ones(4, dtype=torch.float32),
            "running_mean": torch.tensor([2.0, 3.0]),
            "running_var": torch.tensor([0.25, 0.5]),
            "num_batches_tracked": torch.tensor(7, dtype=torch.int64),
        }
        update = quantize_state_update(
            local_state,
            global_state,
            levels=1,
            seed=7,
            lossless_qsgd_encoding=True,
            quantized_names={"weight"},
        )
        raw_names = {spec.name for spec in update.raw_tensor_layout}
        self.assertEqual(
            raw_names,
            {"running_mean", "running_var", "num_batches_tracked"},
        )
        self.assertTrue(
            torch.equal(update.state["running_mean"], local_state["running_mean"])
        )
        self.assertTrue(
            torch.equal(update.state["running_var"], local_state["running_var"])
        )
        self.assertGreater(float(update.state["running_var"].min()), 0.0)

    def test_qsgd_server_averages_raw_batch_norm_buffers_as_values(self) -> None:
        global_state = {
            "weight": torch.zeros(2, dtype=torch.float32),
            "running_var": torch.ones(2, dtype=torch.float32),
        }
        first = quantize_state_update(
            {"weight": torch.ones(2), "running_var": torch.tensor([0.25, 0.5])},
            global_state,
            levels=8,
            seed=1,
            lossless_qsgd_encoding=True,
            quantized_names={"weight"},
        )
        second = quantize_state_update(
            {"weight": torch.ones(2), "running_var": torch.tensor([0.75, 1.0])},
            global_state,
            levels=8,
            seed=2,
            lossless_qsgd_encoding=True,
            quantized_names={"weight"},
        )
        aggregated = _aggregate_quantized_updates(
            global_state,
            [(first, 1), (second, 3)],
        )
        torch.testing.assert_close(
            aggregated["running_var"],
            torch.tensor([0.625, 0.875]),
        )
        self.assertGreater(float(aggregated["running_var"].min()), 0.0)

    def test_dense_baseline_payload_serializes_layout_values_and_metadata(self) -> None:
        state = {
            "weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
            "counter": torch.tensor(9, dtype=torch.int64),
        }
        serialized = serialize_dense_state(
            state,
            client_id=4,
            server_round=6,
            num_examples=321,
        )
        self.assertEqual(serialized.payload_bytes, len(serialized.payload))
        self.assertEqual(serialized.payload_bytes, serialized.byte_counts.total_bytes)
        self.assertGreater(serialized.byte_counts.header_bytes, 0)
        self.assertGreater(serialized.byte_counts.layout_bytes, 0)
        self.assertEqual(serialized.byte_counts.norm_bytes, 0)
        self.assertEqual(serialized.byte_counts.index_bytes, 0)
        self.assertEqual(serialized.byte_counts.value_bytes, 8 * 4 + 8)
        self.assertEqual(serialized.client_id, 4)
        self.assertEqual(serialized.server_round, 6)
        self.assertEqual(serialized.num_examples, 321)
        for name in state:
            self.assertTrue(torch.equal(serialized.state[name], state[name]))

    def test_dadaquant_assigns_more_levels_to_larger_clients(self) -> None:
        levels = DAdaQuantController.client_levels(16, {0: 10, 1: 40, 2: 100})
        self.assertLess(levels[0], levels[1])
        self.assertLess(levels[1], levels[2])
        self.assertGreaterEqual(min(levels.values()), 1)

    def test_dadaquant_time_adaptation_uses_paper_block_length(self) -> None:
        controller = DAdaQuantController(
            SimpleNamespace(
                min_level=1,
                max_level=8,
                moving_average=0.0,
                convergence_interval=2,
            )
        )
        controller.observe_weighted_loss(1.0)
        self.assertEqual(controller.level_for_next_round(), 1)
        controller.observe_weighted_loss(1.1)
        self.assertEqual(controller.level_for_next_round(), 2)
        controller.observe_weighted_loss(1.2)
        self.assertEqual(controller.level_for_next_round(), 2)
        controller.observe_weighted_loss(1.3)
        self.assertEqual(controller.level_for_next_round(), 4)

    def test_fap_prunes_active_indices_without_mutating_source_data(self) -> None:
        features = torch.tensor(
            [[2.0, 0.0], [0.0, 2.0], [1.5, 0.5], [0.5, 1.5]],
            dtype=torch.float32,
        )
        targets = torch.tensor([0, 1, 0, 1])
        source = TensorDataset(features.clone(), targets.clone())
        loader = make_fap_loader(source, [0, 1, 2, 3], batch_size=2, shuffle_seed=42)
        model = nn.Linear(2, 3)
        with torch.no_grad():
            model.weight.copy_(
                torch.tensor([[4.0, 0.0], [0.0, 4.0], [-4.0, -4.0]])
            )
            model.bias.zero_()
        algorithm = SimpleNamespace(
            name="fedavg",
            optimizer="sgd",
            learning_rate=0.01,
            momentum=0.0,
            weight_decay=0.0,
            proximal_mu=0.0,
            local_epochs=1,
        )
        pruning = SimpleNamespace(
            confidence_threshold=0.0,
            fisher_threshold=0.0,
            target_samples=4,
            target_fraction=0.0,
            minimum_target_fraction=0.5,
            warmup_rounds=0,
            max_pruning_fraction_per_round=1.0,
            gradient_clip_norm=1.0,
            dp_epsilon=1e12,
        )
        result = train_fap_model(
            model,
            loader,
            [0, 1, 2, 3],
            algorithm,
            pruning,
            torch.device("cpu"),
            server_round=1,
            original_num_examples=4,
            noise_seed=7,
        )
        self.assertEqual(len(result.active_indices), 2)
        self.assertEqual(result.metrics["fap_pruned_examples"], 2.0)
        self.assertEqual(len(source), 4)
        self.assertTrue(torch.equal(source.tensors[0], features))
        self.assertTrue(torch.equal(source.tensors[1], targets))

    def test_fap_uses_relative_target_warmup_and_round_cap(self) -> None:
        features = torch.tensor(
            [[2.0, 0.0], [0.0, 2.0]] * 10,
            dtype=torch.float32,
        )
        targets = torch.tensor([0, 1] * 10)
        source = TensorDataset(features, targets)
        algorithm = SimpleNamespace(
            name="fedavg",
            optimizer="sgd",
            learning_rate=0.01,
            momentum=0.0,
            weight_decay=0.0,
            proximal_mu=0.0,
            local_epochs=1,
        )
        pruning = SimpleNamespace(
            confidence_threshold=0.0,
            fisher_threshold=0.0,
            target_samples=4,
            target_fraction=0.5,
            minimum_target_fraction=1.0,
            warmup_rounds=2,
            max_pruning_fraction_per_round=0.1,
            gradient_clip_norm=1.0,
            dp_epsilon=1e12,
        )

        def run_round(server_round: int):
            loader = make_fap_loader(
                source,
                range(len(source)),
                batch_size=4,
                shuffle_seed=42,
            )
            model = nn.Linear(2, 3)
            return train_fap_model(
                model,
                loader,
                range(len(source)),
                algorithm,
                pruning,
                torch.device("cpu"),
                server_round=server_round,
                original_num_examples=len(source),
                noise_seed=7,
            )

        warmup_result = run_round(2)
        pruning_result = run_round(3)

        self.assertEqual(len(warmup_result.active_indices), 20)
        self.assertEqual(warmup_result.metrics["fap_warmup_active"], 1.0)
        self.assertEqual(pruning_result.metrics["fap_effective_target_samples"], 10.0)
        self.assertEqual(pruning_result.metrics["fap_pruned_examples"], 2.0)
        self.assertEqual(len(pruning_result.active_indices), 18)


if __name__ == "__main__":
    unittest.main()
