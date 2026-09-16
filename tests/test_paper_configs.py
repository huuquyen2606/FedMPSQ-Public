"""Checks for the exact FedMPSQ configurations reported in the paper."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml

from fl.fedmpsq_config import load_fedmpsq_config


ROOT = Path(__file__).resolve().parents[1]
REPORTED = ROOT / "configs" / "reported"


class PaperConfigTests(unittest.TestCase):
    def test_reported_configs_match_committed_snapshots_exactly(self) -> None:
        snapshots = [
            ROOT
            / "results"
            / "paper"
            / f"{clients}_clients"
            / "fedmpsq"
            / "config.json"
            for clients in (10, 100)
        ]
        if not all(path.is_file() for path in snapshots):
            self.skipTest("Paper result snapshots are not included in this checkout")
        for clients in (10, 100):
            with self.subTest(clients=clients):
                reported = yaml.safe_load(
                    (REPORTED / f"fedmpsq_{clients}clients_seed42.yaml").read_text(
                        encoding="utf-8"
                    )
                )
                snapshot = json.loads(
                    (
                        ROOT
                        / "results"
                        / "paper"
                        / f"{clients}_clients"
                        / "fedmpsq"
                        / "config.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(reported, snapshot)

    def test_reported_configs_cover_both_client_scales(self) -> None:
        ten = load_fedmpsq_config(REPORTED / "fedmpsq_10clients_seed42.yaml")
        hundred = load_fedmpsq_config(REPORTED / "fedmpsq_100clients_seed42.yaml")

        self.assertEqual((ten.data.num_clients, hundred.data.num_clients), (10, 100))
        self.assertEqual((ten.algorithm.num_clients, hundred.algorithm.num_clients), (10, 100))
        self.assertEqual((ten.runtime.seed, hundred.runtime.seed), (42, 42))
        self.assertEqual((ten.data.split_seed, hundred.data.split_seed), (42, 42))
        self.assertEqual((ten.data.target_num_classes, hundred.data.target_num_classes), (34, 34))
        self.assertEqual((ten.algorithm.num_server_rounds, hundred.algorithm.num_server_rounds), (20, 20))

    def test_reported_configs_lock_the_paper_codec(self) -> None:
        settings = [
            load_fedmpsq_config(REPORTED / "fedmpsq_10clients_seed42.yaml"),
            load_fedmpsq_config(REPORTED / "fedmpsq_100clients_seed42.yaml"),
        ]
        for config in settings:
            with self.subTest(num_clients=config.data.num_clients):
                self.assertEqual(config.method.loss, "bounded_cb_lc")
                self.assertEqual(config.method.quant_bits, 2)
                self.assertAlmostEqual(config.method.sparsity, 0.9)
                self.assertTrue(config.method.error_feedback)
                self.assertTrue(config.method.incoherent_rotation)
                self.assertEqual(config.method.quantizer, "gaussian4")
                self.assertEqual(config.method.index_codec, "auto_runs")

        self.assertEqual(settings[0].method.uplink_budget_bytes, 3334)
        self.assertEqual(settings[1].method.uplink_budget_bytes, 3044)


if __name__ == "__main__":
    unittest.main()
