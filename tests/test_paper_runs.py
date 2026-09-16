"""Checks for the nine canonical paper-run identifiers and labels."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.paper_runs import (
    BASELINE_RUNS,
    FEDMPSQ_RUN,
    PAPER_RUNS,
    canonical_run_name,
)
from scripts.run_paper34_campaign import DEFAULT_SEED, build_matrix
from scripts.run_uplink_v4 import RUN_SLUG, SCALES, run_name as uplink_run_name


ROOT = Path(__file__).resolve().parents[1]


class PaperRunNamingTests(unittest.TestCase):
    def test_run_identifiers_and_labels_are_canonical(self) -> None:
        self.assertEqual(
            [(run.slug, run.display_name) for run in PAPER_RUNS],
            [
                ("fedavg", "FedAvg"),
                ("fedprox", "FedProx"),
                ("bdd_hfl", "BDD-HFL"),
                ("bdd_hfl_mu", "BDD-HFL-mu"),
                ("fap", "FAP"),
                ("fap_mu", "FAP-mu"),
                ("fedpaq", "FedPAQ"),
                ("dadaquant", "DAdaQuant"),
                ("fedmpsq", "FedMPSQ"),
            ],
        )

    def test_all_run_configs_use_canonical_filenames(self) -> None:
        for run in (*BASELINE_RUNS, FEDMPSQ_RUN):
            with self.subTest(run=run.slug):
                self.assertEqual(run.config_filename, f"{run.slug}.yaml")
                self.assertTrue(
                    (ROOT / "configs" / "paper34_v1" / run.config_filename).is_file()
                )

    def test_all_eighteen_jobs_have_one_canonical_artifact_name(self) -> None:
        baseline_jobs = build_matrix(ROOT / "results" / "local")
        self.assertEqual(len(baseline_jobs), 16)
        expected_baseline_pairs = {
            (clients, run.slug)
            for clients in (10, 100)
            for run in BASELINE_RUNS
        }
        self.assertEqual(
            {(job["num_clients"], job["scenario"]) for job in baseline_jobs},
            expected_baseline_pairs,
        )
        baseline_names = {job["run_name"] for job in baseline_jobs}
        self.assertEqual(
            baseline_names,
            {
                canonical_run_name(run.slug, clients, DEFAULT_SEED)
                for clients, run in (
                    (clients, run)
                    for clients in (10, 100)
                    for run in BASELINE_RUNS
                )
            },
        )

        self.assertEqual(RUN_SLUG, FEDMPSQ_RUN.slug)
        proposed_names = {
            uplink_run_name(clients, DEFAULT_SEED) for clients in SCALES
        }
        self.assertEqual(
            proposed_names,
            {
                canonical_run_name(FEDMPSQ_RUN.slug, clients, DEFAULT_SEED)
                for clients in (10, 100)
            },
        )
        self.assertEqual(len(baseline_names | proposed_names), 18)


if __name__ == "__main__":
    unittest.main()
