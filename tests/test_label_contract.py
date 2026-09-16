from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from data.ciciot2023 import CICIOT2023_LABELS
from data.fedmpsq_labels import FEDMPSQ_CLASS_NAMES
from data.label_contract import (
    AUDIT_JSON_FILENAME,
    GROUPED_COUNTS_FILENAME,
    MAPPING_CSV_FILENAME,
    MAPPING_JSON_FILENAME,
    audit_statistics_file,
    build_label_contract,
    explicit_mapping_rows,
    scan_raw_label_values,
    write_label_contract_artifacts,
)


def write_split_csv(path: Path, labels: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "label"])
        for index, label in enumerate(labels):
            writer.writerow([index, label])


def write_statistics_csv(path: Path, partitions: dict[str, list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", *CICIOT2023_LABELS, "TOTAL_SAMPLES"])
        for partition, counts in partitions.items():
            writer.writerow([partition, *counts, sum(counts)])


class LabelContractTests(unittest.TestCase):
    def test_contract_exposes_unambiguous_source_and_target_ids(self) -> None:
        contract = build_label_contract()
        rows = contract["mapping"]
        self.assertEqual(len(rows), 34)
        self.assertEqual(rows[0]["source_label_id"], 0)
        self.assertEqual(rows[0]["source_label"], "DDoS-RSTFINFlood")
        self.assertEqual(rows[0]["target_label_id"], 0)
        self.assertEqual(rows[0]["target_label"], "DDoS")
        self.assertEqual(rows[26]["source_label"], "BenignTraffic")
        self.assertEqual(rows[26]["target_label_id"], 5)
        self.assertEqual(rows[-1]["source_label_id"], 33)
        self.assertEqual(rows[-1]["target_label"], "BruteForce")
        self.assertEqual(
            contract["implementation_mapping_sha256"],
            "52e58da35a385d6a48787530bc167da28bef403daa7e0c52e2b8bf355f6b76b0",
        )

    def test_artifacts_validate_headers_statistics_and_totals(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            splits = root / "splits"
            output = root / "output"
            splits.mkdir()
            write_split_csv(splits / "client_0.csv", [CICIOT2023_LABELS[0]])
            write_split_csv(splits / "client_1.csv", [CICIOT2023_LABELS[26]])
            write_split_csv(
                splits / "global_test.csv",
                [CICIOT2023_LABELS[-1]],
            )
            counts_0 = [0] * 34
            counts_0[0] = 3
            counts_1 = [0] * 34
            counts_1[26] = 2
            counts_test = [0] * 34
            counts_test[-1] = 1
            statistics = splits / "ThongKe_NonIID_10clients.csv"
            write_statistics_csv(
                statistics,
                {
                    "Client_0": counts_0,
                    "Client_1": counts_1,
                    "Global_Test": counts_test,
                },
            )

            audit = write_label_contract_artifacts(
                splits_dir=splits,
                statistics_file=statistics,
                output_dir=output,
                expected_clients=2,
            )

            self.assertEqual(audit["status"], "passed")
            self.assertEqual(audit["split_header_audit"]["num_features"], 1)
            self.assertEqual(audit["statistics_audit"]["source"], "statistics_csv")
            self.assertTrue(audit["statistics_audit"]["grouped_total_preserved"])
            for filename in (
                MAPPING_CSV_FILENAME,
                MAPPING_JSON_FILENAME,
                GROUPED_COUNTS_FILENAME,
                AUDIT_JSON_FILENAME,
            ):
                self.assertTrue((output / filename).exists())

            with (output / GROUPED_COUNTS_FILENAME).open(
                encoding="utf-8",
                newline="",
            ) as handle:
                grouped = list(csv.DictReader(handle))
            self.assertEqual(grouped[0]["DDoS"], "3")
            self.assertEqual(grouped[1]["Benign"], "2")
            self.assertEqual(grouped[2]["BruteForce"], "1")
            mapping_payload = json.loads(
                (output / MAPPING_JSON_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(
                mapping_payload["target_label_space"]["labels"],
                list(FEDMPSQ_CLASS_NAMES),
            )

    def test_raw_scan_rejects_unknown_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_split_csv(root / "client_0.csv", ["Unknown_Attack"])
            write_split_csv(root / "global_test.csv", CICIOT2023_LABELS)
            with self.assertRaisesRegex(ValueError, "Unknown raw labels"):
                scan_raw_label_values(
                    root,
                    expected_clients=1,
                    chunk_size=5,
                )

    def test_statistics_rejects_unknown_source_label_column(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stats.csv"
            labels = [*CICIOT2023_LABELS[:-1], "Unknown_Attack"]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["", *labels, "TOTAL_SAMPLES"])
                writer.writerow(["Client_0", *([0] * 34), 0])
            with self.assertRaisesRegex(ValueError, "unknown=.*Unknown_Attack"):
                audit_statistics_file(path, explicit_mapping_rows())

    def test_full_scan_matches_source_statistics_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            splits = root / "splits"
            splits.mkdir()
            write_split_csv(splits / "client_0.csv", [CICIOT2023_LABELS[0]])
            write_split_csv(splits / "global_test.csv", CICIOT2023_LABELS[1:])
            client_counts = [0] * 34
            client_counts[0] = 1
            test_counts = [0] * 34
            for label_id in range(1, 34):
                test_counts[label_id] = 1
            statistics = splits / "ThongKe_NonIID_10clients.csv"
            write_statistics_csv(
                statistics,
                {
                    "Client_0": client_counts,
                    "Global_Test": test_counts,
                },
            )

            audit = write_label_contract_artifacts(
                splits_dir=splits,
                statistics_file=statistics,
                output_dir=root / "output",
                expected_clients=1,
                scan_raw_labels=True,
                chunk_size=1,
            )

            self.assertTrue(audit["raw_label_scan"]["performed"])
            self.assertEqual(
                audit["raw_statistics_consistency"]["rows_compared"],
                34,
            )

    def test_missing_statistics_can_derive_counts_from_full_raw_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_split_csv(root / "client_0.csv", [CICIOT2023_LABELS[0]])
            write_split_csv(root / "global_test.csv", CICIOT2023_LABELS[1:])

            audit = write_label_contract_artifacts(
                splits_dir=root,
                statistics_file=root / "ThongKe_NonIID_10clients.csv",
                output_dir=root / "output",
                expected_clients=1,
                scan_raw_labels=True,
                chunk_size=1,
                allow_missing_statistics=True,
            )

            self.assertEqual(audit["statistics_audit"]["source"], "raw_label_scan")
            self.assertIsNone(audit["statistics_audit"]["file"])
            self.assertEqual(audit["statistics_audit"]["row_count"], 2)
            self.assertFalse(audit["raw_statistics_consistency"]["performed"])
            self.assertIsNone(audit["raw_statistics_consistency"]["matched"])
            self.assertEqual(audit["raw_statistics_consistency"]["raw_rows_used"], 34)
            with (root / "output" / GROUPED_COUNTS_FILENAME).open(
                encoding="utf-8",
                newline="",
            ) as handle:
                grouped = list(csv.DictReader(handle))
            self.assertEqual(grouped[0]["DDoS"], "1")
            self.assertEqual(sum(int(row["total_samples"]) for row in grouped), 34)

    def test_missing_statistics_fallback_requires_full_raw_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_split_csv(root / "client_0.csv", [CICIOT2023_LABELS[0]])
            write_split_csv(root / "global_test.csv", CICIOT2023_LABELS[1:])
            with self.assertRaisesRegex(ValueError, "requires --scan-raw-labels"):
                write_label_contract_artifacts(
                    splits_dir=root,
                    statistics_file=root / "missing.csv",
                    output_dir=root / "output",
                    expected_clients=1,
                    allow_missing_statistics=True,
                )

    def test_full_scan_rejects_statistics_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_split_csv(root / "client_0.csv", [CICIOT2023_LABELS[0]])
            write_split_csv(root / "global_test.csv", CICIOT2023_LABELS[1:])
            client_counts = [0] * 34
            client_counts[0] = 2
            test_counts = [0] * 34
            for label_id in range(1, 34):
                test_counts[label_id] = 1
            statistics = root / "ThongKe_NonIID_10clients.csv"
            write_statistics_csv(
                statistics,
                {
                    "Client_0": client_counts,
                    "Global_Test": test_counts,
                },
            )

            with self.assertRaisesRegex(
                ValueError,
                "Raw label counts differ from statistics",
            ):
                write_label_contract_artifacts(
                    splits_dir=root,
                    statistics_file=statistics,
                    output_dir=root / "output",
                    expected_clients=1,
                    scan_raw_labels=True,
                    chunk_size=1,
                )


if __name__ == "__main__":
    unittest.main()
