# Paper-to-repository mapping

The manuscript uses the following reproducibility artifacts:

| Paper item | Repository location |
| --- | --- |
| Eight baseline run definitions | `configs/paper34_v1/` |
| Reported FedMPSQ configurations | `configs/reported/` |
| Frozen 10-client manifest | `configs/partitions/10_clients_manifest.json` |
| Frozen 100-client manifest | `configs/partitions/100_clients_manifest.json` |
| Round-level snapshots for all 18 runs | `results/paper/` |
| Snapshot integrity manifest | `results/paper/manifest.json` |
| Per-class minority recall values | `results/paper/minority_recall.csv` |
| Reproduction script | `scripts/reproduce_paper_artifacts.py` |
| Reproduced tables | `results/tables/` |
| Reproduced figures | `results/figures/` |
| Reproduced summaries | `results/summaries/` |

The snapshot directory names are the canonical scenario identifiers accepted
by the reproduction script: `fedavg`, `fedprox`, `bdd_hfl`, `bdd_hfl_mu`,
`fap`, `fap_mu`, `fedpaq`, `dadaquant`, and `fedmpsq`.

The repository does not include CICIoT2023 samples, learned parameters, or
full training logs. Use `docs/DATA.md` and `docs/REPRODUCIBILITY.md` for the
data preparation and experiment protocol.
