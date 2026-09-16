# Upload manifest

This directory is the curated GitHub artifact for the final RIVF 2026 paper.

## Included

- Training and evaluation source for eight baseline runs and FedMPSQ.
- Eight baseline configs and the FedMPSQ uplink-v4 template.
- Exact reported FedMPSQ configurations for 10 and 100 clients.
- Frozen partition manifests for both scales; these contain metadata and hashes, not samples.
- Two clean, parameterized Kaggle notebooks.
- Round-level metrics and config snapshots for the 18 runs reported in the paper.
- Compact minority-recall values used by the paper figure.
- Regenerated tables, figures, and K=10/K=100 summaries.
- Core source-level and synthetic tests.
- Paper LaTeX source and publication assets, citation metadata, dependency metadata, and license.

## Excluded

- CICIoT2023 CSV, NPZ, NumPy, Parquet, or HDF5 data.
- Checkpoints, best/final models, private models, or tensor payloads.
- Client-history JSON/CSV, per-round validation dumps, provenance environment dumps, and logs.
- Screening campaigns, unpublished ablations, delivery builders, source bundles, and duplicate configs.
- Virtual environments, caches, editor settings, and raw experiment artifacts.

`SHA256SUMS.txt` records every versioned file except itself.
