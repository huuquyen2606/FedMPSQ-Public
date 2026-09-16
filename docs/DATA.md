# Data protocol

The repository does not redistribute CICIoT2023 samples. The prepared data and reproduction checkpoints are available separately in [Google Drive](https://drive.google.com/drive/folders/10Yf5bLRPGMUSmt-7EfINqieBaItCSCEn?usp=drive_link). To rebuild the partitions from source, obtain CICIoT2023 from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/iotdataset-2023.html), then prepare the NPZ partitions with the scripts in `scripts/`.

The paper evaluates two frozen 34-class partitions:

| Scale | Construction | Manifest | Partition hash |
|---|---|---|---|
| 10 clients | Existing presplit clients | `configs/partitions/10_clients_manifest.json` | `a2a049e3f7a2a443ba137e2b923f62b6b9f83cf7d8a66e0ad9f970fbfcb20e62` |
| 100 clients | Red-packet non-IID repartition of the existing training pool, lognormal sigma 1.25, seed 42 | `configs/partitions/100_clients_manifest.json` | `ca438246c5c92896bad60c10c60a0547c5ac89c785ebd1fc1b3dd39ded7bc07d` |

The two partitions were produced by different partition procedures. Their global-test labels match, while their normalized feature arrays differ because preprocessing was refit. Results across the two scales therefore describe two federation settings; they are not a controlled estimate of the causal effect of client count.

## Expected prepared layout

For 10 clients:

```text
prepared-10/
├── metadata.json
├── global_test.npz
├── client_000.npz
├── ...
└── client_009.npz
```

For 100 clients, the client files run from `client_000.npz` through `client_099.npz`. Every NPZ contains arrays named `x` and `y`.

The manifests contain metadata and SHA-256 hashes only. They contain no traffic samples. Training fails closed when a prepared file differs from its manifest.
