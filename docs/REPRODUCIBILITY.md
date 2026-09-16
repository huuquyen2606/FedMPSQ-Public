# Reproducibility guide

## Environment

The reported runs used Python 3.12, PyTorch with CUDA, 20 communication rounds, full client participation, one local epoch, batch size 256, and seed 42. Install the project with:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[paper]"
```

Run the unit tests before starting a full experiment:

```bash
python -m unittest discover -s tests -v
```

## Baseline runs

The baseline runner accepts both scales. Without `--execute`, it prints the resolved job and performs no training.

```bash
python scripts/run_paper34_campaign.py \
  --scenario fedavg \
  --num-clients 10 \
  --seed 42 \
  --partitions-dir /path/to/prepared-10 \
  --partition-file configs/partitions/10_clients_manifest.json \
  --output-root results/local
```

Append `--execute` after checking the plan. Replace the scenario with any of:

```text
fedavg
fedprox
bdd_hfl
bdd_hfl_mu
fap
fap_mu
fedpaq
dadaquant
```

For the 100-client setting, change `--num-clients`, `--partitions-dir`, and `--partition-file` to the 100-client partition.

## FedMPSQ

The reported arm uses the same method hyperparameters at both scales and changes only the serialized uplink budget: 3,334 bytes for 10 clients and 3,044 bytes for 100 clients.

Planning the reported 10-client arm:

```bash
python scripts/run_uplink_v4.py \
  --data-dir /path/to/prepared-10 \
  --manifest configs/partitions/10_clients_manifest.json \
  --output results/local/fedmpsq-10 \
  --clients 10 \
  --arm rht_g4_b3334 \
  --seed 42 \
  --seeds 42
```

For 100 clients, use the 100-client inputs, `--clients 100`, and `--arm rht_g4_b3044_c100`. Append `--run --evaluate-test` to reproduce the reported confirmation run after checking the generated plan.

The files under `configs/reported/` preserve the exact resolved configurations of the two reported jobs, including their original Kaggle paths. Use them for provenance; use the runners above to generate portable paths for a new machine.

## Recreate tables and figures

`results/paper/` contains only the recorded round-level metrics and configuration snapshots for the 18 reported runs plus the compact per-class recall values used by the minority figure. It contains no samples or weights.

The main utility table uses frozen client-validation metrics for every arm. Following the paper figure caption, `minority_recall.csv` uses global-test per-class recall for the baseline arms and client-validation per-class recall for FedMPSQ; its `evaluation_split` column records this distinction explicitly. BDD-HFL is retained in the CSV for provenance but omitted from the plotted panel because both variants round to zero.

```bash
python scripts/reproduce_paper_artifacts.py
```

Generated CSV tables, JSON summaries, and PDF/PNG figures are written under
`results/tables/`, `results/summaries/`, and `results/figures/`. These compact
outputs are safe to publish; raw logs, checkpoints, and datasets remain
excluded.

## Reviewer smoke test

Run all 18 method-by-client-count paths on deterministic synthetic data and apply the acceptance audit to each run artifact. This command does not require a raw dataset or GPU.

```bash
python scripts/smoke_test_paper34.py
```
