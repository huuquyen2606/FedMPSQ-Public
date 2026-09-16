"""Canonical identifiers and display names for the nine paper runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperRun:
    slug: str
    display_name: str
    config_filename: str
    family: str


BASELINE_RUNS = (
    PaperRun("fedavg", "FedAvg", "fedavg.yaml", "baseline"),
    PaperRun("fedprox", "FedProx", "fedprox.yaml", "baseline"),
    PaperRun("bdd_hfl", "BDD-HFL", "bdd_hfl.yaml", "baseline"),
    PaperRun("bdd_hfl_mu", "BDD-HFL-mu", "bdd_hfl_mu.yaml", "baseline"),
    PaperRun("fap", "FAP", "fap.yaml", "baseline"),
    PaperRun("fap_mu", "FAP-mu", "fap_mu.yaml", "baseline"),
    PaperRun("fedpaq", "FedPAQ", "fedpaq.yaml", "baseline"),
    PaperRun("dadaquant", "DAdaQuant", "dadaquant.yaml", "baseline"),
)
FEDMPSQ_RUN = PaperRun("fedmpsq", "FedMPSQ", "fedmpsq.yaml", "proposed")
PAPER_RUNS = (*BASELINE_RUNS, FEDMPSQ_RUN)
RUN_BY_SLUG = {run.slug: run for run in PAPER_RUNS}
PAPER_SCENARIOS = tuple(run.slug for run in PAPER_RUNS)


def scenario_family(scenario: str) -> str:
    try:
        return RUN_BY_SLUG[scenario].family
    except KeyError as exc:
        raise ValueError(f"Unsupported paper run: {scenario}") from exc


def canonical_run_name(scenario: str, num_clients: int, seed: int) -> str:
    """Return the artifact prefix used consistently across all paper runs."""
    if scenario not in RUN_BY_SLUG:
        raise ValueError(f"Unsupported paper run: {scenario}")
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return f"{scenario}_{num_clients}clients_seed{seed}"
