# Paper sources

This directory contains the LaTeX source and publication assets for the
FedMPSQ paper.

## Contents

- `main.tex` — paper source.
- `IEEEtran.cls` — IEEE conference class used by the manuscript.
- `references.bib` — bibliography database.
- `figures/` — figures included by `main.tex`.
- `tables/` — source tables and supplementary table fragments.
- `result_mapping.md` — mapping from paper claims to repository artifacts.

## Build locally

Run the build from this directory so the relative figure and bibliography
paths resolve correctly:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The generated `main.pdf` is a local build artifact and is not required for
running the code. The repository keeps the source and assets so the manuscript
can be rebuilt without including datasets, checkpoints, or private logs.
