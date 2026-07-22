# Setup and Python Ecosystem

Module 0 is the operating manual for the book. It defines the environment, repository layout, execution policy, and validation checks that make later financial results reproducible {cite}`kluyver2016jupyter,jupyterbook2025`.

```{figure} ../img/generated/module0-reproducible-workflow.png
:alt: Module 0 workflow connecting locked environment, reusable code, notebook execution, cached build, and published book.
:width: 760px
```

## Prerequisites

Readers need a local terminal, Git, and permission to install Python tooling.
No prior Python or financial-modeling experience is assumed. Complete the
steps from the repository root; commands and path checks in this module use that
location as their anchor.

## Expected outcome

By the end of this module, readers should have a working local checkout that can:

- select Python 3.12 or newer with `pyenv`;
- synchronize the locked project environment with `uv`;
- open JupyterLab from the project environment;
- import the course helpers in `src/`;
- find the versioned real-data snapshots used by the published notebooks;
- run setup and environment checks without exposing credentials;
- build the Jupyter Book before sharing changes.

## Module map

Read the module in operational order:

| Step | Page | Role |
| --- | --- | --- |
| 1 | Reproducible Computational Finance Stack | Defines the technical contract: interpreter, dependencies, notebooks, credentials, data snapshots, outputs, and validation. |
| 2 | Initial Repository Setup | Confirms a fresh clone is opened from the project root and that the locked environment is active. |
| 3 | Environment Validation Lab | Runs the full executable check of Python, packages, paths, optional credentials, source helpers, and data snapshots. |
| 4 | Classroom Environment Setup | Gives a short pre-class smoke test before running data, dashboard, or time-series notebooks. |

## Reproducibility standard

A computed result in this book is usable only when the inputs that produced it can be reconstructed:

```{math}
O = F(D, C, E, \theta)
```

where \(D\) is the data, \(C\) is the code, \(E\) is the software environment, \(\theta\) is the parameter set, and \(O\) is the published output. Module 0 focuses on \(E\), the repository layout for \(C\), the snapshot files that make \(D\) reproducible, and the build process that makes \(O\) reviewable.

## Technical decisions

| Area | Course decision | Reason |
| --- | --- | --- |
| Python runtime | Python 3.12 or newer through `pyenv` {cite}`pyenv2025` | Keeps the interpreter explicit and modern. |
| Dependency manager | `uv` with `pyproject.toml` and `uv.lock` {cite}`uv2025` | Provides fast, deterministic environment setup. |
| Notebook interface | JupyterLab launched with `uv run` | Prevents accidental use of the wrong Python kernel. |
| Reusable code | Shared helpers live in `src/` | Keeps notebooks focused on analysis and presentation. |
| Data inputs | Versioned CSV snapshots plus optional live providers | Lets the book build without credentials while preserving real-data workflows. |
| Book build | `make book` with cached execution for reproducible notebooks | Publishes reader-facing outputs without saving generated notebook results in source files. |
| Data credentials | `.env` or shell variables, with `.env.example` as template | Keeps tokens out of Git history. |
| Visual assets | approved PNGs under `img/generated/`, indexed by `visual-assets.json` and per-module manifests | Keeps figures inspectable, reusable, and build-safe. |

## Base workflow

Run these commands from the repository root:

```bash
pyenv install 3.12.12
pyenv local 3.12.12
uv sync
uv run jupyter lab
```

The first two commands select the interpreter. `uv sync` creates or updates `.venv/` from the locked dependency graph. `uv run jupyter lab` opens notebooks from the same environment that the publication build uses.

## Validation workflow

For quick local review:

```bash
make book
```

Before sharing changes:

```bash
make publish-check
```

`make publish-check` verifies whitespace, visual assets, notebook source structure, the Jupyter Book build, `.nojekyll`, and local HTML links. If Jupyter cannot bind local kernel ports inside a restricted environment, rerun the build in a context where local Jupyter kernels can start before changing notebook content.

## Handoff

Module 0 is complete only when the fresh-clone report, full environment report,
and classroom smoke test all pass. After that gate, readers can move into
Module 1 with a clear boundary: notebooks import reusable helpers from `src/`,
read committed real-data snapshots for publication, and use live credentials
only in controlled local sessions. If a path or package check fails, resolve it
here before interpreting any financial output downstream.
