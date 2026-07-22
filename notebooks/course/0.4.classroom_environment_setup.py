# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Classroom Environment Setup
#
# Module: Setup and Python Ecosystem
#
# ## Lesson summary
#
# This classroom setup notebook is the short pre-session smoke test. It assumes
# the first-clone setup in `0.3.initial_repository_setup` and the full validation
# in `0.2.environment_validation_lab` have already passed. Its job is to answer
# one practical question before a live class: is this kernel ready to run the
# next data notebook without downloading market data or exposing credentials
# {cite}`pyenv2025,uv2025,kluyver2016jupyter`.
#
# ## Learning objectives
#
# By the end of this note, readers should be able to:
#
# - confirm that the classroom environment has been synchronized from `uv.lock`;
# - verify data, finance, visualization, widget, and storage libraries such as `pandas`, `yfinance`, `requests`, `plotly`, `ipywidgets`, and `pyarrow`;
# - confirm that snapshot-backed lessons can find the local snapshot manifest;
# - create a concise environment record that can accompany reproducibility questions.
#
# ## Prerequisites
#
# The fresh-clone setup and full environment validation must already pass. Run
# `uv sync` after any lockfile change and launch this notebook from the project
# environment; no live provider credential is needed for the smoke test.
#
# ## Lesson flow
#
# 1. Open the repository from the project environment.
# 2. Run the compact readiness report.
# 3. Record package versions with `watermark`.
# 4. Move to data notebooks only after the check passes.

# %% [markdown]
# ## Setup
#
# Open the notebook from JupyterLab after running `uv sync`; the runtime cells should execute without downloading market data or printing credentials.

# %% [markdown]
# ## Terminal setup with pyenv and uv
#
# Run these commands from the repository root before opening JupyterLab. The project dependencies are declared in `pyproject.toml` and installed by `uv sync`.
#
# ```bash
# pyenv install 3.12.12
# pyenv local 3.12.12
# uv sync
# uv run jupyter lab
# ```

# %% [markdown]
# ## Session readiness check
#
# Use this check before running data notebooks that depend on external providers, interactive widgets, or columnar storage libraries. It is a smoke test, not a replacement for the full environment validation lab.

# %% tags=["setup", "hide-input"]
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

project_root = Path.cwd().resolve()
for candidate in (project_root, *project_root.parents):
    if (candidate / "pyproject.toml").exists() and (candidate / "uv.lock").exists():
        project_root = candidate
        break

classroom_packages = {
    "pandas": "pandas",
    "yfinance": "yfinance",
    "requests": "requests",
    "plotly": "plotly",
    "ipywidgets": "ipywidgets",
    "pyarrow": "pyarrow",
}

package_status = {}
for import_name, distribution_name in classroom_packages.items():
    try:
        package_status[import_name] = version(distribution_name)
    except PackageNotFoundError:
        package_status[import_name] = "missing"

readiness_records = [
    {"check": "python_3_12_or_newer", "ready": sys.version_info >= (3, 12), "detail": sys.version.split()[0]},
    {"check": "project_root", "ready": (project_root / "pyproject.toml").exists(), "detail": str(project_root)},
    {"check": "uv_lock", "ready": (project_root / "uv.lock").exists(), "detail": "uv.lock"},
    {
        "check": "published_notebooks",
        "ready": (project_root / "notebooks" / "course").exists(),
        "detail": "notebooks/course",
    },
    {"check": "snapshot_manifest", "ready": (project_root / "data" / "snapshots" / "metadata.json").exists(), "detail": "data/snapshots/metadata.json"},
]
readiness_records.extend(
    {"check": f"package:{name}", "ready": package_version != "missing", "detail": package_version}
    for name, package_version in package_status.items()
)

session_readiness = pd.DataFrame(readiness_records)
assert session_readiness["ready"].all(), session_readiness
session_readiness

# %% [markdown]
# **Output interpretation.**
#
# Every `ready` value should be `True`. This table is deliberately compact: it confirms that the kernel, repository root, lockfile, notebook folder, snapshot manifest, and classroom packages are available before class starts.

# %% [markdown]
# ## Package version record
#
# The watermark output gives a compact reproducibility stamp for the active classroom session.

# %%
# %load_ext watermark
# %watermark -n -u -v -iv -w -p pandas,numpy,yfinance,requests,aleatory,pyarrow,plotly,ipywidgets

# %% [markdown]
# **Output interpretation.**
#
# The watermark stamp records package versions and the last update time for the notebook run. If a classroom issue appears later, this output helps separate content problems from local environment drift.

# %% [markdown]
# ## Handoff
#
# A fully `True` readiness table is the final Module 0 gate. Preserve the table
# with classroom issue reports, then continue to Module 1's market foundations
# and source inventory. Return to the full validation lab if the problem
# involves helpers, snapshots, or credentials beyond this smoke test.

# %% [markdown] tags=["exercise"]
# ## Checkpoint exercise
#
# Use `session_readiness` and the watermark output to write a pre-class go/no-go
# note.
#
# 1. List the Python version and the installed version of every classroom
#    package.
# 2. Identify any failed readiness row and give the command or repository path
#    that resolves it.
# 3. State whether the next data notebook can run offline without credentials.
# 4. Explain what this smoke test does **not** prove compared with the full
#    environment validation lab.
#

# %% [markdown] tags=["solution"]
# ```{dropdown} Suggested answer rubric
# A complete note cites the readiness table and version record, makes an
# explicit go/no-go decision, maps failures to `uv sync` or the relevant
# repository artifact, and recognizes that this smoke test does not re-audit
# every helper, snapshot row count, or optional credential.
# ```
