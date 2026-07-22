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
# # Initial Repository Setup
#
# Module: Setup and Python Ecosystem
#
# ## Lesson summary
#
# This notebook is the first local checkpoint for a freshly cloned copy of the book. It confirms that the repository is opened from the project root, that the locked `uv` environment is active, and that the core package versions can be recorded for reproducibility {cite}`pyenv2025,uv2025`.
#
# This page is intentionally narrower than the Environment Validation Lab. Use it first after cloning; then run `0.2.environment_validation_lab` for the full package, path, credential, helper, and snapshot validation.
#
# ## Learning objectives
#
# By the end of this note, readers should be able to:
#
# - open the repository from the project root;
# - install the locked environment with `uv sync`;
# - verify that the active kernel is Python 3.12 or newer;
# - confirm that the repository root, lockfile, notebooks, and snapshot manifest are visible;
# - record package versions with `watermark`.
#
# ## Prerequisites
#
# Clone the repository, install Git and `pyenv`, and open a terminal at the
# repository root. This is the first executable gate, so no prior course
# notebook or data credential is required.
#
# ## Lesson flow
#
# 1. Select the local Python runtime.
# 2. Synchronize dependencies from `uv.lock`.
# 3. Open JupyterLab from the project environment.
# 4. Run the setup report and version cells below.

# %% [markdown]
# ## Setup
#
# Run the terminal setup commands from the repository root, then open this notebook with the project kernel created by `uv`. The setup should not require live market data or API credentials.

# %% [markdown]
# ## Terminal setup with pyenv and uv
#
# Run these commands from the repository root before opening JupyterLab. The project requires Python 3.12 or newer.
#
# ```bash
# pyenv install 3.12.12
# pyenv local 3.12.12
# uv sync
# uv run jupyter lab
# ```

# %% [markdown]
# ## Fresh-clone setup report
#
# Run the following cell inside the `uv` environment to confirm that the active kernel can see the repository contract.

# %% tags=["setup", "hide-input"]
import platform
import sys
from pathlib import Path

import pandas as pd

project_root = Path.cwd().resolve()
for candidate in (project_root, *project_root.parents):
    if (candidate / "pyproject.toml").exists() and (candidate / "uv.lock").exists():
        project_root = candidate
        break

setup_report = {
    "python_version": sys.version.split()[0],
    "python_implementation": platform.python_implementation(),
    "project_root": str(project_root),
    "pyproject_present": (project_root / "pyproject.toml").exists(),
    "uv_lock_present": (project_root / "uv.lock").exists(),
    "course_notebooks_present": (project_root / "notebooks" / "course").exists(),
    "snapshot_manifest_present": (project_root / "data" / "snapshots" / "metadata.json").exists(),
}

assert sys.version_info >= (3, 12), setup_report
assert setup_report["pyproject_present"], setup_report
assert setup_report["uv_lock_present"], setup_report
assert setup_report["course_notebooks_present"], setup_report
assert setup_report["snapshot_manifest_present"], setup_report

pd.Series(setup_report, name="status").to_frame()

# %% [markdown]
# **Output interpretation.**
#
# The setup report should show Python 3.12 or newer and `True` for the repository artifacts. If the snapshot manifest is missing, the book can still open, but the real-data publication workflow is incomplete.

# %% [markdown]
# ## Package version record
#
# `watermark` records the exact package versions visible to the active kernel. This is a lightweight audit trail, not a replacement for `uv.lock`.

# %%
# %load_ext watermark
# %watermark -n -u -v -iv -w -p pandas,numpy,yfinance,requests,aleatory,jupyter_book

# %% [markdown]
# **Output interpretation.**
#
# The watermark output confirms that the notebook is running inside the expected scientific Python stack. If package versions are missing or clearly different from the lockfile, rerun `uv sync` before moving to the full validation lab.

# %% [markdown]
# ## Handoff
#
# Keep the setup report as evidence of the fresh-clone state. When every
# repository artifact is present, continue to `0.2.environment_validation_lab`
# for package, helper, snapshot, and optional-credential checks.

# %% [markdown] tags=["exercise"]
# ## Checkpoint exercise
#
# Treat `setup_report` as a fresh-clone acceptance record.
#
# 1. Record the resolved `project_root` and active Python version.
# 2. Confirm that `pyproject.toml`, `uv.lock`, `notebooks/course/`, and the
#    snapshot manifest are all visible.
# 3. If one item is missing, name the repository path that must be restored;
#    if the Python version is too old, give the `pyenv` and `uv sync` commands
#    needed to correct the environment.
# 4. Explain why this narrow report must pass before the broader environment
#    validation lab.
#

# %% [markdown] tags=["solution"]
# ```{dropdown} Suggested answer rubric
# A complete answer reports the actual root and version, verifies all four
# repository artifacts, gives a path-specific remediation for any failure, and
# explains that the next lab tests packages, helpers, snapshots, and optional
# credentials beyond this first-clone gate.
# ```
