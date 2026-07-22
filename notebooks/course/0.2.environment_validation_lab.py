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
# # Environment Validation Lab
#
# Module: Setup and Python Ecosystem
#
# ## Lesson summary
#
# This lab is the executable proof that the Module 0 environment is ready. It
# follows the first-clone setup checkpoint in `0.3.initial_repository_setup` and
# performs the broader validation needed before the course starts using market
# data, dashboards, and time-series notebooks
# {cite}`uv2025,jupyterbook2025,kluyver2016jupyter`.
#
# Run the lab locally from JupyterLab launched with:
#
# ```bash
# uv run jupyter lab
# ```
#
# ## Learning objectives
#
# By the end of this lab, readers should be able to:
#
# - confirm that the active notebook kernel is using Python 3.12 or newer;
# - verify that the scientific finance stack imports correctly;
# - locate the project root and reusable `src/` helpers from any notebook working directory;
# - confirm that the versioned real-data snapshots required by the published book are available;
# - check whether optional API credentials are available without printing secrets;
# - produce a concise environment report for reproducibility.
#
# ## Prerequisites
#
# Complete `0.3.initial_repository_setup` first. The checkout must be open from
# the repository root, `uv sync` must have completed, and the project kernel
# must be active. No live-data credentials are required for this lab.
#
# A local environment is acceptable when the same inputs can reconstruct the same output:
#
# $$
# O = F(D, C, E, \theta).
# $$
#
# This lab validates the environment term, $E$, the project paths that make code, $C$, importable, and the snapshot files that make data, $D$, reproducible.

# %% tags=["setup", "hide-input"]
import json
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

# %%
runtime_report = {
    "python": sys.version.split()[0],
    "implementation": platform.python_implementation(),
    "platform": platform.platform(),
}

assert sys.version_info >= (3, 12), runtime_report
runtime_report

# %% [markdown]
# **Output interpretation.**
#
# The runtime report proves which Python interpreter executed the notebook. The only hard requirement here is Python 3.12 or newer; the platform string is recorded so environment-specific issues can be diagnosed later.

# %% [markdown]
# ## Package checks
#
# The package list covers the libraries that Module 0 expects before later notebooks render tables, diagnostics, models, dashboards, and static book outputs.

# %%
required_packages = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "sklearn": "scikit-learn",
    "arch": "arch",
    "plotly": "plotly",
    "ipywidgets": "ipywidgets",
    "yfinance": "yfinance",
    "jupyter_book": "jupyter-book",
}

package_report = {}
for import_name, distribution_name in required_packages.items():
    try:
        package_report[import_name] = version(distribution_name)
    except PackageNotFoundError:
        package_report[import_name] = "missing"

missing_packages = [
    name for name, package_version in package_report.items()
    if package_version == "missing"
]
assert not missing_packages, missing_packages
package_report


# %% [markdown]
# **Output interpretation.**
#
# Every package should show a concrete version rather than `missing`. This table is not a dependency specification by itself; `uv.lock` remains the source of truth, while the output confirms that the active kernel is using that locked environment.

# %% [markdown]
# ## Project paths
#
# These checks make sure the notebook can locate the same repository structure used by the publication build.

# %%
def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "_config.yml").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root from the current working directory.")


project_root = find_project_root(Path.cwd().resolve())
expected_paths = {
    "book_config": project_root / "_config.yml",
    "book_outputs_config": project_root / "_config.outputs.yml",
    "book_toc": project_root / "_toc.yml",
    "project_metadata": project_root / "pyproject.toml",
    "dependency_lock": project_root / "uv.lock",
    "source_helpers": project_root / "src",
    "course_notebooks": project_root / "notebooks" / "course",
    "snapshot_manifest": project_root / "data" / "snapshots" / "metadata.json",
    "generated_images": project_root / "img" / "generated",
}

path_report = {name: path.exists() for name, path in expected_paths.items()}
assert all(path_report.values()), path_report
path_report

# %% [markdown]
# **Output interpretation.**
#
# All path checks should be `True`. If any value is `False`, the notebook is probably being run from the wrong checkout, an incomplete clone, or a stale branch missing required book artifacts.

# %% [markdown]
# ## Data snapshot check
#
# The published book should render from real, versioned snapshots rather than live API calls. This check verifies the local snapshot manifest and the CSV files it declares.

# %%
metadata_path = expected_paths["snapshot_manifest"]
snapshot_metadata = json.loads(metadata_path.read_text())
required_snapshots = {
    "official_price_panel",
    "official_macro_panel",
    "nasdaq_stock_panel",
}

snapshot_records = []
for snapshot_name, relative_path in snapshot_metadata["outputs"].items():
    snapshot_path = project_root / relative_path
    exists = snapshot_path.exists()
    row_count = None
    column_count = None
    if exists:
        snapshot_frame = pd.read_csv(snapshot_path)
        row_count = len(snapshot_frame)
        column_count = len(snapshot_frame.columns)
    snapshot_records.append(
        {
            "snapshot": snapshot_name,
            "exists": exists,
            "rows": row_count,
            "columns": column_count,
            "declared_rows": snapshot_metadata.get("row_counts", {}).get(snapshot_name),
            "required_for_publication": snapshot_name in required_snapshots,
        }
    )

snapshot_report = pd.DataFrame(snapshot_records).set_index("snapshot")
required_ok = snapshot_report.loc[list(required_snapshots), "exists"].all()
rows_ok = (snapshot_report.loc[list(required_snapshots), "rows"] > 0).all()
assert required_ok and rows_ok, snapshot_report
snapshot_report

# %% [markdown]
# **Output interpretation.**
#
# The required publication snapshots should exist and have positive row counts. Differences between `rows` and `declared_rows` are a signal that the CSV files and `metadata.json` were not refreshed together.

# %% [markdown]
# ## Local credential check
#
# This cell checks whether optional variables exist, but never prints token values.

# %%
credential_report = {
    "BANXICO_TOKEN": bool(os.environ.get("BANXICO_TOKEN")),
    "DBNOMICS_API_KEY": bool(os.environ.get("DBNOMICS_API_KEY")),
}

credential_report

# %% [markdown]
# **Output interpretation.**
#
# `False` values are acceptable for the default publication workflow because the book uses snapshots. Credentials are only needed for controlled live-data refreshes or local instructor-approved provider checks.

# %% [markdown]
# ## Source helper import
#
# Notebooks should import reusable logic from `src/` instead of duplicating provider, quality, or diagnostic code inline.

# %%
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.market_data import official_macro_panel, official_price_panel
from src.market_data_quality import banxico_series_catalog, source_inventory_template
from src.time_series_diagnostics import adf_report

helper_report = {
    "banxico_catalog_rows": len(banxico_series_catalog()),
    "source_inventory_columns": len(source_inventory_template().columns),
    "official_price_panel_callable": callable(official_price_panel),
    "official_macro_panel_callable": callable(official_macro_panel),
    "adf_report_callable": callable(adf_report),
}
helper_report

# %% [markdown]
# **Output interpretation.**
#
# The helper report confirms that the local `src/` package is importable from the notebook kernel. Positive catalog and inventory counts show that provider metadata is available before any market-data lesson starts.

# %% [markdown]
# ## Environment report
#
# The final report is intentionally compact. It records whether the runtime, packages, paths, snapshots, credentials, and helper imports passed without exposing local secrets.

# %%
environment_report = {
    "python_ok": sys.version_info >= (3, 12),
    "packages_ok": not missing_packages,
    "paths_ok": all(path_report.values()),
    "snapshots_ok": bool(required_ok and rows_ok),
    "optional_credentials": credential_report,
    "helpers_ok": all(bool(value) for value in helper_report.values()),
}

environment_report

# %% [markdown]
# **Output interpretation.**
#
# A ready environment has `python_ok`, `packages_ok`, `paths_ok`, `snapshots_ok`, and `helpers_ok` set to `True`. Optional credentials can remain `False` unless the current task is a live provider refresh.

# %% [markdown]
# ## Handoff
#
# Save the compact environment report with any reproducibility issue. Once all
# required fields are `True`, run the classroom smoke test in
# `0.4.classroom_environment_setup`; do not start Module 1 with a failed path,
# package, snapshot, or helper check.
