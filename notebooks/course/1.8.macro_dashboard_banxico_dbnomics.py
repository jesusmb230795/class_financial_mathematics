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
# # Macro Dashboard: Banxico and DB.NOMICS
#
# Module: Markets, Instruments, and Data
#
# ## Lesson summary
#
# This dashboard introduces a reusable macro-data workflow and a disciplined way to communicate macro-financial context. The default offline path reads versioned Banxico and DB.NOMICS snapshots, while the live-data path is centralized in `src.banxico`, `src.dbnomics`, and `src.market_data`.
#
# A dashboard is not just a collection of charts. It should organize information around questions, comparisons, and interpretation {cite}`few2006dashboard,cairo2016truthful`.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - identify which macro variables are better sourced from Banxico or DB.NOMICS;
# - inspect rates, inflation, FX, and inflation-linked levels together;
# - compare macro series in native units and rebase appropriate level paths when requested;
# - compute rolling correlations between appropriately transformed selected variables;
# - distinguish offline, live-data, and cached dashboard modes;
# - include source, date range, and data-quality context in dashboard interpretation;
# - document whether a chart uses the versioned real-data snapshot or a live provider refresh.
#
# ## Prerequisites
#
# Complete macro EDA (`1.5`) and the quality framework (`1.6`). Readers should
# know each selected series' unit, frequency, source, transformation, and
# release limitation before using a dashboard comparison.
#
# ## Setup

# %% tags=["setup", "hide-input", "live-data"]
import os

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from ipywidgets import Checkbox, Dropdown, IntSlider, interact

from src.banxico import example_banxico_catalog
from src.dashboard_fallbacks import build_macro_dashboard_fallback
from src.dashboards import build_macro_dashboard
from src.dbnomics import dbnomics_series_catalog
from src.market_data import dashboard_data_inventory, macro_dashboard_panel

DATA_MODE = os.getenv("DATA_MODE", "offline").lower()
LIVE_DATA_START = os.getenv("LIVE_DATA_START", "2021-01-01")
LIVE_DATA_END = os.getenv("LIVE_DATA_END", "2025-06-30")
RUN_INTERACTIVE_WIDGETS = os.getenv("RUN_INTERACTIVE_WIDGETS", "1") == "1"

# %% [markdown]
# ## Provider map

# %%
dashboard_data_inventory()

# %% [markdown]
# **Output interpretation.**
#
# The dashboard inventory states what each provider contributes to the macro view. It prevents the dashboard from silently mixing domestic monetary indicators with international context without exposing source and frequency differences.

# %%
example_banxico_catalog()

# %% [markdown]
# **Output interpretation.**
#
# The Banxico catalog gives the exact Mexican series definitions behind the dashboard. These labels are part of the interpretation because a rate, an exchange-rate fixing, and an indexed unit answer different economic questions.

# %%
dbnomics_series_catalog()

# %% [markdown]
# **Output interpretation.**
#
# The provider outputs define what the dashboard is allowed to claim. A dashboard that mixes Banxico and DB.NOMICS should keep source, frequency, and transformation notes visible because the charts depend on those choices.
#

# %% [markdown]
# ## Macro panel

# %% tags=["live-data"]
macro = macro_dashboard_panel(
    data_mode=DATA_MODE,
    start=LIVE_DATA_START,
    end=LIVE_DATA_END,
)
macro.tail()

# %% [markdown]
# **Output interpretation.**
#
# The macro tail verifies the values feeding the dashboard. Before interpreting a chart movement, confirm whether it comes from a level, a rate, or an index and whether the latest timestamp is shared across series.
#

# %%
pd.DataFrame(
    [
        {
            "data_mode": macro.attrs.get("data_mode", DATA_MODE),
            "sources": macro.attrs.get("sources", "not specified"),
            "start": macro.index.min(),
            "end": macro.index.max(),
            "observations": len(macro),
        }
    ]
)


# %% [markdown]
# **Output interpretation.**
#
# The metadata table is part of the output, not administrative detail. It tells the reader whether the dashboard came from the versioned snapshot or a live refresh and therefore how reproducible the visible result is.
#

# %% [markdown]
# ## Dashboard design scope
#
# The dashboard has one primary analytical objective: **compare the direction
# and recent movement of two explicitly selected macro-financial variables,
# while keeping their units, sources, frequency, and data mode visible**. It
# supports the following first-level questions:
#
# | Question | Dashboard view |
# | --- | --- |
# | How did key macro variables move over time? | Macro comparison panel |
# | Are the selected variables comparable in levels or should they be indexed? | Normalize switch |
# | How did the selected variables co-move? | Rolling correlation of percentage-point rate changes or percentage/log changes of positive levels, as appropriate |
# | Which source produced the data? | Provider map and panel metadata |
# | Is the view reproducible or live? | `DATA_MODE` summary |
#
# The dashboard should not perform forecasting, causal inference, portfolio optimization, or policy attribution. Those require later modules and stronger assumptions.
#
# ## Snapshot, live-data, and cache discipline
#
# | Mode | Role | Interpretation note |
# | --- | --- | --- |
# | `DATA_MODE=offline` | Reproducible classroom view | Values come from versioned Banxico and DB.NOMICS snapshots |
# | `DATA_MODE=live` | Local demonstration with approved credentials | Values may change because providers update, revise, or limit data |
# | Cached live extracts | Repeatable local refresh workflow | The dashboard should state the retrieval period and source |
#
# Dashboard results should always show data mode, source family, observed date
# coverage, and whether the view uses normalized levels or native units. The
# normalized view shares a base-100 panel; native-unit mode uses aligned,
# separate panels so unlike quantities are never compared on one scale.
#
# ## Interactive dashboard
#
# Run this cell in JupyterLab with `uv run jupyter lab`.


# %% tags=["interactive"]
def plot_macro_dashboard(
    primary_series="banxico_target_rate",
    comparison_series="mexico_inflation",
    normalize=False,
    rolling_window=12,
):
    if RUN_INTERACTIVE_WIDGETS:
        figure = build_macro_dashboard(
            macro,
            primary_series,
            comparison_series,
            normalize=normalize,
            rolling_window=rolling_window,
            data_mode=macro.attrs.get("data_mode", DATA_MODE),
        )
    else:
        figure = build_macro_dashboard_fallback(
            macro,
            primary_series,
            comparison_series,
            normalize=normalize,
            rolling_window=rolling_window,
            data_mode=macro.attrs.get("data_mode", DATA_MODE),
        )
    display(figure)
    if not RUN_INTERACTIVE_WIDGETS:
        plt.close(figure)


SERIES_LABELS = {
    "banxico_target_rate": "Banxico target rate",
    "cetes_28d": "28-day CETES rate",
    "tiie_28d": "28-day TIIE",
    "usd_mxn": "USD/MXN FIX (MXN per USD)",
    "udi": "UDI level (MXN per UDI)",
    "mexico_cpi": "Mexico CPI index",
    "mexico_inflation": "Mexico year-over-year inflation",
    "us_10y": "US 10-year Treasury yield",
}
SERIES_OPTIONS = [(SERIES_LABELS.get(column, column), column) for column in macro.columns]
RATE_OR_INFLATION_SERIES = {
    "banxico_target_rate",
    "cetes_28d",
    "tiie_28d",
    "mexico_inflation",
    "us_10y",
}
DEFAULT_PRIMARY_SERIES = (
    "banxico_target_rate" if "banxico_target_rate" in macro.columns else macro.columns[0]
)
DEFAULT_COMPARISON_SERIES = (
    "mexico_inflation"
    if "mexico_inflation" in macro.columns
    else next(column for column in macro.columns if column != DEFAULT_PRIMARY_SERIES)
)

if RUN_INTERACTIVE_WIDGETS:
    primary_widget = Dropdown(options=SERIES_OPTIONS, value=DEFAULT_PRIMARY_SERIES)
    comparison_widget = Dropdown(options=SERIES_OPTIONS, value=DEFAULT_COMPARISON_SERIES)
    normalize_widget = Checkbox(value=False, description="Rates stay native")

    def update_rebase_control(*_):
        selected = {primary_widget.value, comparison_widget.value}
        rate_selected = bool(selected & RATE_OR_INFLATION_SERIES)
        normalize_widget.disabled = rate_selected
        normalize_widget.description = "Rates stay native" if rate_selected else "Rebase levels"
        if rate_selected:
            normalize_widget.value = False

    primary_widget.observe(update_rebase_control, names="value")
    comparison_widget.observe(update_rebase_control, names="value")
    update_rebase_control()
    interact(
        plot_macro_dashboard,
        primary_series=primary_widget,
        comparison_series=comparison_widget,
        normalize=normalize_widget,
        rolling_window=IntSlider(
            value=12,
            min=6,
            max=36,
            step=1,
            description="Months",
        ),
    )
else:
    plot_macro_dashboard()


# %% [markdown]
# **Output interpretation.**
#
# The dashboard should be read as a comparison surface. Use the top panel for
# level and direction, then read the lower panel as the rolling relationship
# between the **same selected pair** after an explicit transformation: first
# differences for rates and inflation, percentage changes for positive index or
# level series. The source and sample note prevents that descriptive
# co-movement from being read as a causal claim.
# The rebase control is enabled only when both selected variables are positive
# level series. Rates and inflation remain in native percentage units.
#

# %% [markdown]
# ## Real-data extension
#
# The default build uses `DATA_MODE=offline`, which reads the versioned
# Banxico/DB.NOMICS provider snapshot. To run the same dashboard with live
# Banxico and DB.NOMICS data, launch Jupyter locally with
# credentials and explicit dates:
#
# ```bash
# BANXICO_TOKEN=... DBNOMICS_API_KEY=... DATA_MODE=live LIVE_DATA_START=2021-01-01 LIVE_DATA_END=2025-06-30 uv run jupyter lab
# ```
#
# The notebook should not duplicate API request, token, parsing, resampling, unit conversion, or cache logic.
#
# ## Interpretation note
#
# Macro dashboard outputs should be written as observations and hypotheses, not causal claims. A useful sentence structure is:
#
# ```text
# [Macro pattern] coincided with [market pattern], which matters because [financial implication].
# ```
#
# Example:
#
# ```text
# The selected policy-rate series increased during the sample while USD/MXN and
# inflation-linked levels moved through different regimes. The co-movement is a
# reason to examine discount-rate, currency, and inflation context before
# interpreting asset returns; it is not evidence that one series caused another.
# ```
#
# ## Handoff
#
# The selected variables, transformation state, rolling window, source, and
# date range form the dashboard handoff to the integrated Mexican case. Preserve
# them with any exported chart so the image cannot be interpreted outside its
# data contract.
