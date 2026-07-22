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

# %% [markdown] tags=["legacy"]
# # Macro Dashboard: Banxico and FRED
#
# Module: Markets and Data
#
# ## Lesson summary
#
# This dashboard introduces a reusable macro-data workflow and a disciplined way to communicate macro-financial context. The notebook uses a synthetic panel so it can run offline, while the live-data path is centralized in `src.banxico`, `src.fred`, and `src.market_data`.
#
# A dashboard is not just a collection of charts. It should organize information around questions, comparisons, and interpretation {cite}`few2006dashboard,cairo2016truthful`.
#
# ## Learning objectives
#
# By the end of this dashboard, students should be able to:
#
# - identify which macro variables are better sourced from Banxico or FRED;
# - inspect rates, inflation, FX, and equity-index context together;
# - normalize macro series for visual comparison;
# - compute rolling correlations between macro variables and market returns;
# - distinguish offline, live-data, and cached dashboard modes;
# - include source, date range, and data-quality context in dashboard interpretation;
# - document whether a chart uses real data or a reproducible classroom fallback.
#
# ## Setup

# %% tags=["setup", "hide-input", "live-data", "legacy"]
import os

import pandas as pd
import plotly.graph_objects as go
from IPython.display import display
from ipywidgets import Checkbox, Dropdown, IntSlider, interact
from plotly.subplots import make_subplots

from src.banxico import example_banxico_catalog
from src.fred import fred_series_catalog
from src.market_data import dashboard_data_inventory, macro_dashboard_panel

DATA_MODE = os.getenv("DATA_MODE", "offline").lower()
LIVE_DATA_START = os.getenv("LIVE_DATA_START", "2020-01-01")
LIVE_DATA_END = os.getenv("LIVE_DATA_END", "2024-12-31")
RUN_INTERACTIVE_WIDGETS = os.getenv("RUN_INTERACTIVE_WIDGETS", "1") == "1"

# %% [markdown] tags=["legacy"]
# ## Provider map

# %% tags=["legacy"]
dashboard_data_inventory()

# %% tags=["legacy"]
example_banxico_catalog()

# %% tags=["legacy"]
fred_series_catalog()

# %% [markdown] tags=["legacy"]
# ## Macro panel

# %% tags=["live-data", "legacy"]
macro = macro_dashboard_panel(
    data_mode=DATA_MODE,
    start=LIVE_DATA_START,
    end=LIVE_DATA_END,
)
macro.tail()

# %% tags=["legacy"]
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


# %% [markdown] tags=["legacy"]
# ## Dashboard design scope
#
# This dashboard answers first-level macro-financial questions:
#
# | Question | Dashboard view |
# | --- | --- |
# | How did key macro variables move over time? | Macro comparison panel |
# | Are the selected variables comparable in levels or should they be indexed? | Normalize switch |
# | How did market and exchange-rate co-movement change? | Rolling IPC vs USD/MXN correlation |
# | Which source produced the data? | Provider map and panel metadata |
# | Is the view reproducible or live? | `DATA_MODE` summary |
#
# The dashboard should not perform forecasting, causal inference, portfolio optimization, or policy attribution. Those require later modules and stronger assumptions.
#
# ## Offline, live-data, and cache discipline
#
# | Mode | Role | Interpretation note |
# | --- | --- | --- |
# | `DATA_MODE=offline` | Reproducible classroom view | Values are deterministic and suitable for validation |
# | `DATA_MODE=live` | Local demonstration with approved credentials | Values may change because providers update, revise, or limit data |
# | Cached live extracts | Repeatable local refresh workflow | The dashboard should state the retrieval period and source |
#
# Dashboard results should always show data mode, source family, date coverage, and whether the view uses normalized levels or raw levels.
#
# ## Interactive dashboard
#
# Run this cell in JupyterLab with `uv run jupyter lab`.

# %% tags=["interactive", "legacy"]
def plot_macro_dashboard(
    primary_series="banxico_target_rate",
    comparison_series="mexico_inflation",
    normalize=True,
    rolling_window=12,
):
    selected = macro[[primary_series, comparison_series]].copy()
    if normalize:
        selected = selected / selected.iloc[0] * 100

    macro_returns = macro[["ipc_index", "usd_mxn"]].pct_change(fill_method=None)
    rolling_corr = (
        macro_returns["ipc_index"]
        .rolling(rolling_window)
        .corr(macro_returns["usd_mxn"])
        .rename("rolling_correlation")
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Macro comparison", "Rolling IPC vs USD/MXN correlation"),
    )
    fig.add_trace(
        go.Scatter(x=selected.index, y=selected[primary_series], mode="lines", name=primary_series),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=selected.index,
            y=selected[comparison_series],
            mode="lines",
            name=comparison_series,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=rolling_corr.index, y=rolling_corr, mode="lines", name="rolling correlation"),
        row=2,
        col=1,
    )
    fig.update_layout(
        title=f"Macro dashboard ({macro.attrs.get('data_mode', DATA_MODE)} data)",
        template="plotly_white",
        height=650,
    )
    fig.update_yaxes(title_text="Index = 100" if normalize else "Level", row=1, col=1)
    fig.update_yaxes(title_text="Correlation", row=2, col=1)
    display(fig)


DEFAULT_PRIMARY_SERIES = "banxico_target_rate" if "banxico_target_rate" in macro.columns else macro.columns[0]
DEFAULT_COMPARISON_SERIES = "mexico_inflation" if "mexico_inflation" in macro.columns else macro.columns[min(1, len(macro.columns) - 1)]

if RUN_INTERACTIVE_WIDGETS:
    interact(
        plot_macro_dashboard,
        primary_series=Dropdown(options=list(macro.columns), value=DEFAULT_PRIMARY_SERIES),
        comparison_series=Dropdown(options=list(macro.columns), value=DEFAULT_COMPARISON_SERIES),
        normalize=Checkbox(value=True),
        rolling_window=IntSlider(value=12, min=6, max=36, step=1),
    );
else:
    plot_macro_dashboard()

# %% [markdown] tags=["legacy"]
# ## Real-data extension
#
# The default build uses `DATA_MODE=offline`. To run the same dashboard with live Banxico, FRED, and public market data, launch Jupyter locally with credentials and explicit dates:
#
# ```bash
# BANXICO_TOKEN=... FRED_API_KEY=... DATA_MODE=live LIVE_DATA_START=2020-01-01 LIVE_DATA_END=2024-12-31 uv run jupyter lab
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
# The policy-rate proxy increased during the sample while the equity proxy showed periods of higher volatility, which suggests that discount-rate and risk-appetite context should be considered before interpreting returns.
# ```
