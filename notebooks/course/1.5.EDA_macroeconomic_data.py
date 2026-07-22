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
# # Macroeconomic Data Exploratory Analysis
#
# Module: Markets, Instruments, and Data
#
# ## Lesson summary
#
# This notebook introduces exploratory analysis for macroeconomic series. It
# focuses on frequency, units, normalization, co-movement, and source limitations
# before students use the interactive macro dashboard
# {cite}`banxicoSIE2025,dbnomics2025,mishkin2019financial`.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - distinguish levels, rates, indexes, and exchange rates in macro data;
# - keep rates in native percentage units and rebase only positive level paths;
# - compute changes and rolling co-movement without confusing levels and returns;
# - read native-unit panels, normalized index paths, and a standardized regime heatmap;
# - document publication frequency, release lag, and source limitations;
# - prepare a macro panel for the Banxico/DB.NOMICS dashboard.
#
# ## Prerequisites
#
# Complete the source inventory in `1.3` and the quality framework in `1.6`.
# Readers should distinguish rates, index levels, exchange rates, changes, and
# returns, and should understand that macro reference periods and publication
# dates are not interchangeable.
#
# ## Setup

# %% tags=["setup", "hide-input"]
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from src.dbnomics import dbnomics_series_catalog
from src.market_data import official_macro_panel
from src.market_data_quality import data_quality_report, source_inventory_template
from src.module1_visuals import build_macro_eda_overview

pd.set_option("display.max_columns", 80)
ANALYSIS_START = "2021-01-01"
ANALYSIS_END = "2025-06-30"

# %% [markdown]
# ## Macro source inventory

# %%
inventory = source_inventory_template()
inventory.loc[0] = {
    "provider": "Banxico SIE",
    "instrument_or_variable": "target rate, TIIE, CETES, FIX, UDI",
    "frequency": "daily, auction, or publication frequency",
    "start": ANALYSIS_START,
    "end": ANALYSIS_END,
    "field": "published value",
    "currency": "MXN, UDI, or percent",
    "calendar": "Mexican publication calendar",
    "known_limitations": "requires token; observations are not necessarily business-day complete",
}
inventory.loc[1] = {
    "provider": "DB.NOMICS",
    "instrument_or_variable": "FED/H15/RIFLGFCY10_N.B, IMF/CPI/M.MX.PCPI_IX",
    "frequency": "daily or monthly",
    "start": ANALYSIS_START,
    "end": ANALYSIS_END,
    "field": "value",
    "currency": "USD, MXN, index, or percent",
    "calendar": "underlying FED and IMF provider calendars",
    "known_limitations": "publication lag, revisions, and mixed frequencies",
}
inventory

# %% [markdown]
# **Output interpretation.**
#
# The inventory table is a source-risk map. Banxico series are official Mexican
# publications. DB.NOMICS standardizes series from named underlying providers,
# so both the DB.NOMICS dataset code and original provider remain part of the
# provenance chain. Mixed frequency, release lag, and revision policy are
# analytical issues, not cosmetic details.

# %%
dbnomics_series_catalog()

# %% [markdown]
# **Output interpretation.**
#
# The source tables clarify which indicators come from Banxico and which come from DB.NOMICS. Use the provider and frequency fields to decide whether an apparent macro relationship is economically comparable or just an artifact of mixed release calendars.
#

# %% [markdown]
# ## Classroom macro panel
#
# This notebook uses the Module 1 common window from 2021-01-01 through
# 2025-06-30. A monthly observation's publication date generally occurs after
# its reference period. Because this panel is indexed by reference period and
# uses the latest available vintage, it supports exploratory description rather
# than a point-in-time backtest.

# %%
macro = official_macro_panel(start=ANALYSIS_START, end=ANALYSIS_END)
macro.tail()

# %% [markdown]
# **Output interpretation.**
#
# The macro panel tail shows the latest rows of the processed month-end
# intersection used by the book. The underlying source series have different
# release calendars, but this publication panel retains only complete common
# reference periods; it must not be mistaken for a historical-vintage panel.
#

# %%
data_quality_report(macro, expected_frequency="ME")

# %% [markdown]
# **Output interpretation.**
#
# The quality report verifies stored completeness, month-end continuity, and
# duplicated timestamps in this processed intersection. It does not audit the
# source-specific release calendar or prove historical availability.
#

# %% [markdown]
# ## Separate native rates from rebased level paths
#
# Rates, exchange rates, and index levels should not be divided by their first
# values indiscriminately. Rates remain interpretable in percentage units,
# while positive level paths can be rebased for a path comparison. Rebasing
# supports comparison of paths; it does not make their original units or
# economic meanings interchangeable
# {cite}`cleveland1993visualizing,wilke2019dataviz`.

# %%
rate_context_percent = (
    macro[
        [
            "banxico_target_rate",
            "cetes_28d",
            "tiie_28d",
            "mexico_inflation",
            "us_10y",
        ]
    ]
    * 100
)
rebased_level_paths = (
    macro[["usd_mxn", "udi", "mexico_cpi"]]
    .div(macro[["usd_mxn", "udi", "mexico_cpi"]].iloc[0])
    .mul(100)
)

display(rate_context_percent.tail())
display(rebased_level_paths.tail())

# %% [markdown]
# **Output interpretation.**
#
# The first table keeps rates in percent. The second puts only positive level
# paths on a common starting index; values above 100 indicate increases from the
# first observation, but the output no longer represents MXN per USD, MXN per
# UDI, or CPI index units.
#

# %% [markdown]
# ## Changes and co-movement

# %%
rate_changes_pp = macro[["banxico_target_rate", "mexico_inflation", "us_10y"]].diff() * 100
percentage_changes = macro[["usd_mxn", "udi"]].pct_change(fill_method=None)

pd.concat(
    [
        rate_changes_pp.add_suffix("_pp_change"),
        percentage_changes.add_suffix("_pct_change"),
    ],
    axis=1,
).dropna().tail()

# %% [markdown]
# **Output interpretation.**
#
# The table reports arithmetic rate differences in percentage points and
# percentage changes for positive USD/MXN and UDI levels. These transformations are often a more
# appropriate scale for co-movement questions, but levels remain necessary for
# questions about policy stance, purchasing power, or historical range: a high
# level and a rising level are different signals.
#

# %%
rolling_corr = percentage_changes["udi"].rolling(12).corr(percentage_changes["usd_mxn"])
rolling_corr.dropna().tail()

# %% [markdown]
# **Output interpretation.**
#
# The rolling-correlation tail reports the latest local relationship. A changing
# sign or magnitude should be treated as evidence of regime variation, not as a
# single permanent relationship.
#
# ## Macro visual diagnostic board
#
# One normalized line chart cannot faithfully represent rates, inflation,
# exchange rates, and price indexes at once. The board below keeps rates in
# percent, rebases only price/index paths, standardizes changes only for the
# regime heatmap, and preserves the rolling relationship as its own view.
#

# %%
macro_eda_figure, transformed_change_panel = build_macro_eda_overview(
    macro,
    rolling_window=12,
)
display(macro_eda_figure)
plt.close(macro_eda_figure)

# %% [markdown]
# **Output interpretation.**
#
# The top row answers level and path questions without mixing units. In the
# regime map, color represents a within-series z-score, so intensity identifies
# unusual changes but does not compare the economic magnitude of a rate move
# with a USD/MXN percentage change. The lower-right view shows whether UDI and USD/MXN changes
# moved together consistently or only in selected regimes.
#

# %% [markdown]
# ## Macro interpretation checklist
#
# | Question | Interpretation risk |
# | --- | --- |
# | Is the variable a level, rate, index, or return? | Transformations are not interchangeable. |
# | Is the frequency daily, monthly, quarterly, or irregular? | Mixed frequencies can create artificial persistence. |
# | Is the value revised after publication? | Historical analysis can use information unavailable at the time. |
# | Which market calendar applies? | Mexican and US holidays do not line up perfectly. |
# | Does the chart use a versioned provider snapshot or live data? | Claims should state the provider, underlying series, vintage, and whether the table was refreshed live or read from the versioned snapshot. |
#
# ## Handoff
#
# The native-rate table, rebased level paths, and transformation choices are the
# handoff to the macro dashboard. Preserve original units alongside rebased views, and do not use
# latest-vintage macro values as if they had been known on each historical
# observation date.

# %% [markdown] tags=["exercise"]
# ## Checkpoint exercise
#
# Select one rate, one index, and USD/MXN from `macro`. Retain the quote
# convention: USD/MXN is measured as MXN per USD, so an increase denotes USD
# appreciation and MXN depreciation.
#
# 1. Record each series' provider, original unit, frequency, first observation,
#    last observation, and missing count in the common window.
# 2. Show the economically appropriate transformation for each: percentage-
#    point change for the rate, percent change or rebasing for the index, and
#    percent change for USD/MXN.
# 3. Compare the raw-level chart with the normalized chart and identify one
#    conclusion that the raw mixed-unit view cannot support.
# 4. State how release lags or revisions limit a historical interpretation.
#

# %% [markdown] tags=["solution"]
# ```{dropdown} Suggested answer rubric
# A complete response keeps levels, rates, indexes, and exchange rates
# conceptually distinct; reports coverage and missingness; applies and labels
# the appropriate transformation; and explains why normalized co-movement is
# descriptive rather than causal or point-in-time evidence. Reconcile dates,
# counts, and transformations with the executed source inventory, quality report,
# and transformed tables; those outputs are the snapshot-specific self-check.
# ```
