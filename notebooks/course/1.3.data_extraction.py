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
# # Financial Data Extraction
#
# Module: Markets, Instruments, and Data
#
# ## Lesson summary
#
# This notebook introduces provider-aware data extraction for market prices, macroeconomic indicators, reference data, corporate data, and metadata. The default run uses versioned real-data snapshots from Banxico and DB.NOMICS, while live extraction is an instructor-approved extension through reusable clients in `src`.
#
# The purpose is to teach a reproducible extraction workflow: choose sources deliberately, document metadata, preserve raw observations, use cache, and separate retrieval from cleaning, validation, and analysis {cite}`wilkinson2016fair`.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - distinguish official, commercial, and convenience data providers;
# - classify market, macroeconomic, reference, corporate, and metadata sources;
# - evaluate providers by authority, coverage, definition, frequency, latency, revisions, accessibility, reliability, licensing, and reproducibility;
# - document provider, field, calendar, currency, and known limitations before analysis;
# - separate raw, interim, processed, metadata, and cache layers;
# - load classroom-safe price and macro panels without network access;
# - use `MarketDataClient` as the live-data entry point instead of writing provider logic inside notebooks;
# - explain why cached extracts are part of a reproducible quantitative workflow.
#
# ## Prerequisites
#
# Complete Module 0 and the Market Foundations reading. Readers should be able
# to distinguish prices, rates, yields, indexes, and macro levels before
# selecting fields or transformations.
#
# ## Setup

# %% tags=["setup", "hide-input"]
import os

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from src.banxico import example_banxico_catalog
from src.dbnomics import dbnomics_series_catalog
from src.market_data import (
    dashboard_data_inventory,
    live_macro_dashboard_panel,
    live_nasdaq_stock_prices,
    mexican_market_level_panel,
    nasdaq_stock_price_panel,
    official_macro_panel,
    wfe_equity_market_scale_snapshot,
)
from src.market_data_quality import data_quality_report, source_inventory_template
from src.module1_visuals import build_wfe_market_scale_overview
from src.visual_style import set_matplotlib_theme

DATA_MODE = os.getenv("DATA_MODE", "offline")
ANALYSIS_START = "2021-01-01"
ANALYSIS_END = "2025-06-30"
set_matplotlib_theme()

# %% [markdown]
# ## Module workflow preview
#
# This lesson moves from market concepts to the first two questions in the
# module's data workflow. The remaining questions become explicit handoffs, not
# promises that extraction alone has cleaned or modeled the data:
#
# 1. **This lesson:** Where do financial and macroeconomic data come from?
# 2. **This lesson:** How should data be retrieved, cached, and stored?
# 3. **Lesson `1.6`:** How should raw data be audited and validated?
# 4. **Lessons `1.4` and `1.5`:** Which transformations match prices, rates,
#    indexes, and macro levels?
# 5. **Lesson `1.7`:** How can an auditable Mexican market pipeline be built?
#
# Every Module 1 calculation uses the inclusive teaching window from
# `2021-01-01` through `2025-06-30`. Source files can have wider coverage; the
# slice makes comparisons across lessons explicit. Weekends, holidays, and
# release schedules can still make the first or last observed timestamp differ
# by series.
#
# ## Source taxonomy
#
# A financial dataset is only useful if the analyst understands its source, definition, frequency, coverage, limitations, update process, and usage restrictions. Data should not be treated as neutral input; every source reflects institutional rules, market conventions, vendor decisions, and sometimes revision policies.
#
# | Data type | Examples | Typical source family | Main caution |
# | --- | --- | --- | --- |
# | Market data | prices, volumes, bid-ask quotes, index levels, yields, spreads, exchange rates | exchanges, vendors, brokers, public portals | fields and calendars differ by instrument and provider |
# | Macroeconomic data | inflation, rates, GDP, employment, industrial activity, trade, monetary aggregates | central banks, statistical offices, DB.NOMICS and provider release archives | publication lags and revisions matter |
# | Reference data | tickers, identifiers, issuers, calendars, sectors, maturity dates, coupons | exchanges, vendors, internal dictionaries | stale mappings can corrupt joins |
# | Corporate data | statements, dividends, splits, earnings, debt issuance, corporate actions | issuers, exchanges, vendors | events affect return calculation |
# | Metadata | units, frequency, methodology, retrieval date, license, adjustment policy | source docs and local data dictionaries | missing metadata makes results hard to audit |
#
# Market data are often transaction-driven and high-frequency. Macroeconomic data are usually publication-driven, lower-frequency, and sometimes revised after initial release. That difference affects how series should be aligned before analysis.
#
# ## Provider inventory
#
# Start with the source map before requesting data. A provider choice is a modeling assumption because it determines fields, calendars, missing values, revisions, and usage limits.

# %%
dashboard_data_inventory()

# %% [markdown]
# **Output interpretation.**
#
# The inventory frames each provider by analytical role before any extraction
# code runs. Banxico is the official source for the selected Mexican rates and
# reference series. DB.NOMICS standardizes series from underlying providers, so
# both the DB.NOMICS dataset code and the original provider remain part of the
# provenance chain. Keeping the families separate makes units, calendars, and
# revision risks visible {cite}`banxicoSIE2025,dbnomics2025`.

# %%
example_banxico_catalog()

# %% [markdown]
# **Output interpretation.**
#
# The Banxico catalog output should be read as a contract, not just a list. Each series ID fixes the field definition, unit, calendar, and provider dependency that later tables and charts inherit.

# %%
dbnomics_series_catalog()

# %% [markdown]
# **Output interpretation.**
#
# The provider tables show the data contract before any analysis begins. Read them as a map of authority, frequency, credential requirements, and publication role; a series should not be merged into a panel until those fields are clear.
#

# %% [markdown]
# ## Source selection criteria
#
# Do not choose a source only because it is convenient. Evaluate it explicitly.
#
# | Criterion | Practical question |
# | --- | --- |
# | Authority | Is the source official, licensed, audited, or widely accepted? |
# | Coverage | Which instruments, markets, dates, and frequencies are included? |
# | Definition | Are units, methodology, and adjustments documented? |
# | Frequency | Is the data daily, weekly, monthly, quarterly, intraday, or event-based? |
# | Latency | How quickly is the data updated after publication or trading? |
# | Revisions | Can historical values change after publication? |
# | Accessibility | Is there an API, bulk download, manual file, or web interface? |
# | Reliability | Are outages, missing values, or schema changes common? |
# | Licensing | Can the data be stored, redistributed, published, or commercialized? |
# | Reproducibility | Can another analyst retrieve the same data with the same code? |
#
# For educational work, use a simple hierarchy: official sources for macro and regulatory data, exchange or licensed sources for official market data, public portals and wrappers for exploratory learning, and local cached extracts or committed snapshots of real provider data for reproducible exercises.
#
# ## Source inventory template

# %%
inventory = source_inventory_template()
inventory.loc[0] = {
    "provider": "Yahoo Finance",
    "instrument_or_variable": "AAPL, MSFT, NVDA, AMZN, GOOGL",
    "frequency": "daily",
    "start": ANALYSIS_START,
    "end": ANALYSIS_END,
    "field": "adjusted close",
    "currency": "USD",
    "calendar": "trading days",
    "known_limitations": "unofficial yfinance access path; fixed classroom extract only; intended-use terms must be reviewed separately",
}
inventory.loc[1] = {
    "provider": "Banxico SIE",
    "instrument_or_variable": "SF61745, SF60633, SF60648, SF43718, SP68257",
    "frequency": "daily or publication frequency",
    "start": ANALYSIS_START,
    "end": ANALYSIS_END,
    "field": "published value",
    "currency": "percent, MXN per USD, or MXN per UDI",
    "calendar": "Mexican publication calendar",
    "known_limitations": "requires token; macro and rate series can have missing publication dates",
}
inventory.loc[2] = {
    "provider": "DB.NOMICS",
    "instrument_or_variable": "FED/H15/RIFLGFCY10_N.B, IMF/CPI/M.MX.PCPI_IX",
    "frequency": "daily or monthly",
    "start": ANALYSIS_START,
    "end": ANALYSIS_END,
    "field": "value",
    "currency": "USD, MXN, index, or percent",
    "calendar": "underlying FED and IMF provider calendars",
    "known_limitations": "series revisions, publication lags, and mixed frequencies",
}
inventory.loc[3] = {
    "provider": "INEGI",
    "instrument_or_variable": "inflation or economic activity indicator",
    "frequency": "monthly or quarterly",
    "start": ANALYSIS_START,
    "end": ANALYSIS_END,
    "field": "published value",
    "currency": "index, percent, or level",
    "calendar": "Mexican statistical publication calendar",
    "known_limitations": "publication lags, revisions, and indicator-specific units",
}
inventory.loc[4] = {
    "provider": "BMV or licensed market data source",
    "instrument_or_variable": "Mexican equities, ETFs, FIBRAs, or indices",
    "frequency": "daily or intraday",
    "start": ANALYSIS_START,
    "end": ANALYSIS_END,
    "field": "price, volume, index level, or market data product field",
    "currency": "MXN",
    "calendar": "Mexican trading calendar",
    "known_limitations": "licensing, redistribution limits, and instrument-specific field definitions",
}
inventory

# %% [markdown]
# **Output interpretation.**
#
# The source inventory separates official sources from convenience sources and makes the tradeoff visible. A provider with easier access is not automatically better; coverage, definitions, update policy, and reproducibility determine whether it belongs in the workflow.
#

# %% [markdown]
# ## Market scale snapshot
#
# Market datasets are easier to interpret when students see the order of magnitude of the system that produces them. The World Federation of Exchanges publishes dashboard indicators such as market capitalization, value of share trading, listed companies, number of trades, investment flows, and exchange-traded derivative contracts {cite}`wfeDashboardMay2026`.
#
# The static snapshot below is copied into `src.market_data` so the book can render the example without network access. Use it as a scale reference, not as a live market-data feed.
#

# %%
market_scale = wfe_equity_market_scale_snapshot()
market_scale[
    [
        "metric",
        "display_value",
        "display_unit",
        "change_percent",
        "change_basis",
        "source_snapshot",
        "verified_on",
    ]
]


# %% [markdown]
# **Output interpretation.**
#
# The WFE snapshot gives order-of-magnitude context for global markets. Use the table to identify what each metric measures before comparing values, because capitalization, trading value, number of companies, and contracts are different economic objects.
#

# %%
market_scale_figure = build_wfe_market_scale_overview(
    market_scale,
    sample_label="May 2026",
    verified_on="2026-07-20",
)
display(market_scale_figure)
plt.close(market_scale_figure)


# %% [markdown]
# **Output interpretation.**
#
# The metric cards report all seven WFE indicators with their original units
# and published percentage labels. The source page does not state the comparison
# period for those percentages, so the notebook does not call them monthly or
# annual changes. The chart groups only economically comparable units;
# in particular, the derivatives panel compares options and futures contracts.
# Relative lengths across different groups have no quantitative meaning because
# stocks, monetary flows, companies, trades, and contracts are different objects.

# %% [markdown]
# Small multiples preserve scale awareness without placing incompatible units on
# one axis. Before modeling a return series, students should still identify
# whether a field describes a stock, a flow, a transaction count, or a contract
# count—even when two fields happen to share the same display unit.
#

# %% [markdown]
# ## Real-data classroom extraction
#
# The book build should not depend on live external APIs, but it should still use real observations. The published notebooks therefore read versioned CSV snapshots generated from Banxico SIE and DB.NOMICS. Students can regenerate those snapshots with approved credentials, then compare the refreshed metadata with the committed publication snapshot.

# %%
levels = mexican_market_level_panel(start=ANALYSIS_START, end=ANALYSIS_END)
nasdaq_prices = nasdaq_stock_price_panel(start=ANALYSIS_START, end=ANALYSIS_END)
macro = official_macro_panel(start=ANALYSIS_START, end=ANALYSIS_END)

levels.tail()

# %% [markdown]
# **Output interpretation.**
#
# The final rows contain jointly observed USD/MXN and UDI levels together with
# documented synthetic short-rate carry indexes. Treat the observed levels as
# historical data and the carry columns as constructed classroom references;
# neither is a forecast or a general-purpose investment-return panel.
#

# %%
macro.tail()

# %% [markdown]
# **Output interpretation.**
#
# The macro tail is a processed month-end intersection for the publication
# case. It is complete over the stored common sample, but its latest-vintage
# reference periods are not historical release dates and therefore do not
# support a point-in-time backtest.
#

# %%
nasdaq_prices.tail()

# %% [markdown]
# **Output interpretation.**
#
# This separate adjusted-close panel is the equity surface passed to lesson
# `1.4`. It is not merged with the Mexican level panel, whose columns represent
# different economic objects and calendars.

# %%
display(data_quality_report(levels, expected_frequency="B"))
display(data_quality_report(nasdaq_prices, expected_frequency="B"))

# %% [markdown]
# **Output interpretation.**
#
# The quality report is the gate between extraction and analysis. Stored nulls,
# duplicate dates, and dates absent from a Monday-to-Friday benchmark are
# reported separately. An absent weekday can be a Mexican holiday or a genuine
# source gap, so it triggers calendar/source review rather than automatic filling.
#

# %% [markdown]
# ## Live-data extension
#
# Live extraction stays behind an explicit environment switch. Run it locally only when credentials, network access, and rate limits are approved for class.

# %% tags=["live-data"]
if DATA_MODE == "live":
    live_prices = live_nasdaq_stock_prices(start=ANALYSIS_START, end=ANALYSIS_END)
    live_macro = live_macro_dashboard_panel(start=ANALYSIS_START, end=ANALYSIS_END)
else:
    live_prices = nasdaq_prices
    live_macro = macro

live_prices.tail()

# %% [markdown]
# **Output interpretation.**
#
# The price tail checks that the extraction path returns an analysis-ready date-by-asset panel. In offline mode it confirms the committed snapshot shape; in live mode it confirms provider access before any modeling code uses the data.

# %%
live_macro.tail()

# %% [markdown]
# **Output interpretation.**
#
# The switch preserves the same schemas and meanings as the snapshot path:
# NASDAQ adjusted closes and the complete macro-dashboard panel. In live mode,
# row coverage can still differ because provider calendars, revisions, and
# availability change; refresh metadata must therefore travel with the data.

# %% [markdown]
# ## Extraction checklist
#
# | Question | Why it matters |
# | --- | --- |
# | Is the provider official, commercial, open, or unofficial? | The answer affects reliability, legal use, and reproducibility. |
# | What field was used? | Close, adjusted close, settlement, yield, and index levels imply different transformations. |
# | What calendar does the source follow? | Calendar mismatches change missingness, correlations, and volatility. |
# | Are credentials or rate limits involved? | Hidden credentials and unstable limits make notebooks hard to reproduce. |
# | Was the raw extract cached? | Cached data gives students a stable audit trail before modeling. |

# %% [markdown]
# ## Handoff
#
# Preserve the source inventory, actual panel coverage, data mode, and initial
# quality report. Send the Mexican levels, NASDAQ prices, and macro panel first
# to the quality gate in `1.6`. After
# calendar, missingness, field, and transformation policies are accepted, route
# the NASDAQ adjusted-close panel to `1.4` and the macro panel to `1.5`.
# Neither branch may change the common sample, source meaning, or provider-rights
# note silently.

# %% [markdown] tags=["exercise"]
# ## Checkpoint exercise
#
# Build a source decision memo for one Mexican series and one NASDAQ equity in
# the common teaching window.
#
# 1. For each variable, name the provider family, field, unit, calendar,
#    authority level, and known limitation.
# 2. Use the panel metadata and quality report to state the actual first date,
#    last date, row count, and missing count.
# 3. Decide whether the source is suitable for an offline classroom chart, a
#    live demonstration, and a trading or institutional claim; justify each
#    decision without assuming that repository inclusion grants data rights.
# 4. State which raw response, cache record, metadata field, and processed
#    output another analyst would need to reproduce the extraction.
#

# %% [markdown] tags=["solution"]
# ```{dropdown} Suggested answer rubric
# A complete memo separates authority from convenience, reports observed
# coverage and missingness, keeps usage-rights claims limited to reviewed
# evidence, and traces both variables from provider metadata to the processed
# panel used by the lesson. Reconcile its dates, row counts, and missing counts
# with the executed metadata and quality-report tables; those outputs are the
# snapshot-specific self-check.
# ```
