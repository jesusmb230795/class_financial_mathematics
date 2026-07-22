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
# # Mexican Market Data Pipeline
#
# Module: Markets, Instruments, and Data
#
# ## Lesson summary
#
# This lab turns the Module 1 market-data framework into a practical data
# pipeline. Students define a data inventory, route provider access through
# reusable clients, build a clean level matrix, compute economically labeled
# returns or log changes, and audit quality before modeling
# {cite}`fabozzi2019foundations,tsay2010analysis`.
#
# The Mexican market is a useful teaching case because it combines exchange rates, inflation, interest rates, government securities, equity indices, ETFs, FIBRAs, and public market proxies. The purpose is not to build a complete institutional database. The purpose is to create a disciplined educational pipeline that can be reproduced, audited, and extended.
#
# The notebook has one canonical publication path and one opt-in provider check:
#
# - classroom mode with versioned official-source snapshots and documented
#   constructed series, requiring no credentials or network;
# - a live Banxico request demonstration, where students can verify the official
#   series endpoint locally before building a separately reviewed refresh
#   pipeline {cite}`banxicoSIE2025`.
#
# ## Learning objectives
#
# By the end of this lab, students should be able to:
#
# - document market data sources before modeling;
# - design a Mexican market source inventory;
# - separate raw extraction, standardization, validation, and analysis-ready outputs;
# - configure a Banxico SIE request through the shared data client without exposing credentials;
# - understand why adjusted prices matter for equity return calculations;
# - declare an observation-calendar policy without silently filling source levels;
# - compute simple or log returns for appropriate price series, log changes for
#   positive non-contractual levels, endpoint annualized level change,
#   per-observation dispersion, and missingness reports;
# - separate raw data, clean data, modeling data, metadata, and quality reports.
#
# ## Prerequisites
#
# Complete the source-selection and quality lessons (`1.3` and `1.6`). Readers
# should be able to distinguish an observed level from a derived index and must
# retain units, calendars, missingness, and transformation metadata.
#
# ## Setup

# %% tags=["setup", "hide-input"]
import os

import pandas as pd

from src.market_data import MarketDataClient, mexican_market_level_panel
from src.market_data_quality import (
    annualized_log_change_from_levels,
    banxico_series_catalog,
    data_quality_report,
    log_returns,
    source_inventory_template,
)

DATA_MODE = os.getenv("DATA_MODE", "offline")
ANALYSIS_START = "2021-01-01"
ANALYSIS_END = "2025-06-30"
CALENDAR_DAYS_PER_YEAR = 365.2425

# %% [markdown]
# ## Suggested Mexican market dataset
#
# A first version of the Mexican market pipeline can include:
#
# | Variable | Example source | Frequency | Use |
# | --- | --- | ---: | --- |
# | USD/MXN exchange rate | Banxico | Daily | Currency context; quoted as MXN per USD |
# | Overnight policy rate | Banxico | Daily or event-based | Monetary policy |
# | CETES rate | Banxico or Cetesdirecto reference | Daily or weekly | Short-term reference rate |
# | Inflation | INEGI | Monthly | Inflation adjustment and macro context |
# | Economic activity | INEGI | Monthly | Growth proxy |
# | S&P/BMV IPC | BMV, vendor, or public source | Daily | Equity market proxy |
# | FIBRA index or sample FIBRAs | BMV, vendor, or public source | Daily | Real estate vehicle proxy |
# | Selected ETFs | BMV, vendor, or public source | Daily | Listed portfolio exposure |
#
# Banxico SIE is appropriate for central bank and financial time series, INEGI is appropriate for national statistical indicators, and BMV data products are appropriate for official market data from the Mexican exchange ecosystem {cite}`banxicoSIE2025,inegiAPI2025,bmvMarketData`.
#
# ## Source inventory
#
# A data inventory is part of the analytical documentation, not an optional appendix. It should define what each series is, where it comes from, and how it will be used.

# %%
inventory = source_inventory_template()
inventory.loc[0] = {
    "provider": "Yahoo Finance",
    "instrument_or_variable": "^MXX, AMX.MX, WALMEX.MX",
    "frequency": "daily",
    "start": ANALYSIS_START,
    "end": ANALYSIS_END,
    "field": "adjusted close",
    "currency": "MXN",
    "calendar": "trading days",
    "known_limitations": "unofficial access path, rate limits, and intended-use terms that require separate review",
}
inventory.loc[1] = {
    "provider": "Banxico SIE",
    "instrument_or_variable": "SF43718, SF60633, SF60648, SF61745, SP68257",
    "frequency": "daily or auction frequency",
    "start": ANALYSIS_START,
    "end": ANALYSIS_END,
    "field": "published value",
    "currency": "percent, MXN per USD, or MXN per UDI",
    "calendar": "Mexican publication calendar",
    "known_limitations": "requires token and has missing dates",
}
inventory.loc[2] = {
    "provider": "INEGI",
    "instrument_or_variable": "Inflation or economic activity indicator",
    "frequency": "monthly",
    "start": ANALYSIS_START,
    "end": ANALYSIS_END,
    "field": "published value or index level",
    "currency": "percent, index, or level",
    "calendar": "Mexican statistical publication calendar",
    "known_limitations": "publication lags, revisions, and unit conventions",
}
inventory.loc[3] = {
    "provider": "BMV or licensed market data source",
    "instrument_or_variable": "S&P/BMV IPC, selected ETFs, FIBRAs",
    "frequency": "daily",
    "start": ANALYSIS_START,
    "end": ANALYSIS_END,
    "field": "price, index level, or volume",
    "currency": "MXN",
    "calendar": "Mexican trading calendar",
    "known_limitations": "licensing and official data-product access",
}
inventory

# %% [markdown]
# **Output interpretation.**
#
# The inventory output documents the intended source role for the Mexican market pipeline. Read it as a reproducibility checklist: provider, series definition, access method, and limitations should be known before extraction.
#

# %% [markdown]
# ## Pipeline stages
#
# A simple Mexican market pipeline can be organized into six stages:
#
# ```text
# 1. Define series inventory.
# 2. Extract raw data from each source.
# 3. Store raw data and metadata.
# 4. Clean and standardize each series.
# 5. Align dates and frequencies.
# 6. Export analysis-ready panels.
# ```
#
# Each stage should produce files that can be inspected independently.
#
# ## Raw extraction and standardization
#
# Raw extraction should preserve the original response whenever possible:
#
# ```text
# data/raw/banxico/usd_mxn_YYYYMMDD.json
# data/raw/banxico/cetes_28d_YYYYMMDD.json
# data/raw/inegi/inflation_YYYYMMDD.json
# data/raw/dbnomics/us_policy_rate_YYYYMMDD.json
# data/raw/yfinance/ipc_proxy_YYYYMMDD.csv
# ```
#
# Raw files should not be edited manually. If a value appears wrong, the correction belongs in a transformation step and should be documented.
#
# After extraction, each series should be standardized into a common structure:
#
# | Column | Description |
# | --- | --- |
# | `date` | Observation date |
# | `value` | Numeric value |
# | `series_id` | Internal series identifier |
# | `source` | Provider name |
# | `provider_id` | Original provider identifier |
# | `frequency` | Daily, monthly, quarterly |
# | `unit` | Percent, MXN per USD, index level |
# | `retrieval_date` | Date when data were downloaded |
# | `quality_flag` | Optional validation flag |
#
# This schema makes it easier to combine Banxico, INEGI, DB.NOMICS, BMV, and public market sources.
#
# ## Date alignment for Mexico
#
# The pipeline should use explicit date rules:
#
# - daily market data use trading dates;
# - exchange rates use the source observation date;
# - monthly inflation is assigned to the official observation period;
# - dashboard views may use month-end alignment;
# - historical simulations should use publication dates when avoiding look-ahead bias;
# - missing values should not be forward-filled unless the variable is economically persistent and the assumption is documented.
#
# The same source data can support different alignment choices, but each output should state which rule was used.
#
# ## Banxico catalog

# %%
banxico_series_catalog()

# %% [markdown]
# **Output interpretation.**
#
# The Banxico catalog output identifies the official series that can be requested live. Series identifiers are part of the analytical contract, because changing an ID can change the definition, frequency, or unit of the data.
#

# %% [markdown]
# ## Banxico request pattern
#
# Provider-specific API logic belongs in `src`, not inside the notebook. The cell below shows the live-data handoff through `MarketDataClient`; it only runs when `DATA_MODE=live` is set locally and `BANXICO_TOKEN` is configured {cite}`banxicoSIE2025`.

# %% tags=["live-data"]
if DATA_MODE == "live":
    client = MarketDataClient()
    banxico_panel = client.banxico.fetch_series_group(
        ["SF43718", "SF60633", "SF60648", "SF61745", "SP68257"],
        start=ANALYSIS_START,
        end=ANALYSIS_END,
    )
else:
    banxico_panel = pd.DataFrame()

banxico_panel.tail()

# %% [markdown]
# **Output interpretation.**
#
# This live-only Banxico panel is intentionally empty in the offline publication build. When `DATA_MODE=live`, the tail should be used as a provider smoke test; values still need the same date, unit, and missingness checks as the committed snapshot.

# %% [markdown]
# ## Official-source levels and synthetic carry indexes
#
# The course build reads a versioned Banxico snapshot, so diagnostics use
# official observed levels and rates without depending on live APIs during
# publication. The matrix also contains derived classroom series; it is not an
# official Banxico price or total-return product.
#
# The panel contains two official observed levels and three **synthetic
# classroom carry indexes**:
#
# | Column | Meaning | Unit / convention |
# | --- | --- | --- |
# | `usd_mxn` | Banxico FIX exchange-rate level | MXN per USD |
# | `udi` | Banxico UDI level | MXN per UDI |
# | `cetes_28d_carry` | Accumulated index from the published 28-day CETES annualized rate | index, base near 100 |
# | `tiie_28d_carry` | Accumulated index from the published 28-day TIIE annualized rate | index, base near 100 |
# | `policy_rate_carry` | Accumulated index from the policy-rate level | index, base near 100 |
#
# Effective 2025-01-01, Banco de México changed the 28-day TIIE methodology
# from submitted bank quotes to a market-transaction-based method. The carry
# construction preserves the published observations but does not remove this
# measurement break. Any result spanning the date must disclose it and avoid
# describing the pre- and post-change history as homogeneous
# {cite}`banxicoTIIETransition2025`.
#
# A quoted annual rate is carried on an internal calendar with one factor for
# every calendar day \(d\). Each factor uses the rate known on the previous day:
#
# $$
# I_d = I_{d-1}\left(1 + \frac{y_{d-1}/100}{360}\right),
# \qquad I_0=100.
# $$
#
# Between two stored rows, the index is the product of all intervening daily
# factors, so it captures both weekends and any rate publication inside the
# interval. The last published rate is used until a new rate becomes available.
# Stored rows retain dates where both USD/MXN and UDI were observed; neither
# observed level is forward-filled.
# These indexes are pedagogical transformations for comparable log-change
# examples, not observed bond prices, total-return indexes, or investable
# strategies.

# %%
levels = mexican_market_level_panel(start=ANALYSIS_START, end=ANALYSIS_END)
levels.head()


# %% [markdown]
# **Output interpretation.**
#
# The head of the level matrix confirms the starting date, column names, and
# initial levels. This is the first check that the extraction layer combined the
# official-source observations and documented synthetic transformations into a
# rectangular panel suitable for downstream quality checks.
#

# %% [markdown]
# ## Quality report

# %%
data_quality_report(levels, expected_frequency="B")

# %% [markdown]
# **Output interpretation.**
#
# The quality report separates stored nulls from dates absent relative to a
# Monday-to-Friday benchmark. An absent weekday can reflect a Mexican holiday
# or a genuine source gap, so the diagnostic requires calendar/source review;
# it is not permission to synthesize or forward-fill an observation.
#

# %% [markdown]
# ## Quality checks for the Mexican panel
#
# The Mexican market panel should include automated checks:
#
# ```text
# 1. No duplicated date-series observations.
# 2. All values are numeric after parsing.
# 3. Units are documented.
# 4. Expected business-day gaps are reported separately from stored null values.
# 5. Monthly series have one observation per reference month.
# 6. Extreme returns or level changes are flagged.
# 7. Exchange-rate values are positive.
# 8. Interest-rate values are within plausible bounds.
# 9. Inflation series does not mix index levels and percent changes.
# 10. Equity or ETF prices are checked for splits, dividends, and stale values.
# ```
#
# Quality checks should produce a report, not only a cleaned file.
#
# ## Observed-interval log changes

# %%
log_changes = log_returns(levels)

log_changes.head()

# %% [markdown]
# **Output interpretation.**
#
# The log-change matrix uses consecutive stored observations; intervals can span
# more than one calendar day around holidays or source gaps. The first row is
# missing because a change requires a prior level.
# For USD/MXN, UDI, and the synthetic carry indexes, these values describe
# changes in positive levels; they are not interchangeable contract returns.
#

# %%
summary = pd.DataFrame(
    {
        "valid_log_change_observations": log_changes.count(),
        "missing_level_observations": levels.isna().sum(),
        "missing_log_change_observations": len(levels) - log_changes.count(),
        "annualized_log_level_change": levels.apply(
            annualized_log_change_from_levels,
            days_per_year=CALENDAR_DAYS_PER_YEAR,
        ),
        "log_change_std_per_stored_interval": log_changes.std(),
        "minimum_log_change": log_changes.min(),
        "maximum_log_change": log_changes.max(),
    }
)
summary

# %% [markdown]
# **Output interpretation.**
#
# The summary table turns the observed-interval log-change matrix into audit metrics.
# `annualized_log_level_change` is endpoint log growth scaled by elapsed
# calendar time using 365.2425 days per year; for non-contractual levels it must
# not be described as an
# investable or promised return. The dispersion statistic is per stored
# observation interval and is deliberately not annualized. Read both with the
# missing-count fields to decide whether the panel is ready for modeling or
# needs source-level review.
#

# %% [markdown]
# ## Output datasets
#
# The pipeline can export several datasets depending on the analysis.
#
# | Output | Purpose |
# | --- | --- |
# | `mx_macro_long.parquet` | Long-format macro dataset |
# | `mx_market_levels_long.parquet` | Long-format observed and constructed levels |
# | `mx_log_changes_wide.parquet` | Wide-format log changes with economic-object metadata |
# | `mx_monthly_panel.parquet` | Month-end macro-market panel |
# | `quality_report.md` | Data quality summary |
# | `data_dictionary.yml` | Metadata and definitions |
#
# This separation prevents one dataset from trying to solve every analytical need.
#
# ## Educational workflow
#
# A complete educational workflow can follow this sequence:
#
# ```text
# 1. Download the USD/MXN exchange rate from Banxico.
# 2. Download inflation from INEGI.
# 3. Document a market indicator or optional live public-market proxy.
# 4. Store raw files locally.
# 5. Standardize each series to date-value-source format.
# 6. Validate dates, units, missing values, and outliers.
# 7. Convert eligible asset prices into returns and other positive levels into explicitly labeled changes.
# 8. Align daily market data to monthly inflation.
# 9. Build a monthly panel.
# 10. Export a dashboard-ready dataset.
# ```
#
# The final dataset should be simple enough to inspect manually and structured enough to support automated analysis.
#
# ## Minimum documentation
#
# The pipeline should include:
#
# ```text
# README.md
# data_dictionary.yml
# series_inventory.csv
# assumptions_log.md
# quality_report.md
# source_notes.md
# ```
#
# The documentation should explain what data were used, why those sources were chosen, how the data were downloaded, which transformations were applied, how missing values were treated, and which limitations remain.
#
# ## Source-to-model checklist
#
# | Step | Question |
# | --- | --- |
# | Provider | Is the source official, commercial, open, or unofficial? |
# | Field | Is the value an adjusted close, close, fixing, rate, settlement, bid, ask, or mid? |
# | Calendar | Which holidays are represented or missing? |
# | Missingness | Are gaps isolated, structural, or provider failures? |
# | Outliers | Are flagged observations data errors or real stress events? |
# | Transformation | Are models using observed levels, synthetic indexes, simple returns, log returns, log changes, yields, or spreads? |
# | Audit trail | Can another analyst reproduce every cleaning decision? |

# %% [markdown]
# ## Handoff
#
# Pass the accepted Mexican level panel, log-change panel, source inventory,
# carry-index convention, and quality report to the macro dashboard and final
# case. The downstream analysis must keep synthetic carry separate from
# observed prices and must retain the common sample window.

# %% [markdown] tags=["exercise"]
# ## Checkpoint exercise
#
# Prepare the pipeline acceptance report for `usd_mxn`,
# `cetes_28d_carry`, and `udi`.
#
# 1. Record source, original meaning, unit, common-window coverage, missing level
#    count, and missing log-change count for each column.
# 2. Reproduce the first two CETES carry-index updates from the versioned rate
#    path and the stated daily-factor formula; reconcile them with the stored values.
# 3. Explain why the CETES carry index is a synthetic classroom reference and
#    not an observed CETES price or total-return benchmark.
# 4. State the calendar-alignment and forward-fill decisions that must accompany
#    the log-change panel into the integrated case.
#

# %% [markdown] tags=["solution"]
# ```{dropdown} Suggested answer rubric
# A complete report traces all three columns to Banxico, labels levels and
# synthetic transformations correctly, verifies the carry formula numerically,
# reports coverage and missingness, and carries the 360-day/calendar-day
# convention into the downstream case without calling the index investable. It
# must also mark 2025-01-01 as the 28-day TIIE methodology breakpoint and
# reconcile the first two CETES carry updates with the executed level table.
# ```
