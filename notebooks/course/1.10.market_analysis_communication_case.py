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
# # Communication and Integrated Case
#
# Module: Markets, Instruments, and Data
#
# ## Lesson summary
#
# This capstone notebook closes the module by turning accepted data, exploratory
# statistics, dashboards, and basic risk summaries into clear communication. The
# purpose is not advanced app development, production deployment, forecasting,
# portfolio optimization, or institutional reporting. It is to make a clearly
# scoped market analysis understandable, reproducible, and decision-relevant.
#
# Two data contexts appear in the module and they serve different roles. The
# NASDAQ panel supports the earlier asset-level EDA and return-explorer lessons.
# This final case has one scope only: a Mexican macro-financial panel built from
# official-source observations, documented synthetic carry indexes, and
# latest-vintage macro context. It does not merge NASDAQ observations into its
# statistics, charts, or claims.
#
# The central rule is:
#
# ```text
# Clean data are not enough.
# Good analysis is not enough.
# The result must be understandable, reproducible, and decision-relevant.
# ```
#
# ## Learning objectives
#
# By the end of this capstone, students should be able to:
#
# - design exploratory dashboards around analytical questions;
# - distinguish offline, live-data, and hybrid dashboards;
# - choose charts that match financial questions;
# - include data source, update, and quality notes inside dashboard outputs;
# - turn charts and metrics into concise written insights;
# - explain assumptions and limitations without overclaiming;
# - integrate the full module workflow into one reproducible case without mixing incompatible market scopes.
#
# ## Prerequisites
#
# Complete all earlier Module 1 lessons. The Mexican panel's provenance, source
# inventory, quality report, carry-index convention, common sample, macro
# transformation rules, and observed-interval log-change convention must already
# be clear.
#
# ## Exploratory dashboard design
#
# A dashboard is not just a collection of charts. A good dashboard organizes information around questions, decisions, and interpretation. It should help the reader understand what is happening, compare variables, identify patterns, and decide what deserves deeper analysis {cite}`few2006dashboard,cairo2016truthful`.
#
# For this module, dashboards are exploratory. They should help students interact with data and communicate patterns, but they should not perform portfolio optimization, full risk modeling, derivative pricing, bond valuation, forecasting, strategy backtesting, or production monitoring.
#
# The following map applies across the earlier asset and macro dashboards. The
# final Mexican case uses only the rows and labels appropriate to each economic
# object.
#
# | Area | Dashboard content |
# | --- | --- |
# | Market levels | Prices, indices, exchange rates, rates, macro series |
# | Eligible asset returns | Simple returns, log returns, cumulative adjusted-close returns |
# | Non-contractual levels | Level paths, log changes, and cumulative log changes |
# | Risk and path diagnostics | Asset-return volatility and VaR where valid; level-path drawdown, log-change dispersion, and downside percentiles otherwise |
# | Dependence | Correlations, heatmaps, scatterplots |
# | Macro context | Inflation, policy rate, exchange rate, activity indicators |
# | Data quality | Missing values, stale prices, source notes, assumptions |
#
# ## Offline and live-data modes
#
# Students should distinguish three dashboard modes:
#
# | Version | Purpose | Main risk |
# | --- | --- | --- |
# | Offline dashboard | Reproducible analysis using validated data | May be less current |
# | Live-data dashboard | Demonstration of updateable workflows | APIs, revisions, limits, and source drift |
# | Hybrid dashboard | Cached data by default, refresh only when requested | Refresh rules must be explicit |
#
# The default version for this module should be offline. Live data can be introduced only after students understand source quality, cache, and reproducibility.
#
# ## Dashboard architecture
#
# A simple architecture keeps data logic outside the visual interface:
#
# ```text
# Raw data
#   -> validated processed data
#   -> return, level-change, and metric calculations
#   -> dashboard-ready tables
#   -> interactive visual interface
#   -> interpretation and written insight
# ```
#
# The dashboard should not hide extraction, cleaning, or validation. Those steps belong in `src/`, processed datasets, and quality reports.
#
# ## Recommended dashboard sections
#
# | Section | Suggested content | Purpose |
# | --- | --- | --- |
# | Market overview | selected series, observed or constructed level, cumulative path or change, macro context, source note | Give context before statistics |
# | Changes and distributions | eligible asset returns or labeled level changes, histogram, descriptive statistics, rolling mean, and either return volatility or explicitly labeled level-change dispersion | Show behavior across time and distribution |
# | Correlations and relationships | correlation matrix, heatmap, scatterplot, rolling correlation, documented comparison reference | Show co-movement without implying causality |
# | Risk summary | cumulative adjusted-close return or cumulative level change, drawdown or peak-to-trough level decline, appropriately labeled dispersion, and downside percentile; VaR and Sharpe only for eligible return contracts | Connect each transformation with an appropriately limited risk reading |
# | Data notes | source, date range, last update, missingness, adjusted-price policy, assumptions | Make the dashboard auditable |
#
# Each chart should answer a clear question. Use line charts for time paths, histograms for distributions, heatmaps for many pairwise relationships, scatterplots for two-variable relationships, and drawdown charts for peak-to-trough losses. Visual design should prioritize clarity, comparison, and truthful representation rather than decoration {cite}`tufte2001visual,cairo2016truthful,wilke2019dataviz`.
#
# Interactive controls should help interpretation, not distract. Useful controls include series selection, date range, frequency, documented comparison reference, return type, rolling window, and macro variable. Tools such as Plotly support hover labels, zoom, and interactive controls, but interactivity should remain tied to an analytical purpose {cite}`plotlyPythonDocs`.
#
# ## From chart to insight
#
# A chart does not automatically communicate an insight. A useful financial narrative connects data, context, interpretation, and limitations {cite}`knaflic2015storytelling`.
#
# An observation describes what is visible. An insight explains why it matters.
#
# | Observation | Insight |
# | --- | --- |
# | Volatility increased in March | Average return is not enough because the asset became riskier during the stress period |
# | USD/MXN rose, measured as MXN per USD | USD appreciated and MXN depreciated; the portfolio effect depends on the exposure and reporting currency |
# | Two assets have high correlation | Diversification between them may be weaker than expected |
# | Drawdown reached -20% | The investor experienced a severe peak-to-trough loss despite the final return |
# | Inflation and rates rose together | The co-movement motivates a testable policy or discount-rate hypothesis but does not establish causality |
#
# Use this sentence structure:
#
# ```text
# [Metric or pattern] shows that [interpretation], which matters because [financial implication].
# ```
#
# Example:
#
# ```text
# For an eligible asset, the cumulative adjusted-close return chart shows a
# positive ending value, but the drawdown chart reveals a deep interim loss; the
# path therefore carried risk that the endpoint alone does not show.
# ```
#
# ## Narrative structure
#
# A short financial analysis should follow a disciplined structure:
#
# ```text
# 1. Question
# 2. Data
# 3. Method
# 4. Finding
# 5. Interpretation
# 6. Limitation
# 7. Next step
# ```
#
# Students should separate description, interpretation, hypothesis, and causal claim.
#
# | Statement type | Example |
# | --- | --- |
# | Description | Returns were more volatile in the second half of the sample |
# | Interpretation | The higher volatility suggests a less stable market environment |
# | Hypothesis | The change may be related to monetary policy expectations |
# | Causal claim | Monetary policy caused the increase in volatility |
#
# Exploratory analysis can support descriptions, interpretations, and hypotheses. It should not make strong causal claims without deeper analysis.
#
# ## Integrated case: Mexican market context
#
# The earlier NASDAQ lessons remain a separate asset-level example. This closing
# case deliberately uses only the Mexican official-source observations,
# synthetic carry indexes, latest-vintage macro context, and the common
# 2021-01-01 through 2025-06-30 window. Its single research question is:
#
# ```text
# How did USD/MXN, UDI, and the synthetic Mexican rate-carry indexes differ in
# cumulative level change, observed-interval log-change dispersion,
# peak-to-trough level decline, and co-movement, and what rate and inflation
# context accompanied those patterns?
# ```
#
# The case is descriptive. It does not treat macro co-movement as causality or
# the largest cumulative change as an investment recommendation.
#
# The final case should demonstrate that financial analysis is a pipeline, not a single calculation.
#
# ```text
# Question
#   -> market context
#   -> data inventory
#   -> extraction and storage
#   -> validation and cleaning
#   -> economically labeled transformations and panels
#   -> exploratory statistics
#   -> risk summary
#   -> dashboard
#   -> narrative insight
# ```
#
# A strong case theme for this module is:
#
# ```text
# Mexican macro-financial dashboard: exchange-rate and inflation-linked levels,
# documented synthetic rate-carry paths, and latest-vintage rate and inflation
# context.
# ```
#
# ## Case inputs and deliverables
#
# | Category | Case variable | Purpose |
# | --- | --- | --- |
# | Exchange rate | Banxico USD/MXN FIX level | Currency context |
# | Inflation-linked level | Banxico UDI | Inflation-linked purchasing-power context |
# | Synthetic carry | CETES 28d, TIIE 28d, and policy-rate carry indexes | Comparable classroom accrual paths |
# | Macro context | Target rate, CETES 28d rate, TIIE 28d rate, and Mexican inflation | Rate and inflation regimes |
# | Comparison reference | CETES 28d carry index | Documented short-rate classroom reference, not a broad market benchmark |
#
# The final case should include:
#
# ```text
# 1. Data inventory.
# 2. Source notes.
# 3. Raw-data storage structure.
# 4. Processed dataset.
# 5. Data quality report.
# 6. Log-change panel with economic-object labels.
# 7. Descriptive statistics table.
# 8. Correlation matrix and heatmap.
# 9. Drawdown analysis.
# 10. Exploratory dashboard.
# 11. Written narrative.
# 12. Reproducibility notes.
# ```
#
# ## Minimum data inventory
#
# | Variable | Source | Frequency | Unit | Transformation | Limitation |
# | --- | --- | ---: | --- | --- | --- |
# | USD/MXN FIX | Banxico `SF43718` | Publication/business-day panel | MXN per USD | Log change and cumulative change | FIX/source calendar |
# | UDI | Banxico `SP68257` | Publication/business-day panel | MXN per UDI | Log change and cumulative change | Inflation-linked unit, not a traded asset |
# | CETES carry | Derived from Banxico `SF60633` | Daily internal accrual sampled on joint observed-level dates | Index, base near 100 | \(I_d=I_{d-1}[1+(y_{d-1}/100)/360]\) | Synthetic; previous known rate, not an observed price |
# | TIIE carry | Derived from Banxico `SF60648` | Daily internal accrual sampled on joint observed-level dates | Index, base near 100 | Same daily-factor and 360-day convention | Synthetic; underlying TIIE methodology changes on 2025-01-01 |
# | Policy-rate carry | Derived from Banxico `SF61745` | Daily internal accrual sampled on joint observed-level dates | Index, base near 100 | Same daily-factor and 360-day convention | Synthetic, not investable |
# | Mexican inflation | DB.NOMICS/IMF snapshot | Monthly | Decimal year-over-year rate, displayed as percent | CPI year-over-year rate; no second differencing in this case | Latest-vintage/revision risk |
#
# Banco de México changed the 28-day TIIE methodology effective 2025-01-01
# from submitted bank quotes to a market-transaction-based method. The synthetic
# carry path preserves that breakpoint; it does not make the full history
# methodologically homogeneous {cite}`banxicoTIIETransition2025`.
#
# ## Quality and transformation requirements
#
# A minimum data quality report should answer:
#
# ```text
# 1. Which sources were used?
# 2. What sample period was selected?
# 3. Which variables were downloaded?
# 4. Were there missing dates?
# 5. Were there missing values?
# 6. Were there duplicated observations?
# 7. Were there extreme values?
# 8. Which series are observed, provider-adjusted, or constructed?
# 9. How were frequencies aligned?
# 10. What assumptions were made?
# ```
#
# The case includes date parsing, numeric validation, explicit frequency
# separation, level-to-log-change conversion, cumulative level change, rolling
# observed-interval dispersion, level drawdown, correlation, and a contextual—not return-level—macro
# comparison.
#
# ## Required outputs
#
# The integrated case should include at least these outputs:
#
# ```text
# 1. Line chart of selected observed and constructed level paths.
# 2. Cumulative log-change or rebased-level chart for selected series.
# 3. Histogram of observed-interval log changes for one selected series.
# 4. Rolling observed-interval log-change dispersion chart.
# 5. Correlation heatmap.
# 6. Scatterplot of two selected variables.
# 7. Drawdown chart.
# 8. Macro context chart.
# 9. Dashboard summary table.
# 10. Final written insight.
# ```
#
# The written narrative should include research question, data sources, sample period, transformations, main descriptive finding, main risk finding, main relationship finding, macro context, data limitation, and next analytical step.
#
# ## Executable case skeleton
#
# The communication layer is backed by official-source Mexican snapshots,
# documented synthetic carry indexes, and derived publication panels. The
# executable case below produces every
# promised evidence layer: inventory, storage map, processed panels, quality
# report, log-change panel, statistics, correlation, drawdown, dashboard, narrative,
# and reproducibility notes.

# %% tags=["setup", "hide-input"]
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from src.market_data import (
    mexican_market_level_panel,
    official_macro_panel,
    returns_from_prices,
)
from src.market_data_quality import (
    annualized_log_change_from_levels,
    data_quality_report,
    hampel_outlier_flags,
)
from src.module1_visuals import (
    build_mexican_macro_context,
    build_mexican_market_dependence,
    build_mexican_market_paths,
    build_mexican_market_risk,
)
from src.visual_style import set_matplotlib_theme

ANALYSIS_START = "2021-01-01"
ANALYSIS_END = "2025-06-30"
CETES_REFERENCE = "cetes_28d_carry"
ROLLING_WINDOW = 63
CALENDAR_DAYS_PER_YEAR = 365.2425
set_matplotlib_theme()

# %%
case_inventory = pd.DataFrame(
    [
        {
            "series": "usd_mxn",
            "source": "Banxico SIE SF43718",
            "economic_object": "FIX exchange-rate level",
            "unit": "MXN per USD",
            "frequency": "joint observed USD/MXN and UDI dates",
            "transformation": "observed-interval log change",
            "limitation": "FIX calendar; not an investable asset return",
        },
        {
            "series": "udi",
            "source": "Banxico SIE SP68257",
            "economic_object": "inflation-linked unit level",
            "unit": "MXN per UDI",
            "frequency": "joint observed USD/MXN and UDI dates",
            "transformation": "observed-interval log change",
            "limitation": "not a traded total-return asset",
        },
        {
            "series": "cetes_28d_carry",
            "source": "derived from Banxico SIE SF60633",
            "economic_object": "synthetic 28-day CETES carry index",
            "unit": "index, base near 100",
            "frequency": "daily internal accrual sampled on joint observed dates",
            "transformation": "observed-interval log change",
            "limitation": "not an observed CETES price or investable benchmark",
        },
        {
            "series": "tiie_28d_carry",
            "source": "derived from Banxico SIE SF60648",
            "economic_object": "synthetic 28-day TIIE carry index",
            "unit": "index, base near 100",
            "frequency": "daily internal accrual sampled on joint observed dates",
            "transformation": "observed-interval log change",
            "limitation": "not an observed price; TIIE methodology breaks at 2025-01-01",
        },
        {
            "series": "policy_rate_carry",
            "source": "derived from Banxico SIE SF61745",
            "economic_object": "synthetic policy-rate carry index",
            "unit": "index, base near 100",
            "frequency": "daily internal accrual sampled on joint observed dates",
            "transformation": "observed-interval log change",
            "limitation": "policy rate is not directly investable",
        },
        {
            "series": "banxico_target_rate",
            "source": "Banxico SIE SF61745",
            "economic_object": "monetary-policy target-rate level",
            "unit": "decimal rate",
            "frequency": "monthly last observation",
            "transformation": "display as percent",
            "limitation": "event timing is compressed to month-end context",
        },
        {
            "series": "cetes_28d",
            "source": "Banxico SIE SF60633",
            "economic_object": "published 28-day CETES annualized rate",
            "unit": "decimal rate",
            "frequency": "monthly last observation",
            "transformation": "display as percent",
            "limitation": "rate level is not a bond total return",
        },
        {
            "series": "tiie_28d",
            "source": "Banxico SIE SF60648",
            "economic_object": "published 28-day TIIE annualized rate",
            "unit": "decimal rate",
            "frequency": "monthly last observation",
            "transformation": "display as percent",
            "limitation": "not an instrument price; methodology breaks at 2025-01-01",
        },
        {
            "series": "mexico_inflation",
            "source": "DB.NOMICS IMF CPI snapshot",
            "economic_object": "year-over-year inflation rate",
            "unit": "decimal rate",
            "frequency": "monthly",
            "transformation": "year-over-year CPI rate displayed as percent",
            "limitation": "latest-vintage values can be revised",
        },
    ]
).set_index("series")
case_inventory

# %% [markdown]
# **Output interpretation.**
#
# The inventory fixes the meaning of every case variable before statistics are
# computed. In particular, the three carry columns are transparent synthetic
# transformations, and CETES carry is only a short-rate classroom reference.

# %%
project_root = Path.cwd().resolve()
for candidate in (project_root, *project_root.parents):
    if (candidate / "pyproject.toml").exists():
        project_root = candidate
        break

storage_map = pd.DataFrame(
    [
        {"layer": "source snapshot", "path": "data/snapshots/banxico_daily.csv"},
        {"layer": "source snapshot", "path": "data/snapshots/dbnomics_daily.csv"},
        {"layer": "processed publication panel", "path": "data/snapshots/official_price_panel.csv"},
        {"layer": "processed publication panel", "path": "data/snapshots/official_macro_panel.csv"},
        {"layer": "metadata", "path": "data/snapshots/metadata.json"},
    ]
)
storage_map["exists"] = storage_map["path"].map(lambda path: (project_root / path).exists())
assert storage_map["exists"].all(), storage_map
storage_map

# %% [markdown]
# **Output interpretation.**
#
# This is a source-snapshot storage map, not a claim that provider responses are
# raw exchange records. It proves which committed inputs and metadata make the
# offline case traceable.

# %%
levels = mexican_market_level_panel(start=ANALYSIS_START, end=ANALYSIS_END)
macro = official_macro_panel(start=ANALYSIS_START, end=ANALYSIS_END)

processed_panel_evidence = pd.DataFrame(
    [
        {
            "panel": "joint-observation Mexican levels and synthetic carry",
            "start": levels.index.min(),
            "end": levels.index.max(),
            "rows": len(levels),
            "columns": ", ".join(levels.columns),
        },
        {
            "panel": "monthly Mexican macro context",
            "start": macro.index.min(),
            "end": macro.index.max(),
            "rows": len(macro),
            "columns": ", ".join(macro.columns),
        },
    ]
).set_index("panel")
processed_panel_evidence

# %% [markdown]
# **Output interpretation.**
#
# Both processed panels use the common requested window. Actual boundary
# timestamps differ because the level panel retains joint USD/MXN and UDI
# observation dates and the macro
# panel follows month-end reference-period timestamps; they are compared as context rather
# than merged mechanically at daily frequency.

# %%
quality_report = pd.concat(
    {
        "observed_level_panel": data_quality_report(
            levels,
            expected_frequency="B",
        ),
        "monthly_macro_panel": data_quality_report(macro, expected_frequency="ME"),
    },
    names=["panel", "series"],
)
quality_report

# %% [markdown]
# **Output interpretation.**
#
# Coverage and missing counts are reported per series and per frequency. The
# weekday benchmark can include Mexican holidays, so absent dates require
# calendar/source classification rather than automatic filling. A missing
# monthly value is not silently forward-filled into the observed-interval
# log-change panel.

# %%
log_changes = returns_from_prices(levels, method="log")
log_change_panel_evidence = pd.DataFrame(
    {
        "first_valid_log_change": log_changes.apply(lambda series: series.first_valid_index()),
        "last_valid_log_change": log_changes.apply(lambda series: series.last_valid_index()),
        "valid_log_changes": log_changes.count(),
        "missing_log_changes": len(levels) - log_changes.count(),
    }
)
log_change_panel_evidence

# %% [markdown]
# **Output interpretation.**
#
# These are log changes between consecutive stored observations. For synthetic
# carry, each change includes the documented calendar-day accrual since the
# previous stored date; for USD/MXN and UDI, it reflects movement in published
# levels. Intervals can span multiple days, and none is automatically a contract
# return.

# %%
extreme_change_review = pd.DataFrame(
    {
        "hampel_flag_count": log_changes.apply(hampel_outlier_flags).sum(),
        "minimum_log_change": log_changes.min(),
        "maximum_log_change": log_changes.max(),
    }
)
extreme_change_review

# %% [markdown]
# **Output interpretation.**
#
# The centered Hampel diagnostic flags locally unusual observed-interval log
# changes for review; it does not classify them automatically as errors. The
# extrema remain visible beside the flag count so a reviewer can trace each
# candidate back to its source date and economic context.

# %%
cumulative_level_changes = levels.divide(levels.iloc[0]).subtract(1)
level_drawdowns = levels.divide(levels.cummax()).subtract(1)
correlations = log_changes.corr()
rolling_log_change_dispersion = log_changes.rolling(ROLLING_WINDOW).std()
annualized_log_level_changes = levels.apply(
    annualized_log_change_from_levels,
    days_per_year=CALENDAR_DAYS_PER_YEAR,
)

case_summary = pd.DataFrame(
    {
        "cumulative_level_change": cumulative_level_changes.iloc[-1],
        "annualized_log_level_change": annualized_log_level_changes,
        "log_change_std_per_stored_interval": log_changes.std(),
        "p05_downside_log_change_magnitude": (-log_changes.quantile(0.05)).clip(lower=0),
        "maximum_level_drawdown": level_drawdowns.min(),
        "annualized_log_level_change_difference_vs_cetes_reference": (
            annualized_log_level_changes - annualized_log_level_changes[CETES_REFERENCE]
        ),
        "correlation_with_cetes_carry_log_change": log_changes.corrwith(
            log_changes[CETES_REFERENCE]
        ),
        "missing_level_observations": levels.isna().sum(),
        "missing_log_change_observations": len(levels) - log_changes.count(),
    }
)
display(
    case_summary[
        [
            "cumulative_level_change",
            "annualized_log_level_change",
            "maximum_level_drawdown",
        ]
    ]
)
display(
    case_summary[
        [
            "log_change_std_per_stored_interval",
            "p05_downside_log_change_magnitude",
            "correlation_with_cetes_carry_log_change",
        ]
    ]
)
display(
    case_summary[
        [
            "annualized_log_level_change_difference_vs_cetes_reference",
            "missing_level_observations",
            "missing_log_change_observations",
        ]
    ]
)

# %% [markdown]
# **Output interpretation.**
#
# The summary uses explicit labels. Annualized log level change is endpoint
# growth scaled by elapsed calendar time using 365.2425 days per year, not a
# promised return. The 5th-
# percentile downside magnitude and standard deviation describe heterogeneous
# stored intervals; they are not annualized volatility or VaR. CETES carry is a
# documented synthetic reference rather than a broad market benchmark.
#

# %%
scatter_data = log_changes[["usd_mxn", "udi"]].dropna()
macro_context = (
    macro[["banxico_target_rate", "cetes_28d", "tiie_28d", "mexico_inflation"]]
    .multiply(100)
    .rename(columns=lambda column: f"{column}_pct")
)
path_figure = build_mexican_market_paths(
    levels,
    cumulative_level_changes,
)
risk_figure = build_mexican_market_risk(
    log_changes,
    rolling_log_change_dispersion,
    scatter_data,
    rolling_window=ROLLING_WINDOW,
)
dependence_figure = build_mexican_market_dependence(
    correlations,
    level_drawdowns,
)
macro_figure = build_mexican_macro_context(
    macro_context,
)

plt.show()

# %% [markdown]
# **Output interpretation.**
#
# The four-block dashboard fulfills the same visual contract without forcing
# every question into one dense canvas. The path block distinguishes observed
# Banxico levels from constructed carry; distribution and dispersion retain
# question-specific scales; dependence fixes correlation colors to `[-1, 1]`
# and reports peak-to-trough level declines without calling them investor losses;
# and the macro block keeps
# latest-vintage monthly context separate from observed-interval log changes. These views
# support comparison and hypothesis generation, not causal attribution or an
# investment ranking.

# %%
largest_change_series = case_summary["cumulative_level_change"].idxmax()
deepest_drawdown_series = case_summary["maximum_level_drawdown"].idxmin()
most_disperse_series = case_summary["log_change_std_per_stored_interval"].idxmax()
usd_udi_correlation = correlations.loc["usd_mxn", "udi"]
latest_macro = macro_context.dropna().iloc[-1]

insight_table = pd.DataFrame(
    [
        {
            "narrative_element": "research question",
            "evidence": (
                "Compare cumulative level change, path diagnostics, and "
                "co-movement in the versioned Mexican panel."
            ),
            "interpretation": "All evidence uses one scope and one common requested window.",
        },
        {
            "narrative_element": "descriptive finding",
            "evidence": (
                f"{largest_change_series} has the largest cumulative change "
                f"({case_summary.loc[largest_change_series, 'cumulative_level_change']:.2%})."
            ),
            "interpretation": "This compares level paths; it is not an investment-performance ranking.",
        },
        {
            "narrative_element": "risk finding",
            "evidence": (
                f"{deepest_drawdown_series} has the deepest drawdown "
                f"({case_summary.loc[deepest_drawdown_series, 'maximum_level_drawdown']:.2%}), "
                f"while {most_disperse_series} has the highest log-change "
                "standard deviation per stored interval."
            ),
            "interpretation": (
                "Level-path decline and interval dispersion qualify the endpoint "
                "comparison without turning it into an investment ranking."
            ),
        },
        {
            "narrative_element": "relationship finding",
            "evidence": (
                "USD/MXN and UDI observed-interval log changes have correlation "
                f"{usd_udi_correlation:.3f}."
            ),
            "interpretation": "Average linear co-movement is descriptive and can change by regime.",
        },
        {
            "narrative_element": "macro context",
            "evidence": (
                f"At the latest complete monthly row, the target rate is "
                f"{latest_macro['banxico_target_rate_pct']:.2f}% and inflation is "
                f"{latest_macro['mexico_inflation_pct']:.2f}%."
            ),
            "interpretation": "The values provide context but do not identify a causal driver.",
        },
        {
            "narrative_element": "data limitation",
            "evidence": (
                f"Offline joint-observation levels cover {levels.index.min().date()} to "
                f"{levels.index.max().date()}; macro data are latest-vintage monthly observations."
            ),
            "interpretation": (
                "Carry indexes compound synthetic daily factors and are sampled on "
                "joint observed dates, and macro releases "
                "are not aligned to historical availability dates."
            ),
        },
        {
            "narrative_element": "next step",
            "evidence": "Validate release vintages and instrument-level total-return data.",
            "interpretation": "Those additions are required before causal, tradable, or portfolio conclusions.",
        },
    ]
)
insight_table

# %% [markdown]
# **Output interpretation.**
#
# The narrative is generated from the executed case, so its quantitative claims
# remain synchronized with the tables and figures. It answers the stated
# question while preserving the boundary between evidence, interpretation,
# limitation, and next step.
#

# %% [markdown]
# ## Reproducibility notes
#
# The case should be traceable from final chart back to original data source. A minimum project structure is:
#
# ```text
# market_and_data_case/
#   README.md
#   data/
#     source_snapshots/
#     processed/
#     metadata/
#   notebooks/
#     01_data_sources.ipynb
#     02_data_quality.ipynb
#     03_changes_and_analysis.ipynb
#     04_dashboard_outputs.ipynb
#   reports/
#     final_narrative.md
#     figures/
#     quality_report.md
#   references.bib
# ```
#
# The README should explain the project objective, data sources, reproduction steps, required packages, file structure, main outputs, and known limitations. Jupyter Book and MyST Markdown support reproducible analytical publishing with notebooks, Markdown, equations, figures, citations, and cross-references {cite}`jupyterbook2025,mystDocs`.
#
# ## Analysis quality checklist
#
# | Area | Publication-ready evidence |
# | --- | --- |
# | Data inventory | Sources, metadata, frequency, units, coverage, and licensing notes are explicit. |
# | Data quality report | Missing values, calendars, outliers, revisions, and assumptions are visible. |
# | Return, change, and panel construction | Economic-object labels, transformations, alignment rules, and comparison-reference choices are reproducible. |
# | Exploratory statistics | Descriptive metrics are interpreted with scale, sample period, and limitations. |
# | Visualizations | Charts are labeled, relevant, source-aware, and not overloaded. |
# | Dashboard | Controls, offline/live mode, source notes, and quality warnings are clear. |
# | Narrative | Main insight, uncertainty, limitations, and next analytical step are stated. |
#
# ## Common mistakes
#
# | Mistake | Why it matters |
# | --- | --- |
# | Starting with charts before defining the question | Produces unfocused analysis |
# | Downloading data without a source inventory | Weakens reproducibility |
# | Mixing daily and monthly data without rules | Creates misleading relationships |
# | Comparing differently scaled raw levels as performance | Confuses scale with cumulative change or return |
# | Ignoring dividends or distributions for eligible assets | Understates a distribution-adjusted return |
# | Treating correlation as causality | Overclaims evidence |
# | Omitting data quality issues | Hides analytical risk |
# | Showing dashboards without written interpretation | Leaves readers without insight |
# | Reporting VaR as maximum loss | Misinterprets the metric |
# | Calling an unlabeled proxy a benchmark | Hides whether the reference is broad, investable, synthetic, or fit for the decision |
#
# ## Handoff
#
# The full module now follows this logic:
#
# ```text
# Part I: Understand the market.
# Part II: Build reliable data.
# Part III: Explore returns, level changes, relationships, and risk.
# Part IV: Communicate the analysis and integrate the workflow.
# ```
#
# The final message is:
#
# ```text
# Market analysis is not only about finding numbers.
# It is about understanding where the numbers come from,
# what they mean,
# how reliable they are,
# and how to communicate them responsibly.
# ```
#
# The accepted observed-interval log-change panel can now move to Module 2 for stationarity,
# dependence, and volatility diagnostics. Carry forward the common window,
# variable meanings, synthetic-carry convention, missing counts, and source
# notes; the monthly macro panel remains contextual unless release-date
# alignment is added.
