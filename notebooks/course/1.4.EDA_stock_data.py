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
# # NASDAQ Stock Exploratory Analysis
#
# Module: Markets, Instruments, and Data
#
# ## Lesson summary
#
# This notebook turns a real NASDAQ stock panel into exploratory statistics, empirical return distributions, correlations, rolling diagnostics, and visual evidence for later risk and portfolio modeling. Exploratory analysis does not prove a model; it helps decide which questions are worth modeling {cite}`tukey1977eda`.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - audit missing values and usable date ranges across selected NASDAQ-listed equities;
# - compute simple and log returns from a versioned Yahoo Finance snapshot of adjusted daily closes;
# - summarize count, mean, median, volatility, minimum, maximum, percentiles, skewness, kurtosis, and missingness;
# - identify heavy tails, asymmetry, extreme observations, and rolling changes in return behavior;
# - interpret correlations, scatterplots, heatmaps, and rolling relationships without treating correlation as causation;
# - read plots as financial evidence rather than isolated graphics;
# - prepare an equity EDA table that can be reused in time-series, risk, and portfolio modules.
#
# ## Prerequisites
#
# Complete the provider inventory in `1.3` and the quality framework in `1.6`.
# Readers should understand adjusted close, simple and log returns, missing
# dates versus values, and the Module 1 common sample contract.
#
# ## Setup

# %% tags=["setup", "hide-input"]
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from src.market_data import (
    DEFAULT_NASDAQ_STOCK_TICKERS,
    nasdaq_stock_price_panel,
    returns_from_prices,
)
from src.market_data_quality import annualized_volatility, data_quality_report, hampel_outlier_flags
from src.module1_visuals import build_stock_eda_overview, build_stock_risk_map

pd.set_option("display.max_columns", 80)
ANALYSIS_START = "2021-01-01"
ANALYSIS_END = "2025-06-30"

# %% [markdown]
# ## NASDAQ stock selection and Yahoo Finance decision
#
# The EDA uses five liquid NASDAQ-listed equities: Apple (`AAPL`), Microsoft (`MSFT`), NVIDIA (`NVDA`), Amazon (`AMZN`), and Alphabet Class A (`GOOGL`). The published book reads a committed Yahoo Finance snapshot, so the HTML build does not depend on live network access.
#
# For this lesson, Yahoo Finance through `yfinance` is a convenience source for
# classroom EDA of daily adjusted closes: returns, volatility, correlations,
# rolling diagnostics, and visual inspection. It is not an official exchange
# feed or a guaranteed institutional database. The committed extract makes this
# calculation repeatable, but it does not by itself establish redistribution or
# downstream-use rights. Anyone refreshing or reusing the extract must review
# the terms applicable to that use. The `yfinance` project states that it is not
# affiliated with Yahoo, while Yahoo describes its market data as informational
# rather than trading-grade {cite}`yfinance2025,yahooFinanceCoverage2026,yahooTerms2026`.
#
# The lesson uses the Module 1 common sample window, 2021-01-01 through
# 2025-06-30. The snapshot contains additional dates for other uses, but they
# are excluded here so statistics match the rest of the module.
#

# %%
selected_stocks = pd.DataFrame.from_dict(
    DEFAULT_NASDAQ_STOCK_TICKERS,
    orient="index",
    columns=["company_name"],
).rename_axis("ticker")
selected_stocks


# %% [markdown]
# **Output interpretation.**
#
# The stock list defines the EDA universe before any return is computed. These are large, liquid NASDAQ-listed companies, which makes Yahoo Finance acceptable for classroom price-return exploration, but the table should not be read as a diversified portfolio recommendation.
#

# %% [markdown]
# ## Price panel

# %%
prices = nasdaq_stock_price_panel(start=ANALYSIS_START, end=ANALYSIS_END)
prices.tail()

# %% [markdown]
# **Output interpretation.**
#
# The price tail verifies the latest adjusted close observations in the NASDAQ stock snapshot. The values are dollar price levels, so visual differences in scale should not be interpreted as differences in investment performance.
#

# %%
stock_panel_metadata = pd.DataFrame(
    [
        {
            "data_mode": prices.attrs.get("data_mode", "snapshot"),
            "source": prices.attrs.get("sources", "not specified"),
            "start": prices.index.min(),
            "end": prices.index.max(),
            "observations": len(prices),
            "tickers": ", ".join(prices.columns),
        }
    ]
)
stock_panel_metadata


# %% [markdown]
# **Output interpretation.**
#
# The metadata output documents the exact sample behind the EDA. Because the book uses a local snapshot, the displayed statistics are reproducible; a live Yahoo refresh could change the end date, observations, and adjusted price history.
#

# %%
data_quality_report(prices)

# %% [markdown]
# **Output interpretation.**
#
# This report tells whether the panel is analysis-ready. A clean count and a stable date range support return construction; gaps or duplicates would need to be resolved before interpreting volatility or correlation.
#

# %% [markdown]
# ## Returns
#
# Simple returns measure proportional change:
#
# $$
# R_t = \frac{P_t - P_{t-1}}{P_{t-1}}.
# $$
#
# Log returns are additive across time:
#
# $$
# r_t = \ln(P_t) - \ln(P_{t-1}).
# $$

# %%
simple_returns = returns_from_prices(prices, method="simple")
log_returns = returns_from_prices(prices, method="log")

log_returns.head()

# %% [markdown]
# **Output interpretation.**
#
# The first valid return appears only after a prior price exists. Log returns are useful because multi-period returns can be summed, but the reported daily values still depend on the calendar and missing-data policy.
#

# %% [markdown]
# ## Return construction and panel design
#
# Most financial analysis does not use raw prices directly. Prices must often be transformed into returns, aligned across instruments, adjusted for missing values, and organized into panels.
#
# The formulas are simple, but the workflow is fragile. A small mistake in calendars, adjusted prices, missing data, cash distributions, or alignment can produce misleading results.
#
# ### Price return and total return
#
# Price return measures the return from price change only. Total return includes cash distributions such as dividends, coupons, or FIBRA distributions:
#
# $$
# R_t^{total} = \frac{P_t + C_t}{P_{t-1}} - 1.
# $$
#
# where \(C_t\) represents cash distributions during the period.
#
# For equities, raw closes intentionally measure price return and can contain a
# mechanical discontinuity when a split occurs. A provider-documented adjusted
# close can support a cumulative adjusted-close return and may approximate a
# reinvested total-return path only when the split, dividend, and reinvestment
# methodology is known. For bonds, FIBRAs, and other income-oriented
# instruments, distributions, accrued interest, timing, and provider methodology
# require separate treatment.
#
# | Concept | Meaning |
# | --- | --- |
# | Price return | Return from price change only |
# | Total return | Return including price change and cash distributions |
# | Nominal return | Return before inflation adjustment |
# | Real return | Return after inflation adjustment |
# | Local-currency return | Return measured in domestic currency |
# | Foreign-currency return | Return measured in another currency |
#
# ### Aligning assets
#
# A multi-asset panel requires a common date index. Common choices include:
#
# | Alignment choice | Meaning | Main tradeoff |
# | --- | --- | --- |
# | inner join | keep only dates available for all series | clean matrix, but may discard useful observations |
# | outer join | keep all dates and allow missing values | preserves information, but requires missing-data policy |
# | left join | use one reference calendar | useful when one asset or market is the analysis anchor |
# | business-day calendar | create a standard calendar and align all assets | explicit, but may create artificial missingness |
# | month-end alignment | convert daily series to monthly observations | useful for macro joins, but loses daily detail |
#
# There is no universally correct choice. The correct choice depends on the analysis.
#
# ### Wide and long formats
#
# A **wide format** stores one date per row and one asset per column. It is convenient for correlations, covariance matrices, and vectorized return calculations.
#
# A **long format** stores one observation per row. It is often better for storage, metadata, dashboards, and joins with instrument attributes. Tidy data principles help keep variables, observations, and observational units organized {cite}`wickham2014tidy`.
#
# ### Financial panel schema
#
# A cross-provider analysis-ready panel should use identifiers that also work
# for rates and macro variables; `ticker` and `price` can remain optional
# instrument-specific fields:
#
# | Column | Description |
# | --- | --- |
# | `date` | Observation date |
# | `series_id` | Stable instrument or variable identifier |
# | `series_type` | Equity, bond, ETF, FIBRA, FX level, rate, or macro index |
# | `value` | Observed price, rate, fixing, or index level |
# | `value_type` | Raw close, adjusted close, fixing, rate, or index level |
# | `return_simple` | Simple return when the economic object supports that interpretation |
# | `return_log` | Log return for an appropriate price or total-return series |
# | `log_change` | Log change for a positive non-contractual level |
# | `quote_unit` | USD, MXN, MXN per USD, percent, basis points, index units, etc. |
# | `source` | Banxico, DB.NOMICS, INEGI, BMV, licensed provider, or documented public source |
# | `quality_flag` | Missing, stale, outlier, revised, or reviewed |
#
# ### Return checklist
#
# Before calculating returns, confirm:
#
# ```text
# 1. Is the series a price, index, rate, yield, or macro level?
# 2. Are dates sorted?
# 3. Are duplicated dates removed or resolved?
# 4. Are prices adjusted or unadjusted?
# 5. Are missing prices handled?
# 6. Is the frequency defined?
# 7. Is the currency documented?
# 8. Are corporate actions considered?
# 9. Is the return formula appropriate?
# 10. Are results checked for extreme values?
# ```
#
# ## Summary statistics
#
# Descriptive statistics provide a first map of the data. They do not explain the full behavior of a market, but they help identify scale, dispersion, asymmetry, extreme observations, and potential data problems. Financial return distributions often display heavy tails, asymmetry, volatility clustering, and departures from normality {cite}`cont2001empirical`.
#
# Daily volatility is often annualized with the square-root-of-time rule:
#
# $$
# \sigma_{ann} \approx \sigma_{daily}\sqrt{252}.
# $$
#
# This is a convention, not a universal law. It can be misleading when returns are autocorrelated, volatility changes over time, or the sampling frequency is inconsistent {cite}`lo2002sharpe`.

# %%
summary = pd.DataFrame(
    {
        "valid_observations": log_returns.count(),
        "missing_price_observations": prices.isna().sum(),
        "missing_log_return_observations": len(prices) - log_returns.count(),
        "mean_daily_log_return": log_returns.mean(),
        "annualized_mean_log_return": log_returns.mean() * 252,
        "median_daily_log_return": log_returns.median(),
        "annualized_volatility": annualized_volatility(log_returns),
        "minimum_return": log_returns.min(),
        "maximum_return": log_returns.max(),
        "p01": log_returns.quantile(0.01),
        "skewness": log_returns.skew(),
        "excess_kurtosis": log_returns.kurtosis(),
        "p05": log_returns.quantile(0.05),
        "p25": log_returns.quantile(0.25),
        "p50": log_returns.quantile(0.50),
        "p75": log_returns.quantile(0.75),
        "p95": log_returns.quantile(0.95),
        "p99": log_returns.quantile(0.99),
        "negative_return_share": log_returns.lt(0).sum().div(log_returns.count()),
    }
)
summary = summary.sort_values("annualized_volatility", ascending=False)

display(
    summary[
        [
            "valid_observations",
            "missing_price_observations",
            "missing_log_return_observations",
        ]
    ]
)
display(
    summary[
        [
            "mean_daily_log_return",
            "annualized_mean_log_return",
            "median_daily_log_return",
            "annualized_volatility",
            "negative_return_share",
        ]
    ]
)
display(
    summary[
        [
            "minimum_return",
            "p01",
            "p05",
            "p25",
            "p50",
            "p75",
            "p95",
            "p99",
            "maximum_return",
            "skewness",
            "excess_kurtosis",
        ]
    ]
)

# %% [markdown]
# **Output interpretation.**
#
# The summary ranks assets by realized behavior over this sample, not by
# permanent risk. `annualized_mean_log_return` is the arithmetic mean of daily
# log returns multiplied by 252; it is not a guaranteed simple return or a
# forecast. Read it with missing counts, volatility, skewness, and tail
# percentiles rather than as a stand-alone ranking.
#

# %% [markdown]
# The mean and median should be compared. A large difference between them may suggest asymmetry, outliers, or a distribution that is not centered in a simple way. Percentiles are especially useful because they summarize downside and upside behavior without assuming a normal distribution {cite}`mcneil2015quantitative`.
#
# Skewness describes asymmetry through the distribution's third moment.
# Negative sample skewness indicates a relatively longer or heavier left tail;
# by itself, it does not prove that losses occur more frequently.
#
# ```{figure} ../../img/generated/eda-skewness-mean-median-mode.png
# :alt: Three schematic distributions comparing symmetric, right-skewed, and left-skewed relationships among mean, median, mode, and tails.
# :width: 850px
# :name: m1-eda-skewness
#
# Skewness is a directional summary of asymmetry, not a count of positive and
# negative observations.
# ```
#
# Kurtosis describes tail thickness. High excess kurtosis warns that extreme returns occur more often than a normal approximation would suggest.
#
# ```{figure} ../../img/generated/eda-kurtosis-tail-risk.png
# :alt: Schematic comparison of return distributions with different tail thickness and concentration around the center.
# :width: 850px
# :name: m1-eda-kurtosis
#
# Tail-shape comparison used to motivate empirical diagnostics rather than a
# normality assumption.
# ```
#
# ## Rolling descriptive diagnostics
#
# Financial markets change over time. A single mean or volatility for the full sample can hide regimes. Rolling statistics help students see whether return behavior is stable.

# %%
rolling_window = 63
rolling_snapshot = pd.DataFrame(
    {
        "rolling_mean": log_returns.rolling(rolling_window).mean().iloc[-1],
        "rolling_volatility_ann": log_returns.rolling(rolling_window).std().iloc[-1] * (252**0.5),
        "rolling_min": log_returns.rolling(rolling_window).min().iloc[-1],
        "rolling_max": log_returns.rolling(rolling_window).max().iloc[-1],
    }
)
rolling_snapshot

# %% [markdown]
# **Output interpretation.**
#
# The rolling snapshot compresses the most recent 63 trading days into short-horizon risk diagnostics. Annualized volatility is comparable across assets only when the same window and return convention are used; the rolling minimum and maximum highlight tail observations inside that recent window.

# %% [markdown]
# Rolling diagnostics depend on the selected window. A 20-day window, 63-day window, and 252-day window may tell different stories.
#
# ## Correlation and covariance
#
# Correlation is an input to diversification, portfolio optimization, and multi-asset risk. It should be interpreted together with the calendar, frequency, and outlier policy that produced the return matrix.
#
# Correlation measures linear association, not causality, stability, or complete dependence {cite}`mcneil2015quantitative`. Two assets may be correlated because they share the same macro driver, sector, currency, interest-rate exposure, risk sentiment, or crisis period.
#
# ```{figure} ../../img/generated/eda-correlation-gallery.png
# :alt: Scatterplot gallery illustrating positive, negative, and weak linear association while preserving visible dispersion.
# :width: 850px
# :name: m1-eda-correlation-gallery
#
# Linear-correlation patterns. None of the panels establishes causality or
# rules out nonlinear or tail dependence.
# ```

# %%
correlations = log_returns.corr()
correlations

# %% [markdown]
# **Output interpretation.**
#
# The correlation matrix shows average linear co-movement. Values near one reduce diversification benefits, values near zero indicate weaker linear association, and negative values can offset portfolio fluctuations, but none of these statements captures nonlinear tail dependence.
#

# %%
covariance = log_returns.cov()
covariance

# %% [markdown]
# **Output interpretation.**
#
# Covariance combines co-movement with each asset's scale of variation. It is the matrix used by mean-variance portfolio models, so units and return frequency must match before using it for optimization.
#

# %% [markdown]
# ## Rolling correlation
#
# Correlations can change over time. Relationships that look weak in calm periods may strengthen during stress periods.

# %%
asset_a, asset_b = log_returns.columns[:2]
rolling_correlation = log_returns[asset_a].rolling(rolling_window).corr(log_returns[asset_b])
rolling_correlation.dropna().tail()

# %% [markdown]
# **Output interpretation.**
#
# The rolling correlation tail shows the most recent local relationship between the selected pair. Compare it with the full-sample correlation to see whether diversification assumptions are stable or regime-dependent.
#

# %% [markdown]
# Low correlation does not automatically mean low joint risk. Linear correlation can miss nonlinear relationships, asymmetric dependence, and tail dependence.
#
# ## Visual EDA board
#
# A useful EDA figure should separate analytical questions instead of stacking
# every series on the same scale. The board below uses a common-base price view,
# rolling volatility and correlation, drawdowns, distribution small multiples,
# and a correlation heatmap fixed to the full `[-1, 1]` range.

# %%
stock_eda_figure = build_stock_eda_overview(
    prices,
    log_returns,
    rolling_window=rolling_window,
)
display(stock_eda_figure)
plt.close(stock_eda_figure)

# %% [markdown]
# **Output interpretation.**
#
# Read the six views as a sequence. The common-base panel supports performance
# comparison without confusing dollar prices with returns. Rolling volatility,
# drawdown, and rolling correlation reveal time variation that full-sample
# statistics hide. The violin plots preserve separate distributions, while the
# fixed correlation scale makes this matrix comparable with future samples.
#
# ## Risk-return map
#
# A two-dimensional map makes the cross-sectional tradeoff visible. Position
# encodes annualized mean log return and annualized volatility; marker size
# encodes the 5th-percentile daily log-loss magnitude. Labels and marker shapes keep the
# comparison readable without relying on color alone.

# %%
stock_risk_map = build_stock_risk_map(
    log_returns,
)
display(stock_risk_map)
plt.close(stock_risk_map)

# %% [markdown]
# **Output interpretation.**
#
# The upper-left direction is not automatically "best": the map describes one
# realized sample, and marker size exposes downside severity that the two axes
# do not. Compare each label with maximum drawdown, skewness, and tail
# percentiles before writing a ranking or recommendation.
#

# %% [markdown]
# ## Visualization checklist
#
# A financial visualization should answer a clear question.
#
# ```text
# 1. What question does the chart answer?
# 2. What variable is on each axis?
# 3. Is the frequency clear?
# 4. Are units clear?
# 5. Is the time period visible?
# 6. Are missing values or breaks visible?
# 7. Are extreme values explained or flagged?
# 8. Is the chart comparing levels, returns, or cumulative returns?
# 9. Are annotations used only when they add context?
# 10. Can a non-technical reader understand the main point?
# ```
#
# Visualization should support interpretation, not replace it {cite}`cleveland1993visualizing,wilke2019dataviz`.
#
# ## Outlier flags
#
# Large returns are not automatically errors. A flag identifies observations that require review before modeling.

# %%
outlier_counts = pd.Series(
    {
        asset: int(hampel_outlier_flags(log_returns[asset], window=21, n_sigmas=3.0).sum())
        for asset in log_returns.columns
    },
    name="flagged_observations",
)
outlier_counts

# %% [markdown]
# **Output interpretation.**
#
# Outlier counts are review flags, not automatic deletion rules. A flagged observation may be a real market move, a calendar mismatch, or a data error; the next step is source verification before any cleaning decision.
#

# %% [markdown]
# ## Interpretation discipline
#
# Descriptive statistics are not conclusions by themselves. They are prompts for interpretation.
#
# | Observation | Better question |
# | --- | --- |
# | High volatility | Was the asset structurally risky or was there a crisis period? |
# | Negative skewness | Are losses larger or more abrupt than gains? |
# | High kurtosis | Are extreme returns frequent? |
# | Low mean return | Is the sample period unfavorable or is the instrument low-return by design? |
# | Many missing values | Is the asset illiquid or is the source incomplete? |
# | High correlation | Is there a shared macro driver, sector exposure, currency effect, or crisis regime? |
#
# ## Handoff
#
# | Evidence | Downstream use |
# | --- | --- |
# | missingness report | determines whether calendar alignment is safe |
# | return summary | informs annualization, volatility, and tail-risk assumptions |
# | correlation matrix | supports covariance, beta, and diversification analysis |
# | outlier flags | separates data audit from model calibration |
# | source inventory | documents whether the analysis is reproducible and classroom-safe |
#
# The common-window return matrix is the direct handoff to the Return Explorer
# dashboard. Module 2 may reuse it only after retaining the log-return label,
# date window, missingness record, and outlier-review decisions.
