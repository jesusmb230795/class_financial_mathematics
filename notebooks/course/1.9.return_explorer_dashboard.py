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
# # Return Explorer Dashboard
#
# Module: Markets, Instruments, and Data
#
# ## Lesson summary
#
# This dashboard lets students explore prices, returns, rolling volatility, drawdowns, return distributions, and basic risk/performance metrics from the same clean NASDAQ stock panel. It uses reusable data helpers instead of embedding provider-specific logic in the notebook.
#
# The dashboard is exploratory, not a recommendation engine. It should help the reader compare assets, identify risk patterns, and decide what deserves deeper analysis {cite}`few2006dashboard,cairo2016truthful`.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - convert prices into simple or log returns;
# - inspect cumulative adjusted-close return and drawdown;
# - compare rolling volatility across assets;
# - identify skewness and tail behavior from histograms;
# - read introductory metrics such as downside percentiles, positive-loss
#   historical VaR, a correctly defined Sharpe ratio, hit ratio, and
#   peer-basket correlation;
# - connect dashboard views to a short written insight;
# - use the same return matrix as an input to risk and portfolio modules.
#
# ## Prerequisites
#
# Complete NASDAQ EDA (`1.4`) and the quality framework (`1.6`). Readers should
# know the common sample, adjusted-close policy, return convention, missingness
# record, and outlier-review boundary.
#
# ## Setup

# %% tags=["setup", "hide-input", "live-data"]
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import Dropdown, IntSlider, interact

from src.dashboard_fallbacks import build_return_explorer_fallback
from src.dashboards import build_return_explorer
from src.market_data import (
    DEFAULT_RETURN_DASHBOARD_TICKERS,
    return_dashboard_price_panel,
    returns_from_prices,
)

DATA_MODE = os.getenv("DATA_MODE", "offline").lower()
LIVE_DATA_START = os.getenv("LIVE_DATA_START", "2021-01-01")
LIVE_DATA_END = os.getenv("LIVE_DATA_END", "2025-06-30")
RUN_INTERACTIVE_WIDGETS = os.getenv("RUN_INTERACTIVE_WIDGETS", "1") == "1"

# %% [markdown]
# ## Price and return panel

# %% tags=["live-data"]
prices = return_dashboard_price_panel(
    data_mode=DATA_MODE,
    start=LIVE_DATA_START,
    end=LIVE_DATA_END,
)
prices.tail()

# %% [markdown]
# **Output interpretation.**
#
# The price panel tail confirms the observed levels behind the dashboard. Since
# levels have different scales, performance comparisons should be based on
# returns, cumulative adjusted-close returns, drawdowns, or normalized indices.
#

# %%
pd.DataFrame(
    [
        {
            "data_mode": prices.attrs.get("data_mode", DATA_MODE),
            "sources": prices.attrs.get("sources", "not specified"),
            "tickers": prices.attrs.get("tickers", DEFAULT_RETURN_DASHBOARD_TICKERS),
            "start": prices.index.min(),
            "end": prices.index.max(),
            "observations": len(prices),
        }
    ]
)

# %% [markdown]
# **Output interpretation.**
#
# The metadata table tells the reader how the dashboard was produced. A snapshot run is reproducible from the repository, while a live run must be interpreted with the refresh date and provider availability.
#

# %% [markdown]
# ## Dashboard question map
#
# The dashboard has one primary analytical objective: **show how the selected
# stock's return path, distribution, and rolling risk differ from its own price
# path and from the other stocks in the fixed classroom universe**. Each view
# answers a subordinate question.
#
# | Question | Dashboard element |
# | --- | --- |
# | How did the asset evolve over time? | Price chart |
# | How did cumulative performance evolve? | Cumulative adjusted-close return chart |
# | How dispersed are returns? | Return histogram |
# | How did risk change over time? | Rolling volatility chart |
# | How severe were losses from peaks? | Drawdown line |
# | How does the asset co-move with the other stocks? | Risk summary and equal-weight peer-basket correlation |
# | Can another analyst trust the panel? | Data mode, source, date range, and assumptions |
#
# Chart choice should support the analytical question. Visual design should prioritize clarity, labels, units, comparable scales, and truthful representation {cite}`tufte2001visual,wilke2019dataviz`.
#
# ## Exploratory risk and performance metrics
#
# Return alone is not enough. At the exploratory level, risk can be summarized
# through volatility, downside returns, drawdowns, tail percentiles, one-period
# historical VaR, return-to-volatility ratios, and stability over time.
#
# This table is a first reading, not a complete risk model. Drawdown is useful because investors often experience risk through losses from previous highs, while historical VaR is only a distributional threshold, not a worst-case loss {cite}`magdonismail2004drawdown,jorion2007var`. Full VaR modeling, expected shortfall, stress testing, backtesting, volatility modeling, and attribution belong in later modules.
#
# To match the book-wide convention, the exploratory 95% historical VaR is
# reported as a positive one-period log-loss magnitude:
#
# $$
# \operatorname{VaR}_{0.95}^{hist} = \max\left(0,-Q_{0.05}(r_t)\right).
# $$

# %%
dashboard_log_returns = returns_from_prices(prices, method="log")
dashboard_simple_returns = returns_from_prices(prices, method="simple")


def summarize_return_series(
    asset_prices,
    asset_log_returns,
    asset_simple_returns,
    peer_reference_simple_returns=None,
):
    observed_prices = asset_prices.dropna()
    observed_log_returns = asset_log_returns.dropna()
    observed_simple_returns = asset_simple_returns.dropna()
    annualized_log_return_volatility = observed_log_returns.std() * np.sqrt(252)
    arithmetic_sharpe_ratio_zero_rf = (
        observed_simple_returns.mean() / observed_simple_returns.std() * np.sqrt(252)
        if observed_simple_returns.std() > 0
        else np.nan
    )
    peer_correlation = (
        asset_simple_returns.corr(peer_reference_simple_returns)
        if peer_reference_simple_returns is not None
        else np.nan
    )
    return pd.Series(
        {
            "valid_log_return_observations": observed_log_returns.count(),
            "missing_log_return_observations": len(asset_prices) - asset_log_returns.count(),
            "cumulative_adjusted_close_change": (
                observed_prices.iloc[-1] / observed_prices.iloc[0] - 1
            ),
            "annualized_mean_log_return": observed_log_returns.mean() * 252,
            "annualized_log_return_volatility": annualized_log_return_volatility,
            "best_log_return": observed_log_returns.max(),
            "worst_log_return": observed_log_returns.min(),
            "p05_log_return": observed_log_returns.quantile(0.05),
            "historical_log_var_95_loss": max(
                0.0,
                -observed_log_returns.quantile(0.05),
            ),
            "maximum_adjusted_close_drawdown": (
                observed_prices / observed_prices.cummax() - 1
            ).min(),
            "arithmetic_sharpe_ratio_zero_rf": arithmetic_sharpe_ratio_zero_rf,
            "positive_simple_return_share": (observed_simple_returns > 0).mean(),
            "correlation_with_daily_rebalanced_peer_basket": peer_correlation,
        }
    )


risk_summary = pd.DataFrame(
    {
        asset: summarize_return_series(
            prices[asset],
            dashboard_log_returns[asset],
            dashboard_simple_returns[asset],
            peer_reference_simple_returns=dashboard_simple_returns.drop(columns=asset).mean(
                axis=1, skipna=False
            ),
        )
        for asset in prices.columns
    }
).T
display(
    risk_summary[
        [
            "valid_log_return_observations",
            "missing_log_return_observations",
        ]
    ]
)
display(
    risk_summary[
        [
            "cumulative_adjusted_close_change",
            "annualized_mean_log_return",
            "annualized_log_return_volatility",
            "maximum_adjusted_close_drawdown",
            "arithmetic_sharpe_ratio_zero_rf",
        ]
    ]
)
display(
    risk_summary[
        [
            "worst_log_return",
            "p05_log_return",
            "historical_log_var_95_loss",
            "best_log_return",
            "positive_simple_return_share",
            "correlation_with_daily_rebalanced_peer_basket",
        ]
    ]
)


# %% [markdown]
# **Output interpretation.**
#
# The risk summary translates the return panel into exploratory summaries, not
# recommendations or complete decision metrics.
# `annualized_mean_log_return` is the daily mean log return multiplied by 252;
# `cumulative_adjusted_close_change` is the endpoint change in the provider-
# adjusted close, not an independently verified total-return index.
# `historical_log_var_95_loss` is a positive one-period log-loss magnitude. The
# correlation reference is the simple return of a daily-rebalanced equal-weight
# basket of the **other four** stocks, calculated only when every peer return is
# available; it is a classroom peer basket, not a market index, investable
# benchmark, or performance target.
#

# %% [markdown]
# `arithmetic_sharpe_ratio_zero_rf` uses the mean and sample standard deviation
# of daily **simple** excess returns under an explicit zero reference rate, then
# applies the square-root-of-252 convention. A nonzero reference rate would
# need matching currency, frequency, horizon, and compounding
# {cite}`sharpe1994ratio,lo2002sharpe`.
# The dashboard title and axes retain the selected simple/log-return convention,
# and rolling volatility states the 252-trading-day annualization convention.
#
# ## Dashboard function


# %% tags=["interactive"]
def plot_return_explorer(asset=None, return_method="log", rolling_window=63):
    if asset is None:
        asset = prices.columns[0]

    if RUN_INTERACTIVE_WIDGETS:
        figure = build_return_explorer(
            prices,
            asset,
            return_method=return_method,
            rolling_window=rolling_window,
            data_mode=prices.attrs.get("data_mode", DATA_MODE),
        )
    else:
        figure = build_return_explorer_fallback(
            prices,
            asset,
            return_method=return_method,
            rolling_window=rolling_window,
            data_mode=prices.attrs.get("data_mode", DATA_MODE),
        )
    display(figure)
    if not RUN_INTERACTIVE_WIDGETS:
        plt.close(figure)


DEFAULT_ASSET = prices.columns[0]

if RUN_INTERACTIVE_WIDGETS:
    interact(
        plot_return_explorer,
        asset=Dropdown(options=list(prices.columns), value=DEFAULT_ASSET),
        return_method=Dropdown(options=["log", "simple"], value="log"),
        rolling_window=IntSlider(value=63, min=21, max=252, step=21),
    )
else:
    plot_return_explorer()

# %% [markdown]
# **Output interpretation.**
#
# The dashboard links path, distribution, and rolling risk. A strong cumulative return can still hide deep drawdowns or unstable volatility, so the panels should be interpreted together rather than sequentially.
#

# %% [markdown]
# ## Interpretation checklist
#
# Avoid these common mistakes:
#
# | Mistake | Why it is a problem |
# | --- | --- |
# | Ranking assets only by average return | Ignores risk and drawdowns |
# | Treating volatility as complete risk | Ignores asymmetry and tail losses |
# | Ignoring the sample period | Results may be regime-dependent |
# | Annualizing mechanically | Square-root scaling assumptions may not hold |
# | Comparing assets in different currencies | Mixes asset return and FX return |
# | Calling raw price return a distribution-adjusted measure | Ignores dividends, distributions, and provider adjustment methodology |
# | Treating VaR as maximum possible loss | VaR is a threshold, not a worst-case loss |
# | Treating Sharpe ratio as absolute truth | It depends on assumptions and return behavior |
#
# ## Real-data extension
#
# The default build uses `DATA_MODE=offline`, which reads the versioned
# Yahoo-derived adjusted-close classroom snapshot. To run the same dashboard
# with public market prices, launch Jupyter locally with live mode:
#
# ```bash
# DATA_MODE=live LIVE_DATA_START=2021-01-01 LIVE_DATA_END=2025-06-30 uv run jupyter lab
# ```
#
# The live panel is loaded through `return_dashboard_price_panel`, which maps
# public tickers to stable dashboard labels and caches provider extracts before
# analysis. A reproducible cache does not establish redistribution or other
# downstream-use rights; review the applicable provider terms before refreshing
# or sharing an extract.

# %% [markdown]
# ## Handoff
#
# Export a dashboard finding only with the selected ticker, return convention,
# rolling window, source, data mode, sample, and peer-reference definition. The
# final Mexican case uses a different panel and must not inherit NASDAQ results
# or call its peer basket a Mexican market benchmark.

# %% [markdown] tags=["exercise"]
# ## Checkpoint exercise
#
# Build a dashboard decision note for one stock.
#
# 1. State the question and report cumulative adjusted-close return, annualized
#    mean log return, annualized volatility, positive-loss historical VaR,
#    maximum drawdown, arithmetic Sharpe ratio under the explicit zero-rate
#    assumption, and equal-weight peer-basket correlation over the common window.
# 2. Compare 63-day and 252-day rolling volatility and identify one period in
#    which the dashboard changes the full-sample interpretation.
# 3. Switch between simple and log returns; explain which displayed quantities
#    change and which return convention belongs in the written conclusion.
# 4. State why the peer basket is useful for exploratory co-movement but is not
#    an investable or policy benchmark.
#

# %% [markdown] tags=["solution"]
# ```{dropdown} Suggested answer rubric
# A complete note answers one declared question, labels every metric and return
# convention, compares both rolling windows, keeps VaR as a positive loss, and
# describes the peer basket without overstating it as a market benchmark or
# recommendation. Reconcile the reported values with the executed risk-summary
# and dashboard tables; treat the adjusted-close result according to the
# provider methodology rather than assuming a fully specified total-return index.
# ```
