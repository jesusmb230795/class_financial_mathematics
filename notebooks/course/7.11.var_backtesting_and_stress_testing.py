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
# # VaR Backtesting and Stress Testing
#
# Module: Derivatives and Risk Management
#
# ## Lesson summary
#
# This lab moves from risk estimation to validation. It freezes rolling one-step
# Value at Risk (VaR) forecasts before the corresponding return, tests exception
# frequency and clustering, presents the historical Basel 250-day traffic-light
# diagnostic in its proper context, and adds a deterministic adverse scenario
# {cite}`kupiec1995techniques,christoffersen1998evaluating,basel1996MarketRiskAmendment,basel2019marketRisk,baselFrameworkMAR33`.
#
# ## Learning objectives
#
# By the end of this lab, students should be able to:
#
# - construct one-step rolling VaR without look-ahead;
# - identify exceptions under a positive-loss threshold convention;
# - state and test Kupiec and Christoffersen null hypotheses;
# - distinguish a historical Basel diagnostic from the current market-risk framework; and
# - calculate and document an adverse multi-asset scenario loss.
#
# ## Prerequisites
#
# Complete the two preceding market-risk labs. Students should be able to
# compute non-negative one-interval VaR and distinguish a probabilistic loss
# threshold from a deterministic shock.
#
# ## Backtesting equations and hypotheses
#
# A VaR exception occurs when the realized loss exceeds the forecast fixed at
# the end of the preceding interval:
#
# $$
# I_t=\mathbf{1}\!\left\{-R_t>
# \widehat{\operatorname{VaR}}_{\alpha,t\mid t-1}\right\}.
# $$
#
# For $T$ forecasts, let $x=\sum_{t=1}^T I_t$ and $\widehat p=x/T$. Kupiec's
# proportion-of-failures statistic tests $H_0:p=\alpha$ against $H_1:p\ne\alpha$:
#
# $$
# \operatorname{LR}_{\mathrm{POF}}
# =-2\log\left[
# \frac{(1-\alpha)^{T-x}\alpha^x}
# {(1-\widehat p)^{T-x}\widehat p^x}
# \right]
# \xrightarrow{H_0}\chi_1^2.
# $$
#
# To test independence, define $n_{ij}$ as the number of transitions from
# $I_{t-1}=i$ to $I_t=j$, $\widehat\pi_i=n_{i1}/(n_{i0}+n_{i1})$, and
# $\widehat\pi=(n_{01}+n_{11})/\sum_{i,j}n_{ij}$. Christoffersen's statistic
# tests $H_0:\pi_0=\pi_1$ against exception dependence:
#
# $$
# \operatorname{LR}_{\mathrm{IND}}
# =-2\log\left[
# \frac{(1-\widehat\pi)^{n_{00}+n_{10}}
# \widehat\pi^{n_{01}+n_{11}}}
# {(1-\widehat\pi_0)^{n_{00}}\widehat\pi_0^{n_{01}}
# (1-\widehat\pi_1)^{n_{10}}\widehat\pi_1^{n_{11}}}
# \right]
# \xrightarrow{H_0}\chi_1^2.
# $$
#
# The conditional-coverage statistic combines the two diagnostics:
#
# $$
# \operatorname{LR}_{\mathrm{CC}}
# =\operatorname{LR}_{\mathrm{POF}}+\operatorname{LR}_{\mathrm{IND}}
# \xrightarrow{H_0}\chi_2^2.
# $$
#
# This notebook uses a 5% test size. A p-value below 0.05 rejects the relevant
# null; a larger p-value means "do not reject," not "the model is correct"
# {cite}`kupiec1995techniques,christoffersen1998evaluating`.
#
# Stress testing applies a deterministic simple-return shock vector $s$ to
# weights $w$:
#
# $$
# \operatorname{StressLoss}=-w^\top s.
# $$
#
# ## Setup

# %% tags=["setup", "hide-input"]
import pandas as pd
from scipy.stats import norm

from src.market_data import nasdaq_stock_price_panel, returns_from_prices
from src.market_risk import (
    basel_traffic_light,
    conditional_coverage_test,
    exception_series,
    rebalanced_portfolio_returns,
    stress_scenario_loss,
)
from src.module7_visuals import build_var_backtest_figure

pd.options.display.float_format = "{:.6f}".format

# %% [markdown]
# ## Observed equal-weight portfolio
#
# The versioned snapshot contains provider-adjusted closes in USD for AAPL,
# MSFT, NVDA, AMZN, and GOOGL from 2021-01-04 through 2026-06-05. The calculation
# uses simple returns and restores each 20% weight after every observed US
# trading interval. It omits transaction costs, taxes, and foreign-exchange
# conversion {cite}`yfinance2025,yahooFinanceCoverage2026,yahooTerms2026`.

# %%
price_panel = nasdaq_stock_price_panel(start="2021-01-04", end="2026-06-05")
asset_returns = returns_from_prices(price_panel, method="simple").dropna(how="any")
asset_returns.attrs = {
    **price_panel.attrs,
    "method": "simple returns; equal weights rebalanced each observed interval",
}

portfolio_weights = pd.Series(
    {
        "AAPL": 0.20,
        "MSFT": 0.20,
        "NVDA": 0.20,
        "AMZN": 0.20,
        "GOOGL": 0.20,
    },
    name="weight",
)
portfolio_returns = rebalanced_portfolio_returns(asset_returns, portfolio_weights)
portfolio_returns.name = "equal_weight_portfolio_simple_return"

portfolio_returns.describe()

# %%
pd.Series(
    {
        "source": price_panel.attrs["sources"],
        "field_and_currency": "provider-adjusted close, USD",
        "price_sample": f"{price_panel.index.min():%Y-%m-%d} to {price_panel.index.max():%Y-%m-%d}",
        "return_sample": (
            f"{portfolio_returns.index.min():%Y-%m-%d} to {portfolio_returns.index.max():%Y-%m-%d}"
        ),
        "frequency": "observed US trading intervals; no calendar filling",
        "snapshot_generated_at": "2026-06-07T04:55:41.700953+00:00",
        "portfolio_rule": "20% each; rebalanced after every observed interval",
        "omitted": "transaction costs, taxes, and FX conversion",
        "rights_review": "provenance recorded; redistribution rights not independently verified",
    },
    name="data_and_portfolio_contract",
)

# %% [markdown]
# Snapshot provenance supports reproducibility but does not independently
# establish redistribution or downstream-use rights.
#
# ## Rolling VaR forecasts
#
# Each 250-observation estimator is shifted by one interval. Consequently, the
# return at $t$ cannot affect its own $t\mid t-1$ threshold.

# %%
alpha = 0.01
window = 250

historical_var_forecast = (
    -portfolio_returns.rolling(window).quantile(alpha).shift(1).rename("historical_var")
).clip(lower=0.0)

rolling_mean = portfolio_returns.rolling(window).mean().shift(1)
rolling_volatility = portfolio_returns.rolling(window).std().shift(1)
gaussian_var_forecast = (
    -(rolling_mean + norm.ppf(alpha) * rolling_volatility).rename("gaussian_var")
).clip(lower=0.0)

pd.concat(
    [portfolio_returns, historical_var_forecast, gaussian_var_forecast],
    axis=1,
).dropna().head()

# %% [markdown]
# ## Exception series and visual audit

# %%
historical_exceptions = exception_series(portfolio_returns, historical_var_forecast)
gaussian_exceptions = exception_series(portfolio_returns, gaussian_var_forecast)

pd.DataFrame(
    {
        "historical_var": historical_exceptions.value_counts(),
        "gaussian_var": gaussian_exceptions.value_counts(),
    }
).fillna(0).astype(int).rename_axis("exception")

# %% mystnb={"image": {"alt": "Two aligned time-series panels show the equal-weight US equity portfolio's realized signed loss, a one-percent rolling historical Value at Risk threshold fixed one interval earlier, x-shaped markers where signed loss strictly exceeds Value at Risk, and the cumulative exception count."}}
backtest_figure = build_var_backtest_figure(
    portfolio_returns,
    historical_var_forecast,
    historical_exceptions,
)
backtest_figure

# %% [markdown]
# The timeline reveals timing and clustering that a total exception count cannot
# show. An exception is an observation beyond a modeled threshold, not proof of
# a data error.
#
# ## Statistical backtesting and decisions

# %%
backtest_results = pd.DataFrame(
    {
        "historical_var": conditional_coverage_test(
            historical_exceptions,
            alpha=alpha,
        ),
        "gaussian_var": conditional_coverage_test(
            gaussian_exceptions,
            alpha=alpha,
        ),
    }
).T
backtest_results["kupiec_decision_5pct"] = backtest_results["kupiec_p_value"].map(
    lambda p_value: "reject coverage" if p_value < 0.05 else "do not reject coverage"
)
backtest_results["independence_decision_5pct"] = backtest_results["independence_p_value"].map(
    lambda p_value: "reject independence" if p_value < 0.05 else "do not reject independence"
)
backtest_results["conditional_coverage_decision_5pct"] = backtest_results["p_value"].map(
    lambda p_value: (
        "reject conditional coverage" if p_value < 0.05 else "do not reject conditional coverage"
    )
)
backtest_results

# %% [markdown]
# **Output interpretation.** The three decisions answer different questions:
# correct average exception frequency, independence of consecutive exceptions,
# and their joint conditional-coverage requirement. Low test power remains a
# material limitation when $\alpha$ is small.
#
# ## Historical Basel traffic-light diagnostic
#
# The 1996 Basel market-risk amendment attached green, yellow, and red zones to
# exceptions over 250 observations for internal-model capital multiplication.
# This notebook reports that historical diagnostic for teaching; it is not a
# claim that the portfolio or calculation satisfies current regulatory approval
# requirements {cite}`basel1996MarketRiskAmendment`.

# %%
latest_250_exception_count = int(historical_exceptions.tail(250).sum())
basel_traffic_light(latest_250_exception_count).rename("historical_1996_diagnostic")

# %% [markdown]
# ## Deterministic adverse scenario
#
# Every shock below is a negative simple return. Labels must match the portfolio
# exactly; the helper rejects missing or extra assets rather than silently
# dropping them.

# %%
stress_shocks = pd.Series(
    {
        "AAPL": -0.12,
        "MSFT": -0.10,
        "NVDA": -0.25,
        "AMZN": -0.15,
        "GOOGL": -0.10,
    },
    name="adverse_simple_return_shock",
)

stress_result = stress_scenario_loss(portfolio_weights, stress_shocks)
pd.concat(
    [
        portfolio_weights,
        stress_shocks,
        stress_result.drop(index="portfolio_total"),
    ],
    axis=1,
).rename(columns={"scenario_loss": "positive_loss_contribution"})

# %%
pd.Series(
    {
        "portfolio_simple_return_under_scenario": -stress_result["portfolio_total"],
        "positive_portfolio_loss": stress_result["portfolio_total"],
    },
    name="adverse_scenario_summary",
)

# %% [markdown]
# This linear scenario is deliberately adverse for every long position. It is
# not assigned a probability and does not include rebalancing during the shock,
# nonlinear instruments, liquidity costs, taxes, or foreign-exchange effects.
#
# ## Regulatory context
#
# The consolidated Basel market-risk framework is not the 1996 traffic-light
# table. Its internal-models approach uses Expected Shortfall and explicit
# liquidity horizons within a much broader capital framework. VaR backtesting,
# ES measurement, stress testing, model approval, and capital calculation are
# related but distinct claims
# {cite}`basel2019marketRisk,baselFrameworkMAR33`.
#
# ## Model limitations
#
# - Backtests have low power when exceptions are rare; non-rejection is not validation.
# - A rolling window trades responsiveness against estimation noise and can lag a regime shift.
# - Overlapping model development and evaluation samples can overstate apparent performance.
# - The equal-weight portfolio is hypothetical and omits implementation frictions.
# - Stress scenarios depend on judgment and should complement probabilistic estimates.
#
# ## Handoff
#
# The final dashboard varies tail probability, lookback, and EWMA decay. Use the
# tests and adverse scenario here to avoid interpreting dashboard sensitivity as
# evidence of model validity.
