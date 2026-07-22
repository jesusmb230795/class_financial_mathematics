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
# This lab extends VaR estimation into model validation. It generates rolling VaR forecasts, tests exception frequency and exception clustering, maps recent exceptions to Basel traffic-light logic, and complements probabilistic risk metrics with deterministic stress scenarios {cite}`kupiec1995techniques,christoffersen1998evaluating,basel2019marketRisk`.
#
# ## Learning objectives
#
# By the end of this lab, students should be able to:
#
# - build a rolling one-day VaR forecast;
# - identify VaR exceptions from realized portfolio returns;
# - run Kupiec unconditional coverage and Christoffersen independence tests;
# - interpret a Basel traffic-light result;
# - design a deterministic stress test for a multi-asset portfolio.
#
# ## Prerequisites
#
# Complete the two preceding market-risk labs. Students should be able to
# compute a non-negative one-day VaR without look-ahead and distinguish a
# probabilistic loss threshold from a deterministic stress shock.
#
# ## Backtesting equations
#
# A VaR exception occurs when realized loss exceeds the forecast threshold:
#
# $$
# I_t=\mathbf{1}\{-R_t>\widehat{\operatorname{VaR}}_{\alpha,t}\}.
# $$
#
# Under a correctly calibrated one-day VaR model, the expected exception rate is approximately $\alpha$:
#
# $$
# \mathbb{E}[I_t]=\alpha.
# $$
#
# Stress testing applies a deterministic shock vector $s$ to portfolio weights $w$:
#
# $$
# \operatorname{StressLoss}= -w^\top s.
# $$
#
# ## Setup

# %% tags=["setup", "hide-input"]
import pandas as pd
from scipy.stats import norm

from src.market_data import official_price_panel, returns_from_prices
from src.market_risk import (
    basel_traffic_light,
    conditional_coverage_test,
    exception_series,
    stress_scenario_loss,
)

pd.options.display.float_format = "{:.6f}".format

# %% [markdown]
# ## Official portfolio returns
#
# The backtest uses observed returns from the committed Banxico official price-like snapshot. Using the same real-data panel as earlier risk notebooks keeps the exception tests reproducible without generating artificial regimes.

# %%
price_panel = official_price_panel(start="2021-01-01", end="2026-06-05")
asset_returns = returns_from_prices(price_panel, method="log").dropna()
portfolio_weights = (
    pd.Series(
        {
            "usd_mxn": 0.30,
            "udi": 0.20,
            "cetes_28d_carry": 0.20,
            "tiie_28d_carry": 0.20,
            "policy_rate_carry": 0.10,
        },
        name="weight",
    )
    .reindex(asset_returns.columns)
    .fillna(0.0)
)
portfolio_weights = portfolio_weights / portfolio_weights.sum()
portfolio_returns = asset_returns.dot(portfolio_weights).rename("portfolio_return")

portfolio_returns.describe()

# %% [markdown]
# ## Rolling VaR forecasts
#
# Historical VaR uses only information available before the realized return. The `shift(1)` is essential because today's return cannot be used to forecast today's risk.

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
# ## Exception series

# %%
historical_exceptions = exception_series(portfolio_returns, historical_var_forecast)
gaussian_exceptions = exception_series(portfolio_returns, gaussian_var_forecast)

pd.DataFrame(
    {
        "historical_exceptions": historical_exceptions.value_counts(),
        "gaussian_exceptions": gaussian_exceptions.value_counts(),
    }
).fillna(0).astype(int)

# %% [markdown]
# ## Statistical backtesting
#
# Kupiec's test checks whether the total number of exceptions is consistent with the chosen tail probability. Christoffersen's test checks whether exceptions are clustered.

# %%
pd.DataFrame(
    {
        "historical_var": conditional_coverage_test(historical_exceptions, alpha=alpha),
        "gaussian_var": conditional_coverage_test(gaussian_exceptions, alpha=alpha),
    }
)

# %% [markdown]
# **Output interpretation.** A large p-value does not prove the model is correct; it only means the observed exceptions are not inconsistent with the tested calibration under this sample and test power.

# %% [markdown]
# ## Basel traffic-light view
#
# The classic Basel traffic-light table is based on 250 trading days of exceptions. This example maps the latest 250 backtest observations to the zone and multiplier.

# %%
latest_250_exception_count = int(historical_exceptions.tail(250).sum())
basel_traffic_light(latest_250_exception_count)

# %% [markdown]
# ## Deterministic stress scenario
#
# VaR and Expected Shortfall extrapolate from a probability model. Stress testing asks a different question: what happens if a specific macro-financial shock is imposed on the portfolio?

# %%
weights = portfolio_weights.rename("weight")

stress_shocks = (
    pd.Series(
        {
            "usd_mxn": 0.18,
            "udi": -0.04,
            "cetes_28d_carry": -0.03,
            "tiie_28d_carry": -0.04,
            "policy_rate_carry": -0.03,
        },
        name="shock",
    )
    .reindex(weights.index)
    .fillna(0.0)
)

stress_scenario_loss(weights, stress_shocks)

# %% [markdown]
# ## Regulatory context
#
# The Fundamental Review of the Trading Book moved market-risk capital from a VaR-centered framework toward Expected Shortfall. A practical implementation also has to account for liquidity horizons, because a position that takes 60 or 120 days to exit cannot be treated like a highly liquid 10-day risk factor {cite}`basel2019marketRisk`.
#
# For this course, the key modeling lesson is not to memorize a capital formula. The key lesson is that a backtested VaR model, an Expected Shortfall estimate, and a deterministic stress scenario answer different risk questions and should be documented together.
#
# ## Model limitations
#
# - Backtests have low power when exceptions are rare, so passing a test is not proof that the risk model is reliable.
# - Rolling windows trade responsiveness against estimation noise and can react slowly to abrupt regime changes.
# - Stress scenarios are judgment-based and should complement, not replace, probabilistic risk estimates.

# %% [markdown]
# ## Handoff
#
# The final dashboard varies alpha, lookback, and EWMA decay interactively. Use
# the backtesting evidence from this page to avoid interpreting dashboard
# sensitivity as model validation.

# %% [markdown] tags=["exercise"]
# ## Checkpoint exercise
#
# 1. Verify that every one-day VaR forecast is non-negative and that the first
#    usable forecast depends only on the previous 250 returns.
# 2. Report exception count, exception rate, Kupiec p-value, Christoffersen
#    independence p-value, and conditional-coverage p-value for both historical
#    and Gaussian forecasts.
# 3. Recompute the historical backtest with a 500-day window, compare the
#    exception evidence, and explain why passing either test is not proof that
#    the tail model is correct.
#

# %% [markdown] tags=["solution"]
# ```{dropdown} Suggested answer rubric
# A complete answer demonstrates the one-day information lag, reports all five
# diagnostics for both models and both windows, distinguishes exception
# frequency from clustering, and discusses the low power of rare-event tests.
# ```
