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
# # Interactive VaR and CVaR Dashboard
#
# Module: Derivatives and Risk Management
#
# ## Lesson summary
#
# This dashboard uses observed portfolio returns derived from the committed Banxico official price-like snapshot. Students change tail probability, lookback window, and EWMA decay to see how VaR and Expected Shortfall react to the selected real sample {cite}`jorion2007var,mcneil2015quantitative`.
#
# ## Learning objectives
#
# By the end of this dashboard, students should be able to:
#
# - explain VaR as a quantile of the loss distribution;
# - explain CVaR or Expected Shortfall as a tail conditional average;
# - compare historical, Gaussian, Cornish-Fisher, and volatility-weighted VaR;
# - explain why sample window choice changes tail metrics;
# - recognize why model choice affects reported capital.
#
# ## Prerequisites
#
# Complete the estimation and backtesting labs first. Students should
# understand non-negative loss signs, sampling uncertainty, the Cornish-Fisher
# guardrail, and why passing a coverage test is not proof of model correctness.
#
# ## Tail metric definitions
#
# The dashboard displays risk metrics as positive loss numbers. For portfolio return $R$ and loss $L=-R$:
#
# $$
# v_\alpha=Q_{1-\alpha}(L),\qquad
# \operatorname{VaR}_{\alpha}
# =\max\left(0,-Q_\alpha(R)\right)
# =\max(0,v_\alpha),
# $$
#
# $$
# \operatorname{CVaR}_{\alpha}=\operatorname{ES}_{\alpha}
# =\max\left(0,-\mathbb{E}\left[R\mid R\leq Q_\alpha(R)\right]\right)
# =\max\left(0,\mathbb{E}\left[L\mid L\geq v_\alpha\right]\right).
# $$
#
# Changing the lookback window changes the empirical tail of $L$, which is why VaR and CVaR do not move identically.
#
# ## Setup

# %% tags=["setup", "hide-input"]
import os

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from ipywidgets import FloatSlider, IntSlider, interact

from src.dashboard_fallbacks import build_var_cvar_dashboard_fallback
from src.dashboards import build_var_cvar_dashboard
from src.market_data import official_price_panel, returns_from_prices
from src.market_risk import (
    cornish_fisher_moment_report,
    cornish_fisher_var,
    expected_shortfall,
    gaussian_var,
    historical_var,
    volatility_weighted_historical_var,
)

RUN_INTERACTIVE_WIDGETS = os.getenv("RUN_INTERACTIVE_WIDGETS", "1") == "1"
pd.options.display.float_format = "{:.6f}".format

# %% [markdown]
# ## Real return helper

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


def official_portfolio_returns(lookback=1000):
    lookback = max(250, min(int(lookback), len(asset_returns)))
    return asset_returns.dot(portfolio_weights).dropna().tail(lookback).rename("portfolio_return")


# %% [markdown]
# ## Interactive dashboard
#
# Run this cell in JupyterLab with `uv run jupyter lab`. The publication build sets `RUN_INTERACTIVE_WIDGETS=0`, so the book renders a static default view instead of widget controls.


# %% tags=["interactive"]
def plot_var_cvar_dashboard(
    alpha=0.01,
    lambda_=0.94,
    lookback=1000,
):
    returns = official_portfolio_returns(lookback=lookback)
    cornish_fisher_report = cornish_fisher_moment_report(returns)
    try:
        cornish_fisher_estimate = cornish_fisher_var(returns, alpha=alpha)
        cornish_fisher_status = "available"
    except ValueError as exc:
        cornish_fisher_estimate = float("nan")
        cornish_fisher_status = f"unavailable: {exc}"

    risk_metrics = pd.Series(
        {
            "historical_var": historical_var(returns, alpha=alpha),
            "gaussian_var": gaussian_var(returns, alpha=alpha),
            "cornish_fisher_var": cornish_fisher_estimate,
            "volatility_weighted_var": volatility_weighted_historical_var(
                returns,
                alpha=alpha,
                lambda_=lambda_,
            ),
            "expected_shortfall": expected_shortfall(returns, alpha=alpha),
        },
        name="positive_daily_loss",
    )

    builder = (
        build_var_cvar_dashboard
        if RUN_INTERACTIVE_WIDGETS
        else build_var_cvar_dashboard_fallback
    )
    figure = builder(returns, risk_metrics)
    display(figure)
    if not RUN_INTERACTIVE_WIDGETS:
        plt.close(figure)
    display(risk_metrics.to_frame())
    display(
        pd.concat(
            [
                cornish_fisher_report,
                pd.Series({"status": cornish_fisher_status}),
            ]
        ).rename("cornish_fisher_diagnostic")
    )


if RUN_INTERACTIVE_WIDGETS:
    interact(
        plot_var_cvar_dashboard,
        alpha=FloatSlider(value=0.01, min=0.005, max=0.10, step=0.005, readout_format=".3f"),
        lambda_=FloatSlider(value=0.94, min=0.80, max=0.99, step=0.01, readout_format=".2f"),
        lookback=IntSlider(value=1000, min=500, max=len(asset_returns), step=250),
    )
else:
    plot_var_cvar_dashboard()

# %% [markdown]
# ## Model limitations
#
# - The dashboard uses observed returns from the official snapshot; it does not create artificial shock regimes.
# - Parametric VaR can understate losses when skewness, kurtosis, or dependence differs from the assumed form.
# - Cornish-Fisher is unavailable when the displayed sample moments violate the
#   course guardrail; the dashboard never disables that check silently.
# - CVaR estimates can be noisy because they rely on relatively few observations in the tail.
# - Changing the lookback window changes the historical sample and can materially change reported risk.

# %% [markdown]
# ## Handoff
#
# Use the dashboard comparison as input to a risk memo that states the selected
# model, rejected alternatives, backtest evidence, stress loss, liquidity
# assumptions, limit owner, and escalation trigger.
