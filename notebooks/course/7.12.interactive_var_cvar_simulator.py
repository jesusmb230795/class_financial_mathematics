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
# # Interactive VaR and Expected Shortfall Dashboard
#
# Module: Derivatives and Risk Management
#
# ## Lesson summary
#
# This dashboard applies several tail estimators to one documented simple-return
# series. Students vary lower-tail probability, lookback length, and exponentially
# weighted moving-average (EWMA) decay to see which reported differences come from
# data selection and which come from model choice {cite}`jorion2007var,mcneil2015quantitative,riskMetrics1996`.
#
# ## Learning objectives
#
# By the end of this dashboard, students should be able to:
#
# - explain Value at Risk (VaR) as a loss threshold;
# - explain Expected Shortfall (ES) as an average over a fixed tail probability mass;
# - compare historical, Gaussian, Cornish-Fisher, and volatility-weighted VaR;
# - separate tail probability from confidence level; and
# - explain why window selection and model assumptions change reported risk.
#
# ## Prerequisites
#
# Complete the estimation and backtesting labs first. Students should understand
# non-negative loss signs, finite-sample ES, the Cornish-Fisher guardrail, and why
# non-rejection in a coverage test is not proof of model correctness.
#
# ## Tail metric definitions
#
# For simple portfolio return $R$ over one observed US trading interval, the
# dashboard reports positive loss magnitudes:
#
# $$
# \operatorname{VaR}_{\alpha}(R)
# =\max\{0,-Q_\alpha(R)\},
# $$
#
# $$
# \operatorname{ES}_{\alpha}(R)
# =\max\left\{0,-\frac{1}{\alpha}
# \int_0^\alpha Q_u(R)\,du\right\}.
# $$
#
# ES is sometimes called Conditional VaR (CVaR), but that alias can obscure the
# boundary-mass treatment in a discrete sample. This notebook therefore uses
# “Expected Shortfall” for the quantile-integral estimator
# {cite}`acerbiTasche2002,rockafellarUryasev2002`.
#
# The confidence convention is $1-\alpha$. Changing lookback length changes the
# empirical distribution of $R$; changing $\lambda$ changes only the EWMA-scaled
# historical estimate.
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
from src.market_data import nasdaq_stock_price_panel, returns_from_prices
from src.market_risk import (
    cornish_fisher_moment_report,
    cornish_fisher_var,
    expected_shortfall,
    gaussian_var,
    historical_var,
    rebalanced_portfolio_returns,
    volatility_weighted_historical_var,
)

RUN_INTERACTIVE_WIDGETS = os.getenv("RUN_INTERACTIVE_WIDGETS", "1") == "1"
pd.options.display.float_format = "{:.6f}".format

# %% [markdown]
# ## Data and portfolio contract
#
# The committed snapshot contains provider-adjusted closes in USD for AAPL,
# MSFT, NVDA, AMZN, and GOOGL from 2021-01-04 through 2026-06-05. Returns are
# simple changes between consecutive provider trading dates. Each asset receives
# a fixed 20% weight, restored after every observed interval; this is not a
# buy-and-hold portfolio {cite}`yfinance2025,yahooFinanceCoverage2026,yahooTerms2026`.

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


def portfolio_return_window(lookback=1000):
    """Return the latest documented observations without changing their units."""
    bounded_lookback = max(250, min(int(lookback), len(portfolio_returns)))
    window = portfolio_returns.tail(bounded_lookback).copy()
    window.attrs = dict(portfolio_returns.attrs)
    window.attrs["lookback"] = bounded_lookback
    return window


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
# The provider, field, currency, date window, transformation, and portfolio rule
# make the calculation inspectable. Snapshot provenance does not independently
# establish redistribution or downstream-use rights.
#
# ## Interactive dashboard
#
# Run this cell in JupyterLab with `uv run jupyter lab`. The publication build
# sets `RUN_INTERACTIVE_WIDGETS=0`, so the book renders a deterministic static
# view instead of browser-dependent widget controls.


# %% tags=["interactive"] mystnb={"image": {"alt": "Interactive or static two-panel dashboard for an equal-weight US equity portfolio: a positive-loss histogram marks the historical Value at Risk threshold and Expected Shortfall tail mean, while a dot plot compares all available non-negative loss estimates."}}
def plot_tail_risk_dashboard(
    alpha=0.01,
    lambda_=0.94,
    lookback=1000,
):
    returns = portfolio_return_window(lookback=lookback)
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
        name="positive_loss_per_observed_US_trading_interval",
    )

    builder = (
        build_var_cvar_dashboard if RUN_INTERACTIVE_WIDGETS else build_var_cvar_dashboard_fallback
    )
    figure = builder(returns, risk_metrics)
    display(figure)
    if not RUN_INTERACTIVE_WIDGETS:
        plt.close(figure)
    display(risk_metrics.to_frame())
    display(
        pd.Series(
            {
                "lower_tail_probability_alpha": alpha,
                "confidence_level_one_minus_alpha": 1 - alpha,
                "lookback_observations": len(returns),
                "ewma_lambda": lambda_,
            },
            name="dashboard_controls",
        )
    )
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
        plot_tail_risk_dashboard,
        alpha=FloatSlider(
            value=0.01,
            min=0.005,
            max=0.10,
            step=0.005,
            readout_format=".3f",
            description="tail alpha",
        ),
        lambda_=FloatSlider(
            value=0.94,
            min=0.80,
            max=0.99,
            step=0.01,
            readout_format=".2f",
            description="EWMA lambda",
        ),
        lookback=IntSlider(
            value=1000,
            min=500,
            max=len(portfolio_returns),
            step=250,
            description="observations",
        ),
    )
else:
    plot_tail_risk_dashboard()

# %% [markdown]
# ## Interpretation protocol
#
# 1. Read $\alpha$ and $1-\alpha$ before comparing thresholds.
# 2. Confirm the selected observation count and per-interval units.
# 3. Compare empirical ES with historical VaR to assess tail severity beyond the boundary.
# 4. Compare closed-form Gaussian and historical VaR to expose distributional sensitivity.
# 5. Treat a missing Cornish-Fisher estimate as a visible guardrail result.
# 6. Return to the preceding backtest and stress evidence before choosing a model.
#
# ## Model limitations
#
# - The observed sample cannot contain future regimes or every plausible stress.
# - Equal-weight interval rebalancing omits costs, taxes, FX conversion, capacity,
#   and investor-specific constraints.
# - Parametric VaR can understate losses when tails, dependence, or volatility
#   dynamics differ from the assumed model.
# - ES at small $\alpha$ depends on few effective tail observations.
# - Slider sensitivity is exploratory evidence, not backtesting or regulatory validation.
# - Provider-adjusted closes inherit the provider's corporate-action treatment;
#   redistribution rights were not independently verified.
#
# ## Handoff
#
# Use this dashboard only as one input to a risk memo. The memo should state the
# chosen model, rejected alternatives, backtest decisions, stress loss, liquidity
# assumptions, portfolio implementation rule, limit owner, and escalation trigger.
