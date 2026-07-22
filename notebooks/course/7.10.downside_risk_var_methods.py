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
# # Downside Risk and VaR Methods
#
# Module: Derivatives and Risk Management
#
# ## Lesson summary
#
# This lab builds a reproducible workflow for downside risk, Value at Risk, and Expected Shortfall. It compares empirical, Gaussian, Cornish-Fisher, and volatility-weighted estimates on the same official-data portfolio return series {cite}`jorion2007var,mcneil2015quantitative`.
#
# ## Learning objectives
#
# By the end of this lab, students should be able to:
#
# - distinguish volatility from downside-only risk;
# - compute target semideviation and the Sortino ratio;
# - estimate historical, Gaussian, and Cornish-Fisher VaR;
# - explain why Expected Shortfall is more tail-sensitive than VaR;
# - use EWMA volatility to make historical simulation more responsive to recent market regimes.
#
# ## Prerequisites
#
# Complete VaR and Expected Shortfall Foundations first. Students should
# preserve the non-negative loss convention and understand why alpha is a lower
# return-tail probability rather than a confidence level.
#
# ## Risk metric definitions
#
# For target return $\tau$, target semideviation measures only downside observations:
#
# $$
# \sigma_-(\tau)=\sqrt{\frac{1}{T}\sum_{t=1}^{T}\min(r_t-\tau,0)^2}.
# $$
#
# For loss $L=-R$, Value at Risk is the loss quantile:
#
# $$
# v_\alpha=Q_{1-\alpha}(L),\qquad
# \operatorname{VaR}_{\alpha}
# =\max\left(0,-Q_\alpha(R)\right)
# =\max(0,v_\alpha).
# $$
#
# Expected Shortfall averages losses beyond the VaR threshold:
#
# $$
# \operatorname{ES}_{\alpha}
# =\max\left(0,-\mathbb{E}\left[R\mid R\leq Q_\alpha(R)\right]\right)
# =\max\left(0,\mathbb{E}\left[L\mid L\geq v_\alpha\right]\right).
# $$
#
# The Cornish-Fisher adjustment modifies a Gaussian quantile with sample skewness $S$ and excess kurtosis $K$:
#
# $$
# z_{CF}=z+\frac{1}{6}(z^2-1)S+\frac{1}{24}(z^3-3z)K-\frac{1}{36}(2z^3-5z)S^2.
# $$
#
# ## Setup

# %% tags=["setup", "hide-input"]
import pandas as pd
from scipy.stats import kurtosis, skew

from src.market_data import official_price_panel, returns_from_prices
from src.market_risk import (
    cornish_fisher_moment_report,
    cornish_fisher_var,
    ewma_volatility,
    expected_shortfall,
    gaussian_var,
    historical_var,
    sortino_ratio,
    target_semideviation,
    volatility_weighted_historical_var,
)

pd.options.display.float_format = "{:.6f}".format

# %% [markdown]
# ## Official portfolio returns
#
# The portfolio below uses log returns derived from the committed Banxico official price-like snapshot. The columns combine USD/MXN, UDI, and carry indexes built from official CETES, TIIE, and policy-rate series. This keeps downside-risk calculations reproducible without fabricating return histories.

# %%
price_panel = official_price_panel(start="2021-01-01", end="2026-06-05")
asset_returns = returns_from_prices(price_panel, method="log").dropna()

weights = pd.Series(
    {
        "usd_mxn": 0.25,
        "udi": 0.20,
        "cetes_28d_carry": 0.20,
        "tiie_28d_carry": 0.20,
        "policy_rate_carry": 0.15,
    },
    name="weight",
)
weights = weights.reindex(asset_returns.columns).fillna(0.0)
weights = weights / weights.sum()
portfolio_returns = asset_returns.dot(weights).rename("portfolio_return")

asset_returns.head()

# %% [markdown]
# **Output interpretation.** Each column is a real snapshot-backed return series. The weights are reindexed to the available columns before normalization, which prevents silent portfolio drift when the data schema changes.

# %% [markdown]
# ## Distribution diagnostics

# %%
pd.DataFrame(
    {
        "mean": asset_returns.mean(),
        "volatility": asset_returns.std(),
        "skewness": asset_returns.apply(lambda series: skew(series, bias=False)),
        "excess_kurtosis": asset_returns.apply(
            lambda series: kurtosis(series, fisher=True, bias=False)
        ),
    }
)

# %%
pd.Series(
    {
        "portfolio_mean": portfolio_returns.mean(),
        "portfolio_volatility": portfolio_returns.std(),
        "portfolio_skewness": skew(portfolio_returns, bias=False),
        "portfolio_excess_kurtosis": kurtosis(portfolio_returns, fisher=True, bias=False),
    },
    name="portfolio_diagnostics",
)

# %% [markdown]
# ## Downside risk
#
# Volatility penalizes positive and negative surprises symmetrically. Semideviation focuses only on returns below the selected target.

# %%
downside_report = pd.Series(
    {
        "daily_target_semideviation": target_semideviation(portfolio_returns, target=0.0),
        "annualized_target_semideviation": target_semideviation(
            portfolio_returns,
            target=0.0,
            periods_per_year=252,
        ),
        "sortino_ratio": sortino_ratio(portfolio_returns, target=0.0),
    },
    name="downside_report",
)

downside_report

# %% [markdown]
# **Output interpretation.** The Sortino ratio uses downside deviation rather than total volatility. If downside deviation is small because the sample contains few negative days, the ratio can look strong even when tail losses still matter.

# %% [markdown]
# ## VaR and Expected Shortfall
#
# All VaR and Expected Shortfall estimates are reported as non-negative loss
# numbers. For example, a daily VaR of `0.025` means a 2.5% one-day portfolio
# loss threshold.

# %%
alpha = 0.01
cornish_fisher_report = cornish_fisher_moment_report(portfolio_returns)
try:
    cornish_fisher_estimate = cornish_fisher_var(portfolio_returns, alpha=alpha)
    cornish_fisher_status = "available"
except ValueError as exc:
    cornish_fisher_estimate = float("nan")
    cornish_fisher_status = f"unavailable: {exc}"

risk_table = pd.Series(
    {
        "historical_var": historical_var(portfolio_returns, alpha=alpha),
        "gaussian_var": gaussian_var(portfolio_returns, alpha=alpha),
        "cornish_fisher_var": cornish_fisher_estimate,
        "volatility_weighted_historical_var": volatility_weighted_historical_var(
            portfolio_returns,
            alpha=alpha,
            lambda_=0.94,
        ),
        "expected_shortfall": expected_shortfall(portfolio_returns, alpha=alpha),
    },
    name="positive_daily_loss",
).to_frame()

risk_table

# %%
pd.concat(
    [
        cornish_fisher_report,
        pd.Series({"status": cornish_fisher_status}),
    ]
).rename("cornish_fisher_diagnostic")

# %% [markdown]
# **Output interpretation.** Differences across rows are model-risk evidence.
# Historical VaR reads the sample tail, Gaussian VaR imposes symmetry,
# Cornish-Fisher reacts to skewness and kurtosis, and volatility-weighted
# historical VaR gives more influence to the latest volatility state.
# Cornish-Fisher is reported as unavailable rather than extrapolated when the
# displayed sample moments exceed the course guardrail.

# %% [markdown]
# ## EWMA volatility state
#
# EWMA volatility is a simple way to make risk estimates respond faster after volatility shocks. The RiskMetrics convention often uses $\lambda=0.94$ for daily returns.

# %%
ewma_state = ewma_volatility(portfolio_returns, lambda_=0.94)

pd.DataFrame(
    {
        "portfolio_return": portfolio_returns,
        "ewma_volatility": ewma_state,
    }
).tail()

# %% [markdown]
# ## Interpretation checklist
#
# | Question | What to inspect |
# | --- | --- |
# | Is the distribution symmetric? | Skewness and large negative jumps |
# | Is Gaussian VaR plausible? | Difference between Gaussian and historical VaR |
# | Is tail loss material? | Gap between VaR and Expected Shortfall |
# | Is the latest volatility regime unusual? | EWMA volatility relative to unconditional volatility |
# | Is Cornish-Fisher stable? | Whether skewness and kurtosis are within a defensible range |
#
# ## Model limitations
#
# - VaR depends strongly on the selected horizon, confidence level, sign convention, and return distribution.
# - Historical methods reuse past losses and can miss new risks when market structure changes.
# - Expected Shortfall is more tail-sensitive than VaR, but it still inherits the sample and model assumptions used to estimate the tail.

# %% [markdown]
# ## Handoff
#
# Carry the selected tail models into the next lab. Estimation alone is not
# validation: VaR Backtesting and Stress Testing checks exception frequency,
# exception clustering, and deterministic scenario loss.

# %% [markdown] tags=["exercise"]
# ## Checkpoint exercise
#
# 1. Report daily volatility, target semideviation, 1% historical VaR, and 1%
#    ES for the portfolio in decimal-return units.
# 2. Report sample skewness, excess kurtosis, and whether Cornish-Fisher is
#    available under the displayed guardrail; do not override the guardrail.
# 3. Change only EWMA \(\lambda\) from 0.94 to 0.97, compare
#    volatility-weighted VaR, and explain the responsiveness-versus-memory
#    trade-off.
#

# %% [markdown] tags=["solution"]
# ```{dropdown} Suggested answer rubric
# A complete answer distinguishes symmetric volatility from downside metrics,
# reports all requested values and moments, preserves non-negative loss signs,
# respects the Cornish-Fisher availability result, and isolates lambda as the
# only changed assumption.
# ```
