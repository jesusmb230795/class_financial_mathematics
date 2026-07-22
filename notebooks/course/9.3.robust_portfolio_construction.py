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
# # Robust Portfolio Construction
#
# Module: Portfolio Management, Asset Allocation, and Performance
#
# ## Lesson summary
#
# This lab extends classical mean-variance optimization with robust portfolio
# tools. Students compare sample covariance against shrinkage estimators, estimate
# an exploratory factor sensitivity with HAC standard errors, construct risk
# parity weights, and preview Hierarchical Risk Parity. These are estimation-risk
# tools, not guarantees of stable allocations {cite}`damodaran2012investment`.
#
# ## Learning objectives
#
# By the end of this lab, students should be able to:
#
# - explain why sample covariance becomes unstable in small samples;
# - compare Ledoit-Wolf and OAS shrinkage estimates;
# - estimate a documented factor sensitivity with heteroskedasticity and autocorrelation consistent errors;
# - construct a long-only risk parity portfolio;
# - describe why HRP avoids full covariance matrix inversion.
#
# ## Prerequisites
#
# Complete the investment-policy and analytical-frontier lessons first. Readers
# should be able to form returns from aligned price observations, estimate a
# covariance matrix, interpret unconstrained portfolio weights, and distinguish
# an exploratory factor sensitivity from an economically valid CAPM beta. All
# sample windows must be documented and free of future observations.
#
# ## Portfolio equations
#
# Classical minimum-variance allocation solves:
#
# $$
# \min_w w^\top\Sigma w
# \quad\text{subject to}\quad
# \mathbf{1}^\top w=1.
# $$
#
# A generic linear sensitivity to a documented factor is:
#
# $$
# b_i=\frac{\operatorname{Cov}(r_i,f)}{\operatorname{Var}(f)}.
# $$
#
# This becomes a CAPM beta only when $f$ is the excess return on a broad,
# investable market portfolio and the asset return is also measured in excess of
# the same risk-free rate. The Banxico rate/FX panel below does not meet that
# economic contract, so the coefficient is labeled **factor sensitivity**.
#
# Risk parity targets balanced marginal contributions to portfolio volatility:
#
# $$
# \operatorname{RC}_i=\frac{w_i(\Sigma w)_i}{\sqrt{w^\top\Sigma w}}.
# $$
#
# ## Setup

# %% tags=["setup", "hide-input"]
import numpy as np
import pandas as pd

from src.market_data import official_price_panel, returns_from_prices
from src.portfolio_optimization import (
    annualized_mean_returns,
    factor_sensitivity,
    global_minimum_variance_weights,
    hierarchical_risk_parity_weights,
    ledoit_wolf_covariance,
    oas_covariance,
    risk_contribution_percentages,
    risk_parity_weights,
    rolling_factor_sensitivity,
    sample_covariance,
)

# %% [markdown]
# ## Official risk-factor panel and estimation windows
#
# The sample uses log changes from the committed Banxico official price-like
# snapshot. The columns mix FX and synthetic carry indices, so portfolio weights
# are classroom diagnostics rather than implementable allocations. We deliberately
# compare a 60-observation window with a 756-observation window to make
# small-sample instability observable.

# %%
price_panel = official_price_panel(start="2021-01-01", end="2026-06-05")
full_returns = returns_from_prices(price_panel, method="log").dropna().tail(756)
returns = full_returns.tail(60)

returns.head()

# %% [markdown]
# ## Sample covariance versus shrinkage

# %%
expected_returns = annualized_mean_returns(returns)
sample_cov = sample_covariance(returns)
full_sample_cov = sample_covariance(full_returns)
lw_cov, lw_shrinkage = ledoit_wolf_covariance(returns)
oas_cov, oas_shrinkage = oas_covariance(returns)

pd.Series(
    {
        "ledoit_wolf_shrinkage": lw_shrinkage,
        "oas_shrinkage": oas_shrinkage,
        "sample_60_condition_number": np.linalg.cond(sample_cov),
        "sample_756_condition_number": np.linalg.cond(full_sample_cov),
        "ledoit_wolf_condition_number": np.linalg.cond(lw_cov),
        "oas_condition_number": np.linalg.cond(oas_cov),
    }
)

# %% [markdown]
# ## Minimum-variance weights under different covariance estimates

# %%
gmvp_comparison = pd.concat(
    {
        "sample_covariance": global_minimum_variance_weights(sample_cov),
        "ledoit_wolf": global_minimum_variance_weights(lw_cov),
        "oas": global_minimum_variance_weights(oas_cov),
    },
    axis=1,
)

gmvp_comparison

# %% [markdown]
# ## Exploratory factor sensitivity with HAC standard errors
#
# The factor is the equally weighted change across the documented rate/carry
# columns **excluding USD/MXN**, the dependent variable. It is a transparent
# composite for regression mechanics, not a market portfolio and therefore not a
# CAPM proxy.

# %%
composite_factor = (
    returns.drop(columns="usd_mxn").mean(axis=1).rename("rate_carry_composite_factor")
)

factor_sensitivity(
    returns["usd_mxn"],
    composite_factor,
)

# %%
rolling_factor_sensitivity(
    full_returns["usd_mxn"],
    full_returns.drop(columns="usd_mxn").mean(axis=1),
    window=126,
).dropna().tail()

# %% [markdown]
# ## Risk parity
#
# Risk parity avoids direct expected-return forecasts and targets equal contributions to total portfolio volatility.

# %%
rp_weights = risk_parity_weights(lw_cov)
rp_risk_contribution = risk_contribution_percentages(rp_weights, lw_cov)

pd.DataFrame(
    {
        "risk_parity_weight": rp_weights,
        "risk_contribution_pct": rp_risk_contribution,
    }
)

# %% [markdown]
# ## Hierarchical Risk Parity preview
#
# HRP uses clustering and recursive bisection instead of a global inverse covariance matrix. It is especially useful as a robustness benchmark when classical Markowitz weights are unstable.

# %%
hrp_weights = hierarchical_risk_parity_weights(returns)

pd.DataFrame(
    {
        "risk_parity": rp_weights,
        "hrp": hrp_weights,
    }
)

# %% [markdown]
# ## Interpretation checklist
#
# | Question | What to inspect |
# | --- | --- |
# | Is covariance estimation stable? | Condition number before and after shrinkage |
# | Which shrinkage estimator is preferable? | Distributional assumptions and sample size |
# | Is the factor sensitivity statistically useful? | HAC p-value and rolling stability |
# | Does risk parity work as intended? | Risk contribution percentages |
# | Does HRP differ materially? | Weight concentration and sector clustering intuition |
#
# ## Model limitations
#
# - Shrinkage estimators stabilize covariance matrices but do not eliminate model risk in expected returns or factor structure.
# - A factor sensitivity is not a CAPM beta without an economically valid market
#   proxy and consistent excess-return convention.
# - Risk parity and HRP reduce some concentration problems, but they can still be unstable when correlations shift.

# %% [markdown] tags=["exercise"]
# ## Checkpoint exercise
#
# Re-estimate the covariance inputs with 40, 60, and 252 observations.
#
# 1. Report the sample and Ledoit-Wolf condition numbers and shrinkage intensity
#    for each window.
# 2. Compare the resulting GMVP weights and identify the asset with the largest
#    absolute weight change.
# 3. Estimate USD/MXN sensitivity to the documented composite factor, then
#    explain precisely why the coefficient is not a CAPM beta.
# 4. Verify that the risk-parity contributions sum to 100% within numerical
#    tolerance.
#

# %% [markdown] tags=["solution"]
# ```{dropdown} Suggested answer rubric
# A complete answer quantifies conditioning and weight instability across all
# three windows, distinguishes statistical sensitivity from CAPM beta, and
# verifies both full investment and the risk-budget total.
# ```

# %% [markdown]
# ## Handoff
#
# The interactive frontier dashboard exposes the input sensitivity documented in
# this lesson and applies a positive-semidefinite guardrail to correlation
# scenarios. Compare its outputs with the shrinkage, risk-parity, and HRP
# benchmarks before proposing any allocation under the IPS.
