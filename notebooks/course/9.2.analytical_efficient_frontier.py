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
# # Analytical Efficient Frontier
#
# Module: Portfolio Management, Asset Allocation, and Performance
#
# ## Lesson summary
#
# This lab develops a fully analytical efficient-frontier workflow. Students
# compute Merton's constants, the global minimum variance portfolio, efficient
# target-return portfolios, and the tangency portfolio. Risk-adjusted comparisons
# use a risk-free rate with the same currency, horizon, and annual compounding as
# the expected returns {cite}`sharpe1994ratio,damodaran2012investment`.
#
# ## Learning objectives
#
# By the end of this lab, students should be able to:
#
# - express portfolio return and variance with vector notation;
# - compute the constants that define the analytical efficient frontier;
# - derive the global minimum variance portfolio;
# - compute the tangency portfolio for a selected risk-free rate;
# - distinguish unconstrained analytical weights from real-world long-only constraints.
#
# ## Prerequisites
#
# Complete the investment-policy and asset-allocation lesson first. Readers
# should understand annualized returns and volatility, covariance matrices,
# matrix multiplication, and the role of a currency- and horizon-matched
# risk-free proxy. Expected returns, covariance, and the risk-free rate must use
# one annualization and compounding convention.
#
# ## Setup

# %% tags=["setup", "hide-input"]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.portfolio_optimization import (
    efficient_frontier_variance,
    efficient_frontier_weights,
    global_minimum_variance_weights,
    merton_constants,
    portfolio_return,
    portfolio_volatility,
    tangency_weights,
)

# %% [markdown]
# ## Annualized inputs
#
# The example uses stylized annualized inputs for four asset classes. In a production workflow, these inputs would come from a documented data pipeline and would be stress-tested for estimation error.

# %%
assets = ["mexican_equity", "global_equity", "cetes_proxy", "inflation_linked_bond"]

expected_returns = pd.Series(
    [0.125, 0.105, 0.072, 0.087],
    index=assets,
    name="expected_return",
)
volatility = pd.Series(
    [0.22, 0.17, 0.018, 0.075],
    index=assets,
    name="volatility",
)
correlation = pd.DataFrame(
    [
        [1.00, 0.62, 0.05, 0.25],
        [0.62, 1.00, 0.02, 0.18],
        [0.05, 0.02, 1.00, 0.30],
        [0.25, 0.18, 0.30, 1.00],
    ],
    index=assets,
    columns=assets,
)
covariance = correlation.mul(volatility, axis=0).mul(volatility, axis=1)

covariance

# %% [markdown]
# ## Merton constants
#
# For unconstrained mean-variance optimization, the entire frontier is summarized by four constants:
#
# $$
# A = \mathbf{1}^{T}\Sigma^{-1}\mathbf{1},\quad
# B = \mathbf{1}^{T}\Sigma^{-1}\mu,\quad
# C = \mu^{T}\Sigma^{-1}\mu,\quad
# D = AC - B^2.
# $$

# %%
merton_constants(expected_returns, covariance)

# %% [markdown]
# ## Global minimum variance portfolio

# %%
gmvp = global_minimum_variance_weights(covariance)

pd.DataFrame(
    {
        "weight": gmvp,
        "expected_return": expected_returns,
        "volatility": volatility,
    }
)

# %%
pd.Series(
    {
        "gmvp_expected_return": portfolio_return(gmvp, expected_returns),
        "gmvp_volatility": portfolio_volatility(gmvp, covariance),
    }
)

# %% [markdown]
# ## Efficient frontier curve

# %%
target_returns = np.linspace(0.065, 0.135, 60)
frontier_volatility = np.sqrt(
    efficient_frontier_variance(target_returns, expected_returns, covariance)
)

frontier = pd.DataFrame(
    {
        "target_return": target_returns,
        "frontier_volatility": frontier_volatility,
    }
)

frontier.head()

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(frontier["frontier_volatility"], frontier["target_return"], label="Efficient frontier")
ax.scatter(
    portfolio_volatility(gmvp, covariance),
    portfolio_return(gmvp, expected_returns),
    label="GMVP",
)
ax.set_xlabel("Annualized volatility")
ax.set_ylabel("Expected return")
ax.legend()
ax.grid(True, alpha=0.3)

# %% [markdown]
# ## Tangency portfolio
#
# The tangency portfolio maximizes the Sharpe ratio relative to the chosen risk-free rate. In Mexican peso examples, a short sovereign rate such as CETES is a common proxy, but the exact series must be documented.

# %%
risk_free_rate = 0.065
tangency = tangency_weights(expected_returns, covariance, risk_free_rate=risk_free_rate)

pd.DataFrame(
    {
        "gmvp": gmvp,
        "tangency": tangency,
    }
)

# %%
pd.Series(
    {
        "tangency_expected_return": portfolio_return(tangency, expected_returns),
        "tangency_volatility": portfolio_volatility(tangency, covariance),
        "tangency_sharpe": (portfolio_return(tangency, expected_returns) - risk_free_rate)
        / portfolio_volatility(tangency, covariance),
    }
)

# %% [markdown]
# ## Target-return weights

# %%
target_return = 0.10
target_weights = efficient_frontier_weights(target_return, expected_returns, covariance)

pd.Series(
    {
        "weight_sum": target_weights.sum(),
        "expected_return": portfolio_return(target_weights, expected_returns),
        "volatility": portfolio_volatility(target_weights, covariance),
    }
)

# %%
target_weights

# %% [markdown]
# ## Interpretation checklist
#
# | Question | What to inspect |
# | --- | --- |
# | Is the GMVP economically reasonable? | Weight concentration and negative weights |
# | Is the target return feasible? | Whether target weights require leverage or shorting |
# | Is the risk-free proxy defensible? | Source, frequency, currency, and compounding |
# | Is the tangency portfolio stable? | Sensitivity to expected-return assumptions |
# | What constraint is missing? | Long-only, maximum weight, turnover, or regulatory limits |
#
# ## Model limitations
#
# - Mean-variance optimization is highly sensitive to expected-return and covariance estimates.
# - The analytical frontier assumes a single-period problem with stable inputs and frictionless rebalancing.
# - Unconstrained solutions can produce allocations that are mathematically efficient but operationally unrealistic.

# %% [markdown] tags=["exercise"]
# ## Checkpoint exercise
#
# Increase the Mexican-equity expected return by 200 basis points while leaving
# covariance and the 6.5% annually compounded risk-free rate unchanged.
#
# 1. Recompute the GMVP, tangency portfolio, and 10% target-return portfolio.
# 2. Verify that every weight vector sums to one and that the target portfolio
#    reaches 10% to numerical tolerance.
# 3. Report the baseline and shocked tangency weights, annual volatility, and
#    Sharpe ratio.
# 4. Explain why the GMVP is unchanged while the tangency portfolio moves.
#

# %% [markdown] tags=["solution"]
# ```{dropdown} Suggested answer rubric
# A complete answer verifies all constraints, reports return and volatility as
# annual rates, quantifies the weight changes, and connects the contrast to the
# fact that GMVP uses covariance but not expected returns.
# ```

# %% [markdown]
# ## Handoff
#
# The robust-construction lesson tests how these analytical allocations change
# under small samples, covariance shrinkage, factor diagnostics, and alternative
# risk budgets. Carry forward the IPS constraints and treat unconstrained
# frontier weights as a benchmark, not an approved portfolio.
