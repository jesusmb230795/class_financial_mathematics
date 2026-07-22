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
# # Interactive Efficient Frontier Dashboard
#
# Module: Portfolio Management, Asset Allocation, and Performance
#
# ## Lesson summary
#
# This dashboard lets students change expected returns, correlations, and the
# risk-free rate to see how the efficient frontier, global minimum variance
# portfolio, and tangency portfolio respond {cite}`sharpe1994ratio,damodaran2012investment`.
#
# ## Learning objectives
#
# By the end of this dashboard, students should be able to:
#
# - connect expected returns and covariance to frontier geometry;
# - identify the global minimum variance portfolio;
# - interpret the tangency portfolio as the highest Sharpe risky allocation;
# - explain why correlation assumptions change diversification benefits;
# - distinguish analytical intuition from implementable allocation constraints.
#
# ## Prerequisites
#
# Complete the investment-policy, analytical-frontier, and robust-construction
# lessons first. Readers should understand covariance, global-minimum-variance
# and tangency portfolios, weight constraints, and positive-semidefinite
# correlation matrices. Slider values are scenario assumptions, not forecasts or
# automatically investable recommendations.
#
# ## Frontier equations
#
# For weights $w$, expected returns $\mu$, covariance matrix $\Sigma$, and risk-free rate $r_f$:
#
# $$
# \mu_p=w^\top\mu,\qquad \sigma_p=\sqrt{w^\top\Sigma w}.
# $$
#
# The tangency portfolio maximizes the Sharpe ratio:
#
# $$
# \max_w \frac{w^\top\mu-r_f}{\sqrt{w^\top\Sigma w}}
# \quad\text{subject to}\quad
# \mathbf{1}^\top w=1.
# $$
#
# The dashboard changes $\mu$, correlations, and $r_f$ to show how these equations reshape the frontier.
#
# ## Setup

# %% tags=["setup", "hide-input"]
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import FloatSlider, interact

from src.dashboard_fallbacks import build_efficient_frontier_dashboard_fallback
from src.dashboards import build_efficient_frontier_dashboard
from src.portfolio_optimization import (
    project_correlation_to_psd,
    validate_covariance_matrix,
)

RUN_INTERACTIVE_WIDGETS = os.getenv("RUN_INTERACTIVE_WIDGETS", "1") == "1"

# %% [markdown]
# ## Frontier inputs

# %%
assets = ["mexican_equity", "global_equity", "mxn_bond", "inflation_linked_bond"]
base_expected_returns = pd.Series(
    [0.12, 0.10, 0.07, 0.085],
    index=assets,
    name="expected_return",
)
base_volatility = pd.Series([0.22, 0.17, 0.025, 0.075], index=assets, name="volatility")


# %%
def covariance_from_assumptions(
    equity_correlation=0.55,
    bond_correlation=0.25,
    equity_bond_correlation=0.15,
):
    raw_correlation = pd.DataFrame(
        [
            [1.00, equity_correlation, equity_bond_correlation, equity_bond_correlation],
            [equity_correlation, 1.00, equity_bond_correlation, equity_bond_correlation],
            [equity_bond_correlation, equity_bond_correlation, 1.00, bond_correlation],
            [equity_bond_correlation, equity_bond_correlation, bond_correlation, 1.00],
        ],
        index=assets,
        columns=assets,
    )
    correlation = project_correlation_to_psd(raw_correlation)
    covariance = correlation.mul(base_volatility, axis=0).mul(
        base_volatility,
        axis=1,
    )
    validate_covariance_matrix(covariance)
    covariance.attrs["raw_minimum_eigenvalue"] = float(np.linalg.eigvalsh(raw_correlation).min())
    covariance.attrs["projected_minimum_eigenvalue"] = float(np.linalg.eigvalsh(correlation).min())
    covariance.attrs["correlation_projection_norm"] = float(
        np.linalg.norm(correlation - raw_correlation)
    )
    return covariance


# %% [markdown]
# The sliders can propose three pairwise correlations that do not form a valid
# joint correlation matrix. Before optimization, the dashboard projects that
# scenario to the nearest positive-semidefinite correlation matrix by eigenvalue
# clipping and unit-diagonal rescaling. The diagnostics disclose when the
# projection changes the user's raw assumptions.


# %% [markdown]
# ## Interactive dashboard
#
# Run this cell in JupyterLab with `uv run jupyter lab`.


# %% tags=["interactive"]
def plot_efficient_frontier_dashboard(
    mexican_equity_return=0.12,
    global_equity_return=0.10,
    risk_free_rate=0.065,
    equity_correlation=0.55,
    bond_correlation=0.25,
    equity_bond_correlation=0.15,
):
    expected_returns = base_expected_returns.copy()
    expected_returns["mexican_equity"] = mexican_equity_return
    expected_returns["global_equity"] = global_equity_return
    covariance = covariance_from_assumptions(
        equity_correlation=equity_correlation,
        bond_correlation=bond_correlation,
        equity_bond_correlation=equity_bond_correlation,
    )

    builder = (
        build_efficient_frontier_dashboard
        if RUN_INTERACTIVE_WIDGETS
        else build_efficient_frontier_dashboard_fallback
    )
    figure, weights = builder(
        expected_returns,
        covariance,
        risk_free_rate=risk_free_rate,
    )
    display(figure)
    if not RUN_INTERACTIVE_WIDGETS:
        plt.close(figure)
    display(weights)
    display(
        pd.Series(
            {
                "raw_minimum_correlation_eigenvalue": covariance.attrs["raw_minimum_eigenvalue"],
                "projected_minimum_correlation_eigenvalue": covariance.attrs[
                    "projected_minimum_eigenvalue"
                ],
                "correlation_projection_frobenius_norm": covariance.attrs[
                    "correlation_projection_norm"
                ],
            },
            name="PSD guardrail diagnostics",
        )
    )


if RUN_INTERACTIVE_WIDGETS:
    interact(
        plot_efficient_frontier_dashboard,
        mexican_equity_return=FloatSlider(
            value=0.12, min=0.04, max=0.22, step=0.005, readout_format=".3f"
        ),
        global_equity_return=FloatSlider(
            value=0.10, min=0.04, max=0.20, step=0.005, readout_format=".3f"
        ),
        risk_free_rate=FloatSlider(
            value=0.065, min=0.00, max=0.14, step=0.005, readout_format=".3f"
        ),
        equity_correlation=FloatSlider(
            value=0.55, min=-0.20, max=0.95, step=0.05, readout_format=".2f"
        ),
        bond_correlation=FloatSlider(
            value=0.25, min=-0.20, max=0.95, step=0.05, readout_format=".2f"
        ),
        equity_bond_correlation=FloatSlider(
            value=0.15, min=-0.40, max=0.80, step=0.05, readout_format=".2f"
        ),
    )
else:
    plot_efficient_frontier_dashboard()

# %% [markdown]
# ## Model limitations
#
# - The dashboard shows sensitivity to selected inputs; it is not an optimizer with transaction costs, taxes, or mandates.
# - Small changes in expected returns can move the tangency portfolio sharply.
# - Correlation scenarios are stylized and may not capture crisis-period dependence.
# - PSD projection creates a mathematically admissible scenario, but a large
#   projection norm is evidence that the original slider assumptions were not
#   jointly coherent.

# %% [markdown] tags=["exercise"]
# ## Checkpoint exercise
#
# Set equity correlation to 0.95, bond correlation to 0.95, and equity-bond
# correlation to -0.40.
#
# 1. Report the minimum eigenvalue before and after the PSD guardrail and the
#    projection norm.
# 2. Verify that the projected covariance has no materially negative eigenvalue
#    and that both GMVP and tangency weights sum to one.
# 3. Compare GMVP volatility against the baseline slider values.
# 4. Explain why projection makes the calculation admissible but does not make
#    the raw economic assumptions plausible.
#

# %% [markdown] tags=["solution"]
# ```{dropdown} Suggested answer rubric
# A complete answer reports eigenvalues and volatility with units, verifies
# full-investment constraints, and distinguishes a numerical PSD repair from an
# empirically defensible correlation model.
# ```

# %% [markdown]
# ## Handoff
#
# The capstone combines investment policy, macro scenarios, valuation, fixed
# income, derivatives, alternatives, and portfolio analytics in one
# recommendation. Export the selected allocation together with raw assumptions,
# PSD diagnostics, constraints, benchmark, transaction-cost treatment, and
# sensitivity results so the decision remains reproducible and reviewable.
