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
# # Interactive GARCH Persistence Dashboard
#
# Module: Quantitative Methods and Financial Time Series
#
# ## Lesson summary
#
# This dashboard applies transparent GARCH(1,1) and EWMA recursions to
# provider-dated Banxico FIX returns. Instead of allowing invalid independent
# $\alpha$ and $\beta$ choices, readers set total persistence and the share
# assigned to recent shocks. The resulting paths are sensitivity scenarios, not
# maximum-likelihood estimates
# {cite}`engle1982autoregressive,bollerslev1986generalized`.
#
# ## Learning objectives
#
# By the end of this dashboard, readers should be able to:
#
# - distinguish shock reaction $\alpha$ from recursive memory $\beta$;
# - map persistence and shock share into a stationary GARCH parameter pair;
# - compare a GARCH filter with an EWMA filter under the same initial variance;
# - calculate the theoretical half-life of a variance shock; and
# - explain why a sensitivity path is not an estimated forecast.
#
# ## Prerequisites
#
# Lesson 2.5 supplies fitted-model interpretation and diagnostic requirements.
# This page varies assumptions around one observed USD/MXN FIX series; it does
# not mix FIX with UDI or constructed carry indices that have incompatible
# economic meanings and scales.
#
# ## Recursions and parameterization
#
# The dashboard assumes a zero conditional mean and filters returns through
#
# $$
# \sigma_t^2=\omega+\alpha r_{t-1}^2+\beta\sigma_{t-1}^2.
# $$
#
# Let $\rho=\alpha+\beta$ be persistence and let $s=\alpha/\rho$ be the shock
# share. Then $\alpha=\rho s$ and $\beta=\rho(1-s)$. Sliders constrain both
# $\rho$ and $s$ to $(0,1)$, so $\alpha\ge0$, $\beta\ge0$, and $\rho<1$ by
# construction. Given a long-run FIX-interval volatility $\bar\sigma$,
#
# $$
# \omega=\bar\sigma^2(1-\rho).
# $$
#
# EWMA uses
#
# $$
# \sigma_t^2=\lambda\sigma_{t-1}^2+(1-\lambda)r_{t-1}^2.
# $$
#
# Both filters start from $\bar\sigma^2$, so neither initial state uses future
# observations. The expected half-life of a variance shock, assuming no new
# shocks, is $h_{1/2}=\log(0.5)/\log(\rho)$ observations.

# %% [markdown]
# ## Setup

# %% tags=["setup", "hide-input"]
import os

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from ipywidgets import FloatSlider, IntSlider, interact

from src.dashboard_fallbacks import build_volatility_dashboard_fallback
from src.dashboards import build_volatility_dashboard
from src.market_data import banxico_daily_panel
from src.market_data_quality import log_returns
from src.market_risk import ewma_volatility
from src.time_series_diagnostics import (
    garch11_volatility_filter,
    garch_variance_half_life,
)

RUN_INTERACTIVE_WIDGETS = os.getenv("RUN_INTERACTIVE_WIDGETS", "1") == "1"
REQUESTED_START = "2021-01-01"
REQUESTED_END = "2026-06-05"

# %% [markdown]
# ## Provider-dated FIX return series

# %%
banxico = banxico_daily_panel(start=REQUESTED_START, end=REQUESTED_END)
fix_level = banxico["usd_mxn"].dropna().rename("usd_mxn_fix_mxn_per_usd")
OFFICIAL_RETURNS = log_returns(fix_level).rename("usd_mxn_fix_log_return")
calendar_gaps = fix_level.index.to_series().diff().dt.days.dropna()

pd.DataFrame(
    [
        {
            "provider_series": "Banxico SIE SF43718",
            "snapshot_generated_at": banxico.attrs["snapshot_generated_at"],
            "observed_start": fix_level.index.min().date().isoformat(),
            "observed_end": fix_level.index.max().date().isoformat(),
            "return_observations": len(OFFICIAL_RETURNS),
            "maximum_calendar_gap_days": int(calendar_gaps.max()),
            "return_unit": "log change per FIX publication interval",
            "observation_policy": banxico.attrs["observation_policy"],
        }
    ]
)

# %% [markdown]
# **Output interpretation.**
#
# Every modeled change joins two actually published FIX observations. “One
# step” is the next publication interval, which can cross a weekend or holiday.
# FIX is an official reference rate, not a guaranteed transaction price
# {cite}`banxicoSIE2025`.

# %%
def filtered_volatility_paths(
    *,
    long_run_volatility=0.01,
    persistence=0.98,
    shock_share=0.08,
    lambda_=0.94,
    periods=750,
):
    if not 0 < shock_share < 1:
        raise ValueError("shock_share must be strictly between 0 and 1")
    if not 0 < persistence < 1:
        raise ValueError("persistence must be strictly between 0 and 1")
    if long_run_volatility <= 0:
        raise ValueError("long_run_volatility must be positive")
    if periods < 2:
        raise ValueError("periods must be at least 2")

    alpha = persistence * shock_share
    beta = persistence * (1 - shock_share)
    omega = long_run_volatility**2 * (1 - persistence)
    selected_returns = OFFICIAL_RETURNS.tail(periods)
    if len(selected_returns) < 2:
        raise ValueError("The selected lookback has fewer than two returns")

    initial_variance = long_run_volatility**2
    filtered = pd.DataFrame(
        {
            "return": selected_returns,
            "garch_filtered_volatility": garch11_volatility_filter(
                selected_returns,
                omega=omega,
                alpha=alpha,
                beta=beta,
            ),
            "ewma_volatility": ewma_volatility(
                selected_returns,
                lambda_=lambda_,
                initial_variance=initial_variance,
            ),
        }
    )
    filtered.attrs.update(
        {
            "asset": "usd_mxn",
            "data_mode": "provider-dated snapshot",
            "sources": "Banxico SIE SF43718 snapshot",
            "method": (
                "zero-mean filters; "
                f"omega={omega:.8f}, alpha={alpha:.3f}, beta={beta:.3f}, "
                f"lambda={lambda_:.3f}, n={len(selected_returns)}"
            ),
        }
    )
    parameters = pd.Series(
        {
            "long_run_interval_volatility": long_run_volatility,
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "persistence": persistence,
            "shock_share_alpha_over_persistence": shock_share,
            "ewma_lambda": lambda_,
            "variance_shock_half_life_observations": garch_variance_half_life(
                persistence
            ),
            "lookback_observations": len(selected_returns),
        },
        name="value",
    )
    return filtered, parameters


# %% [markdown]
# ## Interactive dashboard
#
# Run this cell in JupyterLab with `uv run jupyter lab`. The publication build
# uses the Matplotlib fallback with the same data, parameter table, trace order,
# units, and conclusion.

# %% tags=["interactive"]
def plot_volatility_dashboard(
    long_run_volatility=0.01,
    persistence=0.98,
    shock_share=0.08,
    lambda_=0.94,
    periods=750,
):
    filtered, parameters = filtered_volatility_paths(
        long_run_volatility=long_run_volatility,
        persistence=persistence,
        shock_share=shock_share,
        lambda_=lambda_,
        periods=periods,
    )

    if RUN_INTERACTIVE_WIDGETS:
        figure = build_volatility_dashboard(
            filtered,
            asset="usd_mxn",
            persistence=persistence,
            return_axis_label="Return per FIX publication interval",
            volatility_axis_label="Volatility per FIX publication interval",
        )
    else:
        figure = build_volatility_dashboard_fallback(
            filtered,
            asset="usd_mxn",
            persistence=persistence,
            return_axis_label="Return per FIX publication interval",
            volatility_axis_label="Volatility per FIX publication interval",
        )
    display(parameters.to_frame())
    display(figure)
    if not RUN_INTERACTIVE_WIDGETS:
        plt.close(figure)


if RUN_INTERACTIVE_WIDGETS:
    interact(
        plot_volatility_dashboard,
        long_run_volatility=FloatSlider(
            value=0.01,
            min=0.004,
            max=0.025,
            step=0.001,
            readout_format=".3f",
            description="Long-run vol",
            continuous_update=False,
        ),
        persistence=FloatSlider(
            value=0.98,
            min=0.60,
            max=0.99,
            step=0.01,
            readout_format=".2f",
            description="Persistence",
            continuous_update=False,
        ),
        shock_share=FloatSlider(
            value=0.08,
            min=0.02,
            max=0.60,
            step=0.02,
            readout_format=".2f",
            description="Shock share",
            continuous_update=False,
        ),
        lambda_=FloatSlider(
            value=0.94,
            min=0.80,
            max=0.99,
            step=0.01,
            readout_format=".2f",
            description="EWMA lambda",
            continuous_update=False,
        ),
        periods=IntSlider(
            value=750,
            min=252,
            max=min(1250, len(OFFICIAL_RETURNS)),
            step=126,
            description="Lookback",
            continuous_update=False,
        ),
    )
else:
    plot_volatility_dashboard()

# %% [markdown]
# **Output interpretation.**
#
# A larger shock share moves persistence from $\beta$ toward $\alpha$, making
# the filter react more sharply to the latest squared return while preserving
# the same expected decay rate. Larger persistence lengthens variance-shock
# half-life. EWMA $\lambda$ controls a separate decay rule. The parameter table
# makes every displayed path reproducible.

# %% [markdown]
# ## Model limitations
#
# - The paths are zero-mean sensitivity filters, not maximum-likelihood fits.
# - The initial variance is a declared long-run assumption rather than an
#   estimate from future observations.
# - FIX publication intervals are not equally spaced in calendar time.
# - A theoretical half-life assumes no new shocks; it is not measured from a
#   realized path that continues to receive innovations.
# - Symmetric GARCH and EWMA omit direction-specific effects, jumps, liquidity,
#   parameter uncertainty, and regime changes.
# - The committed snapshot is reproducible but can differ from a later Banxico
#   vintage.

# %% [markdown] tags=["exercise"]
# ## Checkpoint exercise
#
# 1. Hold long-run volatility and shock share fixed. Compare persistence 0.78
#    with 0.98 and report the derived $\alpha$, $\beta$, and theoretical
#    variance-shock half-life for each scenario.
# 2. Explain why the half-life is an expectation under no new shocks rather than
#    an elapsed time measured from the observed maximum-return date.
# 3. Compare GARCH with EWMA at $\lambda=0.94$ and explain why visual agreement
#    does not validate either parameter set.

# %% tags=["solution"]
checkpoint_rows = []
for scenario_persistence in (0.78, 0.98):
    _, scenario_parameters = filtered_volatility_paths(
        long_run_volatility=0.01,
        persistence=scenario_persistence,
        shock_share=0.08,
        lambda_=0.94,
        periods=750,
    )
    checkpoint_rows.append(
        {
            "persistence": scenario_persistence,
            "alpha": scenario_parameters["alpha"],
            "beta": scenario_parameters["beta"],
            "variance_shock_half_life_observations": scenario_parameters[
                "variance_shock_half_life_observations"
            ],
        }
    )
pd.DataFrame(checkpoint_rows).set_index("persistence")

# %% [markdown] tags=["solution"]
# ```{dropdown} Suggested answer and interpretation
# The higher-persistence scenario has a much longer theoretical variance-shock
# half-life even though the shock share is unchanged. New observed returns keep
# updating both filters, so a realized path cannot identify pure decay from one
# shock. Similar-looking GARCH and EWMA curves show only that two recursions can
# summarize this sample similarly; they do not establish calibration or
# forecasting superiority.
# ```

# %% [markdown]
# ## Handoff
#
# Carry the distinction between estimated parameters and transparent scenarios
# into Module 7 — Derivatives and Risk Management, where positive-loss VaR is
# backtested and complemented by deterministic stress scenarios.
