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
# # Interactive Option Pricing Dashboard
#
# Module: Derivatives and Risk Management
#
# ## Lesson summary
#
# This notebook creates an interactive Black-Scholes-Merton pricer for European
# options. Students can move the spot price, strike, maturity, continuously
# compounded risk-free rate, continuous dividend yield, and volatility to
# observe option prices, payoff diagrams, and Greeks
# {cite}`blackScholes1973,merton1973,hull2022options`.
#
# The objective is to connect the formula with economic intuition: moneyness, time value, volatility exposure, and local hedging sensitivity.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - compute European call and put prices with Black-Scholes-Merton;
# - explain how spot, strike, maturity, rates, dividend yield, and volatility
#   affect option value;
# - interpret Delta, Gamma, Vega, Theta, and Rho;
# - verify put-call parity interactively;
# - use payoff and price curves to explain nonlinear exposure.
#
# ## Prerequisites
#
# Complete the European pricing, numerical-pricing, and implied-volatility
# lessons first. Students should be able to interpret every Black-Scholes input,
# state the units and hedge sign of each Greek, and treat slider comparisons as
# scenarios rather than statistical validation.
#
# ## Black-Scholes-Merton formulas and units
#
# Let $S_0$ and $K$ be in the same currency units per underlying unit,
# $r$ and $q$ be continuously compounded annual decimal rates, $\sigma$ be
# annualized decimal volatility, and $T$ be years. With $\Phi$ denoting the
# standard normal cumulative distribution function,
#
# $$
# d_1=\frac{\ln(S_0/K)+(r-q+\sigma^2/2)T}{\sigma\sqrt{T}},
# \qquad
# d_2=d_1-\sigma\sqrt{T}.
# $$
#
# For a call and a put:
#
# $$
# C=S_0e^{-qT}\Phi(d_1)-Ke^{-rT}\Phi(d_2),
# $$
#
# $$
# P=Ke^{-rT}\Phi(-d_2)-S_0e^{-qT}\Phi(-d_1).
# $$
#
# The local sensitivities shown in the dashboard include:
#
# $$
# \Delta=\frac{\partial V}{\partial S_0},\qquad
# \Gamma=\frac{\partial^2 V}{\partial S_0^2},\qquad
# \nu=\frac{\partial V}{\partial \sigma},\qquad
# \rho_r=\frac{\partial V}{\partial r}.
# $$
#
# The implementation reports vega and interest-rate rho for a **one percentage
# point** change, so it divides derivatives taken with respect to decimal
# $\sigma$ and $r$ by 100. Theta is reported per calendar day. These scaled
# quantities must not be combined with decimal shocks without conversion.
#
# ## Python setup

# %% tags=["setup", "hide-input"]
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import Dropdown, FloatSlider, interact

from src.dashboard_fallbacks import build_black_scholes_dashboard_fallback
from src.dashboards import build_black_scholes_dashboard
from src.derivatives import (
    black_scholes_greeks,
    black_scholes_price,
    option_payoff,
    put_call_parity_gap,
)

RUN_INTERACTIVE_WIDGETS = os.getenv("RUN_INTERACTIVE_WIDGETS", "1") == "1"


# %% [markdown]
# ## Interactive dashboard
#
# Run this cell in JupyterLab with `uv run jupyter lab`. The displayed curves
# are deterministic model scenarios, not observed option quotes or a calibrated
# volatility surface.


# %% tags=["interactive"] mystnb={"image": {"alt": "Interactive or static chart comparing a European option payoff at maturity with its Black-Scholes-Merton value across underlying prices. Separate vertical markers identify the current spot and strike, while a compact table reports the price, five Greeks with units, the delta hedge, and put-call parity gap."}}
def plot_black_scholes_dashboard(
    option_type="call",
    S0=100,
    K=105,
    T=1.0,
    r=0.06,
    q=0.02,
    sigma=0.25,
):
    price = black_scholes_price(S0, K, r, sigma, T, option_type, q)
    greeks = black_scholes_greeks(S0, K, r, sigma, T, option_type, q)

    spot_grid = np.linspace(max(1, S0 * 0.40), S0 * 1.80, 200)
    payoff = option_payoff(spot_grid, K, option_type)
    option_values = [
        black_scholes_price(spot, K, r, sigma, T, option_type, q) for spot in spot_grid
    ]

    call_price = black_scholes_price(S0, K, r, sigma, T, "call", q)
    put_price = black_scholes_price(S0, K, r, sigma, T, "put", q)
    parity_gap = put_call_parity_gap(
        call_price,
        put_price,
        S0,
        K,
        r,
        T,
        q,
    )

    title = f"Black-Scholes-Merton {option_type.title()}: {price:.4f} price units"
    builder = (
        build_black_scholes_dashboard
        if RUN_INTERACTIVE_WIDGETS
        else build_black_scholes_dashboard_fallback
    )
    figure = builder(
        spot_grid,
        payoff,
        option_values,
        current_spot=S0,
        strike=K,
        title=title,
    )
    display(figure)
    if not RUN_INTERACTIVE_WIDGETS:
        plt.close(figure)
    display(
        pd.Series(
            {
                "option_type": option_type,
                "spot_price_units": S0,
                "strike_price_units": K,
                "maturity_years": T,
                "continuous_rate_pct": 100 * r,
                "continuous_dividend_yield_pct": 100 * q,
                "annualized_volatility_pct": 100 * sigma,
                "option_price": price,
                "delta": greeks["delta"],
                "underlying_units_for_long_option_delta_hedge": -greeks["delta"],
                "gamma": greeks["gamma"],
                "vega_per_1_percentage_point": greeks["vega_per_1pct"],
                "theta_per_day": greeks["theta_per_day"],
                "rho_per_1_percentage_point": greeks["rho_per_1pct"],
                "put_call_parity_gap": parity_gap,
            },
            name="synthetic pricing and local hedge report",
        )
    )


if RUN_INTERACTIVE_WIDGETS:
    interact(
        plot_black_scholes_dashboard,
        option_type=Dropdown(options=["call", "put"], value="call"),
        S0=FloatSlider(value=100, min=20, max=250, step=1),
        K=FloatSlider(value=105, min=20, max=250, step=1),
        T=FloatSlider(value=1.0, min=0.05, max=5.0, step=0.05, readout_format=".2f"),
        r=FloatSlider(value=0.06, min=0.00, max=0.20, step=0.005, readout_format=".3f"),
        q=FloatSlider(value=0.02, min=0.00, max=0.15, step=0.005, readout_format=".3f"),
        sigma=FloatSlider(value=0.25, min=0.05, max=1.00, step=0.01, readout_format=".2f"),
    )
else:
    plot_black_scholes_dashboard()

# %% [markdown]
# ## Interpretation guide
#
# | Greek | Practical interpretation |
# | --- | --- |
# | Delta | Option-price units per one underlying-price unit |
# | Gamma | Change in delta per one underlying-price unit |
# | Vega | Option-price units per one volatility percentage point |
# | Theta | Option-price units per calendar day, holding other inputs fixed |
# | Rho | Option-price units per one risk-free-rate percentage point |
#
# ## Model limitations
#
# - The dashboard is a Black-Scholes-Merton teaching tool and keeps volatility,
#   rates, dividend yield, and market frictions constant over the option's life.
# - It assumes a European payoff, continuous trading, no transaction costs, and
#   lognormal underlying dynamics; these assumptions define the calculation but
#   do not validate it against market prices.
# - Slider repricing is scenario-based over the selected range; only the Greeks
#   are local sensitivities. Neither replaces analysis with market-calibrated
#   surfaces.
# - Hedging interpretations should account for discrete rebalancing, transaction costs, and liquidity.
#
# The hedge row is the position in the underlying that offsets the delta of one
# **long** option. A short option uses the opposite hedge. This local position is
# the bridge from a model price to scenario P&L; it is not a risk limit or a
# guarantee against nonlinear moves.

# %% [markdown]
# ## Handoff
#
# Preserve the price, parity, Greek-unit, and hedge-sign checks when moving to
# American and exotic payoffs and then to stochastic volatility. Those models
# add exercise, path, monitoring, calibration, and discretization risks that a
# successful Black-Scholes dashboard run cannot validate.
