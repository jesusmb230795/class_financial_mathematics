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
# This notebook creates an interactive Black-Scholes pricer for European options.
# Students can move the spot price, strike, maturity, risk-free rate, and
# volatility to observe option prices, payoff diagrams, and Greeks
# {cite}`hull2022options`.
#
# The objective is to connect the formula with economic intuition: moneyness, time value, volatility exposure, and local hedging sensitivity.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - compute European call and put prices with Black-Scholes;
# - explain how spot, strike, maturity, rates, and volatility affect option value;
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
# ## Black-Scholes formulas
#
# The dashboard uses the standard European option inputs:
#
# $$
# d_1=\frac{\ln(S_0/K)+(r+\sigma^2/2)T}{\sigma\sqrt{T}},
# \qquad
# d_2=d_1-\sigma\sqrt{T}.
# $$
#
# For a call and a put:
#
# $$
# C=S_0N(d_1)-Ke^{-rT}N(d_2),
# $$
#
# $$
# P=Ke^{-rT}N(-d_2)-S_0N(-d_1).
# $$
#
# The local sensitivities shown in the dashboard are Greeks such as:
#
# $$
# \Delta=\frac{\partial V}{\partial S},\qquad
# \Gamma=\frac{\partial^2 V}{\partial S^2},\qquad
# \nu=\frac{\partial V}{\partial \sigma}.
# $$
#
# ## Python setup

# %% tags=["setup", "hide-input"]
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import Dropdown, FloatSlider, interact
from scipy.stats import norm

from src.dashboard_fallbacks import build_black_scholes_dashboard_fallback
from src.dashboards import build_black_scholes_dashboard

RUN_INTERACTIVE_WIDGETS = os.getenv("RUN_INTERACTIVE_WIDGETS", "1") == "1"


# %% [markdown]
# ## Pricing and Greeks


# %%
def black_scholes_inputs(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes_price(S0, K, r, sigma, T, option_type):
    d1, d2 = black_scholes_inputs(S0, K, r, sigma, T)
    if option_type == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


def black_scholes_greeks(S0, K, r, sigma, T, option_type):
    d1, d2 = black_scholes_inputs(S0, K, r, sigma, T)
    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    vega = S0 * norm.pdf(d1) * np.sqrt(T)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = -S0 * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(
            d2
        )
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = -S0 * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(
            -d2
        )
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    return pd.Series(
        {
            "delta": delta,
            "gamma": gamma,
            "vega": vega / 100,
            "theta_per_day": theta / 365,
            "rho": rho / 100,
        }
    )


# %% [markdown]
# ## Interactive dashboard
#
# Run this cell in JupyterLab with `uv run jupyter lab`.


# %% tags=["interactive"]
def plot_black_scholes_dashboard(
    option_type="call",
    S0=100,
    K=100,
    T=1.0,
    r=0.06,
    sigma=0.25,
):
    price = black_scholes_price(S0, K, r, sigma, T, option_type)
    greeks = black_scholes_greeks(S0, K, r, sigma, T, option_type)

    spot_grid = np.linspace(max(1, S0 * 0.40), S0 * 1.80, 200)
    payoff = np.maximum(spot_grid - K, 0) if option_type == "call" else np.maximum(K - spot_grid, 0)
    option_values = [black_scholes_price(spot, K, r, sigma, T, option_type) for spot in spot_grid]

    parity_gap = (
        black_scholes_price(S0, K, r, sigma, T, "call")
        + K * np.exp(-r * T)
        - black_scholes_price(S0, K, r, sigma, T, "put")
        - S0
    )

    title = (
        f"{option_type.title()} price: {price:.4f} | "
        f"Delta: {greeks['delta']:.4f} | Gamma: {greeks['gamma']:.4f} | "
        f"Vega/1pct: {greeks['vega']:.4f} | Theta/day: {greeks['theta_per_day']:.4f} | "
        f"Parity gap: {parity_gap:.2e}"
    )
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
                "option_price": price,
                "delta": greeks["delta"],
                "underlying_units_for_long_option_delta_hedge": -greeks["delta"],
                "gamma": greeks["gamma"],
                "vega_per_1pct": greeks["vega"],
                "theta_per_day": greeks["theta_per_day"],
                "put_call_parity_gap": parity_gap,
            },
            name="pricing and local hedge report",
        )
    )


if RUN_INTERACTIVE_WIDGETS:
    interact(
        plot_black_scholes_dashboard,
        option_type=Dropdown(options=["call", "put"], value="call"),
        S0=FloatSlider(value=100, min=20, max=250, step=1),
        K=FloatSlider(value=100, min=20, max=250, step=1),
        T=FloatSlider(value=1.0, min=0.05, max=5.0, step=0.05, readout_format=".2f"),
        r=FloatSlider(value=0.06, min=0.00, max=0.20, step=0.005, readout_format=".3f"),
        sigma=FloatSlider(value=0.25, min=0.05, max=1.00, step=0.01, readout_format=".2f"),
    )
else:
    plot_black_scholes_dashboard()

# %% [markdown]
# ## Interpretation guide
#
# | Greek | Practical interpretation |
# | --- | --- |
# | Delta | Approximate option price change for a one-unit change in the underlying price |
# | Gamma | Curvature of the option value with respect to the underlying price |
# | Vega | Approximate price change for a one percentage point change in volatility |
# | Theta | Approximate daily time decay |
# | Rho | Approximate price change for a one percentage point change in the risk-free rate |
#
# ## Model limitations
#
# - The dashboard is a Black-Scholes teaching tool and keeps volatility, rates, and market frictions simplified.
# - Slider sensitivity is local; it does not replace scenario analysis with market-calibrated surfaces.
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
