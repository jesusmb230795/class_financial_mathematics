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
# # Stochastic Volatility and Heston Lab
#
# Module: Derivatives and Risk Management
#
# ## Lesson summary
#
# Black-Scholes assumes constant volatility. Real option markets show smiles,
# skews, and term structures of implied volatility. The Heston model is a
# standard stochastic-volatility extension in which variance mean-reverts and is
# correlated with the underlying return shock {cite}`hull2022options`.
#
# This lab uses simulation rather than Fourier pricing. The objective is to understand dynamics, skew intuition, and model risk before implementing production-level characteristic-function methods.
#
# For pricing, the displayed drift and variance parameters are treated as
# risk-neutral classroom inputs. They are not estimated physical dynamics or a
# calibration to traded option prices.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - state the Heston spot and variance dynamics;
# - explain the roles of mean reversion, long-run variance, vol-of-vol, and correlation;
# - simulate full-truncation Heston paths;
# - compare stochastic-volatility terminal prices against a constant-volatility baseline;
# - connect negative spot-volatility correlation with downside skew.
#
# ## Prerequisites
#
# Complete the implied-volatility, numerical-pricing, and American/exotic option
# lessons first. Students should understand volatility smiles, risk-neutral
# simulation, Monte Carlo standard error, correlation, and the distinction
# between parameter choice, calibration, and model validation.
#
# ## Python setup

# %% tags=["setup", "hide-input"]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.derivatives import black_scholes_price, simulate_gbm_paths, simulate_heston_paths

# %% [markdown]
# ## Model dynamics
#
# The Heston model is:
#
# $$
# dS_t = (r-q)S_tdt + \sqrt{v_t}S_tdW_t^{(1)},
# $$
#
# $$
# dv_t = \kappa(\theta-v_t)dt + \xi\sqrt{v_t}dW_t^{(2)},
# $$
#
# with correlated Brownian shocks:
#
# $$
# dW_t^{(1)}dW_t^{(2)} = \rho dt.
# $$
#
# ## Simulate paths

# %%
spot = 100.0
rate = 0.05
dividend_yield = 0.0
maturity = 1.0
variance0 = 0.25**2
kappa = 2.0
theta = 0.22**2
vol_of_vol = 0.55
rho = -0.65

heston_spots, heston_variances = simulate_heston_paths(
    spot=spot,
    variance0=variance0,
    rate=rate,
    maturity=maturity,
    kappa=kappa,
    theta=theta,
    vol_of_vol=vol_of_vol,
    rho=rho,
    steps=252,
    paths=5_000,
    dividend_yield=dividend_yield,
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(heston_spots[:80, :20], alpha=0.45)
axes[0].set_title("Heston Spot Paths")
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Spot")
axes[0].grid(True, alpha=0.3)

axes[1].plot(np.sqrt(heston_variances[:80, :20]), alpha=0.45)
axes[1].set_title("Instantaneous Volatility Paths")
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Volatility")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Compare with constant volatility

# %%
gbm_paths = simulate_gbm_paths(
    spot=spot,
    rate=rate,
    volatility=np.sqrt(theta),
    maturity=maturity,
    steps=252,
    paths=5_000,
    dividend_yield=dividend_yield,
    seed=2028,
)

terminal = pd.DataFrame(
    {
        "heston": heston_spots[-1],
        "gbm_constant_vol": gbm_paths[-1],
    }
)
terminal.describe(percentiles=[0.01, 0.05, 0.50, 0.95, 0.99])

# %%
terminal.plot(kind="hist", bins=60, alpha=0.45, density=True, figsize=(9, 4))
plt.title("Terminal Price Distributions")
plt.xlabel("Terminal spot")
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## Pricing implication
#
# Use the simulated Heston terminal prices to estimate European call prices across strikes. Compare them with Black-Scholes using the long-run volatility $\sqrt{\theta}$.

# %%
strikes = np.array([80, 90, 100, 110, 120])
discount = np.exp(-rate * maturity)

pricing = pd.DataFrame({"strike": strikes})
pricing["heston_mc_call"] = [
    discount * np.maximum(heston_spots[-1] - strike, 0).mean() for strike in strikes
]
pricing["heston_mc_standard_error"] = [
    discount * np.maximum(heston_spots[-1] - strike, 0).std(ddof=1) / np.sqrt(heston_spots.shape[1])
    for strike in strikes
]
pricing["black_scholes_long_run_vol"] = [
    black_scholes_price(
        spot,
        strike,
        rate,
        np.sqrt(theta),
        maturity,
        option_type="call",
        dividend_yield=dividend_yield,
    )
    for strike in strikes
]
pricing["difference"] = pricing["heston_mc_call"] - pricing["black_scholes_long_run_vol"]
pricing

# %% [markdown]
# ## Model-risk notes
#
# | Issue | Practical implication |
# | --- | --- |
# | Calibration instability | Heston parameters can move sharply across calibration dates |
# | Numerical method | Fourier pricing requires stable characteristic-function implementation |
# | Variance boundary | Discretized variance can become negative without truncation or exact sampling |
# | Smile dynamics | Heston captures skew dynamics better than Black-Scholes, but still misses jumps and local-volatility effects |
#
# ## Model limitations
#
# - Heston simulations depend on discretization, truncation scheme, and parameter stability.
# - The lab uses simulation for intuition and does not implement production characteristic-function pricing.
# - Heston captures stochastic volatility but still misses jumps, local-volatility effects, and liquidity constraints.
#
# ## Handoff
#
# The sequence now has an explicit boundary:
#
# 1. calibrate or choose a pricing model and document numerical uncertainty;
# 2. compute Greeks and scenario P&L for the hedged position;
# 3. aggregate residual P&L across instruments and risk factors;
# 4. estimate positive-loss VaR/Expected Shortfall, stress losses, and exceptions;
# 5. compare them with limits and escalate model or hedge breaches through
#    governance.
#
# VaR is not an option price, and a calibrated Heston price is not evidence that a
# hedge or risk limit is adequate.
