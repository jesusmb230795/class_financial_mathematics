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
# # Short-Rate Calibration Lab
#
# Module: Fixed Income, Credit, and Term Structure
#
# ## Lesson summary
#
# The Vasicek model can be calibrated through an AR(1) bridge, while the CIR
# model requires extra care because volatility depends on the rate level. This
# lab estimates Vasicek parameters from the real Banxico CETES 28-day rate
# history, simulates exact Vasicek paths, checks the CIR Feller condition, and
# compares it with Full Truncation Euler-Maruyama simulation
# {cite}`banxicoSIE2025,fabozzi2019foundations`.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - estimate an AR(1) representation of the short rate;
# - translate AR(1) parameters into Vasicek parameters;
# - simulate exact Vasicek paths from calibrated parameters;
# - evaluate the CIR Feller condition;
# - simulate CIR paths with a boundary-safe discretization;
# - identify model risk in one-factor short-rate models.
#
# ## Prerequisites
#
# Complete the short-rate-model, curve-fitting, and rate-panel scenario lessons
# first. Readers should understand AR(1) estimation, chronological samples,
# simulation seeds, the Vasicek and CIR assumptions, and the difference between
# a short-rate proxy and a complete discount curve. Rates are decimals and
# \(\Delta t\) is expressed in years.
#
# ## Python setup

# %% tags=["setup", "hide-input"]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.term_structure import (
    feller_condition,
    official_mexican_rate_history,
    simulate_cir_full_truncation,
    simulate_vasicek_exact,
    vasicek_ols_calibration,
)

# %% [markdown]
# ## Short-rate proxy
#
# Use the CETES 28-day rate from the committed Banxico daily snapshot as a real Mexican short-rate proxy for classroom calibration.

# %%
rate_history = official_mexican_rate_history(start="2018-01-01", end="2026-06-05")
short_rate = rate_history["cetes_28d"].rename("cetes_28d")
short_rate.tail()

# %%
short_rate.plot(figsize=(10, 4), title="Official Banxico CETES 28-Day Rate")
plt.xlabel("Date")
plt.ylabel("Rate")
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## Vasicek calibration through AR(1)
#
# The discrete bridge is:
#
# $$
# r_{t+\Delta t} = c + \beta r_t + \epsilon_t.
# $$
#
# The continuous-time parameters are:
#
# $$
# \kappa = -\frac{\ln(\beta)}{\Delta t},
# \qquad
# \theta = \frac{c}{1-\beta},
# \qquad
# \sigma =
# \sqrt{\frac{2\kappa\sigma_\epsilon^2}{1-\beta^2}}.
# $$

# %%
calibration = vasicek_ols_calibration(short_rate, dt=1 / 252)
pd.Series(calibration)

# %% [markdown]
# ## Vasicek path simulation

# %%
vasicek_paths = simulate_vasicek_exact(
    r0=float(short_rate.iloc[-1]),
    kappa=calibration["kappa"],
    theta=calibration["theta"],
    sigma=calibration["sigma"],
    years=3,
    paths=200,
)

vasicek_paths.iloc[:, :20].plot(
    figsize=(10, 4), legend=False, alpha=0.35, title="Calibrated Vasicek Paths"
)
plt.xlabel("Years")
plt.ylabel("Rate")
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## CIR Feller condition
#
# The CIR model is:
#
# $$
# dr_t = \kappa(\theta-r_t)dt + \sigma\sqrt{r_t}dW_t.
# $$
#
# The Feller condition is:
#
# $$
# 2\kappa\theta \geq \sigma^2.
# $$

# %%
cir_params = {
    "r0": float(short_rate.iloc[-1]),
    "kappa": 0.75,
    "theta": 0.075,
    "sigma": 0.10,
}

pd.Series(
    {
        **cir_params,
        "feller_condition_satisfied": feller_condition(
            cir_params["kappa"],
            cir_params["theta"],
            cir_params["sigma"],
        ),
    }
)

# %% [markdown]
# ## CIR Full Truncation Euler-Maruyama
#
# Full truncation applies the positive part of the previous rate inside the square-root term:
#
# $$
# r_t^+ = \max(r_t, 0).
# $$

# %%
cir_paths = simulate_cir_full_truncation(
    r0=cir_params["r0"],
    kappa=cir_params["kappa"],
    theta=cir_params["theta"],
    sigma=cir_params["sigma"],
    years=3,
    paths=200,
)

cir_paths.iloc[:, :20].plot(
    figsize=(10, 4), legend=False, alpha=0.35, title="CIR Full Truncation Paths"
)
plt.xlabel("Years")
plt.ylabel("Rate")
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## Distribution comparison

# %%
summary = pd.DataFrame(
    {
        "vasicek_final": vasicek_paths.iloc[-1],
        "cir_final": cir_paths.iloc[-1],
    }
).describe(percentiles=[0.05, 0.50, 0.95])

summary

# %% [markdown]
# ## Model risk checklist
#
# | Risk | Diagnostic question |
# | --- | --- |
# | Parameter uncertainty | Are estimates stable across samples? |
# | Calibration error | Does the model match the current curve? |
# | Recalibration risk | Do parameters jump sharply when new data arrive? |
# | Misspecification risk | Does the model allow behavior that is economically implausible for the use case? |
#
# ## Model limitations
#
# - AR(1)-based calibration is a classroom bridge and can be biased by discretization and measurement noise.
# - The Feller condition is a parameter restriction, not a complete validation of CIR fit.
# - Model-generated rate paths should be interpreted with model-risk notes before being used for valuation.

# %% [markdown] tags=["exercise"]
# ## Checkpoint exercise
#
# Calibrate the CETES 28-day proxy over 2018–2021 and 2022–2026 using
# $\Delta t=1/252$ years.
#
# 1. Report $c$, $\beta$, $\kappa$, $\theta$, and $\sigma$ for each window with
#    rate units stated.
# 2. Simulate 10,000 one-year Vasicek paths from the same initial rate and compare
#    terminal means and 5th/95th percentiles.
# 3. Evaluate the Feller condition for the displayed CIR parameters and for a
#    volatility of 40%.
# 4. Explain why daily observations, parameter instability, and the chosen
#    252-business-day clock limit the calibration.
#

# %% [markdown] tags=["solution"]
# ```{dropdown} Suggested answer rubric
# A complete answer compares both calibration windows, declares the time-step
# convention, reports simulation quantiles in rate units, and distinguishes a
# Feller check from empirical model validation.
# ```

# %% [markdown]
# ## Handoff
#
# Module 7 converts the price, curve, DV01, spread, and calibrated-model outputs
# from this module into derivative valuation, hedging, limits, and model-risk
# decisions. Pass forward units, valuation timestamp, curve source, parameter
# sample, and calibration diagnostics with every reported exposure.
