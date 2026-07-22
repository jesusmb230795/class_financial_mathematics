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
# # Short-Rate Models
#
# Module: Fixed Income, Credit, and Term Structure
#
# ## Lesson summary
#
# Short-rate models describe the evolution of the instantaneous interest rate.
# They are simplified models, but they introduce the core ideas behind
# interest-rate simulation, mean reversion, and stochastic discounting
# {cite}`fabozzi2019foundations`.
#
# This lesson focuses on the Vasicek and Cox-Ingersoll-Ross models.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - explain the role of the short rate in fixed-income modeling;
# - simulate Vasicek short-rate paths;
# - simulate Cox-Ingersoll-Ross short-rate paths;
# - compare mean reversion, volatility, and non-negativity assumptions;
# - discuss why model calibration matters.
#
# ## Prerequisites
#
# Complete the quantitative time-series foundation and yield-curve bootstrap
# first. Readers should understand discount factors, spot and forward rates,
# random shocks, autoregressive models, and reproducible simulation with an
# explicit seed. Rates and model parameters are stored as decimals and time is
# measured in years.
#
# ## Core models
#
# The Vasicek model is:
#
# $$
# dr_t = \kappa(\theta - r_t)dt + \sigma dW_t.
# $$
#
# The Cox-Ingersoll-Ross model is:
#
# $$
# dr_t = \kappa(\theta - r_t)dt + \sigma\sqrt{r_t}dW_t.
# $$
#
# The parameters are:
#
# - $\kappa$: speed of mean reversion;
# - $\theta$: long-run mean;
# - $\sigma$: volatility;
# - $W_t$: Brownian motion.
#
# ## Calibration bridge
#
# The Vasicek model can be linked to a discrete AR(1) regression:
#
# $$
# r_{t+\Delta t} = c + \beta r_t + \epsilon_t.
# $$
#
# The continuous-time parameters are recovered from the regression parameters:
#
# $$
# \kappa = -\frac{\ln(\beta)}{\Delta t},
# $$
#
# $$
# \theta = \frac{c}{1-\beta}.
# $$
#
# This makes the model useful pedagogically: students can estimate a familiar linear model and then translate it into a continuous-time short-rate process.
#
# For the CIR model, students must check the Feller condition:
#
# $$
# 2\kappa\theta \geq \sigma^2.
# $$
#
# When this condition holds, the zero boundary is less problematic in continuous time. In discrete simulation, negative values may still appear if the numerical method is not handled carefully.
#
# ## Python setup

# %% tags=["setup", "hide-input"]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %% [markdown]
# ## Vasicek simulation


# %%
def simulate_vasicek(
    r0=0.08, kappa=0.60, theta=0.06, sigma=0.02, years=5, steps_per_year=252, paths=20, seed=42
):
    rng = np.random.default_rng(seed)
    dt = 1 / steps_per_year
    steps = int(years * steps_per_year)
    rates = np.empty((steps + 1, paths))
    rates[0] = r0

    for step in range(1, steps + 1):
        shock = rng.normal(0, np.sqrt(dt), size=paths)
        rates[step] = rates[step - 1] + kappa * (theta - rates[step - 1]) * dt + sigma * shock

    index = np.linspace(0, years, steps + 1)
    return pd.DataFrame(rates, index=index)


vasicek_paths = simulate_vasicek()
vasicek_paths.head()


# %% [markdown]
# ## Cox-Ingersoll-Ross simulation


# %%
def simulate_cir(
    r0=0.08, kappa=0.60, theta=0.06, sigma=0.08, years=5, steps_per_year=252, paths=20, seed=42
):
    rng = np.random.default_rng(seed)
    dt = 1 / steps_per_year
    steps = int(years * steps_per_year)
    rates = np.empty((steps + 1, paths))
    rates[0] = r0

    for step in range(1, steps + 1):
        previous = np.maximum(rates[step - 1], 0)
        shock = rng.normal(0, np.sqrt(dt), size=paths)
        rates[step] = previous + kappa * (theta - previous) * dt + sigma * np.sqrt(previous) * shock
        rates[step] = np.maximum(rates[step], 0)

    index = np.linspace(0, years, steps + 1)
    return pd.DataFrame(rates, index=index)


cir_paths = simulate_cir()
cir_paths.head()

# %% [markdown]
# ## Full truncation idea
#
# The CIR simulation above uses a simple boundary-safe idea: replace the previous rate by its positive part before evaluating the square-root diffusion term.
#
# $$
# r_t^+ = \max(r_t, 0).
# $$
#
# The discrete approximation is:
#
# $$
# r_{t+\Delta t}
# = r_t
# + \kappa(\theta-r_t^+)\Delta t
# + \sigma\sqrt{r_t^+}\sqrt{\Delta t}Z_{t+\Delta t}.
# $$
#
# This is known as a Full Truncation Euler-Maruyama style scheme. It is not a perfect substitute for exact CIR sampling, but it is a practical classroom method because it prevents invalid square roots and preserves the mean-reverting interpretation near zero.
#
# ## Comparing paths

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

vasicek_paths.iloc[:, :10].plot(ax=axes[0], legend=False, alpha=0.8)
axes[0].set_title("Vasicek Short-Rate Paths")
axes[0].set_xlabel("Years")
axes[0].set_ylabel("Rate")
axes[0].grid(True, alpha=0.3)

cir_paths.iloc[:, :10].plot(ax=axes[1], legend=False, alpha=0.8)
axes[1].set_title("CIR Short-Rate Paths")
axes[1].set_xlabel("Years")
axes[1].grid(True, alpha=0.3)

plt.show()

# %% [markdown]
# ## Model interpretation
#
# | Feature | Vasicek | CIR |
# | --- | --- | --- |
# | Mean reversion | Yes | Yes |
# | Constant volatility | Yes | No |
# | Rate-dependent volatility | No | Yes |
# | Negative rates possible | Yes | Usually avoided by construction |
# | Typical use | Tractable baseline | Non-negative rate modeling |
#
# ## Model limitations
#
# - One-factor short-rate models cannot represent the full shape of the yield curve.
# - Vasicek permits negative rates, while CIR imposes nonnegative rates at the cost of stricter dynamics.
# - Model-generated paths depend heavily on calibration window, discretization, and parameter stability.

# %% [markdown] tags=["exercise"]
# ## Checkpoint exercise
#
# Simulate 5,000 five-year paths after doubling $\kappa$ from 0.40 to 0.80 while
# keeping $r_0$, $\theta$, and $\sigma$ fixed.
#
# 1. Compare the one- and five-year mean, standard deviation, and 5th/95th
#    percentiles for Vasicek.
# 2. Repeat for CIR and report whether the Feller condition holds in each case.
# 3. Count negative Vasicek observations and verify that the full-truncation CIR
#    output remains non-negative.
# 4. Explain how faster mean reversion changes dispersion without claiming that
#    either one-factor model fits the current curve.
#

# %% [markdown] tags=["solution"]
# ```{dropdown} Suggested answer rubric
# A complete answer reports rate distributions in percentage points, verifies the
# boundary and Feller diagnostics, and separates simulation behavior from
# calibration or market-fit evidence.
# ```

# %% [markdown]
# ## Handoff
#
# The curve-fitting lesson compares a parsimonious cross-sectional representation
# with bootstrapped market points, while the calibration lab estimates and tests
# the time-series parameters introduced here. Neither exercise should claim
# instrument valuation until its model reproduces the relevant market curve.
