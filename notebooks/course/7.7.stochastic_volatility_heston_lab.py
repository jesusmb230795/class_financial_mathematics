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
# correlated with the underlying return shock
# {cite}`heston1993,hull2022options`.
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
# - relate negative spot-variance correlation to a mechanism that can contribute
#   to downside skew, without treating a classroom simulation as calibration
#   evidence.
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
from IPython.display import display

from src.derivatives import black_scholes_price, simulate_gbm_paths, simulate_heston_paths
from src.module7_visuals import build_heston_diagnostics_figure

# %% [markdown]
# ## Model dynamics
#
# Under the risk-neutral classroom parameterization used here, the Heston model
# is
#
# $$
# \frac{dS_t}{S_t}=(r-q)dt+\sqrt{v_t}\,dW_t^{(S)},
# $$
#
# $$
# dv_t=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}\,dW_t^{(v)},
# $$
#
# with correlated Brownian shocks:
#
# $$
# dW_t^{(S)}dW_t^{(v)}=\rho_{S,v}\,dt.
# $$
#
# Here $\kappa$ is the mean-reversion speed, $\theta$ is long-run variance,
# $\xi$ is volatility of variance, and $\rho_{S,v}$ is spot-variance shock
# correlation. The sufficient Feller condition for the continuous variance
# process to stay strictly positive is
#
# $$
# 2\kappa\theta\ge \xi^2.
# $$
#
# Violating this condition does not make the model undefined, but it makes the
# variance boundary and discretization choice more important.
#
# The simulator uses full-truncation Euler. It preserves a latent variance
# $\widetilde v_n$, while applying
# $\widetilde v_n^+=\max(\widetilde v_n,0)$ inside drift and diffusion
# {cite}`lordKoekkoekVanDijk2010fullTruncation`:
#
# $$
# \widetilde v_{n+1}=\widetilde v_n
# +\kappa(\theta-\widetilde v_n^+)\Delta t
# +\xi\sqrt{\widetilde v_n^+}\sqrt{\Delta t}\,Z_{n+1}^{(v)},
# $$
#
# $$
# \log S_{n+1}=\log S_n
# +\left(r-q-\tfrac12\widetilde v_n^+\right)\Delta t
# +\sqrt{\widetilde v_n^+}\sqrt{\Delta t}\,Z_{n+1}^{(S)},
# \qquad
# \operatorname{Corr}(Z^{(S)},Z^{(v)})=\rho_{S,v}.
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
rho_spot_variance = -0.65

feller_diagnostic = pd.Series(
    {
        "two_kappa_theta": 2 * kappa * theta,
        "vol_of_vol_squared": vol_of_vol**2,
        "feller_gap": 2 * kappa * theta - vol_of_vol**2,
        "feller_condition_satisfied": 2 * kappa * theta >= vol_of_vol**2,
    },
    name="synthetic Heston boundary diagnostic",
)
assert not bool(feller_diagnostic["feller_condition_satisfied"])
feller_diagnostic

# %%
heston_seed = 2028

heston_spots, heston_variances = simulate_heston_paths(
    spot=spot,
    variance0=variance0,
    rate=rate,
    maturity=maturity,
    kappa=kappa,
    theta=theta,
    vol_of_vol=vol_of_vol,
    rho=rho_spot_variance,
    steps=252,
    paths=5_000,
    dividend_yield=dividend_yield,
    seed=heston_seed,
)

gbm_paths = simulate_gbm_paths(
    spot=spot,
    rate=rate,
    volatility=np.sqrt(theta),
    maturity=maturity,
    steps=252,
    paths=5_000,
    dividend_yield=dividend_yield,
    seed=2029,
)

# %% mystnb={"image": {"alt": "Three panels summarize seeded one-year simulations: the Heston 5th-to-95th percentile spot band and median against the constant-volatility geometric Brownian motion median; the Heston instantaneous-volatility percentile band and median; and both terminal spot densities in generic price units."}}
heston_diagnostics = build_heston_diagnostics_figure(
    heston_spots,
    heston_variances,
    gbm_paths,
    maturity=maturity,
)
display(heston_diagnostics)
plt.close(heston_diagnostics)

# %% [markdown]
# ## Compare with constant volatility
#
# The two simulations use explicit fixed seeds and identical spot, carry,
# maturity, number of steps, and number of paths. They differ in their variance
# dynamics. This controls reproducibility, but independent seeds still leave
# Monte Carlo sampling variation and do not constitute an empirical model test.

# %%
terminal = pd.DataFrame(
    {
        "heston": heston_spots[-1],
        "gbm_constant_vol": gbm_paths[-1],
    }
)
terminal.describe(percentiles=[0.01, 0.05, 0.50, 0.95, 0.99])

# %% [markdown]
# ## Pricing implication
#
# Use the simulated Heston terminal prices to estimate European call prices
# across strikes. Compare them with Black-Scholes-Merton using the long-run
# volatility $\sqrt{\theta}$. The comparison isolates a model choice; it does
# not estimate market implied volatility because no observed option quotes are
# supplied and the Heston parameters are not calibrated.

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
pricing["heston_mc_ci_95_lower"] = (
    pricing["heston_mc_call"] - 1.96 * pricing["heston_mc_standard_error"]
)
pricing["heston_mc_ci_95_upper"] = (
    pricing["heston_mc_call"] + 1.96 * pricing["heston_mc_standard_error"]
)
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
# | Variance boundary | This parameter set violates the sufficient Feller condition, so boundary treatment and time-step convergence require explicit checks |
# | Smile dynamics | Negative $\rho_{S,v}$ can create asymmetric option prices, but only market calibration can establish whether the model explains an observed skew |
#
# ## Model limitations
#
# - Heston simulations depend on discretization, truncation scheme, and parameter stability.
# - The seeded figures and tables are synthetic, use generic price units, and
#   are not observed market evidence.
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
