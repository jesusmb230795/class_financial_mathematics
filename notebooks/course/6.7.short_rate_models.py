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
# {cite}`vasicek1977termStructure,coxIngersollRoss1985termStructure`.
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
# The Vasicek model is {cite}`vasicek1977termStructure`:
#
# $$
# dr_t = \kappa(\theta - r_t)dt + \sigma dW_t.
# $$
#
# The Cox-Ingersoll-Ross (CIR) model is
# {cite}`coxIngersollRoss1985termStructure`:
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
# The same functional form can be written under the physical measure $\mathbb P$
# for scenario analysis or under the risk-neutral measure $\mathbb Q$ for
# valuation. Under $\mathbb Q$, the time-$t$ discount factor satisfies
#
# $$
# D(t,T)
# = \mathbb E_t^{\mathbb Q}\!\left[
# \exp\!\left(-\int_t^T r_u\,du\right)
# \right].
# $$
#
# Historical $\mathbb P$ estimates cannot be inserted into this pricing identity
# without a market-price-of-risk specification or direct cross-sectional
# calibration. The simulations below are illustrative physical-measure
# scenarios, not calibrated prices.
#
# For Vasicek, the exact transition over horizon $h>0$ is Gaussian:
#
# $$
# r_{t+h}\mid r_t
# \sim \mathcal N\!\left(
# \theta+(r_t-\theta)e^{-\kappa h},
# \frac{\sigma^2}{2\kappa}\left(1-e^{-2\kappa h}\right)
# \right).
# $$
#
# The conditional mean moves toward $\theta$, while the conditional variance
# approaches $\sigma^2/(2\kappa)$. Vasicek therefore remains analytically useful
# even though its Gaussian support permits negative rates.
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
# This makes the model useful pedagogically: students can estimate a familiar
# linear model and then translate it into a continuous-time short-rate process.
# The innovation variance must also be mapped through the exact Gaussian
# transition; the calibration lab performs that complete calculation.
#
# For the CIR model, students must check the Feller condition:
#
# $$
# 2\kappa\theta \geq \sigma^2.
# $$
#
# When this condition holds, the zero boundary is unattainable from a positive
# starting value in the continuous-time model. A numerical discretization still
# needs an explicit boundary rule.
#
# ## Python setup

# %% tags=["setup", "hide-input"]
import pandas as pd

from src.module6_visuals import build_short_rate_paths_figure
from src.term_structure import (
    feller_condition,
    simulate_cir_full_truncation,
    simulate_vasicek_exact,
)

# %% [markdown]
# ## Vasicek simulation
#
# The helper samples the exact Gaussian transition, not an Euler approximation.
# Parameters are annualized decimal-rate quantities, time is measured in years,
# the grid has 252 steps per year, and seed 2026 makes the 200 paths reproducible.

# %%
vasicek_params = {
    "r0": 0.08,
    "kappa": 0.60,
    "theta": 0.06,
    "sigma": 0.02,
}
vasicek_paths = simulate_vasicek_exact(
    **vasicek_params,
    years=5,
    steps_per_year=252,
    paths=200,
    seed=2026,
)
vasicek_paths.head()


# %% [markdown]
# ## Cox-Ingersoll-Ross simulation
#
# The CIR parameters are also author-created. They satisfy the Feller condition,
# but that restriction does not establish empirical fit. Seed 2027 distinguishes
# the CIR draws from the Vasicek draws.

# %%
cir_params = {
    "r0": 0.08,
    "kappa": 0.60,
    "theta": 0.06,
    "sigma": 0.08,
}
cir_paths = simulate_cir_full_truncation(
    **cir_params,
    years=5,
    steps_per_year=252,
    paths=200,
    seed=2027,
)
simulation_contract = pd.Series(
    {
        "vasicek_seed": 2026,
        "cir_seed": 2027,
        "steps_per_year": 252,
        "paths_per_model": 200,
        "horizon_years": 5,
        "cir_feller_condition_satisfied": feller_condition(
            cir_params["kappa"],
            cir_params["theta"],
            cir_params["sigma"],
        ),
    }
)
simulation_contract

# %% [markdown]
# ## Full truncation idea
#
# The CIR simulation above uses a full-truncation Euler scheme. It advances an
# auxiliary state $\widetilde r_n$, which may become negative, while applying
# its positive part only inside the drift and diffusion coefficients
# {cite}`lordKoekkoekVanDijk2010fullTruncation`:
#
# $$
# \widetilde r_n^+ = \max(\widetilde r_n, 0),
# $$
#
# $$
# \widetilde r_{n+1}
# = \widetilde r_n
# + \kappa(\theta-\widetilde r_n^+)\Delta t
# + \sigma\sqrt{\widetilde r_n^+}\sqrt{\Delta t}Z_{n+1}.
# $$
#
# The reported non-negative approximation is
#
# $$
# r_n^{\mathrm{FT}}=\widetilde r_n^+.
# $$
#
# The auxiliary update, not the truncated reported value, feeds the next step.
# This distinction separates full truncation from an absorbing-zero Euler rule.
# The scheme is not exact CIR sampling, so results remain sensitive to the
# 252-step grid even though it prevents invalid square roots.
#
# ## Comparing paths

# %% mystnb={"image": {"alt": "Two aligned panels compare median Vasicek and CIR short-rate scenarios with 5th-to-95th percentile bands over five years; dashed horizontal lines mark each model's 6 percent long-run mean."}}
short_rate_source_note = (
    "Source: author-created physical-measure scenarios; annualized decimal-rate "
    "parameters displayed as percentages; 5-year horizon; 252 steps/year; "
    "200 paths/model; Vasicek seed 2026; CIR seed 2027. No market calibration."
)
build_short_rate_paths_figure(
    vasicek_paths,
    cir_paths,
    vasicek_theta=vasicek_params["theta"],
    cir_theta=cir_params["theta"],
    source_note=short_rate_source_note,
)

# %% [markdown]
# Both median paths move from 8% toward the shared 6% long-run mean. The notable
# exception is the distributional shape: CIR volatility shrinks with the rate
# level and the numerical scheme floors the state at zero, whereas Gaussian
# Vasicek paths can cross zero. The bands show simulation dispersion, not
# confidence intervals for estimated parameters, and the comparison is limited
# by hypothetical physical-measure parameters rather than market calibration.

# %%
endpoint_summary = pd.DataFrame(
    {
        "vasicek": vasicek_paths.iloc[-1].quantile([0.05, 0.50, 0.95]),
        "cir": cir_paths.iloc[-1].quantile([0.05, 0.50, 0.95]),
    }
).rename_axis("terminal_quantile")
endpoint_summary

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

# %% [markdown]
# ## Handoff
#
# The curve-fitting lesson compares a parsimonious cross-sectional representation
# with bootstrapped market points, while the calibration lab estimates and tests
# the time-series parameters introduced here. Neither exercise should claim
# instrument valuation until its model reproduces the relevant market curve.
