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
# # Binomial Trees and Monte Carlo Option Pricing
#
# Module: Derivatives and Risk Management
#
# ## Lesson summary
#
# Analytical formulas are valuable benchmarks, but many derivatives require
# numerical methods. A Cox-Ross-Rubinstein (CRR) tree approximates risk-neutral
# price dynamics on a discrete lattice, while Monte Carlo simulation estimates a
# discounted expected payoff from pseudo-random scenarios
# {cite}`coxRossRubinstein1979,boyle1977,hull2022options`.
#
# This lesson separates two numerical uncertainties: lattice discretization
# error and Monte Carlo sampling error. Agreement with Black-Scholes-Merton is a
# controlled implementation check under common assumptions, not evidence that
# those assumptions fit a real market.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - construct CRR up and down factors and the risk-neutral probability $p^*$;
# - price a European option by backward induction;
# - estimate a European option price from risk-neutral terminal simulations;
# - report Monte Carlo standard error and a 95% confidence interval;
# - compare lattice convergence with an analytical benchmark;
# - distinguish numerical error from market-risk profit and loss.
#
# ## Prerequisites
#
# Complete European Option Pricing and Greeks first. Students should be able to
# interpret Black-Scholes-Merton as a model benchmark, calculate a discounted
# risk-neutral expectation, and distinguish a model price from an observed quote.
#
# ## Cox-Ross-Rubinstein lattice
#
# Divide maturity $T$ into $n$ equal intervals $\Delta t=T/n$. The CRR factors
# are {cite}`coxRossRubinstein1979`:
#
# $$
# u=e^{\sigma\sqrt{\Delta t}},
# \qquad
# d=u^{-1}=e^{-\sigma\sqrt{\Delta t}}.
# $$
#
# With annual continuously compounded financing rate $r$ and dividend yield
# $q$, the risk-neutral up probability is
#
# $$
# p^*=\frac{e^{(r-q)\Delta t}-d}{u-d},
# \qquad 0\leq p^*\leq1.
# $$
#
# The star distinguishes this probability from the dividend yield $q$ used in
# the drift. For continuation values $V_u$ and $V_d$ one step ahead,
#
# $$
# V_t=e^{-r\Delta t}\left[p^*V_u+(1-p^*)V_d\right].
# $$
#
# Applying this recursion backward from the terminal payoff gives the time-zero
# European option value.
#
# ## Monte Carlo estimator and uncertainty
#
# Under risk-neutral geometric Brownian motion,
#
# $$
# S_T^{(i)}=S_0\exp\!\left[
# \left(r-q-\frac12\sigma^2\right)T+\sigma\sqrt{T}Z_i
# \right],
# \qquad Z_i\sim\mathcal N(0,1).
# $$
#
# For $N$ simulated terminal prices and payoff $\Pi$, define the discounted
# scenario payoff $Y_i=e^{-rT}\Pi(S_T^{(i)})$. The estimator, estimated standard
# error, and normal-approximation confidence interval are
# {cite}`boyle1977,hull2022options`:
#
# $$
# \widehat V_N=\frac1N\sum_{i=1}^{N}Y_i,
# \qquad
# \widehat{SE}(\widehat V_N)=\frac{s_Y}{\sqrt N},
# $$
#
# $$
# CI_{95\%}=\widehat V_N\pm1.96\,\widehat{SE}(\widehat V_N).
# $$
#
# The interval quantifies pseudo-random sampling uncertainty conditional on the
# model and inputs. It does not include parameter, market, liquidity, or model
# risk.
#
# ## Python setup

# %% tags=["setup", "hide-input"]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from src.derivatives import (
    MonteCarloPriceResult,
    black_scholes_price,
    crr_binomial_option_price,
    monte_carlo_european_option_price,
)
from src.module7_visuals import build_convergence_figure

# %% [markdown]
# ## Deterministic model inputs
#
# All inputs are synthetic. Prices use currency units per share; $r$, $q$, and
# $\sigma$ are annual decimals; rates use continuous compounding; and maturity is
# measured in years.

# %%
spot = 100.0
strike = 105.0
rate = 0.06
dividend_yield = 0.015
volatility = 0.25
maturity = 1.0

pd.Series(
    {
        "spot_currency_per_share": spot,
        "strike_currency_per_share": strike,
        "rate_continuous_pct": 100 * rate,
        "dividend_yield_continuous_pct": 100 * dividend_yield,
        "volatility_annual_pct": 100 * volatility,
        "maturity_years": maturity,
    },
    name="synthetic_pricing_inputs",
)

# %% [markdown]
# ## One-step probability diagnostic
#
# Before running a full tree, calculate the one-step quantities directly. A
# value of $p^*$ outside $[0,1]$ would violate this CRR parameterization's
# no-arbitrage condition and the shared pricer would reject the inputs.

# %%
one_step_dt = maturity
one_step_up = np.exp(volatility * np.sqrt(one_step_dt))
one_step_down = 1.0 / one_step_up
one_step_probability = (np.exp((rate - dividend_yield) * one_step_dt) - one_step_down) / (
    one_step_up - one_step_down
)

pd.Series(
    {
        "up_factor": one_step_up,
        "down_factor": one_step_down,
        "risk_neutral_up_probability_p_star": one_step_probability,
        "risk_neutral_down_probability": 1.0 - one_step_probability,
    }
)

# %%
assert 0.0 <= one_step_probability <= 1.0

# %% [markdown]
# ## CRR price by backward induction

# %%
tree_steps = 101
crr_price = crr_binomial_option_price(
    spot,
    strike,
    rate,
    volatility,
    maturity,
    steps=tree_steps,
    option_type="call",
    dividend_yield=dividend_yield,
)
pd.Series(
    {
        "crr_call_price_currency_per_share": crr_price,
        "tree_steps": tree_steps,
    },
    name="crr_european_call",
)

# %% [markdown]
# ## Monte Carlo price with a declared seed
#
# The fixed seed `42` makes this teaching run reproducible. Changing the seed
# changes the estimate within sampling error; it does not create new market
# information. The shared function returns a `MonteCarloPriceResult`, keeping the
# estimate and its uncertainty together.

# %%
simulation_count = 100_000
simulation_seed = 42
mc_result = monte_carlo_european_option_price(
    spot,
    strike,
    rate,
    volatility,
    maturity,
    simulations=simulation_count,
    option_type="call",
    dividend_yield=dividend_yield,
    seed=simulation_seed,
)
assert isinstance(mc_result, MonteCarloPriceResult)

pd.Series(
    {
        "monte_carlo_price_currency_per_share": mc_result.price,
        "standard_error_currency_per_share": mc_result.standard_error,
        "ci_95_lower_currency_per_share": mc_result.ci_95_lower,
        "ci_95_upper_currency_per_share": mc_result.ci_95_upper,
        "simulations": simulation_count,
        "seed": simulation_seed,
    },
    name="seeded_monte_carlo_european_call",
)

# %% [markdown]
# **Interpretation.** The confidence interval is conditional on risk-neutral
# geometric Brownian motion with fixed parameters. Repeating the experiment with
# independent seeds checks Monte Carlo stability; it does not validate the
# economic model.
#
# ## Convergence against Black-Scholes-Merton
#
# The analytical price under the same $S_0$, $K$, $r$, $q$, $\sigma$, and $T$
# provides a controlled benchmark. CRR error can oscillate with step count near
# a kinked payoff, so a single convenient grid size is not a convergence proof.

# %%
benchmark_price = black_scholes_price(
    spot,
    strike,
    rate,
    volatility,
    maturity,
    option_type="call",
    dividend_yield=dividend_yield,
)
steps_grid = np.array([5, 10, 25, 50, 100, 250, 500])
comparison = pd.DataFrame({"steps": steps_grid})
comparison["binomial_price"] = [
    crr_binomial_option_price(
        spot,
        strike,
        rate,
        volatility,
        maturity,
        steps=int(steps),
        option_type="call",
        dividend_yield=dividend_yield,
    )
    for steps in steps_grid
]
comparison["black_scholes_price"] = benchmark_price
comparison["pricing_error"] = comparison["binomial_price"] - comparison["black_scholes_price"]
comparison.round(6)

# %% mystnb={"image": {"alt": "A two-panel deterministic convergence figure compares Cox-Ross-Rubinstein European call prices with the Black-Scholes-Merton benchmark and shows signed pricing error as the number of tree steps increases from 5 to 500."}}
convergence_figure = build_convergence_figure(comparison)
display(convergence_figure)
plt.close(convergence_figure)

# %% [markdown]
# **Interpretation.** The lattice prices approach the analytical benchmark even
# though the signed error is not monotone. This supports the implementation only
# within the common European-option, constant-parameter model boundary.
#
# ## Cross-method diagnostic

# %%
cross_method = pd.Series(
    {
        "black_scholes_price": benchmark_price,
        "crr_500_step_price": comparison.loc[comparison["steps"] == 500, "binomial_price"].iloc[0],
        "monte_carlo_price": mc_result.price,
        "monte_carlo_ci_95_lower": mc_result.ci_95_lower,
        "monte_carlo_ci_95_upper": mc_result.ci_95_upper,
        "benchmark_inside_mc_ci": mc_result.ci_95_lower <= benchmark_price <= mc_result.ci_95_upper,
    },
    name="conditional_numerical_consistency",
)
cross_method

# %% [markdown]
# ## Source and model notes
#
# - The recombining lattice follows Cox, Ross, and Rubinstein
#   {cite}`coxRossRubinstein1979`.
# - Simulation-based option valuation follows the Monte Carlo approach developed
#   for options by Boyle {cite}`boyle1977`.
# - The analytical benchmark and risk-neutral interpretation follow
#   Black-Scholes-Merton and the standard derivatives treatment
#   {cite}`blackScholes1973,merton1973,hull2022options`.
# - All model inputs are deterministic synthetic teaching assumptions; no option
#   quote or underlying-price observation is loaded.
# - The displayed Monte Carlo estimate uses 100,000 pseudo-random draws and seed
#   42. The seed is part of the reproducibility contract, not a calibrated input.
#
# ## Model limitations
#
# - CRR prices depend on tree construction and step count; convergence can
#   oscillate near payoff kinks.
# - Monte Carlo converges at approximately $N^{-1/2}$ under standard conditions
#   and can be inefficient for rare-event or path-dependent payoffs.
# - The normal-approximation interval can be imperfect for small samples or very
#   skewed payoffs.
# - Numerical agreement cannot validate constant volatility, continuous trading,
#   lognormal diffusion, or any real-market quote.
#
# ## Handoff
#
# Numerical error belongs in valuation controls before risk aggregation. The
# next lesson reverses the pricing map: it starts from a clearly labeled
# synthetic option chain, checks no-arbitrage conditions, and solves the
# Black-Scholes-Merton volatility that reprices each quote.
