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
# Analytical formulas are useful, but many derivative contracts require numerical
# methods. Binomial trees approximate risk-neutral price dynamics on a discrete
# lattice, while Monte Carlo simulation estimates expected discounted payoffs
# from simulated paths {cite}`hull2022options`.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - construct a one-step and multi-step binomial tree;
# - calculate risk-neutral probabilities;
# - price European options by backward induction;
# - simulate terminal stock prices under geometric Brownian motion;
# - estimate option prices with Monte Carlo and evaluate simulation error.
#
# ## Prerequisites
#
# Complete European Option Pricing and Greeks first. Students should be able to
# compute a discounted risk-neutral expectation, interpret Black-Scholes as a
# benchmark rather than observed truth, and distinguish a pricing error from
# portfolio profit and loss.
#
# ## Binomial model
#
# In a one-period binomial model:
#
# $$
# S_u = S_0u,\qquad S_d = S_0d.
# $$
#
# The risk-neutral probability is:
#
# $$
# q = \frac{e^{r\Delta t} - d}{u-d}.
# $$
#
# The option value is:
#
# $$
# V_0 = e^{-r\Delta t}\left(qV_u + (1-q)V_d\right).
# $$
#
# ## Python setup

# %% tags=["setup", "hide-input"]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


# %% [markdown]
# ## Binomial tree pricer


# %%
def binomial_option_price(S0, K, r, sigma, T, steps=100, option_type="call"):
    if S0 <= 0 or K <= 0 or sigma <= 0 or T <= 0 or steps < 1:
        raise ValueError("S0, K, sigma, T, and steps must be positive")
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)
    if not 0 <= q <= 1:
        raise ValueError("risk-neutral probability outside [0, 1]; check inputs")
    discount = np.exp(-r * dt)

    terminal_prices = S0 * u ** np.arange(steps, -1, -1) * d ** np.arange(0, steps + 1)

    if option_type == "call":
        values = np.maximum(terminal_prices - K, 0)
    elif option_type == "put":
        values = np.maximum(K - terminal_prices, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    for _ in range(steps):
        values = discount * (q * values[:-1] + (1 - q) * values[1:])

    return values[0]


binomial_option_price(100, 105, 0.06, 0.25, 1, steps=100, option_type="call")


# %% [markdown]
# ## Monte Carlo pricer


# %%
def monte_carlo_option_price(S0, K, r, sigma, T, simulations=100_000, option_type="call", seed=42):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=simulations)
    terminal_prices = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)

    if option_type == "call":
        payoffs = np.maximum(terminal_prices - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(K - terminal_prices, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted_payoffs = np.exp(-r * T) * payoffs
    return discounted_payoffs.mean(), discounted_payoffs.std(ddof=1) / np.sqrt(simulations)


mc_price, mc_standard_error = monte_carlo_option_price(100, 105, 0.06, 0.25, 1)
pd.Series(
    {
        "monte_carlo_price": mc_price,
        "standard_error": mc_standard_error,
        "ci_95_lower": mc_price - 1.96 * mc_standard_error,
        "ci_95_upper": mc_price + 1.96 * mc_standard_error,
    }
)


# %% [markdown]
# ## Comparing convergence


# %%
def black_scholes_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


steps_grid = [5, 10, 25, 50, 100, 250, 500]
comparison = pd.DataFrame({"steps": steps_grid})
comparison["binomial_price"] = [
    binomial_option_price(100, 105, 0.06, 0.25, 1, steps=steps) for steps in steps_grid
]
comparison["black_scholes_price"] = black_scholes_call(100, 105, 0.06, 0.25, 1)
comparison["pricing_error"] = comparison["binomial_price"] - comparison["black_scholes_price"]
comparison

# %%
ax = comparison.plot(x="steps", y="pricing_error", marker="o", figsize=(8, 4), legend=False)
ax.axhline(0, color="black", linewidth=1)
ax.set_title("Binomial Price Convergence")
ax.set_xlabel("Steps")
ax.set_ylabel("Error versus Black-Scholes")
ax.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## Model limitations
#
# - Binomial prices depend on tree construction, step count, and convergence behavior near payoff kinks.
# - Monte Carlo estimates have sampling error and converge slowly for rare-event or path-dependent payoffs.
# - Numerical agreement with Black-Scholes only validates the implementation under the same simplified assumptions.
#
# ## Handoff
#
# Numerical standard error is part of model valuation uncertainty. It should not
# be mixed with market-risk P&L: first validate convergence and confidence
# intervals, then compute hedge sensitivities and aggregate residual scenario P&L
# at the portfolio level. The next lessons use market-implied volatility and an
# interactive dashboard to expose sensitivity and calibration risk.
