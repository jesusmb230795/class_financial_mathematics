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
# # European Option Pricing and Greeks
#
# Module: Derivatives and Risk Management
#
# ## Lesson summary
#
# Options give the holder the right, but not the obligation, to buy or sell an
# underlying asset. The Black-Scholes model prices European options under
# idealized assumptions and provides sensitivity measures known as Greeks
# {cite}`hull2022options`.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - define call and put option payoffs;
# - state the main Black-Scholes assumptions;
# - price European calls and puts;
# - verify put-call parity;
# - calculate and interpret core Greeks.
#
# ## Prerequisites
#
# Complete Linear Derivatives and Carry first. Students should understand
# discounted present value, continuous compounding, no-arbitrage replication,
# normal-distribution notation, and the difference between a terminal payoff and
# a time-zero price.
#
# ## Payoffs
#
# For a European call with strike $K$ and terminal stock price $S_T$:
#
# $$
# CallPayoff = \max(S_T - K, 0).
# $$
#
# For a European put:
#
# $$
# PutPayoff = \max(K - S_T, 0).
# $$
#
# ## Black-Scholes formulas
#
# For a non-dividend-paying asset:
#
# $$
# C = S_0N(d_1) - Ke^{-rT}N(d_2),
# $$
#
# $$
# P = Ke^{-rT}N(-d_2) - S_0N(-d_1),
# $$
#
# where:
#
# $$
# d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}},
# $$
#
# $$
# d_2 = d_1 - \sigma\sqrt{T}.
# $$
#
# ## Python setup

# %% tags=["setup", "hide-input"]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


# %% [markdown]
# ## Black-Scholes implementation


# %%
def black_scholes(S0, K, r, sigma, T, option_type="call"):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


S0 = 100
K = 105
r = 0.06
sigma = 0.25
T = 1

call_price = black_scholes(S0, K, r, sigma, T, "call")
put_price = black_scholes(S0, K, r, sigma, T, "put")
call_price, put_price

# %% [markdown]
# ## Put-call parity

# %%
left_side = call_price + K * np.exp(-r * T)
right_side = put_price + S0
left_side, right_side, left_side - right_side


# %% [markdown]
# ## Greeks


# %%
def black_scholes_greeks(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta_call = norm.cdf(d1)
    delta_put = norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    vega = S0 * norm.pdf(d1) * np.sqrt(T)
    theta_call = -S0 * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(
        d2
    )
    theta_put = -S0 * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(
        -d2
    )
    rho_call = K * T * np.exp(-r * T) * norm.cdf(d2)
    rho_put = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    return pd.Series(
        {
            "delta_call": delta_call,
            "delta_put": delta_put,
            "gamma": gamma,
            "vega_per_1pct": vega / 100,
            "theta_call_per_day": theta_call / 365,
            "theta_put_per_day": theta_put / 365,
            "rho_call_per_1pct": rho_call / 100,
            "rho_put_per_1pct": rho_put / 100,
        }
    )


base_greeks = black_scholes_greeks(S0, K, r, sigma, T)
base_greeks

# %% [markdown]
# ## From price to a local delta hedge
#
# A dealer short one call has option delta $-\Delta_C$. Buying $\Delta_C$ units
# of the underlying offsets the first-order spot exposure. This is a local hedge:
# gamma, volatility, time, jumps, funding, and discrete rebalancing still create
# P&L.

# %%
spot_scenarios = np.array([S0 - 1, S0, S0 + 1], dtype=float)
delta_hedge = base_greeks["delta_call"]
hedge_pnl = pd.DataFrame({"new_spot": spot_scenarios})
hedge_pnl["short_call_pnl"] = [
    call_price - black_scholes(spot, K, r, sigma, T, "call") for spot in spot_scenarios
]
hedge_pnl["stock_hedge_pnl"] = delta_hedge * (hedge_pnl["new_spot"] - S0)
hedge_pnl["delta_hedged_pnl"] = hedge_pnl["short_call_pnl"] + hedge_pnl["stock_hedge_pnl"]
hedge_pnl

# %% [markdown]
# ## Payoff diagram

# %%
terminal_prices = np.linspace(50, 160, 200)
payoffs = pd.DataFrame(
    {
        "terminal_price": terminal_prices,
        "call_payoff": np.maximum(terminal_prices - K, 0),
        "put_payoff": np.maximum(K - terminal_prices, 0),
    }
)

ax = payoffs.plot(x="terminal_price", y=["call_payoff", "put_payoff"], figsize=(8, 4))
ax.set_title("European Option Payoffs")
ax.set_xlabel("Terminal stock price")
ax.set_ylabel("Payoff")
ax.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## Model limitations
#
# - Black-Scholes assumes continuous trading, constant volatility, lognormal dynamics, and frictionless markets.
# - Real option markets show smiles, skews, jumps, liquidity effects, and discrete hedging error.
# - Greeks are local sensitivities and can change quickly near maturity or around large spot moves.
#
# ## Handoff
#
# Model price establishes present value; Greeks translate that value into local
# exposures; hedge P&L and scenario P&L reveal residual risk. Those residual
# portfolio losses, rather than option prices alone, become inputs to VaR,
# Expected Shortfall, limits, and model-governance review. The next lesson tests
# the same pricing logic with binomial and Monte Carlo numerical evidence.

# %% [markdown] tags=["exercise"]
# ## Checkpoint exercise
#
# For the displayed call, compare spot moves of -MXN 5, -MXN 1, +MXN 1, and
# +MXN 5 with no passage of time.
#
# 1. Verify put-call parity to numerical tolerance.
# 2. Report unhedged short-call P&L, stock-hedge P&L, and delta-hedged P&L for
#    each spot scenario.
# 3. Explain why the delta hedge is more accurate for the one-unit moves and
#    connect the residual to gamma.
# 4. Name two risks that this static, frictionless hedge omits.
#

# %% [markdown] tags=["solution"]
# ```{dropdown} Suggested answer rubric
# A complete answer reports prices and P&L in currency units, verifies parity,
# uses the short-option hedge sign correctly, and distinguishes local Greek
# hedging from a portfolio risk limit.
# ```
