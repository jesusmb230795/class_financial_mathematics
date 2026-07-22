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
# # Implied Volatility and Smiles
#
# Module: Derivatives and Risk Management
#
# ## Lesson summary
#
# Black-Scholes takes volatility as an input. Markets quote option prices, so
# analysts invert the model to recover implied volatility. The resulting
# volatilities vary by strike and maturity, creating a smile or skew that exposes
# the limits of constant-volatility pricing {cite}`hull2022options`.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - solve implied volatility from a market option price;
# - build a simple implied-volatility smile by strike;
# - distinguish model price, market price, and implied volatility;
# - identify basic no-arbitrage checks before inversion;
# - explain why implied volatility is not a direct historical volatility forecast.
#
# ## Prerequisites
#
# Complete European Option Pricing and Greeks and the numerical-pricing lesson
# first. Students should understand option-price bounds, put-call parity, vega,
# numerical root finding, and the difference between a calibrated model input
# and a forecast of realized volatility.
#
# ## Python setup

# %% tags=["setup", "hide-input"]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.derivatives import (
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility,
    put_call_parity_gap,
)

# %% [markdown]
# ## Synthetic option chain
#
# The market prices below are generated from a deterministic skew function. In a live-data lab, this table would come from an option chain, but the same root-finding workflow applies.

# %%
spot = 100.0
rate = 0.055
dividend_yield = 0.01
maturity = 0.75
strikes = np.array([75, 85, 95, 100, 105, 115, 125])

true_smile = 0.22 + 0.10 * ((strikes / spot) - 1) ** 2 - 0.08 * ((strikes / spot) - 1)
market_calls = [
    black_scholes_price(
        spot,
        strike,
        rate,
        vol,
        maturity,
        option_type="call",
        dividend_yield=dividend_yield,
    )
    for strike, vol in zip(strikes, true_smile)
]

chain = pd.DataFrame(
    {
        "strike": strikes,
        "market_call_price": market_calls,
        "generating_volatility": true_smile,
    }
)
chain

# %% [markdown]
# ## Invert Black-Scholes
#
# Implied volatility solves:
#
# $$
# C_{BS}(\sigma_{imp}) - C_{market} = 0.
# $$

# %%
chain["implied_volatility"] = [
    implied_volatility(
        price,
        spot,
        strike,
        rate,
        maturity,
        option_type="call",
        dividend_yield=dividend_yield,
    )
    for price, strike in zip(chain["market_call_price"], chain["strike"])
]
chain["repricing_error"] = [
    black_scholes_price(
        spot,
        strike,
        rate,
        vol,
        maturity,
        option_type="call",
        dividend_yield=dividend_yield,
    )
    - price
    for price, strike, vol in zip(
        chain["market_call_price"],
        chain["strike"],
        chain["implied_volatility"],
    )
]
chain["vega_per_1pct"] = [
    black_scholes_greeks(
        spot,
        strike,
        rate,
        vol,
        maturity,
        option_type="call",
        dividend_yield=dividend_yield,
    )["vega_per_1pct"]
    for strike, vol in zip(chain["strike"], chain["implied_volatility"])
]
chain

# %% [markdown]
# ## Smile visualization

# %%
ax = chain.plot(
    x="strike",
    y=["generating_volatility", "implied_volatility"],
    marker="o",
    figsize=(8, 4),
)
ax.set_title("Synthetic Implied Volatility Smile")
ax.set_xlabel("Strike")
ax.set_ylabel("Volatility")
ax.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## Put-call parity with dividends
#
# For continuous dividend yield $q$:
#
# $$
# C + K e^{-rT} = P + S_0 e^{-qT}.
# $$

# %%
strike = 100
volatility = chain.loc[chain["strike"] == strike, "implied_volatility"].iloc[0]
call = black_scholes_price(spot, strike, rate, volatility, maturity, "call", dividend_yield)
put = black_scholes_price(spot, strike, rate, volatility, maturity, "put", dividend_yield)

pd.Series(
    {
        "call": call,
        "put": put,
        "parity_gap": put_call_parity_gap(
            call,
            put,
            spot,
            strike,
            rate,
            maturity,
            dividend_yield,
        ),
    }
)

# %% [markdown]
# ## Surface extension
#
# An implied volatility surface extends the smile across maturities. A production-quality surface must also be checked for calendar-spread and butterfly arbitrage. In this course, the first objective is operational: invert prices carefully, document inputs, and avoid treating noisy points as a smooth truth.
#
# ## Model limitations
#
# - Implied volatility is model-implied, not directly observed volatility.
# - Failed inversions can come from bad quotes, arbitrage violations, stale data, or numerical bounds.
# - A smile fit at one maturity does not define a complete arbitrage-free volatility surface.
#
# ## Handoff
#
# Implied volatility makes model prices match market quotes at a point. Vega
# translates a volatility-surface shock into local P&L; sticky-strike versus
# sticky-delta rules, smile dynamics, and failed no-arbitrage checks are model-risk
# scenarios that must be carried into hedging and limit reports. The interactive
# dashboard next consolidates price, parity, and Greek diagnostics before the
# sequence moves to early exercise, path dependence, and stochastic volatility.
