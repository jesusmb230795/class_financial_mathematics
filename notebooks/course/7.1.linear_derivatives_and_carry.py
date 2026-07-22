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
# # Linear Derivatives and Carry
#
# Module: Derivatives and Risk Management
#
# ## Lesson summary
#
# Forwards, futures, and swaps are the linear foundation of derivatives pricing.
# Their values are driven by no-arbitrage carry, discounting, and curve
# consistency rather than by payoff convexity. This lesson connects spot price,
# dividend yield, interest rates, mark-to-market value, and par swap rates
# {cite}`hull2022options`.
#
# The examples use deterministic inputs. Mexican TIIE, CETES, and FX series can be connected later through the existing data layer, but this notebook does not require credentials or network access.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - compute a forward price under continuous cost of carry;
# - mark an existing forward contract to market;
# - explain when futures and forwards differ conceptually;
# - compute a par swap rate from discount factors and accrual fractions;
# - interpret carry assumptions for FX and equity-linked derivatives.
#
# ## Prerequisites
#
# Complete the quantitative-foundations lesson and the fixed-income curve
# sequence first. Students should be able to discount a dated cash flow, convert
# between rate and basis-point units, distinguish simple from continuous
# compounding, and read an FX quote without reversing the base and quote
# currencies.
#
# ## Python setup

# %% tags=["setup", "hide-input"]
import numpy as np
import pandas as pd

from src.derivatives import forward_price, forward_value, par_swap_rate

# %% [markdown]
# ## Forward price with continuous carry
#
# For an asset with spot price $S_0$, risk-free rate $r$, dividend or convenience yield $q$, and maturity $T$:
#
# $$
# F(0,T) = S_0 e^{(r-q)T}.
# $$

# %%
spot = 100.0
rate = 0.095
maturity = 0.5
dividend_yield = 0.025

forward = forward_price(spot, rate, maturity, dividend_yield)
forward

# %% [markdown]
# ## Mark-to-market value
#
# If a forward was originally struck at $K$, the value to the long side before maturity is:
#
# $$
# V_t = S_t e^{-q(T-t)} - K e^{-r(T-t)}.
# $$

# %%
strikes = [95, 100, forward, 110]

pd.DataFrame(
    {
        "strike": strikes,
        "long_forward_value": [
            forward_value(spot, strike, rate, maturity, dividend_yield) for strike in strikes
        ],
    }
)

# %% [markdown]
# ## Forward delta hedge
#
# With continuous carry, a long forward has spot delta $e^{-qT}$ per unit of
# underlying. A local delta hedge therefore shorts that many spot units. This
# hedge addresses a spot move; it does not remove funding, basis, dividend,
# counterparty, or liquidity risk.

# %%
forward_delta = np.exp(-dividend_yield * maturity)
pd.Series(
    {
        "long_forward_delta": forward_delta,
        "spot_units_for_delta_hedge": -forward_delta,
        "contract_maturity_years": maturity,
    }
)

# %% [markdown]
# ## Carry scenarios
#
# Carry explains why forward prices can be above or below spot.

# %%
scenarios = pd.DataFrame(
    {
        "scenario": ["high domestic rate", "high dividend yield", "flat carry"],
        "rate": [0.10, 0.04, 0.06],
        "dividend_yield": [0.02, 0.08, 0.06],
    }
)
scenarios["forward_price"] = [
    forward_price(spot, r, 1.0, q) for r, q in zip(scenarios["rate"], scenarios["dividend_yield"])
]
scenarios["forward_minus_spot"] = scenarios["forward_price"] - spot
scenarios

# %% [markdown]
# ## Par swap rate
#
# A fixed-for-floating interest-rate swap can be valued as a fixed leg against a floating leg. At inception, the par fixed rate sets the swap value to zero:
#
# $$
# R_{swap} =
# \frac{1 - D(0,T_n)}
# {\sum_{i=1}^{n}\tau_i D(0,T_i)}.
# $$

# %%
payment_dates = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
zero_rates_continuous = np.array([0.087, 0.085, 0.083, 0.082, 0.081, 0.080])
discount_factors = np.exp(-zero_rates_continuous * payment_dates)
accruals = np.full_like(payment_dates, 0.5)

swap_table = pd.DataFrame(
    {
        "payment_date": payment_dates,
        "zero_rate_continuous": zero_rates_continuous,
        "discount_factor": discount_factors,
        "accrual": accruals,
    }
)
swap_table

# %%
par_rate = par_swap_rate(discount_factors, accruals)
par_rate

# %% [markdown]
# ## Interpretation notes
#
# | Contract | Core pricing input | Main classroom risk |
# | --- | --- | --- |
# | Forward | Spot, carry, maturity | Using an inconsistent dividend or funding assumption |
# | Future | Forward logic plus margining | Ignoring stochastic-rate convexity adjustment |
# | Swap | Discount curve and accrual factors | Mixing discount and projection curves without documentation |
#
# ## Model limitations
#
# - Carry formulas depend on clean inputs for funding, income, storage, convenience yield, and collateral assumptions.
# - Forward and futures prices can differ when rates, margins, taxes, or settlement mechanics matter.
# - Par swap calculations require a reliable discount curve and cash-flow conventions.
#
# ## Handoff
#
# A no-arbitrage forward or par swap rate is an inception price, not a risk
# measure. Desks map the priced contract to spot delta, curve-key-rate exposure,
# carry, basis, and counterparty scenarios; the resulting P&L is then aggregated
# for limits, stress tests, VaR, and Expected Shortfall. The next lesson adds
# nonlinear European option prices and Greeks to this carry-and-discounting
# foundation.

# %% [markdown] tags=["exercise"]
# ## Checkpoint exercise
#
# For a one-year forward, compare $(r,q)$ pairs of (10%, 2%), (6%, 6%), and
# (4%, 8%) under continuous compounding.
#
# 1. Report forward price, forward-minus-spot, long-forward delta, and the spot
#    hedge per unit notional.
# 2. Shock every continuously compounded zero rate in the swap table by +1 basis
#    point and report the change in par swap rate.
# 3. Explain why a zero-value inception price does not imply zero market risk.
# 4. Name the carry, curve, basis, and counterparty inputs required before the
#    positions enter portfolio risk reports.
#

# %% [markdown] tags=["solution"]
# ```{dropdown} Suggested answer rubric
# A complete answer labels continuous compounding, reports prices in currency and
# rates in basis points, uses the hedge sign correctly, and connects valuation
# inputs to portfolio scenario P&L.
# ```
