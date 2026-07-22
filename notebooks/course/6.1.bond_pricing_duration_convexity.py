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
# # Bond Pricing, Duration, and Convexity
#
# Module: Fixed Income, Credit, and Term Structure
#
# ## Lesson summary
#
# A bond is a contract that pays scheduled cash flows. Its price is the present
# value of those promised payments under a discount rate or a full discount
# curve. Because bond prices move inversely with yields, fixed-income risk is
# commonly summarized with duration and convexity {cite}`fabozzi2019foundations`.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - build the cash-flow schedule of a fixed-rate bond;
# - price zero-coupon and coupon bonds;
# - estimate yield to maturity from a market price;
# - calculate Macaulay duration, modified duration, and convexity;
# - approximate price changes caused by yield shocks.
#
# ## Prerequisites
#
# Complete the quantitative-foundations lesson before this notebook. Readers
# should be able to discount dated cash flows, convert a quoted annual rate into
# the matching periodic rate, and keep rates as decimals in calculations. Every
# result below assumes one internally consistent currency, compounding frequency,
# settlement date, and cash-flow schedule.
#
# ## Core formulas
#
# For a fixed-rate bond with coupon rate $C$, face value $F$, nominal annual yield
# $y$ convertible $m$ times per year, and maturity $T$, the price is:
#
# $$
# P = \sum_{i=1}^{mT} \frac{C/m}{(1+y/m)^i} + \frac{F}{(1+y/m)^{mT}}.
# $$
#
# Macaulay duration is the present-value weighted average payment time:
#
# $$
# D_{Mac} = \frac{\sum_i t_i PV(C_i)}{P}.
# $$
#
# Modified duration approximates the percentage price change for a small yield movement:
#
# $$
# D_{mod} = \frac{D_{Mac}}{1+y/m}.
# $$
#
# The duration-convexity approximation is:
#
# $$
# \frac{\Delta P}{P} \approx -D_{mod}\Delta y + \frac{1}{2}Convexity(\Delta y)^2.
# $$
#
# ## Python setup

# %% tags=["setup", "hide-input"]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq


# %% [markdown]
# ## Bond cash-flow table


# %%
def bond_cash_flows(face_value=100, coupon_rate=0.08, maturity=5, frequency=2):
    periods = int(maturity * frequency)
    coupon = face_value * coupon_rate / frequency
    times = np.arange(1, periods + 1) / frequency
    cash_flows = np.full(periods, coupon)
    cash_flows[-1] += face_value
    return pd.DataFrame({"time": times, "cash_flow": cash_flows})


bond_cash_flows()


# %% [markdown]
# ## Pricing a coupon bond


# %%
def bond_price(face_value=100, coupon_rate=0.08, maturity=5, ytm=0.07, frequency=2):
    cf = bond_cash_flows(face_value, coupon_rate, maturity, frequency)
    period_yield = ytm / frequency
    periods = np.arange(1, len(cf) + 1)
    cf["discount_factor"] = 1 / (1 + period_yield) ** periods
    cf["present_value"] = cf["cash_flow"] * cf["discount_factor"]
    return cf["present_value"].sum(), cf


price, table = bond_price()
price

# %%
table.head()


# %% [markdown]
# ## Yield to maturity


# %%
def yield_to_maturity(market_price, face_value=100, coupon_rate=0.08, maturity=5, frequency=2):
    def pricing_error(y):
        price, _ = bond_price(face_value, coupon_rate, maturity, y, frequency)
        return price - market_price

    return brentq(pricing_error, -0.95, 1.00)


yield_to_maturity(price)


# %% [markdown]
# ## Duration and convexity


# %%
def bond_risk_measures(face_value=100, coupon_rate=0.08, maturity=5, ytm=0.07, frequency=2):
    price, cf = bond_price(face_value, coupon_rate, maturity, ytm, frequency)
    period_yield = ytm / frequency
    periods = np.arange(1, len(cf) + 1)
    weights = cf["present_value"] / price
    macaulay_duration = (cf["time"] * weights).sum()
    modified_duration = macaulay_duration / (1 + period_yield)
    convexity = (
        cf["present_value"]
        * periods
        * (periods + 1)
        / (price * (1 + period_yield) ** 2 * frequency**2)
    ).sum()
    return {
        "price": price,
        "macaulay_duration": macaulay_duration,
        "modified_duration": modified_duration,
        "convexity": convexity,
    }


bond_risk_measures()

# %% [markdown]
# ## Scenario analysis

# %%
base = bond_risk_measures()
shocks = np.array([-0.02, -0.01, 0.00, 0.01, 0.02])

scenario = pd.DataFrame({"yield_shock": shocks})
scenario["exact_price"] = [bond_price(ytm=0.07 + shock)[0] for shock in scenario["yield_shock"]]
scenario["duration_approximation"] = base["price"] * (
    1 - base["modified_duration"] * scenario["yield_shock"]
)
scenario["duration_convexity_approximation"] = base["price"] * (
    1
    - base["modified_duration"] * scenario["yield_shock"]
    + 0.5 * base["convexity"] * scenario["yield_shock"] ** 2
)
scenario

# %% [markdown]
# ## Model limitations
#
# - Duration is a local first-order approximation and deteriorates as yield shocks become larger.
# - Convexity improves the approximation but still assumes a simplified yield-shift structure.
# - Bond valuation results depend on day-count, compounding, settlement, and cash-flow convention choices.

# %% [markdown]
# ## Handoff
#
# The interactive sensitivity lesson reuses this cash-flow and risk-measure
# contract to compare exact repricing with duration and convexity approximations
# across user-selected shocks. Preserve the nominal-yield frequency and price
# units when moving from this deterministic calculation to the dashboard.
