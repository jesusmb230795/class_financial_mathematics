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
# commonly summarized with duration and convexity. Macaulay introduced the
# weighted-average time measure that underlies this treatment, while modern
# fixed-income texts connect it to price sensitivity
# {cite}`macaulay1938interest,fabozzi2019foundations`.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - build the cash-flow schedule of a fixed-rate bond;
# - price zero-coupon and coupon bonds;
# - recover yield to maturity in a controlled price-yield round-trip;
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
# For a fixed-rate bond with annual coupon rate $C$ (decimal), face value $F$ in
# currency units, nominal annual yield $y$ (decimal) convertible $m$ times per
# year, and maturity $T$ years, the coupon cash payment each period is
#
# $$
# c=\frac{FC}{m}.
# $$
#
# If $mT$ is an integer and valuation occurs on a coupon date, the price in the
# same currency units as $F$ is:
#
# $$
# P(y) = \sum_{i=1}^{mT} \frac{FC/m}{(1+y/m)^i}
#       + \frac{F}{(1+y/m)^{mT}}.
# $$
#
# A zero-coupon bond is the special case $C=0$, so
# $P(y)=F(1+y/m)^{-mT}$. These formulas assume a flat YTM and omit accrued
# interest; curve-based valuation later replaces the single yield with a
# maturity-specific discount factor.
#
# For cash flows $CF_i$ paid at times $t_i$ in years, Macaulay duration is the
# present-value-weighted average payment time:
#
# $$
# D_{\mathrm{Mac}} = \frac{\sum_i t_i\,PV(CF_i)}{P}.
# $$
#
# Modified duration is measured in years and relates price to a small parallel
# change in the quoted nominal annual yield:
#
# $$
# D_{\mathrm{mod}} = \frac{D_{\mathrm{Mac}}}{1+y/m}
# =-\frac{1}{P}\frac{\partial P}{\partial y}.
# $$
#
# With convexity
# $\mathcal{C}=P^{-1}\partial^2P/\partial y^2$ measured in years squared, the
# second-order approximation is:
#
# $$
# \frac{\Delta P}{P} \approx -D_{\mathrm{mod}}\Delta y
# + \frac{1}{2}\mathcal{C}(\Delta y)^2.
# $$
#
# Here $\Delta y$ is a decimal change in the same yield convention: 100 basis
# points equals $0.01$. Duration and convexity are local measures, and matching
# cash-flow timing rather than only a scalar duration becomes important when the
# curve shifts nonparallel {cite}`fisherWeil1971immunization`.
#
# ## Python setup

# %% tags=["setup", "hide-input"]
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.fixed_income import coupon_bond_cash_flows


# %% [markdown]
# ## Bond cash-flow table


# %%
def bond_cash_flows(face_value=100, coupon_rate=0.08, maturity=5, frequency=2):
    return coupon_bond_cash_flows(face_value, coupon_rate, maturity, frequency)


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

scenario = pd.DataFrame(
    {
        "yield_shock_decimal": shocks,
        "yield_shock_bps": shocks * 10_000,
    }
)
scenario["exact_price_currency_units"] = [
    bond_price(ytm=0.07 + shock)[0] for shock in scenario["yield_shock_decimal"]
]
scenario["duration_approximation"] = base["price"] * (
    1 - base["modified_duration"] * scenario["yield_shock_decimal"]
)
scenario["duration_convexity_approximation"] = base["price"] * (
    1
    - base["modified_duration"] * scenario["yield_shock_decimal"]
    + 0.5 * base["convexity"] * scenario["yield_shock_decimal"] ** 2
)
scenario

# %% [markdown]
# The exact and approximate prices are currency units per 100 units of face
# value. All three values agree at a zero shock. As the absolute shock grows,
# the duration-only error increases; the convexity term captures part of the
# curvature but still assumes one parallel movement in the flat YTM.

# %% [markdown]
# ## Model limitations
#
# - Duration is a local first-order approximation and deteriorates as yield
#   shocks become larger.
# - Convexity improves the approximation but still assumes a parallel change in
#   one flat YTM; it does not represent key-rate or spread risk.
# - The examples value promised cash flows on a coupon date. Real dirty and
#   clean prices depend on settlement, accrued interest, calendars, day count,
#   taxes, optionality, and credit.

# %% [markdown]
# ## Handoff
#
# The interactive sensitivity lesson reuses this cash-flow and risk-measure
# contract to compare exact repricing with duration and convexity approximations
# across user-selected shocks. Preserve the nominal-yield frequency and price
# units when moving from this deterministic calculation to the dashboard.
