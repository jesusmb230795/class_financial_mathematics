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
# # Yield, DV01, and Immunization Lab
#
# Module: Fixed Income, Credit, and Term Structure
#
# ## Lesson summary
#
# Yield to maturity converts a market price into a single internal rate for a
# cash-flow stream. Duration, convexity, and DV01 then translate that valuation
# into interest-rate risk. This lab combines root finding, second-order price
# approximation, and the Redington immunization conditions used in asset-liability
# management {cite}`fabozzi2019foundations`.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - solve yield to maturity from a market price;
# - compute Macaulay duration, modified duration, convexity, and DV01 from cash flows;
# - compare exact repricing with duration-only and duration-convexity approximations;
# - interpret the sign and scale of DV01;
# - evaluate the Redington immunization conditions for a simple asset-liability pair.
#
# ## Prerequisites
#
# Complete the bond-pricing and Mexican-market-convention lessons first. Readers
# should be able to discount a cash-flow vector, solve a price-yield relation,
# and interpret duration and convexity. The lab uses effective annual rates and
# cash-flow times in years unless a calculation explicitly declares another
# convention.
#
# ## Python setup

# %% tags=["setup", "hide-input"]
import numpy as np
import pandas as pd

from src.fixed_income import (
    duration_convexity_price,
    present_value,
    redington_immunization_check,
    risk_measures_from_cash_flows,
    yield_to_maturity_from_cash_flows,
)

# %% [markdown]
# ## Cash-flow stream
#
# The example is a simplified fixed-rate bond with annual cash flows and an
# effective annual YTM (`compounding=1`). The same workflow applies to semiannual
# or irregular cash flows only after the timing and quotation conventions are
# changed consistently.

# %%
times = np.array([1, 2, 3, 4, 5], dtype=float)
cash_flows = np.array([7, 7, 7, 7, 107], dtype=float)
market_price = 96.75

cash_flow_table = pd.DataFrame({"time_years": times, "cash_flow": cash_flows})
cash_flow_table

# %% [markdown]
# ## Yield to maturity by root finding
#
# Yield to maturity is the rate that sets present value equal to market price:
#
# $$
# P_{market} = \sum_i \frac{CF_i}{(1+y)^{t_i}}.
# $$

# %%
ytm = yield_to_maturity_from_cash_flows(
    market_price,
    cash_flows,
    times,
    compounding=1,
)
ytm

# %%
present_value(cash_flows, times, ytm, compounding=1)

# %% [markdown]
# ## Risk measures
#
# DV01 translates modified duration into the currency value of one basis point:
#
# $$
# DV01 = P \times D_{mod} \times 0.0001.
# $$

# %%
risk = risk_measures_from_cash_flows(cash_flows, times, ytm, frequency=1)
pd.Series(risk)

# %% [markdown]
# ## Exact repricing versus approximations
#
# For large shocks, convexity matters because bond prices are nonlinear in yield.

# %%
shocks = np.array([-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02])

scenario = pd.DataFrame({"yield_shock": shocks})
scenario["exact_price"] = [
    present_value(cash_flows, times, ytm + shock, compounding=1) for shock in shocks
]
scenario["duration_only"] = risk["price"] * (1 - risk["modified_duration"] * shocks)
scenario["duration_convexity"] = [
    duration_convexity_price(
        risk["price"],
        risk["modified_duration"],
        risk["convexity"],
        shock,
    )
    for shock in shocks
]
scenario["duration_error"] = scenario["duration_only"] - scenario["exact_price"]
scenario["duration_convexity_error"] = scenario["duration_convexity"] - scenario["exact_price"]
scenario

# %% [markdown]
# ## Redington immunization check
#
# Redington immunization is a local surplus-protection condition for parallel yield shifts. A simple version requires:
#
# 1. present value of assets equals present value of liabilities;
# 2. asset duration equals liability duration;
# 3. asset convexity exceeds liability convexity.

# %%
asset = {
    "pv": 100.0,
    "duration": 5.25,
    "convexity": 43.0,
}

liability = {
    "pv": 100.0,
    "duration": 5.25,
    "convexity": 36.0,
}

redington_immunization_check(
    asset_pv=asset["pv"],
    liability_pv=liability["pv"],
    asset_duration=asset["duration"],
    liability_duration=liability["duration"],
    asset_convexity=asset["convexity"],
    liability_convexity=liability["convexity"],
)

# %% [markdown]
# The condition is local, not universal. It protects against small parallel shocks around the valuation yield. Non-parallel shifts, optionality, credit migration, liquidity, and cash-flow uncertainty require richer models.
#
# ## Model limitations
#
# - Yield to maturity compresses an entire cash-flow stream into one rate and can hide reinvestment assumptions.
# - DV01 and duration are local measures and do not fully capture nonlinear or nonparallel curve moves.
# - Redington immunization is a local surplus condition, not a guarantee against all interest-rate scenarios.

# %% [markdown]
# ## Handoff
#
# Credit-spread and securitization analysis adds default, recovery, liquidity,
# embedded-option, and waterfall risk to these deterministic rate measures.
# Retain separate benchmark-rate and credit-spread assumptions so that DV01,
# spread risk, and expected credit loss are not collapsed into one number.
