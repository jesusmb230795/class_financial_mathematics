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
# approximation, and the Redington immunization conditions used in
# asset-liability management
# {cite}`redington1952immunization,fisherWeil1971immunization,fabozzi2019foundations`.
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
# changed consistently. Cash flows and price are currency units per one modeled
# position, and all times are years from the valuation date.

# %%
times = np.array([1, 2, 3, 4, 5], dtype=float)
cash_flows = np.array([7, 7, 7, 7, 107], dtype=float)
market_price = 96.75  # currency units per modeled position

cash_flow_table = pd.DataFrame(
    {"time_years": times, "cash_flow_currency_units": cash_flows}
)
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
# DV01 is reported here as a **positive price-change magnitude** for a one-basis-
# point parallel yield move. Because a standard option-free bond has positive
# modified duration, a yield increase lowers its price:
#
# $$
# \operatorname{DV01}:=-\frac{\partial P}{\partial y}10^{-4}
# \approx P D_{\mathrm{mod}}10^{-4}>0,
# \qquad
# \Delta y=+1\text{ bp}\Longrightarrow\Delta P\approx-\operatorname{DV01}.
# $$
#
# Price and DV01 share the cash-flow currency unit. In this example, price is
# currency units per one modeled position and DV01 is currency units per basis
# point. The sign of a scenario P&L comes from the yield-shock direction, not
# from changing DV01 into a signed exposure.

# %%
risk = risk_measures_from_cash_flows(cash_flows, times, ytm, frequency=1)
pd.Series(risk).rename(
    {
        "price": "price_currency_units",
        "macaulay_duration": "macaulay_duration_years",
        "modified_duration": "modified_duration_years",
        "convexity": "convexity_years_squared",
        "dv01": "dv01_currency_units_per_bp",
    }
)

# %% [markdown]
# ## Exact repricing versus approximations
#
# For large shocks, convexity matters because bond prices are nonlinear in yield.

# %%
shocks = np.array([-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02])

scenario = pd.DataFrame(
    {
        "yield_shock_decimal": shocks,
        "yield_shock_bps": shocks * 10_000,
    }
)
scenario["exact_price_currency_units"] = [
    present_value(cash_flows, times, ytm + shock, compounding=1) for shock in shocks
]
scenario["duration_only_price"] = risk["price"] * (
    1 - risk["modified_duration"] * shocks
)
scenario["duration_convexity_price"] = [
    duration_convexity_price(
        risk["price"],
        risk["modified_duration"],
        risk["convexity"],
        shock,
    )
    for shock in shocks
]
scenario["duration_error_currency_units"] = (
    scenario["duration_only_price"] - scenario["exact_price_currency_units"]
)
scenario["duration_convexity_error_currency_units"] = (
    scenario["duration_convexity_price"] - scenario["exact_price_currency_units"]
)
scenario

# %% [markdown]
# ## Redington immunization check
#
# Redington immunization is a local surplus-protection condition for parallel
# yield shifts. At the valuation yield $y_0$, a simple version requires:
#
# 1. $PV_A(y_0)=PV_L(y_0)$;
# 2. $D_A(y_0)=D_L(y_0)$; and
# 3. $\mathcal{C}_A(y_0)>\mathcal{C}_L(y_0)$.
#
# These conditions make the surplus $S(y)=PV_A(y)-PV_L(y)$ equal to zero with a
# zero first derivative and positive local curvature. They do not immunize every
# nonparallel change in the term structure
# {cite}`redington1952immunization,fisherWeil1971immunization`.
#
# The following example derives every metric from cash flows. At a 6% effective
# annual valuation yield, a single liability due in year 5 has a present value
# of MXN 100 million. The asset portfolio places half of that present value in a
# year-3 cash flow and half in a year-7 cash flow. Its present-value-weighted
# time is therefore also five years, while dispersion around year 5 gives it
# greater convexity.

# %%
immunization_yield = 0.06
asset_times = np.array([3.0, 7.0])
asset_cash_flows = np.array(
    [
        50.0 * (1 + immunization_yield) ** 3,
        50.0 * (1 + immunization_yield) ** 7,
    ]
)
liability_times = np.array([5.0])
liability_cash_flows = np.array([100.0 * (1 + immunization_yield) ** 5])

asset_risk = risk_measures_from_cash_flows(
    asset_cash_flows,
    asset_times,
    immunization_yield,
    frequency=1,
)
liability_risk = risk_measures_from_cash_flows(
    liability_cash_flows,
    liability_times,
    immunization_yield,
    frequency=1,
)

cash_flow_design = pd.DataFrame(
    {
        "side": ["Asset", "Asset", "Liability"],
        "time_years": [3.0, 7.0, 5.0],
        "cash_flow_mxn_millions": [*asset_cash_flows, *liability_cash_flows],
        "target_present_value_mxn_millions": [50.0, 50.0, 100.0],
    }
)
cash_flow_design

# %%
immunization_metrics = pd.DataFrame(
    {
        "asset": {
            "present_value_mxn_millions": asset_risk["price"],
            "modified_duration_years": asset_risk["modified_duration"],
            "convexity_years_squared": asset_risk["convexity"],
        },
        "liability": {
            "present_value_mxn_millions": liability_risk["price"],
            "modified_duration_years": liability_risk["modified_duration"],
            "convexity_years_squared": liability_risk["convexity"],
        },
    }
)
immunization_metrics

# %%
redington_result = redington_immunization_check(
    asset_pv=asset_risk["price"],
    liability_pv=liability_risk["price"],
    asset_duration=asset_risk["modified_duration"],
    liability_duration=liability_risk["modified_duration"],
    asset_convexity=asset_risk["convexity"],
    liability_convexity=liability_risk["convexity"],
)
pd.Series(redington_result, name="condition_satisfied")

# %% [markdown]
# All three conditions pass because the cash flows were constructed to match
# present value and modified duration while increasing convexity. The condition
# is local, not universal: it protects against small parallel shocks around the
# 6% effective annual valuation yield. Nonparallel shifts, optionality, credit
# migration, liquidity, and cash-flow uncertainty require richer models.
#
# ## Model limitations
#
# - Yield to maturity compresses an entire cash-flow stream into one rate and can hide reinvestment assumptions.
# - DV01 and duration are local measures and do not fully capture nonlinear or
#   nonparallel curve moves. Report the price unit, compounding, and basis-point
#   sign convention with every DV01.
# - Redington immunization is a local surplus condition, not a guarantee against
#   all interest-rate scenarios. The constructed cash flows ignore transaction
#   costs, rebalancing, default, and liability uncertainty.

# %% [markdown]
# ## Handoff
#
# Credit-spread and securitization analysis adds default, recovery, liquidity,
# embedded-option, and waterfall risk to these deterministic rate measures.
# Retain separate benchmark-rate and credit-spread assumptions so that DV01,
# spread risk, and expected credit loss are not collapsed into one number.
