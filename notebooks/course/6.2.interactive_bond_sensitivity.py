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
# # Interactive Bond Duration-Convexity Dashboard
#
# Module: Fixed Income, Credit, and Term Structure
#
# ## Lesson summary
#
# This notebook turns bond pricing, duration, and convexity into an interactive
# classroom tool. Students can change coupon rate, maturity, nominal annual yield,
# payment frequency, and interest-rate shocks to compare exact repricing against
# duration and duration-convexity approximations
# {cite}`fabozzi2019foundations,fisherWeil1971immunization`.
#
# The goal is not only to compute a bond price, but also to build intuition
# about why coupon, maturity, payment frequency, and yield change local
# interest-rate sensitivity.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - explain the inverse relationship between yield and price;
# - use duration to approximate price sensitivity;
# - explain why convexity improves the approximation for larger shocks;
# - identify which bond characteristics increase interest-rate risk;
# - use an interactive widget to test fixed-income scenarios.
#
# ## Prerequisites
#
# Complete the bond-pricing, duration, and convexity lesson first. Readers should
# be able to build a fixed-rate cash-flow schedule, distinguish nominal annual
# yield from its periodic rate, and interpret duration and convexity as local
# sensitivity measures. The interface labels coupon and yield inputs explicitly
# as decimals. Yield shocks are displayed in basis points, where 100 basis
# points equals a decimal change of 0.01.
#
# ## Duration-convexity approximation
#
# For a small parallel yield change $\Delta y$, the local price approximation is:
#
# $$
# \frac{\Delta P}{P}\approx -D_{\text{mod}}\Delta y+\frac{1}{2}C(\Delta y)^2.
# $$
#
# The dashboard compares this approximation against exact repricing. Here $y$ is
# a nominal annual yield convertible $m$ times per year:
#
# $$
# P(y)=\sum_{t=1}^{n}\frac{CF_t}{(1+y/m)^t}.
# $$
#
# ## Python setup

# %% tags=["setup", "hide-input"]
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import FloatSlider, IntSlider, interact

from src.dashboard_fallbacks import build_bond_sensitivity_dashboard_fallback
from src.dashboards import build_bond_sensitivity_dashboard
from src.fixed_income import (
    coupon_bond_cash_flows,
    present_value,
    risk_measures_from_cash_flows,
)

RUN_INTERACTIVE_WIDGETS = os.getenv("RUN_INTERACTIVE_WIDGETS", "1") == "1"


# %% [markdown]
# ## Pricing and risk functions


# %%
def bond_cash_flows(face_value, coupon_rate, maturity, frequency):
    return coupon_bond_cash_flows(face_value, coupon_rate, maturity, frequency)


def bond_price(face_value, coupon_rate, maturity, ytm, frequency):
    cash_flows = bond_cash_flows(face_value, coupon_rate, maturity, frequency)
    price = present_value(
        cash_flows["cash_flow"].to_numpy(),
        cash_flows["time"].to_numpy(),
        ytm,
        compounding=frequency,
    )
    return price, cash_flows


def duration_convexity(face_value, coupon_rate, maturity, ytm, frequency):
    cash_flows = bond_cash_flows(face_value, coupon_rate, maturity, frequency)
    risk = risk_measures_from_cash_flows(
        cash_flows["cash_flow"].to_numpy(),
        cash_flows["time"].to_numpy(),
        ytm,
        frequency=frequency,
    )
    return (
        risk["price"],
        risk["macaulay_duration"],
        risk["modified_duration"],
        risk["convexity"],
    )


# %% [markdown]
# ## Scenario table


# %%
def bond_scenario_table(face_value, coupon_rate, maturity, ytm, frequency, shock_range_bps):
    base_price, _, modified_duration, convexity = duration_convexity(
        face_value, coupon_rate, maturity, ytm, frequency
    )
    shocks = np.linspace(-shock_range_bps, shock_range_bps, 41) / 10_000
    rows = []

    for shock in shocks:
        exact_price, _ = bond_price(face_value, coupon_rate, maturity, ytm + shock, frequency)
        duration_price = base_price * (1 - modified_duration * shock)
        duration_convexity_price = base_price * (
            1 - modified_duration * shock + 0.5 * convexity * shock**2
        )
        rows.append(
            {
                "shock_bps": shock * 10_000,
                "exact_price": exact_price,
                "duration_price": duration_price,
                "duration_convexity_price": duration_convexity_price,
            }
        )

    scenario = pd.DataFrame(rows)
    scenario.attrs["figure_note"] = (
        "Deterministic classroom repricing; price is in currency units per "
        f"{face_value:g} face value; coupon={coupon_rate:.2%}; "
        f"maturity={maturity:g} years; nominal annual YTM={ytm:.2%}; "
        f"payments/year={frequency}; parallel shocks only."
    )
    return scenario


# %% [markdown]
# ## Interactive dashboard
#
# Run this cell in JupyterLab with `uv run jupyter lab`.


# %% tags=["interactive"] mystnb={"image": {"alt": "Interactive or static line chart comparing exact bond repricing, the duration approximation, and the duration-convexity approximation across parallel yield shocks in basis points for explicitly stated coupon, maturity, yield, and payment-frequency assumptions."}}
def plot_bond_sensitivity(
    coupon_rate=0.08,
    maturity=5,
    ytm=0.07,
    frequency=2,
    shock_range_bps=300,
):
    face_value = 100
    scenario = bond_scenario_table(
        face_value, coupon_rate, maturity, ytm, frequency, shock_range_bps
    )
    price, macaulay_duration, modified_duration, convexity = duration_convexity(
        face_value, coupon_rate, maturity, ytm, frequency
    )

    builder = (
        build_bond_sensitivity_dashboard
        if RUN_INTERACTIVE_WIDGETS
        else build_bond_sensitivity_dashboard_fallback
    )
    figure = builder(
        scenario,
        price=price,
        macaulay_duration=macaulay_duration,
        modified_duration=modified_duration,
        convexity=convexity,
        face_value=face_value,
        coupon_rate=coupon_rate,
        maturity=maturity,
        ytm=ytm,
        frequency=frequency,
    )
    display(figure)
    if not RUN_INTERACTIVE_WIDGETS:
        plt.close(figure)


if RUN_INTERACTIVE_WIDGETS:
    slider_style = {"description_width": "initial"}
    interact(
        plot_bond_sensitivity,
        coupon_rate=FloatSlider(
            value=0.08,
            min=0.0,
            max=0.16,
            step=0.005,
            description="Coupon rate (decimal)",
            readout_format=".3f",
            style=slider_style,
        ),
        maturity=IntSlider(
            value=5,
            min=1,
            max=30,
            step=1,
            description="Maturity (years)",
            style=slider_style,
        ),
        ytm=FloatSlider(
            value=0.07,
            min=0.01,
            max=0.18,
            step=0.005,
            description="YTM (decimal)",
            readout_format=".3f",
            style=slider_style,
        ),
        frequency=IntSlider(
            value=2,
            min=1,
            max=4,
            step=1,
            description="Payments per year",
            style=slider_style,
        ),
        shock_range_bps=IntSlider(
            value=300,
            min=50,
            max=800,
            step=50,
            description="Shock range (bp)",
            style=slider_style,
        ),
    )
else:
    plot_bond_sensitivity()

# %% [markdown]
# Every curve slopes downward around the base yield because a higher discount
# rate lowers present value. The duration-convexity curve generally remains
# closer to exact repricing than the duration-only line as the absolute shock
# grows. This is a local comparison under a single parallel YTM shock; it does
# not establish accuracy for nonparallel curve changes, credit-spread moves, or
# bonds with embedded options.

# %% [markdown]
# ## Model limitations
#
# - The dashboard applies simplified parallel yield shocks and does not model full curve reshaping.
# - Duration-convexity approximations are local and can misstate prices under large or nonparallel moves.
# - Coupon, frequency, and yield assumptions are classroom inputs rather than complete market conventions.

# %% [markdown]
# ## Handoff
#
# The Mexican government-security lesson now replaces the dashboard's generic
# bond assumptions with CETES, Bono M, and UDIBONO quotation and settlement
# conventions. Carry forward the price-yield intuition, but revalidate day count,
# coupon timing, accrued interest, and currency units for each instrument.
