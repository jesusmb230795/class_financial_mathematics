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
# duration and duration-convexity approximations {cite}`fabozzi2019foundations`.
#
# The goal is not only to compute a bond price, but to build intuition about why fixed-income instruments react differently to changes in rates.
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
# sensitivity measures. Dashboard inputs remain decimals even when labels show
# percentages or basis points.
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

RUN_INTERACTIVE_WIDGETS = os.getenv("RUN_INTERACTIVE_WIDGETS", "1") == "1"


# %% [markdown]
# ## Pricing and risk functions


# %%
def bond_cash_flows(face_value, coupon_rate, maturity, frequency):
    periods = int(round(maturity * frequency))
    coupon = face_value * coupon_rate / frequency
    times = np.arange(1, periods + 1) / frequency
    cash_flows = np.full(periods, coupon)
    cash_flows[-1] += face_value
    return pd.DataFrame(
        {"period": np.arange(1, periods + 1), "time": times, "cash_flow": cash_flows}
    )


def bond_price(face_value, coupon_rate, maturity, ytm, frequency):
    cash_flows = bond_cash_flows(face_value, coupon_rate, maturity, frequency)
    period_yield = ytm / frequency
    cash_flows["discount_factor"] = 1 / (1 + period_yield) ** cash_flows["period"]
    cash_flows["present_value"] = cash_flows["cash_flow"] * cash_flows["discount_factor"]
    return cash_flows["present_value"].sum(), cash_flows


def duration_convexity(face_value, coupon_rate, maturity, ytm, frequency):
    price, cash_flows = bond_price(face_value, coupon_rate, maturity, ytm, frequency)
    period_yield = ytm / frequency
    weights = cash_flows["present_value"] / price
    macaulay_duration = (cash_flows["time"] * weights).sum()
    modified_duration = macaulay_duration / (1 + period_yield)
    convexity = (
        cash_flows["present_value"]
        * cash_flows["period"]
        * (cash_flows["period"] + 1)
        / (price * (1 + period_yield) ** 2 * frequency**2)
    ).sum()
    return price, macaulay_duration, modified_duration, convexity


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

    return pd.DataFrame(rows)


# %% [markdown]
# ## Interactive dashboard
#
# Run this cell in JupyterLab with `uv run jupyter lab`.


# %% tags=["interactive"]
def plot_bond_sensitivity(coupon_rate=0.08, maturity=5, ytm=0.07, frequency=2, shock_range_bps=300):
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
    )
    display(figure)
    if not RUN_INTERACTIVE_WIDGETS:
        plt.close(figure)


if RUN_INTERACTIVE_WIDGETS:
    interact(
        plot_bond_sensitivity,
        coupon_rate=FloatSlider(value=0.08, min=0.00, max=0.16, step=0.005, readout_format=".3f"),
        maturity=IntSlider(value=5, min=1, max=30, step=1),
        ytm=FloatSlider(value=0.07, min=0.01, max=0.18, step=0.005, readout_format=".3f"),
        frequency=IntSlider(value=2, min=1, max=4, step=1),
        shock_range_bps=IntSlider(value=300, min=50, max=800, step=50),
    )
else:
    plot_bond_sensitivity()

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
