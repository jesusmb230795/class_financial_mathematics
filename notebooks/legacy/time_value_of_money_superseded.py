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

# %% [markdown] tags=["legacy"]
# # Time Value of Money
#
# Module: Quantitative Methods and Financial Time Series (legacy source)
#
# ## Lesson summary
#
# Financial mathematics starts with the idea that money has a time dimension. A cash flow received today is not directly comparable to a cash flow received in the future unless both are moved to the same date with an interest rate or a discount factor {cite}`fabozzi2019foundations`.
#
# This lesson builds the notation used later for bonds, yield curves, derivatives, portfolio valuation, and risk management.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - compute present value and future value under simple, compound, and continuous compounding;
# - convert between interest rates and discount factors;
# - value a finite stream of deterministic cash flows;
# - distinguish nominal, effective, and continuously compounded rates;
# - explain why the discount rate is a valuation assumption, not just a formula input.
#
# ## Core concepts
#
# ### Future value
#
# Under simple interest, interest is calculated only on the original principal:
#
# $$
# FV_T=C_0(1+rT).
# $$
#
# If an amount $C_0$ is invested for $T$ years at a nominal annual rate $r^{(m)}$
# convertible $m$ times per year, the future value is:
#
# $$
# FV_T = C_0\left(1+\frac{r^{(m)}}{m}\right)^{mT}.
# $$
#
# With continuous compounding:
#
# $$
# FV_T = C_0 e^{rT}.
# $$
#
# ### Present value
#
# The present value of a future cash flow $C_T$ is:
#
# $$
# PV_0 = \frac{C_T}{\left(1+r^{(m)}/m\right)^{mT}}.
# $$
#
# With continuous compounding:
#
# $$
# PV_0 = C_T e^{-rT}.
# $$
#
# ### Discount factors
#
# A discount factor $D(0,T)$ is the price today of one monetary unit paid at time $T$:
#
# $$
# D(0,T) = \frac{1}{(1+r_T)^T}.
# $$
#
# Once discount factors are known, any deterministic cash-flow stream can be valued as:
#
# $$
# PV_0 = \sum_{i=1}^{n} C_i D(0,t_i).
# $$
#
# ### Equivalent annual rate quotations
#
# A nominal rate convertible $m$ times, its effective annual rate, and its
# continuously compounded equivalent satisfy:
#
# $$
# 1+r_{\mathrm{eff}}
# =\left(1+\frac{r^{(m)}}{m}\right)^m
# =e^{r_c}.
# $$
#
# A rate is incomplete without its compounding convention. Equivalent quotations
# must produce the same one-year accumulation factor.
#
# ## Python setup

# %% tags=["setup", "hide-input", "legacy"]
import numpy as np
import pandas as pd

from src.fixed_income import (
    continuous_rate_from_effective,
    discount_factor,
    effective_annual_rate,
    effective_rate_from_continuous,
    future_value,
    nominal_rate_from_effective,
    present_value,
    simple_future_value,
    simple_present_value,
)


# %% [markdown] tags=["legacy"]
# ## Simple, periodic, and continuous accumulation

# %% tags=["legacy"]
principal = 1000
rate = 0.08
years = 5

accumulation_comparison = pd.Series(
    {
        "simple_interest": simple_future_value(principal, rate, years),
        "annual_compounding": future_value(principal, rate, years, compounding=1),
        "quarterly_compounding": future_value(principal, rate, years, compounding=4),
        "continuous_compounding": future_value(principal, rate, years, compounding=None),
    },
    name="future_value_mxn",
)
accumulation_comparison

# %% tags=["legacy"]
roundtrip = pd.Series(
    {
        "simple": simple_present_value(
            accumulation_comparison["simple_interest"],
            rate,
            years,
        ),
        "annual": accumulation_comparison["annual_compounding"]
        * discount_factor(rate, years, compounding=1),
        "quarterly": accumulation_comparison["quarterly_compounding"]
        * discount_factor(rate, years, compounding=4),
        "continuous": accumulation_comparison["continuous_compounding"]
        * discount_factor(rate, years, compounding=None),
    },
    name="recovered_present_value_mxn",
)
roundtrip.to_frame()

# %% [markdown] tags=["legacy"]
# Every row must recover MXN 1,000. A roundtrip test catches a frequent error:
# accumulating under one convention and discounting under another.

# %% [markdown] tags=["legacy"]
# ## Nominal, effective, and continuous rate conversion

# %% tags=["legacy"]
nominal_quarterly = 0.08
effective = effective_annual_rate(nominal_quarterly, compounding=4)
continuous = continuous_rate_from_effective(effective)

pd.Series(
    {
        "nominal_rate_convertible_quarterly": nominal_quarterly,
        "effective_annual_rate": effective,
        "continuous_annual_rate": continuous,
        "nominal_roundtrip": nominal_rate_from_effective(effective, compounding=4),
        "effective_roundtrip": effective_rate_from_continuous(continuous),
    }
).to_frame("rate")

# %% [markdown] tags=["legacy"]
# ## Valuing a cash-flow stream

# %% tags=["legacy"]
cash_flows = pd.DataFrame(
    {
        "year": [1, 2, 3, 4, 5],
        "cash_flow": [120, 120, 120, 120, 1120],
    }
)

cash_flows["discount_factor_annual"] = [
    discount_factor(rate, year, compounding=1) for year in cash_flows["year"]
]
cash_flows["present_value"] = cash_flows["cash_flow"] * cash_flows["discount_factor_annual"]
cash_flows

# %% tags=["legacy"]
stream_value = present_value(
    cash_flows["cash_flow"].to_numpy(),
    cash_flows["year"].to_numpy(),
    rate,
    compounding=1,
)

pd.Series(
    {
        "table_present_value_mxn": cash_flows["present_value"].sum(),
        "function_present_value_mxn": stream_value,
        "roundtrip_error_mxn": cash_flows["present_value"].sum() - stream_value,
    }
)

# %% [markdown] tags=["exercise", "legacy"]
# ## Checkpoint exercise
#
# A client can invest MXN 25,000 for 18 months at an 11% quoted annual rate.
#
# 1. Compute future value under simple interest, monthly compounding, and
#    continuous compounding.
# 2. Convert the 11% nominal rate convertible monthly into effective and
#    continuous annual equivalents, then verify both roundtrips numerically.
# 3. Discount each future value with the same convention and report the maximum
#    recovery error in MXN.
# 4. Explain why comparing quotations without their compounding basis can mislead
#    the client.
#

# %% [markdown] tags=["solution", "legacy"]
# ```{dropdown} Suggested answer rubric
# A complete answer reports all three future values in MXN, labels each rate
# convention, recovers MXN 25,000 to numerical tolerance, and distinguishes a
# quotation conversion from a change in the underlying economic return.
# ```
