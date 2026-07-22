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
# # Yield Curve Bootstrapping
#
# Module: Fixed Income, Credit, and Term Structure
#
# ## Lesson summary
#
# A yield curve describes how interest rates vary across maturities. Fixed-income valuation becomes more realistic when each cash flow is discounted with a maturity-specific discount factor instead of a single flat yield.
#
# This lesson introduces the relationship between discount factors, spot rates, forward rates, and par rates, then builds a simple bootstrapping workflow {cite}`fabozzi2019foundations`.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - convert between discount factors and spot rates;
# - bootstrap discount factors from simple market instruments;
# - derive implied forward rates;
# - plot and interpret the shape of a yield curve;
# - explain the limitations of curve construction with sparse inputs.
#
# ## Prerequisites
#
# Complete the bond-pricing, market-convention, and credit-spread lessons first.
# Readers should be able to identify every promised cash flow and distinguish a
# quoted yield from a discount factor. All instruments in one bootstrap must
# share an explicit valuation date, currency, day-count basis, compounding
# convention, and complete coupon-date grid.
#
# ## Core relationships
#
# The discount factor for maturity $T$ under annual compounding is:
#
# $$
# D(0,T) = \frac{1}{(1+s_T)^T},
# $$
#
# where $s_T$ is the spot rate for maturity $T$.
#
# The annual spot rate implied by a discount factor is:
#
# $$
# s_T = D(0,T)^{-1/T} - 1.
# $$
#
# The one-period forward rate between $T_1$ and $T_2$ is:
#
# $$
# f(T_1,T_2) =
# \left(\frac{D(0,T_1)}{D(0,T_2)}\right)^{1/(T_2-T_1)} - 1.
# $$
#
# ## Python setup

# %% tags=["setup", "hide-input"]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.term_structure import (
    bootstrap_coupon_bond_discount_factors,
    forward_rates_from_discount_factors,
    spot_rate_from_discount_factor,
)

# %% [markdown]
# ## Synthetic market instruments
#
# All instruments use semiannual coupon dates. The maturity grid therefore
# includes every half-year node through three years. A coupon-bearing instrument
# cannot be bootstrapped if any earlier coupon-date discount factor is missing:
# the implementation raises an error instead of silently omitting that cash flow.

# %%
market = pd.DataFrame(
    {
        "maturity": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "coupon_rate": [0.00, 0.00, 0.045, 0.050, 0.052, 0.055],
        "price": [98.20, 95.90, 100.10, 100.25, 100.32, 100.40],
    }
)
market


# %% [markdown]
# ## Bootstrap discount factors

# %%
discount_factors = bootstrap_coupon_bond_discount_factors(market, frequency=2)
curve = discount_factors.rename_axis("maturity").reset_index()
curve["spot_rate_annual"] = [
    spot_rate_from_discount_factor(
        discount,
        maturity,
        compounding="annual",
    )
    for maturity, discount in zip(curve["maturity"], curve["discount_factor"])
]
curve["discount_roundtrip_error"] = [
    (1 + spot) ** (-maturity) - discount
    for maturity, spot, discount in zip(
        curve["maturity"],
        curve["spot_rate_annual"],
        curve["discount_factor"],
    )
]
curve

# %% [markdown]
# ## Forward rates

# %%
annual_forwards = forward_rates_from_discount_factors(discount_factors)
curve["forward_rate_annual"] = np.nan
curve.loc[1:, "forward_rate_annual"] = annual_forwards.to_numpy()

curve

# %% [markdown]
# ## Visualizing the curve

# %%
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(curve["maturity"], curve["spot_rate_annual"], marker="o", label="Spot rate")
ax.plot(
    curve["maturity"],
    curve["forward_rate_annual"],
    marker="s",
    label="Forward rate",
)
ax.set_xlabel("Maturity")
ax.set_ylabel("Annually compounded rate")
ax.set_title("Bootstrapped Annual Spot and Forward Rates")
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## Model limitations
#
# - Bootstrapped curves inherit quote noise, missing maturities, and instrument convention errors.
# - Interpolation choices can affect forward rates even when spot rates look smooth.
# - A curve built from simplified instruments should not be used as a production discounting curve.

# %% [markdown] tags=["exercise"]
# ## Checkpoint exercise
#
# Remove the 2.5-year instrument and try to bootstrap the 3-year coupon bond.
#
# 1. Record the exact missing cash-flow date reported by the guardrail.
# 2. Restore the instrument, change its price by MXN 0.25 per MXN 100 face, and
#    recompute the 2.5- and 3-year discount factors, annual spots, and forward.
# 3. Verify that every spot-to-discount roundtrip error is below $10^{-12}$.
# 4. Explain why silently dropping the 2.5-year coupon would overstate the
#    3-year discount factor.
#

# %% [markdown] tags=["solution"]
# ```{dropdown} Suggested answer rubric
# A complete answer identifies the missing 2.5-year cash flow, reports the price
# shock in MXN and rate changes in basis points, verifies the declared annual
# convention, and explains the direction of the bootstrap error.
# ```

# %% [markdown]
# ## Handoff
#
# The short-rate and Nelson-Siegel lessons offer two different ways to move
# beyond a bootstrapped snapshot: a stochastic process for rate evolution and a
# parsimonious cross-sectional curve fit. Keep this lesson's discount-factor
# identities as the valuation check for both extensions.
