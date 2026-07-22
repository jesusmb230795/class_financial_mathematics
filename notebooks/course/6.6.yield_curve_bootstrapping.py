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
# This lesson introduces the relationship between discount factors, spot rates,
# forward rates, and par rates, then builds and validates a recursive
# bootstrapping workflow {cite}`fabozzi2019foundations,banxicoGovSecuritiesTechnical`.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - convert between discount factors and spot rates;
# - bootstrap discount factors from simple market instruments;
# - derive implied forward rates;
# - derive par coupon rates from discount factors;
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
# Let $F$ be face value, $q_T$ the annual coupon rate, $m$ the number of
# payments per year, and $n=mT$. Once the earlier coupon-date discount factors
# are known, a coupon bond supplies the next node recursively:
#
# $$
# P_T
# = \frac{Fq_T}{m}\sum_{j=1}^{n-1}D\!\left(0,\frac{j}{m}\right)
# + \left(F+\frac{Fq_T}{m}\right)D(0,T),
# $$
#
# $$
# D(0,T)
# = \frac{P_T-\frac{Fq_T}{m}\sum_{j=1}^{n-1}D(0,j/m)}
# {F+Fq_T/m}.
# $$
#
# If the bond is priced at par, $P_T=F$, the corresponding annual par coupon
# rate, quoted as a nominal annual coupon convertible $m$ times per year, is
#
# $$
# q_T^{\mathrm{par}}
# = m\frac{1-D(0,T)}{\sum_{j=1}^{n}D(0,j/m)}.
# $$
#
# These identities assume clean synthetic prices with no accrued interest,
# taxes, settlement lag, or day-count adjustment. A dealer curve must encode
# those instrument conventions explicitly.
#
# ## Python setup

# %% tags=["setup", "hide-input"]
import numpy as np
import pandas as pd

from src.module6_visuals import build_bootstrap_curve_figure
from src.term_structure import (
    bootstrap_coupon_bond_discount_factors,
    forward_rates_from_discount_factors,
    par_rate_from_discount_factors,
    spot_rate_from_discount_factor,
)

# %% [markdown]
# ## Synthetic market instruments
#
# The author-created inputs below are deterministic and denominated in generic
# currency units per 100 of face value. All instruments use semiannual coupons,
# annual coupon-rate quotes, annual compounding for reported spot and forward
# rates, and exact half-year maturities. They are not market observations and
# therefore have no provider date or market vintage. A coupon-bearing instrument
# cannot be bootstrapped if an earlier coupon-date discount factor is missing:
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
curve["par_rate_annual"] = [
    par_rate_from_discount_factors(
        discount_factors.loc[:maturity],
        frequency=2,
    )
    for maturity in curve["maturity"]
]
curve

# %% [markdown]
# ## Contractual cash-flow roundtrip
#
# A bootstrap is not validated merely because it returns positive discount
# factors. Reprice every input from the extracted nodes and compare the result
# with the supplied clean price. With face value $F=100$ and semiannual coupons,
# the repricing equation is
#
# $$
# \widehat P_T
# = \sum_{j=1}^{n}\frac{Fq_T}{m}D(0,j/m)+F D(0,T).
# $$

# %%
FACE_VALUE = 100.0
PAYMENTS_PER_YEAR = 2


def reprice_from_bootstrapped_nodes(row: pd.Series) -> float:
    """Reprice one classroom instrument on its contractual coupon grid."""
    periods = int(round(float(row["maturity"]) * PAYMENTS_PER_YEAR))
    payment_times = np.arange(1, periods + 1) / PAYMENTS_PER_YEAR
    coupon_cash_flow = FACE_VALUE * float(row["coupon_rate"]) / PAYMENTS_PER_YEAR
    coupon_pv = sum(coupon_cash_flow * discount_factors.loc[t] for t in payment_times)
    principal_pv = FACE_VALUE * discount_factors.loc[float(row["maturity"])]
    return float(coupon_pv + principal_pv)


roundtrip = market.copy()
roundtrip["repriced_price"] = roundtrip.apply(reprice_from_bootstrapped_nodes, axis=1)
roundtrip["price_error"] = roundtrip["repriced_price"] - roundtrip["price"]
roundtrip

# %% [markdown]
# ## Forward rates

# %%
annual_forwards = forward_rates_from_discount_factors(discount_factors)
curve["forward_rate_annual"] = np.nan
curve.loc[1:, "forward_rate_annual"] = annual_forwards.to_numpy()

curve

# %% [markdown]
# The implied forward at each interval is a no-arbitrage transformation of two
# adjacent discount factors, not a forecast of the future short rate. Likewise,
# the par rate is the coupon that prices a hypothetical bond at 100 on the same
# grid; it need not equal the coupon on the input instrument.

# %% [markdown]
# ## Visualizing the curve

# %% mystnb={"image": {"alt": "A line-and-marker chart compares effective annual spot and one-period forward rates with a nominal annual par coupon convertible semiannually across maturities from 0.5 to 3 years for deterministic synthetic bond prices."}}
bootstrap_source_note = (
    "Source: author-created deterministic instruments; 100 currency-unit face; "
    "semiannual coupons; effective annual spot and forward rates; nominal annual "
    "par coupon convertible semiannually; 0.5- to 3-year contractual grid. No "
    "provider, observation date, or market vintage."
)
build_bootstrap_curve_figure(curve, source_note=bootstrap_source_note)

# %% [markdown]
# The main pattern is the ordering and smooth progression of the rates implied
# by this small synthetic price set. Forward rates can move more sharply than
# spot rates because each forward uses a ratio of adjacent discount factors;
# that is the important exception to reading all three lines as equally smooth.
# The par series is a nominal annual coupon convertible semiannually, whereas
# spot and forward series are effective annual rates, so their vertical gaps
# combine curve information with a disclosed quotation-convention difference.
# The limitation is structural: six clean classroom prices on a complete grid do
# not represent bid-ask spreads, instrument selection, interpolation, settlement,
# or collateral conventions in a production curve.

# %% [markdown]
# ## Model limitations
#
# - Bootstrapped curves inherit quote noise, missing maturities, and instrument convention errors.
# - Interpolation choices can affect forward rates even when spot rates look smooth.
# - A curve built from simplified instruments should not be used as a production discounting curve.
# - An exact repricing roundtrip proves internal consistency with these inputs;
#   it does not prove that the input quotes or conventions are economically correct.

# %% [markdown]
# ## Handoff
#
# The short-rate and Nelson-Siegel lessons offer two different ways to move
# beyond a bootstrapped snapshot: a stochastic process for rate evolution and a
# parsimonious cross-sectional curve fit. Keep this lesson's discount-factor
# identities as the valuation check for both extensions.
