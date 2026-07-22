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
# # Mexican Government Bond Valuation
#
# Module: Fixed Income, Credit, and Term Structure
#
# ## Lesson summary
#
# Mexican sovereign instruments use market conventions that are easy to miss in
# generic bond examples. This lesson connects the time value of money to CETES,
# Bonos M, and UDIBONOS, emphasizing ACT/360 conventions, semiannual coupon
# mechanics, clean price, dirty price, accrued interest, and inflation indexation
# {cite}`banxicoGovSecurities,fabozzi2019foundations`.
#
# The examples are deterministic and classroom-safe. Live Banxico data can be connected later through `src.banxico` and `src.market_data`, but this notebook does not require a token or network access.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - price a CETES instrument from an annualized ACT/360 yield;
# - convert between CETES return yield and discount-rate quotation;
# - build the cash-flow table for a simplified Bono M;
# - distinguish clean price, dirty price, and accrued interest;
# - convert UDIBONO prices in UDIS into MXN settlement amounts;
# - explain why local day-count and quotation conventions matter.
#
# ## Prerequisites
#
# Complete the general bond-pricing and sensitivity lessons first. Readers
# should understand present value, coupon schedules, accrued interest, yield
# compounding, and the inverse price-yield relationship. This lesson then adds
# instrument-specific Mexican quotation, day-count, inflation-indexation, and
# settlement conventions; those conventions must not be mixed across examples.
#
# ## Python setup

# %% tags=["setup", "hide-input"]
import pandas as pd

from src.fixed_income import (
    BonoM,
    Cetes,
    bono_m_price_table,
    cetes_yield_from_discount_rate,
    udibono_settlement_mxn,
)

# %% [markdown]
# ## Instrument map
#
# | Instrument | Cash-flow structure | Common role in the curve | Quotation convention in this lesson |
# | --- | --- | --- | --- |
# | CETES | Zero-coupon discount security | Short end of the nominal curve | ACT/360 money-market yield |
# | Bonos M | Fixed nominal semiannual coupons | Medium and long nominal curve | Clean and dirty price with accrued interest |
# | UDIBONOS | Real fixed semiannual coupons in UDIS | Real curve and inflation-linked valuation | Clean and dirty price in UDIS, settlement in MXN |
#
# ## CETES pricing
#
# For a CETES with face value $VN$, annualized return yield $r$, days to maturity $t$, and ACT/360 basis:
#
# $$
# P = \frac{VN}{1 + r\frac{t}{360}}.
# $$

# %%
cete_28 = Cetes(days_to_maturity=28, annual_yield=0.0950)

pd.Series(
    {
        "face_value": cete_28.face_value,
        "days_to_maturity": cete_28.days_to_maturity,
        "annual_yield": cete_28.annual_yield,
        "price": cete_28.price,
        "discount_rate_quote": cete_28.discount_rate,
    }
).to_frame("CETES 28-day")

# %% [markdown]
# The discount-rate quote $b$ and return-yield quote $r$ are related by:
#
# $$
# b = \frac{r}{1 + r\frac{t}{360}},
# \qquad
# r = \frac{b}{1 - b\frac{t}{360}}.
# $$

# %%
recovered_yield = cetes_yield_from_discount_rate(
    cete_28.discount_rate,
    cete_28.days_to_maturity,
)

pd.Series(
    {
        "original_yield": cete_28.annual_yield,
        "discount_rate": cete_28.discount_rate,
        "recovered_yield": recovered_yield,
        "roundtrip_error": recovered_yield - cete_28.annual_yield,
    }
)

# %% [markdown]
# ## Bono M clean and dirty pricing
#
# For a simplified Bono M, each coupon period is treated as 182 days on ACT/360:
#
# $$
# C = VN \times RC \times \frac{182}{360}.
# $$
#
# If $d$ days have elapsed since the previous coupon, the fractional exponent for cash-flow period $j$ is:
#
# $$
# j - 1 + \frac{182-d}{182}.
# $$

# %%
bono = BonoM(
    coupon_rate=0.075,
    annual_yield=0.088,
    remaining_coupons=10,
    days_since_last_coupon=73,
)

price_table = bono_m_price_table(bono)
price_table.head()

# %%
pd.Series(
    {
        "coupon_payment": bono.coupon_payment,
        "dirty_price": bono.dirty_price,
        "accrued_interest": bono.accrued_interest,
        "clean_price": bono.clean_price,
    }
).to_frame("Bono M")

# %% [markdown]
# The clean price is the quoted economic price. The dirty price is the cash settlement before transaction costs:
#
# $$
# P_{clean} = P_{dirty} - AI.
# $$
#
# The accrued-interest adjustment prevents a buyer from receiving a full coupon without compensating the seller for the coupon period already earned.
#
# ## UDIBONO settlement
#
# UDIBONOS are valued in UDIS first. The MXN settlement amount uses the daily UDI value:
#
# $$
# \text{Settlement MXN} = (P_{clean,UDIS} + AI_{UDIS}) \times UDI_t.
# $$

# %%
udibono_real = BonoM(
    coupon_rate=0.045,
    annual_yield=0.052,
    remaining_coupons=12,
    days_since_last_coupon=120,
    face_value=100.0,
)

udi_value = 8.45
settlement = udibono_settlement_mxn(
    clean_price_udis=udibono_real.clean_price,
    accrued_interest_udis=udibono_real.accrued_interest,
    udi_value=udi_value,
)

pd.Series(
    {
        "clean_price_udis": udibono_real.clean_price,
        "accrued_interest_udis": udibono_real.accrued_interest,
        "dirty_price_udis": udibono_real.dirty_price,
        "udi_value_mxn": udi_value,
        "settlement_mxn": settlement,
    }
)

# %% [markdown]
# ## Convention risk
#
# Small convention errors can become material when the portfolio is large. In
# this simplified `BonoM` class, changing `day_count` changes the entire quotation
# convention consistently: the coupon cash amount, yield per 182-day period, and
# accrued interest. It is therefore incorrect to interpret the comparison as an
# accrued-interest-only change.

# %%
comparison = []
for day_count in [360, 365]:
    bond = BonoM(
        coupon_rate=0.075,
        annual_yield=0.088,
        remaining_coupons=10,
        days_since_last_coupon=73,
        day_count=day_count,
    )
    comparison.append(
        {
            "day_count": day_count,
            "coupon_payment": bond.coupon_payment,
            "yield_per_182_day_period": (bond.annual_yield * bond.coupon_days / bond.day_count),
            "dirty_price": bond.dirty_price,
            "accrued_interest": bond.accrued_interest,
            "clean_price": bond.clean_price,
        }
    )

pd.DataFrame(comparison)

# %% [markdown]
# ## Model limitations
#
# - The examples simplify Mexican market conventions and should not be treated as production settlement logic.
# - Live valuation would require validated curves, calendars, UDI values, tax treatment, and instrument-specific details.
# - Small convention differences can create material price and accrued-interest differences.

# %% [markdown]
# ## Handoff
#
# The next lab solves yield from price and expresses local exposure through DV01,
# duration, convexity, and immunization conditions. Use the instrument convention
# identified here before interpreting any risk number; a correct formula under
# the wrong settlement or compounding convention is not a valid hedge input.
