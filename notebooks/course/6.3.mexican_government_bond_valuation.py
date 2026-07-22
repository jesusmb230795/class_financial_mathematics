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
# mechanics, clean price, dirty price, accrued interest, and inflation indexation.
# The formulas follow Banco de México's government-security descriptions and
# technical appendices {cite}`banxicoGovSecurities,banxicoGovSecuritiesTechnical`.
#
# The examples are deterministic and classroom-safe. They are not observations
# or executable settlement instructions. Live Banco de México data can be
# connected later through `src.banxico` and `src.market_data`, but this notebook
# does not require a token or network access.
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
# For a CETES with face value $VN$ in MXN, annualized simple return yield $r$ as
# a decimal, and $t$ actual days to maturity on an ACT/360 basis:
#
# $$
# P = \frac{VN}{1 + r\frac{t}{360}}.
# $$
#
# This is the return-yield convention in the Banco de México technical
# appendix; it is not periodic-compounding bond YTM
# {cite}`banxicoGovSecuritiesTechnical`.

# %%
cete_28 = Cetes(days_to_maturity=28, annual_yield=0.0950)

pd.Series(
    {
        "face_value_mxn": cete_28.face_value,
        "days_to_maturity": cete_28.days_to_maturity,
        "annual_return_yield_decimal": cete_28.annual_yield,
        "price_mxn": cete_28.price,
        "discount_rate_quote_decimal": cete_28.discount_rate,
    }
).to_frame("CETES 28-day")

# %% [markdown]
# The annualized simple discount-rate quote $b$ and return-yield quote $r$ are
# related by:
#
# $$
# b = \frac{r}{1 + r\frac{t}{360}},
# \qquad
# r = \frac{b}{1 - b\frac{t}{360}}.
# $$
#
# Both $b$ and $r$ are decimals on the same ACT/360 basis. The conversion must
# retain the same face value and days to maturity
# {cite}`banxicoGovSecuritiesTechnical`.

# %%
recovered_yield = cetes_yield_from_discount_rate(
    cete_28.discount_rate,
    cete_28.days_to_maturity,
)

pd.Series(
    {
        "original_yield_decimal": cete_28.annual_yield,
        "discount_rate_decimal": cete_28.discount_rate,
        "recovered_yield_decimal": recovered_yield,
        "roundtrip_error": recovered_yield - cete_28.annual_yield,
    }
)

# %% [markdown]
# ## Bono M clean and dirty pricing
#
# For the simplified Bono M implementation used here, $VN$ is face value in MXN,
# $RC$ is the annual coupon rate as a decimal, and each coupon period is treated
# as 182 days on ACT/360:
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
#
# The exponent discounts the first coupon over the remaining fraction of its
# 182-day period. Banco de México's appendix documents the instrument-specific
# coupon and yield convention {cite}`banxicoGovSecuritiesTechnical`.

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
        "coupon_payment_mxn": bono.coupon_payment,
        "dirty_price_mxn": bono.dirty_price,
        "accrued_interest_mxn": bono.accrued_interest,
        "clean_price_mxn": bono.clean_price,
    }
).to_frame("Bono M")

# %% [markdown]
# The clean price excludes accrued interest. The dirty price is the cash
# settlement amount before transaction costs:
#
# $$
# P_{clean} = P_{dirty} - AI.
# $$
#
# The accrued-interest adjustment prevents a buyer from receiving a full coupon without compensating the seller for the coupon period already earned.
#
# ## UDIBONO settlement
#
# UDIBONOS are valued in UDIS first. The MXN settlement amount uses the UDI value
# for the applicable settlement date:
#
# $$
# \text{Settlement MXN} = (P_{clean,UDIS} + AI_{UDIS}) \times UDI_t.
# $$
#
# The worked input below sets $UDI_t=8.45$ MXN per UDI as an explicitly
# hypothetical classroom assumption. It is not a Banco de México observation,
# a current quote, or a value suitable for settlement. A live application must
# retrieve and validate the date-specific official value
# {cite}`banxicoGovSecurities`.

# %%
udibono_real = BonoM(
    coupon_rate=0.045,
    annual_yield=0.052,
    remaining_coupons=12,
    days_since_last_coupon=120,
    face_value=100.0,
)

hypothetical_udi_mxn_per_udi = 8.45
settlement = udibono_settlement_mxn(
    clean_price_udis=udibono_real.clean_price,
    accrued_interest_udis=udibono_real.accrued_interest,
    udi_value=hypothetical_udi_mxn_per_udi,
)

pd.Series(
    {
        "clean_price_udis": udibono_real.clean_price,
        "accrued_interest_udis": udibono_real.accrued_interest,
        "dirty_price_udis": udibono_real.dirty_price,
        "hypothetical_udi_mxn_per_udi": hypothetical_udi_mxn_per_udi,
        "hypothetical_settlement_mxn": settlement,
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
# - The examples simplify Mexican market conventions and should not be treated
#   as production settlement logic.
# - The rates, price inputs, and UDI value are hypothetical classroom inputs,
#   not observed market data.
# - Live valuation would require validated curves, calendars, date-specific UDI
#   values, tax treatment, and instrument identifiers.
# - Small convention differences can create material price and accrued-interest
#   differences; rounding and settlement rules must also be validated.

# %% [markdown]
# ## Handoff
#
# The next lab solves yield from price and expresses local exposure through DV01,
# duration, convexity, and immunization conditions. Use the instrument convention
# identified here before interpreting any risk number; a correct formula under
# the wrong settlement or compounding convention is not a valid hedge input.
