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
# # Linear Derivatives and Carry
#
# Module: Derivatives and Risk Management
#
# ## Lesson summary
#
# Forwards, futures, and swaps are linear derivatives: before credit and
# funding adjustments, their contractual cash flows are affine in a stated
# underlying price, rate, or index. No-arbitrage links their prices to spot,
# financing, income, settlement, and discounting conventions
# {cite}`hull2022options,cmeFuturesIntro`.
#
# Every number below is a deterministic classroom input. None is an observed
# market quote, a trade recommendation, or evidence about current Mexican or
# international markets.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - compute an equity or FX forward price under continuously compounded carry;
# - mark an existing forward contract to market with units and position sign intact;
# - explain how daily futures variation margin differs from forward settlement;
# - derive a par fixed swap rate from discount factors and accrual fractions;
# - identify the assumptions that must accompany any carry calculation.
#
# ## Prerequisites
#
# Complete the quantitative-foundations lesson and the fixed-income curve
# sequence first. Students should be able to discount a dated cash flow, convert
# between decimal-rate and percentage-point units, distinguish simple from
# continuous compounding, and read an FX quote without reversing its base and
# quote currencies.
#
# ## Conventions and notation
#
# Throughout the lesson, $r$, $q$, $r_d$, and $r_f$ are annual continuously
# compounded rates stored as decimals; $T$ and $\tau=T-t$ are year fractions.
# The symbol $q$ is an income yield here, not a probability. Prices are quoted in
# currency units per underlying unit unless a different unit is stated.
#
# For an equity with spot price $S_0$ and continuous dividend yield $q$,
# cash-and-carry replication gives {cite}`hull2022options`:
#
# $$
# F(0,T)=S_0\exp\!\left[(r-q)T\right].
# $$
#
# More generally, storage costs can be added to carry and convenience benefits
# subtracted, but they are outside this equity example.
#
# For an FX quote $X_0$ in domestic currency per unit of foreign currency,
# covered interest parity implies
#
# $$
# F_X(0,T)=X_0\exp\!\left[(r_d-r_f)T\right].
# $$
#
# The quote direction matters: reversing $X_0$ also reverses which rate is
# domestic.
#
# ## Python setup

# %% tags=["setup", "hide-input"]
import numpy as np
import pandas as pd

from src.derivatives import forward_price, forward_value, par_swap_rate

# %% [markdown]
# ## Equity forward price under continuous carry
#
# The synthetic contract is for one share priced at 100 currency units. The
# annual continuously compounded financing rate is 9.5%, the continuous
# dividend yield is 2.5%, and maturity is six months.

# %%
spot = 100.0
rate = 0.095
maturity = 0.5
dividend_yield = 0.025

forward = forward_price(spot, rate, maturity, dividend_yield)
pd.Series(
    {
        "spot_currency_per_share": spot,
        "financing_rate_continuous_pct": 100 * rate,
        "dividend_yield_continuous_pct": 100 * dividend_yield,
        "maturity_years": maturity,
        "forward_currency_per_share": forward,
    },
    name="synthetic_equity_forward",
)

# %% [markdown]
# **Interpretation.** Positive net carry, $r-q=7\%$, places the synthetic
# forward above spot. This is a replication result under the stated assumptions,
# not a forecast of the future spot price.
#
# ## FX carry and quote direction
#
# Treat the following 18.00 MXN per USD spot quote and both interest rates as
# hypothetical. With MXN as the domestic currency and USD as the foreign
# currency, the helper's `rate` and `dividend_yield` arguments represent $r_d$
# and $r_f$, respectively.

# %%
fx_spot_mxn_per_usd = 18.00
mxn_rate_continuous = 0.085
usd_rate_continuous = 0.045
fx_maturity_years = 0.25

fx_forward_mxn_per_usd = forward_price(
    fx_spot_mxn_per_usd,
    mxn_rate_continuous,
    fx_maturity_years,
    usd_rate_continuous,
)
pd.Series(
    {
        "spot_mxn_per_usd": fx_spot_mxn_per_usd,
        "mxn_rate_continuous_pct": 100 * mxn_rate_continuous,
        "usd_rate_continuous_pct": 100 * usd_rate_continuous,
        "maturity_years": fx_maturity_years,
        "forward_mxn_per_usd": fx_forward_mxn_per_usd,
    },
    name="synthetic_fx_forward",
)

# %% [markdown]
# **Interpretation.** Because the hypothetical MXN rate exceeds the USD rate,
# the forward MXN-per-USD quote exceeds spot. The calculation says nothing about
# expected depreciation; it enforces an internally consistent carry convention.
#
# ## Mark-to-market value and position sign
#
# At time $t$, let $\tau=T-t$ be the remaining maturity and $K$ the contractual
# delivery price. The value per underlying unit to the long side is
#
# $$
# V_t^{\mathrm{long}}
# =S_t e^{-q\tau}-K e^{-r\tau},
# \qquad
# V_t^{\mathrm{short}}=-V_t^{\mathrm{long}}.
# $$
#
# The strike is a contractual price, not an option strike with a nonlinear
# payoff. The table keeps values in currency units per share.

# %%
strikes = np.array([95.0, 100.0, forward, 110.0])
forward_values = pd.DataFrame(
    {
        "delivery_price_currency_per_share": strikes,
        "long_value_currency_per_share": [
            forward_value(spot, strike, rate, maturity, dividend_yield) for strike in strikes
        ],
        "short_value_currency_per_share": [
            forward_value(
                spot,
                strike,
                rate,
                maturity,
                dividend_yield,
                position="short",
            )
            for strike in strikes
        ],
    }
)
forward_values.round(6)

# %%
np.testing.assert_allclose(
    forward_values["long_value_currency_per_share"],
    -forward_values["short_value_currency_per_share"],
)
fair_contract_value = forward_value(
    spot,
    forward,
    rate,
    maturity,
    dividend_yield,
)
assert abs(fair_contract_value) < 1e-10

# %% [markdown]
# ## Local forward delta
#
# Holding $r$, $q$, and $\tau$ fixed,
#
# $$
# \Delta_F=\frac{\partial V_t^{\mathrm{long}}}{\partial S_t}=e^{-q\tau}.
# $$
#
# A dealer long one forward therefore shorts $e^{-q\tau}$ spot units for a
# first-order local hedge. This does not remove funding, dividend, basis,
# counterparty, liquidity, or jump risk.

# %%
forward_delta = np.exp(-dividend_yield * maturity)
pd.Series(
    {
        "long_forward_delta_share_per_contract_unit": forward_delta,
        "spot_units_for_local_delta_hedge": -forward_delta,
        "remaining_maturity_years": maturity,
    }
)

# %% [markdown]
# ## Futures: daily variation margin
#
# A forward normally accumulates value until settlement or close-out. A futures
# position is marked to a daily settlement price, so gains and losses move
# through the margin account each day. For a long position with contract
# multiplier $M$, quantity $Q$, and settlement price $f_t$, daily variation
# margin is {cite}`cmeFuturesIntro`:
#
# $$
# \Delta VM_t=QM(f_t-f_{t-1}).
# $$
#
# The daily reset changes cash-flow timing. If rates and futures prices are
# stochastic and correlated, reinvestment of margin cash flows can make a
# futures price differ from the otherwise comparable forward price
# {cite}`hull2022options`.

# %%
synthetic_settlements = pd.Series(
    [100.00, 101.20, 100.70, 102.10, 101.60],
    index=pd.Index(["Day 0", "Day 1", "Day 2", "Day 3", "Day 4"], name="settlement_day"),
    name="settlement_currency_per_underlying_unit",
)
contract_quantity = 2
contract_multiplier = 100
variation_margin = contract_quantity * contract_multiplier * synthetic_settlements.diff()
margin_ledger = pd.concat(
    [synthetic_settlements, variation_margin.rename("variation_margin_currency")], axis=1
)
margin_ledger["cumulative_margin_currency"] = (
    margin_ledger["variation_margin_currency"].fillna(0.0).cumsum()
)
margin_ledger

# %%
futures_total_change = (
    contract_quantity
    * contract_multiplier
    * (synthetic_settlements.iloc[-1] - synthetic_settlements.iloc[0])
)
assert np.isclose(
    margin_ledger["variation_margin_currency"].sum(),
    futures_total_change,
)

# %% [markdown]
# **Interpretation.** The cumulative variation margin equals the change from the
# first to the last synthetic settlement price times quantity and multiplier.
# The ledger deliberately ignores initial margin, interest on collateral, fees,
# default waterfalls, and exchange-specific settlement rules.
#
# ## Par fixed swap rate
#
# Consider a spot-starting, single-curve, fixed-for-floating interest-rate swap
# with notional $N$, discount factors $D(0,T_i)$, and fixed-leg accrual fractions
# $\tau_i$. Under this simplified setup, the floating-leg present value is
# $N[1-D(0,T_n)]$ and the fixed-leg present value is
# $NR_{\mathrm{swap}}\sum_i\tau_iD(0,T_i)$. Equating them gives
# {cite}`hull2022options`:
#
# $$
# R_{\mathrm{swap}}
# =\frac{1-D(0,T_n)}{\sum_{i=1}^{n}\tau_iD(0,T_i)}.
# $$
#
# Modern collateralized swaps generally require distinct discount and
# projection curves. The one-curve example isolates annuity mechanics rather
# than representing a production valuation.

# %%
payment_dates_years = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
zero_rates_continuous = np.array([0.087, 0.085, 0.083, 0.082, 0.081, 0.080])
discount_factors = np.exp(-zero_rates_continuous * payment_dates_years)
accrual_fractions_years = np.full_like(payment_dates_years, 0.5)

swap_table = pd.DataFrame(
    {
        "payment_time_years": payment_dates_years,
        "zero_rate_continuous_pct": 100 * zero_rates_continuous,
        "discount_factor": discount_factors,
        "fixed_leg_accrual_years": accrual_fractions_years,
    }
)
swap_table.round(6)

# %%
par_rate = par_swap_rate(discount_factors, accrual_fractions_years)
pd.Series(
    {
        "par_swap_rate_decimal": par_rate,
        "par_swap_rate_pct": 100 * par_rate,
        "fixed_leg_annuity_years": np.sum(accrual_fractions_years * discount_factors),
    },
    name="synthetic_single_curve_swap",
)

# %% [markdown]
# ## Interpretation checklist
#
# | Contract | Required unit and convention check | Material omitted risk |
# | --- | --- | --- |
# | Equity forward | currency per share; continuous $r$ and $q$ | dividends, funding basis, credit, liquidity |
# | FX forward | domestic per foreign currency; $r_d-r_f$ | cross-currency basis, collateral currency, settlement |
# | Future | quote unit, quantity, multiplier, daily settlement | margin liquidity, collateral remuneration, convexity |
# | Swap | curve date, discount factors, accrual day count | multi-curve projection, collateral, counterparty adjustment |
#
# ## Source and model notes
#
# - Contract mechanics, carry identities, and swap valuation follow standard
#   no-arbitrage treatments {cite}`hull2022options`.
# - The futures margin example uses generic daily-settlement mechanics described
#   by CME educational material {cite}`cmeFuturesIntro`; actual rulebooks and contract
#   specifications govern a real trade.
# - All spot prices, rates, curves, settlements, maturities, and quantities in
#   this notebook are synthetic deterministic teaching inputs.
# - Continuously compounded rates are stored as decimals in calculations and
#   multiplied by 100 only in explicitly labeled percentage displays.
#
# ## Model limitations
#
# - Carry formulas require documented funding, income, storage, convenience-yield,
#   collateral, tax, and settlement assumptions.
# - Futures and forwards can diverge when margin cash flows and interest rates
#   interact; the deterministic ledger does not estimate that convexity effect.
# - The par swap example is a single-curve abstraction and omits counterparty,
#   funding, collateral, and valuation adjustments.
#
# ## Handoff
#
# A no-arbitrage forward or par swap rate is an inception price, not a risk
# measure. Desks map the priced contract to spot delta, curve exposure, carry,
# basis, counterparty scenarios, and liquidity needs. The next lesson adds
# nonlinear European option prices, Greeks, and option strategies to this
# carry-and-discounting foundation.
