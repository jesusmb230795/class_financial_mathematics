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
# # European Option Pricing and Greeks
#
# Module: Derivatives and Risk Management
#
# ## Lesson summary
#
# A European option gives its holder a right, but not an obligation, exercisable
# only at maturity. The Black-Scholes-Merton framework values this nonlinear
# payoff by replication under idealized continuous-time assumptions and provides
# local sensitivities known as Greeks {cite}`blackScholes1973,merton1973,hull2022options`.
# This lesson keeps payoff, time-zero price, sensitivity, and hedged P&L as four
# distinct objects.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - distinguish a terminal payoff from a time-zero option price;
# - price dividend-paying European calls and puts under Black-Scholes-Merton;
# - verify put-call parity with continuous dividend yield;
# - interpret Delta, Gamma, Vega, Theta, and Rho with explicit units;
# - construct a bull call spread and explain its bounded terminal payoff;
# - describe why a delta hedge is local rather than risk free.
#
# ## Prerequisites
#
# Complete Linear Derivatives and Carry first. Students should understand
# discounted present value, continuous compounding, no-arbitrage replication,
# standard-normal notation, and the difference between a terminal cash flow and
# its present value.
#
# ## Payoffs are not prices
#
# Let $S_T$ be the underlying price at maturity and $K$ the exercise price. The
# terminal payoffs of one European call and one European put are
#
# $$
# \Pi_C(S_T;K)=(S_T-K)^+=\max(S_T-K,0),
# $$
#
# $$
# \Pi_P(S_T;K)=(K-S_T)^+=\max(K-S_T,0).
# $$
#
# A payoff has no discounting and contains no premium. Its time-zero price
# reflects the probability distribution, time value, financing, income, and
# replication assumptions.
#
# ## Black-Scholes-Merton with continuous dividend yield
#
# Let $q$ denote the annual continuously compounded dividend yield, not a
# probability. For $T>0$ and $\sigma>0$, the European prices are
# {cite}`blackScholes1973,merton1973`:
#
# $$
# C_0=S_0e^{-qT}\Phi(d_1)-Ke^{-rT}\Phi(d_2),
# $$
#
# $$
# P_0=Ke^{-rT}\Phi(-d_2)-S_0e^{-qT}\Phi(-d_1),
# $$
#
# where $\Phi$ and $\phi$ are the standard-normal cumulative distribution and
# density functions, respectively, and
#
# $$
# d_1=\frac{\ln(S_0/K)+(r-q+\tfrac12\sigma^2)T}{\sigma\sqrt{T}},
# \qquad
# d_2=d_1-\sigma\sqrt{T}.
# $$
#
# ## Python setup

# %% tags=["setup", "hide-input"]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from src.derivatives import (
    black_scholes_greeks,
    black_scholes_price,
    option_payoff,
    put_call_parity_gap,
)
from src.module7_visuals import build_option_payoff_figure

# %% [markdown]
# ## Synthetic contract and model price
#
# All inputs below are deterministic classroom assumptions. Prices are in
# currency units per share; $r$, $q$, and $\sigma$ are annual decimal inputs;
# $r$ and $q$ use continuous compounding; and $T$ is measured in years.

# %%
spot = 100.0
strike = 105.0
rate = 0.06
dividend_yield = 0.015
volatility = 0.25
maturity = 1.0

call_price = black_scholes_price(
    spot,
    strike,
    rate,
    volatility,
    maturity,
    option_type="call",
    dividend_yield=dividend_yield,
)
put_price = black_scholes_price(
    spot,
    strike,
    rate,
    volatility,
    maturity,
    option_type="put",
    dividend_yield=dividend_yield,
)

pd.Series(
    {
        "call_price_currency_per_share": call_price,
        "put_price_currency_per_share": put_price,
        "spot_currency_per_share": spot,
        "strike_currency_per_share": strike,
        "rate_continuous_pct": 100 * rate,
        "dividend_yield_continuous_pct": 100 * dividend_yield,
        "volatility_annual_pct": 100 * volatility,
        "maturity_years": maturity,
    },
    name="synthetic_european_options",
)

# %% [markdown]
# **Interpretation.** The prices are model outputs under one constant-volatility
# assumption. They are neither observed premiums nor expected terminal payoffs.
#
# ## Put-call parity
#
# Dividend-adjusted European put-call parity is
#
# $$
# C_0+Ke^{-rT}=P_0+S_0e^{-qT}.
# $$
#
# A nonzero gap from synchronized executable inputs would indicate a formula,
# unit, or convention error before it indicated a trading opportunity
# {cite}`hull2022options`.

# %%
parity_gap = put_call_parity_gap(
    call_price,
    put_price,
    spot,
    strike,
    rate,
    maturity,
    dividend_yield,
)
pd.Series(
    {
        "call_plus_discounted_strike": call_price + strike * np.exp(-rate * maturity),
        "put_plus_discounted_spot": put_price + spot * np.exp(-dividend_yield * maturity),
        "parity_gap_currency_per_share": parity_gap,
    }
)

# %%
assert abs(parity_gap) < 1e-10

# %% [markdown]
# ## Greeks: definitions and units
#
# Greeks are local partial derivatives. For the Black-Scholes-Merton call and
# put, key closed forms include
#
# $$
# \Delta_C=e^{-qT}\Phi(d_1),
# \qquad
# \Delta_P=e^{-qT}[\Phi(d_1)-1],
# $$
#
# $$
# \Gamma_C=\Gamma_P
# =\frac{e^{-qT}\phi(d_1)}{S_0\sigma\sqrt{T}},
# \qquad
# \nu_C=\nu_P=S_0e^{-qT}\phi(d_1)\sqrt{T},
# $$
#
# $$
# \rho_C=KTe^{-rT}\Phi(d_2),
# \qquad
# \rho_P=-KTe^{-rT}\Phi(-d_2).
# $$
#
# Here $\Delta=\partial V/\partial S$ is the option-value change per one-unit
# spot move (equivalently, underlying units per option under consistent contract
# multipliers), and $\Gamma=\partial^2V/\partial S^2$ is Delta change per one-unit
# spot move. Vega $\nu=\partial V/\partial\sigma$ is value per unit change in
# decimal volatility, $\Theta=\partial V/\partial t$ is calendar-time decay under
# the helper's sign convention, and $\rho=\partial V/\partial r$ is value per unit
# change in the decimal continuously compounded rate. The shared implementation
# reports Vega and Rho per 1 percentage-point change and Theta per calendar day.

# %%
call_greeks = black_scholes_greeks(
    spot,
    strike,
    rate,
    volatility,
    maturity,
    option_type="call",
    dividend_yield=dividend_yield,
)
put_greeks = black_scholes_greeks(
    spot,
    strike,
    rate,
    volatility,
    maturity,
    option_type="put",
    dividend_yield=dividend_yield,
)

greek_table = pd.concat(
    [call_greeks.rename("call"), put_greeks.rename("put")],
    axis=1,
)
greek_table.rename_axis("reported_sensitivity")

# %% [markdown]
# **Reading the table.** A call Rho of 0.40, for example, means approximately
# 0.40 currency units per share for a +1 percentage-point parallel change in
# the continuously compounded rate, holding all other inputs fixed. It does not
# mean a 40% return.
#
# ## From price to a local delta hedge
#
# A dealer short one call has option Delta $-\Delta_C$. Buying $\Delta_C$ shares
# per short call offsets the first-order spot term for a sufficiently small move:
#
# $$
# \Delta V\approx \Delta\,\Delta S
# +\frac12\Gamma(\Delta S)^2+\nu\,\Delta\sigma
# +\Theta\,\Delta t+\rho\,\Delta r.
# $$
#
# Only the first term is neutralized by a spot-only delta hedge.

# %%
spot_scenarios = np.array([spot - 1.0, spot, spot + 1.0])
delta_hedge_shares = call_greeks["delta"]
hedge_pnl = pd.DataFrame({"new_spot_currency_per_share": spot_scenarios})
hedge_pnl["short_call_pnl_currency_per_share"] = [
    call_price
    - black_scholes_price(
        new_spot,
        strike,
        rate,
        volatility,
        maturity,
        option_type="call",
        dividend_yield=dividend_yield,
    )
    for new_spot in spot_scenarios
]
hedge_pnl["stock_hedge_pnl_currency_per_share"] = delta_hedge_shares * (
    hedge_pnl["new_spot_currency_per_share"] - spot
)
hedge_pnl["delta_hedged_pnl_currency_per_share"] = (
    hedge_pnl["short_call_pnl_currency_per_share"] + hedge_pnl["stock_hedge_pnl_currency_per_share"]
)
hedge_pnl.round(6)

# %% [markdown]
# ## A bounded strategy: bull call spread
#
# Buying a call at $K_L$ and selling one call at $K_H>K_L$ gives terminal payoff
#
# $$
# \Pi_{\mathrm{spread}}(S_T)
# =(S_T-K_L)^+-(S_T-K_H)^+.
# $$
#
# Its gross payoff is bounded between 0 and $K_H-K_L$. Net terminal P&L also
# subtracts the financed initial debit, so payoff and profit must not be used as
# synonyms.

# %%
lower_strike = 95.0
upper_strike = 110.0
terminal_prices = np.linspace(50.0, 160.0, 221)

lower_call_payoff = option_payoff(terminal_prices, lower_strike, "call")
upper_call_payoff = option_payoff(terminal_prices, upper_strike, "call")
bull_call_spread_payoff = lower_call_payoff - upper_call_payoff

spread_debit = black_scholes_price(
    spot,
    lower_strike,
    rate,
    volatility,
    maturity,
    "call",
    dividend_yield,
) - black_scholes_price(
    spot,
    upper_strike,
    rate,
    volatility,
    maturity,
    "call",
    dividend_yield,
)

pd.Series(
    {
        "initial_debit_currency_per_share": spread_debit,
        "maximum_gross_payoff_currency_per_share": bull_call_spread_payoff.max(),
        "maximum_terminal_pnl_before_costs": bull_call_spread_payoff.max()
        - spread_debit * np.exp(rate * maturity),
        "minimum_terminal_pnl_before_costs": -spread_debit * np.exp(rate * maturity),
    },
    name="synthetic_bull_call_spread",
)

# %%
assert np.isclose(bull_call_spread_payoff.min(), 0.0)
assert np.isclose(bull_call_spread_payoff.max(), upper_strike - lower_strike)

# %% [markdown]
# ## Payoff diagram
#
# The shared figure deliberately shows call and put *terminal payoffs* at the
# base strike. It does not overlay time-zero option values or claim that payoff
# equals P&L.

# %% mystnb={"image": {"alt": "A deterministic payoff chart shows European call and put terminal payoffs against the terminal underlying price, with the common strike marked at 105 currency units per share."}}
payoff_figure = build_option_payoff_figure(terminal_prices, strike)
display(payoff_figure)
plt.close(payoff_figure)

# %% [markdown]
# ## Source and model notes
#
# - The pricing equations originate in Black and Scholes and Merton's extension
#   of continuous-time option valuation {cite}`blackScholes1973,merton1973`.
# - Contract interpretation, parity, Greeks, and hedging conventions follow the
#   standard derivatives treatment in {cite:t}`hull2022options`.
# - All contract inputs and scenarios in this notebook are synthetic. No market
#   chain or historical observation is loaded.
# - The code uses shared, tested functions from `src.derivatives`; the notebook
#   does not maintain a second pricing implementation.
#
# ## Model limitations
#
# - Black-Scholes-Merton assumes frictionless continuous trading, lognormal
#   diffusion, and constant parameters; real markets exhibit jumps, discrete
#   trading, liquidity effects, and volatility surfaces.
# - Greeks are local derivatives. Large shocks require repricing and scenario
#   analysis rather than a first-order interpretation alone.
# - The delta-hedge table holds maturity and all other inputs fixed; it is a
#   controlled sensitivity illustration, not a self-financing backtest.
# - The bull spread ignores transaction costs, bid-ask spreads, taxes, funding
#   basis, early exercise, and assignment mechanics.
#
# ## Handoff
#
# Model price establishes present value; Greeks translate that value into local
# exposures; hedge and strategy P&L reveal residual nonlinear risk. The next
# lesson checks the same valuation under binomial and Monte Carlo numerical
# methods, including discretization and sampling uncertainty.
