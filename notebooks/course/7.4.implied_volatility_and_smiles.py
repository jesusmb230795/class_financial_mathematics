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
# # Implied Volatility, Smiles, and Skews
#
# Module: Derivatives and Risk Management
#
# ## Lesson summary
#
# Black-Scholes-Merton takes volatility as an input. An implied volatility is the
# value of $\sigma$ that makes a model price match a specified option quote. When
# the recovered values vary across strikes, the cross-section may form a smile,
# smirk, or skew rather than a constant line
# {cite}`blackScholes1973,merton1973,hull2022options`.
#
# This lesson uses a deterministic synthetic call chain with a *downward skew*.
# It performs price-bound, monotonicity, vertical-spread, and convexity checks
# before inversion, then verifies both volatility recovery and repricing. No
# value in the chain is an observed market quote.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - state model-independent European option price bounds;
# - test a same-maturity call chain for basic static-arbitrage violations;
# - solve a Black-Scholes-Merton implied volatility from an admissible quote;
# - verify volatility recovery and repricing numerically;
# - distinguish a monotone skew from a U-shaped smile;
# - explain why implied volatility is a model coordinate, not a realized-volatility forecast.
#
# ## Prerequisites
#
# Complete European Option Pricing and Greeks and the numerical-pricing lesson
# first. Students should understand continuous discounting, put-call parity,
# Vega, root finding, and the difference between calibration and forecasting.
#
# ## Price bounds before inversion
#
# With continuous dividend yield $q$, a European call and put satisfy
# {cite}`hull2022options`:
#
# $$
# \max\!\left(0,S_0e^{-qT}-Ke^{-rT}\right)
# \leq C_0\leq S_0e^{-qT},
# $$
#
# $$
# \max\!\left(0,Ke^{-rT}-S_0e^{-qT}\right)
# \leq P_0\leq Ke^{-rT}.
# $$
#
# A quote outside these bounds cannot have a valid Black-Scholes-Merton implied
# volatility. A quote exactly at the lower bound corresponds to zero limiting
# volatility; a quote at the finite upper bound has no finite implied volatility.
#
# For ordered strikes at one maturity, an arbitrage-screened call curve is
# non-increasing and convex in strike. In smooth notation,
#
# $$
# \frac{\partial C}{\partial K}\leq0,
# \qquad
# \frac{\partial^2 C}{\partial K^2}\geq0.
# $$
#
# Under additional regularity, the Breeden-Litzenberger identity connects call
# curvature to the risk-neutral terminal density
# {cite}`breedenLitzenberger1978`:
#
# $$
# f_{\mathbb{Q}}(K)=e^{rT}\frac{\partial^2 C(K,T)}{\partial K^2}.
# $$
#
# A finite quote grid only approximates these derivatives, so passing the checks
# is necessary evidence, not proof that a complete surface is arbitrage free.
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
    call_price_arbitrage_diagnostics,
    implied_volatility,
    option_price_bounds,
    put_call_parity_gap,
)
from src.module7_visuals import build_implied_volatility_figure

# %% [markdown]
# ## Deterministic synthetic call chain
#
# The chain is generated from a deliberately specified decreasing volatility
# function. Spot and option prices are currency units per share; $r$, $q$, and
# volatility are annual decimals; rates use continuous compounding; and maturity
# is in years.

# %%
spot = 100.0
rate = 0.055
dividend_yield = 0.01
maturity = 0.75
strikes = np.array([75.0, 85.0, 95.0, 100.0, 105.0, 115.0, 125.0])

generating_skew = 0.22 + 0.10 * ((strikes / spot) - 1.0) ** 2 - 0.08 * ((strikes / spot) - 1.0)
synthetic_call_prices = np.array(
    [
        black_scholes_price(
            spot,
            strike,
            rate,
            volatility,
            maturity,
            option_type="call",
            dividend_yield=dividend_yield,
        )
        for strike, volatility in zip(strikes, generating_skew)
    ]
)

chain = pd.DataFrame(
    {
        "strike": strikes,
        "synthetic_call_price": synthetic_call_prices,
        "generating_volatility": generating_skew,
    }
)
chain

# %% [markdown]
# **Interpretation.** Generating volatility falls as strike rises over the
# displayed domain, so this example is a downward skew rather than a U-shaped
# smile. The prices are exact model outputs, deliberately free from bid-ask
# spreads, rounding, stale timestamps, and microstructure noise.
#
# ## Pointwise no-arbitrage bounds

# %%
call_bounds = [
    option_price_bounds(
        spot,
        strike,
        rate,
        maturity,
        option_type="call",
        dividend_yield=dividend_yield,
    )
    for strike in strikes
]
chain[["call_lower_bound", "call_upper_bound"]] = pd.DataFrame(
    call_bounds,
    index=chain.index,
)
chain["inside_pointwise_bounds"] = (
    chain["synthetic_call_price"] >= chain["call_lower_bound"] - 1e-10
) & (chain["synthetic_call_price"] <= chain["call_upper_bound"] + 1e-10)
chain[
    [
        "strike",
        "synthetic_call_price",
        "call_lower_bound",
        "call_upper_bound",
        "inside_pointwise_bounds",
    ]
].round(8)

# %%
assert chain["inside_pointwise_bounds"].all()

# %% [markdown]
# ## Same-maturity call-curve diagnostics
#
# The shared diagnostic evaluates discrete price bounds, non-increasing call
# prices, the discounted vertical-spread slope bound, and convexity. It expects
# strictly increasing strikes and synchronized same-maturity prices.

# %%
arbitrage_checks = call_price_arbitrage_diagnostics(
    chain["strike"].to_numpy(),
    chain["synthetic_call_price"].to_numpy(),
    spot,
    rate,
    maturity,
    dividend_yield=dividend_yield,
)
arbitrage_checks

# %%
if not bool(arbitrage_checks["all_checks_pass"]):
    raise RuntimeError("Synthetic call chain failed the pre-inversion arbitrage screen")

# %% [markdown]
# **Interpretation.** The deterministic grid passes the implemented checks.
# These checks do not cover calendar-spread arbitrage because the chain has only
# one maturity, and they do not validate an interpolation rule between strikes.
#
# ## Invert Black-Scholes-Merton
#
# For each admissible synthetic quote, implied volatility solves
#
# $$
# C_{\mathrm{BSM}}(S_0,K,r,q,T,\sigma_{\mathrm{imp}})
# -C_{\mathrm{quote}}=0.
# $$
#
# Root finding is numerical. A result must therefore be followed by both a
# repricing check and, in this controlled synthetic example, a recovery check.

# %%
chain["implied_volatility"] = [
    implied_volatility(
        price,
        spot,
        strike,
        rate,
        maturity,
        option_type="call",
        dividend_yield=dividend_yield,
    )
    for price, strike in zip(chain["synthetic_call_price"], chain["strike"])
]
chain["repriced_call"] = [
    black_scholes_price(
        spot,
        strike,
        rate,
        volatility,
        maturity,
        option_type="call",
        dividend_yield=dividend_yield,
    )
    for strike, volatility in zip(chain["strike"], chain["implied_volatility"])
]
chain["repricing_error"] = chain["repriced_call"] - chain["synthetic_call_price"]
chain["volatility_recovery_error"] = chain["implied_volatility"] - chain["generating_volatility"]
chain["vega_per_1pct"] = [
    black_scholes_greeks(
        spot,
        strike,
        rate,
        volatility,
        maturity,
        option_type="call",
        dividend_yield=dividend_yield,
    )["vega_per_1pct"]
    for strike, volatility in zip(chain["strike"], chain["implied_volatility"])
]
chain[
    [
        "strike",
        "synthetic_call_price",
        "generating_volatility",
        "implied_volatility",
        "vega_per_1pct",
        "repricing_error",
        "volatility_recovery_error",
    ]
].round(10)

# %%
np.testing.assert_allclose(
    chain["repriced_call"],
    chain["synthetic_call_price"],
    atol=1e-9,
    rtol=0.0,
)
np.testing.assert_allclose(
    chain["implied_volatility"],
    chain["generating_volatility"],
    atol=1e-9,
    rtol=0.0,
)

# %% [markdown]
# **Interpretation.** Recovery succeeds because the same model generated and
# inverted the quotes. In real data there is no known generating volatility, and
# small price changes can cause large implied-volatility changes when Vega is low.
#
# ## Downward-skew visualization

# %% mystnb={"image": {"alt": "A deterministic line chart compares the decreasing volatility used to generate seven synthetic call prices with the implied volatilities recovered by inversion across strikes from 75 to 125; the two series overlap within numerical tolerance."}}
skew_figure = build_implied_volatility_figure(chain)
display(skew_figure)
plt.close(skew_figure)

# %% [markdown]
# ## Put-call parity at the central strike
#
# For synchronized European options with the same strike and maturity,
#
# $$
# C_0+Ke^{-rT}=P_0+S_0e^{-qT}.
# $$

# %%
central_strike = 100.0
central_volatility = chain.loc[
    chain["strike"] == central_strike,
    "implied_volatility",
].iloc[0]
central_call = black_scholes_price(
    spot,
    central_strike,
    rate,
    central_volatility,
    maturity,
    "call",
    dividend_yield,
)
central_put = black_scholes_price(
    spot,
    central_strike,
    rate,
    central_volatility,
    maturity,
    "put",
    dividend_yield,
)
central_parity_gap = put_call_parity_gap(
    central_call,
    central_put,
    spot,
    central_strike,
    rate,
    maturity,
    dividend_yield,
)
pd.Series(
    {
        "synthetic_call_price": central_call,
        "synthetic_put_price": central_put,
        "parity_gap_currency_per_share": central_parity_gap,
    },
    name="central_strike_parity_check",
)

# %%
assert abs(central_parity_gap) < 1e-10

# %% [markdown]
# ## From a strike slice to a surface
#
# A volatility surface extends the strike slice across maturities. Production
# construction requires synchronized quote timestamps, contract and dividend
# conventions, bid-ask filters, and checks for strike and calendar arbitrage.
# Breeden-Litzenberger explains why convexity in strike is economically
# consequential: negative call curvature would imply a negative state-price
# density under its assumptions {cite}`breedenLitzenberger1978`.
#
# Interpolation and extrapolation are model choices. A visually smooth surface
# is not necessarily an arbitrage-free or stable surface, and implied volatility
# should not be presented as a direct forecast of realized volatility.
#
# ## Source and model notes
#
# - The constant-volatility pricing map being inverted originates in
#   Black-Scholes and Merton {cite}`blackScholes1973,merton1973`; price bounds,
#   parity, and inversion conventions follow the standard derivatives treatment
#   in {cite:t}`hull2022options`.
# - The connection between call-price curvature and risk-neutral distributions
#   follows Breeden and Litzenberger {cite}`breedenLitzenberger1978`.
# - The seven call prices are generated deterministically inside this notebook
#   from the displayed downward-skew function; they are not observed quotes.
# - The recovery test is intentionally in-sample and same-model. It verifies code
#   consistency, not empirical fit or forecasting value.
#
# ## Model limitations
#
# - The diagnostic checks only one finite strike grid at one maturity; it cannot
#   establish global surface arbitrage freedom.
# - Implied volatility depends on the pricing model and all non-volatility inputs,
#   including dividends, rates, exercise style, and settlement convention.
# - Live quotes can be stale, crossed, illiquid, or asynchronous. Bid-ask width
#   should be propagated into implied-volatility intervals rather than hidden.
# - Very low Vega makes inversion ill-conditioned even when a quote passes
#   simple price bounds.
#
# ## Handoff
#
# Implied volatility calibrates a model to a quote at a point. Vega maps local
# volatility changes into P&L, while smile dynamics, interpolation, stale quotes,
# and failed no-arbitrage checks become model-risk scenarios. The next lessons
# add interactive sensitivity analysis, early exercise, path dependence, and
# stochastic volatility.
