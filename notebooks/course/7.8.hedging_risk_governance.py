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
# # Hedging, Risk Limits, and Governance
#
# Module: Derivatives and Risk Management
#
# ## Lesson summary
#
# A derivative model produces a price or sensitivity; a risk-management process
# decides which exposure is intentional, which exposure should be hedged, what
# residual risk is acceptable, and who can approve exceptions. This lesson
# connects derivative sensitivities with hedge construction, effectiveness
# testing, limits, escalation, and model governance
# {cite}`hull2022options,mcneil2015quantitative,basel2019marketRisk`.
#
# ## Learning objectives
#
# By the end of this lesson, readers should be able to:
#
# - translate an exposure into a hedge ratio using matched sensitivity units;
# - distinguish market, basis, liquidity, counterparty, funding, operational,
#   legal, and model risk;
# - separate ex-ante hedge design from ex-post effectiveness measurement;
# - design a limit with a metric, threshold, observation frequency, owner, and
#   escalation action; and
# - explain why a hedge can reduce one risk while increasing another.
#
# ## Prerequisites
#
# Readers should understand present value, return and loss signs, duration and
# DV01, FX quote direction, and the macro scenario template. Rates and returns
# remain decimals in calculations. One basis point is \(0.0001\) in rate units.

# %% [markdown]
# ## The position-to-governance chain
#
# A controlled hedge process preserves an auditable sequence:
#
# 1. **Position:** verify notional, currency, maturity, optionality, settlement,
#    legal entity, and valuation timestamp.
# 2. **Risk factor:** map the position to rates, FX, equity, volatility, credit,
#    liquidity, or other drivers.
# 3. **Sensitivity:** calculate DV01, delta, vega, spread duration, or scenario
#    P&L in consistent units.
# 4. **Hedge instrument:** document contract, multiplier, maturity, liquidity,
#    collateral, counterparty, and basis.
# 5. **Hedge ratio:** size the hedge against the selected sensitivity.
# 6. **Residual risk:** recalculate all relevant sensitivities and stress P&L,
#    not only the hedged metric.
# 7. **Limit and monitoring:** assign thresholds, data frequency, owner,
#    exceptions, and escalation.
# 8. **Effectiveness review:** compare realized changes with the hedge objective
#    and investigate drift.
#
# A hedge is defined by its documented objective. A trade that happens to offset
# historical P&L is not automatically a hedge if its exposure, horizon, and
# rebalance rule are undefined.

# %% [markdown]
# ## Sensitivity-based hedge ratios
#
# If a portfolio has first-order sensitivity \(S_P\) and one unit of hedge
# instrument has sensitivity \(S_H\), the idealized hedge count is:
#
# ```{math}
# N^*=\frac{S_P}{S_H}.
# ```
#
# Direction must be attached separately. A long fixed-rate bond portfolio has
# positive DV01 as a reported loss magnitude for a one-basis-point yield rise.
# A short rate-futures position is commonly used to create the offsetting
# first-order exposure. Contract rounding leaves a residual sensitivity.
#
# Delta, duration, and DV01 are local approximations. Gamma, convexity, basis
# moves, curve shape changes, volatility changes, and jumps remain after a
# first-order hedge.

# %% [markdown]
# ### Applied case: DV01 hedge and residual risk
#
# A MXN 50 million fixed-rate portfolio has modified duration 6.2. Its
# approximate DV01 loss magnitude is:
#
# ```{math}
# DV01_P=50{,}000{,}000(6.2)(0.0001)=\text{MXN }31{,}000.
# ```
#
# Suppose one liquid futures contract has MXN 850 of DV01 in the relevant
# direction after its contract and delivery conventions are applied. The
# unrounded ratio is \(31{,}000/850=36.47\). Using 36 short contracts gives:
#
# | Quantity | Result |
# | --- | ---: |
# | Portfolio DV01 | MXN 31,000 per bp |
# | Hedge DV01 from 36 contracts | MXN 30,600 per bp |
# | Residual first-order DV01 | MXN 400 per bp |
# | Approximate residual loss for +25 bp | MXN 10,000 |
#
# The table does not prove a MXN 10,000 realized loss. It assumes a matched
# parallel move, stable conversion factors and duration, no convexity, and no
# transaction or funding costs. A 37-contract hedge would reverse the residual
# direction and create MXN 450 per bp of over-hedge.

# %% [markdown]
# ## Risk does not disappear; it changes form
#
# | Risk | Hedge-process question | Example control |
# | --- | --- | --- |
# | Market risk | Which factor and shock is being reduced? | DV01, delta, vega, scenario P&L |
# | Basis risk | Can the position and hedge move differently? | Bucketed sensitivities and historical basis stress |
# | Liquidity risk | Can the hedge be entered, resized, and exited? | Bid-ask, depth, days-to-liquidate limit |
# | Counterparty risk | What happens if the hedge provider defaults? | CSA, collateral, netting, exposure limit |
# | Funding risk | Can margin calls be met under stress? | Liquidity buffer and margin stress |
# | Operational risk | Are confirmations, prices, and positions correct? | Reconciliation and independent price verification |
# | Legal risk | Is close-out, collateral, and settlement enforceable? | Approved documentation and legal review |
# | Model risk | Are price and sensitivity fit for the decision? | Independent validation, benchmark, and change control |
#
# A portfolio can show lower market VaR after hedging and still become harder to
# fund because daily variation margin has increased.

# %% [markdown]
# ## Ex-ante design and ex-post effectiveness
#
# **Ex-ante** analysis asks whether the selected hedge reduces the intended risk
# under defined scenarios. Evidence can include residual DV01, delta-gamma P&L,
# key-rate duration, basis shocks, volatility shocks, and liquidity stress.
#
# **Ex-post** analysis compares realized position and hedge changes over the
# documented horizon. A simple effectiveness series is:
#
# ```{math}
# \Delta V_{\mathrm{net},t}
# =\Delta V_{\mathrm{position},t}+\Delta V_{\mathrm{hedge},t}.
# ```
#
# Report both gross legs and the net result. A small net P&L can conceal two
# large offsetting and unstable positions. Effectiveness drift can come from
# changing sensitivities, time decay, cash flows, nonlinear payoff, basis,
# volatility, rebalancing delay, or stale prices.

# %% [markdown]
# ## Limits and escalation
#
# "Keep risk low" is not a limit. A usable limit specification contains:
#
# | Field | Observable requirement |
# | --- | --- |
# | Scope | legal entity, desk, portfolio, strategy, and instruments |
# | Metric | defined formula, sign, horizon, currency, and aggregation |
# | Threshold | warning, hard limit, and stress threshold |
# | Data | price source, timestamp, quality flags, and fallback |
# | Frequency | intraday, daily, weekly, or event-driven |
# | Owner | named first-line owner and independent oversight |
# | Breach action | stop, reduce, hedge, investigate, or seek exception |
# | Exception | approving authority, rationale, size, expiry, and remediation |
# | Evidence | immutable position, model version, input, result, and approval log |
#
# Risk appetite is a governance decision; a model does not choose it. Limits
# should cover complementary views because one statistic cannot represent all
# loss mechanisms. VaR or Expected Shortfall can sit beside stress loss, gross
# notional, concentration, sensitivity, liquidity, and counterparty exposure.

# %% [markdown]
# ## Model governance
#
# A pricing or risk model should have:
#
# - a defined intended use and prohibited uses;
# - versioned code, parameters, market data, and valuation date;
# - independent conceptual review and benchmark tests;
# - calibration, backtesting, sensitivity, and stress evidence;
# - known limitations and conservative fallbacks;
# - change approval and reproducible release artifacts; and
# - performance monitoring with thresholds for recalibration or withdrawal.
#
# Model output must retain its uncertainty. Rounding a hedge ratio to a contract
# count is an implementation choice; calibrating on illiquid quotes is a model
# choice; accepting residual exposure is a governance choice. The audit trail
# should distinguish all three.

# %% [markdown]
# ## Limitations
#
# - The DV01 case assumes one matched rate shock and ignores curve buckets,
#   convexity, futures basis, cheapest-to-deliver dynamics, and margin.
# - Sensitivities are local and can change materially with price, volatility,
#   time, and optionality.
# - Historical hedge effectiveness can fail after a regime or liquidity shift.
# - Risk limits can create false comfort when data are stale, aggregation omits
#   positions, or economically equivalent exposures use inconsistent units.
# - Governance cannot eliminate judgment; it makes authority, evidence,
#   exceptions, and accountability observable.

# %% [markdown] tags=["exercise"]
# ## Assessment
#
# A long fixed-income portfolio has a first-order DV01 loss magnitude of MXN
# 42,000 per basis point. One hedge contract provides MXN 1,050 per basis point
# in the opposite direction.
#
# 1. Calculate the whole-contract hedge count and direction.
# 2. Calculate the first-order portfolio loss and hedge gain for a +30 basis
#    point matched yield shock.
# 3. Report residual first-order P&L.
# 4. Name three risks that remain even when the residual is zero.
# 5. Define one observable escalation trigger for hedge drift.

# %% [markdown] tags=["solution"]
# ```{dropdown} Suggested answer
# The ratio is \(42{,}000/1{,}050=40\), so use 40 contracts in the direction
# opposite the portfolio DV01 (short under the stated long-bond setup). A +30 bp
# shock produces an approximate MXN 1,260,000 portfolio loss and MXN 1,260,000
# hedge gain, leaving zero first-order residual. Basis risk, convexity,
# liquidity/margin risk, counterparty risk, and model risk remain. One observable
# trigger is: if absolute residual DV01 exceeds MXN 2,000 per bp at the daily
# close, the desk must investigate and rebalance or obtain a documented
# exception before the next trading day.
# ```

# %% [markdown]
# ## Handoff
#
# The preceding lessons priced linear derivatives and options, calculated
# Greeks, and compared numerical methods. The following lessons estimate and
# backtest portfolio tail risk. Every result returns to this governance frame:
# objective, sensitivity, hedge, residual risk, limit, evidence, owner, and
# escalation.
