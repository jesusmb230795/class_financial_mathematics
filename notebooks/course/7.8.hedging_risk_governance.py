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
# {cite}`hull2022options,mcneil2015quantitative,basel2019marketRisk,fed2026ModelRiskGuidance`.
#
# ## Learning objectives
#
# By the end of this lesson, readers should be able to:
#
# - translate an exposure into a hedge ratio using matched sensitivity units;
# - approximate nonlinear option profit and loss with delta, gamma, and vega;
# - size a futures overlay for cash equitization or a target portfolio beta;
# - distinguish a volatility risk contribution from an approved risk budget;
# - distinguish market, basis, liquidity, counterparty, funding, operational,
#   legal, and model risk;
# - separate ex-ante hedge design from ex-post effectiveness measurement;
# - design a limit with a metric, threshold, observation frequency, owner, and
#   escalation action; and
# - explain why a hedge can reduce one risk while increasing another.
#
# ## Prerequisites
#
# Readers should understand present value, return and loss signs, duration, the
# value of a one-basis-point rate move (DV01), foreign-exchange (FX) quote
# direction, and the macro scenario template. Rates and returns remain decimals
# in calculations. One basis point is $0.0001$ in rate units.

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
#    profit and loss (P&L) in consistent units.
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
# If a portfolio has signed first-order value change $S_P$ for a stated factor
# shock and one **long** unit of hedge instrument has signed change $S_H$ for
# the same shock, the idealized signed hedge count is
#
# ```{math}
# n_H^*=-\frac{S_P}{S_H}.
# ```
#
# A positive $n_H^*$ means long hedge units and a negative value means short
# units. If sensitivities are reported instead as positive loss magnitudes,
# $\lvert n_H^*\rvert=\lvert S_P\rvert/\lvert S_H\rvert$ determines only the
# magnitude; the direction must be recovered from signed scenario P&L. Mixing a
# signed sensitivity with a loss magnitude silently reverses the hedge.
#
# Delta, duration, and DV01 are local approximations. Gamma, convexity, basis
# moves, curve shape changes, volatility changes, and jumps remain after a
# first-order hedge.

# %% [markdown]
# ### Applied case: DV01 hedge and residual risk
#
# A fixed-rate portfolio with a market value of MXN 50 million has modified
# duration 6.2. For a one-basis-point yield increase, its approximate signed P&L
# and positive DV01 loss magnitude are
#
# ```{math}
# S_P=-50{,}000{,}000(6.2)(0.0001)
#     =-\text{MXN }31{,}000,
# \qquad
# DV01_P=\text{MXN }31{,}000.
# ```
#
# Suppose one **long** liquid rate-futures contract has signed P&L
# $S_H=-\text{MXN }850$ for the same yield shock after its contract and
# delivery conventions are applied. Then
# $n_H^*=-(-31{,}000)/(-850)=-36.47$: the idealized direction is short.
# Rounding to 36 short contracts gives:
#
# | Quantity | Result |
# | --- | ---: |
# | Portfolio signed P&L for +1 bp | −MXN 31,000 |
# | Hedge signed P&L for +1 bp | +MXN 30,600 |
# | Net signed P&L for +1 bp | −MXN 400 |
# | Approximate net P&L for +25 bp | −MXN 10,000 |
#
# The table does not prove a MXN 10,000 realized loss. It assumes a matched
# parallel move, stable conversion factors and duration, no convexity, and no
# transaction or funding costs. A 37-contract hedge would reverse the residual
# direction and create MXN 450 per bp of over-hedge.

# %% [markdown]
# ## Nonlinear option P&L
#
# A delta hedge neutralizes only the local linear term. For a small joint spot
# and volatility shock, a delta-gamma-vega approximation is
#
# ```{math}
# \Delta V
# \approx
# \Delta\,\Delta S
# +\frac{1}{2}\Gamma(\Delta S)^2
# +\nu\,\Delta\sigma.
# ```
#
# Units must match. If $\nu$ is calculated per one-unit change in decimal
# volatility, then $\Delta\sigma$ is a decimal shock such as $0.01$. If the
# report scales vega per percentage point, use a one-point shock instead.
# Cross-Greeks, theta, jumps, discrete rebalancing, transaction costs, and
# changes in implied-volatility shape remain outside this approximation.

# %% [markdown]
# ## Futures overlays, cash equitization, and risk budgets
#
# Let $V_P$ be portfolio market value in currency units, $\beta_P$ its
# current exposure to an equity index, $\beta_{\mathrm{target}}$ the approved
# target, $F_0$ the futures price in index points, and $m$ the contract
# multiplier in currency units per index point {cite}`hull2022options`.
# An idealized futures overlay is:
#
# ```{math}
# n_F^*
# =\frac{(\beta_{\mathrm{target}}-\beta_P)V_P}{F_0m}.
# ```
#
# A cash portfolio can be approximately equitized by setting
# $\beta_P\approx0$ and $\beta_{\mathrm{target}}\approx1$. The formula is a
# first-order sizing rule, not permission to trade: contract rounding, dividend
# assumptions, futures basis, currency conversion, liquidity, and variation
# margin must enter the implementation record.
#
# For asset weights $w$ and return covariance matrix $\Sigma$, portfolio
# volatility and Euler volatility contributions are
# {cite}`mcneil2015quantitative`:
#
# ```{math}
# \sigma_P=\sqrt{w^\top\Sigma w},
# \qquad
# RC_i=\frac{w_i(\Sigma w)_i}{\sigma_P},
# \qquad
# b_i=\frac{RC_i}{\sigma_P},
# \qquad
# \sum_i RC_i=\sigma_P,\quad \sum_i b_i=1.
# ```
#
# $RC_i$ is a model-derived risk contribution and $b_i$ is its share under
# the chosen covariance estimate. An approved **risk budget** is a governance
# target or limit against which those estimates are compared; it is not
# automatically equal to the latest measured $b_i$.

# %% [markdown]
# ## Risk does not disappear; it changes form
#
# | Risk | Hedge-process question | Example control |
# | --- | --- | --- |
# | Market risk | Which factor and shock is being reduced? | DV01, delta, vega, scenario P&L |
# | Basis risk | Can the position and hedge move differently? | Bucketed sensitivities and historical basis stress |
# | Liquidity risk | Can the hedge be entered, resized, and exited? | Bid-ask, depth, days-to-liquidate limit |
# | Counterparty risk | What happens if the hedge provider defaults? | Credit Support Annex (CSA), collateral, netting, exposure limit |
# | Funding risk | Can margin calls be met under stress? | Liquidity buffer and margin stress |
# | Operational risk | Are confirmations, prices, and positions correct? | Reconciliation and independent price verification |
# | Legal risk | Is close-out, collateral, and settlement enforceable? | Approved documentation and legal review |
# | Model risk | Are price and sensitivity fit for the decision? | Independent validation, benchmark, and change control |
#
# A portfolio can show lower market Value at Risk (VaR) after hedging and still
# become harder to fund because daily variation margin has increased.

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
# loss mechanisms. VaR or Expected Shortfall (ES) can sit beside stress loss,
# gross notional, concentration, sensitivity, liquidity, and counterparty
# exposure.

# %% [markdown]
# ## Model governance
#
# A pricing or risk model should have the following controls, consistent with a
# lifecycle view of model-risk management
# {cite}`fed2026ModelRiskGuidance`:
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

# %% [markdown]
# ## Handoff
#
# The preceding lessons priced linear derivatives and options, calculated
# Greeks, and compared numerical methods. The following lessons estimate and
# backtest portfolio tail risk. Every result returns to this governance frame:
# objective, sensitivity, hedge, residual risk, limit, evidence, owner, and
# escalation.
