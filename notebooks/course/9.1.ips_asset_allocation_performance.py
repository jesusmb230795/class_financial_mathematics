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
# # Investment Policy, Asset Allocation, and Performance
#
# Module: Portfolio Management, Asset Allocation, and Performance
#
# ## Lesson summary
#
# Portfolio optimization is not the starting decision. An investor first needs
# an Investment Policy Statement (IPS) that converts purpose, liabilities,
# constraints, and governance into an implementable allocation and monitoring
# process. This lesson connects the IPS to capital market expectations,
# strategic and tactical allocation, implementation costs, rebalancing, and
# performance evaluation {cite}`damodaran2012investment,sharpe1994ratio,lo2002sharpe`.
#
# ## Learning objectives
#
# By the end of this lesson, readers should be able to:
#
# - distinguish investment objectives, risk capacity, risk tolerance, and
#   constraints;
# - convert a real objective and inflation assumption into a nominal return
#   requirement;
# - document strategic allocation ranges and a rules-based rebalancing policy;
# - separate manager performance from external cash-flow timing; and
# - calculate a simplified allocation contribution relative to a benchmark.
#
# ## Prerequisites
#
# Readers should understand simple and log returns, annualization, volatility,
# covariance, drawdown, VaR/Expected Shortfall, and the macro scenario template.
# Portfolio returns and rates are stored as decimals. Every reported performance
# number must state period, currency, gross/net status, and benchmark.

# %% [markdown]
# ## The Investment Policy Statement
#
# A usable IPS is a decision contract, not a marketing description.
#
# | IPS field | Observable content |
# | --- | --- |
# | Purpose | liability, spending, purchase, reserve, retirement, or mandate |
# | Return objective | nominal/real, gross/net, currency, horizon, and probability |
# | Risk capacity | economic ability to absorb loss without failing the purpose |
# | Risk tolerance | approved willingness to accept uncertainty and drawdown |
# | Horizon | decision, liability, review, and liquidity horizons |
# | Liquidity | scheduled and stressed cash needs, collateral, and reserves |
# | Constraints | legal, tax, concentration, leverage, ESG, currency, and instrument |
# | Allocation | targets, ranges, benchmarks, and permitted implementation |
# | Governance | owner, delegate, review frequency, exceptions, and escalation |
#
# Risk capacity and risk tolerance can disagree. A long horizon may increase
# capacity, while a near-term distribution, covenant, or governance constraint
# can reduce it. The lower effective boundary should constrain the portfolio
# until the conflict is resolved.

# %% [markdown]
# ## Return requirement and capital market expectations
#
# If the required real return is \(r_{\mathrm{real}}\) and expected inflation is
# \(\pi\), the exact nominal return requirement before fees is:
#
# ```{math}
# 1+r_{\mathrm{nominal}}
# =(1+r_{\mathrm{real}})(1+\pi).
# ```
#
# Fees, taxes, cash drag, and implementation shortfall must then be included
# under explicit conventions. Capital market expectations are uncertain inputs,
# not guaranteed asset-class returns. A forecast set should include expected
# return, volatility, correlation, horizon, currency, inflation, valuation
# regime, and scenario sensitivity.
#
# ### Applied case: a spending portfolio
#
# A MXN 100 million portfolio seeks a 5.00% real annual return before fees, with
# expected inflation of 4.00%. The exact nominal requirement is:
#
# ```{math}
# (1.05)(1.04)-1=9.20\%.
# ```
#
# This is a planning hurdle, not a forecast. If the approved capital market
# expectations cannot support it at acceptable risk, governance must change the
# spending rule, contribution, horizon, or risk budget rather than forcing an
# optimizer to manufacture the return.

# %% [markdown]
# ## Strategic and tactical allocation
#
# **Strategic asset allocation (SAA)** expresses the long-horizon risk and
# return structure consistent with the IPS. **Tactical asset allocation (TAA)**
# is a controlled, time-bounded deviation based on an active view. TAA requires
# a thesis, size limit, horizon, benchmark, stop/review condition, and owner.
#
# A target without a range causes unnecessary trading; a range without a
# rebalance rule invites discretion after losses. One auditable policy is:
#
# - review monthly;
# - rebalance when any asset class leaves its approved range;
# - use external cash flows first;
# - trade toward target or the nearest approved point;
# - include tax lots, bid-ask spreads, market impact, and liquidity;
# - record approvals and exceptions.
#
# ### Applied rebalance
#
# Assume target weights of 50% bonds, 35% equities, 10% real assets, and 5% cash
# for a MXN 100 million portfolio. After market moves, weights are 46%, 40%, 9%,
# and 5%. Returning exactly to target implies:
#
# | Asset class | Current MXN m | Target MXN m | Trade MXN m |
# | --- | ---: | ---: | ---: |
# | Bonds | 46 | 50 | +4 |
# | Equities | 40 | 35 | -5 |
# | Real assets | 9 | 10 | +1 |
# | Cash | 5 | 5 | 0 |
#
# Gross one-way traded notional is MXN 10 million: buys plus sells. If the
# all-in cost assumption is 20 basis points of each traded amount, estimated
# implementation cost is MXN 20,000. Some institutions report turnover as half
# the sum of absolute trades; the convention must be stated before comparison.

# %% [markdown]
# ## Portfolio construction and risk budgeting
#
# Mean-variance, minimum-variance, risk-parity, hierarchical, and resampled
# portfolios are implementation tools. Their output must be constrained by the
# IPS and tested for:
#
# - estimation error and unstable weights;
# - concentration by issuer, factor, country, currency, and liquidity;
# - leverage, shorting, derivatives, and collateral requirements;
# - normal and stressed liquidity;
# - drawdown and tail loss under macro scenarios;
# - turnover, taxes, bid-ask spread, and market impact; and
# - sensitivity to expected returns, covariance, and rebalance dates.
#
# Risk contribution is not the same as capital weight. A small volatile or
# highly correlated allocation can dominate marginal portfolio risk.

# %% [markdown]
# ## Measuring performance
#
# **Time-weighted return (TWR)** geometrically links subperiod returns and removes
# the mechanical effect of external cash-flow timing. It is usually appropriate
# for evaluating a manager who does not control contributions and withdrawals.
#
# **Money-weighted return (MWR)** is the internal rate of return on dated investor
# cash flows. It measures the investor experience but can differ from manager
# skill because the investor controls timing.
#
# Performance must be reported:
#
# - for a defined date range and valuation calendar;
# - in a defined currency;
# - gross and/or net of defined fees and costs;
# - against an investable, relevant, and consistently rebalanced benchmark; and
# - with risk, drawdown, and exposure context.
#
# Sharpe ratios require the same return frequency and a matched risk-free series.
# Serial correlation changes their sampling properties, so annualization should
# not be applied mechanically {cite}`sharpe1994ratio,lo2002sharpe`.

# %% [markdown]
# ## Attribution: an allocation-only case
#
# Consider two asset classes. A benchmark holds 60% equities and 40% bonds; the
# portfolio holds 50% and 50%. Equities return 12% and bonds return 4% during the
# period, with no security-selection difference.
#
# ```{math}
# R_B=0.60(0.12)+0.40(0.04)=8.80\%,
# ```
#
# ```{math}
# R_P=0.50(0.12)+0.50(0.04)=8.00\%.
# ```
#
# Active return is -0.80 percentage points. A simplified allocation
# contribution relative to total benchmark return is:
#
# | Asset | Calculation | Contribution |
# | --- | --- | ---: |
# | Equities | \((0.50-0.60)(0.12-0.088)\) | -0.32 pp |
# | Bonds | \((0.50-0.40)(0.04-0.088)\) | -0.48 pp |
# | Total | sum | -0.80 pp |
#
# This two-asset example isolates allocation because portfolio and benchmark
# asset returns are identical. Full attribution requires a declared methodology
# for allocation, selection, interaction, cash, derivatives, currency, fees,
# and geometric linking.

# %% [markdown]
# ## Monitoring and governance
#
# A portfolio review should separate four questions:
#
# 1. **Objective:** Are liabilities, spending, horizon, or constraints changing?
# 2. **Positioning:** Are weights, factors, liquidity, and currency within range?
# 3. **Outcome:** What return, risk, drawdown, cost, and attribution occurred?
# 4. **Decision:** Maintain, rebalance, change assumptions, revise the IPS, or
#    escalate an exception?
#
# Record the valuation source, benchmark version, external cash flows, corporate
# actions, fee treatment, classification changes, and approvals. A performance
# number that cannot be regenerated from positions and cash flows is not a
# governance artifact.

# %% [markdown]
# ## Limitations
#
# - A single real-return hurdle does not model the timing or probability of
#   liabilities.
# - Expected inflation and capital market expectations are uncertain and can be
#   internally inconsistent.
# - Target weights conceal factor, liquidity, currency, and nonlinear exposures.
# - Rebalancing examples omit taxes, market impact, settlement, and minimum lots.
# - TWR, MWR, and attribution answer different questions and can conflict when
#   cash flows or classifications are large.
# - Historical performance and optimized allocations do not establish future
#   suitability.

# %% [markdown]
# ## Handoff
#
# The following portfolio lessons estimate return and covariance inputs, trace
# efficient frontiers, compare robust allocations, and visualize sensitivity.
# Their outputs become decision-ready only after they are checked against this
# IPS, implementation, performance, and governance contract.
