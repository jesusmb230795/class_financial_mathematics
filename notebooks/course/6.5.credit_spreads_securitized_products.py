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
# # Credit Spreads, Credit Loss, and Securitized Products
#
# Module: Fixed Income, Credit, and Term Structure
#
# ## Lesson summary
#
# Government-curve valuation is only the first layer of fixed-income analysis.
# Credit instruments add default, recovery, migration, liquidity, seniority, and
# contractual optionality. Securitized products add a pool, a payment waterfall,
# prepayment behavior, and tranche-specific loss allocation. This lesson supplies
# the vocabulary and decision checks required before those risks are added to the
# bond and term-structure calculations elsewhere in Module 6. The treatment
# follows standard fixed-income and quantitative-risk distinctions; the
# securitization section also anchors its terminology in the Basel framework
# {cite}`fabozzi2019foundations,mcneil2015quantitative,baselFrameworkCreditSecuritisation2026`.
#
# ## Learning objectives
#
# By the end of this lesson, readers should be able to:
#
# - distinguish a nominal yield spread, a zero-volatility spread, and an
#   option-adjusted spread;
# - calculate expected credit loss from probability of default, loss given
#   default, and exposure at default;
# - explain why an observed spread is not identical to expected loss;
# - allocate collateral loss through a simplified securitization waterfall; and
# - identify extension, contraction, model, liquidity, and legal-structure risk.
#
# ## Prerequisites
#
# Readers should be able to discount deterministic cash flows, distinguish clean
# from dirty bond price, and interpret spot rates and discount factors. Rate
# comparisons require matched currency, maturity, compounding, day count, and
# cash-flow timing. The examples use annual decimal rates and simplified
# one-period loss calculations.

# %% [markdown]
# ## Python setup

# %% tags=["setup", "hide-input"]
import pandas as pd

from src.fixed_income import tranche_loss
from src.module6_visuals import build_securitization_waterfall_figure

# %% [markdown]
# ## From a government curve to a credit spread
#
# A nominal spread subtracts a selected benchmark yield from a credit
# instrument's yield:
#
# ```{math}
# s_{\mathrm{nominal}}=y_{\mathrm{credit}}-y_{\mathrm{benchmark}}.
# ```
#
# This number is meaningful only when the comparison is economically matched.
# Two equal-maturity instruments can still differ in coupon, amortization,
# liquidity, tax, seniority, collateral, and embedded options.
#
# A **zero-volatility spread** (Z-spread) is the constant spread added to each
# point on a benchmark spot curve so that discounted promised cash flows match
# the observed price. An **option-adjusted spread** (OAS) removes the modeled
# value of an embedded option from the spread calculation. OAS therefore depends
# on the interest-rate and prepayment model; it is not directly observed
# {cite}`fabozzi2019foundations`.

# %% [markdown]
# ## Expected loss is a component, not the whole spread
#
# For a one-period exposure,
#
# ```{math}
# EL = PD \times LGD \times EAD,
# ```
#
# where \(PD\) is probability of default, \(LGD=1-\text{recovery rate}\), and
# \(EAD\) is exposure at default. Expected loss is a mean loss under the selected
# horizon and assumptions {cite}`mcneil2015quantitative`. Investors can also
# require compensation for
# unexpected loss, migration, liquidity, model uncertainty, taxes, funding,
# optionality, and risk aversion.
#
# ### Applied case: a five-year corporate bond
#
# Assume:
#
# - matched government benchmark YTM: 8.00% nominal annual, convertible
#   semiannually;
# - corporate YTM: 10.20% nominal annual, convertible semiannually;
# - the two hypothetical quotes use the same valuation and settlement date,
#   five-year final maturity, and 30/360 day-count basis;
# - one-year probability of default: 2.00%;
# - recovery rate: 40.00%; and
# - exposure at default: MXN 10,000,000.
#
# Under that shared quotation convention, the illustrative nominal spread is:
#
# ```{math}
# s_{\mathrm{nominal}}=(0.1020-0.0800)10^4=220\ \text{bp}.
# ```
#
# The one-year expected loss is:
#
# ```{math}
# EL=0.02(1-0.40)(10{,}000{,}000)=\text{MXN }120{,}000,
# ```
#
# or 1.20% of exposure. It would be incorrect to call the remaining 100 basis
# points "profit." The yield spread and expected loss differ in horizon, timing,
# discounting, liquidity, risk premia, and contractual assumptions. A defensible
# memo records those differences instead of forcing a residual interpretation.

# %% [markdown]
# ## Credit migration and spread risk
#
# A bond can lose value without default. A rating downgrade or deterioration in
# market-implied credit quality can widen spreads and lower price. Conversely, a
# tightening spread can raise price even if the issuer's promised cash flows do
# not change.
#
# Keep three related but different calculations separate:
#
# | Calculation | Question | Main limitation |
# | --- | --- | --- |
# | Expected loss | What is the mean credit loss over a horizon? | Sensitive to PD, LGD, EAD, and dependence |
# | Spread duration | How does price change for a small spread move? | Local approximation; holds the base curve fixed |
# | Transition scenario | What happens after migration or default? | Requires a horizon, transition matrix, recovery, and repricing rule |
#
# Spread duration cannot be added mechanically to interest-rate duration when
# the two shocks are correlated or when the instrument contains options.

# %% [markdown]
# ## Securitization structure
#
# A securitization transforms a pool of contractual cash flows into securities
# with different priorities. A simplified structure contains:
#
# 1. collateral and a servicing process;
# 2. fees and liquidity reserves;
# 3. a waterfall that allocates interest and principal;
# 4. credit enhancement such as subordination or overcollateralization; and
# 5. equity, mezzanine, and senior tranches.
#
# Tranching reallocates pool losses; it does not eliminate them. Dependence among
# borrowers is especially important because correlated defaults can move losses
# through several attachment points at once. Regulatory securitization
# treatments likewise distinguish tranche attachment and detachment points and
# recognize that contractual waterfalls and structural features change the
# allocation of risk {cite}`baselFrameworkCreditSecuritisation2026`.

# %% [markdown]
# ### Applied case: allocate pool loss through a waterfall
#
# Consider a MXN 100 million pool. The equity tranche absorbs losses from 0 to
# MXN 4 million, the mezzanine tranche from MXN 4 million to MXN 10 million, and
# the senior tranche absorbs losses above MXN 10 million.
#
# Let $L\geq0$ be collateral loss in MXN millions. For tranche $j$ with
# attachment point $A_j$, detachment point $D_j$, and notional $D_j-A_j$, its
# allocated loss is
#
# ```{math}
# L_j(L)=\min\!\left(\max(L-A_j,0),D_j-A_j\right),
# \qquad
# \ell_j(L)=\frac{L_j(L)}{D_j-A_j}.
# ```
#
# Thus $L_j$ is a currency amount and $\ell_j$ is the tranche loss fraction. In
# each hypothetical stress, total collateral loss is
#
# ```{math}
# L=\text{pool notional}\times PD_{\mathrm{pool}}\times(1-R),
# ```
#
# where the scenario default share $PD_{\mathrm{pool}}$ and recovery rate $R$
# are deterministic assumptions, not estimated probabilities.
#
# **Base stress.** If 6% of the pool defaults and recovery is 50%, collateral
# loss is:
#
# ```{math}
# 100(0.06)(1-0.50)=\text{MXN }3\text{ million}.
# ```
#
# Equity absorbs all MXN 3 million; mezzanine and senior absorb zero.
#
# **Severe stress.** If 15% defaults and recovery falls to 40%, collateral loss
# is:
#
# ```{math}
# 100(0.15)(1-0.40)=\text{MXN }9\text{ million}.
# ```
#
# Equity absorbs MXN 4 million and is exhausted. Mezzanine absorbs MXN 5
# million. Senior absorbs zero, but its remaining protection has fallen from
# MXN 10 million to MXN 1 million. A zero current senior loss is not evidence of
# zero senior risk.

# %%
pool_notional = 100.0  # MXN millions
tranche_specification = pd.DataFrame(
    {
        "tranche": ["Equity", "Mezzanine", "Senior"],
        "attachment": [0.0, 4.0, 10.0],
        "detachment": [4.0, 10.0, pool_notional],
    }
)
stress_scenarios = pd.DataFrame(
    {
        "scenario": ["Base stress", "Severe stress", "Tail stress"],
        "default_rate": [0.06, 0.15, 0.25],
        "recovery_rate": [0.50, 0.40, 0.30],
    }
)
stress_scenarios["collateral_loss"] = pool_notional * stress_scenarios[
    "default_rate"
] * (1 - stress_scenarios["recovery_rate"])

allocation_rows = []
for scenario_row in stress_scenarios.itertuples(index=False):
    for tranche_row in tranche_specification.itertuples(index=False):
        allocated_loss = tranche_loss(
            scenario_row.collateral_loss,
            tranche_row.attachment,
            tranche_row.detachment,
        )
        allocation_rows.append(
            {
                "scenario": scenario_row.scenario,
                "default_rate": scenario_row.default_rate,
                "recovery_rate": scenario_row.recovery_rate,
                "collateral_loss": scenario_row.collateral_loss,
                "tranche": tranche_row.tranche,
                "attachment": tranche_row.attachment,
                "detachment": tranche_row.detachment,
                "tranche_loss": allocated_loss,
            }
        )

waterfall_allocation = pd.DataFrame(allocation_rows)
waterfall_allocation

# %% mystnb={"image": {"alt": "Stacked bars allocate three hypothetical collateral-loss scenarios across equity, mezzanine, and senior tranches of a MXN 100 million pool. The base stress reaches only equity, the severe stress reaches mezzanine, and the tail stress reaches senior."}}
waterfall_figure = build_securitization_waterfall_figure(
    waterfall_allocation,
    pool_notional=pool_notional,
    source_note=(
        "Hypothetical deterministic classroom stresses; MXN millions; "
        "attachments 0, 4, and 10; no observed collateral data."
    ),
)
waterfall_figure

# %% [markdown]
# The base stress allocates MXN 3 million entirely to equity. The severe stress
# exhausts equity at MXN 4 million and allocates MXN 5 million to mezzanine. The
# tail stress produces MXN 17.5 million of collateral loss: equity absorbs MXN 4
# million, mezzanine absorbs MXN 6 million, and senior absorbs MXN 7.5 million.
# Position, labels, and the table communicate the ordering without relying on
# color alone. The stepwise result is contractual arithmetic, not a forecast of
# loss likelihood or timing.

# %% [markdown]
# ## Prepayment and embedded options
#
# Borrowers can often prepay mortgages or other amortizing loans. Falling rates
# may accelerate prepayments, return principal when reinvestment yields are
# lower, and shorten duration (**contraction risk**). Rising rates may slow
# prepayments and extend duration when discount rates are higher (**extension
# risk**). The cash-flow timing is therefore endogenous to rates.
#
# A Z-spread discounts one promised cash-flow path and cannot by itself separate
# this option. OAS requires simulated rate paths, a prepayment rule, and a
# valuation model. The result is conditional on all three. Comparing OAS across
# products is defensible only when models, curves, volatility assumptions, and
# collateral conventions are comparable {cite}`fabozzi2019foundations`.

# %% [markdown]
# ## Applied review checklist
#
# Before recommending a credit or securitized instrument, document:
#
# - legal obligor, seniority, collateral, covenants, and governing waterfall;
# - currency, settlement, day-count, coupon, amortization, and call/prepayment
#   terms;
# - benchmark curve and whether the quoted measure is nominal spread, Z-spread,
#   or OAS;
# - PD horizon, LGD/recovery convention, EAD, migration assumption, and source;
# - liquidity, price source, valuation timestamp, and bid-ask evidence;
# - concentration and dependence within the issuer or collateral pool;
# - base, adverse, and severe loss allocation by tranche; and
# - model validation, exception approval, monitoring trigger, and owner.

# %% [markdown]
# ## Limitations
#
# - The expected-loss formula is a one-period mean and does not represent the
#   full loss distribution or time of default.
# - Yield spreads combine several risks and may use imperfect benchmarks.
# - Recovery can depend on seniority, collateral values, legal process, and the
#   economic cycle.
# - The deterministic waterfall examples omit fees, excess spread, triggers,
#   reserves, principal timing, servicing advances, and reinvestment.
# - Real securitized-product valuation requires loan-level data, dependence,
#   prepayment, delinquency, default, recovery, and interest-rate models.
# - Attachment and detachment arithmetic does not establish regulatory capital
#   treatment; contractual documents and the applicable framework must be
#   reviewed {cite}`baselFrameworkCreditSecuritisation2026`.
# - Ratings and modeled OAS are inputs to due diligence, not substitutes for it.

# %% [markdown]
# ## Handoff
#
# The curve-construction lessons now replace a flat benchmark yield with
# maturity-specific discount factors, parametric fits, and rate scenarios.
# Preserve the deterministic bond, duration, DV01, credit-spread, recovery, and
# waterfall assumptions from Lessons 6.1–6.5 as separate inputs; a fitted
# government curve does not absorb issuer or structure risk.
