# Module 8: Alternative Investments

Alternative investments are not a single asset class. They are combinations of
assets, legal vehicles, contractual cash flows, manager decisions, leverage,
and liquidity constraints that differ materially from a daily traded stock or
bond. A private-credit loan, a buyout fund, a listed real-estate trust, a
commodity future, and a market-neutral hedge fund therefore require different
valuation evidence even when they appear in the same portfolio allocation.
This layered view is consistent with valuation practice that begins by defining
the claim and its cash flows, and with institutional private-markets guidance
that separates fund terms, fees, governance, and reporting
{cite}`damodaran2012investment,ilpaPrinciples2019`.

This module follows the instrument, valuation, fixed-income, derivatives, and
risk foundations in earlier modules. It prepares Module 9 by translating
alternative holdings into cash-flow, exposure, liquidity, and benchmarking
inputs that can be incorporated into an investment policy and an asset
allocation. The governing principle is to analyze the **asset**, the
**vehicle**, and the **strategy** separately before making a portfolio claim.

## Learning objectives

By the end of this module, readers should be able to:

- distinguish private equity, venture capital, growth equity, buyouts, private
  debt, hedge funds, real assets, commodities, and digital assets by their
  economic cash flows rather than by labels;
- map fund commitments, capital calls, distributions, management fees, carried
  interest, waterfalls, and co-investments into investor-level cash flows;
- value a private company, direct real estate, and a listed real-estate vehicle
  using transparent assumptions and sensitivity ranges;
- explain NAV, NOI, cap rates, FFO, AFFO, roll return, contango,
  backwardation, leverage, gates, side pockets, and stale pricing;
- evaluate manager selection, operational due diligence, liquidity terms,
  factor exposures, and after-fee performance;
- construct an alternatives allocation subject to an explicit liquidity
  budget, concentration limits, and an evidence-based benchmark;
- identify the evidence needed to analyze Mexican FIBRAs, CKDs, and private
  capital vehicles without treating unavailable or estimated data as observed.

## Prerequisites

Readers should be able to:

- distinguish market, instrument, issuer, and provider data using
  [Market Foundations](../notebooks/course/1.1.stock_markets.md);
- document source, unit, frequency, calendar, vintage, and transformation using
  the [Market Data Quality Framework](../notebooks/course/1.6.market_data_quality_framework.py);
- connect macro assumptions to cash flows, discount rates, and risk premia
  using [Macro Scenarios and Capital Market Expectations](../notebooks/course/3.4.macro_scenarios_cme.md);
- interpret enterprise value, equity value, free cash flow, and scenario
  valuation from the corporate-analysis modules;
- distinguish return thresholds from positive-loss VaR and Expected Shortfall,
  and state the risk horizon and confidence level.

## Conceptual spine

### Analyze three layers

Every alternatives position is recorded in three linked layers:

| Layer | Central question | Examples |
| --- | --- | --- |
| Asset | What produces the economic cash flow? | company, property, toll road, crop, commodity exposure, loan |
| Vehicle | How does the investor own the exposure? | limited partnership, trust, listed security, separate account, swap, token |
| Strategy | What decisions change the exposure through time? | buyout, direct lending, long/short, trend following, development, roll schedule |

The vehicle can add fees, leverage, tax, governance, reporting, transfer, and
liquidity effects that are absent from the underlying asset. The strategy can
add timing, selection, hedging, financing, and operational risk. A comparison
that ignores either layer is incomplete.

```{figure} ../img/generated/m8-alternatives-asset-vehicle-strategy-map.png
:alt: Alternatives map separating the underlying asset, legal vehicle, and strategy before tracing investor cash flows, fees, carry, leverage, liquidity, due diligence, and portfolio role.
:width: 900px
:align: center

Investor-level cash flows and constraints emerge from the combination of
asset, vehicle, and strategy, not from the asset-class label alone.
```

### Cash-flow and value identity

For an investor with dated contributions \(C_t\), distributions \(D_t\), and a
reported residual net asset value \(NAV_T\), the cash-flow record is:

```{math}
CF_t = D_t-C_t,\qquad
CF_T = D_T-C_T+NAV_T.
```

With actual dates $d_i$, the annualized money-weighted return $x$ solves:

```{math}
0=\sum_{i=0}^{n}\frac{CF_i}{(1+x)^{\tau_i}},
\qquad
\tau_i=\frac{d_i-d_0}{365}.
```

This XIRR-style convention requires $x>-1$ and may have no unique solution
when cash-flow signs change more than once. The dates, day-count convention,
and root-selection rule are therefore part of the reported method.

Useful private-fund multiples are:

```{math}
DPI=\frac{\text{Cumulative distributions}}{\text{Paid-in capital}},
\qquad
RVPI=\frac{NAV_T}{\text{Paid-in capital}},
```

```{math}
TVPI=DPI+RVPI.
```

These multiples do not incorporate timing. An internal rate of return (IRR)
does incorporate timing, but it can be sensitive to subscription facilities,
intermediate cash flows, valuation policy, and the chosen measurement date.
Both money-weighted and multiple-based evidence should therefore be reported.
ILPA Principles guidance likewise distinguishes cash-flow
inputs, gross and net results, and subscription-facility effects rather than
treating one headline IRR as complete evidence {cite}`ilpaPrinciples2019`.

For a cash-flow-producing asset, value remains an expectation about future cash
flows discounted for time and risk:

```{math}
V_0=\sum_{t=1}^{T}\mathbb{E}[CF_t]P(0,t),
\qquad
P(0,t)=\prod_{j=1}^{t}\frac{1}{1+k_j},
```

where $k_j$ is the effective per-period required return and $P(0,t)$ is the
matching discount factor. Cash-flow scenarios and discount rates must not count
the same risk twice. Comparable-company, transaction, NAV, and cap-rate methods
are cross-checks, not substitutes for defining the cash-flow claim
{cite}`damodaran2012investment`.

### Liquidity is a state, not a label

Liquidity analysis records notice periods, settlement time, lockups, gates,
side pockets, transfer restrictions, financing capacity, market depth, and the
time required to exit without unacceptable price impact. The book reports
positive-loss risk measures. If \(VaR_\alpha\) is a non-negative loss magnitude
at lower-tail probability \(\alpha\), a simple liquidity adjustment is:

```{math}
LVaR_\alpha = VaR_\alpha + C_{\text{liquidation}},
```

where \(C_{\text{liquidation}}\) is an explicit spread, market-impact, funding,
and delay estimate over the liquidation horizon. This additive form is a
scenario convention, not a universal model; it must not double-count losses
already embedded in the return distribution {cite}`mcneil2015quantitative`.

## Lesson map

| Order | Page | Role | Evidence or output |
| --- | --- | --- | --- |
| 1 | [Private Capital Structures and Fund Economics](../notebooks/course/8.1.private_capital_structures_and_economics.md) | Separate asset economics from fund and waterfall economics | Investor cash-flow ledger and fee/carry calculation |
| 2 | [Private-Company Valuation and Due Diligence](../notebooks/course/8.2.private_company_valuation_and_due_diligence.md) | Connect valuation ranges to manager and deal evidence | Valuation bridge, sensitivity table, and diligence log |
| 3 | [Real Assets and Commodities](../notebooks/course/8.3.real_assets_and_commodities.md) | Analyze property, infrastructure, natural resources, and commodity exposures | NAV/cap-rate case and commodity roll calculation |
| 4 | [Hedge Funds, Digital Assets, and Liquidity Risk](../notebooks/course/8.4.hedge_funds_digital_assets_and_liquidity.md) | Translate strategy labels into leverage, factor, custody, and redemption risks | Exposure map and liquidity-adjusted loss scenario |
| 5 | [Portfolio Role and Mexican Alternatives Case](../notebooks/course/8.5.alternatives_portfolio_role_and_mexico_case.md) | Convert individual holdings into an allocation decision | Liquidity budget and investment-committee recommendation |

## Data and reproducibility contract

An alternatives analysis must preserve both observed and estimated inputs:

| Field | Minimum evidence |
| --- | --- |
| Position identity | asset, legal vehicle, strategy, currency, jurisdiction, ownership share |
| Measurement date | valuation date, cash-flow dates, reporting lag, retrieval date, snapshot vintage |
| Value status | traded price, manager NAV, appraisal, model estimate, transaction value, or scenario |
| Cash flows | gross and net contributions, distributions, income, fees, carry, taxes, and financing |
| Valuation | method, forecast horizon, compounding convention, discount rate, terminal assumption, comparable set |
| Liquidity | notice, lockup, gate, side pocket, settlement, transfer restriction, liquidation horizon |
| Risk | leverage definition, factor exposure, concentration, currency, counterparty, custody, model limitation |
| Rights and provenance | provider, document or series identifier, license or access note, revision policy |

Rates and returns are stored as decimals and displayed with an explicit `%`.
Currencies and units appear beside every value. Appraisals, manager marks, and
constructed indexes must never be labeled as market prices. A data room or
manager report is not automatically redistributable; only permitted snapshots
belong in the repository.

For Mexican examples, use the controlling security or trust documents and
auditable exchange or issuer disclosures. Grupo BMV describes FIBRAs and CKDs
as distinct listed-market instruments {cite}`bmvFibras,bmvCkds`; the analyst
must still verify each vehicle's current terms, distributions, valuation date,
and data rights. If private-vehicle cash flows are unavailable, use a clearly
labeled simulation rather than infer them from a public index.

## Reading sequence

1. Begin with fund and investor cash flows so gross asset performance is not
   confused with net investor performance.
2. Establish valuation and due-diligence evidence before comparing managers.
3. Add real assets and commodities, keeping physical cash flow separate from
   derivative roll mechanics.
4. Translate hedge-fund and digital-asset labels into actual exposures,
   leverage, custody, and redemption terms.
5. End with a portfolio case constrained by liquidity, concentration, and
   governance.

## Sources and further reading

The module uses sources by analytical purpose rather than as interchangeable
endorsements:

- Damodaran supplies the cash-flow, claim, and valuation-method foundation
  {cite}`damodaran2012investment`.
- ILPA Principles 3.0 supplies an institutional review framework for LP/GP
  alignment, waterfall terms, fees, subscription facilities, governance, and
  reporting {cite}`ilpaPrinciples2019`.
- The SEC risk alert documents observed due-diligence practices and failures;
  it is supervisory evidence, not a safe-harbor checklist
  {cite}`secAlternativeDueDiligence2014`.
- Nareit supplies the industry definition and implementation guidance for FFO;
  AFFO remains issuer- or analyst-defined {cite}`nareitFfo2002`.
- Grupo BMV supplies public instrument descriptions for Mexican FIBRAs and
  CKDs; each prospectus, trust agreement, and disclosure controls the specific
  investment {cite}`bmvFibras,bmvCkds`.
- Hull, McNeil, Frey, Embrechts, Jorion, Sharpe, and Lo support the derivatives,
  positive-loss risk, performance, and serial-correlation cautions used in the
  lessons {cite}`hull2022options,mcneil2015quantitative,jorion2007var,sharpe1994ratio,lo2002sharpe`.
- The BIS chapter supplies the tokenization framework and its governance,
  settlement, and legal-claim limitations {cite}`bisTokenisation2023`.

## Module practice

Create an alternatives position card for one real or simulated holding. It
must name the asset, vehicle, strategy, cash-flow pattern, fee layer, valuation
method, valuation date, liquidity terms, currency, leverage definition,
benchmark, and three evidence limitations. Then state one portfolio role that
the evidence supports and one role it does not support.

```{dropdown} Review standard
A complete card keeps asset, vehicle, and strategy separate; identifies whether
each value is observed, reported, appraised, or modeled; reports gross and net
cash flows in one currency and unit; states a liquidation horizon; and links
the portfolio claim to a measurable exposure. "Diversification" without a
factor, cash-flow, scenario, or liquidity explanation is not sufficient.
```

## Handoff

Start with [Private Capital Structures and Fund Economics](../notebooks/course/8.1.private_capital_structures_and_economics.md).
The final alternatives case then hands a cash-flow schedule, valuation range,
exposure map, and liquidity budget to Module 9 for portfolio construction and
performance evaluation.
