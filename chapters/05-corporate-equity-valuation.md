# Module 5: Corporate Issuers and Equity Valuation

Corporate finance asks how a company should invest, fund operations, manage
liquidity, and return capital. Equity valuation asks what the resulting cash
flows and risks imply for the value of the operating business and the residual
claim held by common shareholders. These are one connected problem: investment
decisions affect growth and ROIC, financing affects risk and the allocation of
cash flows, and competitive conditions affect how long excess returns can
persist. Governance defines the decision rights, disclosure, oversight, and
accountability around those choices
{cite}`oecd2023CorporateGovernance,jensenMeckling1976TheoryFirm`.

This module begins with projects and governance, then develops capital
structure, operating forecasts, valuation frameworks, and a decision-ready
equity memo. It uses the normalized statements and integrated model from Module
4. Market price remains an observation; intrinsic value is an estimate
conditional on assumptions {cite}`damodaran2012investment,fama1970EfficientMarkets`.
The lessons use separate, clearly labeled simulations to isolate each method;
the final Northstar case recombines the controls but is not the filing-backed
issuer case still identified in the editorial plan.

## Learning objectives

By the end of this module, readers should be able to:

- connect governance, stakeholders, working capital, investment, financing,
  and payout decisions;
- calculate and interpret NPV, IRR, payback, profitability index, and
  incremental ROIC;
- explain real options and identify when managerial flexibility has value;
- estimate cost of equity, after-tax cost of debt, WACC, and a target capital
  structure using consistent market-value weights;
- translate industry structure, business model, pricing power, and unit
  economics into a driver-based forecast;
- explain how market efficiency, behavioral biases, and anomalies constrain
  valuation confidence;
- value equity using dividend discount, FCFF, FCFE, residual-income,
  market-multiple, and sum-of-the-parts frameworks;
- construct an evidence-based investment thesis with catalysts, risks,
  monitoring indicators, scenario values, and a review date.

## Prerequisites

Readers should complete:

- [Financial Statement Analysis and Financial Modeling](04-financial-statements-modeling.md),
  especially the normalized cash-flow and three-statement model;
- [Market Foundations](../notebooks/course/1.1.stock_markets.md) for the
  difference between observable price and estimated value;
- [Currency, Parity, and FX](../notebooks/course/3.3.currency_parity_fx.md) when
  cash flows, debt, or discount rates span currencies.

Readers should be able to discount a cash flow, distinguish enterprise value
from equity value, and keep rates as decimals in calculations.

## Conceptual spine

### Value creation

For a project with initial outlay $I_0$, incremental after-tax cash flow
$CF_t$, required return $r$, and horizon $N$:

```{math}
\text{NPV}
=-I_0+\sum_{t=1}^{N}\frac{CF_t}{(1+r)^t}.
```

A positive NPV means the project is expected to earn more than the required
return under the stated cash-flow, timing, and risk assumptions. At the
company level, growth creates value when incremental return on invested capital
exceeds the risk-consistent cost of capital for long enough to offset required
reinvestment {cite}`damodaran2012investment,myers1974FinancingInvestment`.

### Required return

For a company financed with market values $E$ of equity and $D$ of
interest-bearing debt:

```{math}
\mathrm{WACC}
=\frac{E}{D+E}k_e
+\frac{D}{D+E}k_d(1-\tau_{\mathrm{shield}}),
```

where $k_e$ is cost of equity, $k_d$ is marginal pretax cost of debt, and
$\tau_{\mathrm{shield}}$ is the marginal tax rate expected to produce a usable
interest deduction. WACC is not a universal company hurdle rate; project risk,
currency, duration, and financing context must match the cash flow
{cite}`modiglianiMiller1958CostCapital,hamada1972CapitalStructure`.

### Enterprise and equity claims

Free cash flow to the firm (FCFF) is available to all capital providers and is
discounted at WACC to estimate enterprise value. Free cash flow to equity
(FCFE) is available to common equity after debt cash flows and is discounted at
cost of equity {cite}`damodaran2012investment`. With constant discount rates,
an explicit horizon $N$, and terminal values at $N$:

```{math}
\text{Enterprise value}
=\sum_{t=1}^{N}\frac{\mathrm{FCFF}_t}{(1+\mathrm{WACC})^t}
+\frac{TV_N^{\mathrm{FCFF}}}{(1+\mathrm{WACC})^N},
```

```{math}
\text{Equity value}
=\sum_{t=1}^{N}\frac{\mathrm{FCFE}_t}{(1+k_e)^t}
+\frac{TV_N^{\mathrm{FCFE}}}{(1+k_e)^N}.
```

Unless a lesson states otherwise, annual cash flows occur at year-end and rates
are effective annual rates. A nominal rate is inflation-inclusive; it is not an
unconverted quoted annual percentage rate.

A simplified bridge is:

```{math}
\text{Common equity value}
=\text{Enterprise value}
-\text{Debt and debt-like claims}
-\text{NCI}
+\text{Excess cash and non-operating assets}.
```

Here NCI means non-controlling interests. Every bridge item needs the same
valuation date, currency, and perimeter.
Residual-income and dividend methods provide equity-claim cross-checks under
their own accounting and payout assumptions
{cite}`gordonShapiro1956CapitalEquipment,ohlson1995EarningsBookValues`.

```{figure} ../img/generated/m5-corporate-value-creation-valuation-map.png
:alt: Corporate valuation map from capital allocation and operating forecasts through FCFF or FCFE and matched discount rates to enterprise value, equity value, thesis, and monitoring.
:width: 900px
:align: center

Cash-flow claim and discount rate must match before enterprise and equity values
can support a monitored investment thesis.
```

## Lesson map

| Order | Page | Role | Evidence or output |
| --- | --- | --- | --- |
| 1 | [Corporate Decisions and Capital Budgeting](../notebooks/course/5.1.corporate_decisions_and_capital_budgeting.md) | Connect governance and capital allocation to project cash flow | NPV decision and capital-allocation map |
| 2 | [Capital Structure and WACC](../notebooks/course/5.2.capital_structure_and_wacc.md) | Estimate financing risk and required return | WACC build and target-leverage case |
| 3 | [Business Model and Forecasting](../notebooks/course/5.3.business_model_and_forecasting.md) | Convert competitive economics into operating drivers | Unit-economic forecast and scenarios |
| 4 | [Equity Valuation Frameworks](../notebooks/course/5.4.equity_valuation_frameworks.md) | Triangulate intrinsic and relative value | DCF, residual-income, multiples, and SOTP bridge |
| 5 | [Investment Thesis and Valuation Memo](../notebooks/course/5.5.investment_thesis_and_valuation_memo.md) | Turn analysis into a monitored decision record | Research note with catalysts, risks, and indicators |

## Data and reproducibility contract

Every valuation should state:

- valuation date, market-price timestamp, share class, diluted share count, and
  reporting currency;
- historical filing vintage, forecast start date, fiscal calendar, and model
  perimeter;
- whether cash flows and rates are nominal or real and before or after tax;
- source and date for risk-free rate, equity risk premium, beta, credit spread,
  debt value, tax rate, and target capital structure;
- treatment of leases, pensions, preferred claims, NCI, associates,
  non-operating assets, options, restricted cash, and contingent consideration;
- forecast scenario, explicit horizon, terminal method, and sensitivity range;
- peer-selection rule, metric definition, market-data timestamp, and outlier
  policy;
- formula, source, owner, and review status for every material assumption.

The revised IFRS management-commentary framework is a useful source for
connecting strategy, resources, risks, performance, and cash-flow prospects
{cite}`ifrs2025ManagementCommentary`. For model governance, the inventory,
independent challenge, validation, and monitoring principles in the 2026 U.S.
interagency guidance are a high-discipline benchmark; that guidance formally
applies to banking organizations and is not a regulatory requirement for this
educational valuation workflow {cite}`fed2026ModelRiskGuidance`.

Observed market values can change continuously while filings update
periodically. Preserve each valuation as an as-of snapshot; do not combine a
current price with stale debt, cash, shares, or peer values without an explicit
bridge.

## Reading sequence

1. Evaluate investments and capital-allocation governance.
2. Estimate financing costs and target capital structure.
3. Forecast operating economics from competitive and unit drivers.
4. Value the claims with multiple internally consistent methods.
5. End with a thesis, scenario range, catalysts, risks, and monitoring plan.

## Module practice

For a public issuer, write a value-driver tree with four levels:

1. operating driver;
2. financial-statement line;
3. cash-flow effect;
4. valuation effect.

Include at least one growth driver, one margin driver, one reinvestment driver,
and one financing driver. Mark which quantities are reported facts, analyst
estimates, and market observations.

```{dropdown} Review standard
A complete tree traces a driver such as customer retention to revenue, operating
profit, working capital or reinvestment, free cash flow, and value. It also
identifies the source date and avoids treating a valuation multiple or share
price as an operating cause. The labels fact, estimate, and observation must be
visible.
```

## Handoff

Begin with [Corporate Decisions and Capital Budgeting](../notebooks/course/5.1.corporate_decisions_and_capital_budgeting.md).
The lesson establishes the incremental cash-flow and governance discipline used
throughout the module.
