# Economics, Macro, and Currency

Module 3 turns macroeconomic context into an investment input. Module 1 introduced market data and exploratory dashboards; this module makes the economic mechanism explicit: macro variables affect expected cash flows, discount rates, risk premia, currency returns, and investor positioning {cite}`mishkin2019financial,damodaran2012investment`.

The module combines four concise conceptual lessons with one reproducible
macro-to-FX case. The conceptual pages provide hand-checkable calculations and
concrete assessments; the executable case then reuses those contracts with a
versioned real-data panel, an availability lag for inflation, a chronological
holdout, explicit benchmarks, and a model-rejection decision gate.

```{figure} ../img/generated/m3-macro-transmission-currency-map.png
:alt: Flow from observed macro releases and vintages through scenario assumptions and four valuation channels to an investment decision and monitoring rule.
:width: 960px
:name: module-3-macro-transmission-map

Macro evidence becomes decision-relevant only after its timing and mechanism
have been mapped to cash flows, rates, risk premia, or currency conversion.
```

## Expected outcome

By the end of this module, readers should be able to:

- explain how supply, demand, elasticity, market structure, and price controls affect economic interpretation;
- read GDP, inflation, employment, productivity, output gaps, policy rates, yield curves, fiscal balances, public debt, and financial conditions;
- distinguish monetary-policy and fiscal-policy channels;
- quote exchange rates and cross-rates without reversing base and quote currencies;
- evaluate purchasing power parity, covered interest parity, uncovered interest parity, forward premium or discount, carry, and currency-crisis risk;
- translate macro scenarios into capital market expectations for equities, fixed income, FX, commodities, and real estate;
- document macro dashboards with source, frequency, revision, unit, and limitation notes.

## Prerequisites

Students should complete [Module 1](01-markets-data.md) before the conceptual
lessons. The required numerical tools are percentage changes,
decimal-versus-percent discipline, one-period discounting, and careful source
metadata. The executable case also requires the log-return, regression,
regularization, MAE, benchmark, and chronological-validation foundations from
[Module 2](02-time-series.md). Yield-curve construction, derivative pricing,
and portfolio optimization are later applications, not prerequisites here.

## Investment macro contract

A macro view becomes useful only when it changes at least one investment input:

```{math}
P_0 \approx \sum_{t=1}^{T}
\frac{\mathbb{E}[CF_t \mid M]}{(1+k_t(M))^t},
\qquad
k_t(M)=r_t(M)+\lambda_t(M),
```

where \(M\) is the macro state, \(CF_t\) is the cash flow expected at the end of
period \(t\), and \(k_t\) is the effective spot required return per period for
maturity \(t\). The period unit and rate convention must match—for example,
annual effective rates with \(t\) measured in years. The return is decomposed here
into a maturity-matched reference rate \(r_t\) and an incremental risk premium
\(\lambda_t\) for uncertainty, liquidity, credit, duration, currency, or policy
risk. A path of one-period rates would instead require a product of period-by-
period discount factors.

This framing keeps the module concise. The analyst should not list macro facts for their own sake. Each fact should connect to expected cash flows, rates, risk premia, currency conversion, or portfolio positioning.

## Data contract

Module 3 uses the same reproducibility standard as earlier modules. Publication
examples should prefer versioned snapshots from official or stable public
providers, while live notebooks should use the shared provider/cache layer in
`src/`.

| Data family | Examples | Preferred classroom sources |
| --- | --- | --- |
| Mexico macro and rates | inflation, policy rate, TIIE, CETES, UDI, USD/MXN FIX | Banxico SIE and INEGI where available {cite}`banxicoSIE2025,inegiAPI2025` |
| Global macro | GDP, inflation, employment, productivity, trade, debt | DB.NOMICS, World Bank, IMF WEO, and OECD {cite}`dbnomics2025,worldBankOpenData2025,imfWEO2025,oecdData2025` |
| Currency and rates | spot FX, forwards, local and foreign interest rates | official central bank data, exchange feeds, or reviewed public market sources |
| Market response | equity indices, yields, credit spreads, commodities, real estate vehicles | versioned market snapshots or approved provider-backed panels |

The applied case uses a frozen latest-vintage panel generated on 2026-06-07
with observations through 2025-06-30. Its monthly labels are period ends, not
historical release timestamps. A one-month inflation lag is therefore imposed
as a conservative availability proxy. That prevents the known same-month CPI
look-ahead in the teaching design, but it does not turn the snapshot into a
real-time-vintage database.

## Lesson map

| Page | Role | Main output |
| --- | --- | --- |
| [Economic Foundations for Investors](../notebooks/course/3.1.economic_foundations.md) | Builds the microeconomic language behind market interpretation. | Midpoint elasticity, revenue effect, contribution margin, and identification limits. |
| [Macro Indicators and Policy](../notebooks/course/3.2.macro_indicators_policy.md) | Connects macro aggregates and policy to rates, growth, inflation, fiscal sustainability, and risk appetite. | Indicator classification, real-rate sensitivity, and debt arithmetic. |
| [Currency, Parity, and FX](../notebooks/course/3.3.currency_parity_fx.md) | Establishes exchange-rate conventions, parity relationships, carry, and external-stress logic. | Unit-safe cross-rate, PPP, CIP, and unhedged-carry calculations. |
| [Macro Scenarios and Capital Market Expectations](../notebooks/course/3.4.macro_scenarios_cme.md) | Converts macro states into governed, horizon-specific asset assumptions. | Quantified valuation bridge, CME contract, and vintage-aware monitoring requirements. |
| [Applied Macro-to-FX Model Review Case](../notebooks/course/3.5.applied_macro_fx_case.ipynb) | Audits a small predictive workflow on a versioned Mexico data panel. | Availability-aware features, two naive benchmarks, explicit rejection gate, and scenario-stability diagnostic. |

## Reading sequence

1. Start with supply, demand, elasticity, and market structure so price changes have an economic interpretation.
2. Move to GDP, inflation, employment, productivity, business cycles, monetary policy, fiscal policy, and financial conditions.
3. Add exchange-rate mechanics, cross-rates, parity conditions, carry trades, and balance-of-payments pressure.
4. Convert macro views into scenarios that state assumptions, transmission channels, affected assets, uncertainty, and monitoring indicators.
5. Complete the applied macro-to-FX case using the committed monthly panel,
   declared availability assumption, chronological holdout, and two benchmark
   comparisons. Treat model rejection as a valid analytical result.
6. Extend the case into a dashboard only after source, unit, frequency,
   revision policy, and limitations remain explicit.

Each lesson ends with an assessment that requires a numerical output, a
mechanism statement, and one limitation. Completion means producing those
three forms of evidence, not only repeating definitions.

## Scope boundaries

This module does not replace later valuation, fixed-income, derivatives, or portfolio modules. It supplies the economic inputs those modules need:

| Topic introduced here | Later development |
| --- | --- |
| Discount-rate and risk-premium channels | Equity valuation, fixed income, and portfolio management |
| Yield curves and policy-rate paths | Fixed income and term-structure modeling |
| FX forwards, currency hedging, and overlays | Derivatives and portfolio management |
| Stress scenarios and macro surprises | Risk management and investment committee cases |
| Capital market expectations | Strategic and tactical asset allocation |
