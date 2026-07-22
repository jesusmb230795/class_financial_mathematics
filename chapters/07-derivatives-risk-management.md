# Derivatives and Risk Management

This module integrates derivative payoff and pricing methods with hedging,
portfolio tail risk, validation, limits, and governance. A price is not a risk
decision: every model output must connect to an exposure, hedge objective,
residual risk, stress, limit, owner, and escalation rule
{cite}`hull2022options,blackScholes1973,merton1973,mcneil2015quantitative`.

Readers should first understand discount factors and rate sensitivity from
Module 6, return and volatility models from Module 2, and macro and
foreign-exchange (FX) scenarios from Module 3.

## Expected outcome

By the end of this module, readers should be able to:

- define linear and option payoffs with consistent position signs;
- apply no-arbitrage, replication, risk-neutral, tree, and Monte Carlo logic;
- calculate and interpret Delta, Gamma, Vega, Theta, Rho, duration, and dollar
  value of a basis point (DV01);
- distinguish implied volatility, smile/skew, stochastic volatility, and
  model-price error;
- estimate VaR and Expected Shortfall as non-negative loss magnitudes;
- backtest risk forecasts and supplement them with deterministic stress; and
- design a hedge and risk limit with observable governance evidence.

```{figure} ../img/generated/m7-option-payoff-greeks-map.png
:alt: Derivatives map linking payoffs, replication, pricing inputs, Greeks, implied volatility, hedge selection, residual profit and loss, limits, and governance.
:width: 900px
:align: center

Pricing intuition is carried through hedge selection, residual risk, stress,
limits, and escalation rather than ending at a model value.
```

## Conceptual spine

The intended sequence is:

```text
position and governance objective
→ payoff and no-arbitrage value
→ local and scenario sensitivities
→ hedge instrument and residual basis
→ nonlinear/volatility model risk
→ portfolio loss distribution
→ backtesting, stress, limits, and escalation
```

Option prices, Greeks, hedge P&L, returns, and losses require explicit signs and
units. For tail risk, the book defines loss as \(L=-R\) and reports VaR and
Expected Shortfall as non-negative loss magnitudes.

## Published lesson map

| Lesson | Main role |
| --- | --- |
| `7.1.linear_derivatives_and_carry` | Forwards, carry, mark-to-market, and par swaps |
| `7.2.european_option_pricing_and_greeks` | European option values, assumptions, and local sensitivities |
| `7.3.binomial_and_monte_carlo` | Tree and simulation pricing with convergence diagnostics |
| `7.4.implied_volatility_and_smiles` | No-arbitrage checks, implied-volatility inversion, and skew interpretation |
| `7.5.interactive_option_pricing_dashboard` | Static and interactive price/Greek sensitivity |
| `7.6.american_and_exotic_options` | Early exercise, Asian, and barrier methods |
| `7.7.stochastic_volatility_heston_lab` | Heston simulation and stochastic-volatility model risk |
| `7.8.hedging_risk_governance` | Hedge P&L, residual basis, limits, effectiveness, and escalation |
| `7.9.value_at_risk_foundations` | Canonical non-negative VaR and Expected Shortfall estimates |
| `7.10.downside_risk_var_methods` | Downside, distributional, and volatility-weighted risk methods |
| `7.11.var_backtesting_and_stress_testing` | Chronological exceptions, backtests, traffic-light context, and stress |
| `7.12.interactive_var_cvar_simulator` | VaR and Expected Shortfall assumption sensitivity with positive-loss labels |

The superseded derivatives and market-risk overviews are retained under
`chapters/legacy/` for provenance only. The `.py` files above are canonical
Jupytext sources; their `.ipynb` pairs are generated deterministically.

## Data and reproducibility contract

Observed portfolio-risk examples use a committed, versioned matrix of
provider-adjusted U.S. equity closing prices. They must state provider lineage,
instruments, field, currency, actual date range, trading calendar, return method,
annualization, and portfolio construction. Reproducibility does not establish
redistribution permission: the snapshot remains subject to an owner-authorized
rights review. Option chains, Heston paths, stress shocks, and some payoff
examples are synthetic and must preserve parameters and seeds. Interactive pages
require a meaningful static result when widgets are disabled
{cite}`yfinance2025,yahooFinanceCoverage2026,yahooTerms2026`.

Backtests must preserve time order and use forecasts formed only from prior
observations. A successful notebook execution does not validate a risk model:
signs, units, exceptions, output size, economic interpretation, and governance
remain separate gates. The 1996 Basel traffic-light procedure is presented as
historical backtesting context, not as a substitute for the current market-risk
framework {cite}`kupiec1995techniques,christoffersen1998evaluating,basel1996MarketRiskAmendment,basel2019marketRisk,baselFrameworkMAR33`.

## Evidence and reference map

The module anchors its principal models and diagnostics in primary sources:
Black--Scholes--Merton for European pricing, Cox--Ross--Rubinstein and
Leisen--Reimer for trees, Boyle for Monte Carlo pricing, Breeden--Litzenberger
for strike-convex call prices, Kemna--Vorst and Broadie--Glasserman--Kou for
path-dependent options, Heston and full-truncation research for stochastic
volatility, Sortino and Cornish--Fisher for downside and distributional
diagnostics, and the original backtesting and Expected Shortfall literature for
portfolio tail risk
{cite}`blackScholes1973,merton1973,coxRossRubinstein1979,leisenReimer1996,boyle1977,breedenLitzenberger1978,kemnaVorst1990,broadieGlassermanKou1997,heston1993,lordKoekkoekVanDijk2010fullTruncation,sortinoPrice1994,cornishFisher1938,kupiec1995techniques,christoffersen1998evaluating,acerbiTasche2002,rockafellarUryasev2002`.

## Reading sequence

1. Carry the overview's hedge objective, residual-risk, limit, and governance
   questions through every pricing page.
2. Price linear derivatives under explicit carry conventions.
3. Establish European option payoffs, Black-Scholes assumptions, and Greeks.
4. Compare trees and Monte Carlo simulation.
5. Infer volatility and interpret smiles without treating them as arbitrage-free
   surfaces by default.
6. Explore static and interactive sensitivities, then add American, exotic,
   and stochastic-volatility methods.
7. Convert sensitivities into hedge P&L, residual basis, limits, and escalation.
8. Aggregate portfolio returns into VaR and Expected Shortfall.
9. Compare tail methods, backtest forecasts, and apply stress scenarios.
10. Close with liquidity horizons, model risk, limit ownership, and escalation.

## Module limitations

The module does not constitute a complete counterparty-credit, xVA,
margin, collateral, legal, liquidity, or production model-governance framework.
Black-Scholes, trees, Monte Carlo, and VaR are teaching implementations. Their
assumptions and numerical limitations must travel with every promoted page.

## Handoff

Module 8 applies discounting, risk premia, scenario assumptions, and
uncertainty to asset valuation. Module 9 then combines instrument-level
exposures into an investment policy statement (IPS), allocation, and
performance process.
