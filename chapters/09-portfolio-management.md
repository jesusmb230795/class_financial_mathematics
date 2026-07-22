# Portfolio Management, Asset Allocation, and Performance

This module turns return, covariance, risk, and valuation inputs into a
governed portfolio process. The Investment Policy Statement (IPS) comes first;
optimization, implementation, monitoring, and performance evaluation follow.
The objective is a portfolio that can be explained, funded, rebalanced, and
reviewed—not merely one that is optimal for one estimated matrix
{cite}`damodaran2012investment,sharpe1994ratio`.

Readers should complete the quantitative, macro, risk, and valuation modules
before treating expected returns, covariance, and scenarios as allocation
inputs.

## Expected outcome

By the end of this module, readers should be able to:

- translate purpose, liabilities, horizon, liquidity, constraints, and
  governance into an IPS;
- construct and interpret minimum-variance, efficient, tangency, shrinkage,
  risk-parity, and hierarchical benchmark portfolios;
- separate strategic allocation from governed tactical views;
- include concentration, liquidity, turnover, fees, taxes, and stress in
  implementation;
- calculate benchmark-relative return and simplified attribution; and
- monitor allocation, risk, performance, assumptions, and exceptions.

## Conceptual spine

The module follows one decision chain:

```text
investor purpose and constraints
→ return objective and risk budget
→ capital market expectations
→ strategic allocation and ranges
→ portfolio construction
→ implementation and rebalancing
→ performance and attribution
→ monitoring, exceptions, and IPS review
```

Expected return is not promised return. Covariance is estimated, not known.
Risk capacity and risk tolerance are distinct. Performance must state period,
currency, benchmark, cash-flow treatment, and gross/net convention.

```{figure} ../img/generated/m9-efficient-frontier-risk-budget-map.png
:alt: Governed portfolio lifecycle from the investment policy statement and risk budget through expectations, allocation, construction, implementation, performance, monitoring, exceptions, and policy review.
:width: 900px
:align: center

The efficient frontier is one construction tool inside a governed lifecycle;
the IPS, implementation, performance evidence, and review loop remain primary.
```

## Published lesson map

| Lesson | Main role |
| --- | --- |
| `9.1.ips_asset_allocation_performance` | IPS, strategic allocation, rebalancing, performance, attribution, and governance |
| `9.2.analytical_efficient_frontier` | Merton constants, global minimum variance, analytical frontier, and tangency portfolio |
| `9.3.robust_portfolio_construction` | Shrinkage, factor sensitivity, risk parity, and hierarchical benchmark allocations |
| `9.4.interactive_efficient_frontier_dashboard` | Correlation, risk-free-rate, PSD, and frontier assumption sensitivity |

The superseded portfolio overview and monolithic MPT notebook remain under
`chapters/legacy/` and `notebooks/legacy/` for provenance only. Canonical edits
belong in the scoped Module 9 sources above, followed by deterministic Jupytext
synchronization.

## Data and reproducibility contract

Observed portfolio examples must identify provider, instruments or constructed
series, actual sample boundaries, frequency, annualization factor, currency,
missing-data policy, and snapshot vintage. Synthetic expected-return or
covariance examples must be labeled as assumptions. Optimization output must
record:

- estimator and estimation window;
- constraints, bounds, target, and risk-free rate;
- solver or closed-form method and diagnostics;
- transaction-cost and turnover treatment; and
- sensitivity to plausible input changes.

A committed snapshot supports reproducibility but does not establish
redistribution rights. Portfolio weights from classroom data are not investment
recommendations.

## Reading sequence

1. Define the IPS, nominal/real objective, risk capacity, constraints, and
   governance.
2. Translate Module 3 scenarios and Module 8 valuation assumptions into capital
   market expectations.
3. Review return, volatility, covariance, diversification, and feasible
   portfolios.
4. Derive the global minimum-variance and analytical efficient frontier.
5. Add the risk-free asset, Sharpe ratio, and tangency interpretation.
6. Test shrinkage, risk parity, and hierarchical benchmark allocations.
7. Apply concentration, liquidity, turnover, cost, and stress constraints.
8. Use the dashboard to test input sensitivity rather than select a single
   "best" weight vector.
9. Rebalance under explicit ranges and evaluate time-weighted, money-weighted,
   benchmark-relative, and attributed performance.
10. Monitor assumptions, exposures, outcomes, and exceptions against the IPS.

## Module limitations

The module does not include a full liability model, tax-lot optimization,
multi-period allocation, currency overlay, derivatives collateral process,
production transaction-cost model, and complete performance-attribution engine.
The legacy notebook is long and internally multi-purpose and is not an
alternative published route.

## Handoff

Module 10 can apply this governed portfolio process to alternatives,
professional standards, and integrative cases while preserving liquidity,
valuation, benchmark, and reporting limitations.
