# Market Risk

This module introduces quantitative metrics for estimating potential losses in assets and portfolios.

## Expected outcome

By the end of this module, students should be able to:

- calculate semideviation;
- estimate VaR with parametric and historical methods;
- apply Cornish-Fisher and volatility-weighted VaR adjustments;
- interpret CVaR or Expected Shortfall;
- validate VaR exceptions with Kupiec and Christoffersen tests;
- design deterministic stress scenarios;
- use an interactive VaR and CVaR simulator;
- compare distributional assumptions;
- explain the practical limitations of each metric.

## Included notes

- Value at Risk and Conditional Value at Risk;
- Downside Risk and VaR Methods;
- VaR Backtesting and Stress Testing;
- Interactive VaR and CVaR Simulator.

## Recommended next improvements

- integrate Monte Carlo simulation;
- expand FRTB liquidity-horizon examples;
- connect the module with the mean-variance portfolio.

## Class sequence

1. Define downside risk and semideviation.
2. Estimate historical VaR and CVaR from empirical returns.
3. Estimate Gaussian VaR and compare assumptions against empirical data.
4. Use skewness and kurtosis to motivate Cornish-Fisher modified VaR.
5. Add EWMA volatility weighting to make historical simulation more adaptive.
6. Backtest VaR exceptions with coverage and independence tests.
7. Add deterministic stress scenarios and discuss regulatory context.
8. Use the VaR/CVaR simulator to compare tail regimes.
9. Discuss limitations, model risk, liquidity risk, and judgment.

## In-class practice

Students should calculate downside risk, VaR, and Expected Shortfall for one asset and one portfolio, then backtest a rolling VaR model, document one deterministic stress scenario, and compare two simulator parameter settings.

## Module checkpoint

The checkpoint is a risk memo that states the confidence level, method, estimated loss, exception behavior, stress scenario, assumptions, and limitations for each risk metric.
