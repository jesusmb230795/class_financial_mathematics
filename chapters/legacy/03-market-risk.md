# Legacy source: Market Risk

Status: source material for the market-risk block of canonical Module 7,
Derivatives and Risk Management. The `03-` filename is a legacy physical path
and is not the curriculum module number.

This module turns cleaned return series into market-risk estimates, model-validation checks, and stress narratives. The goal is not to find a single definitive loss number. The goal is to document how the answer changes when the analyst changes the tail probability, distributional assumption, lookback window, volatility model, or stress scenario {cite}`jorion2007var,mcneil2015quantitative`.

The executable notebooks use the versioned Banxico official price-like snapshot exposed through `src.market_data.official_price_panel`. That keeps the published book reproducible while preserving a clear path to live official APIs for local exploration {cite}`banxicoSIE2025`. Module 2 supplies the return, residual, and conditional-volatility prerequisites; this block adds portfolio loss aggregation, tail estimation, validation, and stress governance.

## Expected outcome

By the end of this module, students should be able to:

- calculate semideviation;
- estimate VaR with parametric and historical methods;
- apply Cornish-Fisher and volatility-weighted VaR adjustments;
- interpret CVaR or Expected Shortfall as a tail conditional average;
- validate VaR exceptions with Kupiec and Christoffersen coverage tests {cite}`kupiec1995techniques,christoffersen1998evaluating`;
- design deterministic stress scenarios;
- distinguish statistical tail estimates from liquidity-horizon and regulatory capital context;
- use an interactive VaR and CVaR dashboard;
- compare distributional assumptions;
- explain the practical limitations of each metric.

## Bridge from pricing and hedging to portfolio risk

Derivative valuation produces prices and sensitivities; market risk asks how
those positions change under observed or imposed risk-factor moves. Use this
handoff:

1. value positions under a declared market snapshot;
2. record delta, gamma, vega, duration, FX, or other relevant exposures;
3. map risk-factor changes into position profit and loss, including hedge P&L;
4. aggregate position P&L into portfolio return \(R_t\) and loss \(L_t=-R_t\);
5. estimate VaR and ES under the contract below;
6. backtest exceptions, compare stress loss, and document limit breaches,
   escalation ownership, liquidity, and model limitations.

VaR does not validate a pricing model, and a locally delta-neutral hedge does
not eliminate gap, basis, volatility, liquidity, or nonlinear risk.

## Measurement contract

Let $r_t$ be an asset-return vector, $w$ a portfolio-weight vector, and

$$
R_t=w^\top r_t,\qquad L_t=-R_t.
$$

For tail probability $\alpha$, define the raw loss quantile

$$
v_\alpha=Q_{1-\alpha}(L).
$$

This book reports risk as a non-negative loss. The implementation therefore
uses:

$$
\operatorname{VaR}_{\alpha}
=\max\left(0,-Q_\alpha(R)\right)
=\max(0,v_\alpha),
$$

$$
\operatorname{ES}_{\alpha}
=\max\left(0,-\mathbb{E}\left[R\mid R\leq Q_\alpha(R)\right]\right)
=\max\left(0,\mathbb{E}\left[L\mid L\geq v_\alpha\right]\right).
$$

For ordinary loss samples both values are positive without binding at zero.
The floor matters for samples whose complete left return tail is still a gain:
the reporting layer does not label an expected gain as a positive risk charge.
Every table must state whether returns are decimals or percentages.

For parametric one-step VaR with standardized return-innovation quantile
\(q_\alpha\):

$$
\operatorname{VaR}_{\alpha,t+1}
=\max\left\{0,-\left(\mu_{t+1}
+\sigma_{t+1}q_\alpha\right)\right\}.
$$

`arch` uses a unit-variance Student's t innovation. A SciPy t quantile must
therefore be standardized as
\(q_\alpha=t_\nu^{-1}(\alpha)\sqrt{(\nu-2)/\nu}\), with \(\nu>2\), before it is
multiplied by conditional volatility.

The backtesting notebooks define a one-day exception as

$$
I_t=\mathbf{1}\{-R_t>\widehat{\operatorname{VaR}}_{\alpha,t}\}.
$$

Stress testing asks a different question. Given a deterministic shock vector $s$, the portfolio stress loss is

$$
\operatorname{StressLoss}=-w^\top s.
$$

Cornish-Fisher is an approximation, not a license to extrapolate arbitrary
sample moments. The course helper reports skewness and excess kurtosis and
marks the adjustment unavailable outside its documented stability guardrail;
notebooks must not silently disable that check.

## Included notes

- VaR and Expected Shortfall Foundations;
- Downside Risk and VaR Methods;
- VaR Backtesting and Stress Testing;
- Interactive VaR and CVaR Dashboard.

## Reading sequence

1. Establish the sign convention: returns can be positive or negative, while reported risk metrics are non-negative losses.
2. Define VaR and Expected Shortfall with LaTeX formulas before looking at code.
3. Load the official Banxico snapshot and inspect the return sample used by the module.
4. Estimate historical VaR and Expected Shortfall from empirical returns.
5. Compare Gaussian, Cornish-Fisher, and volatility-weighted VaR on the same portfolio.
6. Use skewness and kurtosis to decide whether Gaussian assumptions are credible and whether Cornish-Fisher is available under the course guardrail.
7. Backtest VaR exceptions with coverage and independence tests.
8. Add deterministic stress scenarios and discuss liquidity-horizon regulatory context {cite}`basel2019marketRisk`.
9. Use the dashboard to compare tail probability, lookback window, and EWMA decay.
10. Document limitations, model risk, liquidity risk, and judgment.
