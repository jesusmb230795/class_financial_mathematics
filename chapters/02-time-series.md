# Quantitative Methods and Financial Time Series

This module establishes the quantitative conventions used throughout the book
and then applies them to financial time series. The sequence moves from rate
and return measurement through reproducible simulation, inference, and
chronological validation before treating stationarity, conditional-mean models,
volatility dynamics, and one-step tail risk
{cite}`box2015time,hamilton1994time,tsay2010analysis`.

The foundations lesson is deliberately compact. It introduces the methods that
later modules assume without turning Bayesian methods, clustering, or broader
machine learning into hidden prerequisites.

```{figure} ../img/generated/m2-price-return-volume-volatility-bridge.png
:alt: Two-lane bridge. The modeling spine runs from provider-dated price levels through return transformation, diagnostics, conditional models, and forecast or risk. A separate context lane routes volume and liquidity only to interpretation and limitations.
:width: 900px
:align: center
:name: module-2-time-series-bridge

The module separates observed price and activity context from return
transformations, diagnostics, and conditional-volatility models. Volume and
liquidity support interpretation; they are not automatic model inputs.
```

## Prerequisites and conventions

Complete the Module 1 market-data quality path first. Readers should already be
able to identify a provider, series, observation date, unit, frequency,
transformation, and missing-data policy.

The module follows the repository notation contract:

- rates and returns are stored as decimals unless a table or model explicitly
  labels percentage units;
- $R_t=P_t/P_{t-1}-1$ denotes a simple return, while
  $g_t=\log(P_t/P_{t-1})$ denotes a log return; model cells that use percentage
  points write $g_t^{(\%)}=100g_t$ explicitly;
- annualized log return is converted with `expm1` before it is called an
  annualized simple-return equivalent;
- $A=252$ FIX publication intervals per year is the reporting convention in
  this module, so displayed annualized volatility is $\sqrt{A}$ times
  per-interval volatility;
- `GARCH(p, q)` follows the `arch` convention: $p$ counts squared-shock lags and
  $q$ counts conditional-variance lags; and
- VaR is a non-negative loss magnitude for a stated position, with tail
  probability $\alpha_{\mathrm{tail}}$ and confidence
  $1-\alpha_{\mathrm{tail}}$. This tail symbol is distinct from the GARCH shock
  coefficient $\alpha$.

## Data and evidence boundary

Quantitative Foundations uses seeded synthetic data. Its paths illustrate known
assumptions and make no empirical security claim.

The empirical lessons use the committed Banxico SIE snapshot for series
SF43718, the FIX exchange rate quoted as MXN per USD
{cite}`banxicoSIE2025`. They calculate changes only between dates on which a FIX
observation was actually published. The business-day-aligned classroom panel
and its forward-filled rows are not used to estimate ADF, ARIMA, ARCH, GARCH, or
VaR results.

For the current snapshot, the requested window is 2021-01-01 through
2026-06-05 and the observed FIX sample begins on 2021-01-04. A return is a log
change from one published FIX to the next; that interval can cross a weekend or
holiday. FIX is an official reference fixing, not a guaranteed executable
transaction price. Every executable lesson prints its actual sample, vintage,
unit, and calendar-gap evidence so a refreshed snapshot cannot silently inherit
old claims.

## Published outcomes

After completing the module, readers should be able to:

- accumulate and discount cash flows under simple, periodic, and continuous
  rate conventions;
- distinguish price or level, first difference, simple return, and log return;
- reproduce a seeded simulation while distinguishing assumptions from observed
  evidence;
- estimate regression coefficients with uncertainty and construct lagged
  features without look-ahead;
- compare a predictive pipeline with a chronological benchmark;
- formulate weak stationarity and interpret ADF, ACF, and PACF evidence;
- distinguish raw-series screens from fitted-residual diagnostics;
- select an admissible ARIMA candidate with a declared criterion and evaluate
  it in a later retrospective block;
- distinguish ARCH shock memory from GARCH variance recursion;
- diagnose and compare Gaussian and standardized Student-t GARCH
  specifications while flagging adequacy failures;
- calculate persistence, long-run variance, long-run volatility, and theoretical
  variance-shock half-life; and
- report one-step VaR as a non-negative loss under an explicit position, tail
  probability, confidence level, horizon, and return unit.

## Lesson ownership and evidence

Stable lesson identifiers are not numerically consecutive in the preferred
reading order. The production TOC follows the conceptual sequence below.

| Lesson | Primary role | Evidence or output |
| --- | --- | --- |
| [Quantitative Foundations](../notebooks/course/2.0.quantitative_foundations.ipynb) | Introduces rates, returns, probability, inference, regression, and no-look-ahead validation. | Seeded simulation, uncertainty interval, chronological folds, model-versus-benchmark evaluation. |
| [Financial Time Series: Levels, Returns, and White Noise](../notebooks/course/2.1.time_series_1.ipynb) | Transforms observed FIX levels and distinguishes mean dependence from variance dependence. | Provider-dated sample audit, level/return diagnostics, constant-mean residual and squared-residual evidence. |
| [Time Series Pre-model Diagnostics](../notebooks/course/2.3.time_series_diagnostics_and_volatility_extensions.ipynb) | Develops the decision process used before selecting a mean or variance specification. | ADF, 95% ACF/PACF, raw-return Ljung-Box, Jarque-Bera, and ARCH-LM. |
| [ARIMA Selection and Diagnostic Workflow](../notebooks/course/2.4.arima_diagnostic_workflow.ipynb) | Covers conditional-mean selection, chronological evaluation, and residual diagnostics. | Training-only grid, AIC rule, BIC comparison, later evaluation block, residual tests. |
| [ARCH and GARCH Model Comparison](../notebooks/course/2.2.time_series_2.ipynb) | Isolates shock-only versus recursive variance dynamics. | Convergence checks, information criteria, persistence measures, common-scale volatility paths. |
| [GARCH Volatility, Diagnostics, and One-Step Risk](../notebooks/course/2.5.garch_volatility_risk_workflow.ipynb) | Connects innovation-distribution comparison with the volatility-to-VaR bridge. | Standardized-residual tests, Gaussian versus Student-t fits, exact log-to-simple loss conversion, and VaR decomposition. |
| [Interactive GARCH Persistence Dashboard](../notebooks/course/2.6.interactive_volatility_garch_dashboard.ipynb) | Explores transparent parameter sensitivity without calling it estimation. | Stationary slider parameterization, shared initial variance, theoretical variance-shock half-life. |

## Assessment and interpretation discipline

Each lesson closes with a reproducible handoff to the next analytical step.
A complete response reports the evidence, threshold, unit, actual sample, and
limitation. It must not turn failure to reject into proof, in-sample information
criteria into forecast superiority, association into causality, or a slider
scenario into a calibrated model.

The current sequence is substantial for its declared scope. Optional Bayesian,
clustering, and broader machine-learning extensions must preserve time order,
use an interpretable benchmark, and avoid duplicating the portfolio module.
Module 7 then owns VaR backtesting, exception independence, and stress testing;
Module 9 owns portfolio-level performance and allocation decisions.
