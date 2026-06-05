# Assessment and Rubrics

This page defines the course assessment structure, module checkpoints, dashboard work, and final project requirements.

## Assessment structure

| Component | Weight | Evidence |
| --- | ---: | --- |
| Module checkpoints | 35% | Short technical memos or reproducible notebooks |
| Dashboard labs | 20% | Interactive notebook outputs with interpretation |
| Final project | 35% | End-to-end quantitative finance notebook and presentation |
| Reproducibility and participation | 10% | Environment checks, version-control hygiene, and in-class work |

## General rubric

| Criterion | Excellent | Satisfactory | Needs improvement |
| --- | --- | --- | --- |
| Technical correctness | Methods, formulas, and code are correct and clearly connected. | Main method is correct with minor gaps or unclear assumptions. | Method is incorrect, incomplete, or not connected to the question. |
| Data discipline | Sources, dates, fields, transformations, and missingness are documented. | Main source and transformations are stated, but limitations are thin. | Data source, cleaning, or transformation choices are unclear. |
| Interpretation | Results are explained in financial terms with limitations. | Results are described, but implications or caveats are limited. | Output is reported without meaningful interpretation. |
| Reproducibility | Notebook runs from a clean environment and uses project helpers. | Notebook mostly runs, with small manual steps. | Notebook depends on hidden state, credentials, or untracked files. |
| Communication | Tables and charts are readable, labeled, and aligned with the question. | Most outputs are readable, with minor labeling issues. | Outputs are difficult to read or not connected to the deliverable. |

## Module checkpoints

| Module | Checkpoint | Required evidence |
| --- | --- | --- |
| 0. Setup and Python ecosystem | Reproducibility report | Python version, `uv sync` status, `make book` status, and setup issues |
| 1. Markets, data, and EDA | EDA brief | Source inventory, quality report, return statistics, correlation chart, and caveat |
| 2. Financial time series | Modeling report | Stationarity, ARIMA or GARCH specification, diagnostics, and limitations |
| 3. Market risk | Risk memo | VaR, Expected Shortfall, exceptions, stress scenario, and assumptions |
| 4. Portfolio theory | Allocation brief | Weights, expected return, volatility, Sharpe ratio, covariance method, and risk contributions |
| 5. Fixed income | Bond sensitivity memo | Clean and dirty price, accrued interest, yield, duration, convexity, DV01, shocked prices, and immunization interpretation |
| 6. Term structure | Curve and rate simulation report | Discount factors, spot and forward rates, Nelson-Siegel fit, PCA scenarios, calibrated paths, Feller check, and model-risk notes |
| 7. Derivatives | Option pricing notebook | Forward or swap value, option prices, Greeks, put-call parity, implied volatility, numerical comparison, exotic-method interpretation, and hedging/model-risk notes |

## Dashboard rubric

| Dashboard | Minimum requirement | Interpretation requirement |
| --- | --- | --- |
| Macro dashboard | At least three macro or market variables with documented source logic | Explain one macro-financial relationship and one data limitation |
| Return explorer | Price, return, volatility, drawdown, and distribution view | Compare two assets and state which risk feature matters most |
| Volatility/GARCH dashboard | Parameter-controlled conditional volatility simulation | Explain how persistence changes risk forecasts |
| VaR/CVaR simulator | At least three VaR methods and Expected Shortfall | Explain why tail assumptions change reported risk |
| Efficient frontier dashboard | Frontier, GMVP, tangency portfolio, and weights | Explain how correlation or expected returns change allocation |
| Bond duration-convexity dashboard | Exact repricing and duration-convexity approximation | Explain when duration-only approximation fails |
| Term-structure scenario work | Fitted curve, PCA loadings, and at least three curve shocks | Explain level, slope, curvature, and one ALM implication |
| Black-Scholes Greeks dashboard | Price curve, payoff, and Greeks | Explain one hedging implication from the Greeks |
| Implied-volatility work | Root-finding results across at least five strikes | Explain smile/skew behavior and one failed-inversion risk |

## Final project requirements

The final project should be a reproducible quantitative finance notebook that uses at least three course modules.

Required sections:

1. Research question and scope.
2. Data source inventory.
3. Data cleaning and quality report.
4. Exploratory analysis.
5. Model or valuation method.
6. Risk or sensitivity analysis.
7. Interactive dashboard or scenario tool.
8. Financial interpretation.
9. Limitations and model-risk discussion.
10. Reproducibility appendix with environment and commands.

## Final project topic examples

| Topic | Required modules |
| --- | --- |
| Mexican macro and equity risk dashboard | Markets, time series, market risk |
| Portfolio allocation with VaR overlay | Markets, portfolio theory, market risk |
| Fixed-income sensitivity under rate scenarios | Fixed income, term structure, dashboards |
| Option pricing and Greeks monitor | Derivatives, market risk, dashboards |
| Implied volatility and exotic option analysis | Derivatives, time series, market risk |
| Yield curve and bond portfolio analysis | Term structure, fixed income, portfolio theory |

## Submission checklist

- The notebook runs from the project root after `uv sync`.
- API credentials are not committed.
- Data source and field choices are documented.
- The notebook uses helpers from `src/` when available.
- Charts have titles, axis labels, and units.
- Risk metrics state confidence level, horizon, and sign convention.
- The conclusion states what the model can and cannot support.
