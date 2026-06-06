# Course Roadmap

This roadmap takes the classic financial mathematics syllabus and updates it into an applied version built around Python, real data, simulation, optimization, and interactivity {cite}`mckinney2010data,harris2020array,kluyver2016jupyter`.

The published Jupyter Book currently focuses on Modules 0 through 2. Modules 3 through 7 are tracked here as a work-in-progress roadmap and remain outside the published table of contents until they are reviewed and promoted.

## Module 0: Setup and Python ecosystem

Goal: prepare a reproducible environment for quantitative analysis.

Core topics:

- Python 3.12+ installation with `pyenv`;
- project environment management with `uv`;
- using JupyterLab and Jupyter Book;
- Git workflow;
- environment validation and package reproducibility checks;
- foundations of `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, and `ipywidgets` {cite}`mckinney2010data,harris2020array,hunter2007matplotlib,waskom2021seaborn,plotly2015collaborative`;
- connection to data sources such as Yahoo Finance, FRED, and Banxico SIE {cite}`yfinance2025,fredAPI2025,banxicoSIE2025`;
- local caching and reusable data-access helpers.

## Module 1: Markets and Data

Goal: understand how markets operate and how to convert financial data into analytical inputs.

Core topics:

- market structure, participants, trading venues, post-trade infrastructure, and regulation;
- equities, fixed income, funds, ETFs, FIBRAs, CKDs, derivatives, indices, and foreign exchange;
- quotation conventions, liquidity, bid-ask spreads, calendars, frequencies, and instrument-specific data fields;
- price formation through supply, demand, expectations, discount rates, risk premia, liquidity, and macroeconomic context;
- source taxonomy for market data, macroeconomic data, reference data, corporate data, and metadata;
- source selection criteria: authority, coverage, definition, frequency, latency, revisions, accessibility, reliability, licensing, and reproducibility;
- extraction of prices, rates, and macroeconomic data;
- raw/interim/processed data layers, local cache, metadata, and audit trails;
- cleaning, calendar alignment, frequency alignment, corporate action awareness, revisions, returns, volatility, and correlations;
- wide and long financial panels with source and quality flags;
- descriptive return statistics: count, mean, median, volatility, minimum, maximum, percentiles, skewness, kurtosis, and missingness;
- empirical distributions, rolling means, rolling volatility, rolling correlations, and rolling drawdowns;
- correlation matrices, heatmaps, scatterplots, dependence caveats, and macro-financial exploratory relationships;
- introductory risk and performance metrics: downside percentiles, drawdown, historical VaR as a percentile, Sharpe ratio, hit ratio, cumulative return, and benchmark comparison;
- robust outlier flags and data quality reporting;
- exploratory visualization of assets and macro variables;
- Mexican market data pipeline design;
- macro dashboard with Banxico/FRED source mapping;
- return explorer dashboard;
- dashboard design principles: analytical question, chart purpose, offline/live mode, source note, and data quality visibility;
- data narrative from observation to insight, including audience, uncertainty, limitations, and next analytical step;
- integrated Mexican market case that connects data inventory, quality report, return panel, exploratory metrics, dashboard, and written interpretation.

## Module 2: Financial time series

Goal: analyze the time dynamics of prices, returns, and volatility.

Core topics:

- components of a time series;
- stationarity, autocorrelation, and lags;
- ARIMA models as a baseline {cite}`box2015time,hamilton1994time`;
- model diagnostics with ADF, Ljung-Box, and Jarque-Bera tests {cite}`dickey1979distribution,ljung1978measure,jarque1980efficient`;
- conditional heteroskedasticity;
- ARCH and GARCH models {cite}`engle1982autoregressive,bollerslev1986generalized`;
- asymmetric volatility extensions and heavy-tailed residuals;
- ARIMA and GARCH model-building labs;
- interactive volatility and GARCH dashboard.

## Work in progress: Module 3 - Market risk

Goal: quantify potential losses under different distributional assumptions and scenarios.

Core topics:

- semideviation and loss metrics;
- parametric, historical, and simulated Value at Risk;
- Cornish-Fisher modified VaR;
- volatility-weighted historical simulation with EWMA volatility;
- Conditional Value at Risk or Expected Shortfall;
- Kupiec and Christoffersen VaR backtesting;
- Basel traffic-light interpretation;
- deterministic stress testing and scenario analysis;
- interactive VaR and CVaR simulator.

## Work in progress: Module 4 - Modern portfolio theory

Goal: build and evaluate portfolios using the mean-variance framework.

Core topics:

- estimation of returns, volatility, and covariance;
- portfolio simulation;
- efficient frontier;
- analytical efficient frontier and Merton constants;
- minimum variance portfolio;
- Sharpe ratio and weight allocation;
- tangency portfolio with a documented risk-free proxy;
- CAPM beta with robust standard errors;
- covariance shrinkage with Ledoit-Wolf and OAS;
- risk parity and Hierarchical Risk Parity;
- interactive efficient frontier dashboard.

## Work in progress: Module 5 - Time value of money and fixed income

Goal: value deterministic cash flows and measure bond price sensitivity.

Core topics:

- present value, future value, discount factors, and compounding;
- annuities and general cash-flow valuation;
- zero-coupon and coupon bond pricing;
- clean price, dirty price, accrued interest, and yield to maturity;
- CETES, Bonos M, and UDIBONOS valuation under simplified ACT/360 conventions;
- Macaulay duration, modified duration, dollar duration, and convexity;
- DV01, second-order price approximation, and Redington immunization;
- bond repricing under interest-rate shocks;
- interactive bond duration-convexity dashboard.

## Work in progress: Module 6 - Term structure and interest rate models

Goal: move from flat-yield valuation to curve-based valuation and rate simulation.

Core topics:

- spot rates, forward rates, par rates, and discount factors;
- curve bootstrapping from market instruments;
- interpolation and curve visualization;
- Nelson-Siegel curve fitting and Nelson-Siegel-Svensson as an extension;
- yield curve level, slope, curvature, and PCA scenario design;
- Vasicek AR(1) calibration and Cox-Ingersoll-Ross Feller-condition checks;
- short-rate simulation and model-risk interpretation.

## Work in progress: Module 7 - Derivatives

Goal: price derivative contracts using no-arbitrage reasoning and numerical methods.

Core topics:

- forwards, futures, calls, puts, and payoff diagrams;
- cost-of-carry pricing, forward mark-to-market value, and par swap rates;
- one-period replication and risk-neutral pricing;
- binomial trees and convergence;
- Black-Scholes pricing for European options;
- Greeks and local risk sensitivity;
- implied volatility inversion, smiles, and surface interpretation;
- Monte Carlo simulation for option pricing;
- American exercise, Asian control variates, barrier continuity correction, and Heston simulation;
- interactive Black-Scholes Greeks dashboard.

## Expansion track

The following topics complete the course vision, but still require richer notebooks, real data integration, or project-level expansion:

| Expansion topic | Primary module | Promotion note |
| --- | --- | --- |
| Live-data variants for dashboards using Banxico, FRED, and public market sources | Module 1 - Markets and Data | Implemented as opt-in `DATA_MODE=live` dashboard panels; reuse Module 1 provider, cache, and quality patterns before adding live data to later modules. |
| Calendar-aware Mexican fixed-income examples using Banxico and public market sources | Module 5 - Time value of money and fixed income | Promote after fixed-income conventions, settlement dates, accrued interest, CETES, Bonos M, and UDIBONOS examples are stable. |
| Liability-driven portfolio optimization examples | Module 5 - Time value of money and fixed income | Treat as an asset-liability extension that can reference Module 4 portfolio optimization once the fixed-income cash-flow mechanics are ready. |
| Monte Carlo market risk with PCA covariance stabilization | Module 3 - Market risk | Add after VaR, CVaR, stress testing, and covariance estimation references are stable; connect to Module 4 as a supporting method. |
| FRTB liquidity-horizon Expected Shortfall examples | Module 3 - Market risk | Keep as an advanced regulatory risk extension after Expected Shortfall and backtesting are clear. |
| Nelson-Siegel-Svensson calibration and yield curve PCA with historical data | Module 6 - Term structure and interest rate models | Promote after bootstrapping, Nelson-Siegel, PCA scenarios, and historical yield-curve data handling are reviewed. |
| Constrained portfolio optimization with turnover and transaction costs | Module 4 - Modern portfolio theory | Add after the analytical frontier and robust allocation notebooks, because the extension depends on constraints and implementation frictions. |
| Full implied-volatility surface fitting with arbitrage checks | Module 7 - Derivatives | Promote after implied-volatility inversion and option-smile interpretation are stable. |
| Option strategy analysis and portfolio hedging P&L attribution | Module 7 - Derivatives | Add after Greeks, implied volatility, and numerical pricing are ready enough to support hedging interpretation. |
| Capstone projects with interactive dashboards and real Mexican data | Final project track after Module 7 | Keep as a cross-module capstone drawing from Module 1 data foundations and the promoted modeling modules. |
