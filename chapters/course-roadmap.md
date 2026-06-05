# Course Roadmap

This roadmap takes the classic financial mathematics syllabus and updates it into an applied version built around Python, real data, simulation, optimization, and interactivity.

For implementation standards, use `chapters/course-standards.md`. For graded practice and module checkpoints, use `chapters/exercises.md`.

## Module 0: Setup and Python ecosystem

Goal: prepare a reproducible environment for quantitative analysis.

Core topics:

- Python 3.12+ installation with `pyenv`;
- project environment management with `uv`;
- using JupyterLab and Jupyter Book;
- Git workflow;
- environment validation and package reproducibility checks;
- foundations of `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, and `ipywidgets`;
- connection to data sources such as Yahoo Finance, FRED, and Banxico SIE;
- local caching and reusable data-access helpers.

## Module 1: Markets, data, and exploratory analysis

Goal: understand how markets operate and how to convert financial data into analytical inputs.

Core topics:

- equity markets, indices, fixed income, derivatives, and foreign exchange;
- extraction of prices, rates, and macroeconomic data;
- market microstructure and Mexican financial institutions;
- cleaning, calendar alignment, corporate action awareness, returns, volatility, and correlations;
- robust outlier flags and data quality reporting;
- exploratory visualization of assets and macro variables;
- Mexican market data pipeline design;
- macro dashboard with Banxico/FRED source mapping;
- return explorer dashboard.

## Module 2: Financial time series

Goal: analyze the time dynamics of prices, returns, and volatility.

Core topics:

- components of a time series;
- stationarity, autocorrelation, and lags;
- ARIMA models as a baseline;
- model diagnostics with ADF, Ljung-Box, and Jarque-Bera tests;
- conditional heteroskedasticity;
- ARCH and GARCH models;
- asymmetric volatility extensions and heavy-tailed residuals;
- ARIMA and GARCH model-building labs;
- interactive volatility and GARCH dashboard.

## Module 3: Market risk

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

## Module 4: Modern portfolio theory

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

## Module 5: Time value of money and fixed income

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

## Module 6: Term structure and interest rate models

Goal: move from flat-yield valuation to curve-based valuation and rate simulation.

Core topics:

- spot rates, forward rates, par rates, and discount factors;
- curve bootstrapping from market instruments;
- interpolation and curve visualization;
- Nelson-Siegel curve fitting and Nelson-Siegel-Svensson as an extension;
- yield curve level, slope, curvature, and PCA scenario design;
- Vasicek AR(1) calibration and Cox-Ingersoll-Ross Feller-condition checks;
- short-rate simulation and model-risk interpretation.

## Module 7: Derivatives

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

- live-data variants for dashboards using Banxico, FRED, and public market sources;
- calendar-aware Mexican fixed-income examples using Banxico and public market sources;
- liability-driven portfolio optimization examples;
- Monte Carlo market risk with PCA covariance stabilization;
- FRTB liquidity-horizon Expected Shortfall examples;
- Nelson-Siegel-Svensson calibration and yield curve PCA with historical data;
- constrained portfolio optimization with turnover and transaction costs;
- full implied-volatility surface fitting with arbitrage checks;
- option strategy analysis and portfolio hedging P&L attribution;
- capstone projects with interactive dashboards and real Mexican data.
