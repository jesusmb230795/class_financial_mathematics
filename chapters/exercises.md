# Exercises and Checkpoints

This page collects applied exercises for the Jupyter Book. Each notebook should end with a student deliverable, and each module should have a checkpoint that can be graded as a short technical memo or reproducible notebook.

## Module 0: Setup and Python Ecosystem

| Notebook | Exercise | Deliverable |
| --- | --- | --- |
| Reproducible Computational Finance Stack | Explain the difference between `pyenv`, `uv`, `.venv`, and `uv.lock`. | One-page reproducibility note |
| Environment Validation Lab | Run the environment checks locally and record package versions. | Environment report |
| Initial Repository Setup | Identify the purpose of each top-level repository file. | Repository map |
| Classroom Environment Setup | Launch JupyterLab from the project environment. | Screenshot or command log summary |

Checkpoint: submit a reproducibility report with Python version, `uv sync` status, `make book` status, and any local setup issues.

## Module 1: Markets, Data, and EDA

| Notebook | Exercise | Deliverable |
| --- | --- | --- |
| Stock Market Operations | Map each market participant to its function in price discovery. | Market structure diagram or table |
| Financial Data Extraction | Download one equity index and one macro series. | Data source inventory |
| Stock Data EDA | Compare return, volatility, skewness, and kurtosis for three assets. | EDA summary table |
| Macroeconomic Data EDA | Compare two macro variables and describe their co-movement. | Macro interpretation memo |
| Market Data Quality Framework | Flag missingness, outliers, and corporate action risks. | Data quality memo |
| Mexican Market Data Pipeline | Build a clean price matrix and return matrix. | Reproducible data pipeline notebook |
| Macro Dashboard - Banxico and FRED | Build a macro dashboard view with documented provider logic. | Macro dashboard memo |
| Return Explorer Dashboard | Compare two assets using price, return, volatility, drawdown, and histogram views. | Return explorer brief |

Checkpoint: submit an EDA brief with source inventory, missingness report, adjusted-price policy, return statistics, dashboard view, correlation chart, and one modeling caveat.

## Module 2: Financial Time Series

| Notebook | Exercise | Deliverable |
| --- | --- | --- |
| Time Series I | Compare price levels and log returns for stationarity. | Stationarity diagnostic note |
| ARCH and GARCH | Fit one volatility model and interpret parameters. | Volatility model summary |
| Time Series Diagnostics and Volatility Extensions | Run ADF, Ljung-Box, and Jarque-Bera diagnostics. | Residual diagnostic report |
| ARIMA Diagnostic Workflow | Search a small ARIMA grid and select a model. | ARIMA model selection memo |
| GARCH Volatility and Risk Workflow | Compare Gaussian and Student's t GARCH models. | Volatility and VaR memo |
| Interactive Volatility and GARCH Dashboard | Compare low- and high-persistence volatility regimes. | Volatility dashboard interpretation |

Checkpoint: submit a modeling report that states the selected model, diagnostics, volatility behavior, dashboard scenario, distributional assumption, and model limitations.

## Module 3: Market Risk

| Notebook | Exercise | Deliverable |
| --- | --- | --- |
| VaR and CVaR | Estimate historical, Gaussian, and modified VaR for one asset and one portfolio. | Risk metric comparison memo |
| Downside Risk and VaR Methods | Compare semideviation, Sortino ratio, VaR, and Expected Shortfall on the same return series. | Tail-risk method table |
| VaR Backtesting and Stress Testing | Backtest a rolling VaR model and design one deterministic scenario. | Backtesting and stress memo |
| Interactive VaR and CVaR Simulator | Compare two tail regimes and explain metric sensitivity. | VaR/CVaR simulator memo |

Checkpoint: submit a risk report with confidence level, method, estimated loss, exception behavior, stress scenario, simulator scenario, assumptions, and limitations.

## Module 4: Modern Portfolio Theory

| Notebook | Exercise | Deliverable |
| --- | --- | --- |
| Mean-Variance and the Efficient Frontier | Build equal-weight, minimum-variance, and maximum-Sharpe portfolios. | Portfolio allocation brief |
| Analytical Efficient Frontier | Compute Merton constants, GMVP, target-return weights, and tangency weights. | Analytical frontier notebook |
| Robust Portfolio Construction | Compare shrinkage, CAPM beta, risk parity, and HRP allocations. | Robust allocation memo |
| Interactive Efficient Frontier Dashboard | Compare two correlation or risk-free-rate assumptions. | Frontier dashboard memo |

Checkpoint: submit selected weights, expected return, volatility, Sharpe ratio, covariance method, beta estimate, risk contributions, dashboard scenario, concentration risk, and one robustness concern.

## Module 5: Time Value of Money and Fixed Income

| Notebook | Exercise | Deliverable |
| --- | --- | --- |
| Time Value of Money | Value three cash-flow streams under multiple discount rates. | Cash-flow valuation table |
| Bond Pricing, Duration, and Convexity | Compare exact repricing against duration-convexity approximation. | Bond sensitivity memo |
| Interactive Bond Duration-Convexity Dashboard | Use sliders to test coupon, maturity, yield, and shock scenarios. | Scenario interpretation brief |
| Mexican Government Bond Valuation | Price CETES, Bonos M, and UDIBONOS with stated day-count and settlement assumptions. | Mexican bond valuation note |
| Yield, DV01, and Immunization Lab | Solve YTM, compute DV01, compare approximation errors, and test Redington conditions. | YTM and immunization memo |

Checkpoint: submit a bond valuation memo with clean and dirty price, accrued interest, yield, duration, convexity, DV01, exact shocked prices, approximation errors, convention risk, and interest-rate risk interpretation.

## Module 6: Term Structure and Interest Rate Models

| Notebook | Exercise | Deliverable |
| --- | --- | --- |
| Yield Curve Bootstrapping | Bootstrap discount factors and derive spot and forward rates. | Yield curve notebook |
| Short-Rate Models | Simulate Vasicek and CIR paths and compare assumptions. | Short-rate simulation memo |
| Nelson-Siegel Curve Fitting | Fit a smooth curve and interpret level, slope, curvature, and residuals. | Curve-fitting note |
| Yield Curve PCA and Scenarios | Run PCA on curve changes and create level, slope, and curvature stress scenarios. | PCA scenario memo |
| Short-Rate Calibration Lab | Calibrate Vasicek through AR(1), simulate Vasicek and CIR, and check Feller condition. | Calibration and model-risk memo |

Checkpoint: submit a curve and simulation report with bootstrapped rates, Nelson-Siegel fit, PCA scenario interpretation, simulated paths, calibration output, Feller-condition result, and model-risk notes.

## Module 7: Derivatives

| Notebook | Exercise | Deliverable |
| --- | --- | --- |
| Options and Black-Scholes | Price calls and puts across multiple strikes and verify put-call parity. | Option pricing table |
| Binomial Trees and Monte Carlo | Compare Black-Scholes, binomial, and Monte Carlo prices. | Numerical pricing comparison |
| Interactive Black-Scholes Greeks Dashboard | Use sliders to explain moneyness and Greeks. | Greeks interpretation memo |
| Linear Derivatives and Carry | Price a forward, mark it to market, and compute a par swap rate. | Carry and swap memo |
| Implied Volatility and Smiles | Invert synthetic market prices into implied volatilities by strike. | Implied-volatility smile note |
| American and Exotic Option Methods | Compare American put methods, Asian control variates, and BGK barrier adjustment. | Exotic methods memo |
| Stochastic Volatility and Heston Lab | Simulate Heston paths and compare terminal distributions and option prices. | Stochastic-volatility memo |

Checkpoint: submit an option pricing notebook with assumptions, forward or swap valuation, prices, Greeks, parity check, implied volatility, numerical comparison, exotic-method interpretation, and hedging or model-risk discussion.
