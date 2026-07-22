# Quantitative Financial Mathematics with Python

This Jupyter Book organizes the financial mathematics course notes for actuarial science students into a reproducible, practical, data-oriented path {cite}`jupyterbook2025,kluyver2016jupyter`.

The goal is to turn the course into a modern learning resource: executable notebooks, financial analysis with Python, visualizations, and applied examples connected to Mexican and global market data.

```{figure} img/generated/fm-course-hero.png
:alt: Course hero showing Python, market data, distributions, yield curves, and portfolio optimization.
:width: 760px
```

## How to use this book

- Read the course roadmap first to understand the module progression.
- Use each module page as a study map before entering the notebooks.
- Run notebooks from JupyterLab when you want to reproduce calculations, inspect intermediate data, or use live providers.
- Start with the Module 0 environment checks before running market data or time-series notebooks.

## Who this book is for

This book is written for actuarial science, finance, economics, and data-oriented students who want a practical bridge between financial mathematics and reproducible Python analysis.

It is also useful for instructors or analysts who need concise notebooks for market data, exploratory analysis, time series, risk measurement, portfolio construction, fixed income, interest-rate models, and derivatives.

The book is designed as a reference-style Jupyter Book. It focuses on concepts, formulas, reproducible code, interpretation, and model limitations.

## Mathematical spine

The course uses a small notation core and expands it module by module. Prices and returns are the first building block:

```{math}
R_t = \frac{P_t}{P_{t-1}} - 1,
\qquad
r_t = \log\left(\frac{P_t}{P_{t-1}}\right).
```

Here \(P_t\) is the price or adjusted price at time \(t\), \(R_t\) is the simple return, and \(r_t\) is the log return. Later modules reuse the same time index for volatility, risk, portfolios, term structures, and derivative payoffs.

Reproducibility is treated as part of the mathematics. A numerical claim in the book should be traceable to data, code, environment, and parameters:

```{math}
\text{result} = F(\text{data}, \text{code}, \text{environment}, \theta).
```

Module 0 makes that contract operational before the course moves into market data and time-series modeling.

## Current publication scope

The published book is being developed incrementally. The current Jupyter Book focuses on Modules 0 through 3:

- reproducible setup, Python environment management, and JupyterLab workflow;
- markets, instruments, data extraction, exploratory data analysis, data quality, and first dashboards;
- quantitative methods, financial time series, ARIMA diagnostics, volatility modeling, and GARCH interpretation {cite}`box2015time,engle1982autoregressive,bollerslev1986generalized`;
- economics, macro indicators, policy, currency parity, FX interpretation, and capital market expectations {cite}`mishkin2019financial,banxicoCentralBank,dbnomics2025`.

Later modules on financial statements, corporate finance, valuation, fixed income, derivatives and risk management, alternatives, portfolio management, and capstone pathways remain as work-in-progress source files or roadmap structure in the repository. They are intentionally kept out of the published navigation until each module is reviewed and promoted.

Publication builds use cached execution for curated notebooks while excluding API-sensitive notebooks and disabling live widget execution. To work interactively, install the environment and open the notebooks in JupyterLab.
