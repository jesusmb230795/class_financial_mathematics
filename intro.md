# Quantitative Financial Mathematics with Python

This Jupyter Book organizes the financial mathematics course notes for actuarial science students into a reproducible, practical, data-oriented path {cite}`jupyterbook2025,kluyver2016jupyter`.

The goal is to preserve the existing notes and gradually turn them into a modern learning resource: executable notebooks, financial analysis with Python, visualizations, and applied examples connected to Mexican and global market data.

```{figure} img/generated/fm-course-hero.png
:alt: Finance and international markets students
:width: 760px
```

## How to use this book

- Read the course path first to understand the module progression.
- Run notebooks from JupyterLab when you want to reproduce calculations or download external data.
- Use each module page as a study map before entering the original notes.
- Treat the current notebooks as the base source: some are already ready for class, while others are marked for cleanup or expansion.

## Who this book is for

This book is written for actuarial science, finance, economics, and data-oriented students who want a practical bridge between financial mathematics and reproducible Python analysis.

It is also useful for instructors or analysts who need concise notebooks for market data, exploratory analysis, time series, risk measurement, portfolio construction, fixed income, interest-rate models, and derivatives.

The book is designed as a reference-style Jupyter Book. It focuses on concepts, formulas, reproducible code, interpretation, and model limitations.

## Current publication scope

The published book is being developed incrementally. The current Jupyter Book focuses on Modules 0 through 2:

- reproducible setup, Python environment management, and JupyterLab workflow;
- markets, data extraction, exploratory data analysis, data quality, and first dashboards.
- financial time series, ARIMA diagnostics, volatility modeling, and GARCH interpretation {cite}`box2015time,engle1982autoregressive,bollerslev1986generalized`.

Later modules on market risk, portfolio theory, fixed income, term structure, and derivatives remain as work-in-progress source files in the repository. They are intentionally kept out of the published navigation until each module is reviewed and promoted.

Publication builds use cached execution for curated notebooks while excluding API-sensitive notebooks and disabling live widget execution. To work interactively, install the environment and open the notebooks in JupyterLab.
