# Quantitative Financial Mathematics with Python

This Jupyter Book organizes the financial mathematics course notes for actuarial science students into a reproducible, practical, data-oriented path.

The goal is to preserve the existing notes and gradually turn them into a modern learning resource: executable notebooks, financial analysis with Python, visualizations, exercises, and projects applied to the Mexican and global market context.

```{figure} img/msc-international-financial-degree.jpg
:alt: Finance and international markets students
:width: 760px
```

## How to use this book

- Read the course path first to understand the module progression.
- Run notebooks from JupyterLab when you want to reproduce calculations or download external data.
- Use each module page as a study map before entering the original notes.
- Treat the current notebooks as the base source: some are already ready for class, while others are marked for cleanup or expansion.

## Current status

The first version of the book integrates the notes that already exist in `notebooks/class`:

- financial markets and data extraction;
- exploratory analysis of stock and macroeconomic data;
- time series, ARCH, and GARCH;
- VaR and CVaR;
- modern portfolio theory;
- starter notes for time value of money, fixed income, term structure, interest-rate models, and derivatives.

Automatic execution is disabled during the build to avoid failures caused by external APIs, tokens, or network changes. To work interactively, install the environment and open the notebooks in JupyterLab.
