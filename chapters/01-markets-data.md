# Markets, Data, and EDA

This module connects the basic theory of financial markets with the practice of obtaining, cleaning, and exploring data.

## Expected outcome

By the end of this module, students should be able to:

- describe the main financial markets;
- download asset and macroeconomic data;
- calculate returns and descriptive metrics;
- detect data quality issues;
- communicate findings with tables and charts;
- use reusable data clients and local caching for provider access;
- build macro and return dashboards from clean panels;
- document provider limitations, missingness, calendar alignment, and corporate action assumptions.

## Included notes

- stock market operations;
- financial data extraction;
- stock data EDA;
- macroeconomic data EDA;
- market data quality framework;
- Mexican market data pipeline;
- Macro Dashboard: Banxico and FRED;
- Return Explorer Dashboard.

## Recommended next improvements

- add live Banxico SIE examples for TIIE, INPC, and the FIX exchange rate;
- add real-data dashboard variants with instructor-approved API credentials;
- add MexDer and Mexican fixed-income instrument examples in the later modules.

## Class sequence

1. Build the conceptual foundation with stock market operations.
2. Download price and macroeconomic data from external providers.
3. Audit missing values, date ranges, and provider-specific structure.
4. Transform prices into returns and log returns.
5. Summarize return distributions and correlations.
6. Document data quality assumptions before moving into modeling.
7. Build a Mexican market data pipeline with source inventory, quality report, and return summary.
8. Use the macro dashboard and return explorer to communicate results interactively.

## In-class practice

Students should work in pairs to select three assets, download or use instructor-provided adjusted close prices, compute returns, and compare return, volatility, skewness, kurtosis, and correlation patterns. They should also create a data source inventory, flag at least one data quality risk, and produce one dashboard view.

## Module checkpoint

The checkpoint is an EDA brief with a data source table, quality notes, calendar and corporate action assumptions, summary statistics, and at least one dashboard view that supports a financial interpretation.
