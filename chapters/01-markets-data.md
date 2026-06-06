# Markets and Data

This module connects the basic theory of financial markets with the practice of obtaining, cleaning, and exploring data {cite}`mishkin2019financial,fabozzi2019foundations`.

## Expected outcome

By the end of this module, students should be able to:

- describe the main financial markets, participants, venues, and post-trade infrastructure;
- distinguish primary and secondary markets, exchange-traded and OTC markets, and price discovery and valuation;
- identify the main features of equities, fixed income, funds, ETFs, FIBRAs, CKDs, and derivatives;
- read core quotation conventions such as price, rate, yield, return, spread, volume, maturity, coupon, and duration;
- explain how expectations, liquidity, risk premia, interest rates, inflation, exchange rates, growth, and monetary policy affect asset prices;
- classify market, macroeconomic, reference, corporate, and metadata sources;
- evaluate financial data sources by authority, coverage, definition, frequency, latency, revisions, accessibility, reliability, licensing, and reproducibility;
- download asset and macroeconomic data;
- separate raw extracts, cached responses, processed datasets, metadata, and analysis-ready panels;
- calculate returns and descriptive metrics;
- align multiple instruments across calendars, frequencies, currencies, and source conventions;
- summarize empirical return distributions with mean, median, volatility, percentiles, skewness, kurtosis, and rolling diagnostics;
- explore dependence with correlations, scatterplots, heatmaps, and rolling relationships without treating correlation as causation;
- read basic exploratory risk and performance metrics such as volatility, downside percentiles, drawdown, historical VaR as a percentile, Sharpe ratio, hit ratio, cumulative return, and benchmark comparison;
- detect data quality issues;
- communicate findings with tables and charts;
- design exploratory dashboards around analytical questions, source notes, and limitations;
- turn charts and metrics into concise data narratives for technical and non-technical readers;
- integrate the full workflow into a reproducible Mexican market analysis case;
- use reusable data clients and local caching for provider access;
- build macro and return dashboards from clean panels;
- document provider limitations, missingness, calendar alignment, and corporate action assumptions.

## Included notes

- market foundations;
- financial data extraction;
- stock data EDA;
- macroeconomic data EDA;
- market data quality framework;
- Mexican market data pipeline;
- Macro Dashboard: Banxico and FRED;
- Return Explorer Dashboard;
- Communication and Integrated Case.

## Conceptual structure

The module is organized as a bridge from market language to usable datasets:

| Part | Main question | Concepts |
| --- | --- | --- |
| Market foundations | What is the market and what is traded? | Market structure, participants, trading venues, instruments, infrastructure, and regulation. |
| Quotation and price formation | How is market information observed? | Prices, rates, yields, returns, bid-ask spreads, liquidity, market calendars, expectations, risk premia, and macro context. |
| Data preparation | How do raw observations become analytical inputs? | Source hierarchy, extraction design, raw/interim/processed layers, cache, metadata, quality checks, calendar alignment, adjusted prices, returns, and financial panels. |
| Exploratory analysis and communication | How are market datasets interpreted and delivered? | Descriptive statistics, empirical distributions, rolling diagnostics, correlations, dependence checks, heatmaps, drawdowns, basic risk and performance metrics, dashboards, data narratives, and an integrated Mexican market analysis case. |

Several topics appear here only as introductory concepts. Duration, yield to maturity, convexity, VaR, expected shortfall, stress testing, volatility modeling, beta, tracking error, factor exposure, derivatives pricing, option Greeks, implied volatility, forecasting, causal inference, dashboard deployment, automated refresh, production reporting, and valuation are deferred to later specialized modules.

## Lesson map

The detailed syllabus is integrated as a four-part module rather than as sixteen separate pages. This keeps the reader moving from market language to usable data, then to exploratory analysis and communication.

| Syllabus block | Integrated lessons | Treatment in this module |
| --- | --- | --- |
| Market structure | `1.1` Market Foundations | Financial markets, participants, venues, trading infrastructure, regulators, and price discovery. |
| Financial instruments | `1.1` Market Foundations | Introductory comparison of equities, fixed income, derivatives, ETFs, funds, FIBRAs, and CKDs; instrument-specific valuation is deferred to later modules. |
| Quotation, liquidity, and market conventions | `1.1`, `1.3`, `1.6`, `1.7` | Prices, rates, yields, spreads, calendars, volume, maturity, coupon, duration, metadata, and provider conventions. |
| Price formation and macro context | `1.1`, `1.5`, `1.8` | Expectations, information, liquidity, inflation, interest rates, exchange rates, growth, monetary policy, and macro-financial dashboards. |
| Data sources | `1.3`, `1.7`, `1.8` | Banxico, FRED, public market sources, official macro providers, source hierarchy, coverage, frequency, reliability, and reproducibility. |
| Extraction, storage, and cache | `1.3`, `1.7` | Download design, local cache, raw extracts, processed panels, metadata, and reproducible analysis architecture. |
| Data quality and integrity | `1.6`, `1.7` | Missing dates, incomplete series, duplicated records, units, calendars, revisions, provider limits, corporate actions, and documented assumptions. |
| Prices, returns, and panels | `1.4`, `1.7`, `1.9` | Simple returns, log returns, alignment, missing data handling, multi-asset panels, and return dashboards. |
| Mexican market pipeline | `1.7`, `1.10` | Local source inventory, cleaning, validation, assumptions, and a traceable Mexican market analysis workflow. |
| Descriptive statistics | `1.4`, `1.5`, `1.9` | Mean, median, volatility, percentiles, skewness, kurtosis, empirical distributions, and rolling diagnostics. |
| Correlation and exploratory dependence | `1.4`, `1.9` | Correlation matrices, scatterplots, heatmaps, rolling relationships, dependence caveats, and regime-aware interpretation. |
| Basic risk and performance metrics | `1.4`, `1.9`, `1.10` | Volatility, drawdown, historical VaR as a percentile, Sharpe ratio, hit ratio, cumulative return, and benchmark comparison as exploratory measures. |
| Dashboards | `1.8`, `1.9` | Offline and live-data dashboard variants using Banxico, FRED, and public market sources through shared data helpers. |
| Data narrative and integrated case | `1.10` | Translation from chart to insight, source caveats, interpretation, limitations, and final Mexican market case. |

## Data source notes

This module should prioritize real provider data when examples can remain reproducible and classroom-safe. The working source map is:

| Use case | Preferred sources | Classroom role |
| --- | --- | --- |
| Mexico official indicators | INEGI API, Banxico SIE {cite}`inegiAPI2025,banxicoSIE2025` | National macro, inflation, rates, exchange rates, and official context. |
| US macro and rates | FRED {cite}`fredAPI2025` | Reference macro series, Treasury rates, and US comparison panels. |
| Global and LATAM comparisons | World Bank Open Data, DBnomics, IMF WEO, OECD Data {cite}`worldBankOpenData2025,dbnomics2025,imfWEO2025,oecdData2025` | Cross-country projects, macro dashboards, and development indicators. |
| Equity and market prices | Finnhub, EODHD, Alpha Vantage, Financial Modeling Prep, Yahoo Finance | Returns, technical indicators, fundamentals, global end-of-day history, and quick classroom exploration. |
| Mexican market infrastructure | BMV, BIVA, MexDer, CNBV, PIP, and Valmer | Listed securities context, derivatives conventions, regulatory reference, and independent valuation context. |

Yahoo Finance remains useful for rapid demonstrations through `yfinance`, but it should not be the only source for serious projects because it is an unofficial access path and can have unstable limits {cite}`yfinance2025`. For Mexico-specific claims, prefer INEGI or Banxico when they cover the required series {cite}`inegiAPI2025,banxicoSIE2025`.

For educational work, use this source hierarchy:

1. official sources for macroeconomic and regulatory series;
2. exchange or licensed sources for official market data;
3. public portals and open-source wrappers for exploratory learning;
4. local cached datasets for reproducible exercises;
5. synthetic or sample datasets when licenses prevent redistribution.

The macro and return dashboards now support two data modes:

| Mode | How to use it | Role |
| --- | --- | --- |
| `DATA_MODE=offline` | Default book build and classroom validation | Uses deterministic synthetic panels so the book can run without credentials or network access. |
| `DATA_MODE=live` | Local JupyterLab sessions with approved credentials and network access | Uses Banxico, FRED, and public market prices through cached helpers in `src.market_data`. |

## Reading sequence

1. Build the conceptual foundation with market structure, instruments, quotation conventions, price formation, and macro context.
2. Build a source inventory that records provider, field, calendar, unit, coverage, license note, and known limitations.
3. Download price and macroeconomic data from external providers or deterministic classroom panels.
4. Preserve raw extracts and use cache to avoid unnecessary live API calls.
5. Audit missing values, missing dates, date ranges, duplicates, units, revisions, and provider-specific structure.
6. Transform prices into simple returns, log returns, and documented return panels.
7. Summarize return distributions with descriptive statistics, tail percentiles, skewness, kurtosis, and rolling diagnostics.
8. Explore relationships with correlations, scatterplots, heatmaps, rolling correlations, and macro-financial visuals.
9. Compute introductory risk and performance metrics such as volatility, drawdown, historical VaR as a percentile, Sharpe ratio, hit ratio, cumulative return, and benchmark comparison.
10. Document data quality assumptions before moving into modeling.
11. Build a Mexican market data pipeline with source inventory, quality report, and return summary.
12. Use the macro dashboard and return explorer to communicate results interactively without overclaiming.
13. Convert exploratory charts into a short narrative that states question, data, method, finding, interpretation, limitation, and next step.
14. Close the module with an integrated Mexican market case that is traceable from final chart back to original source.
