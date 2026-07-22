# Markets, Instruments, and Data

This module connects the basic theory of financial markets with the practice of obtaining, cleaning, and exploring data {cite}`mishkin2019financial,fabozzi2019foundations`.

## Prerequisites

Complete the three Module 0 readiness gates before running these lessons.
Readers should be comfortable with percentages, logarithms, averages, standard
deviations, and reading a date-indexed table. Statistical modeling is not a
prerequisite: this module prepares the panels that later modules model.

## Expected outcome

By the end of this module, students should be able to:

- explain how market structure, instruments, quotation conventions, expectations,
  liquidity, risk premia, and macroeconomic conditions shape financial
  observations;
- distinguish market, macroeconomic, reference, corporate, and metadata sources
  and evaluate their authority, coverage, definitions, frequency, latency,
  revisions, access, licensing, and reproducibility;
- load versioned classroom panels and, when access is approved, retrieve provider
  data through reusable clients and a documented cache;
- preserve auditable raw, interim, processed, metadata, and quality layers while
  aligning calendars, frequencies, currencies, units, and corporate actions;
- construct and label simple returns, log returns, and non-contractual level
  changes without treating rates, yields, indexes, or macro levels as if they
  were interchangeable asset returns;
- use descriptive statistics, rolling diagnostics, dependence checks, and
  introductory loss metrics to answer declared exploratory questions without
  implying causality, permanence, or investment suitability; and
- communicate a reproducible Mexican macro-financial case through source-aware
  tables, accessible figures, dashboards, limitations, and a concise narrative.

## Conceptual structure

The module is organized as a bridge from market language to usable datasets:

| Part | Main question | Concepts |
| --- | --- | --- |
| Market foundations | What is the market and what is traded? | Market structure, participants, trading venues, instruments, infrastructure, and regulation. |
| Quotation and price formation | How is market information observed? | Prices, rates, yields, returns, bid-ask spreads, liquidity, market calendars, expectations, risk premia, and macro context. |
| Data preparation | How do raw observations become analytical inputs? | Source hierarchy, extraction design, raw/interim/processed layers, cache, metadata, quality checks, calendar alignment, adjusted prices, returns, and financial panels. |
| Exploratory analysis and communication | How are market datasets interpreted and delivered? | Descriptive statistics, empirical distributions, rolling diagnostics, correlations, dependence checks, heatmaps, drawdowns, basic risk and performance metrics, dashboards, data narratives, and an integrated Mexican market analysis case. |

Several topics appear here only as introductory concepts. Full VaR and expected
shortfall modeling, backtesting, stress testing, volatility modeling, duration,
yield to maturity, convexity, beta, tracking error, factor exposure, derivatives
pricing, option Greeks, implied volatility, forecasting, causal inference,
dashboard deployment, automated refresh, production reporting, and valuation are
deferred to later specialized modules. Module 1 retains only a clearly labeled
historical VaR reading and other descriptive diagnostics needed for data
literacy.

## Visual language

Module 1 uses the same visual grammar across static Matplotlib figures and
interactive Plotly dashboards. Teal identifies the primary analytical series,
muted blue supports comparisons, dark amber marks an important reference or
highlight, coral identifies losses or stress, and ink anchors labels and zero
lines. Meaning never depends on color alone: persistent series also use direct
labels, markers, line styles, position, or sign.

Every analytical figure should make its question, units, sample, source, and
transformation visible in the chart or its adjacent interpretation. Equity EDA
moves from comparable paths to rolling risk, distributions, drawdowns, and
dependence. Macro EDA keeps rates in native percentage units, rebases only
price or index paths, and uses standardized changes only when the visual
explicitly labels a within-series regime comparison. The full authoring and
accessibility contract is owned by `content/README.md`; the executable tokens
and backend templates live in `src/visual_style.py`.

## Publication sample contract

All time-series panel comparisons in Module 1 use the common inclusive analysis
window from **2021-01-01 through 2025-06-30**. The committed source files may contain
earlier or later observations because they support other modules and future
refreshes. Each notebook slices its input to the common window before computing
statistics. The first actual observation can occur after 2021-01-01 because of
weekends, holidays, or release frequency; the final available observation can
occur before 2025-06-30 for the same reason.

Two dated context exhibits sit outside that time-series contract: the May 2026
WFE market-scale snapshot and the exchange-ranking snapshot in the foundations
reading. They are labeled as contextual cross-sections, are not merged with the
teaching panels, and do not enter any 2021--2025 statistic.

The quality-framework lesson also contains a small, explicitly synthetic 2026
series. It demonstrates missing dates, missing values, and outlier review and is
never combined with the empirical teaching panels.

The two empirical panels have deliberately different roles:

- the NASDAQ adjusted-close panel is a bounded classroom example for equity EDA
  and the return explorer;
- the Mexican macro-financial panel is the primary Module 1 case and combines
  official-source USD/MXN and UDI levels, documented synthetic rate-carry
  indexes, and latest-vintage monthly macro context.

They are not merged into a single return claim.

## Lesson map

The detailed syllabus is integrated across nine published lessons grouped into
four parts. This keeps the reader moving from market language to usable data,
then to exploratory analysis and communication. Lesson identifier `1.2` remains
intentionally reserved; the production TOC defines reading order, so the gap is
not missing content and must not be silently reused or renumbered.

| Syllabus block | Integrated lessons | Treatment in this module |
| --- | --- | --- |
| Market structure | `1.1` Market Foundations | Financial markets, participants, venues, trading infrastructure, regulators, and price discovery. |
| Financial instruments | `1.1` Market Foundations | Introductory comparison of equities, fixed income, derivatives, ETFs, funds, FIBRAs, and CKDs; instrument-specific valuation is deferred to later modules. |
| Quotation, liquidity, and market conventions | `1.1`, `1.3`, `1.6`, `1.7` | Prices, rates, yields, spreads, calendars, volume, maturity, coupon, duration, metadata, and provider conventions. |
| Price formation and macro context | `1.1`, `1.5`, `1.8` | Expectations, information, liquidity, inflation, interest rates, exchange rates, growth, monetary policy, and macro-financial dashboards. |
| Data sources | `1.3`, `1.7`, `1.8` | Banxico, DB.NOMICS, public market sources, official macro providers, source hierarchy, coverage, frequency, reliability, and reproducibility. |
| Extraction, storage, and cache | `1.3`, `1.7` | Download design, local cache, raw extracts, processed panels, metadata, and reproducible analysis architecture. |
| Data quality and integrity | `1.6`, `1.7` | Missing dates, incomplete series, duplicated records, units, calendars, revisions, provider limits, corporate actions, and documented assumptions. |
| Prices, level changes, returns, and panels | `1.4`, `1.7`, `1.9` | Simple and log returns for appropriate price series, log changes for non-contractual levels, alignment, missing-data handling, multi-series panels, and exploratory dashboards. |
| Mexican market pipeline | `1.7`, `1.10` | Local source inventory, cleaning, validation, assumptions, and a traceable Mexican market analysis workflow. |
| Descriptive statistics | `1.4`, `1.5`, `1.9` | Mean, median, volatility, percentiles, skewness, kurtosis, empirical distributions, and rolling diagnostics. |
| Correlation and exploratory dependence | `1.4`, `1.9` | Correlation matrices, scatterplots, heatmaps, rolling relationships, dependence caveats, and regime-aware interpretation. |
| Basic risk and performance metrics | `1.4`, `1.9`, `1.10` | NASDAQ return volatility, adjusted-close drawdown, positive-loss historical VaR, arithmetic Sharpe under an explicit zero-rate assumption, hit ratio, and peer comparison; the Mexican case instead uses level drawdown, observed-interval dispersion, downside magnitude, and an explicit synthetic CETES carry reference. |
| Dashboards | `1.8`, `1.9` | Offline and live-data dashboard variants using Banxico, DB.NOMICS, and public market sources through shared data helpers. |
| Data narrative and integrated case | `1.10` | Translation from chart to insight, source caveats, interpretation, limitations, and final Mexican market case. |

## Data source notes

This module should prioritize real provider data when examples can remain reproducible and classroom-safe. The working source map is:

| Use case | Preferred sources | Classroom role |
| --- | --- | --- |
| Mexico official indicators | INEGI API, Banxico SIE {cite}`inegiAPI2025,banxicoSIE2025` | National macro, inflation, rates, exchange rates, and official context. |
| US macro and rates | DB.NOMICS {cite}`dbnomics2025` | Reference macro series, Treasury rates, and US comparison panels. |
| Global and LATAM comparisons | World Bank Open Data, DB.NOMICS, IMF WEO, OECD Data {cite}`worldBankOpenData2025,dbnomics2025,imfWEO2025,oecdData2025` | Cross-country projects, macro dashboards, and development indicators. |
| Equity and market prices | Exchange or licensed feeds, Finnhub, EODHD, Alpha Vantage, Financial Modeling Prep, Yahoo Finance | Returns, technical indicators, fundamentals, global end-of-day history, and quick classroom exploration. |
| Mexican market infrastructure | BMV, BIVA, MexDer, CNBV, PIP, and Valmer | Listed securities context, derivatives conventions, regulatory reference, and independent valuation context. |

Yahoo Finance remains useful for rapid demonstrations through `yfinance`. The
NASDAQ lesson uses a dated adjusted-close extract for `AAPL`, `MSFT`, `NVDA`,
`AMZN`, and `GOOGL`, so its calculations do not change during an offline build.
That reproducibility claim concerns the **fixed analytical input**, not data
ownership, redistribution permission, trading suitability, or continued
provider availability.

`yfinance` is an unofficial access library and Yahoo describes its market data
as informational rather than trading-grade {cite}`yfinance2025,yahooFinanceCoverage2026,yahooTerms2026`.
Before refreshing, sharing, or reusing any provider extract, contributors must
review the terms that apply to their intended use; the presence of a snapshot in
the repository does not itself establish downstream rights. For Mexican
macro-financial claims, prefer Banxico or INEGI when they publish the required
series {cite}`inegiAPI2025,banxicoSIE2025`.

For educational work, use this source hierarchy:

1. official sources for macroeconomic and regulatory series;
2. exchange or licensed sources for official market data;
3. public portals and open-source wrappers for exploratory learning;
4. local cached extracts or committed snapshots of real provider data for reproducible exercises.

The macro and return dashboards use real data in both reproducible and live modes:

| Mode | How to use it | Role |
| --- | --- | --- |
| `DATA_MODE=offline` | Default book build and classroom validation | Uses the versioned Banxico/DB.NOMICS panels for macro views and the dated Yahoo-derived NASDAQ adjusted-close extract for the return explorer, so neither dashboard requires credentials or network access. |
| `DATA_MODE=live` | Local JupyterLab sessions with approved credentials and network access | Uses Banxico, DB.NOMICS, and public market prices through cached helpers in `src.market_data`. |

## Reading sequence

1. Build the conceptual foundation with market structure, instruments, quotation conventions, price formation, and macro context.
2. Build a source inventory that records provider, field, calendar, unit, coverage, license note, and known limitations.
3. Download price and macroeconomic data from external providers or read versioned real-data snapshots, including the NASDAQ equity snapshot used for stock EDA.
4. Preserve raw extracts and use cache to avoid unnecessary live API calls.
5. Audit missing values, missing dates, date ranges, duplicates, units, revisions, and provider-specific structure.
6. Transform prices into simple returns, log returns, and documented return panels.
7. Summarize return distributions with descriptive statistics, tail percentiles, skewness, kurtosis, and rolling diagnostics.
8. Explore relationships with correlations, scatterplots, heatmaps, rolling correlations, and macro-financial visuals.
9. Compute introductory risk and performance summaries only when the economic
   object supports them: NASDAQ return volatility, adjusted-close drawdown,
   positive-loss historical VaR, arithmetic zero-rate Sharpe, hit ratio, and
   peer comparison; or explicitly labeled level-path diagnostics in the Mexican
   case.
10. Document data quality assumptions before moving into modeling.
11. Build a Mexican market data pipeline with a source inventory, quality
    report, and economically labeled return or log-change summary.
12. Use the macro dashboard and return explorer to communicate results interactively without overclaiming.
13. Convert exploratory charts into a short narrative that states question, data, method, finding, interpretation, limitation, and next step.
14. Close the module with the Mexican official-source and synthetic-carry case,
    using the CETES carry index only as a documented classroom reference rather
    than a broad market benchmark.

## Handoff

The module is ready to hand off when the common sample window, transformations,
missing counts, source notes, and reference choice are explicit in every
reported result. The documented return and log-change panels then feed Module 2
time-series diagnostics. Later risk and portfolio modules carry forward and
develop the positive-loss convention, model validation, and benchmark design.
