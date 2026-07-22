# Course Roadmap

This roadmap records the complete eleven-module publication path. The course
remains global-first, data-oriented, and Python-based, with Mexican market
examples used as local applications rather than as the only frame of reference
{cite}`mckinney2010data,harris2020array,kluyver2016jupyter`.

Modules 0 through 10 are part of the canonical navigation. Narrative lessons
and executable notebooks use one terminology contract, explicit prerequisites
and handoffs, resolved citations, and reviewed data or simulation boundaries.

## Concision principle

The detailed syllabus is intentionally compressed here. Each module records the goal, required coverage, and promotion boundary. Long topic lists belong inside module pages, notebooks, dashboards, and project briefs after the corresponding content is ready.

## At a glance

| Module | Title | Publication status |
| --- | --- | --- |
| 0 | Setup and Python Ecosystem | Published; substantial |
| 1 | Markets, Instruments, and Data | Published; substantial |
| 2 | Quantitative Methods and Financial Time Series | Published; substantial |
| 3 | Economics, Macro, and Currency | Published; substantial |
| 4 | Financial Statement Analysis and Financial Modeling | Published; substantial |
| 5 | Corporate Issuers and Equity Valuation | Published; substantial |
| 6 | Fixed Income, Credit, and Term Structure | Published; substantial |
| 7 | Derivatives and Risk Management | Published; substantial |
| 8 | Alternative Investments | Published; substantial |
| 9 | Portfolio Management, Asset Allocation, and Performance | Published; substantial |
| 10 | Advanced Pathways and Capstone | Published; substantial |

`content/plan.json` is the machine-validated source for these statuses, target
paths, known gaps, and next text. Run `make content-status` before starting a
new lesson.

## Canonical and legacy boundaries

The older fixed-income, derivatives, market-risk, and portfolio sources have
been consolidated under Modules 6, 7, and 9. Their promoted lessons now live
under `notebooks/course/` with canonical module numbers. Superseded chapter
overviews live under `chapters/legacy/`; the historical provider and portfolio
notebooks, plus the superseded time-value-of-money notebook, live under
`notebooks/legacy/`. None of those legacy files appears in the published TOCs.

Lesson identifiers are stable source identifiers, not an instruction to sort
files lexically. The TOCs define reading order. The reserved `1.2` gap remains
intentional; do not silently reuse or renumber an existing page.

## Module 0: Setup and Python Ecosystem

Goal: prepare a reproducible environment for investment analysis, quantitative labs, interactive notebooks, and applied case studies.

Required coverage:

- Python 3.12+, `pyenv`, `uv`, JupyterLab, Jupyter Book, and Git workflow;
- environment validation, tests, cached execution, and publication checks;
- foundations of `pandas`, `numpy`, visualization, Plotly, and `ipywidgets` {cite}`mckinney2010data,harris2020array,plotly2015collaborative`;
- provider connections, local caching, reusable data loaders, and audit metadata;
- folder structure for raw, interim, processed, and publication snapshot data.

## Module 1: Markets, Instruments, and Data

Goal: understand how financial markets operate and how to convert market observations into reliable analytical inputs.

Required coverage:

- market structure, participants, venues, clearing, settlement, regulation, and trading costs;
- equities, fixed income, funds, ETFs, FIBRAs, CKDs, derivatives, indices, commodities, and FX;
- quotation conventions, liquidity, market depth, calendars, settlement cycles, and price formation;
- source selection across market, macro, corporate, alternative, reference, and metadata providers;
- extraction, caching, raw/interim/processed layers, cleaning, alignment, corporate actions, revisions, and return panels;
- descriptive statistics, rolling diagnostics, correlations, dashboards, data narratives, and an integrated Mexican market case.

## Module 2: Quantitative Methods and Financial Time Series

Goal: build the statistical, probabilistic, inferential, and computational foundation that supports valuation, risk analysis, and portfolio construction.

Published core coverage:

- time value of money, compounding, annualization, discount factors, equivalent rates, and return measurement;
- descriptive statistics, seeded probability simulation, sampling uncertainty, bootstrap intervals, Monte Carlo, and sensitivity analysis;
- regression, model diagnostics, heteroskedasticity, serial correlation, omitted-variable bias, and model risk;
- stationarity, autocorrelation, ARIMA, ADF, Ljung-Box, Jarque-Bera, ARCH, GARCH, asymmetric volatility, and heavy-tailed residuals {cite}`box2015time,hamilton1994time,engle1982autoregressive,bollerslev1986generalized`;
- lagged feature engineering, chronological cross-validation, holdout benchmarking, and overfitting control.

Bayesian inference and clustering remain optional Module 2 extensions. Portfolio
mathematics and PCA belong to Module 9, while full risk backtesting belongs to
Module 7; those topics are not hidden prerequisites for the published time-series
sequence.

## Module 3: Economics, Macro, and Currency

Goal: connect economic theory with capital market expectations and investment decisions.

Required coverage:

- supply, demand, elasticity, price controls, market structure, and industrial organization;
- GDP, inflation, employment, productivity, output gaps, business cycles, and financial conditions;
- monetary policy, fiscal policy, public debt, deficits, sustainability, yield curves, and transmission mechanisms;
- exchange rates, cross-rates, triangular arbitrage, forward premium or discount, and parity conditions;
- purchasing power parity, interest rate parity, uncovered interest rate parity, carry trades, currency crises, capital controls, and balance of payments;
- macro scenarios, nowcasting discipline, economic surprises, capital market expectations, and macro dashboards using Banxico, DB.NOMICS, and global data {cite}`banxicoSIE2025,dbnomics2025,worldBankOpenData2025,imfWEO2025,oecdData2025`.

## Module 4: Financial Statement Analysis and Financial Modeling

Goal: transform financial statements into inputs for valuation, credit analysis, and business analysis.

Required coverage:

- financial statement analysis framework, filings, notes, management commentary, audit reports, and comparability;
- income statement, balance sheet, cash flow statement, ratios, DuPont analysis, ROE, ROIC, liquidity, solvency, leverage, and coverage;
- revenue recognition, expense recognition, non-recurring items, inventories, long-lived assets, intangibles, leases, pensions, stock compensation, income taxes, and reporting quality;
- banks, insurers, multinational operations, intercorporate investments, business combinations, and special-purpose entities;
- normalized earnings, owner earnings, free cash flow, pro forma models, three-statement modeling, sensitivity, stress cases, model review, and traceability.

## Module 5: Corporate Issuers and Equity Valuation

Goal: connect corporate decisions, competitive analysis, and valuation frameworks with market prices.

Required coverage:

- governance, stakeholders, working capital, liquidity, capital investments, NPV, IRR, payback, profitability index, ROIC, real options, and payout policy;
- capital structure, WACC, Modigliani-Miller, taxes, distress costs, beta, cost of equity, cost of debt, and target leverage;
- business models, industry analysis, Porter's Five Forces, PESTLE, pricing power, unit economics, and forecasting;
- market efficiency, anomalies, behavioral finance, dividend discount models, FCFF, FCFE, residual income, multiples, and sum-of-the-parts valuation;
- investment thesis, catalysts, risks, monitoring indicators, valuation memos, and equity research notes.

## Module 6: Fixed Income, Credit, and Term Structure

Goal: move from bond mathematics to complete fixed-income analysis and its use in portfolios.

Required coverage:

- bond pricing, annuities, zero-coupon and coupon bonds, clean price, dirty price, accrued interest, day counts, settlement, and yield measures;
- duration, convexity, DV01, effective duration, key-rate duration, and second-order price approximation;
- credit risk, default probability, loss given default, exposure at default, ratings, credit spreads, spread duration, downgrade risk, and liquidity premia;
- spot rates, forward rates, par rates, discount factors, bootstrapping, interpolation, Nelson-Siegel, Nelson-Siegel-Svensson, and curve PCA;
- interest-rate trees, embedded options, option-adjusted spread, convertibles, CDS, securitized products, immunization, liability-driven cases, CETES, Bonos M, and UDIBONOS.

## Module 7: Derivatives and Risk Management

Goal: understand derivative pricing, strategic use, and risk at both the instrument and portfolio levels.

Required coverage:

- forwards, futures, swaps, options, credit derivatives, payoff diagrams, long/short economics, replication, arbitrage, cost of carry, and risk-neutral pricing;
- forward valuation, futures margining, swaps, binomial trees, Black-Scholes-Merton, Greeks, delta hedging, implied volatility, smiles, skew, and option strategies {cite}`hull2022options`;
- American, Asian, barrier, and Monte Carlo pricing, Heston simulation, derivative overlays, rebalancing, hedging, and cash equitization;
- semideviation, historical VaR, parametric VaR, simulated VaR, Expected Shortfall, Cornish-Fisher VaR, volatility-weighted historical simulation, stress testing, backtesting, and traffic-light interpretation {cite}`jorion2007var,mcneil2015quantitative`;
- risk governance, limits, risk budgeting, capital allocation, pricing dashboards, Greeks dashboards, VaR dashboards, and scenario tools.

## Module 8: Alternative Investments

Goal: cover the structures, risks, valuation methods, and portfolio role of alternative assets.

Required coverage:

- private equity, venture capital, growth equity, buyouts, private debt, fund structures, fees, carried interest, waterfalls, and co-investments;
- due diligence, manager selection, benchmarking challenges, private-company valuation, direct and listed real estate, NAV, FFO, AFFO, cap rates, and NOI;
- infrastructure, natural resources, timberland, farmland, commodities, roll return, contango, backwardation, swaps, and commodity indexes;
- hedge fund strategies, leverage, liquidity terms, gates, side pockets, factor exposures, digital assets, tokenization, stale pricing, liquidity-adjusted risk, and after-fee performance;
- Mexican examples with FIBRAs, CKDs, and private capital vehicles when data availability permits.

## Module 9: Portfolio Management, Asset Allocation, and Performance

Goal: move from instrument analysis to portfolio construction, monitoring, and evaluation for different investor types.

Required coverage:

- portfolio management process, IPS, objectives, constraints, risk tolerance, return objectives, liquidity needs, investor types, and behavioral biases;
- mean-variance analysis, efficient frontier, minimum variance portfolio, tangency portfolio, CAPM, expected returns, covariance estimation, shrinkage, and resampling;
- multifactor models, APT, active risk, tracking risk, information ratio, fundamental law of active management, risk parity, HRP, and factor allocation;
- ETFs, index construction, active versus passive implementation, strategic and tactical allocation, capital market expectations, tax-aware investing, rebalancing, currency management, and liability-driven investing;
- benchmark selection, attribution, appraisal, reporting, transaction costs, implementation shortfall, turnover constraints, monitoring, governance dashboards, and investment committee cases.

## Module 10: Advanced Pathways and Capstone

Goal: close the curriculum with advanced pathways and an integrative project that combines data, valuation, risk, portfolio construction, and executive communication.

Required coverage:

- Portfolio Management pathway: index strategies, active equity, active fixed income, yield curve strategies, credit strategies, execution, and institutional cases;
- Private Markets pathway: deal screening, valuation, structuring, fund vehicles, agreements, economics, value creation, exits, capital calls, distributions, and carried interest;
- Private Wealth pathway: wealth management industry, client management, family dynamics, wealth planning, liquidity planning, concentrated positions, human capital, entrepreneurs, philanthropy, and wealth transfer;
- structured response labs, memo writing, investment committee communication, and a final capstone with real data, dashboard, valuation memo, risk memo, and presentation.

## Expansion track

The core path is complete; future work should deepen it without creating a
second curriculum.

| Expansion topic | Primary module | Integration boundary |
| --- | --- | --- |
| Bayesian inference, clustering, and broader ML cases | Module 2 | Add only with time-aware validation, an interpretable benchmark, and no duplication of later portfolio methods. |
| Nowcasting and currency-overlay dashboards | Modules 3 and 7 | Preserve vintages, parity conventions, hedging P&L, and an offline publication mode. |
| Filing-backed three-statement and equity-research cases | Modules 4 and 5 | Use licensed filings, reconcile every statement, and retain a model audit trail. |
| Calendar-aware Mexican fixed-income and curve cases | Module 6 | Review settlement calendars, day counts, quotations, identifiers, and redistribution rights. |
| Credit curves, CDS, OAS, and securitized case packs | Module 6 | Add contractual cash-flow validation and clearly separate structural assumptions from observed quotes. |
| Arbitrage-checked implied-volatility surfaces and xVA | Module 7 | Add only after numerical, counterparty, collateral, and model-governance boundaries are explicit. |
| Manager due diligence and private-wealth planning labs | Modules 8 and 10 | Use auditable inputs, privacy-safe cases, after-fee cash flows, and explicit liquidity constraints. |
| Full attribution and performance reporting | Module 9 | Reconcile benchmark, cash-flow timing, fees, taxes, turnover, and implementation shortfall. |
| Additional capstone datasets and committee cases | Module 10 | Keep one evidence contract and rubric across pathways so projects remain comparable. |
