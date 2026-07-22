# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Market Data Quality Framework
#
# Module: Markets, Instruments, and Data
#
# ## Lesson summary
#
# Financial data analysis starts before the first model is fit. Market data must be interpreted through the trading venue, instrument type, calendar, corporate action policy, source limitations, and data-cleaning assumptions that produced it {cite}`fabozzi2019foundations,tsay2010analysis`.
#
# This lesson synthesizes the data-quality part of Module 1 into a practical framework for actuarial and quantitative finance work. A structured table can still contain serious errors: missing dates, stale prices, inconsistent calendars, duplicated records, incorrect units, unadjusted corporate actions, or undocumented macro revisions.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - identify the main institutions and data sources in the Mexican financial market;
# - distinguish price, volume, macroeconomic, fixed-income, FX, and derivative data;
# - build a source inventory before modeling;
# - distinguish missing dates from missing values;
# - detect missing data, asynchronous calendars, frequency mismatches, outliers, stale prices, revisions, and corporate action problems;
# - transform adjusted prices into simple and log returns;
# - document data assumptions clearly enough for a risk, actuarial, or reproducibility review.
#
# ## Prerequisites
#
# Complete `1.3`. Readers should know the field, unit, provider, and calendar of
# a series and be comfortable with percentages and logarithms before applying a
# missing-data, return, or outlier rule.
#
# ## Setup
#
# Use the deterministic classroom examples and helper functions from `src.market_data_quality`; no external provider call is required for the default run.
#
# The short 2026 series below is a synthetic, deterministic quality ticket. It
# deliberately omits one business date, contains one explicit missing value,
# and includes one extreme price jump followed by a reversal. These features
# make the audit steps visible; they are not claims about a real instrument. The
# example is never combined with the empirical 2021-01-01 through 2025-06-30
# Module 1 panels.
#
# ## Mexican market structure
#
# | Institution or venue | Role in the data workflow | Typical course use |
# | --- | --- | --- |
# | Banxico | Central bank, monetary policy, exchange rates, rates, macro-financial series | Target rate, TIIE, CETES, FIX, UDI {cite}`banxicoSIE2025` |
# | [CNBV](https://www.gob.mx/cnbv/es/que-hacemos) | Financial-sector authorization, regulation, supervision, and enforcement | Institutional and regulatory context |
# | BMV and [BIVA](https://www.biva.mx/nosotros/acerca_de) | Equity and listed-securities venues | Mexican equity prices and index context {cite}`bmvMarketData` |
# | MexDer | Listed derivatives venue | FX, TIIE, and equity-index hedging examples {cite}`grupoBmvAbout` |
# | [PiP](https://priv.piplatam.com/Private/Index) and [Valmer](https://valmer.com.mx/es/valmer/home) | Price and valuation-data providers | Valuation context for less liquid instruments {cite}`grupoBmvAbout` |
#
# The key modeling lesson is that prices are not abstract numbers. They are produced by institutions, market conventions, trading calendars, and quotation rules.
#
# ## Data source inventory
#
# Every dataset used in the course should be documented before modeling:
#
# | Field | Example |
# | --- | --- |
# | Provider | Banxico, DB.NOMICS, Yahoo Finance, exchange file, instructor sample {cite}`banxicoSIE2025,dbnomics2025,yfinance2025` |
# | Instrument or variable | `^MXX`, USD/MXN FIX, 28-day TIIE, CPI |
# | Frequency | daily, weekly, monthly, intraday |
# | Date range | first and last available observation |
# | Price field | close, adjusted close, settlement price, bid, ask, mid |
# | Quote or denomination unit | MXN, USD, MXN per USD, MXN per UDI, percent, basis points, or real terms |
# | Calendar | local market calendar, US business calendar, merged business calendar |
# | Known limitations | missing values, rate limits, unofficial endpoints, revisions |
#
# ## Quality problem map
#
# Structure is not the same as quality. The most common problems in financial and macroeconomic datasets are:
#
# | Problem | Example |
# | --- | --- |
# | Missing dates | A market holiday, API outage, or absent observation |
# | Missing values | A price field is blank for one instrument |
# | Duplicates | The same date and ticker appear more than once |
# | Calendar mismatch | U.S. and Mexican markets have different holidays |
# | Frequency mismatch | Daily prices are merged with monthly inflation |
# | Unit mismatch | Percent, decimal, index level, pesos, dollars, and basis points are mixed |
# | Stale prices | An illiquid asset repeats the same value for many days |
# | Outliers | A return appears extremely large because of a true event or a data error |
# | Corporate actions | Splits or dividends are not reflected correctly |
# | Revisions | Macroeconomic values change after initial publication |
# | Survivorship bias | Only currently listed instruments are included |
# | Look-ahead bias | Future information is accidentally used in a historical decision |
#
# The purpose is not to solve every advanced bias immediately. The purpose is to train students to suspect the data before trusting the result.
#
# ## Example Banxico series
#
# These series are useful for future Mexican market notebooks. Verify availability before live classroom use.
#
# | Series | Meaning | Possible use |
# | --- | --- | --- |
# | `SF61745` | Target rate | monetary policy context |
# | `SF60648` | 28-day TIIE | interbank and floating-rate examples |
# | `SF60633` | 28-day CETES | short sovereign-rate reference |
# | `SF43718` | USD/MXN FIX | FX risk and macro dashboards |
# | `SP68257` | UDI | inflation-linked valuation context |
#
# The 28-day TIIE series also has a documented methodology breakpoint. Effective
# 2025-01-01, Banco de México changed its methodology from submitted bank quotes
# to a market-transaction-based method. Analyses spanning that date must disclose
# the break and must not interpret the full history as one unchanged measurement
# process {cite}`banxicoTIIETransition2025`.
#
# ## Data quality pipeline
#
# 1. Define the instrument universe and provider.
# 2. Download raw data and preserve the original field names.
# 3. Build a data source inventory.
# 4. Normalize the index to a `DatetimeIndex`.
# 5. Align calendars only after understanding market holidays.
# 6. Use adjusted prices for equity return calculations.
# 7. Convert prices to simple or log returns.
# 8. Audit missingness and outliers.
# 9. Document every cleaning assumption.
# 10. Pass only cleaned data into models.
#
# ## Missing dates versus missing values
#
# A **missing date** means an expected date is absent from the index. A **missing value** means the date exists, but one or more fields are empty.
#
# This distinction matters because a missing trading day may be normal if the market was closed, while a missing price on an active trading day may indicate an extraction or data-quality issue. A correct pipeline should distinguish among market closed, data not yet published, provider error, source unavailability, and values that are not applicable to the instrument.
#
# ## Calendar and frequency differences
#
# Financial instruments do not all follow the same calendar. Mexican equities, U.S. equities, government securities, exchange rates, and macroeconomic indicators may have different holidays, time zones, and publication schedules.
#
# A typical panel may combine:
#
# - daily stock prices;
# - daily exchange rates;
# - weekly monetary data;
# - monthly inflation;
# - monthly industrial activity;
# - quarterly GDP;
# - annual financial statements.
#
# These series cannot be merged mechanically. The analyst must define the alignment rule. Examples include using month-end values for market variables, assigning macro releases to publication dates instead of reference periods, avoiding forward-fill unless the assumption is explicit, and using only information available at the historical decision date when testing a strategy.
#
# ## Returns
#
# Simple return:
#
# $$
# R_t = \frac{P_t - P_{t-1}}{P_{t-1}}.
# $$
#
# Log return:
#
# $$
# r_t = \ln(P_t) - \ln(P_{t-1}).
# $$
#
# Log returns are additive across time and are the default input for many statistical models of asset returns {cite}`tsay2010analysis`.

# %% tags=["setup", "hide-input"]
import pandas as pd

from src.market_data_quality import data_quality_report, log_returns, simple_returns

# %%
prices = pd.Series(
    [100.0, 100.4, 100.8, float("nan"), 101.0, 101.3, 101.6, 140.0, 101.8, 102.0, 102.2],
    index=pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-05",
            "2026-01-07",
            "2026-01-08",
            "2026-01-09",
            "2026-01-12",
            "2026-01-13",
            "2026-01-14",
            "2026-01-15",
            "2026-01-16",
            "2026-01-19",
        ]
    ),
    name="example_price",
)

return_input = prices.dropna()
pd.DataFrame(
    {
        "price": return_input,
        "simple_return": simple_returns(return_input),
        "log_return": log_returns(return_input),
    }
)

# %% [markdown]
# **Output interpretation.**
#
# The table computes returns only from non-missing observed prices. A return after
# a gap spans the last two available observations; it is not automatically a
# one-business-day return. The extreme jump and reversal are intentional review
# candidates, not values to delete automatically.
#

# %% [markdown]
# ## Missing data and asynchronous calendars
#
# Cross-market data often mixes Mexican and U.S. holidays. Dropping every row
# with a missing observation can remove useful information and distort
# correlations. Forward-filling a last-traded mark is permissible only when the
# analysis intentionally maps a closed-market value onto a declared reference
# calendar, limits the fill horizon, and reports the induced zero changes. It is
# not a default cleaning step before volatility or correlation analysis.
#
# | Data type | Defensible treatment | Warning |
# | --- | --- | --- |
# | Adjusted equity prices | preserve the native calendar; align only for a declared comparison | a fill creates unchanged marks and must be bounded and disclosed |
# | Returns | compute after price alignment | do not forward-fill returns directly |
# | Macroeconomic levels | align by release frequency | revisions and publication lags matter |
# | Yield curve points | interpolate carefully | avoid artificial zero-volatility segments |

# %%
from src.market_data_quality import align_to_business_calendar

aligned_unfilled = align_to_business_calendar(prices, fill_method=None)
aligned_with_explicit_ffill = align_to_business_calendar(prices, fill_method="ffill")
alignment_audit = pd.concat(
    {
        "observed": prices,
        "business_calendar_unfilled": aligned_unfilled,
        "explicit_ffill_example": aligned_with_explicit_ffill,
    },
    axis=1,
)
alignment_audit.loc["2026-01-02":"2026-01-09"]

# %%
data_quality_report(alignment_audit)

# %% [markdown]
# **Output interpretation.**
#
# The comparison separates two defects: 2026-01-06 is absent from the observed
# index, while 2026-01-08 exists with a missing value. The unfilled business
# calendar exposes both gaps. The explicitly filled column shows how a fill rule
# changes the data definition; it is displayed for audit, not endorsed as the
# modeling input.
#

# %% [markdown]
# ## Data revisions
#
# Macroeconomic data may be revised after initial publication. This creates a distinction between latest available data, first-release data, vintage data, and real-time data available on a historical date. A strategy tested using today's revised macro data may accidentally use information that was not available at the time of the decision {cite}`dbnomics2025`.
#
# For a classroom dashboard, latest data can be acceptable if the limitation is stated. For a historical simulation, the release calendar and revision policy become part of the model design.
#
# ## Corporate actions
#
# Raw equity closes intentionally measure price return. A split can create a
# mechanical discontinuity in that series; an ex-dividend decline is an economic
# price return but omits the cash distribution from total return. Use a
# provider-documented adjusted series only after confirming whether and how it
# treats splits, dividends, and reinvestment. Label the result as a cumulative
# adjusted-close return unless the provider methodology supports a stronger
# total-return claim.
#
# When using `yfinance`, prefer adjusted price fields or use `auto_adjust=True` when appropriate. Always state which field was used.
#
# ## Robust outlier flags
#
# Financial returns are heavy-tailed, so a large move is not automatically bad data. The goal is to flag observations for audit, not to erase real market stress.
#
# The Hampel idea compares each observation with a rolling local median {cite}`hampel1974influence`:
#
# $$
# MAD_t = \text{median}(|X_{t-k} - M_t|,\dots,|X_{t+k}-M_t|).
# $$
#
# An observation is flagged when:
#
# $$
# |X_t - M_t| > n \times 1.4826 \times MAD_t.
# $$

# %%
from src.market_data_quality import hampel_outlier_flags

returns = log_returns(prices.dropna())
hampel_outlier_flags(returns, window=7, n_sigmas=3.0)

# %% [markdown]
# **Output interpretation.**
#
# The Hampel output flags returns that are unusual relative to a local window. A flag is a prompt for investigation; it does not prove an observation is wrong or should be removed.
#

# %% [markdown]
# ## Validation checklist
#
# Before using a dataset in a model or dashboard, confirm:
#
# ```text
# 1. Are dates parsed correctly?
# 2. Are there duplicated rows?
# 3. Are expected columns present?
# 4. Are numeric columns numeric?
# 5. Are units documented?
# 6. Are missing dates expected?
# 7. Are missing values explained?
# 8. Are extreme values flagged?
# 9. Are series observed, provider-adjusted, or constructed?
# 10. Are time zones and calendars documented?
# 11. Are source and retrieval date stored?
# 12. Are transformations reproducible?
# ```
#
# Validation does not guarantee truth, but it reduces avoidable errors.
#
# ## Assumptions log
#
# Every dataset should include a short assumptions log.
#
# | Field | Example |
# | --- | --- |
# | Dataset | `mx_market_panel.parquet` |
# | Source | Banxico, INEGI, BMV, or Yahoo Finance accessed through `yfinance` |
# | Retrieval date | `YYYY-MM-DD` |
# | Calendar | Joint observed USD/MXN and UDI dates for the level panel; provider/reference calendars elsewhere |
# | Missing values | No fill for observed levels; synthetic carry accrues daily using the last published rate |
# | Equity prices | Provider-documented adjusted close; methodology retained |
# | Inflation | Monthly year-over-year CPI rate; displayed as percent |
# | Exchange rate | Banxico FIX observation, quoted as MXN per USD |
# | Known limitations | Convenience-source terms and intended use require separate review; repository inclusion does not establish downstream rights |
#
# A documented assumption is not automatically correct, but it is auditable.
#
# ## Handoff
#
# Data quality decisions affect later modules:
#
# | Decision | Downstream effect |
# | --- | --- |
# | adjusted versus raw prices | return calculation, VaR, volatility, portfolio optimization |
# | calendar alignment | correlation, covariance, beta, spread analysis |
# | outlier treatment | volatility estimates, GARCH parameters, tail-risk metrics |
# | macro release frequency | time series model selection and interpretation |
# | currency and inflation units | fixed-income and real-return analysis |
#
# This quality log is the gate for both EDA branches and the Mexican pipeline.
# A downstream notebook should receive accepted values together with the source
# inventory, calendar rule, missingness counts, outlier review, and
# transformation labels. Apply it to the NASDAQ panel before `1.4`, to the macro
# panel before `1.5`, and to the official-source levels before `1.7`.
