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
# # Downside Risk and VaR Methods
#
# Module: Derivatives and Risk Management
#
# ## Lesson summary
#
# This lab compares downside deviation, historical and parametric Value at Risk
# (VaR), Expected Shortfall (ES), Gaussian Monte Carlo, and an exponentially
# weighted moving-average (EWMA) volatility adjustment. Every estimator uses the
# same simple-return series for an equal-weight portfolio rebalanced at each
# observed US trading interval {cite}`jorion2007var,mcneil2015quantitative,riskMetrics1996`.
#
# ## Learning objectives
#
# By the end of this lab, students should be able to:
#
# - distinguish total volatility from target semideviation;
# - compute and interpret an annualized Sortino ratio;
# - compare historical, Gaussian, Cornish-Fisher, and Gaussian Monte Carlo VaR;
# - estimate ES from exactly the worst empirical probability mass; and
# - explain the timing of an EWMA one-step-ahead volatility forecast.
#
# ## Prerequisites
#
# Complete Value at Risk and Expected Shortfall Foundations first. Preserve the
# non-negative loss convention and remember that $\alpha$ is a lower return-tail
# probability, not a confidence level.
#
# ## Risk metric definitions
#
# For per-interval target return $\tau$ and $T$ observations, target
# semideviation is
#
# $$
# \sigma_-(\tau)
# =\sqrt{\frac{1}{T}\sum_{t=1}^{T}\min(r_t-\tau,0)^2}.
# $$
#
# With $N$ observed intervals per year, the course helper annualizes both the
# arithmetic excess return and the semideviation:
#
# $$
# \operatorname{Sortino}_{N}
# =\frac{N(\bar r-\tau)}{\sqrt{N}\,\sigma_-(\tau)}
# =\sqrt{N}\,\frac{\bar r-\tau}{\sigma_-(\tau)}.
# $$
#
# This downside-risk performance formulation follows Sortino and Price
# {cite}`sortinoPrice1994`.
#
# VaR and ES are reported as positive loss magnitudes:
#
# $$
# \operatorname{VaR}_{\alpha}(R)
# =\max\{0,-Q_\alpha(R)\},
# $$
#
# $$
# \operatorname{ES}_{\alpha}(R)
# =\max\left\{0,-\frac{1}{\alpha}
# \int_0^\alpha Q_u(R)\,du\right\}.
# $$
#
# The conditional-tail expectation is equivalent only under suitable
# continuity at the quantile. The empirical helper instead uses complete worst
# observations plus the fractional boundary observation required to represent
# exactly $\alpha T$ observations {cite}`acerbiTasche2002,rockafellarUryasev2002`.
#
# The Cornish-Fisher approximation modifies a Gaussian quantile with sample
# skewness $S$ and excess kurtosis $K$:
#
# $$
# z_{CF}=z+\frac{z^2-1}{6}S+\frac{z^3-3z}{24}K
# -\frac{2z^3-5z}{36}S^2.
# $$
#
# The course applies a moment guardrail rather than treating this asymptotic
# expansion as universally reliable {cite}`cornishFisher1938`.
#
# ## Setup

# %% tags=["setup", "hide-input"]
import pandas as pd
from scipy.stats import kurtosis, skew

from src.market_data import nasdaq_stock_price_panel, returns_from_prices
from src.market_risk import (
    cornish_fisher_moment_report,
    cornish_fisher_var,
    ewma_next_volatility,
    ewma_volatility,
    expected_shortfall,
    gaussian_monte_carlo_var,
    gaussian_var,
    historical_var,
    rebalanced_portfolio_returns,
    sortino_ratio,
    target_semideviation,
    volatility_weighted_historical_var,
)
from src.module7_visuals import build_ewma_volatility_figure

pd.options.display.float_format = "{:.6f}".format

# %% [markdown]
# ## Observed equal-weight equity portfolio
#
# The committed snapshot contains provider-adjusted closes in USD for AAPL,
# MSFT, NVDA, AMZN, and GOOGL. Prices cover 2021-01-04 through 2026-06-05;
# simple returns begin on the next observed trading date. The fixed 20% weights
# are restored after every observed interval, so the portfolio return is
# $R_{p,t}=\sum_i w_iR_{i,t}$. This is an explicit rebalancing assumption, not a
# buy-and-hold reconstruction {cite}`yfinance2025,yahooFinanceCoverage2026,yahooTerms2026`.

# %%
price_panel = nasdaq_stock_price_panel(start="2021-01-04", end="2026-06-05")
asset_returns = returns_from_prices(price_panel, method="simple").dropna(how="any")
asset_returns.attrs = {
    **price_panel.attrs,
    "method": "simple returns; equal weights rebalanced each observed interval",
}

weights = pd.Series(
    {
        "AAPL": 0.20,
        "MSFT": 0.20,
        "NVDA": 0.20,
        "AMZN": 0.20,
        "GOOGL": 0.20,
    },
    name="weight",
)
portfolio_returns = rebalanced_portfolio_returns(asset_returns, weights)
portfolio_returns.name = "equal_weight_portfolio_simple_return"

asset_returns.head()

# %%
pd.Series(
    {
        "source": price_panel.attrs["sources"],
        "field_and_currency": "provider-adjusted close, USD",
        "price_sample": f"{price_panel.index.min():%Y-%m-%d} to {price_panel.index.max():%Y-%m-%d}",
        "return_sample": (
            f"{portfolio_returns.index.min():%Y-%m-%d} to {portfolio_returns.index.max():%Y-%m-%d}"
        ),
        "frequency": "observed US trading intervals; no calendar filling",
        "snapshot_generated_at": "2026-06-07T04:55:41.700953+00:00",
        "portfolio_rule": "20% each; rebalanced after every observed interval",
        "omitted": "transaction costs, taxes, and FX conversion",
        "rights_review": "provenance recorded; redistribution rights not independently verified",
    },
    name="data_and_portfolio_contract",
)

# %% [markdown]
# The versioned snapshot makes the calculation reproducible, but its provenance
# does not independently establish redistribution or downstream-use rights.
#
# ## Distribution diagnostics

# %%
pd.DataFrame(
    {
        "mean_simple_return_per_observed_interval": asset_returns.mean(),
        "volatility_per_observed_interval": asset_returns.std(),
        "skewness": asset_returns.apply(lambda series: skew(series, bias=False)),
        "excess_kurtosis": asset_returns.apply(
            lambda series: kurtosis(series, fisher=True, bias=False)
        ),
    }
)

# %%
pd.Series(
    {
        "portfolio_mean_simple_return_per_observed_interval": portfolio_returns.mean(),
        "portfolio_volatility_per_observed_interval": portfolio_returns.std(),
        "portfolio_skewness": skew(portfolio_returns, bias=False),
        "portfolio_excess_kurtosis": kurtosis(
            portfolio_returns,
            fisher=True,
            bias=False,
        ),
    },
    name="per_observed_US_trading_interval",
)

# %% [markdown]
# ## Downside risk
#
# Volatility penalizes positive and negative surprises symmetrically.
# Semideviation measures only shortfalls from the stated target. The following
# annualization assumes 252 intervals per year; it is a scaling convention, not
# a claim that every calendar year contains exactly 252 observations.

# %%
periods_per_year = 252
target = 0.0

downside_report = pd.Series(
    {
        "target_per_interval": target,
        "target_semideviation_per_interval": target_semideviation(
            portfolio_returns,
            target=target,
        ),
        "annualized_target_semideviation": target_semideviation(
            portfolio_returns,
            target=target,
            periods_per_year=periods_per_year,
        ),
        "annualized_sortino_ratio": sortino_ratio(
            portfolio_returns,
            target=target,
            periods_per_year=periods_per_year,
        ),
    },
    name="downside_report",
)
downside_report

# %% [markdown]
# **Output interpretation.** A high Sortino ratio can still coexist with a
# material tail loss. It is a mean-to-downside-deviation ratio, not a replacement
# for VaR, ES, stress testing, or a drawdown analysis.
#
# ## VaR, Expected Shortfall, and Gaussian Monte Carlo
#
# Gaussian Monte Carlo below simulates independent one-interval draws using the
# sample mean and standard deviation. The fixed seed makes the numerical
# benchmark reproducible; it does not make the Gaussian model empirically true.

# %%
alpha = 0.01
cornish_fisher_report = cornish_fisher_moment_report(portfolio_returns)
try:
    cornish_fisher_estimate = cornish_fisher_var(portfolio_returns, alpha=alpha)
    cornish_fisher_status = "available"
except ValueError as exc:
    cornish_fisher_estimate = float("nan")
    cornish_fisher_status = f"unavailable: {exc}"

risk_table = pd.Series(
    {
        "historical_var": historical_var(portfolio_returns, alpha=alpha),
        "gaussian_closed_form_var": gaussian_var(portfolio_returns, alpha=alpha),
        "gaussian_monte_carlo_var": gaussian_monte_carlo_var(
            portfolio_returns,
            confidence=1 - alpha,
            simulations=100_000,
            seed=20260722,
        ),
        "cornish_fisher_var": cornish_fisher_estimate,
        "volatility_weighted_historical_var": volatility_weighted_historical_var(
            portfolio_returns,
            alpha=alpha,
            lambda_=0.94,
        ),
        "expected_shortfall": expected_shortfall(portfolio_returns, alpha=alpha),
    },
    name="positive_loss_per_observed_US_trading_interval",
).to_frame()
risk_table

# %%
pd.concat(
    [
        cornish_fisher_report,
        pd.Series({"status": cornish_fisher_status}),
    ]
).rename("cornish_fisher_diagnostic")

# %% [markdown]
# **Output interpretation.** Differences across rows are model-risk evidence.
# Historical VaR reads the observed sample tail; Gaussian methods impose
# symmetry; Cornish-Fisher reacts to sample skewness and kurtosis; and
# volatility-weighted historical VaR rescales observations to the latest EWMA
# state. A close Monte Carlo and closed-form Gaussian result checks the
# simulation implementation, not the normality assumption.
#
# ## EWMA timing and next-state forecast
#
# With decay $\lambda$, the forecast available after observing return $r_t$ is
#
# $$
# \widehat\sigma_{t+1\mid t}^{2}
# =\lambda\widehat\sigma_{t\mid t-1}^{2}
# +(1-\lambda)r_t^2.
# $$
#
# The value used to scale the next interval must therefore incorporate the last
# observed return. The commonly cited RiskMetrics daily convention uses
# $\lambda=0.94$ {cite}`riskMetrics1996`.

# %%
lambda_ = 0.94
ewma_state = ewma_volatility(portfolio_returns, lambda_=lambda_)
next_ewma_forecast = ewma_next_volatility(portfolio_returns, lambda_=lambda_)

pd.Series(
    {
        "last_observed_return": portfolio_returns.iloc[-1],
        "sigma_t_given_t_minus_1": ewma_state.iloc[-1],
        "sigma_t_plus_1_given_t": next_ewma_forecast,
        "lambda": lambda_,
    },
    name="EWMA_timing_check",
)

# %% mystnb={"image": {"alt": "Two aligned time-series panels show equal-weight portfolio simple returns and the descriptive EWMA volatility state for observed US trading intervals from January 2021 through June 2026."}}
ewma_figure = build_ewma_volatility_figure(portfolio_returns, ewma_state)
ewma_figure

# %% [markdown]
# The full-sample default used to initialize this descriptive EWMA path is not a
# valid no-look-ahead backtest initialization. A production filter would set its
# initial variance from a prior training period and preserve the complete state
# history.
#
# ## Interpretation checklist
#
# | Question | Evidence to inspect |
# | --- | --- |
# | Is the sample symmetric? | Skewness and large negative simple returns |
# | Does the Gaussian model matter? | Gaussian versus historical VaR |
# | Does simulation reproduce its own model? | Monte Carlo versus closed-form Gaussian VaR |
# | Is the tail material? | Gap between historical VaR and empirical ES |
# | Has volatility moved recently? | EWMA path and the $t+1\mid t$ state |
# | Is Cornish-Fisher defensible here? | Moment report and guardrail status |
#
# ## Model limitations
#
# - All estimates depend on horizon, sample, weighting, sign, and return conventions.
# - Equal weights rebalanced every interval are hypothetical and omit transaction
#   costs, taxes, foreign-exchange conversion, capacity, and trading constraints.
# - Historical methods cannot reveal events absent from the sample.
# - Gaussian Monte Carlo preserves the fitted normal model's omissions, including
#   nonlinear dependence, volatility clustering, and heavy tails.
# - Empirical ES is tail-sensitive but statistically noisy at small $\alpha$.
#
# ## Handoff
#
# The next lab separates estimation from validation. It freezes each rolling
# forecast before observing the corresponding return, tests exception coverage
# and dependence, and adds an adverse deterministic scenario.
