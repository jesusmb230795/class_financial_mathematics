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
# # Value at Risk and Expected Shortfall Foundations
#
# Module: Derivatives and Risk Management
#
# ## Lesson summary
#
# This notebook fixes one sign convention and one data contract before comparing
# tail-risk estimators. It uses simple returns from a committed panel of five
# provider-adjusted US equity closes; no synthetic price or return history is
# mixed into the sample. Value at Risk (VaR) remains a quantile, whereas Expected
# Shortfall (ES) measures the average loss over exactly the worst probability
# mass {cite}`jorion2007var,mcneil2015quantitative,acerbiTasche2002,rockafellarUryasev2002`.
#
# ## Learning objectives
#
# By the end of this notebook, students should be able to:
#
# - convert a simple return into a signed loss and a non-negative reported risk;
# - distinguish a lower-tail probability from a confidence level;
# - estimate empirical VaR and ES without mishandling a finite-sample boundary;
# - compare empirical tail estimates with Gaussian VaR; and
# - read VaR and ES annotations in positive-loss space without reversing signs.
#
# ## Prerequisites
#
# Complete the Module 2 return and volatility lessons first. Students should
# distinguish decimal from percentage returns, identify a lower-tail
# probability, and interpret an empirical quantile.
#
# ## Tail-risk notation
#
# Let $R$ denote the simple return over one observed US trading interval and
# $L=-R$ the corresponding signed loss. For a lower-tail probability
# $\alpha\in(0,1)$, this book reports VaR as a non-negative loss magnitude:
#
# $$
# \operatorname{VaR}_{\alpha}(R)
# =\max\left\{0,-Q_{\alpha}(R)\right\}.
# $$
#
# ![A one-period loss density with gains to the left, a dashed 99 percent Value at Risk cutoff, and the worst 1 percent probability mass shaded to the right as more severe loss.](../../img/generated/risk-var-tail-loss.png)
#
# VaR is a threshold, not the worst possible loss. The diagram is drawn in
# loss space, so larger losses appear farther to the right.
#
# The definition of ES that remains valid for discrete empirical samples is the
# lower-quantile integral:
#
# $$
# \operatorname{ES}_{\alpha}(R)
# =\max\left\{0,-\frac{1}{\alpha}
# \int_{0}^{\alpha}Q_u(R)\,du\right\}.
# $$
#
# For a continuous return distribution with no probability mass at
# $Q_\alpha(R)$, this reduces to
#
# $$
# \operatorname{ES}_{\alpha}(R)
# =\max\left\{0,-\mathbb{E}\!\left[R\mid
# R\leq Q_\alpha(R)\right]\right\}.
# $$
#
# In a finite sample, the implementation averages complete worst observations
# plus the fraction of the boundary observation needed to represent exactly
# $\alpha T$ observations. A conditional average below an interpolated sample
# quantile need not produce that same estimator {cite}`acerbiTasche2002,rockafellarUryasev2002`.
#
# ![A one-period loss density with the worst 1 percent tail shaded beyond the 99 percent Value at Risk cutoff and a diamond marking Expected Shortfall as the average loss within that tail, not as another cutoff.](../../img/generated/risk-cvar-expected-shortfall.png)
#
# Expected Shortfall summarizes the entire selected probability mass rather
# than only its boundary.
#
# If $R\sim\mathcal{N}(\mu,\sigma^2)$, Gaussian VaR under the same convention is
#
# $$
# \operatorname{VaR}_{\alpha}^{\mathrm{Gaussian}}(R)
# =\max\left\{0,-\left(\mu+\sigma\Phi^{-1}(\alpha)\right)\right\}.
# $$
#
# VaR is not generally coherent, whereas ES satisfies the coherent-risk axioms
# under its standard loss formulation {cite}`artzner1999coherent,acerbiTasche2002`.
#
# ![Three standardized return-density panels compare positive, symmetric, and negative skew; each panel shades its characteristic tail and marks the median calculated from the plotted distribution.](../../img/generated/risk-skewness-tail-orientation.png)
#
# Skewness changes tail shape. For a long position, loss risk still comes from
# negative returns even when the diagram also highlights the characteristic
# direction of positive or negative skew.
#
# ## Setup

# %% tags=["setup", "hide-input"]
import pandas as pd

from src.market_data import nasdaq_stock_price_panel, returns_from_prices
from src.market_risk import expected_shortfall, gaussian_var, historical_var
from src.module7_visuals import build_tail_risk_figure

pd.options.display.float_format = "{:.6f}".format

# %% [markdown]
# ## Versioned observed return sample
#
# The committed snapshot contains provider-adjusted closing prices in USD for
# AAPL, MSFT, NVDA, AMZN, and GOOGL. The requested price window runs from
# 2021-01-04 through 2026-06-05 on provider trading dates. Simple returns begin
# one observed interval later because the first price has no prior observation.
# The snapshot was generated on 2026-06-07 through `yfinance` from Yahoo-derived
# data {cite}`yfinance2025,yahooFinanceCoverage2026,yahooTerms2026`.

# %%
price_panel = nasdaq_stock_price_panel(start="2021-01-04", end="2026-06-05")
asset_returns = returns_from_prices(price_panel, method="simple").dropna(how="any")
asset_returns.attrs = {
    **price_panel.attrs,
    "method": "simple return over each observed US trading interval",
}

asset = "AAPL"
returns = asset_returns[asset].rename("AAPL_simple_return")
returns.attrs = dict(asset_returns.attrs)
losses = (-returns).rename("AAPL_signed_loss")

sample = pd.concat(
    [price_panel[asset].rename("AAPL_adjusted_close_USD"), returns, losses],
    axis=1,
).dropna()
sample.tail()

# %% [markdown]
# **Output interpretation.** A negative simple return becomes a positive signed
# loss. The risk helpers additionally floor the reported metric at zero, so an
# all-gain sample is not presented as a positive loss charge.

# %%
pd.Series(
    {
        "source": price_panel.attrs["sources"],
        "field_and_currency": "provider-adjusted close, USD",
        "price_sample": f"{price_panel.index.min():%Y-%m-%d} to {price_panel.index.max():%Y-%m-%d}",
        "return_sample": f"{returns.index.min():%Y-%m-%d} to {returns.index.max():%Y-%m-%d}",
        "frequency": "observed US trading intervals; no calendar filling",
        "snapshot_generated_at": "2026-06-07T04:55:41.700953+00:00",
        "observations": int(returns.shape[0]),
        "rights_review": "provenance recorded; redistribution rights not independently verified",
    },
    name="data_contract",
)

# %% [markdown]
# Recording a provider, field, currency, dates, and transformation makes the
# calculation reproducible. It does not itself establish redistribution or
# downstream-use rights; those rights require a separate review.
#
# ## Historical, Gaussian, and tail-average estimates
#
# Historical VaR reads the empirical quantile. Gaussian VaR compresses the
# sample into a mean and standard deviation. Empirical ES integrates the worst
# $\alpha$ probability mass, including a fractional boundary observation when
# $\alpha T$ is not an integer.

# %%
alpha_levels = [0.05, 0.025, 0.01]

risk_table = pd.DataFrame(
    {
        f"alpha_{alpha:.3f}": {
            "historical_var": historical_var(returns, alpha=alpha),
            "gaussian_var": gaussian_var(returns, alpha=alpha),
            "expected_shortfall": expected_shortfall(returns, alpha=alpha),
        }
        for alpha in alpha_levels
    }
).T
risk_table.index.name = "lower_return_tail_probability"
risk_table.columns.name = "positive_loss_per_observed_US_trading_interval"
risk_table

# %% [markdown]
# **Output interpretation.** `alpha_0.010` means a 1% lower return tail, or a
# 99% confidence convention. It does not mean that 1% is itself the confidence
# level. ES is at least as tail-sensitive as the VaR boundary because it uses
# losses throughout the selected probability mass.
#
# ## Tail-loss chart

# %% mystnb={"image": {"alt": "A histogram expresses AAPL simple returns as signed losses over observed US trading intervals; the positive-loss axis marks historical Value at Risk, Gaussian Value at Risk, and the Expected Shortfall tail mean at alpha equal to one percent."}}
alpha = 0.01
var_99 = historical_var(returns, alpha=alpha)
es_99 = expected_shortfall(returns, alpha=alpha)
gaussian_99 = gaussian_var(returns, alpha=alpha)

tail_figure = build_tail_risk_figure(
    returns,
    historical_var=var_99,
    expected_shortfall=es_99,
    gaussian_var=gaussian_99,
)
tail_figure

# %% [markdown]
# **Output interpretation.** The figure converts returns into signed loss space,
# so larger adverse outcomes appear farther to the right. ES is shown as a tail
# average, not as an additional empirical quantile.
#
# ## Cross-asset comparison
#
# Applying the same convention to all five equities isolates cross-sectional
# differences without combining incompatible units or synthetic carry indexes.

# %%
asset_tail_table = pd.DataFrame(
    {
        "historical_var_1pct": asset_returns.apply(historical_var, alpha=0.01),
        "gaussian_var_1pct": asset_returns.apply(gaussian_var, alpha=0.01),
        "expected_shortfall_1pct": asset_returns.apply(expected_shortfall, alpha=0.01),
        "simple_return_volatility": asset_returns.std(),
    }
).sort_values("expected_shortfall_1pct", ascending=False)
asset_tail_table.columns.name = "per_observed_US_trading_interval"
asset_tail_table

# %% [markdown]
# ## Model limitations
#
# - VaR is not a worst-case loss; it is a quantile for a stated horizon and sample.
# - Empirical ES is more informative about the selected tail mass, but a 1% tail
#   remains statistically sparse in a sample of this length.
# - Gaussian VaR is a benchmark, not evidence that observed equity returns are normal.
# - Provider-adjusted closes embed the provider's corporate-action treatment;
#   the notebook does not independently reconstruct those adjustments.
# - No transaction costs, taxes, foreign-exchange conversion, or investor-specific
#   constraints enter this single-asset illustration.
#
# ## Handoff
#
# The next notebook extends this foundation to an exactly rebalanced five-equity
# portfolio, downside deviation, guarded Cornish-Fisher VaR, Gaussian Monte Carlo,
# and volatility-weighted historical simulation.
