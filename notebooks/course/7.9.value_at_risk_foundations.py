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
# # VaR and Expected Shortfall Foundations
#
# Module: Derivatives and Risk Management
#
# ## Lesson summary
#
# This notebook establishes the sign convention, notation, and first empirical estimates for Value at Risk and Expected Shortfall. The examples use real returns from the versioned Banxico official price-like snapshot, so the published output can be reproduced without inventing a return series {cite}`banxicoSIE2025,jorion2007var,mcneil2015quantitative`.
#
# ## Learning objectives
#
# By the end of this notebook, students should be able to:
#
# - convert portfolio returns into loss variables and non-negative risk metrics;
# - estimate historical VaR and Expected Shortfall from observed returns;
# - compare empirical tail estimates against Gaussian VaR;
# - explain why Expected Shortfall is more tail-sensitive than VaR;
# - read a tail-loss chart without confusing returns and losses.
#
# ## Prerequisites
#
# Complete the Module 2 return and volatility lessons first. Students should
# distinguish decimal from percentage returns, identify a lower-tail
# probability, and interpret an empirical quantile.
#
# ## Tail-risk notation
#
# Let $R_t$ be a one-period return and $L_t=-R_t$ the corresponding signed
# loss. For tail probability $\alpha$, first define the raw loss quantile
# \(v_\alpha=Q_{1-\alpha}(L)\). This book reports risk as a non-negative loss:
#
# $$
# \operatorname{VaR}_{\alpha}
# =\max\left(0,-Q_\alpha(R)\right)
# =\max(0,v_\alpha),
# $$
#
# ![Value at Risk left-tail loss diagram](../../img/generated/risk-var-tail-loss.png)
#
# and
#
# $$
# \operatorname{ES}_{\alpha}
# =\max\left(0,-\mathbb{E}\left[R\mid R\leq Q_\alpha(R)\right]\right)
# =\max\left(0,\mathbb{E}\left[L\mid L\geq v_\alpha\right]\right).
# $$
#
# ![Conditional Value at Risk expected shortfall diagram](../../img/generated/risk-cvar-expected-shortfall.png)
#
# If returns are modeled as $R\sim\mathcal{N}(\mu,\sigma^2)$, Gaussian VaR under the same positive-loss convention is
#
# $$
# \operatorname{VaR}_{\alpha}(R)
# =\max\left\{0,-\left(\mu+\sigma\Phi^{-1}(\alpha)\right)\right\}.
# $$
#
# VaR is a threshold. Expected Shortfall is an average beyond that threshold, which is why the two metrics can rank portfolios differently when tails are asymmetric or heavy {cite}`artzner1999coherent`.
#
# ![Skewness and tail orientation in return distributions](../../img/generated/risk-skewness-tail-orientation.png)
#
# ## Setup

# %% tags=["setup", "hide-input"]
import pandas as pd
import matplotlib.pyplot as plt

from src.market_data import official_price_panel, returns_from_prices
from src.market_risk import expected_shortfall, gaussian_var, historical_var

pd.options.display.float_format = "{:.6f}".format

# %% [markdown]
# ## Official return sample
#
# The first empirical example uses daily USD/MXN FIX returns from the committed Banxico snapshot. Later notebooks reuse the same panel for portfolio-level risk, backtesting, and dashboards.

# %%
price_panel = official_price_panel(start="2021-01-01", end="2026-06-05")
asset_returns = returns_from_prices(price_panel, method="log").dropna()

asset = "usd_mxn"
returns = asset_returns[asset].rename("usd_mxn_log_return")
losses = (-returns).rename("usd_mxn_loss")

sample = pd.concat(
    [price_panel[asset].rename("usd_mxn_fix"), returns, losses],
    axis=1,
).dropna()

sample.tail()

# %% [markdown]
# **Output interpretation.** The table keeps the price level, return, and signed
# loss side by side. A negative return becomes a positive loss. The VaR and
# Expected Shortfall helpers additionally floor reported risk at zero so an
# all-gain sample is not labeled as a positive risk charge.

# %%
pd.Series(
    {
        "source": price_panel.attrs.get("sources", "not recorded"),
        "start": returns.index.min().strftime("%Y-%m-%d"),
        "end": returns.index.max().strftime("%Y-%m-%d"),
        "observations": int(returns.shape[0]),
        "mean_return": returns.mean(),
        "volatility": returns.std(),
        "worst_return": returns.min(),
        "largest_loss": losses.max(),
    },
    name="usd_mxn_sample",
)

# %% [markdown]
# ## Historical and Gaussian tail estimates
#
# Historical VaR reads the empirical quantile directly from observed returns. Gaussian VaR compresses the same sample into a mean and standard deviation, then uses the normal quantile. Expected Shortfall averages losses at or beyond the empirical VaR threshold.

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

risk_table

# %% [markdown]
# **Output interpretation.** The rows are tail probabilities, not confidence levels. `alpha_0.010` corresponds to a 99% one-day loss threshold. When Expected Shortfall is materially larger than VaR, the realized tail contains losses beyond the threshold that should not be hidden by the quantile alone.

# %% [markdown]
# ## Tail-loss chart

# %%
alpha = 0.01
var_99 = historical_var(returns, alpha=alpha)
es_99 = expected_shortfall(returns, alpha=alpha)
gaussian_99 = gaussian_var(returns, alpha=alpha)

fig, ax = plt.subplots(figsize=(9, 4.8))
returns.hist(bins=70, ax=ax, color="#4f6f8f", alpha=0.78)
ax.axvline(-var_99, color="#b42318", linestyle="--", linewidth=2, label="Historical VaR")
ax.axvline(-es_99, color="#7f1d1d", linestyle=":", linewidth=2.5, label="Expected Shortfall")
ax.axvline(-gaussian_99, color="#175cd3", linestyle="-.", linewidth=2, label="Gaussian VaR")
ax.set_title("USD/MXN daily log returns and 1% tail thresholds")
ax.set_xlabel("Daily log return")
ax.set_ylabel("Frequency")
ax.legend()
fig.tight_layout()

# %% [markdown]
# **Output interpretation.** The chart is drawn in return space, so the risk thresholds appear on the left tail as negative returns. The reported table stores the same thresholds as positive losses.

# %% [markdown]
# ## Cross-asset comparison
#
# The official panel combines USD/MXN, UDI, and carry indexes constructed from official Mexican rate series. Comparing the same tail metrics across all columns shows why a portfolio risk report must state its data source, horizon, and sign convention before discussing model choice.

# %%
asset_tail_table = pd.DataFrame(
    {
        "historical_var_1pct": asset_returns.apply(historical_var, alpha=0.01),
        "gaussian_var_1pct": asset_returns.apply(gaussian_var, alpha=0.01),
        "expected_shortfall_1pct": asset_returns.apply(expected_shortfall, alpha=0.01),
        "volatility": asset_returns.std(),
    }
).sort_values("expected_shortfall_1pct", ascending=False)

asset_tail_table

# %% [markdown]
# ## Model limitations
#
# - VaR is not a worst-case loss; it is a quantile under a chosen horizon and sample.
# - Expected Shortfall is more informative about the average tail loss, but it can be noisy when few observations fall beyond the VaR threshold.
# - Gaussian VaR can be useful as a benchmark, but it should not be treated as evidence that the empirical tail is normal.
# - Tail estimates from a reproducible snapshot are appropriate for publication, while live-data runs should record provider, timestamp, transformation, and cache metadata.
#
# ## Handoff
#
# The next notebook extends this foundation from a single asset to portfolio
# semideviation, Sortino ratio, guarded Cornish-Fisher VaR, and EWMA volatility
# weighting.

# %% [markdown] tags=["exercise"]
# ## Checkpoint exercise
#
# 1. For USD/MXN, report the 1% raw return quantile, historical VaR, and
#    Expected Shortfall in decimal daily-return units.
# 2. Verify numerically that the plotted VaR line in return space is the
#    negative of the reported positive-loss threshold.
# 3. Repeat at \(\alpha=5\%\) and explain why changing tail probability changes
#    both the threshold and the number of observations averaged by ES.
#

# %% [markdown] tags=["solution"]
# ```{dropdown} Suggested answer rubric
# A complete answer preserves return and loss signs, reports both alpha levels
# and tail observation counts, demonstrates the chart/table sign conversion,
# and notes that empirical ES is noisy when the tail contains few observations.
# ```
