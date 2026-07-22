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
# # ARIMA Selection and Diagnostic Workflow
#
# Module: Quantitative Methods and Financial Time Series
#
# ## Lesson summary
#
# This lab turns the pre-model evidence from Lesson 2.3 into a declared ARIMA
# selection rule. Candidate orders are ranked on an earlier training sample,
# the selected mean model is compared with a chronological benchmark on a later
# evaluation sample, and a full-sample refit is checked for residual
# autocorrelation and non-Gaussianity {cite}`box2015time,hamilton1994time`.
#
# ## Learning objectives
#
# By the end of this lab, readers should be able to:
#
# - distinguish the order-selection sample from the later evaluation sample;
# - select the lowest-AIC admissible candidate in a declared grid and report
#   whether BIC agrees;
# - verify convergence, stationarity, and invertibility before interpretation;
# - compare a fixed-origin ARIMA forecast with a training-mean benchmark on
#   later observations;
# - run residual autocorrelation and normality checks; and
# - write a model-selection conclusion that separates in-sample fit,
#   out-of-sample error, and remaining model risk.
#
# ## Prerequisites
#
# [Lesson 2.3](2.3.time_series_diagnostics_and_volatility_extensions.ipynb)
# establishes the stationarity decision and explains why its
# Ljung-Box calculation on raw returns is a pre-model screen. This lab assumes
# readers can interpret ADF, ACF, and PACF without treating a p-value as proof
# of a model. Because Lesson 2.3 already displays full-sample diagnostics, the
# later block here is a transparent retrospective evaluation—not a pristine
# ex-ante research holdout. Inside this notebook, however, every executable
# selection decision is recomputed from the training block only.
#
# ## Notation
#
# | Symbol | Meaning | Convention in this lesson |
# | --- | --- | --- |
# | $L$ | lag operator, $Lg_t=g_{t-1}$ | one FIX publication step |
# | $\phi(L)$ | autoregressive polynomial | roots outside the unit circle |
# | $\theta(L)$ | moving-average polynomial | roots outside the unit circle |
# | $\mu$ | constant mean for $d=0$ | log return per publication interval |
# | $\varepsilon_t$ | innovation | zero-mean model disturbance |
# | $k$ | number of estimated parameters used by an information criterion | dimensionless |
# | $n$ | number of training observations | count |
#
# ## Model and decision rule
#
# For the $d=0$ return candidates used here, statsmodels' constant-trend
# parameterization can be written as
#
# $$
# \phi(L)(g_t-\mu)=\theta(L)\varepsilon_t.
# $$
#
# The Banxico FIX log-return transformation and the training-only ADF screen
# support keeping $d=0$. The grid below admits a candidate only when
# optimization converges and its AR and MA representations are stationary and
# invertible. For maximized log likelihood $\ell$, the reported criteria are
#
# $$
# \operatorname{AIC}=-2\ell+2k,
# \qquad
# \operatorname{BIC}=-2\ell+k\log n.
# $$
#
# The predeclared rule selects the lowest-AIC admissible candidate on the first
# 80% of observations; BIC and later-sample MAE are reported as separate
# evidence, not silently folded into the choice.

# %% [markdown]
# ## Setup

# %% tags=["setup", "hide-input"]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

from src.market_data import banxico_daily_panel
from src.market_data_quality import log_returns
from src.module2_visuals import (
    build_forecast_comparison,
    build_residual_diagnostics,
)
from src.time_series_diagnostics import (
    adf_report,
    arima_order_search,
    jarque_bera_report,
    ljung_box_report,
)

REQUESTED_START = "2021-01-01"
REQUESTED_END = "2026-06-05"
SIGNIFICANCE_LEVEL = 0.05
TRAINING_FRACTION = 0.80
MATERIAL_RELATIVE_MAE_IMPROVEMENT = 0.01

# %% [markdown]
# ## Provider-dated Banxico FIX observations
#
# The committed snapshot preserves Banxico publication dates. Missing calendar
# dates are not forward-filled before returns are calculated, so a zero return
# is never created merely to align this series with another panel
# {cite}`banxicoSIE2025`.

# %%
banxico = banxico_daily_panel(start=REQUESTED_START, end=REQUESTED_END)
prices = banxico["usd_mxn"].dropna().rename("usd_mxn_fix_mxn_per_usd")
returns = log_returns(prices).rename("usd_mxn_fix_log_return")
calendar_gaps = prices.index.to_series().diff().dt.days.dropna()

sample_inventory = pd.DataFrame(
    [
        {
            "provider_series": "Banxico SIE SF43718",
            "data_mode": banxico.attrs["data_mode"],
            "snapshot_generated_at": banxico.attrs["snapshot_generated_at"],
            "observed_start": prices.index.min().date().isoformat(),
            "observed_end": prices.index.max().date().isoformat(),
            "level_observations": len(prices),
            "return_observations": len(returns),
            "maximum_calendar_gap_days": int(calendar_gaps.max()),
            "return_unit": "log change per FIX publication interval",
            "observation_policy": banxico.attrs["observation_policy"],
            "rights_note": (
                "provenance recorded; redistribution terms require external review"
            ),
        }
    ]
)
sample_inventory

# %%
source_note = (
    "Banxico SIE SF43718; observed "
    f"{prices.index.min():%Y-%m-%d} to {prices.index.max():%Y-%m-%d}; "
    f"snapshot {banxico.attrs['snapshot_generated_at']}; "
    "provider-dated observations, no forward fill"
)

# %% [markdown]
# **Output interpretation.**
#
# The inventory reports actual inclusive sample boundaries and the longest gap
# between consecutive FIX publications. Returns span publication intervals,
# which are usually one business day but can cross weekends or holidays. They
# are not returns from a mechanically forward-filled business-day panel.

# %%
pd.DataFrame({"level_mxn_per_usd": prices, "log_return": returns}).head()

# %% [markdown]
# **Output interpretation.**
#
# The first valid log return requires two consecutive published FIX levels.
# Banxico FIX is an official exchange-rate fixing, not an adjusted equity close
# or a guaranteed executable transaction price.

# %% [markdown]
# ## Chronological training and evaluation samples

# %%
split_index = int(len(returns) * TRAINING_FRACTION)
training_returns = returns.iloc[:split_index].copy()
testing_returns = returns.iloc[split_index:].copy()
training_prices = prices.loc[: training_returns.index.max()].copy()
assert training_returns.index.max() < testing_returns.index.min()
assert len(training_prices) == len(training_returns) + 1

pd.DataFrame(
    {
        "sample": ["training", "later evaluation"],
        "start": [training_returns.index.min(), testing_returns.index.min()],
        "end": [training_returns.index.max(), testing_returns.index.max()],
        "observations": [len(training_returns), len(testing_returns)],
    }
).set_index("sample")

# %% [markdown]
# **Output interpretation.**
#
# Every evaluation observation occurs after the order-selection sample. The
# executable workflow does not use the later block until the candidate grid and
# lowest-AIC rule have been fixed. The earlier full-sample lesson still means
# this is retrospective evidence rather than a pristine ex-ante experiment.

# %% [markdown]
# ## Training-only stationarity decision

# %%
level_adf = adf_report(training_prices, regression="c")
return_adf = adf_report(training_returns, regression="c")
if return_adf["p_value"] >= SIGNIFICANCE_LEVEL:
    raise RuntimeError(
        "The training-return ADF screen does not support the predeclared d=0 grid; "
        "review the transformation before fitting ARIMA candidates."
    )
adf_decision = pd.DataFrame(
    {
        "series": ["training FIX level", "training FIX log return"],
        "adf_statistic": [level_adf["statistic"], return_adf["statistic"]],
        "p_value": [level_adf["p_value"], return_adf["p_value"]],
        "decision_at_5pct": [
            "reject unit-root null"
            if report["p_value"] < SIGNIFICANCE_LEVEL
            else "do not reject unit-root null"
            for report in (level_adf, return_adf)
        ],
        "used_lag": [int(level_adf["used_lag"]), int(return_adf["used_lag"])],
        "observations": [int(level_adf["nobs"]), int(return_adf["nobs"])],
    }
).set_index("series")
adf_decision

# %% [markdown]
# **Output interpretation.**
#
# The 5% decision column uses only training observations and states the evidence
# without converting failure to reject into proof. The level and return tests
# use a constant deterministic term and AIC lag selection. A different trend
# specification is a sensitivity check, not a hidden way to force the desired
# conclusion.

# %% [markdown]
# ## Admissible ARIMA order search
#
# The grid is intentionally small for classroom use. AIC and BIC estimate
# different complexity penalties; neither is a forecast score
# {cite}`akaike1974new,schwarz1978estimating`.

# %%
candidate_results = arima_order_search(
    training_returns,
    p_values=range(0, 4),
    d_values=[0],
    q_values=range(0, 4),
)
admissible = candidate_results.loc[
    candidate_results["converged"]
    & candidate_results["stationary"]
    & candidate_results["invertible"]
].dropna(subset=["aic", "bic"])
if admissible.empty:
    raise RuntimeError("No converged, stationary, invertible ARIMA candidate was found")

aic_best = admissible.sort_values("aic").iloc[0]
bic_best = admissible.sort_values("bic").iloc[0]
selected_order = (int(aic_best["p"]), int(aic_best["d"]), int(aic_best["q"]))
bic_order = (int(bic_best["p"]), int(bic_best["d"]), int(bic_best["q"]))

candidate_results.head(10)

# %%
pd.DataFrame(
    {
        "criterion": ["AIC selection rule", "BIC comparison"],
        "order": [selected_order, bic_order],
        "criterion_value": [aic_best["aic"], bic_best["bic"]],
        "same_order_as_aic": [True, bic_order == selected_order],
    }
).set_index("criterion")

# %% [markdown]
# **Output interpretation.**
#
# “Selected” means lowest AIC inside this declared grid and training sample. The
# BIC row makes agreement or disagreement visible. Neither row establishes that
# the data-generating process is literally ARIMA or stable across regimes.

# %% [markdown]
# ## Chronological fixed-origin evaluation
#
# On the later sample, mean absolute error is
#
# $$
# \operatorname{MAE}
# =\frac{1}{n_{\mathrm{eval}}}
# \sum_{t\in\mathcal E}|g_t-\widehat g_{t\mid\mathrm{train}}|.
# $$
#
# The selected ARIMA, training-mean, and zero-return forecasts remain fixed
# after the training endpoint. The zero-return path is a no-change benchmark.
# If ARIMA(0,0,0) is selected, its conditional-mean forecast and the separately
# computed training-mean benchmark belong to the same constant-forecast class;
# a sub-threshold numerical difference is therefore reported as equivalent, not
# as a model victory. The ARIMA interval
# is a 95% model-based Gaussian interval; it is displayed as uncertainty under
# the fitted specification, not as a guaranteed empirical coverage rate.

# %%
training_model = ARIMA(
    training_returns.to_numpy(),
    order=selected_order,
    enforce_stationarity=True,
    enforce_invertibility=True,
).fit()
if not bool(training_model.mle_retvals.get("converged", True)):
    raise RuntimeError(f"Selected ARIMA{selected_order} did not converge on training data")

training_forecast_result = training_model.get_forecast(steps=len(testing_returns))
model_forecast = pd.Series(
    training_forecast_result.predicted_mean,
    index=testing_returns.index,
    name="arima_forecast",
)
model_forecast_interval = pd.DataFrame(
    np.asarray(training_forecast_result.conf_int(alpha=0.05)),
    index=testing_returns.index,
    columns=["lower", "upper"],
)
training_mean_forecast = pd.Series(
    training_returns.mean(),
    index=testing_returns.index,
    name="training_mean_forecast",
)
zero_return_forecast = pd.Series(
    0.0,
    index=testing_returns.index,
    name="zero_return_forecast",
)

forecast_evidence = pd.DataFrame(
    {
        "model": [
            f"ARIMA{selected_order}",
            "training-mean benchmark",
            "zero-return benchmark",
        ],
        "evaluation_mae": [
            mean_absolute_error(testing_returns, model_forecast),
            mean_absolute_error(testing_returns, training_mean_forecast),
            mean_absolute_error(testing_returns, zero_return_forecast),
        ],
        "unit": ["decimal log return per FIX publication interval"] * 3,
        "evaluation_start": [testing_returns.index.min()] * 3,
        "evaluation_end": [testing_returns.index.max()] * 3,
        "evaluation_observations": [len(testing_returns)] * 3,
    }
).set_index("model")
forecast_evidence

# %%
arima_mae = float(
    forecast_evidence.loc[f"ARIMA{selected_order}", "evaluation_mae"]
)
training_mean_mae = float(
    forecast_evidence.loc["training-mean benchmark", "evaluation_mae"]
)
zero_return_mae = float(
    forecast_evidence.loc["zero-return benchmark", "evaluation_mae"]
)
relative_improvement_vs_training_mean = (
    (training_mean_mae - arima_mae) / training_mean_mae
    if training_mean_mae > 0
    else np.nan
)
relative_improvement_vs_zero_return = (
    (zero_return_mae - arima_mae) / zero_return_mae
    if zero_return_mae > 0
    else np.nan
)
forecast_comparison = pd.Series(
    {
        "selected_order": selected_order,
        "same_constant_forecast_class": selected_order == (0, 0, 0),
        "materiality_threshold_relative_mae": (
            MATERIAL_RELATIVE_MAE_IMPROVEMENT
        ),
        "arima_mae_minus_training_mean_mae": arima_mae - training_mean_mae,
        "relative_improvement_vs_training_mean": (
            relative_improvement_vs_training_mean
        ),
        "assessment_vs_training_mean": (
            "material improvement"
            if relative_improvement_vs_training_mean
            >= MATERIAL_RELATIVE_MAE_IMPROVEMENT
            else "no material improvement"
        ),
        "arima_mae_minus_zero_return_mae": arima_mae - zero_return_mae,
        "relative_improvement_vs_zero_return": relative_improvement_vs_zero_return,
        "assessment_vs_zero_return": (
            "material improvement"
            if relative_improvement_vs_zero_return
            >= MATERIAL_RELATIVE_MAE_IMPROVEMENT
            else "no material improvement"
        ),
    },
    name="later-sample comparison",
)
forecast_comparison

# %%
forecast_paths = pd.DataFrame(
    {
        f"ARIMA{selected_order} fixed forecast": model_forecast,
        "Training-mean benchmark": training_mean_forecast,
        "Zero-return benchmark": zero_return_forecast,
    },
    index=testing_returns.index,
)
# %% mystnb={"image": {"alt": "Two aligned panels compare observed Banxico FIX log returns with fixed-origin ARIMA, training-mean, and zero-return forecasts, then show cumulative absolute error for all three paths."}}
forecast_figure = build_forecast_comparison(
    testing_returns,
    forecast_paths,
    title="Banxico FIX (MXN per USD): chronological mean-forecast evaluation",
    scale="decimal",
    prediction_interval=model_forecast_interval,
    prediction_interval_label=f"ARIMA{selected_order} 95% model interval",
    source=(
        f"{source_note}; evaluation block "
        f"{testing_returns.index.min():%Y-%m-%d} to "
        f"{testing_returns.index.max():%Y-%m-%d}"
    ),
    data_mode=banxico.attrs["data_mode"],
)
display(forecast_figure)
plt.close(forecast_figure)

# %% [markdown]
# **Output interpretation.**
#
# Later-sample MAE asks whether the selected linear mean model materially
# improves on two transparent fixed forecasts. The declared 1% relative-MAE
# threshold prevents floating-point or optimizer differences between constant
# forecasts from becoming a win/loss claim. The cumulative-error panel shows
# when any difference arises; it does not convert forecast error into trading
# profit or loss. Every path is produced once at the end of training, not by a
# rolling refit that consumes evaluation returns. The shaded interval remains
# conditional on a Gaussian ARIMA specification, so later residual-shape
# diagnostics are essential.

# %% [markdown]
# ## Full-sample refit and residual diagnostics
#
# After the order and later-sample comparison are recorded, the chosen specification
# is refitted to all available returns for descriptive residual diagnostics.

# %%
full_model = ARIMA(
    returns.to_numpy(),
    order=selected_order,
    enforce_stationarity=True,
    enforce_invertibility=True,
).fit(cov_type="robust")
if not bool(full_model.mle_retvals.get("converged", True)):
    raise RuntimeError(f"Selected ARIMA{selected_order} did not converge on full data")

residuals = pd.Series(full_model.resid, index=returns.index, name="arima_residual")
parameter_table = pd.DataFrame(
    {
        "estimate": full_model.params,
        "robust_standard_error": full_model.bse,
    },
    index=full_model.param_names,
).rename_axis("parameter")
parameter_table

# %% [markdown]
# **Output interpretation.**
#
# The compact table reports parameter estimates and robust sandwich standard
# errors. Robust covariance reduces reliance on a correctly specified Gaussian
# likelihood, but it cannot repair mean dependence, conditional
# heteroskedasticity, or non-Gaussian residual shape. Coefficients describe
# conditional-mean dependence, not profitability or causality.

# %%
residual_model_df = selected_order[0] + selected_order[2]
residual_lags = [lag for lag in (5, 10, 20) if lag > residual_model_df]
if not residual_lags:
    raise RuntimeError("No declared Ljung-Box lag exceeds the fitted ARMA parameter count")
residual_ljung_box = ljung_box_report(
    residuals,
    lags=residual_lags,
    model_df=residual_model_df,
)
residual_jarque_bera = jarque_bera_report(residuals)

if not np.isfinite(residual_ljung_box["lb_pvalue"].to_numpy()).all():
    raise RuntimeError("Residual Ljung-Box returned a non-finite p-value")
residual_rows = [
    {
        "diagnostic": f"Ljung-Box lag {lag}",
        "p_value": residual_ljung_box.loc[lag, "lb_pvalue"],
        "decision_at_5pct": (
            "remaining linear autocorrelation"
            if residual_ljung_box.loc[lag, "lb_pvalue"] < SIGNIFICANCE_LEVEL
            else "no rejection of zero residual autocorrelations"
        ),
    }
    for lag in residual_lags
]
residual_rows.append(
    {
        "diagnostic": "Jarque-Bera normality",
        "p_value": residual_jarque_bera["p_value"],
        "decision_at_5pct": (
            "reject Gaussian residuals"
            if residual_jarque_bera["p_value"] < SIGNIFICANCE_LEVEL
            else "do not reject Gaussian residuals"
        ),
    }
)
residual_evidence = pd.DataFrame(residual_rows).set_index("diagnostic")
residual_evidence

# %% mystnb={"image": {"alt": "Four diagnostic panels show the selected ARIMA residual path, residual ACF, squared-residual ACF, and normal quantile-quantile plot for Banxico FIX log returns."}}
residual_figure = build_residual_diagnostics(
    residuals,
    title=f"ARIMA{selected_order} residual diagnostics",
    residual_unit="decimal log return / FIX publication interval",
    lags=20,
    source=source_note,
    data_mode=banxico.attrs["data_mode"],
)
display(residual_figure)
plt.close(residual_figure)

# %% [markdown]
# **Output interpretation.**
#
# The Ljung-Box rows evaluate remaining *linear* autocorrelation after adjusting
# degrees of freedom for the selected AR and MA terms. Declared lags that do not
# exceed that parameter count are omitted rather than mislabeled from an
# undefined p-value. The squared-residual ACF
# in the figure can still reveal variance dependence even when the mean
# diagnostics are satisfactory. Jarque-Bera rejection warns against Gaussian
# interval or tail claims; it does not by itself invalidate the conditional-mean
# specification.

# %% [markdown]
# ## Model limitations
#
# - The FIX series is an official reference fixing, not a continuously traded
#   executable price.
# - Publication intervals are not equally spaced in calendar time; weekend and
#   holiday moves are contained in the next published change.
# - ADF, information criteria, and residual tests are sample- and
#   specification-dependent.
# - The grid excludes seasonal, exogenous, nonlinear, and regime-switching mean
#   models.
# - One retrospective chronological evaluation is more informative than
#   in-sample fit but is not a pristine ex-ante experiment and does not
#   establish stable forecast value.
# - ARIMA models can miss volatility clustering and heavy-tailed innovations.

# %% [markdown]
# ## Handoff
#
# The [next comparison](2.2.time_series_2.ipynb) fits ARCH and GARCH with a
# constant mean. That is aligned
# with this page only when the selected mean specification is ARIMA(0,0,0). If a
# refreshed snapshot selects nontrivial AR or MA terms, the ARCH/GARCH page
# remains a didactic variance comparison; an applied workflow should fit the
# selected mean and variance equations jointly rather than pretending raw
# returns are already mean-model residuals.
