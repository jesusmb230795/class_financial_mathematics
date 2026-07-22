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
# # Quantitative Foundations for Financial Time Series
#
# Module: Quantitative Methods and Financial Time Series
#
# ## Lesson summary
#
# Quantitative finance joins cash flows across time, market observations across
# dates, and models across estimation and evaluation samples. This notebook
# establishes one reproducible foundation for those tasks: explicit interest-rate
# conventions, simple and log returns, probability and simulation, statistical
# inference, regression, and time-ordered model validation. The examples use
# synthetic data so that every result is deterministic and no empirical claim is
# confused with a simulated illustration {cite}`harris2020array,mckinney2010data,tsay2010analysis`.
#
# ## Learning objectives
#
# By the end of this lesson, readers should be able to:
#
# - discount and accumulate cash flows under simple, periodic, and continuous
#   compounding;
# - calculate simple and log returns and annualize each quantity with the correct
#   transformation;
# - reproduce a simulation from an explicit random seed and summarize its
#   probability distribution;
# - estimate a regression with uncertainty reported around the coefficients;
# - construct lagged predictors without using future observations; and
# - evaluate a predictive model with chronological splits and a genuine
#   out-of-sample benchmark.
#
# ## Prerequisites
#
# Complete the Module 1 market-data quality lessons first. Readers should be able
# to store rates and returns as decimals, distinguish an observation from a
# constructed field, and identify the date and unit of every input.
#
# ## Notation and conventions
#
# | Symbol | Meaning | Convention in this lesson |
# | --- | --- | --- |
# | $PV$, $FV$ | present and future value | currency units |
# | $r$ | quoted annual rate | decimal, with compounding stated |
# | $m$ | periodic compounding frequency | periods per year |
# | $R_t$ | simple return | $P_t/P_{t-1}-1$ |
# | $g_t$ | log return | $\log(P_t/P_{t-1})=\log(1+R_t)$ |
# | $A$ | annualization factor | 252 business days per year |
#
# A rate is not complete until its compounding convention, horizon, payment
# frequency, and day-count basis are known. The classroom examples below use exact
# year fractions and do not model settlement calendars {cite}`fabozzi2019foundations`.

# %% [markdown]
# ## Setup

# %% tags=["setup", "hide-input"]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from IPython.display import display
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.fixed_income import (
    continuous_rate_from_effective,
    effective_annual_rate,
    effective_rate_from_continuous,
    future_value,
    simple_future_value,
)
from src.visual_style import (
    SEMANTIC_COLORS,
    add_figure_note,
    matplotlib_style,
    style_axes,
)

SEED = 20260720
PERIODS_PER_YEAR = 252
np.set_printoptions(precision=6, suppress=True)
pd.set_option("display.float_format", lambda value: f"{value:,.6f}")

# %% [markdown]
# ## Time value of money and rate conventions
#
# For an annual quoted rate $r$, a horizon $T$ in years, and $m$ periodic
# compounding intervals per year, the accumulation factors are:
#
# ```{math}
# a_{\mathrm{simple}}(T)=1+rT,\qquad
# a_{\mathrm{periodic}}(T)=\left(1+\frac{r}{m}\right)^{mT},\qquad
# a_{\mathrm{continuous}}(T)=e^{rT}.
# ```
#
# Present value and future value use the same factor:
#
# ```{math}
# FV=PV\,a(T),\qquad PV=\frac{FV}{a(T)}.
# ```
#
# The formulas do not make quotations interchangeable. A nominal 8% rate
# compounded monthly is not the same economic input as an 8% effective annual
# rate or an 8% continuously compounded rate.

# %%
principal = 10_000.0
quoted_rate = 0.08
horizon_years = 2.0
convention_rows = [
    {
        "convention": "simple",
        "compounding_periods_per_year": np.nan,
        "future_value_mxn": simple_future_value(
            principal,
            quoted_rate,
            horizon_years,
        ),
    },
    {
        "convention": "periodic",
        "compounding_periods_per_year": 1,
        "future_value_mxn": future_value(
            principal,
            quoted_rate,
            horizon_years,
            compounding=1,
        ),
    },
    {
        "convention": "periodic",
        "compounding_periods_per_year": 12,
        "future_value_mxn": future_value(
            principal,
            quoted_rate,
            horizon_years,
            compounding=12,
        ),
    },
    {
        "convention": "continuous",
        "compounding_periods_per_year": np.nan,
        "future_value_mxn": future_value(
            principal,
            quoted_rate,
            horizon_years,
            compounding=None,
        ),
    },
]
for row in convention_rows:
    row["accumulation_factor"] = row["future_value_mxn"] / principal

rate_convention_table = pd.DataFrame(convention_rows)[
    [
        "convention",
        "compounding_periods_per_year",
        "accumulation_factor",
        "future_value_mxn",
    ]
]
rate_convention_table

# %% [markdown]
# **Output interpretation.**
#
# Every row uses the same numerical quote but a different accumulation rule.
# The resulting future values differ because the quote alone does not determine
# the cash-flow conversion.

# %%
monthly_factor_one_year = future_value(
    1.0,
    quoted_rate,
    1.0,
    compounding=12,
)
effective_annual_rate_value = effective_annual_rate(quoted_rate, 12)
equivalent_continuous_rate = continuous_rate_from_effective(
    effective_annual_rate_value
)
conversion_check = pd.Series(
    {
        "nominal_rate_monthly_compounding": quoted_rate,
        "effective_annual_rate": effective_annual_rate_value,
        "equivalent_continuous_rate": equivalent_continuous_rate,
    }
)
assert np.isclose(np.exp(equivalent_continuous_rate), monthly_factor_one_year)
assert np.isclose(
    effective_rate_from_continuous(equivalent_continuous_rate),
    effective_annual_rate_value,
)
conversion_check

# %% [markdown]
# **Output interpretation.**
#
# The effective annual and continuous rates are *equivalent* because they
# produce the same one-year factor. The original 8% nominal quotation is retained
# separately so that the conversion cannot be mistaken for a change in market
# conditions.

# %% [markdown]
# ## Reproducible probability model
#
# A simulation is an experiment under stated assumptions, not observed evidence.
# The next cell creates 900 synthetic business-day observations. A persistent
# signal affects the next return, and volatility increases after observation 600.
# Those choices make time ordering and model instability visible. They are not
# estimates of any security.

# %%
rng = np.random.default_rng(SEED)
n_observations = 900
signal_burn_in = 500
signal_ar_coefficient = 0.88
signal_shock_standard_deviation = 0.45
signal_unconditional_standard_deviation = signal_shock_standard_deviation / np.sqrt(
    1 - signal_ar_coefficient**2
)
dates = pd.bdate_range("2022-01-03", periods=n_observations)

signal_path = np.zeros(n_observations + signal_burn_in)
signal_shocks = rng.normal(
    0.0,
    signal_shock_standard_deviation,
    len(signal_path),
)
for index in range(1, len(signal_path)):
    signal_path[index] = (
        signal_ar_coefficient * signal_path[index - 1] + signal_shocks[index]
    )
signal = signal_path[signal_burn_in:] / signal_unconditional_standard_deviation

daily_volatility = np.where(np.arange(n_observations) < 600, 0.008, 0.013)
return_shocks = rng.normal(0.0, 1.0, n_observations)
synthetic_log_returns = np.empty(n_observations)
synthetic_log_returns[0] = 0.00015 + daily_volatility[0] * return_shocks[0]
synthetic_log_returns[1:] = (
    0.00015
    + 0.00120 * signal[:-1]
    + daily_volatility[1:] * return_shocks[1:]
)

synthetic = pd.DataFrame(
    {
        "signal": signal,
        "daily_volatility_parameter": daily_volatility,
        "log_return": synthetic_log_returns,
    },
    index=dates,
)
synthetic.index.name = "date"
synthetic["price"] = 100.0 * np.exp(synthetic["log_return"].cumsum())
synthetic[["price", "log_return", "signal"]].head()

# %% [markdown]
# **Output interpretation.**
#
# The fixed seed makes the table identical on every run. A 500-observation
# burn-in reduces dependence on the zero initialization, and the signal is
# scaled by its theoretical unconditional standard deviation rather than by
# full-sample statistics. The dates are a generic Monday-to-Friday index; they
# do not encode exchange holidays, releases, or investable prices.

# %%
with matplotlib_style():
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True,
        constrained_layout=True,
    )
    synthetic["price"].plot(
        ax=axes[0],
        color=SEMANTIC_COLORS["primary"],
        title="Synthetic price path",
    )
    synthetic["log_return"].plot(
        ax=axes[1],
        color=SEMANTIC_COLORS["comparison"],
        alpha=0.8,
        title="Synthetic daily log returns",
    )
    axes[0].set_ylabel("Index level (base = 100)")
    axes[1].set_ylabel("Daily log return (decimal)")
    axes[1].set_xlabel("Date")
    style_axes(axes[0], grid_axis="y")
    style_axes(axes[1], grid_axis="y", show_zero_line=True)
    fig.suptitle("Seeded synthetic sample with a final-third volatility shift")
    add_figure_note(
        fig,
        (
            f"Seed: {SEED} | Sample: {dates.min():%Y-%m-%d} to "
            f"{dates.max():%Y-%m-%d} | {n_observations} synthetic "
            "Monday-to-Friday observations | Returns: decimal daily log returns"
        ),
    )
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0, 0.075, 1, 0.94))
display(fig)
plt.close(fig)

# %% [markdown]
# **Output interpretation.**
#
# The final third of the return chart contains a deliberately higher-volatility
# regime. A validation scheme that mixes early and late observations at random
# would conceal part of that distribution shift.

# %% [markdown]
# ## Simple returns, log returns, and annualization
#
# Simple returns aggregate across assets when portfolio weights are known. Log
# returns aggregate through time. For one asset and one period,
# $g_t=\log(1+R_t)$, but the quantities must retain distinct names
# {cite}`cont2001empirical,tsay2010analysis`.

# %%
synthetic["simple_return"] = synthetic["price"].pct_change()
synthetic["log_return_from_price"] = np.log(synthetic["price"]).diff()
return_sample = synthetic.dropna().copy()

identity_error = (
    np.log1p(return_sample["simple_return"])
    - return_sample["log_return_from_price"]
).abs().max()
assert identity_error < 1e-12

annualized_log_return = (
    return_sample["log_return_from_price"].mean() * PERIODS_PER_YEAR
)
annualized_simple_return_from_log = np.expm1(annualized_log_return)
annualized_volatility = (
    return_sample["log_return_from_price"].std(ddof=1)
    * np.sqrt(PERIODS_PER_YEAR)
)
annualization_table = pd.Series(
    {
        "periods_per_year": PERIODS_PER_YEAR,
        "annualized_log_return": annualized_log_return,
        "annualized_simple_return_from_log": annualized_simple_return_from_log,
        "annualized_volatility": annualized_volatility,
        "maximum_return_identity_error": identity_error,
    }
)
annualization_table

# %% [markdown]
# **Output interpretation.**
#
# Multiplying the mean log return by 252 gives an annualized **log** return.
# Applying `expm1` converts that result to its annualized simple-return
# equivalent. Volatility uses the square root of 252. These are descriptive
# annualizations of the synthetic daily sample, not forecasts.

# %% [markdown]
# ## Monte Carlo probabilities
#
# Under a geometric Brownian motion illustration with annual drift $\mu$,
# annual volatility $\sigma$, and horizon $T$, a terminal price is:
#
# ```{math}
# S_T=S_0\exp\left[\left(\mu-\frac{1}{2}\sigma^2\right)T
# +\sigma\sqrt{T}Z\right],\qquad Z\sim N(0,1).
# ```
#
# The probability estimate changes with the assumed distribution, drift,
# volatility, horizon, and simulation count.

# %%
simulation_rng = np.random.default_rng(SEED + 1)
n_paths = 25_000
spot = 100.0
annual_drift = 0.06
annual_volatility_parameter = 0.20
years = 1.0
normal_draws = simulation_rng.standard_normal(n_paths)
terminal_prices = spot * np.exp(
    (annual_drift - 0.5 * annual_volatility_parameter**2) * years
    + annual_volatility_parameter * np.sqrt(years) * normal_draws
)
monte_carlo_summary = pd.Series(
    {
        "seed": SEED + 1,
        "paths": n_paths,
        "probability_terminal_loss": np.mean(terminal_prices < spot),
        "terminal_price_q05": np.quantile(terminal_prices, 0.05),
        "terminal_price_median": np.median(terminal_prices),
        "terminal_price_q95": np.quantile(terminal_prices, 0.95),
    }
)
monte_carlo_summary

# %% [markdown]
# **Output interpretation.**
#
# The loss probability is a Monte Carlo frequency under the stated lognormal
# model. It is not a historical loss rate and does not incorporate jumps,
# transaction costs, parameter uncertainty, or changing volatility.

# %% [markdown]
# ## Inference and regression
#
# An estimate without uncertainty invites false precision. First, an
# independent-and-identically-distributed bootstrap illustrates a confidence
# interval for the synthetic mean daily log return. Then a training-sample
# regression estimates the known lagged-signal relationship with
# heteroskedasticity-and-autocorrelation-consistent standard errors. The
# bootstrap is pedagogical; dependent market returns may require a block
# bootstrap or a model-based procedure {cite}`hamilton1994time,statsmodels2010`.

# %%
bootstrap_rng = np.random.default_rng(SEED + 2)
observed_log_returns = return_sample["log_return_from_price"].to_numpy()
n_bootstrap = 4_000
bootstrap_indices = bootstrap_rng.integers(
    0,
    len(observed_log_returns),
    size=(n_bootstrap, len(observed_log_returns)),
)
bootstrap_means = observed_log_returns[bootstrap_indices].mean(axis=1)
bootstrap_interval = np.quantile(bootstrap_means, [0.025, 0.975])
pd.Series(
    {
        "sample_mean_daily_log_return": observed_log_returns.mean(),
        "iid_bootstrap_ci_lower": bootstrap_interval[0],
        "iid_bootstrap_ci_upper": bootstrap_interval[1],
        "bootstrap_resamples": n_bootstrap,
    }
)

# %%
model_frame = synthetic.assign(
    signal_lag_1=synthetic["signal"].shift(1),
    log_return_lag_1=synthetic["log_return"].shift(1),
    rolling_volatility_20=(
        synthetic["log_return"].shift(1).rolling(20).std(ddof=1)
    ),
).dropna()

split_index = int(len(model_frame) * 0.70)
training = model_frame.iloc[:split_index].copy()
testing = model_frame.iloc[split_index:].copy()
assert training.index.max() < testing.index.min()

inference_design = sm.add_constant(training[["signal_lag_1"]])
inference_model = sm.OLS(training["log_return"], inference_design).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": 5},
)
coefficient_interval = inference_model.conf_int().loc["signal_lag_1"]
inference_table = pd.Series(
    {
        "training_start": training.index.min().date().isoformat(),
        "training_end": training.index.max().date().isoformat(),
        "signal_coefficient": inference_model.params["signal_lag_1"],
        "hac_ci_lower": coefficient_interval.iloc[0],
        "hac_ci_upper": coefficient_interval.iloc[1],
        "training_observations": len(training),
    }
)
inference_table

# %% [markdown]
# **Output interpretation.**
#
# Only the first 70% of dates enter the inferential regression. The estimated
# coefficient can be compared with the simulation parameter of 0.00120, but a
# confidence interval describes repeated-sample uncertainty under an estimator;
# it does not prove causality.

# %% [markdown]
# ## Temporal validation without look-ahead
#
# Every predictor below is available before the target return:
#
# - `signal_lag_1` is yesterday's signal;
# - `log_return_lag_1` is yesterday's return;
# - `rolling_volatility_20` uses returns ending yesterday.
#
# Preprocessing and estimation are wrapped in one pipeline. Each
# `TimeSeriesSplit` fold fits the scaler and model only on dates earlier than its
# validation block. The final test block remains untouched until the model
# choices are fixed.

# %%
feature_columns = [
    "signal_lag_1",
    "log_return_lag_1",
    "rolling_volatility_20",
]
pipeline = Pipeline(
    [
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=5.0)),
    ]
)
temporal_cv = TimeSeriesSplit(n_splits=5)
cv_negative_mae = cross_val_score(
    pipeline,
    training[feature_columns],
    training["log_return"],
    cv=temporal_cv,
    scoring="neg_mean_absolute_error",
)

pipeline.fit(training[feature_columns], training["log_return"])
test_prediction = pipeline.predict(testing[feature_columns])
training_mean_baseline = np.repeat(
    training["log_return"].mean(),
    len(testing),
)
validation_table = pd.Series(
    {
        "cv_mean_mae": -cv_negative_mae.mean(),
        "cv_mae_standard_deviation": cv_negative_mae.std(ddof=1),
        "test_model_mae": mean_absolute_error(
            testing["log_return"],
            test_prediction,
        ),
        "test_training_mean_baseline_mae": mean_absolute_error(
            testing["log_return"],
            training_mean_baseline,
        ),
        "test_start": testing.index.min().date().isoformat(),
        "test_end": testing.index.max().date().isoformat(),
        "test_observations": len(testing),
    }
)
validation_table

# %% [markdown]
# **Output interpretation.**
#
# Cross-validation error measures several expanding time-ordered experiments.
# Final test MAE measures one later, higher-volatility regime. The model should
# be judged against the training-mean benchmark rather than against in-sample
# fit. Even if it wins in this seeded simulation, that does not establish a
# profitable strategy.

# %%
split_audit = []
for fold, (train_positions, validation_positions) in enumerate(
    temporal_cv.split(training),
    start=1,
):
    train_end = training.index[train_positions].max()
    validation_start = training.index[validation_positions].min()
    split_audit.append(
        {
            "fold": fold,
            "train_end": train_end.date().isoformat(),
            "validation_start": validation_start.date().isoformat(),
            "ordered": train_end < validation_start,
        }
    )
split_audit_table = pd.DataFrame(split_audit)
assert split_audit_table["ordered"].all()
split_audit_table

# %% [markdown]
# **Output interpretation.**
#
# The audit table makes the no-look-ahead condition observable: every training
# fold ends before its validation fold begins. A shuffled split cannot provide
# this guarantee for a forecasting problem.

# %% [markdown]
# ## Limitations and failure modes
#
# - The business-day calendar is synthetic and omits exchange holidays.
# - The simulated signal is deliberately predictive; real signals are weaker,
#   noisier, crowded, revised, and exposed to data-mining bias.
# - The lognormal Monte Carlo model omits jumps, stochastic volatility,
#   liquidity, costs, and parameter uncertainty.
# - The i.i.d. bootstrap ignores serial dependence and volatility clustering.
# - HAC standard errors reduce sensitivity to some dependence patterns but do
#   not repair omitted variables, endogeneity, or structural breaks.
# - A single chronological holdout is necessary but not sufficient. Production
#   validation also needs repeated vintages, transaction-cost assumptions,
#   stability monitoring, and a documented retraining rule.

# %% [markdown] tags=["exercise"]
# ## Checkpoint exercise
#
# 1. Calculate the two-year future value of MXN 10,000 at a 9% nominal annual
#    rate compounded monthly, and state the compounding convention.
# 2. Convert a 12% annualized log return to an annualized simple return.
# 3. Inspect the temporal split audit and state why `ordered=True` is necessary.
# 4. Replace `SEED` with `SEED + 10`, rerun the simulation, and identify which
#    outputs change and which rate-conversion outputs remain invariant.

# %% tags=["solution"]
checkpoint_future_value = future_value(
    10_000.0,
    0.09,
    2.0,
    compounding=12,
)
checkpoint_simple_return = np.expm1(0.12)
assert split_audit_table["ordered"].all()
pd.Series(
    {
        "future_value_mxn": checkpoint_future_value,
        "annualized_simple_return": checkpoint_simple_return,
        "all_temporal_folds_ordered": split_audit_table["ordered"].all(),
    }
)

# %% [markdown]
# **Checkpoint interpretation.**
#
# The future value is approximately MXN 11,964.14 under nominal monthly
# compounding. A 12% annualized log return is approximately a 12.7497%
# annualized simple return. Ordered folds prevent observations dated after a
# validation target from influencing estimation or preprocessing. Changing the
# seed changes simulated paths, regression estimates, and probability summaries,
# but it does not change deterministic rate conversions.

# %% [markdown]
# ## Handoff
#
# The next lesson, Financial Time Series: Levels, Returns, and White Noise,
# replaces synthetic levels with provider-dated Banxico FIX observations. It
# applies the return conventions from this notebook before introducing
# stationarity, white-noise diagnostics, and time-series model selection.
