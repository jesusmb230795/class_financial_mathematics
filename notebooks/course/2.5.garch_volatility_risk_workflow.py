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
# # GARCH Volatility, Diagnostics, and One-Step Risk
#
# Module: Quantitative Methods and Financial Time Series
#
# ## Lesson summary
#
# This lab is the module's authoritative conditional-volatility-to-risk
# workflow. It compares Gaussian and standardized Student-t GARCH(1,1)
# specifications, verifies convergence and standardized-residual diagnostics,
# and calculates one-FIX-interval VaR as a non-negative loss for a declared
# long-USD/short-MXN exposure
# {cite}`engle1982autoregressive,bollerslev1986generalized,tsay2010analysis`.
#
# ## Learning objectives
#
# By the end of this lab, readers should be able to:
#
# - fit and validate Gaussian and standardized Student-t GARCH models;
# - interpret $\omega$, $\alpha$, $\beta$, persistence, and long-run volatility;
# - test standardized residuals and their squares for remaining dependence;
# - separate innovation-quantile effects from model-forecast effects; and
# - report one-step VaR with an explicit tail probability, confidence level,
#   horizon, position, sign, and return unit.
#
# ## Prerequisites
#
# Complete Lessons 2.3, 2.4, and 2.2 first. They establish the stationarity,
# conditional-mean, ARCH-effect, and variance-model evidence used here. The
# current snapshot's mean-model result determines whether a constant-mean GARCH
# is an aligned applied specification or only a didactic comparison.
#
# ## Model and risk equations
#
# The GARCH(1,1) recursion is
#
# $$
# r_t = \mu + \varepsilon_t, \qquad \varepsilon_t = \sigma_t z_t,
# $$
#
# $$
# \sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2
# + \beta \sigma_{t-1}^2.
# $$
#
# When $\alpha+\beta<1$, the parameter-implied long-run variance and volatility
# are
#
# $$
# \bar{\sigma}^2 = \frac{\omega}{1-\alpha-\beta}, \qquad
# \bar{\sigma}=\sqrt{\bar{\sigma}^2}.
# $$
#
# For a long-USD/short-MXN position whose value changes with the MXN-per-USD
# quote, return loss is $L_{t+1}=-r_{t+1}$. With left-tail probability
# $\alpha_{\mathrm{tail}}=0.01$ and confidence
# $1-\alpha_{\mathrm{tail}}=0.99$, the book reports one-step VaR as
#
# $$
# \operatorname{VaR}_{\alpha_{\mathrm{tail}},t+1}
# =\max\left\{0,-\left(\mu_{t+1\mid t}
# +\sigma_{t+1\mid t}q_{\alpha_{\mathrm{tail}}}\right)\right\}.
# $$

# %% [markdown]
# ## Setup

# %% tags=["setup", "hide-input"]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from IPython.display import display
from scipy.stats import norm

from src.market_data import banxico_daily_panel
from src.market_data_quality import log_returns
from src.market_risk import parametric_var, standardized_student_t_quantile
from src.module2_visuals import build_volatility_comparison
from src.time_series_diagnostics import (
    arch_lm_report,
    garch_long_run_variance,
    garch_persistence,
    ljung_box_report,
)

REQUESTED_START = "2021-01-01"
REQUESTED_END = "2026-06-05"
TAIL_PROBABILITY = 0.01
CONFIDENCE_LEVEL = 1 - TAIL_PROBABILITY

# %% [markdown]
# ## Provider-dated Banxico FIX returns
#
# Returns are calculated only between published Banxico FIX observations; the
# business-day-aligned classroom panel is not used for model estimation
# {cite}`banxicoSIE2025`.

# %%
banxico = banxico_daily_panel(start=REQUESTED_START, end=REQUESTED_END)
fix_level = banxico["usd_mxn"].dropna().rename("usd_mxn_fix_mxn_per_usd")
returns_decimal = log_returns(fix_level).rename("usd_mxn_fix_log_return")
returns = returns_decimal.mul(100).rename("usd_mxn_fix_log_return_pct")
calendar_gaps = fix_level.index.to_series().diff().dt.days.dropna()

pd.DataFrame(
    [
        {
            "provider_series": "Banxico SIE SF43718",
            "data_mode": banxico.attrs["data_mode"],
            "snapshot_generated_at": banxico.attrs["snapshot_generated_at"],
            "observed_start": fix_level.index.min().date().isoformat(),
            "observed_end": fix_level.index.max().date().isoformat(),
            "return_observations": len(returns),
            "maximum_calendar_gap_days": int(calendar_gaps.max()),
            "model_unit": "percentage log return per FIX publication interval",
            "observation_policy": banxico.attrs["observation_policy"],
        }
    ]
)

# %% [markdown]
# **Output interpretation.**
#
# Percent scaling improves numerical conditioning but changes the units of
# $\omega$, conditional variance, volatility, and VaR. Publication intervals
# can cross weekends or holidays; “one step” means the next published FIX, not
# a guaranteed one-calendar-day horizon.

# %% [markdown]
# ## Fit Gaussian and standardized Student-t GARCH
#
# The `arch` package uses $p$ for squared-shock lags and $q$ for conditional-
# variance lags. Both models use the same returns and constant-mean equation;
# only the standardized innovation distribution changes
# {cite}`sheppard2024arch`.

# %%
garch_normal_result = arch_model(
    returns,
    mean="Constant",
    vol="Garch",
    p=1,
    q=1,
    dist="normal",
    rescale=False,
).fit(disp="off")
garch_t_result = arch_model(
    returns,
    mean="Constant",
    vol="Garch",
    p=1,
    q=1,
    dist="StudentsT",
    rescale=False,
).fit(disp="off")

for model_name, fitted_model in {
    "Gaussian GARCH": garch_normal_result,
    "Student-t GARCH": garch_t_result,
}.items():
    if fitted_model.convergence_flag != 0:
        raise RuntimeError(f"{model_name} did not converge")

# %%
model_rows = []
for model_name, fitted_model in {
    "Gaussian GARCH": garch_normal_result,
    "Student-t GARCH": garch_t_result,
}.items():
    fitted_params = fitted_model.params
    persistence = garch_persistence(
        fitted_params["alpha[1]"],
        fitted_params["beta[1]"],
    )
    long_run_variance = garch_long_run_variance(
        fitted_params["omega"],
        fitted_params["alpha[1]"],
        fitted_params["beta[1]"],
    )
    model_rows.append(
        {
            "model": model_name,
            "mu_pct": fitted_params["mu"],
            "omega_pct_squared": fitted_params["omega"],
            "alpha_1": fitted_params["alpha[1]"],
            "beta_1": fitted_params["beta[1]"],
            "persistence": persistence,
            "long_run_variance_pct_squared": long_run_variance,
            "long_run_interval_volatility_pct": np.sqrt(long_run_variance),
            "student_t_degrees_of_freedom": fitted_params.get("nu", np.nan),
            "aic": fitted_model.aic,
            "bic": fitted_model.bic,
            "convergence_flag": fitted_model.convergence_flag,
        }
    )

model_comparison = pd.DataFrame(model_rows).set_index("model")
model_comparison

# %% [markdown]
# **Output interpretation.**
#
# The table keeps variance and volatility in distinct units. Lower AIC or BIC
# supports one specification only within this same-sample comparison.
# Persistence near one implies slow expected variance decay, while a zero
# convergence flag is a prerequisite—not proof of a stable model.

# %%
volatility_paths = pd.DataFrame(
    {
        "Gaussian GARCH(1,1)": garch_normal_result.conditional_volatility,
        "Student-t GARCH(1,1)": garch_t_result.conditional_volatility,
    },
    index=returns.index,
)
volatility_figure = build_volatility_comparison(
    returns,
    volatility_paths,
    title="USD/MXN FIX conditional-volatility comparison",
    return_scale="percent",
    volatility_scale="percent",
    source="Banxico SIE SF43718 snapshot",
    data_mode="provider-dated snapshot",
)
display(volatility_figure)
plt.close(volatility_figure)

# %% [markdown]
# **Output interpretation.**
#
# Both paths use the same percent-return sample. Differences therefore reflect
# the fitted innovation distribution and parameter estimates, not a calendar or
# scale change. The figure is descriptive and does not establish forecast
# superiority.

# %% [markdown]
# ## Standardized-residual validation
#
# A fitted variance model should leave little linear dependence in standardized
# residuals $z_t=\varepsilon_t/\sigma_t$ or their squares. ARCH-LM provides a
# complementary check for remaining conditional heteroskedasticity.

# %%
diagnostic_rows = []
for model_name, fitted_model in {
    "Gaussian GARCH": garch_normal_result,
    "Student-t GARCH": garch_t_result,
}.items():
    standardized_residuals = pd.Series(
        fitted_model.std_resid,
        index=returns.index,
        name="standardized_residual",
    ).dropna()
    residual_lb = ljung_box_report(standardized_residuals, lags=[10])
    squared_lb = ljung_box_report(standardized_residuals.pow(2), lags=[10])
    arch_lm = arch_lm_report(standardized_residuals, lags=10, model_df=1)
    diagnostic_rows.append(
        {
            "model": model_name,
            "standardized_residual_ljung_box_p10": residual_lb.loc[10, "lb_pvalue"],
            "squared_residual_ljung_box_p10": squared_lb.loc[10, "lb_pvalue"],
            "arch_lm_p10": arch_lm["lm_p_value"],
        }
    )

standardized_residual_diagnostics = pd.DataFrame(diagnostic_rows).set_index("model")
standardized_residual_diagnostics["mean_dependence_decision_at_5pct"] = np.where(
    standardized_residual_diagnostics[
        "standardized_residual_ljung_box_p10"
    ]
    < 0.05,
    "remaining linear dependence",
    "no rejection of zero linear dependence",
)
standardized_residual_diagnostics["variance_dependence_decision_at_5pct"] = np.where(
    (
        standardized_residual_diagnostics["squared_residual_ljung_box_p10"]
        < 0.05
    )
    | (standardized_residual_diagnostics["arch_lm_p10"] < 0.05),
    "remaining variance dependence",
    "no rejection of absorbed variance dependence",
)
standardized_residual_diagnostics

# %% [markdown]
# **Output interpretation.**
#
# At a 5% threshold, a small first p-value flags remaining conditional-mean
# dependence; small squared-residual or ARCH-LM p-values flag variance dynamics
# the GARCH model has not absorbed. The decision columns are computed from the
# displayed p-values. A rejection does not prevent the arithmetic below, but it
# prevents presenting the resulting VaR as a validated risk forecast.

# %% [markdown]
# ## One-step positive-loss VaR
#
# First calculate an end-to-end VaR from each model's own mean, variance, and
# innovation quantile. Then hold the Student-t GARCH mean and variance fixed and
# change only the quantile; that counterfactual isolates the tail-shape effect.

# %%
normal_forecast = garch_normal_result.forecast(horizon=1)
student_t_forecast = garch_t_result.forecast(horizon=1)

normal_mean_forecast = float(normal_forecast.mean.iloc[-1, 0])
normal_volatility_forecast = float(np.sqrt(normal_forecast.variance.iloc[-1, 0]))
student_t_mean_forecast = float(student_t_forecast.mean.iloc[-1, 0])
student_t_volatility_forecast = float(
    np.sqrt(student_t_forecast.variance.iloc[-1, 0])
)
degrees_of_freedom = float(garch_t_result.params["nu"])
normal_quantile = float(norm.ppf(TAIL_PROBABILITY))
student_t_quantile = standardized_student_t_quantile(
    TAIL_PROBABILITY,
    degrees_of_freedom,
)

normal_model_var = parametric_var(
    normal_mean_forecast,
    normal_volatility_forecast,
    normal_quantile,
)
student_t_model_var = parametric_var(
    student_t_mean_forecast,
    student_t_volatility_forecast,
    student_t_quantile,
)
student_t_forecast_gaussian_quantile_var = parametric_var(
    student_t_mean_forecast,
    student_t_volatility_forecast,
    normal_quantile,
)

var_comparison = pd.DataFrame(
    {
        "mean_forecast_pct": [normal_mean_forecast, student_t_mean_forecast],
        "volatility_forecast_pct": [
            normal_volatility_forecast,
            student_t_volatility_forecast,
        ],
        "standardized_quantile": [normal_quantile, student_t_quantile],
        "one_step_var_positive_loss_pct": [normal_model_var, student_t_model_var],
        "tail_probability": [TAIL_PROBABILITY] * 2,
        "confidence": [CONFIDENCE_LEVEL] * 2,
        "horizon": ["next published FIX"] * 2,
        "position": ["long USD / short MXN"] * 2,
    },
    index=["Gaussian GARCH", "Student-t GARCH"],
)
var_comparison["diagnostic_status"] = standardized_residual_diagnostics[
    "mean_dependence_decision_at_5pct"
]
var_comparison

# %%
var_decomposition = pd.Series(
    {
        "gaussian_model_var_pct": normal_model_var,
        "student_t_forecast_with_gaussian_quantile_var_pct": (
            student_t_forecast_gaussian_quantile_var
        ),
        "student_t_model_var_pct": student_t_model_var,
        "mean_and_variance_forecast_effect_pct": (
            student_t_forecast_gaussian_quantile_var - normal_model_var
        ),
        "student_t_quantile_effect_pct": (
            student_t_model_var - student_t_forecast_gaussian_quantile_var
        ),
    }
)
var_decomposition

# %% [markdown]
# **Output interpretation.**
#
# The first table compares complete fitted models. The decomposition then holds
# the Student-t GARCH forecast fixed, so only its final difference can be called
# a quantile or tail-shape effect. Every VaR is a non-negative percentage loss
# for the next published FIX and the declared long-USD/short-MXN position. A
# short-USD position would require reversing the return sign before applying the
# same loss convention. The diagnostic-status column prevents an arithmetically
# valid VaR from being mislabeled as a validated forecast. The standardized
# Student-t quantile uses $\sqrt{(\nu-2)/\nu}$; a raw SciPy t quantile would
# double-count scale.

# %% [markdown]
# ## Extension: directional asymmetry in FX
#
# A GJR-GARCH specification adds a sign indicator
# {cite}`glosten1993relation`:
#
# ```python
# arch_model(
#     returns,
#     mean="Constant",
#     vol="Garch",
#     p=1,
#     o=1,
#     q=1,
#     dist="StudentsT",
#     rescale=False,
# )
# ```
#
# Under the book's MXN-per-USD quote, a negative return means MXN appreciation
# and USD depreciation. A positive asymmetry coefficient therefore describes a
# direction-specific response; it should not be called an adverse “leverage
# effect” without first stating the position whose loss is being modeled.

# %% [markdown]
# ## Model limitations
#
# - FIX is a reference fixing, not a continuously traded executable price.
# - Publication intervals can span different numbers of calendar days.
# - Constant-mean GARCH is an applied candidate only when Lesson 2.4 selects an
#   intercept-only mean *and* fitted residual diagnostics are satisfactory;
#   otherwise the mean and variance should be fitted jointly.
# - Parameters and innovation shape can change across policy and crisis regimes.
# - Student-t innovations allow heavy tails but impose one fixed symmetric shape.
# - One-step VaR omits position size, nonlinear payoffs, liquidity, jumps,
#   parameter uncertainty, and backtesting evidence.

# %% [markdown] tags=["exercise"]
# ## Checkpoint exercise
#
# 1. For both fitted models, report $\hat\alpha$, $\hat\beta$, persistence,
#    long-run variance, and long-run FIX-interval volatility with correct units.
# 2. Report both end-to-end 99% VaRs and verify the Student-t unit-variance
#    scaling $\sqrt{(\nu-2)/\nu}$.
# 3. Use the decomposition to distinguish the mean/variance forecast effect from
#    the innovation-quantile effect.
# 4. Apply a 5% decision threshold to the three standardized-residual diagnostics
#    and state one model risk those tests cannot detect.

# %% tags=["solution"]
checkpoint_solution = pd.concat(
    [
        model_comparison[
            [
                "alpha_1",
                "beta_1",
                "persistence",
                "long_run_variance_pct_squared",
                "long_run_interval_volatility_pct",
            ]
        ],
        var_comparison[["one_step_var_positive_loss_pct"]],
        standardized_residual_diagnostics,
    ],
    axis=1,
)
checkpoint_solution

# %% [markdown] tags=["solution"]
# ```{dropdown} Suggested answer and interpretation
# Use the executed tables rather than fixed copied values. The Student-t scale
# check is `scipy.stats.t.ppf(alpha, nu) * sqrt((nu - 2) / nu)`. Diagnose mean
# dependence, variance dependence, and tail shape separately. Even satisfactory
# standardized-residual tests do not establish VaR calibration; that requires
# the exception and independence backtests developed in Module 7.
# ```

# %% [markdown]
# ## Handoff
#
# Lesson 2.6 varies transparent recursion assumptions without calling them
# estimates. Then Module 7 — Derivatives and Risk Management, especially its
# VaR foundations and backtesting lessons — evaluates tail forecasts against
# realized exceptions and deterministic stress scenarios.
