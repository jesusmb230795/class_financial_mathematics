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
# and calculates one-FIX-interval VaR on both log-threshold and exact
# simple-loss scales for a declared long-USD/short-MXN exposure
# {cite}`engle1982autoregressive,bollerslev1986generalized,tsay2010analysis`.
#
# ## Learning objectives
#
# By the end of this lab, readers should be able to:
#
# - fit and diagnose Gaussian and standardized Student-t GARCH models;
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
# ## Notation and units
#
# | Symbol | Meaning | Convention in this lesson |
# | --- | --- | --- |
# | $g_t$ | log return of MXN per USD | decimal; code fits $100g_t$ in percentage points |
# | $R_t$ | corresponding simple return, $\exp(g_t)-1$ | decimal unless displayed as a percentage |
# | $\mu_{t+1\mid t}$ | one-step conditional log-return mean | decimal in equations; percentage points in code tables |
# | $\sigma_{t+1\mid t}$ | one-step conditional log-return volatility | decimal in equations; percentage points in code tables |
# | $z_t$ | standardized innovation | unit variance |
# | $q_{\alpha}$ | left-tail quantile of $z_t$ | dimensionless |
# | $\alpha,\beta$ | GARCH shock and variance-memory coefficients | dimensionless; distinct from $\alpha_{\mathrm{tail}}$ |
# | $\alpha_{\mathrm{tail}}$ | left-tail probability | 0.01 in this lesson |
#
# ## Model and risk equations
#
# The GARCH(1,1) recursion is
#
# $$
# g_t = \mu + \varepsilon_t, \qquad \varepsilon_t = \sigma_t z_t,
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
# For a long-USD/short-MXN position whose MXN value changes with the MXN-per-USD
# quote, the conditional log-return quantile is
#
# $$
# q^{(g)}_{\alpha_{\mathrm{tail}},t+1\mid t}
# =\mu_{t+1\mid t}
# +\sigma_{t+1\mid t}q_{\alpha_{\mathrm{tail}}}.
# $$
#
# With $R=\exp(g)-1$, the book distinguishes a positive log-return loss
# threshold from the exact simple-return loss magnitude:
#
# $$
# \operatorname{VaR}^{(g)}_{\alpha_{\mathrm{tail}},t+1}
# =\max\left\{0,-\left(
# \mu_{t+1\mid t}
# +\sigma_{t+1\mid t}q_{\alpha_{\mathrm{tail}}}
# \right)\right\},
# $$
#
# $$
# \operatorname{VaR}^{(R)}_{\alpha_{\mathrm{tail}},t+1}
# =\max\left\{0,1-
# \exp\!\left(q^{(g)}_{\alpha_{\mathrm{tail}},t+1\mid t}\right)\right\}.
# $$
#
# The reported tail probability is $\alpha_{\mathrm{tail}}=0.01$ and the
# corresponding confidence level is $1-\alpha_{\mathrm{tail}}=0.99$. Because
# the fitted model uses $100g_t$, code divides its forecast quantile by 100
# before applying the exponential conversion.

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
from src.market_risk import (
    parametric_var,
    simple_loss_from_log_return,
    standardized_student_t_quantile,
)
from src.module2_visuals import (
    build_innovation_tail_risk_figure,
    build_volatility_comparison,
)
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
            "rights_note": (
                "provenance recorded; redistribution terms require external review"
            ),
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
# For Student-t degrees of freedom $\nu>2$, the unit-variance quantile is
#
# $$
# q_{\alpha,\nu}^{\mathrm{std}}
# =t_{\nu}^{-1}(\alpha)\sqrt{\frac{\nu-2}{\nu}}.
# $$
#
# The scale factor is essential: a raw Student-t quantile has variance
# $\nu/(\nu-2)$ and cannot be combined directly with an `arch` volatility
# forecast that already refers to standardized residuals.

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
# %% mystnb={"image": {"alt": "Two aligned panels show Banxico FIX log-return percentage points and conditional-volatility paths from Gaussian and standardized Student-t GARCH models."}}
volatility_figure = build_volatility_comparison(
    returns,
    volatility_paths,
    title="Banxico FIX (MXN per USD): conditional-volatility comparison",
    return_scale="percent",
    volatility_scale="percent",
    source=(
        "Banxico SIE SF43718; observed "
        f"{fix_level.index.min():%Y-%m-%d} to {fix_level.index.max():%Y-%m-%d}; "
        f"snapshot {banxico.attrs['snapshot_generated_at']}"
    ),
    data_mode=banxico.attrs["data_mode"],
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
    arch_lm = arch_lm_report(standardized_residuals, lags=10, model_df=0)
    diagnostic_rows.append(
        {
            "model": model_name,
            "standardized_residual_ljung_box_p10": residual_lb.loc[10, "lb_pvalue"],
            "squared_residual_ljung_box_p10": squared_lb.loc[10, "lb_pvalue"],
            "arch_lm_p10": arch_lm["lm_p_value"],
        }
    )

standardized_residual_diagnostics = pd.DataFrame(diagnostic_rows).set_index("model")
diagnostic_pvalue_columns = [
    "standardized_residual_ljung_box_p10",
    "squared_residual_ljung_box_p10",
    "arch_lm_p10",
]
if not np.isfinite(
    standardized_residual_diagnostics[diagnostic_pvalue_columns].to_numpy()
).all():
    raise RuntimeError("Standardized-residual diagnostics returned a non-finite p-value")
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
diagnostic_rejections = (
    standardized_residual_diagnostics[
        diagnostic_pvalue_columns
    ]
    < 0.05
)
standardized_residual_diagnostics["failed_diagnostic_count"] = (
    diagnostic_rejections.sum(axis=1).astype(int)
)
standardized_residual_diagnostics["overall_diagnostic_status"] = np.where(
    diagnostic_rejections.any(axis=1),
    "diagnostic warning: at least one 5% rejection",
    "no 5% rejection in the listed dependence diagnostics",
)
assert (
    standardized_residual_diagnostics["overall_diagnostic_status"].str.startswith(
        "diagnostic warning"
    )
    == (standardized_residual_diagnostics["failed_diagnostic_count"] > 0)
).all()
standardized_residual_diagnostics

# %% [markdown]
# **Output interpretation.**
#
# At a 5% threshold, a small first p-value flags remaining conditional-mean
# dependence; small squared-residual or ARCH-LM p-values flag variance dynamics
# the GARCH model has not absorbed. Because the conditional mean has no AR or MA
# lags, the ARCH-LM degrees-of-freedom adjustment is `model_df=0`. The overall
# status combines all three displayed tests. A rejection does not prevent the
# arithmetic below, but it prevents presenting the resulting VaR as a validated
# risk forecast.

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

normal_log_return_quantile_pct = (
    normal_mean_forecast + normal_volatility_forecast * normal_quantile
)
student_t_log_return_quantile_pct = (
    student_t_mean_forecast + student_t_volatility_forecast * student_t_quantile
)
student_t_gaussian_quantile_counterfactual_pct = (
    student_t_mean_forecast + student_t_volatility_forecast * normal_quantile
)

normal_log_loss_threshold_pct = parametric_var(
    normal_mean_forecast,
    normal_volatility_forecast,
    normal_quantile,
)
student_t_log_loss_threshold_pct = parametric_var(
    student_t_mean_forecast,
    student_t_volatility_forecast,
    student_t_quantile,
)
student_t_gaussian_quantile_log_loss_threshold_pct = parametric_var(
    student_t_mean_forecast,
    student_t_volatility_forecast,
    normal_quantile,
)
normal_simple_loss_var_pct = 100 * simple_loss_from_log_return(
    normal_log_return_quantile_pct / 100
)
student_t_simple_loss_var_pct = 100 * simple_loss_from_log_return(
    student_t_log_return_quantile_pct / 100
)
student_t_gaussian_quantile_simple_loss_var_pct = (
    100
    * simple_loss_from_log_return(
        student_t_gaussian_quantile_counterfactual_pct / 100
    )
)

var_comparison = pd.DataFrame(
    {
        "mean_forecast_pct": [normal_mean_forecast, student_t_mean_forecast],
        "volatility_forecast_pct": [
            normal_volatility_forecast,
            student_t_volatility_forecast,
        ],
        "standardized_quantile": [normal_quantile, student_t_quantile],
        "conditional_log_return_quantile_pct": [
            normal_log_return_quantile_pct,
            student_t_log_return_quantile_pct,
        ],
        "one_step_var_log_return_loss_threshold_pct": [
            normal_log_loss_threshold_pct,
            student_t_log_loss_threshold_pct,
        ],
        "one_step_var_simple_return_loss_pct": [
            normal_simple_loss_var_pct,
            student_t_simple_loss_var_pct,
        ],
        "tail_probability": [TAIL_PROBABILITY] * 2,
        "confidence": [CONFIDENCE_LEVEL] * 2,
        "horizon": ["next published FIX"] * 2,
        "position": ["long USD / short MXN"] * 2,
    },
    index=["Gaussian GARCH", "Student-t GARCH"],
)
var_comparison["diagnostic_status"] = standardized_residual_diagnostics[
    "overall_diagnostic_status"
]
var_comparison

# %%
var_decomposition = pd.Series(
    {
        "gaussian_model_log_loss_threshold_pct": normal_log_loss_threshold_pct,
        "student_t_forecast_with_gaussian_quantile_log_loss_threshold_pct": (
            student_t_gaussian_quantile_log_loss_threshold_pct
        ),
        "student_t_model_log_loss_threshold_pct": (
            student_t_log_loss_threshold_pct
        ),
        "mean_and_variance_effect_on_log_threshold_pct": (
            student_t_gaussian_quantile_log_loss_threshold_pct
            - normal_log_loss_threshold_pct
        ),
        "student_t_quantile_effect_on_log_threshold_pct": (
            student_t_log_loss_threshold_pct
            - student_t_gaussian_quantile_log_loss_threshold_pct
        ),
        "gaussian_model_simple_loss_var_pct": normal_simple_loss_var_pct,
        "student_t_forecast_with_gaussian_quantile_simple_loss_var_pct": (
            student_t_gaussian_quantile_simple_loss_var_pct
        ),
        "student_t_model_simple_loss_var_pct": student_t_simple_loss_var_pct,
        "mean_and_variance_effect_on_simple_loss_pct": (
            student_t_gaussian_quantile_simple_loss_var_pct
            - normal_simple_loss_var_pct
        ),
        "student_t_quantile_effect_on_simple_loss_pct": (
            student_t_simple_loss_var_pct
            - student_t_gaussian_quantile_simple_loss_var_pct
        ),
    }
)
var_decomposition

# %%
tail_risk_figure_input = var_comparison[
    [
        "one_step_var_log_return_loss_threshold_pct",
        "one_step_var_simple_return_loss_pct",
    ]
].rename(
    columns={
        "one_step_var_log_return_loss_threshold_pct": (
            "log_return_loss_threshold_pct"
        ),
        "one_step_var_simple_return_loss_pct": "simple_return_loss_pct",
    }
)
# %% mystnb={"image": {"alt": "The first panel compares unit-variance Gaussian and Student-t left tails with their one-percent quantiles. The second compares one-step log-return loss thresholds with exact simple-return VaR for both fitted GARCH models."}}
tail_risk_figure = build_innovation_tail_risk_figure(
    tail_risk_figure_input,
    alpha=TAIL_PROBABILITY,
    degrees_of_freedom=degrees_of_freedom,
    source=(
        "Banxico SIE SF43718; observed "
        f"{fix_level.index.min():%Y-%m-%d} to {fix_level.index.max():%Y-%m-%d}; "
        f"snapshot {banxico.attrs['snapshot_generated_at']}"
    ),
    data_mode=banxico.attrs["data_mode"],
)
display(tail_risk_figure)
plt.close(tail_risk_figure)

# %% [markdown]
# **Output interpretation.**
#
# The first table compares complete fitted models. The decomposition then holds
# the Student-t GARCH forecast fixed, so only its final difference can be called
# a quantile or tail-shape effect. The log-threshold columns remain in the
# fitted model's additive log-return scale. The simple-loss columns apply the
# exact $1-\exp(g)$ conversion and are the stated long-USD position-loss
# percentages. The overall diagnostic status prevents arithmetically valid VaR
# values from being mislabeled as validated forecasts; empirical calibration
# still requires the backtests in Module 7. The figure makes both the
# unit-variance tail comparison and the small but exact log-to-simple conversion
# visible without implying that either fitted model passed calibration.

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

# %% [markdown]
# ## Handoff
#
# [Lesson 2.6](2.6.interactive_volatility_garch_dashboard.ipynb) varies
# transparent recursion assumptions without calling them
# estimates. Then Module 7 — Derivatives and Risk Management, especially its
# VaR foundations and backtesting lessons — evaluates tail forecasts against
# realized exceptions and deterministic stress scenarios.
