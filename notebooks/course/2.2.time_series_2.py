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
# # ARCH and GARCH Model Comparison
#
# Module: Quantitative Methods and Financial Time Series
#
# ## Lesson summary
#
# This lesson holds the conditional mean and sample fixed, then compares one
# ARCH(3) variance specification with one GARCH(1,1) specification. The narrow
# comparison makes the extra lagged-variance term visible without turning this
# page into a general model tournament
# {cite}`engle1982autoregressive,bollerslev1986generalized`.
#
# ## Learning objectives
#
# By the end of this lesson, readers should be able to:
#
# - distinguish squared-shock memory from lagged-variance memory;
# - fit ARCH(3) and GARCH(1,1) on the same calculated MXN-per-USD FIX log returns;
# - verify convergence before interpreting a fitted model;
# - compare information criteria and annualized conditional-volatility paths;
# - explain why an in-sample comparison is not a forecasting claim.
#
# ## Prerequisites and scope
#
# [Lesson 2.3](2.3.time_series_diagnostics_and_volatility_extensions.ipynb)
# covers the pre-model diagnostics, and
# [Lesson 2.4](2.4.arima_diagnostic_workflow.ipynb) covers
# conditional-mean selection. A constant mean is a controlled assumption for
# isolating the two variance recursions, not a validated applied mean
# specification. This notebook does not import fitted objects from Lesson 2.4;
# readers must compare its executed mean-model evidence before treating this as
# an aligned applied specification. A zero-lag conditional mean does **not**
# imply independent returns: ARCH and GARCH explicitly model dependence in the
# second conditional moment.

# %% [markdown]
# ## Notation and admissibility
#
# | Symbol | Meaning | Unit or condition |
# | --- | --- | --- |
# | $g_t^{(\%)}=100\log(P_t/P_{t-1})$ | FIX log return at publication step $t$ | percentage points |
# | $\mu$ | constant conditional mean | percentage points per publication interval |
# | $\varepsilon_t$ | return innovation, $g_t^{(\%)}-\mu$ | percentage points |
# | $z_t$ | standardized innovation, $\varepsilon_t/\sigma_t$ | $\mathbb E[z_t]=0$, $\operatorname{Var}(z_t)=1$ |
# | $\sigma_t^2$ | conditional variance given information through $t-1$ | percentage points squared |
# | $\omega$ | variance intercept | strictly positive |
# | $\alpha_i,\beta_j$ | shock and variance-memory coefficients | non-negative |

# %% [markdown]
# ## Model definitions
#
# The common constant-mean equation is
#
# $$
# g_t^{(\%)}=\mu+\varepsilon_t,
# \qquad
# \varepsilon_t=\sigma_t z_t.
# $$
#
# An ARCH($p$) model updates conditional variance from the previous $p$ squared
# innovations:
#
# $$
# \sigma_t^2
# = \omega + \sum_{i=1}^{p}\alpha_i\varepsilon_{t-i}^2.
# $$
#
# A GARCH($p,q$) model adds $q$ lagged conditional variances:
#
# $$
# \sigma_t^2
# = \omega
# + \sum_{i=1}^{p}\alpha_i\varepsilon_{t-i}^2
# + \sum_{j=1}^{q}\beta_j\sigma_{t-j}^2.
# $$
#
# With unit-variance innovations, the usual finite unconditional-variance
# condition is
#
# $$
# \sum_{i=1}^{p}\alpha_i+\sum_{j=1}^{q}\beta_j<1.
# $$
#
# This page follows the `arch` package convention: $p$ counts squared-shock
# lags and $q$ counts conditional-variance lags. Reporting the equation avoids
# ambiguity across textbook conventions
# {cite}`bollerslev1986generalized,sheppard2024arch`.

# %% tags=["setup", "hide-input"]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from IPython.display import display

from src.market_data import banxico_daily_panel
from src.market_data_quality import log_returns
from src.module2_visuals import build_volatility_comparison

# %% [markdown]
# ## Reproducible source inventory
#
# The analysis uses only the dates on which Banxico published FIX series
# SF43718, quoted as MXN per USD {cite}`banxicoSIE2025`. It does not reindex to
# an artificial business-day calendar and does not forward-fill weekends,
# holidays, or missing observations.

# %%
ANALYSIS_START = "2021-01-01"
ANALYSIS_END = "2026-06-05"
ANNUALIZATION_FACTOR = 252

banxico_panel = banxico_daily_panel(start=ANALYSIS_START, end=ANALYSIS_END)
series_ids = banxico_panel.attrs.get("series_ids", {})
snapshot_generated_at = banxico_panel.attrs.get("snapshot_generated_at")
if series_ids.get("usd_mxn") != "SF43718":
    raise ValueError("The publication snapshot must map usd_mxn to Banxico SF43718.")

usd_mxn_fix = (
    pd.to_numeric(banxico_panel["usd_mxn"], errors="coerce")
    .dropna()
    .rename("usd_mxn_fix_mxn_per_usd")
)
if usd_mxn_fix.empty or not usd_mxn_fix.index.is_monotonic_increasing:
    raise ValueError("SF43718 must contain ordered published observations.")
if usd_mxn_fix.index.has_duplicates:
    raise ValueError("SF43718 must contain at most one published value per date.")
if not np.isfinite(usd_mxn_fix.to_numpy()).all() or (usd_mxn_fix <= 0).any():
    raise ValueError("SF43718 levels must be finite and strictly positive.")

interval_returns_pct = log_returns(usd_mxn_fix).mul(100).rename(
    "usd_mxn_fix_interval_log_return_pct"
)
observed_gap_days = usd_mxn_fix.index.to_series().diff().dt.days.dropna()

source_inventory = pd.DataFrame(
    [
        ("provider", "Banco de México, SIE"),
        ("series", "SF43718 — Banxico FIX, MXN per USD"),
        ("snapshot vintage", snapshot_generated_at or "not recorded"),
        ("observed start", usd_mxn_fix.index.min().strftime("%Y-%m-%d")),
        ("observed end", usd_mxn_fix.index.max().strftime("%Y-%m-%d")),
        ("published level observations", f"{len(usd_mxn_fix):,}"),
        ("computed return observations", f"{len(interval_returns_pct):,}"),
        ("level unit", "MXN per USD"),
        ("return unit", "log-return percentage points per FIX publication interval"),
        (
            "observed calendar gaps",
            (
                f"min {int(observed_gap_days.min())}, "
                f"median {observed_gap_days.median():.0f}, "
                f"max {int(observed_gap_days.max())} days"
            ),
        ),
        (
            "annualization convention",
            "square-root-of-time with 252 FIX publication intervals/year",
        ),
        ("observation policy", banxico_panel.attrs.get("observation_policy", "not recorded")),
        (
            "rights note",
            "provenance recorded; redistribution terms require external review",
        ),
    ],
    columns=["inventory item", "value"],
)
source_inventory

# %% [markdown]
# **How to read the inventory.**
#
# The level count and return count differ by one because the first published
# level has no previous published observation. A return across a multi-day
# calendar gap remains one observed-to-observed FIX return; no synthetic daily
# values are inserted between the two publication dates.

# %% [markdown]
# ## Fit ARCH(3) and GARCH(1,1)
#
# Both models use Gaussian innovations, `rescale=False`, the same constant
# mean, the same percentage-point return series, and the same estimation
# window. Only the conditional-variance specification changes.

# %%
arch_result = arch_model(
    interval_returns_pct,
    mean="Constant",
    vol="ARCH",
    p=3,
    dist="normal",
    rescale=False,
).fit(disp="off", cov_type="robust")

garch_result = arch_model(
    interval_returns_pct,
    mean="Constant",
    vol="Garch",
    p=1,
    q=1,
    dist="normal",
    rescale=False,
).fit(disp="off", cov_type="robust")

fitted_results = {
    "ARCH(3)": arch_result,
    "GARCH(1,1)": garch_result,
}
for model_name, fitted_result in fitted_results.items():
    if int(fitted_result.convergence_flag) != 0:
        raise RuntimeError(
            f"{model_name} did not converge: convergence_flag={fitted_result.convergence_flag}"
        )

# %% [markdown]
# ## Curated parameter evidence
#
# The table replaces the package's timestamped full summary with the estimates
# needed for this comparison. Confidence intervals are 95% Wald intervals from
# the explicitly requested robust covariance estimator; both likelihoods use
# Gaussian innovations.

# %%
parameter_rows = []
for model_name, fitted_result in fitted_results.items():
    confidence_intervals = fitted_result.conf_int(alpha=0.05)
    for parameter_name, estimate in fitted_result.params.items():
        if parameter_name == "mu":
            unit = "return percentage points per FIX publication interval"
        elif parameter_name == "omega":
            unit = "percentage points squared"
        else:
            unit = "dimensionless"
        parameter_rows.append(
            {
                "model": model_name,
                "parameter": parameter_name,
                "estimate": estimate,
                "standard error": fitted_result.std_err[parameter_name],
                "95% lower": confidence_intervals.loc[parameter_name, "lower"],
                "95% upper": confidence_intervals.loc[parameter_name, "upper"],
                "p-value": fitted_result.pvalues[parameter_name],
                "unit": unit,
            }
        )

parameter_table = pd.DataFrame(parameter_rows)
parameter_table

# %% [markdown]
# **Interpretation boundary.**
#
# The mean coefficient describes the conditional center of the percentage
# return series. The variance coefficients describe magnitude dynamics, not
# the direction of the next exchange-rate change. A confidence interval is
# conditional on the chosen likelihood and specification; it is not an
# out-of-sample forecast interval.

# %% [markdown]
# ## Compare fit and variance memory

# %%
arch_alpha_names = [
    parameter_name
    for parameter_name in arch_result.params.index
    if parameter_name.startswith("alpha[")
]
arch_alpha_sum = float(arch_result.params.loc[arch_alpha_names].sum())
garch_alpha_plus_beta = float(garch_result.params["alpha[1]"] + garch_result.params["beta[1]"])
if not np.isfinite(arch_alpha_sum) or not 0 <= arch_alpha_sum < 1:
    raise RuntimeError("ARCH(3) variance memory must be finite and below one")
if not np.isfinite(garch_alpha_plus_beta) or not 0 <= garch_alpha_plus_beta < 1:
    raise RuntimeError("GARCH(1,1) persistence must be finite and below one")

comparison_table = pd.DataFrame(
    [
        {
            "model": "ARCH(3)",
            "log likelihood": arch_result.loglikelihood,
            "AIC": arch_result.aic,
            "BIC": arch_result.bic,
            "variance-memory measure": arch_alpha_sum,
            "measure definition": "sum(alpha_i), i=1,...,3",
            "convergence flag": int(arch_result.convergence_flag),
            "N returns": int(arch_result.nobs),
        },
        {
            "model": "GARCH(1,1)",
            "log likelihood": garch_result.loglikelihood,
            "AIC": garch_result.aic,
            "BIC": garch_result.bic,
            "variance-memory measure": garch_alpha_plus_beta,
            "measure definition": "alpha[1] + beta[1]",
            "convergence flag": int(garch_result.convergence_flag),
            "N returns": int(garch_result.nobs),
        },
    ]
).set_index("model")
comparison_table

# %% [markdown]
# Lower AIC and BIC are preferred only within this same-sample, same-likelihood
# comparison. The ARCH sum and GARCH sum have related persistence
# interpretations, but they arise from different recursions and should not be
# treated as interchangeable model scores.

# %% [markdown]
# ## Annualized conditional-volatility paths
#
# Conditional volatility is in percentage points per FIX publication interval
# because the model input uses that horizon. For display, each path is multiplied
# by $\sqrt{252}$ and labeled as annualized percentage points under a declared
# square-root-of-time convention.

# %% mystnb={"image": {"alt": "Two aligned panels show Banxico FIX publication-interval log returns and annualized conditional-volatility paths from ARCH(3) and GARCH(1,1)."}}
interval_volatility_paths = pd.concat(
    {
        "ARCH(3)": arch_result.conditional_volatility,
        "GARCH(1,1)": garch_result.conditional_volatility,
    },
    axis=1,
).dropna()
annualized_volatility_paths = interval_volatility_paths.mul(
    np.sqrt(ANNUALIZATION_FACTOR)
)

volatility_figure = build_volatility_comparison(
    interval_returns_pct,
    interval_volatility_paths,
    title="Banxico FIX (MXN per USD): ARCH(3) versus GARCH(1,1) volatility",
    return_scale="percent",
    volatility_scale="percent",
    annualization_factor=ANNUALIZATION_FACTOR,
    source=(
        "Banxico SIE SF43718; observed "
        f"{usd_mxn_fix.index.min():%Y-%m-%d} to {usd_mxn_fix.index.max():%Y-%m-%d}; "
        f"snapshot {snapshot_generated_at}; provider-dated observations, no forward fill"
    ),
    data_mode=banxico_panel.attrs.get("data_mode"),
)
display(volatility_figure)
plt.close(volatility_figure)

# %%
volatility_path_summary = annualized_volatility_paths.agg(
    ["min", "median", "max"]
).T.rename(
    columns={
        "min": "minimum_annualized_volatility_pct",
        "median": "median_annualized_volatility_pct",
        "max": "maximum_annualized_volatility_pct",
    }
)
volatility_path_summary["observations"] = annualized_volatility_paths.notna().sum()
volatility_path_summary["unit"] = (
    "annualized percentage points; sqrt(252) FIX intervals/year"
)
volatility_path_summary

# %% [markdown]
# The shared axis supports a direct path comparison, while the summary reports
# each path on the same scale without relying on visual estimation. Spikes are
# conditional standard-deviation estimates, not realized losses. Annualization
# is a scale convention; it does not assert that volatility is constant across
# 252 future observations.

# %%
preferred_by_aic = comparison_table["AIC"].idxmin()
preferred_by_bic = comparison_table["BIC"].idxmin()
model_decision = pd.DataFrame(
    [
        {
            "criterion": "AIC",
            "programmatic result": preferred_by_aic,
            "interpretation": "lowest value in this two-model in-sample comparison",
        },
        {
            "criterion": "BIC",
            "programmatic result": preferred_by_bic,
            "interpretation": "lowest value with a stronger complexity penalty",
        },
        {
            "criterion": "forecast evidence",
            "programmatic result": "not tested",
            "interpretation": "requires a dated holdout or rolling backtest",
        },
    ]
)
model_decision

# %% [markdown]
# ## Limitations
#
# - The comparison is limited to ARCH(3) and Gaussian GARCH(1,1); it is not a
#   search across orders or innovation distributions.
# - Parameters can change after market or policy regime shifts.
# - Extreme returns can dominate Gaussian maximum-likelihood estimates.
# - AIC and BIC summarize in-sample fit and complexity, not future accuracy.
# - The 252-factor annualization is a reporting convention applied to an
#   irregular provider-dated publication series.
# - A stationary variance recursion is not evidence that the exchange rate
#   level or the return-generating process is structurally stable.

# %% [markdown]
# ## Handoff
#
# [Lesson 2.5](2.5.garch_volatility_risk_workflow.ipynb) extends the volatility
# workflow with heavy-tailed innovations,
# forecasts, and positive-loss VaR. Carry forward the source inventory,
# convergence guard, explicit return scale, and distinction between in-sample
# fit and forecast evidence.
