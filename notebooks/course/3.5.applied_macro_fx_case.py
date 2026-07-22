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
# # Applied Macro-to-FX Model Review Case
#
# Module: Economics, Macro, and Currency
#
# ## Lesson summary
#
# This case audits a small macro-to-FX predictive workflow rather than assuming
# that a fitted model deserves investment use. It loads a frozen monthly panel,
# preserves the canonical `MXN per USD` quote, lags inflation to approximate its
# release availability, keeps forecast origins chronological, and compares Ridge
# with both a training-mean benchmark and a zero-change FX benchmark. A model
# rejection is a valid result {cite}`mishkin2019financial,banxicoSIE2025,dbnomics2025`.
#
# The final section applies explicit macro shocks only as a **rejected-model
# stability diagnostic**. It does not publish an FX target, cash-flow conversion,
# hedge recommendation, or causal policy claim.
#
# ## Learning objectives
#
# By the end of this lesson, readers should be able to:
#
# - audit provider, series, unit, sample, retrieval timestamp, and release timing;
# - construct a conservative inflation-availability proxy without same-month CPI
#   look-ahead;
# - evaluate a fixed Ridge model with a chronological holdout and two naive
#   benchmarks;
# - apply an explicit acceptance gate and communicate model rejection directly;
# - compare scenario sensitivities across estimation windows and against observed
#   forecast error; and
# - write a short decision note that separates mechanism, evidence, alternative
#   explanations, and limitations.
#
# ## Prerequisites
#
# Complete the Module 3 sequence from
# [Economic Foundations](3.1.economic_foundations.md) through
# [Macro Scenarios and Capital Market Expectations](3.4.macro_scenarios_cme.md).
# The case also assumes the log-return, regression, regularization, MAE, and
# chronological-validation foundations from
# [Module 2](../../chapters/02-time-series.md). Retain the quote convention
# \(S_{\mathrm{MXN/USD}}\): an increase means USD appreciation and MXN
# depreciation. Rates and returns are stored as decimals in code.

# %% [markdown]
# ## Setup

# %% tags=["setup", "hide-input"]
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.macro_fx import (
    FEATURE_COLUMNS,
    REQUIRED_MACRO_COLUMNS,
    TARGET_COLUMN,
    chronological_split,
    prepare_macro_fx_features,
    scenario_feature_frame,
)
from src.module3_visuals import (
    build_macro_fx_evaluation_figure,
    build_macro_fx_history_figure,
    build_macro_fx_scenario_stability_figure,
)

PROJECT_ROOT = Path.cwd().resolve()
for candidate in (PROJECT_ROOT, *PROJECT_ROOT.parents):
    if (candidate / "pyproject.toml").is_file():
        PROJECT_ROOT = candidate
        break
else:
    raise FileNotFoundError(
        "Could not locate the repository root containing pyproject.toml"
    )

SNAPSHOT_PATH = PROJECT_ROOT / "data/snapshots/official_macro_panel.csv"
METADATA_PATH = PROJECT_ROOT / "data/snapshots/metadata.json"

# %% [markdown]
# ## Snapshot, series, sample, and timing
#
# `official_macro_panel.csv` is a committed latest-vintage teaching snapshot.
# Banxico supplies the domestic rates and USD/MXN FIX. DB.NOMICS distributes the
# IMF Mexico CPI index and Federal Reserve H.15 US 10-year Treasury yield. The
# case uses only the fields listed below; TIIE, UDI, and the CPI level remain in
# the shared panel for other lessons but are not predictors here.
#
# | Case field | Provider and series | Stored unit | Monthly construction and use |
# | --- | --- | --- | --- |
# | `banxico_target_rate` | Banxico SIE `SF61745` | decimal annual target rate | Last available observation in reference month |
# | `cetes_28d` | Banxico SIE `SF60633` | decimal annual quoted rate | Last available observation; forms a tenor-mismatched proxy gap |
# | `usd_mxn` | Banxico SIE `SF43718` | MXN per USD FIX | Last available level; target is next-month log return |
# | `mexico_inflation` | DB.NOMICS `IMF/CPI/M.MX.PCPI_IX` | decimal year-over-year CPI change, constructed as \(CPI_t/CPI_{t-12}-1\) | Shifted one month before modeling as an availability proxy |
# | `us_10y` | DB.NOMICS `FED/H15/RIFLGFCY10_N.B` | decimal annual yield | Last available observation; forms a maturity-mismatched proxy spread |
#
# A month label is a period end, not a release timestamp. INEGI's corresponding
# June 2025 CPI release was published on 2025-07-09, after the June month-end
# forecast origin {cite}`inegiINPCJune2025`. That evidence supports a
# conservative one-month lag, but it is not an exact availability timestamp for
# the IMF/DB.NOMICS extract. This file is not a historical real-time-vintage
# database.

# %%
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
macro = pd.read_csv(SNAPSHOT_PATH, parse_dates=["date"]).set_index("date").sort_index()

missing_columns = REQUIRED_MACRO_COLUMNS - set(macro.columns)
if missing_columns:
    raise ValueError(f"snapshot is missing required columns: {sorted(missing_columns)}")
if not macro.index.is_monotonic_increasing or macro.index.has_duplicates:
    raise ValueError("snapshot dates must be unique and increasing")

recorded_coverage = metadata["output_coverage"]["official_macro_panel"]
sample_start = macro.index.min().date().isoformat()
sample_end = macro.index.max().date().isoformat()
generation_timestamp = metadata["generated_at"]
generation_date = generation_timestamp[:10]
assert sample_start == recorded_coverage["start"]
assert sample_end == recorded_coverage["end"]

sample_audit = pd.DataFrame(
    [
        {"Audit item": "Snapshot", "Value": SNAPSHOT_PATH.relative_to(PROJECT_ROOT)},
        {"Audit item": "Generation / retrieval timestamp (UTC)", "Value": generation_timestamp},
        {"Audit item": "Actual sample start (inclusive)", "Value": sample_start},
        {"Audit item": "Actual sample end (inclusive)", "Value": sample_end},
        {"Audit item": "Monthly period rows", "Value": len(macro)},
        {
            "Audit item": "Missing values in case fields",
            "Value": int(macro[list(REQUIRED_MACRO_COLUMNS)].isna().sum().sum()),
        },
        {"Audit item": "Execution mode", "Value": "Offline frozen snapshot"},
    ]
)
display(
    sample_audit.style.hide(axis="index").set_caption(
        "Table 1. Snapshot and observed-sample audit"
    )
)

# %% [markdown]
# **Output interpretation.**
#
# The panel contains 78 consecutive monthly period labels from 2019-01-31
# through 2025-06-30 and no missing case fields. The 2026-06-07 timestamp records
# when this current-vintage extract was generated; it is not an observation date
# and the panel must not be presented as current market data.

# %% [markdown]
# ## From period-labeled observations to available features
#
# At each month-end origin \(t\), the target is the next monthly USD/MXN log
# return:
#
# ```{math}
# g^{FX}_{t+1}=\log\left(\frac{S_{t+1}}{S_t}\right).
# ```
#
# A positive target is USD appreciation and MXN depreciation. The feature
# contract is deliberately narrow:
#
# | Feature at origin \(t\) | Construction | Availability interpretation |
# | --- | --- | --- |
# | Policy-minus-US-10-year proxy | Mexico policy rate minus US 10-year yield | Both are period-\(t\) market/policy observations; maturities do not match |
# | Lagged Mexico inflation | Year-over-year inflation labeled \(t-1\) | One-month release-lag proxy; not a true vintage join |
# | CETES-policy proxy gap | CETES 28-day rate minus policy rate | Period-\(t\) values; tenor and quotation conventions differ |
# | FX momentum | \(\log(S_t/S_{t-3})\) | Uses spot levels at \(t\) or earlier |
#
# Neither rate spread is covered interest parity or a tradable carry basis.

# %%
macro_features, model_data = prepare_macro_fx_features(
    macro,
    inflation_release_lag_months=1,
)
model_origin_start = model_data.index.min().date().isoformat()
model_origin_end = model_data.index.max().date().isoformat()
model_realization_end = macro.index[macro.index.get_loc(model_data.index.max()) + 1]

model_sample_audit = pd.DataFrame(
    [
        {"Audit item": "First usable forecast origin", "Value": model_origin_start},
        {"Audit item": "Last usable forecast origin", "Value": model_origin_end},
        {
            "Audit item": "Last target realization date",
            "Value": model_realization_end.date().isoformat(),
        },
        {"Audit item": "Complete model origins", "Value": len(model_data)},
        {"Audit item": "Inflation availability lag", "Value": "1 month"},
        {"Audit item": "FX momentum lookback", "Value": "3 months"},
    ]
)
display(
    model_sample_audit.style.hide(axis="index").set_caption(
        "Table 2. Aligned feature and target sample"
    )
)

history_display = macro_features[
    [
        "banxico_target_rate",
        "cetes_28d",
        "mexico_inflation_lag1m",
        "us_10y",
        "usd_mxn",
    ]
].dropna()
history_sample_label = (
    f"{history_display.index.min().date().isoformat()} to "
    f"{history_display.index.max().date().isoformat()}"
)
history_figure = build_macro_fx_history_figure(
    macro_features,
    sample_label=history_sample_label,
    vintage_label=generation_date,
)
display(history_figure)
plt.close(history_figure)

# %% [markdown]
# **Figure description and takeaway.**
#
# The three aligned panels keep annual rates, the lagged inflation proxy, and
# MXN-per-USD levels in their native units. They show the information entering
# the case without implying that visual co-movement is stable or causal. The
# inflation line begins one period later because same-month inflation is not
# treated as available at the forecast origin. The remaining limitation is
# material: the lag is an approximation applied to a current-vintage panel, not
# a reconstruction of every historical release set.

# %% [markdown]
# ## Chronological estimation, transparent benchmarks, and gate
#
# The first 75% of usable forecast origins form the training sample; the final
# 25% form a later holdout. Scaling and Ridge estimation are fitted inside one
# pipeline using training data only. `alpha=4.0` is a fixed classroom shrinkage
# setting declared before viewing the holdout; it is not tuned on test data.
#
# Two naive benchmarks make the decision standard explicit:
#
# - **training mean:** predict the training-sample mean log return every month;
# - **zero change:** predict a zero log return, equivalent to a random-walk next
#   USD/MXN level.
#
# The model is accepted only if its holdout MAE is lower than **both** naive
# benchmarks. This gate is intentionally simple and does not establish
# deployability even if passed.

# %%
training, testing = chronological_split(model_data, training_fraction=0.75)

evaluation_model = Pipeline(
    [
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=4.0)),
    ]
)
evaluation_model.fit(training[list(FEATURE_COLUMNS)], training[TARGET_COLUMN])

actual_test = testing[TARGET_COLUMN]
prediction_map = {
    "Ridge prediction": evaluation_model.predict(testing[list(FEATURE_COLUMNS)]),
    "Training-mean benchmark": np.repeat(training[TARGET_COLUMN].mean(), len(testing)),
    "Zero-change benchmark": np.zeros(len(testing)),
}
evaluation_frame = pd.DataFrame(
    {"Actual next-month return": actual_test, **prediction_map},
    index=testing.index,
)

metric_rows = []
for model_name, predictions in prediction_map.items():
    directional_accuracy = (
        np.nan
        if model_name == "Zero-change benchmark"
        else np.mean(np.sign(actual_test) == np.sign(predictions))
    )
    metric_rows.append(
        {
            "Model": model_name,
            "mae": mean_absolute_error(actual_test, predictions),
            "rmse": mean_squared_error(actual_test, predictions) ** 0.5,
            "r2": r2_score(actual_test, predictions),
            "directional_accuracy": directional_accuracy,
        }
    )
evaluation_metrics = pd.DataFrame(metric_rows).set_index("Model")
benchmark_names = ["Training-mean benchmark", "Zero-change benchmark"]
best_benchmark_name = evaluation_metrics.loc[benchmark_names, "mae"].idxmin()
best_benchmark_mae = evaluation_metrics.loc[best_benchmark_name, "mae"]
ridge_mae = evaluation_metrics.loc["Ridge prediction", "mae"]
model_accepted = bool(
    all(ridge_mae < evaluation_metrics.loc[name, "mae"] for name in benchmark_names)
)
skill_vs_best_benchmark = 1 - ridge_mae / best_benchmark_mae
assert not model_accepted

metrics_display = pd.DataFrame(
    {
        "Test MAE (log-return percentage points)": 100 * evaluation_metrics["mae"],
        "Test RMSE (log-return percentage points)": 100 * evaluation_metrics["rmse"],
        "R-squared": evaluation_metrics["r2"],
        "Direction correct (%)": 100 * evaluation_metrics["directional_accuracy"],
    }
)
display(
    metrics_display.style.format(
        {
            "Test MAE (log-return percentage points)": "{:.3f}",
            "Test RMSE (log-return percentage points)": "{:.3f}",
            "R-squared": "{:.3f}",
            "Direction correct (%)": "{:.1f}",
        },
        na_rep="not applicable",
    ).set_caption("Table 3. Frozen chronological holdout results")
)

decision_gate = pd.DataFrame(
    [
        {"Gate item": "Training origins", "Result": len(training)},
        {"Gate item": "Test origins", "Result": len(testing)},
        {
            "Gate item": "Training cutoff",
            "Result": training.index.max().date().isoformat(),
        },
        {"Gate item": "Best naive benchmark", "Result": best_benchmark_name},
        {
            "Gate item": "Ridge skill vs best benchmark",
            "Result": f"{100 * skill_vs_best_benchmark:.2f}%",
        },
        {
            "Gate item": "Decision",
            "Result": "REJECTED — Ridge MAE does not beat both benchmarks",
        },
    ]
)
display(
    decision_gate.style.hide(axis="index").set_caption(
        "Table 4. Predeclared model-acceptance gate"
    )
)

# %% [markdown]
# **Output interpretation.**
#
# The training sample contains 55 origins through 2023-10-31; the test contains
# 19 later origins from 2023-11-30 through 2025-05-31, with target realizations
# through 2025-06-30. Ridge MAE is 0.02394 in log-return units, or 2.394
# percentage points after multiplying by 100, versus 2.176 and 2.190 percentage
# points for the training-mean and zero-change benchmarks. Ridge is 10.02%
# worse than the best naive benchmark, its \(R^2\) is -0.123, and it gets the
# return sign right in only 6 of 19 months. The model is therefore **rejected**.
# Failure to beat a transparent benchmark is the main result, not an output to
# hide or repair by tuning against the same holdout.

# %%
mae_by_model = evaluation_metrics["mae"].to_dict()
test_sample_label = (
    f"{testing.index.min().date().isoformat()} to "
    f"{testing.index.max().date().isoformat()}"
)
evaluation_figure = build_macro_fx_evaluation_figure(
    evaluation_frame,
    mae_by_model,
    sample_label=test_sample_label,
    realization_end=model_realization_end.date().isoformat(),
    vintage_label=generation_date,
)
display(evaluation_figure)
plt.close(evaluation_figure)

# %% [markdown]
# **Figure description and takeaway.**
#
# The upper panel aligns each realized return with its month-end forecast origin;
# the lower panel answers the decision question directly. Ridge has the largest
# MAE of the three methods, so the figure provides no visual or numerical basis
# for promoting it. The result is specific to 19 held-out origins in one frozen
# latest-vintage sample; it is not evidence that macro variables never matter
# for FX.

# %% [markdown]
# ## Rejected-model scenario stability diagnostic
#
# The gate prevents a production refit and prevents publication of a next-month
# FX level or converted cash flow. The code below is retained only to answer a
# model-risk question: **would the response to the same explicit shocks remain
# stable if the estimation window were updated?**
#
# `Observed-input baseline` is a mechanical reference, not the most probable
# macro scenario. Every non-baseline case changes only the listed annual-rate
# inputs; three-month FX momentum remains observed.
#
# | Scenario | Trigger and mechanism hypothesis | Monitoring evidence and limitation |
# | --- | --- | --- |
# | Sticky inflation and domestic tightening | Lagged-inflation proxy +100 bp, policy +100 bp, CETES +125 bp | CPI release and Banxico decision; model coefficients are predictive, not causal |
# | Global long-yield shock | US 10-year +75 bp, CETES +25 bp | H.15 yield and global risk conditions; maturities and instruments do not match |
# | Disinflation and domestic easing | Lagged-inflation proxy -100 bp, policy -100 bp, CETES -125 bp | CPI and policy path; symmetry is an imposed stress design, not a forecast |
#
# The **evaluation fit** remains trained only through 2023-10. An **updated fit**
# uses all 74 labeled origins through 2025-05 solely to diagnose window
# sensitivity. It is not a validated production model.

# %%
latest = macro_features.iloc[-1].copy()
latest_date = macro_features.index[-1]

scenario_inputs = pd.DataFrame(
    [
        {
            "scenario": "Observed-input baseline",
            "policy_shock": 0.0000,
            "cetes_shock": 0.0000,
            "lagged_inflation_proxy_shock": 0.0000,
            "us_10y_shock": 0.0000,
        },
        {
            "scenario": "Sticky inflation and domestic tightening",
            "policy_shock": 0.0100,
            "cetes_shock": 0.0125,
            "lagged_inflation_proxy_shock": 0.0100,
            "us_10y_shock": 0.0000,
        },
        {
            "scenario": "Global long-yield shock",
            "policy_shock": 0.0000,
            "cetes_shock": 0.0025,
            "lagged_inflation_proxy_shock": 0.0000,
            "us_10y_shock": 0.0075,
        },
        {
            "scenario": "Disinflation and domestic easing",
            "policy_shock": -0.0100,
            "cetes_shock": -0.0125,
            "lagged_inflation_proxy_shock": -0.0100,
            "us_10y_shock": 0.0000,
        },
    ]
).set_index("scenario")
scenario_features = scenario_feature_frame(scenario_inputs, latest)

updated_fit = clone(evaluation_model)
updated_fit.fit(model_data[list(FEATURE_COLUMNS)], model_data[TARGET_COLUMN])
diagnostic_models = {
    "Evaluation fit": evaluation_model,
    "Updated fit": updated_fit,
}
scenario_sensitivity = pd.DataFrame(index=scenario_inputs.index)
for fit_name, fitted_model in diagnostic_models.items():
    predictions = pd.Series(
        fitted_model.predict(scenario_features[list(FEATURE_COLUMNS)]),
        index=scenario_features.index,
    )
    baseline_prediction = predictions.loc["Observed-input baseline"]
    scenario_sensitivity[fit_name] = 10_000 * (
        predictions - baseline_prediction
    )

expected_sensitivity_bps = np.array(
    [
        [0.0, 0.0],
        [-33.124177, 46.681668],
        [-15.469700, 26.453655],
        [33.124177, -46.681668],
    ]
)
np.testing.assert_allclose(
    scenario_sensitivity[["Evaluation fit", "Updated fit"]],
    expected_sensitivity_bps,
    atol=1e-6,
)

scenario_assumptions_display = 10_000 * scenario_inputs.rename(
    columns={
        "policy_shock": "Policy shock (bp)",
        "cetes_shock": "CETES shock (bp)",
        "lagged_inflation_proxy_shock": "Lagged-inflation-proxy shock (bp)",
        "us_10y_shock": "US 10-year shock (bp)",
    }
)
display(
    scenario_assumptions_display.style.format("{:+.0f}").set_caption(
        "Table 5. Explicit scenario shocks in annual-rate basis points"
    )
)

scenario_sensitivity_display = scenario_sensitivity.drop(
    index="Observed-input baseline"
)
display(
    scenario_sensitivity_display.style.format("{:+.1f}").set_caption(
        "Table 6. Change versus each fit's baseline, basis points in one-month USD/MXN log-return units"
    )
)

scenario_figure = build_macro_fx_scenario_stability_figure(
    scenario_sensitivity_display,
    model_mae_bps=10_000 * ridge_mae,
    fit_window_labels={
        "Evaluation fit": (
            f"Evaluation fit (origins through {training.index.max():%Y-%m})"
        ),
        "Updated fit": (
            f"Updated fit (origins through {model_data.index.max():%Y-%m})"
        ),
    },
    sample_label=f"scenario origin {latest_date.date().isoformat()}",
    vintage_label=generation_date,
)
display(scenario_figure)
plt.close(scenario_figure)

# %% [markdown]
# **Figure description and takeaway.**
#
# Every non-baseline scenario reverses direction when the estimation window is
# updated. The largest displayed sensitivity is about 46.7 basis points of
# monthly log-return units, while Ridge's held-out MAE is about 239.4 basis points.
# MAE is not a prediction interval, but the scale comparison and sign reversals
# show that the scenario response is neither stable nor large relative to the
# observed error. No fitted FX level should be passed to valuation, hedging, or
# portfolio decisions from this model.

# %% [markdown]
# ## Limitations and decision boundary
#
# - The 78-row panel is small, monthly, and ends on 2025-06-30; it is neither
#   live nor a real-time-vintage database.
# - The one-month inflation lag removes the known same-month CPI leak but cannot
#   reproduce exact release dates, revisions, or information available at each
#   historical origin.
# - Month-end dates are period labels; some fall on weekends and are not source
#   observation or publication dates.
# - Policy-minus-US-10-year and CETES-minus-policy features mix maturities,
#   instruments, calendars, and quotation conventions. They are not tradable
#   carry spreads, yield-curve slopes, CIP bases, or causal policy shocks.
# - `alpha=4.0` is fixed for a transparent holdout demonstration. Future tuning
#   would require time-aware validation inside the training period, not reuse of
#   this test sample.
# - Ridge fails both naive benchmarks. The updated fit is shown only to expose
#   instability and has no separate validation claim.
# - The model omits intervention, positioning, global risk appetite, trade,
#   fiscal news, liquidity, bid-ask spreads, nonlinear regimes, and many release
#   surprises.
# - Snapshot provenance supports reproducibility but does not by itself establish
#   redistribution or other downstream-use rights.

# %% [markdown] tags=["exercise"]
# ## Assessment
#
# Create a `Domestic 75 bp tightening` diagnostic from the 2025-06-30 row:
#
# - policy rate: +75 basis points;
# - CETES 28-day rate: +100 basis points;
# - lagged Mexican inflation and US 10-year yield: unchanged; and
# - three-month FX momentum: unchanged.
#
# 1. Report the policy-minus-US-10-year proxy and CETES-policy proxy gap.
# 2. Calculate the change versus the observed-input baseline under both the
#    evaluation fit and updated fit, in basis points of one-month log return.
# 3. Compare the larger absolute sensitivity with Ridge test MAE.
# 4. Write a three- or four-sentence decision note: state the hypothesized
#    mechanism, one alternative explanation, the benchmark decision, and one
#    limitation. Do not publish an FX target.

# %% tags=["solution"]
checkpoint_inputs = pd.DataFrame(
    [
        {
            "scenario": "Domestic 75 bp tightening",
            "policy_shock": 0.0075,
            "cetes_shock": 0.0100,
            "lagged_inflation_proxy_shock": 0.0000,
            "us_10y_shock": 0.0000,
        }
    ]
).set_index("scenario")
checkpoint_features = scenario_feature_frame(checkpoint_inputs, latest)
baseline_features = scenario_features.loc[["Observed-input baseline"]]

checkpoint_deltas = {}
for fit_name, fitted_model in diagnostic_models.items():
    baseline_prediction = fitted_model.predict(
        baseline_features[list(FEATURE_COLUMNS)]
    )[0]
    checkpoint_prediction = fitted_model.predict(
        checkpoint_features[list(FEATURE_COLUMNS)]
    )[0]
    checkpoint_deltas[fit_name] = 10_000 * (
        checkpoint_prediction - baseline_prediction
    )

np.testing.assert_allclose(
    [checkpoint_deltas["Evaluation fit"], checkpoint_deltas["Updated fit"]],
    [-1.799063, 72.680753],
    atol=1e-6,
)

checkpoint_answer = pd.DataFrame(
    [
        {"Assessment item": "As-of date", "Result": latest_date.date().isoformat()},
        {
            "Assessment item": "Policy-minus-US-10-year proxy",
            "Result": f"{100 * checkpoint_features.iloc[0]['mx_policy_minus_us_10y']:.2f}%",
        },
        {
            "Assessment item": "CETES-policy proxy gap",
            "Result": f"{100 * checkpoint_features.iloc[0]['cetes_policy_gap']:.2f}%",
        },
        {
            "Assessment item": "Evaluation-fit delta vs baseline",
            "Result": f"{checkpoint_deltas['Evaluation fit']:+.2f} bp",
        },
        {
            "Assessment item": "Updated-fit delta vs baseline",
            "Result": f"{checkpoint_deltas['Updated fit']:+.2f} bp",
        },
        {
            "Assessment item": "Ridge holdout MAE",
            "Result": f"{10_000 * ridge_mae:.2f} bp",
        },
        {
            "Assessment item": "Decision",
            "Result": "REJECTED — benchmark and stability evidence fail",
        },
    ]
)
assert checkpoint_deltas["Evaluation fit"] * checkpoint_deltas["Updated fit"] < 0
display(
    checkpoint_answer.style.hide(axis="index").set_caption(
        "Table 7. Suggested assessment calculation"
    )
)

# %% [markdown]
# **Assessment interpretation.**
#
# The policy-minus-US-10-year proxy is 4.51% and the CETES-policy gap is 0.25%.
# Relative to each fit's baseline, the same tightening diagnostic changes the
# predicted one-month log return by about -1.80 basis points under the evaluation
# fit and +72.68 basis points under the updated fit. The signs disagree, and even
# the larger move is well below the 239.41-basis-point holdout MAE. Tighter policy
# could affect relative rates and currency demand, but the fitted response is not
# a causal estimate and could instead reflect sample composition or omitted
# global-risk conditions. The model remains rejected, so no FX target is issued.

# %% [markdown]
# ## Handoff
#
# This case closes Module 3 with a governed negative result. Later modules may
# reuse the **explicit macro shocks**, their units, horizon, monitoring evidence,
# and limitations. They must not reuse these rejected fitted FX outputs.
# [Module 4](../../chapters/04-financial-statements-modeling.md) can apply the
# documented assumptions to revenue, margins, working capital, and discount
# rates; fixed-income, derivatives, and portfolio modules can independently
# validate curve, hedge, and allocation consequences.
