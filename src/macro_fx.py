"""Time-aware data preparation helpers for the Module 3 macro-to-FX case.

The committed macro panel contains latest-vintage monthly observations rather
than historical release timestamps.  These helpers therefore impose a one-month
availability lag on CPI inflation and keep forecast origins in chronological
order.  The lag is an explicit classroom approximation, not a real-time-vintage
substitute.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


REQUIRED_MACRO_COLUMNS = frozenset(
    {
        "banxico_target_rate",
        "cetes_28d",
        "usd_mxn",
        "mexico_inflation",
        "us_10y",
    }
)
FEATURE_COLUMNS = (
    "mx_policy_minus_us_10y",
    "mexico_inflation_lag1m",
    "cetes_policy_gap",
    "fx_momentum_3m",
)
TARGET_COLUMN = "next_fx_log_return"
SCENARIO_SHOCK_COLUMNS = (
    "policy_shock",
    "cetes_shock",
    "lagged_inflation_proxy_shock",
    "us_10y_shock",
)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _validate_time_index(frame: pd.DataFrame, name: str) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must use a DatetimeIndex")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError(f"{name} dates must be unique and increasing")


def prepare_macro_fx_features(
    macro: pd.DataFrame,
    *,
    inflation_release_lag_months: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create point-in-time proxies and the aligned next-month FX target.

    ``inflation_release_lag_months`` must be at least one because a CPI value
    labeled for month ``t`` is not assumed to have been public at that month's
    close.  The returned feature frame preserves every input row; ``model_data``
    contains only complete origins with a realized next-month target.
    """

    _require_columns(macro, tuple(REQUIRED_MACRO_COLUMNS), "macro")
    _validate_time_index(macro, "macro")
    if inflation_release_lag_months < 1:
        raise ValueError("inflation_release_lag_months must be at least 1")
    if macro.empty:
        raise ValueError("macro must contain observations")
    observed_months = macro.index.to_period("M")
    expected_months = pd.period_range(
        observed_months[0], observed_months[-1], freq="M"
    )
    if not observed_months.equals(expected_months):
        raise ValueError("macro must contain exactly one consecutive row per month")
    if not np.isfinite(macro[list(REQUIRED_MACRO_COLUMNS)].to_numpy()).all():
        raise ValueError("macro required columns must contain finite values")
    if (macro["usd_mxn"] <= 0).any():
        raise ValueError("usd_mxn must be strictly positive for log returns")

    features = macro.copy()
    features["mx_policy_minus_us_10y"] = (
        features["banxico_target_rate"] - features["us_10y"]
    )
    features["mexico_inflation_lag1m"] = features["mexico_inflation"].shift(
        inflation_release_lag_months
    )
    features["cetes_policy_gap"] = (
        features["cetes_28d"] - features["banxico_target_rate"]
    )
    features["fx_momentum_3m"] = np.log(
        features["usd_mxn"] / features["usd_mxn"].shift(3)
    )
    features[TARGET_COLUMN] = np.log(
        features["usd_mxn"].shift(-1) / features["usd_mxn"]
    )

    model_data = features[[*FEATURE_COLUMNS, TARGET_COLUMN]].replace(
        [np.inf, -np.inf], np.nan
    )
    model_data = model_data.dropna()
    if model_data.empty:
        raise ValueError("feature construction produced no complete model rows")
    return features, model_data


def chronological_split(
    model_data: pd.DataFrame,
    *,
    training_fraction: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ordered forecast origins without shuffling or future leakage."""

    _validate_time_index(model_data, "model_data")
    if not 0 < training_fraction < 1:
        raise ValueError("training_fraction must be between 0 and 1")
    split_position = int(len(model_data) * training_fraction)
    if split_position == 0 or split_position == len(model_data):
        raise ValueError("training_fraction must leave non-empty train and test samples")
    training = model_data.iloc[:split_position].copy()
    testing = model_data.iloc[split_position:].copy()
    if training.index.max() >= testing.index.min():
        raise ValueError("training origins must end before testing origins begin")
    return training, testing


def scenario_feature_frame(
    inputs: pd.DataFrame,
    observed: pd.Series,
) -> pd.DataFrame:
    """Map explicit rate and lagged-inflation-proxy shocks into fitted features."""

    _require_columns(inputs, SCENARIO_SHOCK_COLUMNS, "scenario inputs")
    required_observed = (
        "banxico_target_rate",
        "cetes_28d",
        "mexico_inflation_lag1m",
        "us_10y",
        "fx_momentum_3m",
    )
    missing_observed = set(required_observed) - set(observed.index)
    if missing_observed:
        raise ValueError(
            f"observed row is missing required fields: {sorted(missing_observed)}"
        )

    scenario = pd.DataFrame(index=inputs.index)
    scenario["policy_rate"] = observed["banxico_target_rate"] + inputs["policy_shock"]
    scenario["cetes_28d"] = observed["cetes_28d"] + inputs["cetes_shock"]
    scenario["mexico_inflation_lag1m"] = (
        observed["mexico_inflation_lag1m"]
        + inputs["lagged_inflation_proxy_shock"]
    )
    scenario["us_10y"] = observed["us_10y"] + inputs["us_10y_shock"]
    scenario["mx_policy_minus_us_10y"] = (
        scenario["policy_rate"] - scenario["us_10y"]
    )
    scenario["cetes_policy_gap"] = scenario["cetes_28d"] - scenario["policy_rate"]
    scenario["fx_momentum_3m"] = observed["fx_momentum_3m"]
    if not np.isfinite(scenario.to_numpy()).all():
        raise ValueError("scenario features must contain finite values")
    return scenario


__all__ = [
    "FEATURE_COLUMNS",
    "REQUIRED_MACRO_COLUMNS",
    "SCENARIO_SHOCK_COLUMNS",
    "TARGET_COLUMN",
    "chronological_split",
    "prepare_macro_fx_features",
    "scenario_feature_frame",
]
