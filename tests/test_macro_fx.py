from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.macro_fx import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    chronological_split,
    prepare_macro_fx_features,
    scenario_feature_frame,
)


def _macro_frame(periods: int = 10) -> pd.DataFrame:
    index = pd.date_range("2024-01-31", periods=periods, freq="ME")
    return pd.DataFrame(
        {
            "banxico_target_rate": np.linspace(0.11, 0.08, periods),
            "cetes_28d": np.linspace(0.108, 0.079, periods),
            "usd_mxn": np.linspace(17.0, 18.8, periods),
            "mexico_inflation": np.linspace(0.048, 0.041, periods),
            "us_10y": np.linspace(0.038, 0.044, periods),
        },
        index=index,
    )


def test_prepare_macro_fx_features_lags_inflation_and_aligns_target() -> None:
    macro = _macro_frame()
    original = macro.copy(deep=True)

    features, model_data = prepare_macro_fx_features(macro)

    assert features.loc[macro.index[1], "mexico_inflation_lag1m"] == pytest.approx(
        macro.loc[macro.index[0], "mexico_inflation"]
    )
    assert features.loc[macro.index[3], "fx_momentum_3m"] == pytest.approx(
        np.log(macro.loc[macro.index[3], "usd_mxn"] / macro.loc[macro.index[0], "usd_mxn"])
    )
    assert features.loc[macro.index[4], TARGET_COLUMN] == pytest.approx(
        np.log(macro.loc[macro.index[5], "usd_mxn"] / macro.loc[macro.index[4], "usd_mxn"])
    )
    assert model_data.index.min() == macro.index[3]
    assert model_data.index.max() == macro.index[-2]
    assert list(model_data.columns) == [*FEATURE_COLUMNS, TARGET_COLUMN]
    pd.testing.assert_frame_equal(macro, original)


def test_prepare_macro_fx_features_rejects_same_month_inflation() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        prepare_macro_fx_features(
            _macro_frame(),
            inflation_release_lag_months=0,
        )


def test_prepare_macro_fx_features_requires_one_row_per_consecutive_month() -> None:
    daily = _macro_frame()
    daily.index = pd.date_range("2024-01-01", periods=len(daily), freq="D")

    with pytest.raises(ValueError, match="one consecutive row per month"):
        prepare_macro_fx_features(daily)


def test_chronological_split_preserves_order_and_non_overlap() -> None:
    _, model_data = prepare_macro_fx_features(_macro_frame(periods=12))

    training, testing = chronological_split(model_data, training_fraction=0.75)

    assert len(training) == int(len(model_data) * 0.75)
    assert training.index.max() < testing.index.min()
    assert training.index.is_monotonic_increasing
    assert testing.index.is_monotonic_increasing


def test_scenario_feature_frame_maps_shocks_without_changing_momentum() -> None:
    macro = _macro_frame()
    features, _ = prepare_macro_fx_features(macro)
    observed = features.iloc[-1]
    inputs = pd.DataFrame(
        {
            "policy_shock": [0.0075],
            "cetes_shock": [0.0100],
            "lagged_inflation_proxy_shock": [0.0025],
            "us_10y_shock": [-0.0010],
        },
        index=["Diagnostic"],
    )

    scenario = scenario_feature_frame(inputs, observed)

    assert scenario.loc["Diagnostic", "policy_rate"] == pytest.approx(
        observed["banxico_target_rate"] + 0.0075
    )
    assert scenario.loc["Diagnostic", "mx_policy_minus_us_10y"] == pytest.approx(
        observed["banxico_target_rate"]
        + 0.0075
        - (observed["us_10y"] - 0.0010)
    )
    assert scenario.loc["Diagnostic", "cetes_policy_gap"] == pytest.approx(
        observed["cetes_28d"]
        + 0.0100
        - (observed["banxico_target_rate"] + 0.0075)
    )
    assert scenario.loc["Diagnostic", "mexico_inflation_lag1m"] == pytest.approx(
        observed["mexico_inflation_lag1m"] + 0.0025
    )
    assert scenario.loc["Diagnostic", "fx_momentum_3m"] == pytest.approx(
        observed["fx_momentum_3m"]
    )


def test_frozen_snapshot_reproduces_published_rejection_metrics() -> None:
    root = Path(__file__).resolve().parents[1]
    macro = (
        pd.read_csv(
            root / "data/snapshots/official_macro_panel.csv",
            parse_dates=["date"],
        )
        .set_index("date")
        .sort_index()
    )
    _, model_data = prepare_macro_fx_features(macro)
    training, testing = chronological_split(model_data)
    model = Pipeline(
        [("scale", StandardScaler()), ("ridge", Ridge(alpha=4.0))]
    )
    model.fit(training[list(FEATURE_COLUMNS)], training[TARGET_COLUMN])

    actual = testing[TARGET_COLUMN]
    ridge_mae = mean_absolute_error(
        actual,
        model.predict(testing[list(FEATURE_COLUMNS)]),
    )
    training_mean_mae = mean_absolute_error(
        actual,
        np.repeat(training[TARGET_COLUMN].mean(), len(testing)),
    )
    zero_change_mae = mean_absolute_error(actual, np.zeros(len(testing)))

    assert (len(model_data), len(training), len(testing)) == (74, 55, 19)
    assert training.index.max() == pd.Timestamp("2023-10-31")
    assert testing.index.min() == pd.Timestamp("2023-11-30")
    assert testing.index.max() == pd.Timestamp("2025-05-31")
    assert ridge_mae == pytest.approx(0.0239407825, abs=1e-10)
    assert training_mean_mae == pytest.approx(0.0217597546, abs=1e-10)
    assert zero_change_mae == pytest.approx(0.0218968136, abs=1e-10)
    assert ridge_mae > training_mean_mae
    assert ridge_mae > zero_change_mae
