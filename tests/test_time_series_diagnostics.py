from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.time_series_diagnostics import (
    arch_lm_report,
    arima_order_search,
    garch11_volatility_filter,
    garch_long_run_variance,
    garch_variance_half_life,
    ljung_box_report,
)


def test_arima_search_reports_warnings_without_emitting_them() -> None:
    rng = np.random.default_rng(42)
    series = pd.Series(rng.normal(size=120))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = arima_order_search(
            series,
            p_values=[0, 1],
            d_values=[0],
            q_values=[0],
        )

    assert caught == []
    assert result[["p", "d", "q"]].shape == (2, 3)
    assert {
        "aic",
        "bic",
        "stationary",
        "invertible",
        "converged",
        "warning_types",
    } <= set(result.columns)
    assert result["converged"].any()
    assert result.loc[result["converged"], "stationary"].all()
    assert result.loc[result["converged"], "invertible"].all()


def test_ljung_box_report_uses_model_degrees_of_freedom_for_residuals() -> None:
    rng = np.random.default_rng(7)
    residuals = pd.Series(rng.normal(size=200))

    result = ljung_box_report(residuals, lags=[5, 10], model_df=2)

    assert list(result.index) == [5, 10]
    assert result["lb_pvalue"].between(0, 1).all()


def test_ljung_box_report_rejects_negative_model_degrees_of_freedom() -> None:
    with np.testing.assert_raises_regex(ValueError, "model_df"):
        ljung_box_report(pd.Series([0.1, -0.1, 0.2]), model_df=-1)


def test_arch_lm_report_detects_conditional_variance_dependence() -> None:
    rng = np.random.default_rng(12)
    shocks = rng.normal(size=800)
    variance = np.empty_like(shocks)
    variance[0] = 1.0
    for index in range(1, len(shocks)):
        variance[index] = 0.1 + 0.82 * variance[index - 1] + 0.12 * shocks[index - 1] ** 2
    residuals = pd.Series(np.sqrt(variance) * shocks)

    report = arch_lm_report(residuals, lags=10)

    assert report["lags"] == 10
    assert report["model_df"] == 0
    assert report["lm_p_value"] < 0.05


def test_arch_lm_report_validates_lags_and_sample_size() -> None:
    with np.testing.assert_raises_regex(ValueError, "lags"):
        arch_lm_report(pd.Series([0.1, -0.1, 0.2]), lags=0)
    with np.testing.assert_raises_regex(ValueError, "more observations"):
        arch_lm_report(pd.Series([0.1, -0.1, 0.2]), lags=2)


def test_garch_filter_uses_parameter_implied_initial_variance() -> None:
    returns = pd.Series(
        [0.01, -0.02, 0.005],
        index=pd.date_range("2026-01-01", periods=3),
    )
    omega = 0.000002
    alpha = 0.08
    beta = 0.90

    volatility = garch11_volatility_filter(
        returns,
        omega=omega,
        alpha=alpha,
        beta=beta,
    )

    expected_initial_variance = garch_long_run_variance(omega, alpha, beta)
    assert np.isclose(volatility.iloc[0] ** 2, expected_initial_variance)
    assert np.isclose(
        volatility.iloc[1] ** 2,
        omega + alpha * returns.iloc[0] ** 2 + beta * expected_initial_variance,
    )
    assert volatility.index.equals(returns.index)


def test_garch_helpers_reject_invalid_parameters() -> None:
    returns = pd.Series([0.01, -0.02])
    invalid_parameters = [
        {"omega": 0.0, "alpha": 0.08, "beta": 0.90},
        {"omega": 0.000002, "alpha": -0.01, "beta": 0.90},
        {"omega": 0.000002, "alpha": 0.08, "beta": -0.10},
        {"omega": 0.000002, "alpha": 0.20, "beta": 0.80},
    ]
    for parameters in invalid_parameters:
        with np.testing.assert_raises(ValueError):
            garch11_volatility_filter(returns, **parameters)


def test_garch_variance_half_life_matches_closed_form() -> None:
    assert np.isclose(garch_variance_half_life(0.98), np.log(0.5) / np.log(0.98))
    with np.testing.assert_raises_regex(ValueError, "strictly between"):
        garch_variance_half_life(1.0)
