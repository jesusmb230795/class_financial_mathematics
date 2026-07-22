from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, t

from src.market_risk import (
    cornish_fisher_moment_report,
    cornish_fisher_var,
    ewma_volatility,
    exception_series,
    expected_shortfall,
    gaussian_var,
    historical_var,
    parametric_var,
    standardized_student_t_quantile,
)
from src.time_series_diagnostics import parametric_var as time_series_parametric_var


def test_var_and_expected_shortfall_use_non_negative_loss_convention() -> None:
    returns = pd.Series([-0.05, -0.03, -0.01, 0.02, 0.04])
    losses = -returns
    alpha = 0.2

    assert historical_var(returns, alpha=alpha) == pytest.approx(max(0.0, -returns.quantile(alpha)))
    assert historical_var(returns, alpha=alpha) == pytest.approx(
        max(0.0, losses.quantile(1 - alpha))
    )
    assert expected_shortfall(returns, alpha=alpha) == pytest.approx(0.05)
    assert gaussian_var(returns, alpha=0.05) >= 0

    all_gains = pd.Series([0.01, 0.02, 0.03, 0.04])
    assert historical_var(all_gains, alpha=0.05) == 0.0
    assert expected_shortfall(all_gains, alpha=0.05) == 0.0


def test_m2_and_market_risk_parametric_var_have_exact_parity() -> None:
    mean = 0.03
    volatility = 1.25
    quantile = norm.ppf(0.01)
    expected = -(mean + volatility * quantile)

    assert parametric_var(mean, volatility, quantile) == pytest.approx(expected)
    assert time_series_parametric_var(mean, volatility, quantile) == pytest.approx(expected)


def test_parametric_var_rejects_invalid_scale_and_never_reports_a_gain_as_var() -> None:
    with pytest.raises(ValueError, match="volatility"):
        parametric_var(mean=0.0, volatility=-0.1, quantile=-2.0)

    assert parametric_var(mean=2.0, volatility=0.1, quantile=-1.0) == 0.0


def test_standardized_student_t_quantile_matches_unit_variance_parameterization() -> None:
    alpha = 0.01
    nu = 7.5
    expected = t.ppf(alpha, df=nu) * np.sqrt((nu - 2.0) / nu)

    assert standardized_student_t_quantile(alpha, nu) == pytest.approx(expected)
    assert abs(standardized_student_t_quantile(alpha, nu)) < abs(t.ppf(alpha, df=nu))

    with pytest.raises(ValueError, match="greater than 2"):
        standardized_student_t_quantile(alpha, 2.0)


def test_cornish_fisher_guardrail_is_reported_and_enforced() -> None:
    ordinary = pd.Series(np.linspace(-1.0, 1.0, 101))
    report = cornish_fisher_moment_report(ordinary)
    assert bool(report["cornish_fisher_available"])
    assert cornish_fisher_var(ordinary, alpha=0.05) >= 0

    extreme = pd.Series([0.0] * 99 + [-100.0])
    extreme_report = cornish_fisher_moment_report(extreme)
    assert not bool(extreme_report["cornish_fisher_available"])
    with pytest.raises(ValueError, match="outside the course guardrail"):
        cornish_fisher_var(extreme, alpha=0.05)


def test_exception_series_rejects_negative_var_forecasts() -> None:
    returns = pd.Series([-0.03, 0.01])

    with pytest.raises(ValueError, match="non-negative"):
        exception_series(returns, -0.02)
    with pytest.raises(ValueError, match="non-negative"):
        exception_series(returns, pd.Series([0.02, -0.01]))
    with pytest.raises(ValueError, match="finite"):
        exception_series(returns, pd.Series([0.02, np.inf]))


def test_ewma_can_use_a_declared_initial_variance_without_look_ahead() -> None:
    returns = pd.Series([0.01, 0.50, -0.40])
    initial_variance = 0.0004

    volatility = ewma_volatility(
        returns,
        lambda_=0.94,
        initial_variance=initial_variance,
    )

    assert volatility.iloc[0] == pytest.approx(np.sqrt(initial_variance))
    assert volatility.iloc[1] ** 2 == pytest.approx(
        0.94 * initial_variance + 0.06 * returns.iloc[0] ** 2
    )
    with pytest.raises(ValueError, match="initial_variance"):
        ewma_volatility(returns, initial_variance=0.0)
