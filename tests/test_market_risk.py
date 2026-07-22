from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, t

from src.market_risk import (
    basel_traffic_light,
    christoffersen_independence_test,
    cornish_fisher_moment_report,
    cornish_fisher_var,
    ewma_next_volatility,
    ewma_volatility,
    exception_series,
    expected_shortfall,
    gaussian_monte_carlo_var,
    gaussian_var,
    historical_var,
    kupiec_pof_test,
    parametric_var,
    rebalanced_portfolio_returns,
    simple_loss_from_log_return,
    standardized_student_t_quantile,
    stress_scenario_loss,
    volatility_weighted_historical_var,
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


def test_expected_shortfall_uses_exact_fractional_empirical_tail_mass() -> None:
    returns = pd.Series([-0.30, -0.20, 0.10])

    # alpha*n = 1.5: one complete worst observation plus half of the next.
    assert expected_shortfall(returns, alpha=0.5) == pytest.approx(-(-0.30 - 0.5 * 0.20) / 1.5)


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


def test_log_return_threshold_converts_to_exact_simple_loss() -> None:
    log_return_threshold = -0.025
    expected_simple_loss = 1 - np.exp(log_return_threshold)

    assert simple_loss_from_log_return(log_return_threshold) == pytest.approx(expected_simple_loss)
    assert simple_loss_from_log_return(0.01) == 0.0
    with pytest.raises(ValueError, match="finite"):
        simple_loss_from_log_return(np.inf)


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


def test_ewma_next_forecast_incorporates_the_last_observation() -> None:
    returns = pd.Series([0.01, -0.02, 0.08])
    lambda_ = 0.94
    initial_variance = 0.0004
    filtered = ewma_volatility(
        returns,
        lambda_=lambda_,
        initial_variance=initial_variance,
    )
    expected_next = np.sqrt(
        lambda_ * filtered.iloc[-1] ** 2 + (1 - lambda_) * returns.iloc[-1] ** 2
    )

    assert ewma_next_volatility(
        returns,
        lambda_=lambda_,
        initial_variance=initial_variance,
    ) == pytest.approx(expected_next)
    assert volatility_weighted_historical_var(returns, alpha=0.34, lambda_=lambda_) >= 0


def test_rebalanced_portfolio_returns_requires_exact_labels_and_preserves_provenance() -> None:
    asset_returns = pd.DataFrame(
        {"A": [0.10, -0.05], "B": [-0.02, 0.04]},
        index=pd.date_range("2026-01-01", periods=2),
    )
    asset_returns.attrs = {"source": "versioned fixture"}
    weights = pd.Series({"A": 0.6, "B": 0.4})

    portfolio = rebalanced_portfolio_returns(asset_returns, weights)

    pd.testing.assert_series_equal(
        portfolio,
        pd.Series(
            [0.052, -0.014],
            index=asset_returns.index,
            name="portfolio_simple_return",
        ),
    )
    assert portfolio.attrs["source"] == "versioned fixture"
    assert "rebalanced" in portfolio.attrs["portfolio_rule"]

    with pytest.raises(ValueError, match="match the return columns exactly"):
        rebalanced_portfolio_returns(asset_returns, pd.Series({"A": 1.0}))
    with pytest.raises(ValueError, match="sum to 1"):
        rebalanced_portfolio_returns(asset_returns, pd.Series({"A": 0.4, "B": 0.4}))


def test_gaussian_monte_carlo_var_is_reproducible_and_close_to_closed_form() -> None:
    rng = np.random.default_rng(2026)
    returns = pd.Series(rng.normal(0.0004, 0.012, size=1_000))

    first = gaussian_monte_carlo_var(
        returns,
        confidence=0.99,
        simulations=200_000,
        seed=7,
    )
    second = gaussian_monte_carlo_var(
        returns,
        confidence=0.99,
        simulations=200_000,
        seed=7,
    )

    assert first == second
    assert first == pytest.approx(gaussian_var(returns, alpha=0.01), abs=5e-4)
    with pytest.raises(ValueError, match="confidence"):
        gaussian_monte_carlo_var(returns, confidence=1.0)


@pytest.mark.parametrize("test_function", [kupiec_pof_test, christoffersen_independence_test])
def test_backtests_reject_non_binary_exception_inputs(test_function) -> None:
    with pytest.raises(ValueError, match="binary"):
        test_function([0, 2, 0])


def test_traffic_light_and_stress_scenario_validate_their_contracts() -> None:
    with pytest.raises(ValueError, match="integer"):
        basel_traffic_light(True)

    assert basel_traffic_light(4)["zone"] == "green"
    assert basel_traffic_light(5)["zone"] == "yellow"
    assert basel_traffic_light(5)["multiplier"] == pytest.approx(3.40)
    assert basel_traffic_light(9)["multiplier"] == pytest.approx(3.85)
    assert basel_traffic_light(10)["zone"] == "red"

    weights = pd.Series({"A": 0.6, "B": 0.4})
    shocks = pd.Series({"A": -0.10, "B": -0.20})
    losses = stress_scenario_loss(weights, shocks)
    assert losses.loc["portfolio_total"] == pytest.approx(0.14)

    with pytest.raises(ValueError, match="identical labels"):
        stress_scenario_loss(weights, pd.Series({"A": -0.10, "C": -0.20}))
