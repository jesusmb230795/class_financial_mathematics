from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.derivatives import (
    arithmetic_asian_call_control_variate,
    bgk_adjusted_barrier,
    black_scholes_greeks,
    black_scholes_price,
    call_price_arbitrage_diagnostics,
    crr_binomial_option_price,
    forward_price,
    forward_value,
    implied_volatility,
    leisen_reimer_american_put,
    monte_carlo_european_option_price,
    option_price_bounds,
    par_swap_rate,
    put_call_parity_gap,
    simulate_heston_paths,
)


def test_forward_and_swap_helpers_respect_continuous_carry_contracts() -> None:
    spot = 20.0
    rate = 0.08
    dividend_yield = 0.02
    maturity = 0.75
    strike = forward_price(spot, rate, maturity, dividend_yield)

    assert strike == pytest.approx(spot * np.exp((rate - dividend_yield) * maturity))
    assert forward_value(spot, strike, rate, maturity, dividend_yield) == pytest.approx(0.0)
    assert forward_value(
        spot, strike, rate, maturity, dividend_yield, position="short"
    ) == pytest.approx(0.0)

    discount_factors = np.array([0.97, 0.94, 0.91])
    accrual_factors = np.array([0.5, 0.5, 0.5])
    expected_rate = (1 - discount_factors[-1]) / np.sum(accrual_factors * discount_factors)
    assert par_swap_rate(discount_factors, accrual_factors) == pytest.approx(expected_rate)


def test_black_scholes_prices_greeks_and_parity_are_internally_consistent() -> None:
    inputs = {
        "spot": 100.0,
        "strike": 105.0,
        "rate": 0.05,
        "volatility": 0.22,
        "maturity": 0.75,
        "dividend_yield": 0.015,
    }
    call = black_scholes_price(**inputs, option_type="call")
    put = black_scholes_price(**inputs, option_type="put")
    greeks = black_scholes_greeks(**inputs, option_type="call")

    assert put_call_parity_gap(
        call,
        put,
        inputs["spot"],
        inputs["strike"],
        inputs["rate"],
        inputs["maturity"],
        inputs["dividend_yield"],
    ) == pytest.approx(0.0, abs=1e-12)
    assert set(greeks.index) == {
        "delta",
        "gamma",
        "vega_per_1pct",
        "theta_per_day",
        "rho_per_1pct",
    }
    assert 0 < greeks["delta"] < 1
    assert greeks["gamma"] > 0


def test_implied_volatility_obeys_bounds_and_recovers_the_generating_value() -> None:
    spot, strike, rate, maturity, dividend_yield = 100.0, 95.0, 0.04, 0.5, 0.01
    generating_volatility = 0.27
    price = black_scholes_price(
        spot,
        strike,
        rate,
        generating_volatility,
        maturity,
        "call",
        dividend_yield,
    )
    lower, upper = option_price_bounds(
        spot,
        strike,
        rate,
        maturity,
        "call",
        dividend_yield,
    )

    assert lower <= price < upper
    assert implied_volatility(
        price,
        spot,
        strike,
        rate,
        maturity,
        "call",
        dividend_yield,
    ) == pytest.approx(generating_volatility)
    with pytest.raises(ValueError, match="lower bound"):
        implied_volatility(
            lower - 0.01,
            spot,
            strike,
            rate,
            maturity,
            "call",
            dividend_yield,
        )


def test_call_chain_diagnostics_detect_convexity_and_monotonicity_failures() -> None:
    strikes = np.array([90.0, 100.0, 110.0, 120.0])
    valid_prices = np.array([14.0, 8.0, 4.0, 2.0])
    valid = call_price_arbitrage_diagnostics(
        strikes,
        valid_prices,
        spot=100.0,
        rate=0.03,
        maturity=0.5,
    )

    assert bool(valid["all_checks_pass"])

    invalid_prices = np.array([14.0, 8.0, 8.5, 2.0])
    invalid = call_price_arbitrage_diagnostics(
        strikes,
        invalid_prices,
        spot=100.0,
        rate=0.03,
        maturity=0.5,
    )
    assert not bool(invalid["non_increasing_in_strike"])
    assert not bool(invalid["all_checks_pass"])


def test_tree_and_monte_carlo_prices_converge_to_european_benchmark() -> None:
    inputs = {
        "spot": 100.0,
        "strike": 100.0,
        "rate": 0.04,
        "volatility": 0.20,
        "maturity": 1.0,
    }
    benchmark = black_scholes_price(**inputs, option_type="call")
    tree = crr_binomial_option_price(**inputs, option_type="call", steps=501)
    monte_carlo = monte_carlo_european_option_price(
        **inputs,
        option_type="call",
        simulations=200_000,
        seed=2026,
    )

    assert tree == pytest.approx(benchmark, abs=0.02)
    assert monte_carlo.price == pytest.approx(benchmark, abs=0.08)
    assert monte_carlo.ci_95_lower < monte_carlo.price < monte_carlo.ci_95_upper
    assert monte_carlo.standard_error > 0


def test_american_and_path_dependent_helpers_report_model_features() -> None:
    european_put = crr_binomial_option_price(
        95.0,
        100.0,
        0.05,
        0.25,
        1.0,
        steps=501,
        option_type="put",
    )
    american_put = leisen_reimer_american_put(
        95.0,
        100.0,
        0.05,
        0.25,
        1.0,
        steps=501,
    )
    asian = arithmetic_asian_call_control_variate(
        100.0,
        100.0,
        0.03,
        0.20,
        1.0,
        observations=24,
        paths=8_000,
        seed=2026,
    )

    assert american_put >= european_put
    assert asian.control_variate_standard_error < asian.naive_standard_error
    assert bgk_adjusted_barrier(90.0, 0.20, 1 / 252, "down") < 90.0
    assert bgk_adjusted_barrier(110.0, 0.20, 1 / 252, "up") > 110.0


def test_asian_control_variate_falls_back_when_control_payoff_is_constant() -> None:
    result = arithmetic_asian_call_control_variate(
        100.0,
        1_000.0,
        0.03,
        0.20,
        1.0,
        observations=24,
        paths=100,
        seed=1,
    )

    assert result.beta == 0.0
    assert np.isfinite(result.control_variate_price)
    assert np.isfinite(result.control_variate_standard_error)
    assert result.control_variate_price == pytest.approx(result.naive_price)
    assert result.control_variate_standard_error == pytest.approx(
        result.naive_standard_error
    )


def test_heston_full_truncation_paths_are_finite_non_negative_and_reproducible() -> None:
    kwargs = {
        "spot": 100.0,
        "variance0": 0.04,
        "rate": 0.03,
        "maturity": 1.0,
        "kappa": 1.2,
        "theta": 0.04,
        "vol_of_vol": 0.8,
        "rho": -0.65,
        "steps": 50,
        "paths": 200,
        "seed": 17,
    }

    first_spots, first_variances = simulate_heston_paths(**kwargs)
    second_spots, second_variances = simulate_heston_paths(**kwargs)

    assert first_spots.shape == (51, 200)
    assert first_variances.shape == (51, 200)
    assert np.isfinite(first_spots).all() and (first_spots > 0).all()
    assert np.isfinite(first_variances).all() and (first_variances >= 0).all()
    np.testing.assert_array_equal(first_spots, second_spots)
    np.testing.assert_array_equal(first_variances, second_variances)


def test_heston_full_truncation_preserves_latent_variance_recovery() -> None:
    """A negative latent state must recover from itself, not from a zero projection."""
    parameters = {
        "spot": 100.0,
        "variance0": 0.0001,
        "rate": 0.0,
        "maturity": 1.0,
        "kappa": 4.0,
        "theta": 0.04,
        "vol_of_vol": 2.0,
        "rho": 0.0,
        "steps": 4,
        "paths": 1,
        "seed": 3,
    }

    rng = np.random.default_rng(parameters["seed"])
    dt = parameters["maturity"] / parameters["steps"]
    latent_variance = parameters["variance0"]
    displayed_reference = [latent_variance]
    latent_reference = [latent_variance]
    for _ in range(parameters["steps"]):
        z1 = rng.normal(size=1)
        z2 = (
            parameters["rho"] * z1
            + np.sqrt(1 - parameters["rho"] ** 2) * rng.normal(size=1)
        )
        variance_positive = max(latent_variance, 0.0)
        latent_variance = float(
            latent_variance
            + parameters["kappa"]
            * (parameters["theta"] - variance_positive)
            * dt
            + parameters["vol_of_vol"]
            * np.sqrt(variance_positive)
            * np.sqrt(dt)
            * z2[0]
        )
        latent_reference.append(latent_variance)
        displayed_reference.append(max(latent_variance, 0.0))

    _, simulated_variances = simulate_heston_paths(**parameters)

    assert latent_reference[2] < 0 < latent_reference[3]
    assert latent_reference[3] != pytest.approx(parameters["kappa"] * parameters["theta"] * dt)
    np.testing.assert_allclose(simulated_variances[:, 0], displayed_reference)


def test_derivative_helpers_reject_invalid_contract_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        forward_price(0.0, 0.03, 1.0)
    with pytest.raises(ValueError, match="option_type"):
        black_scholes_price(100, 100, 0.03, 0.2, 1.0, option_type="straddle")
    with pytest.raises(ValueError, match="strictly increasing"):
        call_price_arbitrage_diagnostics(
            np.array([100.0, 100.0, 110.0]),
            np.array([8.0, 7.0, 3.0]),
            100.0,
            0.03,
            1.0,
        )
