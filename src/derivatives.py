"""Derivative pricing helpers for classroom notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


def forward_price(spot: float, rate: float, maturity: float, dividend_yield: float = 0.0) -> float:
    """No-arbitrage forward price under continuous carry."""
    return float(spot * np.exp((rate - dividend_yield) * maturity))


def forward_value(
    spot: float,
    strike: float,
    rate: float,
    maturity: float,
    dividend_yield: float = 0.0,
    position: str = "long",
) -> float:
    """Mark-to-market value of an existing forward contract."""
    value = spot * np.exp(-dividend_yield * maturity) - strike * np.exp(-rate * maturity)
    if position == "long":
        return float(value)
    if position == "short":
        return float(-value)
    raise ValueError("position must be 'long' or 'short'")


def par_swap_rate(discount_factors: np.ndarray, accrual_factors: np.ndarray) -> float:
    """Fixed par swap rate implied by a discount curve."""
    discount_factors = np.asarray(discount_factors, dtype=float)
    accrual_factors = np.asarray(accrual_factors, dtype=float)
    annuity = np.sum(accrual_factors * discount_factors)
    return float((1 - discount_factors[-1]) / annuity)


def option_payoff(terminal_price: np.ndarray, strike: float, option_type: str = "call") -> np.ndarray:
    """European call or put payoff."""
    terminal_price = np.asarray(terminal_price, dtype=float)
    if option_type == "call":
        return np.maximum(terminal_price - strike, 0.0)
    if option_type == "put":
        return np.maximum(strike - terminal_price, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def black_scholes_inputs(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    """Return Black-Scholes d1 and d2."""
    if spot <= 0 or strike <= 0 or volatility <= 0 or maturity <= 0:
        raise ValueError("spot, strike, volatility, and maturity must be positive")
    d1 = (
        np.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility**2) * maturity
    ) / (volatility * np.sqrt(maturity))
    d2 = d1 - volatility * np.sqrt(maturity)
    return float(d1), float(d2)


def black_scholes_price(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
) -> float:
    """Black-Scholes-Merton European option price with continuous dividend yield."""
    d1, d2 = black_scholes_inputs(spot, strike, rate, volatility, maturity, dividend_yield)
    discounted_spot = spot * np.exp(-dividend_yield * maturity)
    discounted_strike = strike * np.exp(-rate * maturity)
    if option_type == "call":
        return float(discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2))
    if option_type == "put":
        return float(discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1))
    raise ValueError("option_type must be 'call' or 'put'")


def black_scholes_greeks(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
) -> pd.Series:
    """Core Black-Scholes-Merton Greeks."""
    d1, d2 = black_scholes_inputs(spot, strike, rate, volatility, maturity, dividend_yield)
    exp_q = np.exp(-dividend_yield * maturity)
    exp_r = np.exp(-rate * maturity)
    gamma = exp_q * norm.pdf(d1) / (spot * volatility * np.sqrt(maturity))
    vega = spot * exp_q * norm.pdf(d1) * np.sqrt(maturity)

    if option_type == "call":
        delta = exp_q * norm.cdf(d1)
        theta = (
            -spot * exp_q * norm.pdf(d1) * volatility / (2 * np.sqrt(maturity))
            - rate * strike * exp_r * norm.cdf(d2)
            + dividend_yield * spot * exp_q * norm.cdf(d1)
        )
        rho = strike * maturity * exp_r * norm.cdf(d2)
    elif option_type == "put":
        delta = exp_q * (norm.cdf(d1) - 1)
        theta = (
            -spot * exp_q * norm.pdf(d1) * volatility / (2 * np.sqrt(maturity))
            + rate * strike * exp_r * norm.cdf(-d2)
            - dividend_yield * spot * exp_q * norm.cdf(-d1)
        )
        rho = -strike * maturity * exp_r * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return pd.Series(
        {
            "delta": float(delta),
            "gamma": float(gamma),
            "vega_per_1pct": float(vega / 100),
            "theta_per_day": float(theta / 365),
            "rho_per_1pct": float(rho / 100),
        }
    )


def put_call_parity_gap(
    call_price: float,
    put_price: float,
    spot: float,
    strike: float,
    rate: float,
    maturity: float,
    dividend_yield: float = 0.0,
) -> float:
    """Return the put-call parity gap; zero means internally consistent inputs."""
    return float(
        call_price
        + strike * np.exp(-rate * maturity)
        - put_price
        - spot * np.exp(-dividend_yield * maturity)
    )


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    rate: float,
    maturity: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
    lower: float = 1e-6,
    upper: float = 5.0,
) -> float:
    """Solve Black-Scholes implied volatility by robust bracketing."""
    intrinsic_floor = max(
        0.0,
        spot * np.exp(-dividend_yield * maturity) - strike * np.exp(-rate * maturity),
    )
    if option_type == "put":
        intrinsic_floor = max(
            0.0,
            strike * np.exp(-rate * maturity) - spot * np.exp(-dividend_yield * maturity),
        )
    if market_price < intrinsic_floor - 1e-10:
        raise ValueError("market_price violates the no-arbitrage lower bound")

    def objective(volatility: float) -> float:
        return (
            black_scholes_price(
                spot,
                strike,
                rate,
                volatility,
                maturity,
                option_type,
                dividend_yield,
            )
            - market_price
        )

    return float(brentq(objective, lower, upper))


def crr_binomial_option_price(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    steps: int = 100,
    option_type: str = "call",
    dividend_yield: float = 0.0,
    american: bool = False,
) -> float:
    """Cox-Ross-Rubinstein binomial option price."""
    dt = maturity / steps
    u = np.exp(volatility * np.sqrt(dt))
    d = 1 / u
    growth = np.exp((rate - dividend_yield) * dt)
    q = (growth - d) / (u - d)
    if not 0 <= q <= 1:
        raise ValueError("risk-neutral probability outside [0, 1]; check inputs")
    discount = np.exp(-rate * dt)
    terminal = spot * u ** np.arange(steps, -1, -1) * d ** np.arange(0, steps + 1)
    values = option_payoff(terminal, strike, option_type)

    for step in range(steps - 1, -1, -1):
        values = discount * (q * values[:-1] + (1 - q) * values[1:])
        if american:
            nodes = spot * u ** np.arange(step, -1, -1) * d ** np.arange(0, step + 1)
            values = np.maximum(values, option_payoff(nodes, strike, option_type))
    return float(values[0])


def _peizer_pratt_inversion(z: float, steps: int) -> float:
    """Peizer-Pratt inversion used by the Leisen-Reimer tree."""
    adjusted_steps = steps + 1 / 3 + 0.1 / (steps + 1)
    exponent = -((z / adjusted_steps) ** 2) * (steps + 1 / 6)
    return float(0.5 + np.sign(z) * 0.5 * np.sqrt(1 - np.exp(exponent)))


def leisen_reimer_american_put(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    steps: int = 101,
    dividend_yield: float = 0.0,
) -> float:
    """Leisen-Reimer American put price with odd-step adjustment."""
    if steps % 2 == 0:
        steps += 1
    dt = maturity / steps
    d1, d2 = black_scholes_inputs(spot, strike, rate, volatility, maturity, dividend_yield)
    p_d1 = _peizer_pratt_inversion(d1, steps)
    p_d2 = _peizer_pratt_inversion(d2, steps)
    p = p_d2
    u = np.exp((rate - dividend_yield) * dt) * p_d1 / p_d2
    d = np.exp((rate - dividend_yield) * dt) * (1 - p_d1) / (1 - p_d2)
    discount = np.exp(-rate * dt)

    up_moves = np.arange(steps, -1, -1)
    down_moves = np.arange(0, steps + 1)
    nodes = spot * u**up_moves * d**down_moves
    values = np.maximum(strike - nodes, 0.0)

    for step in range(steps - 1, -1, -1):
        nodes = nodes[:-1] / u
        continuation = discount * (p * values[:-1] + (1 - p) * values[1:])
        exercise = np.maximum(strike - nodes, 0.0)
        values = np.maximum(continuation, exercise)
    return float(values[0])


def simulate_gbm_paths(
    spot: float,
    rate: float,
    volatility: float,
    maturity: float,
    steps: int = 252,
    paths: int = 10_000,
    dividend_yield: float = 0.0,
    seed: int = 2026,
) -> np.ndarray:
    """Simulate risk-neutral GBM paths."""
    rng = np.random.default_rng(seed)
    dt = maturity / steps
    shocks = rng.normal(size=(steps, paths))
    increments = (rate - dividend_yield - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shocks
    log_paths = np.vstack([np.zeros(paths), np.cumsum(increments, axis=0)])
    return spot * np.exp(log_paths)


def geometric_asian_call_price_discrete(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    observations: int,
    dividend_yield: float = 0.0,
) -> float:
    """Closed-form price of a discretely monitored geometric Asian call."""
    observation_times = np.linspace(maturity / observations, maturity, observations)
    mean_log = np.log(spot) + (rate - dividend_yield - 0.5 * volatility**2) * observation_times.mean()
    covariance_sum = np.minimum.outer(observation_times, observation_times).sum()
    variance_log = volatility**2 * covariance_sum / observations**2
    stdev_log = np.sqrt(variance_log)
    d1 = (mean_log - np.log(strike) + variance_log) / stdev_log
    d2 = d1 - stdev_log
    expectation_term = np.exp(mean_log + 0.5 * variance_log) * norm.cdf(d1)
    strike_term = strike * norm.cdf(d2)
    return float(np.exp(-rate * maturity) * (expectation_term - strike_term))


@dataclass(frozen=True)
class AsianControlVariateResult:
    """Result bundle for arithmetic Asian call control-variate pricing."""

    naive_price: float
    naive_standard_error: float
    control_variate_price: float
    control_variate_standard_error: float
    geometric_price: float
    beta: float


def arithmetic_asian_call_control_variate(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    observations: int = 52,
    paths: int = 20_000,
    dividend_yield: float = 0.0,
    seed: int = 2026,
) -> AsianControlVariateResult:
    """Price an arithmetic Asian call using a geometric Asian control variate."""
    simulated = simulate_gbm_paths(
        spot,
        rate,
        volatility,
        maturity,
        steps=observations,
        paths=paths,
        dividend_yield=dividend_yield,
        seed=seed,
    )[1:]
    arithmetic_average = simulated.mean(axis=0)
    geometric_average = np.exp(np.log(simulated).mean(axis=0))
    discount = np.exp(-rate * maturity)
    arithmetic_payoffs = discount * np.maximum(arithmetic_average - strike, 0.0)
    geometric_payoffs = discount * np.maximum(geometric_average - strike, 0.0)
    geometric_price = geometric_asian_call_price_discrete(
        spot,
        strike,
        rate,
        volatility,
        maturity,
        observations,
        dividend_yield,
    )
    covariance = np.cov(arithmetic_payoffs, geometric_payoffs, ddof=1)
    beta = covariance[0, 1] / covariance[1, 1]
    adjusted = arithmetic_payoffs - beta * (geometric_payoffs - geometric_price)
    return AsianControlVariateResult(
        naive_price=float(arithmetic_payoffs.mean()),
        naive_standard_error=float(arithmetic_payoffs.std(ddof=1) / np.sqrt(paths)),
        control_variate_price=float(adjusted.mean()),
        control_variate_standard_error=float(adjusted.std(ddof=1) / np.sqrt(paths)),
        geometric_price=float(geometric_price),
        beta=float(beta),
    )


def bgk_adjusted_barrier(
    barrier: float,
    volatility: float,
    monitoring_interval: float,
    barrier_type: str = "down",
    beta: float = 0.5826,
) -> float:
    """Broadie-Glasserman-Kou continuity-corrected barrier level."""
    if barrier_type == "down":
        sign = -1
    elif barrier_type == "up":
        sign = 1
    else:
        raise ValueError("barrier_type must be 'down' or 'up'")
    return float(barrier * np.exp(sign * beta * volatility * np.sqrt(monitoring_interval)))


def simulate_heston_paths(
    spot: float,
    variance0: float,
    rate: float,
    maturity: float,
    kappa: float,
    theta: float,
    vol_of_vol: float,
    rho: float,
    steps: int = 252,
    paths: int = 5_000,
    dividend_yield: float = 0.0,
    seed: int = 2027,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-truncation Euler simulation for Heston spot and variance paths."""
    rng = np.random.default_rng(seed)
    dt = maturity / steps
    spots = np.empty((steps + 1, paths))
    variances = np.empty((steps + 1, paths))
    spots[0] = spot
    variances[0] = variance0
    for step in range(1, steps + 1):
        z1 = rng.normal(size=paths)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.normal(size=paths)
        variance_positive = np.maximum(variances[step - 1], 0.0)
        variances[step] = (
            variances[step - 1]
            + kappa * (theta - variance_positive) * dt
            + vol_of_vol * np.sqrt(variance_positive) * np.sqrt(dt) * z2
        )
        variances[step] = np.maximum(variances[step], 0.0)
        spots[step] = spots[step - 1] * np.exp(
            (rate - dividend_yield - 0.5 * variance_positive) * dt
            + np.sqrt(variance_positive) * np.sqrt(dt) * z1
        )
    return spots, variances
