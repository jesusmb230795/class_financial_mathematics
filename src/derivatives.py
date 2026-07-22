"""Derivative pricing helpers for classroom notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


def _require_finite(**values: float) -> None:
    """Reject non-finite scalar model inputs with an actionable field name."""
    for name, value in values.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")


def _validate_option_type(option_type: str) -> None:
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")


def forward_price(
    spot: float,
    rate: float,
    maturity: float,
    dividend_yield: float = 0.0,
) -> float:
    """No-arbitrage forward price under continuous carry."""
    _require_finite(
        spot=spot,
        rate=rate,
        maturity=maturity,
        dividend_yield=dividend_yield,
    )
    if spot <= 0 or maturity < 0:
        raise ValueError("spot must be positive and maturity cannot be negative")
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
    _require_finite(
        spot=spot,
        strike=strike,
        rate=rate,
        maturity=maturity,
        dividend_yield=dividend_yield,
    )
    if spot <= 0 or strike <= 0 or maturity < 0:
        raise ValueError("spot and strike must be positive and maturity cannot be negative")
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
    if discount_factors.ndim != 1 or accrual_factors.ndim != 1:
        raise ValueError("discount_factors and accrual_factors must be one-dimensional")
    if discount_factors.size == 0 or discount_factors.shape != accrual_factors.shape:
        raise ValueError("discount_factors and accrual_factors must have equal non-zero length")
    if not np.isfinite(discount_factors).all() or not np.isfinite(accrual_factors).all():
        raise ValueError("discount_factors and accrual_factors must be finite")
    if (discount_factors <= 0).any() or (accrual_factors <= 0).any():
        raise ValueError("discount_factors and accrual_factors must be positive")
    annuity = np.sum(accrual_factors * discount_factors)
    return float((1 - discount_factors[-1]) / annuity)


def option_payoff(
    terminal_price: np.ndarray, strike: float, option_type: str = "call"
) -> np.ndarray:
    """European call or put payoff."""
    terminal_price = np.asarray(terminal_price, dtype=float)
    _require_finite(strike=strike)
    _validate_option_type(option_type)
    if strike <= 0:
        raise ValueError("strike must be positive")
    if not np.isfinite(terminal_price).all() or (terminal_price < 0).any():
        raise ValueError("terminal_price must contain finite non-negative values")
    if option_type == "call":
        return np.maximum(terminal_price - strike, 0.0)
    return np.maximum(strike - terminal_price, 0.0)


def black_scholes_inputs(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    """Return Black-Scholes d1 and d2."""
    _require_finite(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        dividend_yield=dividend_yield,
    )
    if spot <= 0 or strike <= 0 or volatility <= 0 or maturity <= 0:
        raise ValueError("spot, strike, volatility, and maturity must be positive")
    d1 = (np.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * maturity) / (
        volatility * np.sqrt(maturity)
    )
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
    _validate_option_type(option_type)
    d1, d2 = black_scholes_inputs(spot, strike, rate, volatility, maturity, dividend_yield)
    discounted_spot = spot * np.exp(-dividend_yield * maturity)
    discounted_strike = strike * np.exp(-rate * maturity)
    if option_type == "call":
        return float(discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2))
    return float(discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1))


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
    _validate_option_type(option_type)
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
    _require_finite(
        call_price=call_price,
        put_price=put_price,
        spot=spot,
        strike=strike,
        rate=rate,
        maturity=maturity,
        dividend_yield=dividend_yield,
    )
    return float(
        call_price
        + strike * np.exp(-rate * maturity)
        - put_price
        - spot * np.exp(-dividend_yield * maturity)
    )


def option_price_bounds(
    spot: float,
    strike: float,
    rate: float,
    maturity: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    """Return model-independent European option bounds under continuous carry."""
    _validate_option_type(option_type)
    _require_finite(
        spot=spot,
        strike=strike,
        rate=rate,
        maturity=maturity,
        dividend_yield=dividend_yield,
    )
    if spot <= 0 or strike <= 0 or maturity <= 0:
        raise ValueError("spot, strike, and maturity must be positive")
    discounted_spot = spot * np.exp(-dividend_yield * maturity)
    discounted_strike = strike * np.exp(-rate * maturity)
    if option_type == "call":
        return float(max(0.0, discounted_spot - discounted_strike)), float(discounted_spot)
    return float(max(0.0, discounted_strike - discounted_spot)), float(discounted_strike)


def call_price_arbitrage_diagnostics(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    spot: float,
    rate: float,
    maturity: float,
    dividend_yield: float = 0.0,
    tolerance: float = 1e-10,
) -> pd.Series:
    """Check same-maturity call quotes for bounds, monotonicity, and convexity."""
    strikes = np.asarray(strikes, dtype=float)
    call_prices = np.asarray(call_prices, dtype=float)
    _require_finite(
        spot=spot,
        rate=rate,
        maturity=maturity,
        dividend_yield=dividend_yield,
        tolerance=tolerance,
    )
    if strikes.ndim != 1 or call_prices.ndim != 1 or strikes.shape != call_prices.shape:
        raise ValueError("strikes and call_prices must be equal-length one-dimensional arrays")
    if strikes.size < 3:
        raise ValueError("at least three ordered strikes are required")
    if not np.isfinite(strikes).all() or not np.isfinite(call_prices).all():
        raise ValueError("strikes and call_prices must be finite")
    if (strikes <= 0).any() or (np.diff(strikes) <= 0).any():
        raise ValueError("strikes must be positive and strictly increasing")
    if (call_prices < 0).any() or tolerance < 0:
        raise ValueError("call_prices and tolerance cannot be negative")

    discounted_spot = spot * np.exp(-dividend_yield * maturity)
    discounted_strikes = strikes * np.exp(-rate * maturity)
    lower_bounds = np.maximum(discounted_spot - discounted_strikes, 0.0)
    slopes = np.diff(call_prices) / np.diff(strikes)
    slope_changes = np.diff(slopes)
    lower_slope_bound = -np.exp(-rate * maturity)

    checks = {
        "pointwise_lower_bound": bool(np.all(call_prices >= lower_bounds - tolerance)),
        "pointwise_upper_bound": bool(np.all(call_prices <= discounted_spot + tolerance)),
        "non_increasing_in_strike": bool(np.all(slopes <= tolerance)),
        "vertical_spread_bound": bool(np.all(slopes >= lower_slope_bound - tolerance)),
        "convex_in_strike": bool(np.all(slope_changes >= -tolerance)),
    }
    return pd.Series(
        {
            **checks,
            "all_checks_pass": bool(all(checks.values())),
            "minimum_call_spread_slope": float(slopes.min()),
            "minimum_slope_change": float(slope_changes.min()),
        },
        name="same_maturity_call_arbitrage_checks",
    )


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    rate: float,
    maturity: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
    lower: float = 1e-8,
    upper: float = 5.0,
) -> float:
    """Solve Black-Scholes implied volatility by robust bracketing."""
    _validate_option_type(option_type)
    _require_finite(market_price=market_price, lower=lower, upper=upper)
    if lower <= 0 or upper <= lower:
        raise ValueError("lower and upper must define a positive volatility bracket")
    lower_bound, upper_bound = option_price_bounds(
        spot,
        strike,
        rate,
        maturity,
        option_type,
        dividend_yield,
    )
    tolerance = 1e-10
    if market_price < lower_bound - tolerance:
        raise ValueError("market_price violates the no-arbitrage lower bound")
    if market_price >= upper_bound - tolerance:
        raise ValueError("market_price violates the finite no-arbitrage upper bound")
    if np.isclose(market_price, lower_bound, atol=tolerance, rtol=0.0):
        return 0.0

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

    lower_value = objective(lower)
    upper_value = objective(upper)
    if lower_value * upper_value > 0:
        raise ValueError("implied volatility is not bracketed by lower and upper")
    return float(brentq(objective, lower, upper))


@dataclass(frozen=True)
class MonteCarloPriceResult:
    """Discounted European option estimate and sampling uncertainty."""

    price: float
    standard_error: float
    ci_95_lower: float
    ci_95_upper: float


def monte_carlo_european_option_price(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    simulations: int = 100_000,
    option_type: str = "call",
    dividend_yield: float = 0.0,
    seed: int = 42,
) -> MonteCarloPriceResult:
    """Price a European option from terminal risk-neutral GBM draws."""
    _validate_option_type(option_type)
    black_scholes_inputs(spot, strike, rate, volatility, maturity, dividend_yield)
    if not isinstance(simulations, (int, np.integer)) or simulations < 2:
        raise ValueError("simulations must be an integer of at least 2")
    rng = np.random.default_rng(seed)
    shocks = rng.normal(size=simulations)
    terminal_prices = spot * np.exp(
        (rate - dividend_yield - 0.5 * volatility**2) * maturity
        + volatility * np.sqrt(maturity) * shocks
    )
    discounted_payoffs = np.exp(-rate * maturity) * option_payoff(
        terminal_prices,
        strike,
        option_type,
    )
    price = float(discounted_payoffs.mean())
    standard_error = float(discounted_payoffs.std(ddof=1) / np.sqrt(simulations))
    critical_value = float(norm.ppf(0.975))
    return MonteCarloPriceResult(
        price=price,
        standard_error=standard_error,
        ci_95_lower=price - critical_value * standard_error,
        ci_95_upper=price + critical_value * standard_error,
    )


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
    _validate_option_type(option_type)
    black_scholes_inputs(spot, strike, rate, volatility, maturity, dividend_yield)
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
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
    black_scholes_inputs(spot, strike, rate, volatility, maturity, dividend_yield)
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
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
    black_scholes_inputs(spot, spot, rate, volatility, maturity, dividend_yield)
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not isinstance(paths, (int, np.integer)) or paths < 1:
        raise ValueError("paths must be a positive integer")
    rng = np.random.default_rng(seed)
    dt = maturity / steps
    shocks = rng.normal(size=(steps, paths))
    increments = (rate - dividend_yield - 0.5 * volatility**2) * dt + volatility * np.sqrt(
        dt
    ) * shocks
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
    black_scholes_inputs(spot, strike, rate, volatility, maturity, dividend_yield)
    if not isinstance(observations, (int, np.integer)) or observations < 1:
        raise ValueError("observations must be a positive integer")
    observation_times = np.linspace(maturity / observations, maturity, observations)
    mean_log = (
        np.log(spot) + (rate - dividend_yield - 0.5 * volatility**2) * observation_times.mean()
    )
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
    if not isinstance(paths, (int, np.integer)) or paths < 2:
        raise ValueError("paths must be an integer of at least 2")
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
    geometric_variance = float(np.var(geometric_payoffs, ddof=1))
    variance_scale = max(1.0, float(np.mean(np.square(geometric_payoffs))))
    if (
        not np.isfinite(geometric_variance)
        or geometric_variance <= np.finfo(float).eps * variance_scale
    ):
        beta = 0.0
        adjusted = arithmetic_payoffs.copy()
    else:
        covariance = float(np.cov(arithmetic_payoffs, geometric_payoffs, ddof=1)[0, 1])
        beta = covariance / geometric_variance
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
    _require_finite(
        barrier=barrier,
        volatility=volatility,
        monitoring_interval=monitoring_interval,
        beta=beta,
    )
    if barrier <= 0 or volatility < 0 or monitoring_interval <= 0 or beta <= 0:
        raise ValueError(
            "barrier, monitoring_interval, and beta must be positive; volatility cannot be negative"
        )
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
    _require_finite(
        spot=spot,
        variance0=variance0,
        rate=rate,
        maturity=maturity,
        kappa=kappa,
        theta=theta,
        vol_of_vol=vol_of_vol,
        rho=rho,
        dividend_yield=dividend_yield,
    )
    if spot <= 0 or variance0 < 0 or maturity <= 0:
        raise ValueError("spot and maturity must be positive; variance0 cannot be negative")
    if kappa < 0 or theta < 0 or vol_of_vol < 0:
        raise ValueError("kappa, theta, and vol_of_vol cannot be negative")
    if not -1 <= rho <= 1:
        raise ValueError("rho must be between -1 and 1")
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not isinstance(paths, (int, np.integer)) or paths < 1:
        raise ValueError("paths must be a positive integer")
    rng = np.random.default_rng(seed)
    dt = maturity / steps
    spots = np.empty((steps + 1, paths))
    latent_variances = np.empty((steps + 1, paths))
    spots[0] = spot
    latent_variances[0] = variance0
    for step in range(1, steps + 1):
        z1 = rng.normal(size=paths)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.normal(size=paths)
        variance_positive = np.maximum(latent_variances[step - 1], 0.0)
        latent_variances[step] = (
            latent_variances[step - 1]
            + kappa * (theta - variance_positive) * dt
            + vol_of_vol * np.sqrt(variance_positive) * np.sqrt(dt) * z2
        )
        spots[step] = spots[step - 1] * np.exp(
            (rate - dividend_yield - 0.5 * variance_positive) * dt
            + np.sqrt(variance_positive) * np.sqrt(dt) * z1
        )
    return spots, np.maximum(latent_variances, 0.0)
