"""Market risk helpers used by the course notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2, kurtosis, norm, skew, t


CORNISH_FISHER_MAX_ABS_SKEWNESS = 2.0
CORNISH_FISHER_MIN_EXCESS_KURTOSIS = -2.0
CORNISH_FISHER_MAX_EXCESS_KURTOSIS = 12.0


def _validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")


def _clean_series(returns: pd.Series) -> pd.Series:
    clean = returns.dropna()
    if clean.empty:
        raise ValueError("returns must contain at least one non-missing value")
    clean = clean.astype(float)
    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError("returns must contain only finite values")
    return clean


def _positive_loss(value: float) -> float:
    """Map a signed loss estimate to the book's non-negative reporting convention."""
    if not np.isfinite(value):
        raise ValueError("loss estimate must be finite")
    return float(max(0.0, value))


def rebalanced_portfolio_returns(
    simple_returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """Aggregate simple asset returns for a portfolio rebalanced each interval."""
    if simple_returns.empty:
        raise ValueError("simple_returns must contain at least one observation")
    if simple_returns.columns.has_duplicates or weights.index.has_duplicates:
        raise ValueError("asset labels must be unique")
    missing_weights = simple_returns.columns.difference(weights.index)
    extra_weights = weights.index.difference(simple_returns.columns)
    if not missing_weights.empty or not extra_weights.empty:
        raise ValueError(
            "weights must match the return columns exactly; "
            f"missing={missing_weights.tolist()}, extra={extra_weights.tolist()}"
        )
    numeric_returns = simple_returns.astype(float)
    numeric_weights = weights.reindex(simple_returns.columns).astype(float)
    if not np.isfinite(numeric_returns.to_numpy()).all():
        raise ValueError("simple_returns must contain only finite values")
    if not np.isfinite(numeric_weights.to_numpy()).all():
        raise ValueError("weights must contain only finite values")
    if not np.isclose(float(numeric_weights.sum()), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("weights must sum to 1")
    result = numeric_returns.mul(numeric_weights, axis="columns").sum(axis="columns")
    result.name = "portfolio_simple_return"
    result.attrs = dict(simple_returns.attrs)
    result.attrs["portfolio_rule"] = "constant weights rebalanced each observed interval"
    result.attrs["weights"] = numeric_weights.to_dict()
    return result


def parametric_var(mean: float, volatility: float, quantile: float) -> float:
    """Return one-step VaR as a non-negative loss.

    ``quantile`` is the standardized left-tail return quantile. For example,
    pass ``norm.ppf(alpha)`` for Gaussian innovations. ``mean`` and
    ``volatility`` must use the same units, such as daily decimal returns or
    daily percentage returns.
    """
    values = np.asarray([mean, volatility, quantile], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("mean, volatility, and quantile must be finite")
    if volatility < 0:
        raise ValueError("volatility cannot be negative")
    return _positive_loss(-(mean + volatility * quantile))


def simple_loss_from_log_return(log_return: float) -> float:
    """Convert a decimal log-return threshold to non-negative simple loss.

    If ``g`` is a log return, the corresponding simple return is
    ``expm1(g)``.  A long-position loss is therefore ``-expm1(g)`` before the
    book's non-negative reporting floor is applied.
    """
    if not np.isfinite(log_return):
        raise ValueError("log_return must be finite")
    return _positive_loss(-np.expm1(log_return))


def standardized_student_t_quantile(alpha: float, degrees_of_freedom: float) -> float:
    """Return a unit-variance Student's t left-tail quantile.

    ``arch`` parameterizes Student's t innovations to have unit variance. A
    raw SciPy t quantile has variance ``nu / (nu - 2)``, so it must be scaled
    before it can be combined with an ``arch`` conditional-volatility forecast.
    """
    _validate_alpha(alpha)
    if not np.isfinite(degrees_of_freedom) or degrees_of_freedom <= 2:
        raise ValueError("degrees_of_freedom must be finite and greater than 2")
    scale = np.sqrt((degrees_of_freedom - 2.0) / degrees_of_freedom)
    return float(t.ppf(alpha, df=degrees_of_freedom) * scale)


def target_semideviation(
    returns: pd.Series,
    target: float = 0.0,
    periods_per_year: int | None = None,
) -> float:
    """Compute downside deviation relative to a target return."""
    clean = _clean_series(returns)
    downside = np.minimum(clean - target, 0.0)
    semideviation = float(np.sqrt(np.mean(np.square(downside))))
    if periods_per_year is None:
        return semideviation
    return semideviation * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: pd.Series,
    target: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute an annualized Sortino ratio against a per-period target."""
    clean = _clean_series(returns)
    downside = target_semideviation(clean, target=target, periods_per_year=periods_per_year)
    if downside == 0:
        return float("nan")
    annualized_excess_return = float((clean - target).mean() * periods_per_year)
    return annualized_excess_return / downside


def historical_var(returns: pd.Series, alpha: float = 0.01) -> float:
    """Return historical VaR as a non-negative loss number."""
    _validate_alpha(alpha)
    clean = _clean_series(returns)
    return _positive_loss(-clean.quantile(alpha))


def expected_shortfall(returns: pd.Series, alpha: float = 0.01) -> float:
    """Return empirical Expected Shortfall from the worst ``alpha`` probability mass.

    The finite-sample estimate averages complete worst observations plus the
    fractional boundary observation needed to represent exactly ``alpha * n``
    observations. This is the empirical quantile-integral definition and avoids
    over-weighting ties at an interpolated VaR threshold.
    """
    _validate_alpha(alpha)
    clean = _clean_series(returns)
    ordered = np.sort(clean.to_numpy())
    tail_mass = alpha * ordered.size
    complete_observations = int(np.floor(tail_mass))
    fractional_observation = tail_mass - complete_observations
    tail_sum = float(ordered[:complete_observations].sum())
    if fractional_observation > 0:
        tail_sum += fractional_observation * float(ordered[complete_observations])
    return _positive_loss(-(tail_sum / tail_mass))


def gaussian_var(returns: pd.Series, alpha: float = 0.01) -> float:
    """Return Gaussian VaR as a non-negative loss number."""
    _validate_alpha(alpha)
    clean = _clean_series(returns)
    return parametric_var(
        mean=float(clean.mean()),
        volatility=float(clean.std(ddof=1)),
        quantile=float(norm.ppf(alpha)),
    )


def gaussian_monte_carlo_var(
    returns: pd.Series,
    *,
    confidence: float = 0.99,
    simulations: int = 100_000,
    seed: int = 42,
) -> float:
    """Estimate one-period Gaussian VaR with reproducible fitted simulations.

    ``confidence`` is the reported loss confidence level, so the simulated
    left-tail return probability is ``1 - confidence``. The fitted mean and
    sample standard deviation retain the units of ``returns``.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if not isinstance(simulations, (int, np.integer)) or isinstance(simulations, bool):
        raise ValueError("simulations must be an integer of at least 2")
    if simulations < 2:
        raise ValueError("simulations must be an integer of at least 2")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    clean = _clean_series(returns)
    if clean.size < 2:
        raise ValueError("returns must contain at least two observations")
    rng = np.random.default_rng(int(seed))
    simulated_returns = rng.normal(
        loc=float(clean.mean()),
        scale=float(clean.std(ddof=1)),
        size=int(simulations),
    )
    return _positive_loss(-float(np.quantile(simulated_returns, 1 - confidence)))


def cornish_fisher_z(alpha: float, skewness: float, excess_kurtosis: float) -> float:
    """Adjust a Gaussian z-score with the Cornish-Fisher expansion."""
    _validate_alpha(alpha)
    z = norm.ppf(alpha)
    return float(
        z
        + ((z**2 - 1) * skewness / 6)
        + ((z**3 - 3 * z) * excess_kurtosis / 24)
        - ((2 * z**3 - 5 * z) * skewness**2 / 36)
    )


def cornish_fisher_moment_report(returns: pd.Series) -> pd.Series:
    """Report sample moments and whether the course guardrail permits CF VaR."""
    clean = _clean_series(returns)
    skewness = float(skew(clean, bias=False))
    excess_kurtosis = float(kurtosis(clean, fisher=True, bias=False))
    finite = bool(np.isfinite(skewness) and np.isfinite(excess_kurtosis))
    stable = bool(
        finite
        and abs(skewness) <= CORNISH_FISHER_MAX_ABS_SKEWNESS
        and CORNISH_FISHER_MIN_EXCESS_KURTOSIS
        <= excess_kurtosis
        <= CORNISH_FISHER_MAX_EXCESS_KURTOSIS
    )
    return pd.Series(
        {
            "skewness": skewness,
            "excess_kurtosis": excess_kurtosis,
            "cornish_fisher_available": stable,
        }
    )


def cornish_fisher_var(
    returns: pd.Series,
    alpha: float = 0.01,
) -> float:
    """Return guarded Cornish-Fisher VaR as a non-negative loss number."""
    _validate_alpha(alpha)
    clean = _clean_series(returns)
    moment_report = cornish_fisher_moment_report(clean)
    skewness = float(moment_report["skewness"])
    excess_kurtosis = float(moment_report["excess_kurtosis"])

    if not np.isfinite(skewness) or not np.isfinite(excess_kurtosis):
        raise ValueError("skewness and excess kurtosis must be finite")
    if not bool(moment_report["cornish_fisher_available"]):
        raise ValueError(
            "Cornish-Fisher adjustment is unavailable: "
            f"skewness={skewness:.3f}, excess_kurtosis={excess_kurtosis:.3f} "
            "is outside the course guardrail"
        )

    adjusted_z = cornish_fisher_z(alpha, skewness, excess_kurtosis)
    return parametric_var(
        mean=float(clean.mean()),
        volatility=float(clean.std(ddof=1)),
        quantile=adjusted_z,
    )


def ewma_volatility(
    returns: pd.Series,
    lambda_: float = 0.94,
    initial_variance: float | None = None,
) -> pd.Series:
    """Estimate one-step-ahead EWMA volatility from a return series.

    The historical default initializes from full-sample variance. Pass an
    explicit positive ``initial_variance`` in chronological filtering workflows
    to avoid using later observations in the initial state.
    """
    if not 0 < lambda_ < 1:
        raise ValueError("lambda_ must be between 0 and 1")
    clean = _clean_series(returns)

    variance = np.empty(len(clean), dtype=float)
    if initial_variance is None:
        if len(clean) < 2:
            raise ValueError(
                "returns must contain at least two observations when "
                "initial_variance is not supplied"
            )
        variance[0] = clean.var(ddof=1)
    else:
        if not np.isfinite(initial_variance) or initial_variance <= 0:
            raise ValueError("initial_variance must be finite and positive")
        variance[0] = initial_variance
    squared_returns = np.square(clean.to_numpy())

    for idx in range(1, len(clean)):
        variance[idx] = lambda_ * variance[idx - 1] + (1 - lambda_) * squared_returns[idx - 1]

    return pd.Series(np.sqrt(variance), index=clean.index, name="ewma_volatility")


def ewma_next_volatility(
    returns: pd.Series,
    lambda_: float = 0.94,
    initial_variance: float | None = None,
) -> float:
    """Return the EWMA volatility forecast after incorporating the last return."""
    clean = _clean_series(returns)
    volatility = ewma_volatility(
        clean,
        lambda_=lambda_,
        initial_variance=initial_variance,
    )
    next_variance = (
        lambda_ * float(volatility.iloc[-1]) ** 2 + (1 - lambda_) * float(clean.iloc[-1]) ** 2
    )
    return float(np.sqrt(next_variance))


def volatility_weighted_historical_var(
    returns: pd.Series,
    alpha: float = 0.01,
    lambda_: float = 0.94,
) -> float:
    """Return volatility-weighted historical VaR using EWMA scaling."""
    _validate_alpha(alpha)
    clean = _clean_series(returns)
    volatility = ewma_volatility(clean, lambda_=lambda_)
    standardized = (clean / volatility).replace([np.inf, -np.inf], np.nan).dropna()
    if standardized.empty:
        return historical_var(clean, alpha=alpha)
    forecast_volatility = ewma_next_volatility(clean, lambda_=lambda_)
    return _positive_loss(-standardized.quantile(alpha) * forecast_volatility)


def exception_series(returns: pd.Series, var_forecast: pd.Series | float) -> pd.Series:
    """Flag observations where realized loss exceeds a positive VaR forecast."""
    clean = _clean_series(returns)
    if isinstance(var_forecast, pd.Series):
        aligned = pd.concat(
            [clean.rename("return"), var_forecast.rename("var")],
            axis=1,
            join="inner",
        ).dropna()
        if not np.isfinite(aligned["var"].to_numpy()).all() or (aligned["var"] < 0).any():
            raise ValueError("var_forecast must contain finite non-negative loss thresholds")
        return (aligned["return"] < -aligned["var"]).rename("exception")
    threshold = float(var_forecast)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("var_forecast must be a finite non-negative loss threshold")
    return (clean < -threshold).rename("exception")


def _bernoulli_log_likelihood(successes: int, trials: int, probability: float) -> float:
    failures = trials - successes
    return float(xlogy(successes, probability) + xlogy(failures, 1 - probability))


def _validated_exceptions(exceptions: pd.Series | np.ndarray) -> pd.Series:
    """Return a missing-free binary exception series without truthy coercion."""
    observed = pd.Series(exceptions).dropna()
    if observed.empty:
        raise ValueError("exceptions must contain at least one observation")
    if pd.api.types.is_bool_dtype(observed.dtype):
        return observed.astype(int)
    numeric = pd.to_numeric(observed, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all() or not numeric.isin([0, 1]).all():
        raise ValueError("exceptions must contain only binary 0/1 or boolean values")
    return numeric.astype(int)


def kupiec_pof_test(exceptions: pd.Series | np.ndarray, alpha: float = 0.01) -> pd.Series:
    """Run Kupiec's unconditional coverage test for VaR exceptions."""
    _validate_alpha(alpha)
    observed = _validated_exceptions(exceptions)
    n_obs = int(observed.shape[0])

    n_exceptions = int(observed.sum())
    empirical_alpha = n_exceptions / n_obs
    restricted = _bernoulli_log_likelihood(n_exceptions, n_obs, alpha)
    unrestricted = _bernoulli_log_likelihood(n_exceptions, n_obs, empirical_alpha)
    lr_pof = max(0.0, -2 * (restricted - unrestricted))

    return pd.Series(
        {
            "observations": n_obs,
            "exceptions": n_exceptions,
            "expected_exceptions": n_obs * alpha,
            "exception_rate": empirical_alpha,
            "lr_pof": lr_pof,
            "p_value": float(chi2.sf(lr_pof, df=1)),
        }
    )


def christoffersen_independence_test(exceptions: pd.Series | np.ndarray) -> pd.Series:
    """Run Christoffersen's independence test for exception clustering."""
    observed = _validated_exceptions(exceptions)
    if observed.shape[0] < 2:
        raise ValueError("exceptions must contain at least two observations")

    previous = observed.iloc[:-1].to_numpy()
    current = observed.iloc[1:].to_numpy()
    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())

    p0 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    p1 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    p = (n01 + n11) / (n00 + n01 + n10 + n11)

    restricted = xlogy(n00 + n10, 1 - p) + xlogy(n01 + n11, p)
    unrestricted = xlogy(n00, 1 - p0) + xlogy(n01, p0) + xlogy(n10, 1 - p1) + xlogy(n11, p1)
    lr_independence = max(0.0, float(-2 * (restricted - unrestricted)))

    return pd.Series(
        {
            "n00": n00,
            "n01": n01,
            "n10": n10,
            "n11": n11,
            "lr_independence": lr_independence,
            "p_value": float(chi2.sf(lr_independence, df=1)),
        }
    )


def conditional_coverage_test(
    exceptions: pd.Series | np.ndarray,
    alpha: float = 0.01,
) -> pd.Series:
    """Combine Kupiec and Christoffersen tests into a conditional coverage test."""
    kupiec = kupiec_pof_test(exceptions, alpha=alpha)
    independence = christoffersen_independence_test(exceptions)
    lr_cc = float(kupiec["lr_pof"] + independence["lr_independence"])
    return pd.Series(
        {
            "lr_conditional_coverage": lr_cc,
            "p_value": float(chi2.sf(lr_cc, df=2)),
            "kupiec_p_value": kupiec["p_value"],
            "independence_p_value": independence["p_value"],
        }
    )


def basel_traffic_light(exception_count: int) -> pd.Series:
    """Map 250-day VaR exceptions to Basel traffic-light zones and multipliers."""
    if (
        not isinstance(exception_count, (int, np.integer))
        or isinstance(exception_count, bool)
        or not 0 <= exception_count <= 250
    ):
        raise ValueError("exception_count must be an integer between 0 and 250")

    yellow_multipliers = {
        5: 3.40,
        6: 3.50,
        7: 3.65,
        8: 3.75,
        9: 3.85,
    }
    if exception_count <= 4:
        zone = "green"
        multiplier = 3.00
    elif exception_count >= 10:
        zone = "red"
        multiplier = 4.00
    else:
        zone = "yellow"
        multiplier = yellow_multipliers[exception_count]

    return pd.Series(
        {
            "exceptions": exception_count,
            "zone": zone,
            "multiplier": multiplier,
        }
    )


def stress_scenario_loss(weights: pd.Series, shocks: pd.Series) -> pd.Series:
    """Compute asset and total portfolio losses from deterministic return shocks."""
    if weights.index.has_duplicates or shocks.index.has_duplicates:
        raise ValueError("weights and shocks must use unique asset labels")
    missing_shocks = weights.index.difference(shocks.index)
    extra_shocks = shocks.index.difference(weights.index)
    if not missing_shocks.empty or not extra_shocks.empty:
        raise ValueError(
            "weights and shocks must have identical labels; "
            f"missing_shocks={missing_shocks.tolist()}, extra_shocks={extra_shocks.tolist()}"
        )
    aligned = pd.concat(
        [weights.rename("weight"), shocks.reindex(weights.index).rename("shock")],
        axis=1,
    ).astype(float)
    if not np.isfinite(aligned.to_numpy()).all():
        raise ValueError("weights and shocks must contain only finite values")
    asset_losses = -(aligned["weight"] * aligned["shock"])
    asset_losses.loc["portfolio_total"] = asset_losses.sum()
    return asset_losses.rename("scenario_loss")
