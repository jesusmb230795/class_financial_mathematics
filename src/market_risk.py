"""Market risk helpers used by the course notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2, kurtosis, norm, skew


def _validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")


def _clean_series(returns: pd.Series) -> pd.Series:
    clean = returns.dropna()
    if clean.empty:
        raise ValueError("returns must contain at least one non-missing value")
    return clean.astype(float)


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
    """Return historical VaR as a positive loss number."""
    _validate_alpha(alpha)
    clean = _clean_series(returns)
    return float(-clean.quantile(alpha))


def expected_shortfall(returns: pd.Series, alpha: float = 0.01) -> float:
    """Return Expected Shortfall as the average loss beyond historical VaR."""
    _validate_alpha(alpha)
    clean = _clean_series(returns)
    threshold = clean.quantile(alpha)
    tail = clean[clean <= threshold]
    if tail.empty:
        return historical_var(clean, alpha=alpha)
    return float(-tail.mean())


def gaussian_var(returns: pd.Series, alpha: float = 0.01) -> float:
    """Return Gaussian VaR as a positive loss number."""
    _validate_alpha(alpha)
    clean = _clean_series(returns)
    quantile = clean.mean() + clean.std(ddof=1) * norm.ppf(alpha)
    return float(-quantile)


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


def cornish_fisher_var(
    returns: pd.Series,
    alpha: float = 0.01,
    validate_moments: bool = True,
) -> float:
    """Return Cornish-Fisher modified VaR as a positive loss number."""
    _validate_alpha(alpha)
    clean = _clean_series(returns)
    skewness = float(skew(clean, bias=False))
    excess_kurtosis = float(kurtosis(clean, fisher=True, bias=False))

    if validate_moments:
        if not np.isfinite(skewness) or not np.isfinite(excess_kurtosis):
            raise ValueError("skewness and excess kurtosis must be finite")
        if abs(skewness) > 2.0 or excess_kurtosis > 12.0 or excess_kurtosis < -2.0:
            raise ValueError(
                "Cornish-Fisher adjustment is unstable for extreme skewness or kurtosis"
            )

    adjusted_z = cornish_fisher_z(alpha, skewness, excess_kurtosis)
    modified_quantile = clean.mean() + clean.std(ddof=1) * adjusted_z
    return float(-modified_quantile)


def ewma_volatility(returns: pd.Series, lambda_: float = 0.94) -> pd.Series:
    """Estimate one-step-ahead EWMA volatility from a return series."""
    if not 0 < lambda_ < 1:
        raise ValueError("lambda_ must be between 0 and 1")
    clean = _clean_series(returns)

    variance = np.empty(len(clean), dtype=float)
    variance[0] = clean.var(ddof=1)
    squared_returns = np.square(clean.to_numpy())

    for idx in range(1, len(clean)):
        variance[idx] = lambda_ * variance[idx - 1] + (1 - lambda_) * squared_returns[idx - 1]

    return pd.Series(np.sqrt(variance), index=clean.index, name="ewma_volatility")


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
    latest_volatility = float(volatility.iloc[-1])
    return float(-standardized.quantile(alpha) * latest_volatility)


def exception_series(returns: pd.Series, var_forecast: pd.Series | float) -> pd.Series:
    """Flag observations where realized loss exceeds a positive VaR forecast."""
    clean = _clean_series(returns)
    if isinstance(var_forecast, pd.Series):
        aligned = pd.concat(
            [clean.rename("return"), var_forecast.rename("var")],
            axis=1,
            join="inner",
        ).dropna()
        return (aligned["return"] < -aligned["var"]).rename("exception")
    return (clean < -float(var_forecast)).rename("exception")


def _bernoulli_log_likelihood(successes: int, trials: int, probability: float) -> float:
    failures = trials - successes
    return float(xlogy(successes, probability) + xlogy(failures, 1 - probability))


def kupiec_pof_test(exceptions: pd.Series | np.ndarray, alpha: float = 0.01) -> pd.Series:
    """Run Kupiec's unconditional coverage test for VaR exceptions."""
    _validate_alpha(alpha)
    observed = pd.Series(exceptions).dropna().astype(bool)
    n_obs = int(observed.shape[0])
    if n_obs == 0:
        raise ValueError("exceptions must contain at least one observation")

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
    observed = pd.Series(exceptions).dropna().astype(int)
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

    restricted = (
        xlogy(n00 + n10, 1 - p)
        + xlogy(n01 + n11, p)
    )
    unrestricted = (
        xlogy(n00, 1 - p0)
        + xlogy(n01, p0)
        + xlogy(n10, 1 - p1)
        + xlogy(n11, p1)
    )
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
    if exception_count < 0:
        raise ValueError("exception_count cannot be negative")

    amber_multipliers = {
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
        zone = "amber"
        multiplier = amber_multipliers[exception_count]

    return pd.Series(
        {
            "exceptions": exception_count,
            "zone": zone,
            "multiplier": multiplier,
        }
    )


def stress_scenario_loss(weights: pd.Series, shocks: pd.Series) -> pd.Series:
    """Compute asset and total portfolio losses from deterministic return shocks."""
    aligned = pd.concat([weights.rename("weight"), shocks.rename("shock")], axis=1).dropna()
    asset_losses = -(aligned["weight"] * aligned["shock"])
    asset_losses.loc["portfolio_total"] = asset_losses.sum()
    return asset_losses.rename("scenario_loss")
