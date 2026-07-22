"""Time series diagnostic helpers for financial return modeling."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from src.market_risk import parametric_var as positive_loss_parametric_var


def adf_report(series: pd.Series, regression: str = "c") -> pd.Series:
    """Run the Augmented Dickey-Fuller test and return a compact report."""
    clean = series.dropna()
    statistic, p_value, used_lag, nobs, critical_values, icbest = adfuller(
        clean,
        regression=regression,
        autolag="AIC",
    )
    report = {
        "statistic": statistic,
        "p_value": p_value,
        "used_lag": used_lag,
        "nobs": nobs,
        "icbest": icbest,
    }
    report.update({f"critical_value_{key}": value for key, value in critical_values.items()})
    return pd.Series(report)


def ljung_box_report(
    residuals: pd.Series,
    lags: int | list[int] = 10,
    model_df: int = 0,
) -> pd.DataFrame:
    """Run a Ljung-Box autocorrelation test on a series or fitted residuals.

    When ``residuals`` come from ARIMA(p,d,q), pass ``model_df=p+q``. For an
    unfitted return series, leave ``model_df=0`` and describe the result as a
    pre-model serial-dependence screen rather than a residual diagnostic.
    """
    if model_df < 0:
        raise ValueError("model_df cannot be negative")
    return acorr_ljungbox(
        residuals.dropna(),
        lags=lags,
        model_df=model_df,
        return_df=True,
    )


def jarque_bera_report(residuals: pd.Series) -> pd.Series:
    """Run a Jarque-Bera normality test on residuals."""
    statistic, p_value = jarque_bera(residuals.dropna())
    return pd.Series({"statistic": statistic, "p_value": p_value})


def arch_lm_report(
    residuals: pd.Series,
    lags: int = 10,
    model_df: int = 0,
) -> pd.Series:
    """Run Engle's ARCH-LM test on a residual or demeaned-return series.

    The null hypothesis is that the first ``lags`` squared-residual lags have no
    explanatory power for current squared residuals. ``model_df`` accounts for
    parameters estimated in the conditional-mean model.
    """
    if lags < 1:
        raise ValueError("lags must be at least 1")
    if model_df < 0:
        raise ValueError("model_df cannot be negative")
    clean = pd.to_numeric(residuals, errors="coerce").dropna()
    if len(clean) <= lags + model_df + 1:
        raise ValueError("residuals must contain more observations than lags + model_df + 1")
    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError("residuals must contain only finite values")

    lm_statistic, lm_p_value, f_statistic, f_p_value = het_arch(
        clean,
        nlags=lags,
        ddof=model_df,
    )
    return pd.Series(
        {
            "lags": lags,
            "model_df": model_df,
            "lm_statistic": lm_statistic,
            "lm_p_value": lm_p_value,
            "f_statistic": f_statistic,
            "f_p_value": f_p_value,
        }
    )


def arima_order_search(
    series: pd.Series,
    p_values: range | list[int],
    d_values: range | list[int],
    q_values: range | list[int],
) -> pd.DataFrame:
    """Fit an ARIMA grid and reject candidates that do not converge.

    The search uses a NumPy array because the lesson compares in-sample
    information criteria and does not need a forecast date index. This avoids
    presenting an inferred business-day frequency as if it were a verified
    market calendar.
    """
    clean = series.dropna()
    rows = []

    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        result = ARIMA(
                            clean.to_numpy(),
                            order=order,
                            enforce_stationarity=True,
                            enforce_invertibility=True,
                        ).fit()
                    converged = bool(result.mle_retvals.get("converged", True))
                    warning_types = sorted(
                        {
                            warning.category.__name__
                            for warning in caught
                            if issubclass(warning.category, Warning)
                        }
                    )
                    if any(issubclass(warning.category, ConvergenceWarning) for warning in caught):
                        converged = False
                    rows.append(
                        {
                            "p": p,
                            "d": d,
                            "q": q,
                            "aic": result.aic if converged else float("nan"),
                            "bic": result.bic if converged else float("nan"),
                            "stationary": bool(
                                np.all(np.abs(result.arroots) > 1)
                            ),
                            "invertible": bool(
                                np.all(np.abs(result.maroots) > 1)
                            ),
                            "converged": converged,
                            "warning_types": ", ".join(warning_types),
                        }
                    )
                except Exception as exc:  # pragma: no cover - used as notebook guardrail
                    rows.append(
                        {
                            "p": p,
                            "d": d,
                            "q": q,
                            "aic": float("nan"),
                            "bic": float("nan"),
                            "stationary": False,
                            "invertible": False,
                            "converged": False,
                            "warning_types": "",
                            "error": str(exc),
                        }
                    )

    return pd.DataFrame(rows).sort_values("aic", na_position="last").reset_index(drop=True)


def garch_persistence(alpha: float, beta: float) -> float:
    """Return GARCH(1,1) volatility persistence."""
    parameters = np.asarray([alpha, beta], dtype=float)
    if not np.isfinite(parameters).all():
        raise ValueError("alpha and beta must be finite")
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta cannot be negative")
    return float(alpha + beta)


def garch_long_run_variance(omega: float, alpha: float, beta: float) -> float:
    """Return long-run variance for stationary GARCH(1,1)."""
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError("omega must be finite and positive")
    persistence = garch_persistence(alpha, beta)
    if persistence >= 1:
        raise ValueError("GARCH long-run variance requires alpha + beta < 1")
    return float(omega / (1 - persistence))


def garch_variance_half_life(persistence: float) -> float:
    """Return the expected GARCH variance-shock half-life in observations."""
    if not np.isfinite(persistence) or not 0 < persistence < 1:
        raise ValueError("persistence must be finite and strictly between 0 and 1")
    return float(np.log(0.5) / np.log(persistence))


def garch11_volatility_filter(
    returns: pd.Series,
    *,
    omega: float,
    alpha: float,
    beta: float,
) -> pd.Series:
    """Filter a zero-mean return series through a stationary GARCH(1,1).

    The recursion starts from the parameter-implied long-run variance, so the
    initial state does not use future sample observations. Returns and
    volatility share the same unit; for example, decimal returns produce
    decimal volatility.
    """
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if len(clean) < 2:
        raise ValueError("returns must contain at least two finite observations")
    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError("returns must contain only finite values")

    initial_variance = garch_long_run_variance(omega, alpha, beta)
    variance = np.empty(len(clean), dtype=float)
    variance[0] = initial_variance
    values = clean.to_numpy(dtype=float)
    for index in range(1, len(clean)):
        variance[index] = (
            omega
            + alpha * values[index - 1] ** 2
            + beta * variance[index - 1]
        )

    return pd.Series(
        np.sqrt(variance),
        index=clean.index,
        name="garch11_filtered_volatility",
    )


def parametric_var(mean: float, volatility: float, quantile: float) -> float:
    """Compatibility wrapper for positive-loss parametric VaR.

    New risk code should import :func:`src.market_risk.parametric_var`
    directly. The wrapper keeps existing time-series lessons stable while both
    modules share exactly one sign convention.
    """
    return positive_loss_parametric_var(mean, volatility, quantile)
