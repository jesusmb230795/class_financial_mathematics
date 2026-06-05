"""Time series diagnostic helpers for financial return modeling."""

from __future__ import annotations

import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


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
    """Run a Ljung-Box residual autocorrelation test."""
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


def arima_order_search(
    series: pd.Series,
    p_values: range | list[int],
    d_values: range | list[int],
    q_values: range | list[int],
) -> pd.DataFrame:
    """Fit a small ARIMA grid and return comparable information criteria."""
    clean = series.dropna()
    rows = []

    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    result = ARIMA(
                        clean,
                        order=order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit()
                    rows.append(
                        {
                            "p": p,
                            "d": d,
                            "q": q,
                            "aic": result.aic,
                            "bic": result.bic,
                            "converged": bool(result.mle_retvals.get("converged", True)),
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
                            "converged": False,
                            "error": str(exc),
                        }
                    )

    return pd.DataFrame(rows).sort_values("aic", na_position="last").reset_index(drop=True)


def garch_persistence(alpha: float, beta: float) -> float:
    """Return GARCH(1,1) volatility persistence."""
    return float(alpha + beta)


def garch_long_run_variance(omega: float, alpha: float, beta: float) -> float:
    """Return long-run variance for stationary GARCH(1,1)."""
    persistence = garch_persistence(alpha, beta)
    if persistence >= 1:
        raise ValueError("GARCH long-run variance requires alpha + beta < 1")
    return float(omega / (1 - persistence))


def parametric_var(mean: float, volatility: float, quantile: float) -> float:
    """Compute one-step parametric VaR from mean, volatility, and a quantile."""
    return float(mean + volatility * quantile)
