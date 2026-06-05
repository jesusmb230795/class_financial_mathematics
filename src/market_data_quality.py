"""Market data quality helpers used across class notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def source_inventory_template() -> pd.DataFrame:
    """Return an empty template for documenting financial data sources."""
    return pd.DataFrame(
        columns=[
            "provider",
            "instrument_or_variable",
            "frequency",
            "start",
            "end",
            "field",
            "currency",
            "calendar",
            "known_limitations",
        ]
    )


def banxico_series_catalog() -> pd.DataFrame:
    """Return a compact catalog of Banxico series used in the course."""
    return pd.DataFrame(
        [
            {
                "series": "SF61745",
                "meaning": "Target rate",
                "course_use": "monetary policy and short-rate context",
            },
            {
                "series": "SF60648",
                "meaning": "28-day TIIE",
                "course_use": "interbank and floating-rate examples",
            },
            {
                "series": "SF60633",
                "meaning": "28-day CETES",
                "course_use": "short risk-free rate proxy",
            },
            {
                "series": "SF43718",
                "meaning": "USD/MXN FIX",
                "course_use": "FX risk and macro dashboards",
            },
            {
                "series": "SP68257",
                "meaning": "UDI",
                "course_use": "inflation-linked valuation context",
            },
        ]
    )


def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute simple returns from a price series or price matrix."""
    return prices.pct_change().dropna(how="all")


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute continuously compounded returns from a price series or matrix."""
    return np.log(prices / prices.shift(1)).dropna(how="all")


def annualized_volatility(
    returns: pd.Series | pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.Series | float:
    """Annualize return volatility using a square-root-of-time convention."""
    volatility = returns.std() * np.sqrt(periods_per_year)
    if isinstance(volatility, pd.Series):
        return volatility
    return float(volatility)


def align_to_business_calendar(
    data: pd.Series | pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    fill_method: str | None = "ffill",
) -> pd.Series | pd.DataFrame:
    """Reindex data to a business-day calendar and optionally fill missing values."""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("data must use a DatetimeIndex")

    start_date = pd.Timestamp(start) if start is not None else data.index.min()
    end_date = pd.Timestamp(end) if end is not None else data.index.max()
    calendar = pd.bdate_range(start=start_date, end=end_date)
    aligned = data.sort_index().reindex(calendar)

    if fill_method is None:
        return aligned
    if fill_method not in {"ffill", "bfill"}:
        raise ValueError("fill_method must be 'ffill', 'bfill', or None")
    return getattr(aligned, fill_method)()


def data_quality_report(data: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Summarize missingness and date coverage for market or macro data."""
    frame = data.to_frame(name=data.name or "value") if isinstance(data, pd.Series) else data
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("data must use a DatetimeIndex")

    return pd.DataFrame(
        {
            "start": frame.apply(lambda col: col.first_valid_index()),
            "end": frame.apply(lambda col: col.last_valid_index()),
            "observations": frame.count(),
            "missing": frame.isna().sum(),
            "missing_ratio": frame.isna().mean(),
        }
    )


def hampel_outlier_flags(
    series: pd.Series,
    window: int = 21,
    n_sigmas: float = 3.0,
) -> pd.Series:
    """Flag local outliers with a rolling Hampel filter."""
    if window < 3:
        raise ValueError("window must be at least 3")
    if n_sigmas <= 0:
        raise ValueError("n_sigmas must be positive")

    rolling_median = series.rolling(window=window, center=True, min_periods=window // 2).median()
    absolute_deviation = (series - rolling_median).abs()
    rolling_mad = absolute_deviation.rolling(
        window=window,
        center=True,
        min_periods=window // 2,
    ).median()
    scaled_mad = 1.4826 * rolling_mad
    threshold = n_sigmas * scaled_mad
    return (absolute_deviation > threshold).fillna(False)
