"""Market data quality helpers used across class notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _drop_all_missing_rows(
    data: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """Drop rows that contain no usable return without changing the container type."""
    if isinstance(data, pd.Series):
        return data.dropna()
    return data.dropna(how="all")


def _numeric_data(
    data: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """Coerce a price object to numeric values while preserving its pandas shape."""
    if isinstance(data, pd.Series):
        return pd.to_numeric(data, errors="raise")
    return data.apply(pd.to_numeric, errors="raise")


def _longest_constant_run(series: pd.Series) -> int:
    """Return the longest consecutive run of one non-missing value."""
    longest = 0
    current = 0
    previous: object | None = None
    has_previous = False

    for value in series.array:
        if pd.isna(value):
            current = 0
            previous = None
            has_previous = False
            continue
        if has_previous and value == previous:
            current += 1
        else:
            current = 1
        longest = max(longest, current)
        previous = value
        has_previous = True
    return longest


def _missing_expected_observations(
    series: pd.Series,
    expected_index: pd.DatetimeIndex,
) -> int:
    """Count expected dates without a non-missing observation for one series."""
    observed_index = series.dropna().index.unique()
    return int(len(expected_index.difference(observed_index)))


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
                "methodology_break": None,
                "methodology_note": None,
            },
            {
                "series": "SF60648",
                "meaning": "28-day TIIE",
                "course_use": "interbank and floating-rate examples",
                "methodology_break": "2025-01-01",
                "methodology_note": (
                    "Banco de Mexico changed the 28-day TIIE methodology effective "
                    "2025-01-01; comparisons across the break are not homogeneous."
                ),
            },
            {
                "series": "SF60633",
                "meaning": "28-day CETES",
                "course_use": "short sovereign-rate reference",
                "methodology_break": None,
                "methodology_note": None,
            },
            {
                "series": "SF43718",
                "meaning": "USD/MXN FIX",
                "course_use": "FX risk and macro dashboards",
                "methodology_break": None,
                "methodology_note": None,
            },
            {
                "series": "SP68257",
                "meaning": "UDI",
                "course_use": "inflation-linked valuation context",
                "methodology_break": None,
                "methodology_note": None,
            },
        ]
    )


def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute simple returns without implicitly filling missing price levels."""
    numeric = _numeric_data(prices)
    return _drop_all_missing_rows(numeric.pct_change(fill_method=None))


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute log returns from strictly positive levels, without implicit filling."""
    numeric = _numeric_data(prices)
    non_positive = numeric.le(0) & numeric.notna()
    if bool(non_positive.to_numpy().any()):
        raise ValueError("log returns require strictly positive, non-missing price levels")
    return _drop_all_missing_rows(np.log(numeric / numeric.shift(1)))


def annualized_volatility(
    returns: pd.Series | pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.Series | float:
    """Annualize return volatility using a square-root-of-time convention."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    volatility = returns.std() * np.sqrt(periods_per_year)
    if isinstance(volatility, pd.Series):
        return volatility
    return float(volatility)


def annualized_log_change_from_levels(
    series: pd.Series,
    days_per_year: float = 365.2425,
) -> float:
    """Annualize endpoint log growth for a positive level series.

    This is a descriptive change in a level between its first and last observed
    dates, not an asset-return or total-return contract.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must use a DatetimeIndex")
    if days_per_year <= 0:
        raise ValueError("days_per_year must be positive")

    observed = pd.to_numeric(series, errors="raise").dropna().sort_index()
    if len(observed) < 2:
        raise ValueError("series must contain at least two observed levels")
    if observed.index.has_duplicates:
        raise ValueError("series contains duplicate observation dates")
    if (observed <= 0).any():
        raise ValueError("annualized log change requires strictly positive levels")

    elapsed_days = (observed.index[-1] - observed.index[0]).total_seconds() / 86_400
    if elapsed_days <= 0:
        raise ValueError("first and last observations must have distinct dates")
    return float(np.log(observed.iloc[-1] / observed.iloc[0]) * days_per_year / elapsed_days)


def align_to_business_calendar(
    data: pd.Series | pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    fill_method: str | None = None,
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


def data_quality_report(
    data: pd.Series | pd.DataFrame,
    *,
    expected_frequency: str | None = None,
) -> pd.DataFrame:
    """Summarize structural and value-level quality for time-indexed data.

    Calendar gaps are only evaluated when ``expected_frequency`` is supplied.
    ``missing`` counts nulls on stored rows, ``missing_index_dates`` counts
    expected dates absent from the frame, and ``missing_expected_observations``
    counts expected dates without a non-null value for each series. The report
    does not create or imply an imputation mask.
    """
    frame = data.to_frame(name=data.name or "value") if isinstance(data, pd.Series) else data
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("data must use a DatetimeIndex")
    if expected_frequency is not None:
        try:
            pd.date_range("2000-01-01", periods=2, freq=expected_frequency)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid expected_frequency: {expected_frequency!r}") from exc

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    non_numeric = (frame.notna() & numeric.isna()).sum()
    non_finite = numeric.apply(
        lambda column: int(np.isinf(column.to_numpy(dtype=float, na_value=np.nan)).sum())
    )
    duplicate_rows = int(frame.index.duplicated(keep=False).sum())
    report = pd.DataFrame(
        {
            "start": frame.apply(lambda column: column.first_valid_index()),
            "end": frame.apply(lambda column: column.last_valid_index()),
            "observations": frame.count(),
            "missing": frame.isna().sum(),
            "missing_ratio": frame.isna().mean(),
            "non_numeric": non_numeric,
            "non_finite": non_finite,
            "duplicate_timestamp_rows": duplicate_rows,
            "longest_constant_run": frame.apply(_longest_constant_run),
            "calendar_checked": expected_frequency is not None,
            "expected_frequency": expected_frequency,
        }
    )
    if expected_frequency is None:
        report["missing_index_dates"] = pd.Series(
            pd.array([pd.NA] * len(report), dtype="Int64"),
            index=report.index,
        )
        report["missing_expected_observations"] = pd.Series(
            pd.array([pd.NA] * len(report), dtype="Int64"),
            index=report.index,
        )
    else:
        if frame.empty:
            expected_index = pd.DatetimeIndex([])
        else:
            expected_index = pd.date_range(
                start=frame.index.min(),
                end=frame.index.max(),
                freq=expected_frequency,
            )
        missing_index_dates = int(len(expected_index.difference(frame.index.unique())))
        report["missing_index_dates"] = pd.Series(
            pd.array([missing_index_dates] * len(report), dtype="Int64"),
            index=report.index,
        )
        report["missing_expected_observations"] = frame.apply(
            _missing_expected_observations,
            expected_index=expected_index,
        ).astype("Int64")
    return report


def hampel_outlier_flags(
    series: pd.Series,
    window: int = 21,
    n_sigmas: float = 3.0,
) -> pd.Series:
    """Flag local outliers with the canonical rolling-median Hampel rule."""
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if window < 3:
        raise ValueError("window must be at least 3")
    if n_sigmas <= 0:
        raise ValueError("n_sigmas must be positive")

    numeric = pd.to_numeric(series, errors="coerce")
    min_periods = (window + 1) // 2
    rolling = numeric.rolling(
        window=window,
        center=True,
        min_periods=min_periods,
    )
    rolling_median = rolling.median()
    rolling_mad = rolling.apply(
        lambda values: float(np.median(np.abs(values - np.median(values)))),
        raw=True,
    )
    absolute_deviation = (numeric - rolling_median).abs()
    scaled_mad = 1.4826 * rolling_mad
    threshold = n_sigmas * scaled_mad
    return (absolute_deviation > threshold).fillna(False)
