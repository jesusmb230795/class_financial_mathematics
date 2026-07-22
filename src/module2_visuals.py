"""Pure, publication-safe time-series figures for Module 2.

The builders centralize the Module 2 visual contract so notebook cells can
select data, call one function, and interpret the result.  They do not mutate
their inputs, acquire data, display figures, or write files.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real
import textwrap
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf as sample_acf
from statsmodels.tsa.stattools import pacf as sample_pacf

from src.visual_style import (
    AMBER_DARK,
    INK,
    INLINE_FIGURE_DPI,
    MUTED_BLUE,
    TEAL,
    add_figure_note,
    matplotlib_style,
    series_color,
    series_linestyle,
    series_marker,
    style_axes,
)


Scale = Literal["decimal", "percent"]


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_text(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, name)


def _numeric_series(
    series: pd.Series,
    name: str,
    *,
    minimum_observations: int = 2,
    require_dates: bool = False,
) -> pd.Series:
    """Return an isolated numeric copy after validating the time-series boundary."""
    if not isinstance(series, pd.Series):
        raise ValueError(f"{name} must be a pandas Series")
    if series.empty:
        raise ValueError(f"{name} must contain observations")
    if isinstance(series.index, pd.MultiIndex):
        raise ValueError(f"{name} must use a one-dimensional index")
    if require_dates and not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must use a DatetimeIndex")
    if isinstance(series.index, pd.DatetimeIndex) and series.index.hasnans:
        raise ValueError(f"{name} index cannot contain missing dates")
    if series.index.has_duplicates:
        raise ValueError(f"{name} index cannot contain duplicates")
    if not series.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be sorted in increasing order")

    isolated = series.copy(deep=True)
    numeric = pd.to_numeric(isolated, errors="coerce")
    invalid_numeric = isolated.notna() & numeric.isna()
    if invalid_numeric.any():
        raise ValueError(f"{name} must contain numeric values or missing observations")

    values = numeric.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError(f"{name} cannot contain infinite values")
    if int(np.isfinite(values).sum()) < minimum_observations:
        raise ValueError(
            f"{name} must contain at least {minimum_observations} finite observations"
        )
    return numeric.astype(float)


def _validate_window(window: int, finite_observations: int) -> int:
    if isinstance(window, bool) or not isinstance(window, Integral):
        raise ValueError("rolling_window must be an integer")
    normalized = int(window)
    if normalized < 2:
        raise ValueError("rolling_window must be at least 2")
    if normalized > finite_observations:
        raise ValueError(
            "rolling_window cannot exceed the number of finite return observations"
        )
    return normalized


def _validate_lags(lags: int, observations: int, *, include_pacf: bool) -> int:
    if isinstance(lags, bool) or not isinstance(lags, Integral):
        raise ValueError("lags must be an integer")
    normalized = int(lags)
    if normalized < 1:
        raise ValueError("lags must be at least 1")
    if normalized >= observations:
        raise ValueError("lags must be smaller than the number of finite observations")
    if include_pacf and normalized >= observations / 2:
        raise ValueError(
            "lags must be less than half the number of finite observations for PACF"
        )
    return normalized


def _validate_scale(scale: str, name: str) -> Scale:
    if scale not in {"decimal", "percent"}:
        raise ValueError(f"{name} must be 'decimal' or 'percent'")
    return scale


def _sample_summary(series: pd.Series) -> str:
    observed = series.dropna()
    first = observed.index[0]
    last = observed.index[-1]
    if isinstance(observed.index, pd.DatetimeIndex):
        return (
            f"{first.strftime('%Y-%m-%d')} to {last.strftime('%Y-%m-%d')} "
            f"(inclusive; n={len(observed)})"
        )
    return (
        f"index {first} to {last} (n={len(observed)}; "
        "calendar dates not supplied)"
    )


def _metadata_note(
    *,
    sample: str,
    source: str | None,
    data_mode: str | None,
    method: str,
) -> str:
    source_text = source if source is not None else "not supplied"
    mode_text = data_mode if data_mode is not None else "not supplied"
    sections = (
        f"Sample: {sample}.",
        f"Source: {source_text}. Data mode: {mode_text}.",
        f"Method note: {method}",
    )
    return "\n".join(
        textwrap.fill(
            section,
            width=126,
            break_long_words=False,
            break_on_hyphens=False,
        )
        for section in sections
    )


def _format_time_axis(axis: Axes, index: pd.Index) -> None:
    if isinstance(index, pd.DatetimeIndex):
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        axis.set_xlabel("Date")
    else:
        axis.set_xlabel("Observation")


def _return_label(series: pd.Series) -> str:
    normalized = str(series.name or "").lower().replace("-", "_").replace(" ", "_")
    if "log" in normalized and "return" in normalized:
        return "Log return"
    if "simple" in normalized and "return" in normalized:
        return "Simple return"
    return "Return"


def _plot_acf_panel(
    axis: Axes,
    values: np.ndarray,
    *,
    lags: int,
    title: str,
    color: str,
    marker: str,
    bartlett_confint: bool,
) -> None:
    plot_acf(
        values,
        ax=axis,
        lags=lags,
        zero=False,
        alpha=0.05,
        bartlett_confint=bartlett_confint,
        title=title,
        color=color,
        marker=marker,
        markersize=4.5,
        vlines_kwargs={"colors": color, "linewidth": 1.2},
    )
    axis.set_xlabel("Lag (observations)")
    axis.set_ylabel("Correlation")
    correlations = sample_acf(values, nlags=lags, fft=True, adjusted=False)[1:]
    axis.set_ylim(_correlation_limits(correlations, len(values)))
    style_axes(axis, grid_axis="y")


def _correlation_limits(
    correlations: np.ndarray,
    observations: int,
) -> tuple[float, float]:
    """Return symmetric, labeled limits that retain values and reference bands."""
    approximate_bound = 1.96 / np.sqrt(observations)
    largest = max(
        approximate_bound,
        float(np.max(np.abs(correlations))) if len(correlations) else 0.0,
    )
    limit = min(1.0, max(0.15, 1.25 * largest))
    return -limit, limit


def build_level_return_diagnostics(
    levels: pd.Series,
    returns: pd.Series,
    *,
    level_label: str,
    level_unit: str,
    rolling_window: int = 63,
    source: str | None = None,
    data_mode: str | None = None,
) -> Figure:
    """Plot a dated level, returns, and two rolling sample diagnostics.

    ``returns`` must contain decimal returns.  The builder converts them to
    percentage points for display.  Rolling volatility is per observation and
    is deliberately not annualized because no frequency factor is accepted by
    this diagnostic.
    """
    level_label = _required_text(level_label, "level_label")
    level_unit = _required_text(level_unit, "level_unit")
    source = _optional_text(source, "source")
    data_mode = _optional_text(data_mode, "data_mode")
    level_values = _numeric_series(levels, "levels", require_dates=True)
    return_values = _numeric_series(returns, "returns", require_dates=True)
    window = _validate_window(rolling_window, int(return_values.notna().sum()))

    rolling_mean = return_values.rolling(window, min_periods=window).mean()
    rolling_volatility = return_values.rolling(window, min_periods=window).std(ddof=1)
    if not rolling_mean.notna().any() or not rolling_volatility.notna().any():
        raise ValueError(
            "returns do not contain a complete rolling window of finite observations"
        )

    return_name = _return_label(return_values)
    return_percent = return_values.multiply(100)
    rolling_mean_percent = rolling_mean.multiply(100)
    rolling_volatility_percent = rolling_volatility.multiply(100)

    with matplotlib_style():
        figure, axes = plt.subplots(
            4,
            1,
            figsize=(7.5, 12.5),
            dpi=INLINE_FIGURE_DPI,
            sharex=True,
        )
        level_axis, return_axis, mean_axis, volatility_axis = axes

        level_axis.plot(
            level_values.index,
            level_values,
            color=TEAL,
            linestyle="-",
            label=level_label,
        )
        level_axis.set_title(f"{level_label} level")
        level_axis.set_ylabel(f"Level ({level_unit})")

        return_axis.plot(
            return_percent.index,
            return_percent,
            color=MUTED_BLUE,
            linestyle="-",
            label=return_name,
        )
        return_axis.axhline(0, color=INK, linestyle="--", linewidth=0.9, alpha=0.7)
        return_axis.set_title(f"{return_name} observations")
        return_axis.set_ylabel(f"{return_name} (%)")
        return_axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))

        mean_axis.plot(
            rolling_mean_percent.index,
            rolling_mean_percent,
            color=AMBER_DARK,
            linestyle="--",
            label=f"{window}-observation mean",
        )
        mean_axis.axhline(0, color=INK, linestyle=":", linewidth=0.9, alpha=0.7)
        mean_axis.set_title(f"{window}-observation rolling sample mean")
        mean_axis.set_ylabel("Rolling mean (%)")
        mean_axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=2))

        volatility_axis.plot(
            rolling_volatility_percent.index,
            rolling_volatility_percent,
            color=series_color(3),
            linestyle="-.",
            label=f"{window}-observation volatility",
        )
        volatility_axis.set_title(f"{window}-observation rolling sample volatility")
        volatility_axis.set_ylabel("Per-observation volatility (%)")
        volatility_axis.yaxis.set_major_formatter(
            PercentFormatter(xmax=100, decimals=2)
        )

        _format_time_axis(volatility_axis, return_values.index)
        for axis in axes:
            style_axes(axis, grid_axis="y")

        figure.suptitle(
            f"{level_label} | level, return, and rolling sample diagnostics",
            x=0.01,
            ha="left",
        )
        add_figure_note(
            figure,
            _metadata_note(
                sample=(
                    f"levels {_sample_summary(level_values)}; "
                    f"returns {_sample_summary(return_values)}"
                ),
                source=source,
                data_mode=data_mode,
                method=(
                    f"Returns are decimal inputs displayed as percentages. The {window}-"
                    "observation rolling mean and sample standard deviation are in-sample "
                    "descriptions, not forecasts; volatility is not annualized."
                ),
            ),
        )
        figure.tight_layout(rect=(0, 0.105, 1, 0.965))
    return figure


def build_acf_pacf_figure(
    series: pd.Series,
    *,
    title: str,
    lags: int = 30,
    source: str | None = None,
    data_mode: str | None = None,
) -> Figure:
    """Build side-by-side ACF and PACF panels with 95% confidence bands."""
    title = _required_text(title, "title")
    source = _optional_text(source, "source")
    data_mode = _optional_text(data_mode, "data_mode")
    values = _numeric_series(series, "series", minimum_observations=4)
    clean = values.dropna()
    lag_count = _validate_lags(lags, len(clean), include_pacf=True)
    if float(np.ptp(clean.to_numpy(dtype=float))) == 0.0:
        raise ValueError("series must vary for ACF and PACF diagnostics")

    with matplotlib_style():
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(7.5, 7.2),
            dpi=INLINE_FIGURE_DPI,
            sharex=True,
        )
        acf_axis, pacf_axis = axes
        _plot_acf_panel(
            acf_axis,
            clean.to_numpy(dtype=float),
            lags=lag_count,
            title="Autocorrelation function (95% bands)",
            color=TEAL,
            marker=series_marker(0),
            bartlett_confint=False,
        )
        acf_axis.set_xlabel("")
        plot_pacf(
            clean.to_numpy(dtype=float),
            ax=pacf_axis,
            lags=lag_count,
            alpha=0.05,
            method="ywm",
            zero=False,
            title="Partial autocorrelation function (95% bands)",
            color=MUTED_BLUE,
            marker=series_marker(1),
            markersize=4.5,
            vlines_kwargs={"colors": MUTED_BLUE, "linewidth": 1.2},
        )
        pacf_axis.set_xlabel("Lag (observations)")
        pacf_axis.set_ylabel("Partial correlation")
        pacf_values = sample_pacf(
            clean.to_numpy(dtype=float),
            nlags=lag_count,
            method="ywm",
        )[1:]
        pacf_axis.set_ylim(_correlation_limits(pacf_values, len(clean)))
        style_axes(pacf_axis, grid_axis="y")

        figure.suptitle(title, x=0.01, ha="left")
        add_figure_note(
            figure,
            _metadata_note(
                sample=_sample_summary(clean),
                source=source,
                data_mode=data_mode,
                method=(
                    "ACF and PACF are sample diagnostics. Shaded intervals are the "
                    "approximate 95% confidence bands from statsmodels (alpha=0.05) "
                    "under a white-noise standard-error reference, not model-selection "
                    "guarantees. Symmetric axes adapt to the plotted values and bands; "
                    "read magnitudes from the labeled scale."
                ),
            ),
        )
        figure.tight_layout(rect=(0, 0.17, 1, 0.94))
    return figure


def build_residual_diagnostics(
    series: pd.Series,
    *,
    title: str,
    residual_unit: str = "model unit",
    lags: int = 20,
    source: str | None = None,
    data_mode: str | None = None,
) -> Figure:
    """Plot residual path, linear and squared ACFs, and a normal QQ panel."""
    title = _required_text(title, "title")
    residual_unit = _required_text(residual_unit, "residual_unit")
    source = _optional_text(source, "source")
    data_mode = _optional_text(data_mode, "data_mode")
    values = _numeric_series(series, "series", minimum_observations=4)
    clean = values.dropna()
    lag_count = _validate_lags(lags, len(clean), include_pacf=False)
    if float(np.ptp(clean.to_numpy(dtype=float))) == 0.0:
        raise ValueError("series must vary for residual diagnostics")
    if float(np.ptp(clean.pow(2).to_numpy(dtype=float))) == 0.0:
        raise ValueError("squared residuals must vary for variance diagnostics")

    with matplotlib_style():
        figure, axes = plt.subplots(
            4,
            1,
            figsize=(7.5, 12.5),
            dpi=INLINE_FIGURE_DPI,
        )
        path_axis, acf_axis, squared_acf_axis, qq_axis = axes

        path_axis.plot(
            clean.index,
            clean,
            color=TEAL,
            linestyle="-",
            label="Fitted-model residual",
        )
        path_axis.axhline(0, color=INK, linestyle="--", linewidth=0.9, alpha=0.7)
        path_axis.set_title("Residual path")
        path_axis.set_ylabel(f"Residual\n({residual_unit})")
        _format_time_axis(path_axis, clean.index)
        style_axes(path_axis, grid_axis="y")

        _plot_acf_panel(
            acf_axis,
            clean.to_numpy(dtype=float),
            lags=lag_count,
            title="Residual ACF (95% bands)",
            color=MUTED_BLUE,
            marker=series_marker(1),
            bartlett_confint=False,
        )
        _plot_acf_panel(
            squared_acf_axis,
            clean.pow(2).to_numpy(dtype=float),
            lags=lag_count,
            title="Squared-residual ACF (95% bands)",
            color=AMBER_DARK,
            marker=series_marker(2),
            bartlett_confint=False,
        )

        (theoretical, ordered), (slope, intercept, _) = stats.probplot(
            clean.to_numpy(dtype=float),
            dist="norm",
        )
        qq_axis.scatter(
            theoretical,
            ordered,
            color=series_color(3),
            edgecolor=INK,
            linewidth=0.5,
            marker=series_marker(3),
            s=24,
            alpha=0.8,
            label="Ordered residuals",
        )
        qq_axis.plot(
            theoretical,
            slope * theoretical + intercept,
            color=INK,
            linestyle="--",
            linewidth=1.2,
            label="Fitted normal reference",
        )
        qq_axis.set_title("Normal QQ plot")
        qq_axis.set_xlabel("Theoretical normal quantile")
        qq_axis.set_ylabel(f"Ordered residual\n({residual_unit})")
        qq_axis.legend(loc="best")
        style_axes(qq_axis, grid_axis="both")

        figure.suptitle(title, x=0.01, ha="left")
        add_figure_note(
            figure,
            _metadata_note(
                sample=_sample_summary(clean),
                source=source,
                data_mode=data_mode,
                method=(
                    "These are in-sample residual diagnostics, not proof of white noise "
                    "or Gaussian innovations. ACF bands are approximate 95% intervals "
                    "with alpha=0.05; squared residuals screen for remaining variance "
                    "dependence. Symmetric correlation axes adapt to the displayed "
                    f"values and bands. Residual unit: {residual_unit}."
                ),
            ),
        )
        figure.tight_layout(rect=(0, 0.11, 1, 0.965))
    return figure


def _volatility_frame(
    volatility_paths: pd.DataFrame | Mapping[str, pd.Series],
) -> pd.DataFrame:
    if isinstance(volatility_paths, pd.DataFrame):
        if volatility_paths.empty or len(volatility_paths.columns) == 0:
            raise ValueError("volatility_paths must contain at least one series")
        if volatility_paths.columns.has_duplicates:
            raise ValueError("volatility_paths labels must be unique")
        frame = volatility_paths.copy(deep=True)
    elif isinstance(volatility_paths, Mapping):
        if not volatility_paths:
            raise ValueError("volatility_paths must contain at least one series")
        isolated: dict[str, pd.Series] = {}
        for label, path in volatility_paths.items():
            clean_label = _required_text(label, "volatility path label")
            if clean_label in isolated:
                raise ValueError("volatility_paths labels must be unique")
            isolated[clean_label] = _numeric_series(
                path,
                f"volatility_paths[{clean_label!r}]",
                require_dates=True,
            )
        frame = pd.DataFrame(isolated)
    else:
        raise ValueError("volatility_paths must be a DataFrame or mapping of Series")

    if len(frame.columns) > 5:
        raise ValueError("volatility_paths supports at most five distinguishable series")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("volatility_paths must use a DatetimeIndex")
    if frame.index.hasnans:
        raise ValueError("volatility_paths index cannot contain missing dates")
    if frame.index.has_duplicates:
        raise ValueError("volatility_paths index cannot contain duplicates")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("volatility_paths index must be sorted in increasing order")

    normalized: dict[str, pd.Series] = {}
    for column in frame.columns:
        label = _required_text(column, "volatility path label")
        if label in normalized:
            raise ValueError("volatility_paths labels must be unique")
        normalized[label] = _numeric_series(
            frame[column],
            f"volatility_paths[{label!r}]",
            require_dates=True,
        )
    return pd.DataFrame(normalized, index=frame.index.copy())


def build_volatility_comparison(
    returns: pd.Series,
    volatility_paths: pd.DataFrame | Mapping[str, pd.Series],
    *,
    title: str,
    return_scale: Scale = "decimal",
    volatility_scale: Scale = "decimal",
    annualization_factor: float | None = None,
    source: str | None = None,
    data_mode: str | None = None,
) -> Figure:
    """Compare dated returns with one to five non-negative volatility paths.

    ``return_scale`` and ``volatility_scale`` describe input units.  Both
    panels display percentage points.  When supplied, ``annualization_factor``
    annualizes each volatility path with the square-root-of-time convention;
    it does not annualize returns.
    """
    title = _required_text(title, "title")
    source = _optional_text(source, "source")
    data_mode = _optional_text(data_mode, "data_mode")
    return_scale = _validate_scale(return_scale, "return_scale")
    volatility_scale = _validate_scale(volatility_scale, "volatility_scale")
    return_values = _numeric_series(returns, "returns", require_dates=True)
    paths = _volatility_frame(volatility_paths)
    if not return_values.index.equals(paths.index):
        raise ValueError("returns and volatility_paths must use the same dated index")

    path_values = paths.to_numpy(dtype=float, na_value=np.nan)
    if np.isfinite(path_values).any() and np.nanmin(path_values) < 0:
        raise ValueError("volatility_paths cannot contain negative values")

    if annualization_factor is None:
        annualization_multiplier = 1.0
        volatility_horizon = "Per-observation volatility (%)"
        annualization_note = "volatility is not annualized"
    else:
        if (
            isinstance(annualization_factor, bool)
            or not isinstance(annualization_factor, Real)
            or not np.isfinite(float(annualization_factor))
            or float(annualization_factor) <= 0
        ):
            raise ValueError("annualization_factor must be finite and positive")
        factor = float(annualization_factor)
        annualization_multiplier = np.sqrt(factor)
        volatility_horizon = "Annualized volatility (%)"
        annualization_note = f"volatility annualized with sqrt({factor:g})"

    return_multiplier = 100.0 if return_scale == "decimal" else 1.0
    volatility_multiplier = 100.0 if volatility_scale == "decimal" else 1.0
    plotted_returns = return_values.multiply(return_multiplier)
    plotted_paths = paths.multiply(volatility_multiplier * annualization_multiplier)

    with matplotlib_style():
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(7.5, 7.4),
            dpi=INLINE_FIGURE_DPI,
            sharex=True,
        )
        return_axis, volatility_axis = axes

        return_axis.plot(
            plotted_returns.index,
            plotted_returns,
            color=TEAL,
            linestyle="-",
            label=_return_label(return_values),
        )
        return_axis.axhline(0, color=INK, linestyle="--", linewidth=0.9, alpha=0.7)
        return_axis.set_title("Observed returns")
        return_axis.set_ylabel("Return (%)")
        return_axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))

        marker_interval = max(1, len(plotted_paths) // 12)
        for index, column in enumerate(plotted_paths.columns):
            volatility_axis.plot(
                plotted_paths.index,
                plotted_paths[column],
                color=series_color(index),
                linestyle=series_linestyle(index),
                marker=series_marker(index),
                markevery=marker_interval,
                markersize=3.5,
                linewidth=1.7,
                label=column,
            )
        volatility_axis.set_title("Volatility paths")
        volatility_axis.set_ylabel(volatility_horizon)
        volatility_axis.yaxis.set_major_formatter(
            PercentFormatter(xmax=100, decimals=1)
        )
        volatility_axis.legend(
            loc="best",
            ncols=min(3, len(plotted_paths.columns)),
        )

        _format_time_axis(volatility_axis, return_values.index)
        for axis in axes:
            style_axes(axis, grid_axis="y")

        figure.suptitle(title, x=0.01, ha="left")
        add_figure_note(
            figure,
            _metadata_note(
                sample=(
                    f"returns {_sample_summary(return_values)}; "
                    "volatility paths "
                    + "; ".join(
                        f"{column} {_sample_summary(paths[column])}"
                        for column in paths.columns
                    )
                ),
                source=source,
                data_mode=data_mode,
                method=(
                    f"Return input scale: {return_scale}; volatility input scale: "
                    f"{volatility_scale}; both panels display percentage points and "
                    f"{annualization_note}. Paths are in-sample estimates, not forecasts "
                    "unless the caller explicitly supplies forecast series."
                ),
            ),
        )
        figure.tight_layout(rect=(0, 0.19, 1, 0.94))
    return figure


def build_forecast_comparison(
    observed: pd.Series,
    forecasts: pd.DataFrame | Mapping[str, pd.Series],
    *,
    title: str,
    scale: Scale = "decimal",
    prediction_interval: pd.DataFrame | None = None,
    prediction_interval_label: str = "95% model-based interval",
    source: str | None = None,
    data_mode: str | None = None,
) -> Figure:
    """Compare aligned evaluation returns and cumulative absolute forecast errors.

    ``scale`` describes the input unit for both the observed series and every
    forecast.  The figure displays returns as percentages and cumulative
    absolute error as percentage points, preserving one common evaluation
    sample across all forecast paths.
    """
    title = _required_text(title, "title")
    source = _optional_text(source, "source")
    data_mode = _optional_text(data_mode, "data_mode")
    scale = _validate_scale(scale, "scale")
    observed_values = _numeric_series(
        observed,
        "observed",
        require_dates=True,
    )
    forecast_values = _volatility_frame(forecasts)
    if not observed_values.index.equals(forecast_values.index):
        raise ValueError("observed and forecasts must use the same dated index")
    if observed_values.isna().any() or forecast_values.isna().any().any():
        raise ValueError("observed and forecasts cannot contain missing values")

    interval_values: pd.DataFrame | None = None
    if prediction_interval is not None:
        prediction_interval_label = _required_text(
            prediction_interval_label,
            "prediction_interval_label",
        )
        if not isinstance(prediction_interval, pd.DataFrame):
            raise ValueError("prediction_interval must be a pandas DataFrame")
        if list(prediction_interval.columns) != ["lower", "upper"]:
            raise ValueError(
                "prediction_interval columns must be exactly ['lower', 'upper']"
            )
        interval_values = pd.DataFrame(
            {
                column: _numeric_series(
                    prediction_interval[column],
                    f"prediction_interval[{column!r}]",
                    require_dates=True,
                )
                for column in prediction_interval.columns
            },
            index=prediction_interval.index.copy(),
        )
        if not observed_values.index.equals(interval_values.index):
            raise ValueError(
                "observed and prediction_interval must use the same dated index"
            )
        if interval_values.isna().any().any():
            raise ValueError("prediction_interval cannot contain missing values")
        if (interval_values["lower"] > interval_values["upper"]).any():
            raise ValueError("prediction_interval lower values cannot exceed upper values")

    multiplier = 100.0 if scale == "decimal" else 1.0
    plotted_observed = observed_values.multiply(multiplier)
    plotted_forecasts = forecast_values.multiply(multiplier)
    plotted_interval = (
        interval_values.multiply(multiplier)
        if interval_values is not None
        else None
    )
    cumulative_absolute_error = plotted_forecasts.sub(
        plotted_observed,
        axis=0,
    ).abs().cumsum()

    with matplotlib_style():
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(7.5, 7.4),
            dpi=INLINE_FIGURE_DPI,
            sharex=True,
        )
        forecast_axis, error_axis = axes

        if plotted_interval is not None:
            forecast_axis.fill_between(
                plotted_interval.index,
                plotted_interval["lower"],
                plotted_interval["upper"],
                facecolor=MUTED_BLUE,
                edgecolor=MUTED_BLUE,
                alpha=0.16,
                hatch="//",
                linewidth=0.7,
                label=prediction_interval_label,
            )

        forecast_axis.plot(
            plotted_observed.index,
            plotted_observed,
            color=TEAL,
            linestyle="-",
            linewidth=1.5,
            label=f"Observed {_return_label(observed_values).lower()}",
        )
        marker_interval = max(1, len(plotted_forecasts) // 12)
        for index, column in enumerate(plotted_forecasts.columns, start=1):
            forecast_axis.plot(
                plotted_forecasts.index,
                plotted_forecasts[column],
                color=series_color(index),
                linestyle=series_linestyle(index),
                marker=series_marker(index),
                markevery=marker_interval,
                markersize=3.5,
                linewidth=1.5,
                label=column,
            )
            error_axis.plot(
                cumulative_absolute_error.index,
                cumulative_absolute_error[column],
                color=series_color(index),
                linestyle=series_linestyle(index),
                marker=series_marker(index),
                markevery=marker_interval,
                markersize=3.5,
                linewidth=1.7,
                label=column,
            )

        forecast_axis.axhline(
            0,
            color=INK,
            linestyle="--",
            linewidth=0.9,
            alpha=0.7,
        )
        forecast_axis.set_title("Observed evaluation returns and fixed forecasts")
        forecast_axis.set_ylabel("Return (%)")
        forecast_axis.yaxis.set_major_formatter(
            PercentFormatter(xmax=100, decimals=1)
        )
        forecast_axis.legend(
            loc="best",
            ncols=2,
            fontsize=9,
        )

        error_axis.set_title("Cumulative absolute forecast error")
        error_axis.set_ylabel("Cumulative absolute error (percentage points)")
        error_axis.legend(loc="best", ncols=min(3, len(forecast_values)))

        _format_time_axis(error_axis, observed_values.index)
        for axis in axes:
            style_axes(axis, grid_axis="y")

        figure.suptitle(title, x=0.01, ha="left")
        add_figure_note(
            figure,
            _metadata_note(
                sample=_sample_summary(observed_values),
                source=source,
                data_mode=data_mode,
                method=(
                    f"Observed and forecast input scale: {scale}; displayed returns are "
                    "percentages. Cumulative absolute error is the running sum of "
                    "|observed - forecast| in percentage points; lower is better on "
                    "this shared evaluation sample, but the measure is not trading "
                    "profit or loss. "
                    "Any shaded band is a model-based interval, not an empirical "
                    "coverage guarantee."
                ),
            ),
        )
        figure.tight_layout(rect=(0, 0.17, 1, 0.94))
    return figure


def build_innovation_tail_risk_figure(
    risk_measures: pd.DataFrame,
    *,
    alpha: float,
    degrees_of_freedom: float,
    source: str | None = None,
    data_mode: str | None = None,
) -> Figure:
    """Compare Gaussian and unit-variance Student-t tails with VaR measures.

    ``risk_measures`` must use model labels as its index and contain
    ``log_return_loss_threshold_pct`` and ``simple_return_loss_pct`` columns.
    Both columns are displayed in percentage points.
    """
    source = _optional_text(source, "source")
    data_mode = _optional_text(data_mode, "data_mode")
    if not isinstance(alpha, Real) or isinstance(alpha, bool):
        raise ValueError("alpha must be a real number")
    alpha_value = float(alpha)
    if not np.isfinite(alpha_value) or not 0 < alpha_value < 0.5:
        raise ValueError("alpha must be finite and between 0 and 0.5")
    if not isinstance(degrees_of_freedom, Real) or isinstance(
        degrees_of_freedom,
        bool,
    ):
        raise ValueError("degrees_of_freedom must be a real number")
    nu = float(degrees_of_freedom)
    if not np.isfinite(nu) or nu <= 2:
        raise ValueError("degrees_of_freedom must be finite and greater than 2")
    if not isinstance(risk_measures, pd.DataFrame):
        raise ValueError("risk_measures must be a pandas DataFrame")
    expected_columns = [
        "log_return_loss_threshold_pct",
        "simple_return_loss_pct",
    ]
    if list(risk_measures.columns) != expected_columns:
        raise ValueError(f"risk_measures columns must be exactly {expected_columns}")
    if risk_measures.empty or len(risk_measures) > 4:
        raise ValueError("risk_measures must contain between one and four models")
    if risk_measures.index.has_duplicates:
        raise ValueError("risk_measures model labels must be unique")
    labels = [_required_text(label, "risk model label") for label in risk_measures.index]
    values = risk_measures.copy(deep=True)
    values.index = labels
    for column in values.columns:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    value_array = values.to_numpy(dtype=float)
    if not np.isfinite(value_array).all() or (value_array < 0).any():
        raise ValueError("risk_measures must contain finite non-negative values")

    student_scale = np.sqrt((nu - 2.0) / nu)
    gaussian_quantile = float(stats.norm.ppf(alpha_value))
    student_quantile = float(stats.t.ppf(alpha_value, df=nu) * student_scale)
    x_min = min(-4.5, student_quantile - 1.0)
    x_values = np.linspace(x_min, 4.5, 1_200)
    gaussian_density = stats.norm.pdf(x_values)
    student_density = stats.t.pdf(x_values / student_scale, df=nu) / student_scale

    with matplotlib_style():
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(7.5, 8.4),
            dpi=INLINE_FIGURE_DPI,
        )
        density_axis, risk_axis = axes

        density_axis.plot(
            x_values,
            gaussian_density,
            color=MUTED_BLUE,
            linestyle="-",
            linewidth=1.8,
            label="Gaussian, unit variance",
        )
        density_axis.plot(
            x_values,
            student_density,
            color=series_color(3),
            linestyle="--",
            linewidth=1.8,
            label=f"Student-t, unit variance (nu={nu:.2f})",
        )
        gaussian_tail = x_values <= gaussian_quantile
        student_tail = x_values <= student_quantile
        density_axis.fill_between(
            x_values[gaussian_tail],
            0,
            gaussian_density[gaussian_tail],
            facecolor=MUTED_BLUE,
            edgecolor=MUTED_BLUE,
            alpha=0.16,
            hatch="//",
        )
        density_axis.fill_between(
            x_values[student_tail],
            0,
            student_density[student_tail],
            facecolor=series_color(3),
            edgecolor=series_color(3),
            alpha=0.12,
            hatch="xx",
        )
        density_axis.axvline(
            gaussian_quantile,
            color=MUTED_BLUE,
            linestyle=":",
            linewidth=1.2,
            label=f"Gaussian {alpha_value:.0%} quantile = {gaussian_quantile:.2f}",
        )
        density_axis.axvline(
            student_quantile,
            color=series_color(3),
            linestyle="-.",
            linewidth=1.2,
            label=f"Student-t {alpha_value:.0%} quantile = {student_quantile:.2f}",
        )
        density_axis.set_title("Standardized innovation distributions and left tails")
        density_axis.set_xlabel("Standardized innovation")
        density_axis.set_ylabel("Probability density")
        density_axis.legend(loc="best", ncols=2)
        style_axes(density_axis, grid_axis="y")

        positions = np.arange(len(values))
        bar_width = 0.36
        log_bars = risk_axis.bar(
            positions - bar_width / 2,
            values["log_return_loss_threshold_pct"],
            width=bar_width,
            color=MUTED_BLUE,
            edgecolor=INK,
            linewidth=0.7,
            hatch="//",
            label="Log-return loss threshold",
        )
        simple_bars = risk_axis.bar(
            positions + bar_width / 2,
            values["simple_return_loss_pct"],
            width=bar_width,
            color=series_color(3),
            edgecolor=INK,
            linewidth=0.7,
            hatch="xx",
            label="Exact simple-return loss",
        )
        risk_axis.bar_label(log_bars, fmt="%.3f", padding=3, fontsize=9)
        risk_axis.bar_label(simple_bars, fmt="%.3f", padding=3, fontsize=9)
        risk_axis.set_title("One-step VaR on log-return and simple-loss scales")
        risk_axis.set_ylabel("Positive loss magnitude (%)")
        risk_axis.set_xticks(positions, labels)
        risk_axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=2))
        risk_axis.legend(loc="best", ncols=2)
        style_axes(risk_axis, grid_axis="y")

        figure.suptitle(
            "Innovation-tail and one-step risk comparison",
            x=0.01,
            ha="left",
        )
        add_figure_note(
            figure,
            _metadata_note(
                sample=f"{len(values)} fitted risk specifications",
                source=source,
                data_mode=data_mode,
                method=(
                    f"Left-tail probability alpha={alpha_value:.4f}; confidence="
                    f"{1 - alpha_value:.4f}. Student-t density and quantile are scaled "
                    "to unit variance. Bars distinguish additive log-return thresholds "
                    "from exact simple-return position losses."
                ),
            ),
        )
        figure.tight_layout(rect=(0, 0.14, 1, 0.95))
    return figure


__all__ = [
    "build_acf_pacf_figure",
    "build_forecast_comparison",
    "build_innovation_tail_risk_figure",
    "build_level_return_diagnostics",
    "build_residual_diagnostics",
    "build_volatility_comparison",
]
