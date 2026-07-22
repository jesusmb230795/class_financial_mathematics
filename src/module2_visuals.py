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
    style_axes(axis, grid_axis="y")


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
            2,
            2,
            figsize=(13.5, 8.8),
            dpi=INLINE_FIGURE_DPI,
        )
        level_axis, return_axis, mean_axis, volatility_axis = axes.ravel()

        level_axis.plot(
            level_values.index,
            level_values,
            color=TEAL,
            linestyle="-",
            label=level_label,
        )
        level_axis.set_title(f"{level_label} level")
        level_axis.set_ylabel(f"Level ({level_unit})")
        level_axis.legend(loc="best")

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
        return_axis.legend(loc="best")

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
        mean_axis.legend(loc="best")

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
        volatility_axis.legend(loc="best")

        for axis, index in (
            (level_axis, level_values.index),
            (return_axis, return_values.index),
            (mean_axis, return_values.index),
            (volatility_axis, return_values.index),
        ):
            _format_time_axis(axis, index)
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
        figure.tight_layout(rect=(0, 0.13, 1, 0.95))
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
            1,
            2,
            figsize=(12.5, 4.8),
            dpi=INLINE_FIGURE_DPI,
        )
        acf_axis, pacf_axis = axes
        _plot_acf_panel(
            acf_axis,
            clean.to_numpy(dtype=float),
            lags=lag_count,
            title="Autocorrelation function (95% bands)",
            color=TEAL,
            marker=series_marker(0),
            bartlett_confint=True,
        )
        plot_pacf(
            clean.to_numpy(dtype=float),
            ax=pacf_axis,
            lags=lag_count,
            alpha=0.05,
            method="ywm",
            title="Partial autocorrelation function (95% bands)",
            color=MUTED_BLUE,
            marker=series_marker(1),
            markersize=4.5,
            vlines_kwargs={"colors": MUTED_BLUE, "linewidth": 1.2},
        )
        pacf_axis.set_xlabel("Lag (observations)")
        pacf_axis.set_ylabel("Partial correlation")
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
                    "approximate 95% confidence bands from statsmodels (alpha=0.05), "
                    "not model-selection guarantees."
                ),
            ),
        )
        figure.tight_layout(rect=(0, 0.22, 1, 0.91))
    return figure


def build_residual_diagnostics(
    series: pd.Series,
    *,
    title: str,
    lags: int = 20,
    source: str | None = None,
    data_mode: str | None = None,
) -> Figure:
    """Plot residual path, linear and squared ACFs, and a normal QQ panel."""
    title = _required_text(title, "title")
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
            2,
            2,
            figsize=(12.5, 8.6),
            dpi=INLINE_FIGURE_DPI,
        )
        path_axis, acf_axis, squared_acf_axis, qq_axis = axes.ravel()

        path_axis.plot(
            clean.index,
            clean,
            color=TEAL,
            linestyle="-",
            label="Fitted-model residual",
        )
        path_axis.axhline(0, color=INK, linestyle="--", linewidth=0.9, alpha=0.7)
        path_axis.set_title("Residual path")
        path_axis.set_ylabel("Residual (model units)")
        path_axis.legend(loc="best")
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
        qq_axis.set_ylabel("Ordered residual (model units)")
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
                    "dependence."
                ),
            ),
        )
        figure.tight_layout(rect=(0, 0.16, 1, 0.94))
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
            figsize=(12.5, 7.8),
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
        return_axis.legend(loc="best")

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

        for axis in axes:
            _format_time_axis(axis, return_values.index)
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


__all__ = [
    "build_acf_pacf_figure",
    "build_level_return_diagnostics",
    "build_residual_diagnostics",
    "build_volatility_comparison",
]
