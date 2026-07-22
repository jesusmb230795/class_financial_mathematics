"""Deterministic Matplotlib fallbacks for published dashboard notebooks.

The interactive notebooks keep their Plotly figures for local exploration.
Publication builds use these builders so MyST-NB receives a standard
``image/png`` representation without a browser runtime or remote assets.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from src.market_data import returns_from_prices
from src.portfolio_optimization import (
    efficient_frontier_variance,
    global_minimum_variance_weights,
    portfolio_return,
    portfolio_volatility,
    tangency_weights,
)
from src.visual_style import (
    INLINE_FIGURE_DPI,
    PLOT_BACKGROUND,
    SEMANTIC_COLORS,
    add_figure_note,
    matplotlib_style,
    style_axes,
)


_RATE_CHANGE_SERIES = frozenset(
    {
        "banxico_target_rate",
        "cetes_28d",
        "tiie_28d",
        "mexico_inflation",
        "us_10y",
    }
)
_MATPLOTLIB_THRESHOLD_STYLES = ("--", ":", "-.", (0, (5, 2)), (0, (3, 1, 1, 1)))


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    """Raise an actionable error when a dashboard input is incomplete."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{name} must contain at least one observation")


def _require_rolling_window(rolling_window: int) -> None:
    """Validate a rolling-window input shared by the dashboard builders."""
    if rolling_window < 2:
        raise ValueError("rolling_window must be at least 2")


def _display_label(value: str) -> str:
    """Convert a data-column identifier into a compact chart label."""
    known_labels = {
        "banxico_target_rate": "Banxico target rate",
        "cetes_28d": "CETES 28-day rate",
        "cetes_28d_carry": "CETES 28-day carry",
        "mexico_inflation": "Mexico inflation",
        "mxn_per_usd_fix": "Banxico FIX (MXN per USD)",
        "policy_rate_carry": "Policy-rate carry",
        "tiie_28d": "TIIE 28-day rate",
        "tiie_28d_carry": "TIIE 28-day carry",
        "udi": "UDI",
        "us_10y": "US 10-year yield",
        "usd_mxn": "USD/MXN",
    }
    if value in known_labels:
        return known_labels[value]
    if value.isupper():
        return value
    return value.replace("_", " ").title()


def _macro_change_series(macro: pd.DataFrame, column: str) -> tuple[pd.Series, str]:
    """Return the economically appropriate change series and its display label."""
    values = macro[column]
    if column in _RATE_CHANGE_SERIES:
        return values.diff().multiply(100), "percentage-point change"

    observed = values.dropna()
    if observed.empty:
        raise ValueError(f"macro series {column!r} must contain a finite observation")
    if (observed <= 0).any():
        raise ValueError(f"macro series {column!r} must be positive to compute percent changes")
    return values.pct_change(fill_method=None), "percent change"


def _native_macro_axis_title(column: str) -> str:
    """Return an explicit unit label for an unnormalized macro series."""
    label = _display_label(column)
    if column in _RATE_CHANGE_SERIES:
        return f"{label} (%)"
    native_units = {
        "udi": "MXN per UDI",
        "usd_mxn": "MXN per USD",
    }
    unit = native_units.get(column, "source units")
    return f"{label} ({unit})"


def _context_note(
    data: pd.Series | pd.DataFrame,
    *,
    data_mode: str | None = None,
) -> str:
    """Build a source/sample note only from supplied data and metadata."""
    attrs = data.attrs
    parts: list[str] = []
    mode = data_mode or attrs.get("data_mode")
    if mode:
        parts.append(f"Data mode: {mode}")

    if isinstance(data.index, pd.DatetimeIndex) and len(data.index):
        start = data.index.min().strftime("%Y-%m-%d")
        end = data.index.max().strftime("%Y-%m-%d")
    else:
        start = attrs.get("start")
        end = attrs.get("end")
    if start is not None and end is not None:
        parts.append(f"Sample: {start} to {end}")

    source = attrs.get("sources") or attrs.get("source")
    if source:
        parts.append(f"Source: {source}")
    method = attrs.get("method")
    if method:
        parts.append(f"Method: {method}")
    return " | ".join(parts)


def _add_context_note(
    figure: Figure,
    data: pd.Series | pd.DataFrame,
    *,
    data_mode: str | None = None,
) -> None:
    note = _context_note(data, data_mode=data_mode)
    if note:
        add_figure_note(figure, note)
        layout_engine = figure.get_layout_engine()
        if layout_engine is not None:
            layout_engine.set(rect=(0, 0.065, 1, 0.935))


def _format_date_axis(axis: Axes) -> None:
    """Use compact, non-overlapping dates on publication-width panels."""
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def build_macro_dashboard_fallback(
    macro: pd.DataFrame,
    primary_series: str,
    comparison_series: str,
    *,
    normalize: bool = False,
    rolling_window: int = 12,
    data_mode: str = "offline",
) -> Figure:
    """Build the publication-safe macro comparison and correlation figure."""
    _require_columns(macro, (primary_series, comparison_series), "macro")
    _require_rolling_window(rolling_window)
    if primary_series == comparison_series:
        raise ValueError("primary_series and comparison_series must be different")
    if normalize and ({primary_series, comparison_series} & _RATE_CHANGE_SERIES):
        raise ValueError(
            "normalize=True is only available for positive level series; "
            "rates and inflation must remain in native percentage units"
        )

    primary_values = macro[primary_series].copy()
    comparison_values = macro[comparison_series].copy()
    if normalize:
        bases = (primary_values.iloc[0], comparison_values.iloc[0])
        invalid_labels = [
            label
            for label, base in zip(
                (primary_series, comparison_series),
                bases,
                strict=True,
            )
            if not np.isfinite(base) or base == 0
        ]
        if invalid_labels:
            raise ValueError(
                "Cannot normalize macro series with a missing or zero first "
                f"observation: {sorted(set(invalid_labels))}"
            )
        primary_values = primary_values.divide(bases[0]).multiply(100)
        comparison_values = comparison_values.divide(bases[1]).multiply(100)

    primary_changes, primary_transformation = _macro_change_series(
        macro,
        primary_series,
    )
    comparison_changes, comparison_transformation = _macro_change_series(
        macro,
        comparison_series,
    )
    rolling_correlation = primary_changes.rolling(rolling_window).corr(comparison_changes)
    primary_label = _display_label(primary_series)
    comparison_label = _display_label(comparison_series)
    correlation_title = (
        f"{rolling_window}-month rolling correlation\n"
        f"{primary_label} ({primary_transformation}) vs "
        f"{comparison_label} ({comparison_transformation})"
    )

    with matplotlib_style():
        row_count = 2 if normalize else 3
        figure, axes = plt.subplots(
            row_count,
            1,
            figsize=(7.5, 7 if normalize else 10),
            dpi=INLINE_FIGURE_DPI,
            sharex=True,
            constrained_layout=True,
        )
        if normalize:
            primary_axis = comparison_axis = axes[0]
            correlation_axis = axes[1]
        else:
            primary_axis, comparison_axis, correlation_axis = axes

        primary_axis.plot(
            macro.index,
            primary_values,
            color=SEMANTIC_COLORS["primary"],
            linestyle="-",
            linewidth=1.8,
            label=primary_label,
        )
        comparison_axis.plot(
            macro.index,
            comparison_values,
            color=SEMANTIC_COLORS["comparison"],
            linestyle="--",
            linewidth=1.8,
            label=comparison_label,
        )
        if normalize:
            comparison_axis.set_title("Macro comparison (base = 100)")
            comparison_axis.set_ylabel("Index (base = 100)")
            comparison_axis.legend(frameon=False, ncols=2)
        else:
            primary_axis.set_title(f"{primary_label} in native units")
            primary_axis.set_ylabel(_native_macro_axis_title(primary_series))
            if primary_series in _RATE_CHANGE_SERIES:
                primary_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
            primary_axis.legend(frameon=False)
            comparison_axis.set_title(f"{comparison_label} in native units")
            comparison_axis.set_ylabel(_native_macro_axis_title(comparison_series))
            if comparison_series in _RATE_CHANGE_SERIES:
                comparison_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
            comparison_axis.legend(frameon=False)

        correlation_axis.plot(
            rolling_correlation.index,
            rolling_correlation,
            color=SEMANTIC_COLORS["highlight"],
            linestyle=":",
            linewidth=1.7,
            label="Rolling correlation",
        )
        correlation_axis.set_title(correlation_title)
        correlation_axis.set_xlabel("Date")
        correlation_axis.set_ylabel("Correlation coefficient")
        correlation_axis.set_ylim(-1.05, 1.05)
        correlation_axis.legend(frameon=False)
        _format_date_axis(correlation_axis)

        style_axes(primary_axis, grid_axis="y")
        if not normalize:
            style_axes(comparison_axis, grid_axis="y")
        style_axes(correlation_axis, grid_axis="y", show_zero_line=True)
        figure.suptitle(
            f"Macro dashboard | {primary_label} vs {comparison_label} | {data_mode} data",
            fontsize=14,
        )
        _add_context_note(figure, macro, data_mode=data_mode)
    return figure


def build_return_explorer_fallback(
    prices: pd.DataFrame,
    asset: str,
    *,
    return_method: str = "log",
    rolling_window: int = 63,
    data_mode: str = "offline",
) -> Figure:
    """Build a publication-safe price, return, drawdown, and risk figure."""
    _require_columns(prices, (asset,), "prices")
    _require_rolling_window(rolling_window)
    if return_method not in {"log", "simple"}:
        raise ValueError("return_method must be 'log' or 'simple'")

    asset_prices = prices[asset].copy()
    observed_prices = asset_prices.replace([np.inf, -np.inf], np.nan).dropna()
    if len(observed_prices) < 2:
        raise ValueError(f"prices must contain at least two finite values for {asset!r}")
    if (observed_prices <= 0).any():
        raise ValueError(f"adjusted prices must be positive for asset {asset!r}")

    asset_returns = returns_from_prices(
        prices.loc[:, [asset]],
        method=return_method,
    )[asset].dropna()
    if asset_returns.empty:
        raise ValueError(f"prices do not produce returns for asset {asset!r}")
    cumulative = asset_prices.divide(observed_prices.iloc[0]).subtract(1)
    asset_drawdown = asset_prices.divide(asset_prices.cummax()).subtract(1)
    rolling_volatility = asset_returns.rolling(rolling_window).std() * np.sqrt(252)
    return_label = f"{return_method} return"

    with matplotlib_style():
        figure, axes = plt.subplots(
            4,
            1,
            figsize=(7.5, 14),
            dpi=INLINE_FIGURE_DPI,
            constrained_layout=True,
        )
        price_axis, performance_axis, volatility_axis, histogram_axis = axes

        price_axis.plot(
            asset_prices.index,
            asset_prices,
            color=SEMANTIC_COLORS["primary"],
            linestyle="-",
            linewidth=1.7,
            label=_display_label(asset),
        )
        price_axis.set_title("Adjusted price")
        price_axis.set_xlabel("Date")
        price_axis.set_ylabel("Adjusted price (USD)")
        price_axis.legend(frameon=False)
        _format_date_axis(price_axis)

        performance_axis.plot(
            cumulative.index,
            cumulative,
            color=SEMANTIC_COLORS["primary"],
            linestyle="-",
            linewidth=1.7,
            label="Cumulative change from adjusted price",
        )
        performance_axis.plot(
            asset_drawdown.index,
            asset_drawdown,
            color=SEMANTIC_COLORS["negative"],
            linestyle="--",
            linewidth=1.4,
            label="Adjusted-price drawdown",
        )
        performance_axis.set_title("Cumulative adjusted-close change and drawdown")
        performance_axis.set_xlabel("Date")
        performance_axis.set_ylabel("Adjusted-close change / drawdown")
        performance_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        performance_axis.legend(frameon=False)
        _format_date_axis(performance_axis)

        volatility_axis.plot(
            rolling_volatility.index,
            rolling_volatility,
            color=SEMANTIC_COLORS["highlight"],
            linestyle="-.",
            linewidth=1.7,
            label=f"{rolling_window}-trading-day rolling volatility",
        )
        volatility_axis.set_title(f"{rolling_window}-trading-day rolling volatility")
        volatility_axis.set_xlabel("Date")
        volatility_axis.set_ylabel("Annualized volatility (252 trading days/year)")
        volatility_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        volatility_axis.legend(frameon=False)
        _format_date_axis(volatility_axis)

        histogram_axis.hist(
            asset_returns,
            bins=50,
            color=SEMANTIC_COLORS["comparison"],
            edgecolor=PLOT_BACKGROUND,
            linewidth=0.5,
            alpha=0.85,
        )
        histogram_axis.axvline(
            0,
            color=SEMANTIC_COLORS["reference"],
            linewidth=0.8,
            alpha=0.65,
        )
        histogram_axis.set_title(f"{return_method.title()}-return histogram")
        histogram_axis.set_xlabel(f"Daily {return_label}")
        histogram_axis.set_ylabel("Observations")
        histogram_axis.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))

        style_axes(price_axis, grid_axis="y")
        style_axes(performance_axis, grid_axis="y", show_zero_line=True)
        style_axes(volatility_axis, grid_axis="y")
        style_axes(histogram_axis, grid_axis="y")
        figure.suptitle(
            f"Return explorer: {_display_label(asset)} | daily {return_label}s | {data_mode} data",
            fontsize=14,
        )
        _add_context_note(figure, prices, data_mode=data_mode)
    return figure


def build_volatility_dashboard_fallback(
    filtered: pd.DataFrame,
    *,
    asset: str,
    persistence: float,
    return_axis_label: str = "Daily return",
    volatility_axis_label: str = "Daily volatility",
) -> Figure:
    """Build publication-safe return and volatility panels with explicit units."""
    _require_columns(
        filtered,
        ("return", "garch_filtered_volatility", "ewma_volatility"),
        "filtered",
    )
    if not np.isfinite(persistence):
        raise ValueError("persistence must be finite")
    if not isinstance(return_axis_label, str) or not return_axis_label.strip():
        raise ValueError("return_axis_label must be a non-empty string")
    if not isinstance(volatility_axis_label, str) or not volatility_axis_label.strip():
        raise ValueError("volatility_axis_label must be a non-empty string")

    with matplotlib_style():
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(7.5, 7),
            dpi=INLINE_FIGURE_DPI,
            sharex=True,
            constrained_layout=True,
        )
        returns_axis, volatility_axis = axes

        returns_axis.plot(
            filtered.index,
            filtered["return"],
            color=SEMANTIC_COLORS["comparison"],
            linestyle="-",
            linewidth=1.2,
            label=f"{_display_label(asset)} return",
        )
        returns_axis.set_title("Observed returns")
        returns_axis.set_ylabel(return_axis_label.strip())
        returns_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))

        volatility_axis.plot(
            filtered.index,
            filtered["garch_filtered_volatility"],
            color=SEMANTIC_COLORS["highlight"],
            linestyle="--",
            linewidth=1.7,
            label="GARCH-filtered volatility",
        )
        volatility_axis.plot(
            filtered.index,
            filtered["ewma_volatility"],
            color=SEMANTIC_COLORS["primary"],
            linestyle="-.",
            linewidth=1.7,
            label="EWMA volatility",
        )
        volatility_axis.set_title("GARCH-filtered volatility versus EWMA volatility")
        volatility_axis.set_xlabel("Date")
        volatility_axis.set_ylabel(volatility_axis_label.strip())
        volatility_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        volatility_axis.legend(frameon=False, ncols=2)
        _format_date_axis(volatility_axis)

        style_axes(returns_axis, grid_axis="y", show_zero_line=True)
        style_axes(volatility_axis, grid_axis="y")
        figure.suptitle(
            "Observed-return volatility dashboard | "
            f"{_display_label(asset)} | alpha + beta = {persistence:.3f}",
            fontsize=14,
        )
        _add_context_note(figure, filtered)
    return figure


def build_bond_sensitivity_dashboard_fallback(
    scenario: pd.DataFrame,
    *,
    price: float,
    macaulay_duration: float,
    modified_duration: float,
    convexity: float,
) -> Figure:
    """Build publication-safe exact and approximated bond repricing curves."""
    _require_columns(
        scenario,
        (
            "shock_bps",
            "exact_price",
            "duration_price",
            "duration_convexity_price",
        ),
        "scenario",
    )
    metrics = np.asarray(
        [price, macaulay_duration, modified_duration, convexity],
        dtype=float,
    )
    if not np.all(np.isfinite(metrics)):
        raise ValueError("bond price and sensitivity metrics must be finite")

    with matplotlib_style():
        figure, axis = plt.subplots(
            figsize=(10, 5.5),
            dpi=INLINE_FIGURE_DPI,
            constrained_layout=True,
        )
        for column, label, color, linestyle in (
            (
                "exact_price",
                "Exact repricing",
                SEMANTIC_COLORS["reference"],
                "-",
            ),
            (
                "duration_price",
                "Duration approximation",
                SEMANTIC_COLORS["comparison"],
                "--",
            ),
            (
                "duration_convexity_price",
                "Duration-convexity approximation",
                SEMANTIC_COLORS["primary"],
                "-.",
            ),
        ):
            axis.plot(
                scenario["shock_bps"],
                scenario[column],
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                label=label,
            )
        axis.axvline(
            0,
            color=SEMANTIC_COLORS["reference"],
            linewidth=0.8,
            alpha=0.65,
        )
        axis.set_title("Exact repricing versus local approximations")
        axis.set_xlabel("Parallel yield shock (basis points)")
        axis.set_ylabel("Bond price")
        axis.legend(frameon=False)
        style_axes(axis, grid_axis="y")
        figure.suptitle(
            f"Price {price:.2f} | Macaulay duration {macaulay_duration:.2f} | "
            f"Modified duration {modified_duration:.2f} | Convexity {convexity:.2f}",
            fontsize=14,
        )
        _add_context_note(figure, scenario)
    return figure


def build_black_scholes_dashboard_fallback(
    spot_grid: Sequence[float],
    payoff: Sequence[float],
    option_values: Sequence[float],
    *,
    current_spot: float,
    strike: float,
    title: str,
) -> Figure:
    """Build publication-safe payoff and Black-Scholes value curves."""
    spot_values = np.asarray(spot_grid, dtype=float)
    payoff_values = np.asarray(payoff, dtype=float)
    model_values = np.asarray(option_values, dtype=float)
    if (
        spot_values.ndim != 1
        or len(spot_values) == 0
        or payoff_values.shape != spot_values.shape
        or model_values.shape != spot_values.shape
    ):
        raise ValueError(
            "spot_grid, payoff, and option_values must be non-empty "
            "one-dimensional sequences of equal length"
        )
    if not np.all(
        np.isfinite(
            np.concatenate(
                (
                    spot_values,
                    payoff_values,
                    model_values,
                    np.asarray([current_spot, strike], dtype=float),
                )
            )
        )
    ):
        raise ValueError("option dashboard inputs must be finite")

    with matplotlib_style():
        figure, axis = plt.subplots(
            figsize=(10, 5.8),
            dpi=INLINE_FIGURE_DPI,
            constrained_layout=True,
        )
        axis.plot(
            spot_values,
            payoff_values,
            color=SEMANTIC_COLORS["reference"],
            linestyle="--",
            linewidth=1.8,
            label="Payoff at maturity",
        )
        axis.plot(
            spot_values,
            model_values,
            color=SEMANTIC_COLORS["primary"],
            linestyle="-",
            linewidth=1.8,
            label="Black-Scholes value",
        )
        axis.axvline(
            current_spot,
            color=SEMANTIC_COLORS["comparison"],
            linestyle="--",
            linewidth=1.2,
            label="Current spot",
        )
        axis.axvline(
            strike,
            color=SEMANTIC_COLORS["highlight"],
            linestyle=":",
            linewidth=1.4,
            label="Strike",
        )
        axis.set_title("Payoff and model value")
        axis.set_xlabel("Underlying price")
        axis.set_ylabel("Option value")
        axis.legend(frameon=False, ncols=2)
        style_axes(axis, grid_axis="y")
        figure.suptitle(title, fontsize=14, wrap=True)
    return figure


def build_var_cvar_dashboard_fallback(
    returns: pd.Series,
    risk_metrics: pd.Series,
) -> Figure:
    """Build a publication-safe loss-distribution view with VaR/ES thresholds."""
    observed_returns = pd.to_numeric(returns, errors="coerce")
    if observed_returns.empty or not np.all(np.isfinite(observed_returns)):
        raise ValueError("returns must contain finite observations")

    metrics = pd.to_numeric(risk_metrics, errors="coerce")
    if np.isinf(metrics.to_numpy(dtype=float)).any():
        raise ValueError("risk_metrics must not contain infinite values")
    finite_metrics = metrics[np.isfinite(metrics)]
    if finite_metrics.empty:
        raise ValueError("risk_metrics must contain at least one finite estimate")
    if (finite_metrics < 0).any():
        raise ValueError("risk_metrics must be non-negative loss estimates")

    threshold_colors = (
        SEMANTIC_COLORS["negative"],
        SEMANTIC_COLORS["highlight"],
        SEMANTIC_COLORS["primary"],
        SEMANTIC_COLORS["comparison"],
        SEMANTIC_COLORS["reference"],
    )
    with matplotlib_style():
        figure, axis = plt.subplots(
            figsize=(10, 5.8),
            dpi=INLINE_FIGURE_DPI,
            constrained_layout=True,
        )
        axis.hist(
            observed_returns,
            bins=70,
            color=SEMANTIC_COLORS["comparison"],
            edgecolor=PLOT_BACKGROUND,
            linewidth=0.5,
            alpha=0.8,
            label="Observed returns",
        )
        for index, (metric, value) in enumerate(finite_metrics.items()):
            axis.axvline(
                -value,
                color=threshold_colors[index % len(threshold_colors)],
                linestyle=_MATPLOTLIB_THRESHOLD_STYLES[index % len(_MATPLOTLIB_THRESHOLD_STYLES)],
                linewidth=1.4,
                label=metric.replace("_", " "),
            )
        axis.set_title("Observed return distribution and loss thresholds")
        axis.set_xlabel("Daily return")
        axis.set_ylabel("Frequency")
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
        axis.legend(frameon=False, fontsize=9, ncols=2)
        style_axes(axis, grid_axis="y")
        figure.suptitle(
            "VaR and Expected Shortfall from official returns "
            f"| lookback = {len(observed_returns)}",
            fontsize=14,
        )
        _add_context_note(figure, returns)
    return figure


def build_efficient_frontier_dashboard_fallback(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    *,
    risk_free_rate: float,
    points: int = 80,
) -> tuple[Figure, pd.DataFrame]:
    """Build a publication-safe efficient frontier and reference portfolios."""
    if points < 2:
        raise ValueError("points must be at least 2")
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite")
    if list(covariance.index) != list(covariance.columns):
        raise ValueError("covariance index and columns must have identical order")
    if set(expected_returns.index) != set(covariance.columns):
        raise ValueError("expected_returns index must match the covariance asset labels")
    aligned_returns = expected_returns.reindex(covariance.columns)

    target_returns = np.linspace(
        aligned_returns.min() * 0.9,
        aligned_returns.max() * 1.1,
        points,
    )
    frontier_variance = efficient_frontier_variance(
        target_returns,
        aligned_returns,
        covariance,
    )
    if np.min(frontier_variance) < -1e-10:
        raise ValueError("efficient frontier produced materially negative variance")
    frontier_volatility = np.sqrt(np.maximum(frontier_variance, 0))
    gmvp = global_minimum_variance_weights(covariance)
    tangency = tangency_weights(
        aligned_returns,
        covariance,
        risk_free_rate=risk_free_rate,
    )
    gmvp_volatility = portfolio_volatility(gmvp, covariance)
    gmvp_return = portfolio_return(gmvp, aligned_returns)
    tangency_volatility = portfolio_volatility(tangency, covariance)
    tangency_return = portfolio_return(tangency, aligned_returns)
    tangency_sharpe = (tangency_return - risk_free_rate) / tangency_volatility

    with matplotlib_style():
        figure, axis = plt.subplots(
            figsize=(9.5, 6),
            dpi=INLINE_FIGURE_DPI,
            constrained_layout=True,
        )
        axis.plot(
            frontier_volatility,
            target_returns,
            color=SEMANTIC_COLORS["primary"],
            linestyle="-",
            linewidth=2,
            label="Efficient frontier",
        )
        axis.scatter(
            [gmvp_volatility],
            [gmvp_return],
            color=SEMANTIC_COLORS["highlight"],
            marker="o",
            s=70,
            zorder=3,
            label="GMVP",
        )
        axis.scatter(
            [tangency_volatility],
            [tangency_return],
            color=SEMANTIC_COLORS["comparison"],
            marker="D",
            s=65,
            zorder=3,
            label="Tangency",
        )
        axis.set_title("Analytical frontier and reference portfolios")
        axis.set_xlabel("Annualized volatility")
        axis.set_ylabel("Expected annual return")
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        axis.legend(frameon=False)
        style_axes(axis, grid_axis="y")
        figure.suptitle(
            f"GMVP volatility {gmvp_volatility:.2%} | Tangency Sharpe {tangency_sharpe:.2f}",
            fontsize=14,
        )
        context_data = expected_returns if _context_note(expected_returns) else covariance
        _add_context_note(figure, context_data)
    return figure, pd.DataFrame({"gmvp": gmvp, "tangency": tangency})
