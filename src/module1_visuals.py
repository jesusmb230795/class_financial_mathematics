"""Pure, publication-safe exploratory figures for Module 1.

The builders keep calculations and chart contracts outside notebook cells so
the rendered lessons remain concise and the visual evidence can be tested.
"""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import dates as mdates
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from src.visual_style import (
    CORAL,
    INLINE_FIGURE_DPI,
    FINMATH_DIVERGING_CMAP,
    INK,
    MUTED_BLUE,
    PLOT_BACKGROUND,
    SOFT_GRID,
    TEAL,
    add_figure_note,
    matplotlib_style,
    series_color,
    series_linestyle,
    series_marker,
    style_axes,
)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{name} must contain at least one observation")


def _require_finite_observations(frame: pd.DataFrame, name: str) -> None:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.replace([np.inf, -np.inf], np.nan).count().eq(0).any():
        columns = numeric.columns[numeric.replace([np.inf, -np.inf], np.nan).count().eq(0)].tolist()
        raise ValueError(f"{name} has no finite observations for columns: {columns}")


def _display_label(value: str) -> str:
    special_labels = {
        "cetes_28d": "CETES 28-day",
        "mexico_cpi": "Mexico CPI",
        "mexico_inflation": "Mexico inflation",
        "tiie_28d": "TIIE 28-day",
        "udi": "UDI",
        "us_10y": "US 10-year yield",
        "usd_mxn": "USD/MXN",
    }
    if value.isupper():
        return value
    return special_labels.get(value, value.replace("_", " ").title())


def _format_date_axis(axis) -> None:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _observed_sample_label(
    frame: pd.DataFrame,
    sample_label: str | None,
    name: str,
) -> str:
    """Return an explicit label or derive the actual inclusive date sample."""
    if sample_label:
        return sample_label
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.empty:
        raise ValueError(f"{name} requires a sample_label or a non-empty date index")
    observed_index = frame.index[frame.index.notna()]
    if observed_index.empty:
        raise ValueError(f"{name} date index has no valid observations")
    return (
        f"{observed_index.min().strftime('%Y-%m-%d')} to "
        f"{observed_index.max().strftime('%Y-%m-%d')}"
    )


def _plot_panel(
    axis,
    frame: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    percent_xmax: float | None = None,
) -> None:
    for index, column in enumerate(frame.columns):
        axis.plot(
            frame.index,
            frame[column],
            color=series_color(index),
            linestyle=series_linestyle(index),
            linewidth=1.7,
            label=_display_label(str(column)),
        )
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel(ylabel)
    if percent_xmax is not None:
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=percent_xmax, decimals=1))
    axis.legend(frameon=False, ncols=min(3, len(frame.columns)), loc="best")
    _format_date_axis(axis)
    style_axes(axis, grid_axis="y")


def build_stock_eda_overview(
    prices: pd.DataFrame,
    log_returns: pd.DataFrame,
    *,
    rolling_window: int = 63,
    sample_label: str | None = None,
) -> Figure:
    """Build a six-view equity EDA board on comparable analytical scales."""
    if rolling_window < 2:
        raise ValueError("rolling_window must be at least 2")
    if prices.empty or log_returns.empty:
        raise ValueError("prices and log_returns must contain observations")
    columns = [column for column in prices.columns if column in log_returns.columns]
    if len(columns) < 2:
        raise ValueError("prices and log_returns must share at least two columns")
    _require_finite_observations(prices[columns], "prices")
    _require_finite_observations(log_returns[columns], "log_returns")

    clean_prices = prices[columns].copy()
    clean_returns = log_returns[columns].copy()
    sample_label = _observed_sample_label(clean_prices, sample_label, "prices")
    base_values = clean_prices.apply(lambda series: series.dropna().iloc[0])
    if (~np.isfinite(base_values) | base_values.eq(0)).any():
        invalid = base_values.index[~np.isfinite(base_values) | base_values.eq(0)]
        raise ValueError(f"prices cannot be rebased for columns: {invalid.tolist()}")

    rebased_prices = clean_prices.divide(base_values).multiply(100)
    rolling_volatility = clean_returns.rolling(rolling_window).std().multiply(np.sqrt(252))
    drawdowns = clean_prices.divide(clean_prices.cummax()).subtract(1)
    correlations = clean_returns.corr()
    long_returns = (
        clean_returns.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="asset", value_name="return")
        .dropna()
    )

    with matplotlib_style():
        figure, axes = plt.subplots(
            6,
            1,
            figsize=(7.5, 22),
            dpi=INLINE_FIGURE_DPI,
            gridspec_kw={"height_ratios": [1, 1, 1, 1, 1.15, 1.35]},
        )
        (
            indexed_axis,
            dependence_axis,
            volatility_axis,
            drawdown_axis,
            distribution_axis,
            correlation_axis,
        ) = axes

        _plot_panel(
            indexed_axis,
            rebased_prices,
            title="Adjusted prices on a common base",
            ylabel="Index (first observation = 100)",
        )
        asset_a, asset_b = columns[:2]
        rolling_correlation = (
            clean_returns[asset_a].rolling(rolling_window).corr(clean_returns[asset_b])
        )
        dependence_axis.plot(
            rolling_correlation.index,
            rolling_correlation,
            color=MUTED_BLUE,
            linewidth=1.6,
            label=f"{asset_a} vs {asset_b}",
        )
        dependence_axis.fill_between(
            rolling_correlation.index,
            0,
            rolling_correlation.clip(lower=0),
            color=TEAL,
            alpha=0.24,
        )
        dependence_axis.fill_between(
            rolling_correlation.index,
            0,
            rolling_correlation.clip(upper=0),
            color=CORAL,
            alpha=0.22,
        )
        dependence_axis.set_ylim(-1.05, 1.05)
        dependence_axis.set_title(f"{rolling_window}-trading-day correlation is not constant")
        dependence_axis.set_xlabel("Date")
        dependence_axis.set_ylabel("Rolling correlation")
        dependence_axis.legend(frameon=False)
        _format_date_axis(dependence_axis)
        style_axes(dependence_axis, grid_axis="y", show_zero_line=True)
        _plot_panel(
            volatility_axis,
            rolling_volatility,
            title=f"{rolling_window}-trading-day rolling risk",
            ylabel="Annualized volatility",
            percent_xmax=1,
        )
        _plot_panel(
            drawdown_axis,
            drawdowns,
            title="Loss from each running peak",
            ylabel="Drawdown",
            percent_xmax=1,
        )

        asset_palette = {column: series_color(index) for index, column in enumerate(columns)}
        sns.violinplot(
            data=long_returns,
            x="return",
            y="asset",
            hue="asset",
            palette=asset_palette,
            cut=0,
            density_norm="width",
            inner="quart",
            linewidth=1,
            legend=False,
            ax=distribution_axis,
        )
        distribution_axis.axvline(0, color=INK, linewidth=0.8, alpha=0.65)
        distribution_axis.set_title("Daily log-return distributions")
        distribution_axis.set_xlabel("Daily log return")
        distribution_axis.set_ylabel("")
        distribution_axis.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        style_axes(distribution_axis, grid_axis="x")

        upper_triangle = np.triu(np.ones_like(correlations, dtype=bool), k=1)
        sns.heatmap(
            correlations,
            mask=upper_triangle,
            annot=True,
            fmt=".2f",
            cmap=FINMATH_DIVERGING_CMAP,
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.7,
            linecolor=SOFT_GRID,
            square=False,
            cbar_kws={"label": "Correlation", "shrink": 0.8},
            ax=correlation_axis,
        )
        correlation_axis.set_title("Return dependence on a fixed [-1, 1] scale")
        correlation_axis.set_xlabel("")
        correlation_axis.set_ylabel("")
        style_axes(correlation_axis, grid_axis=None)

        figure.suptitle(
            "NASDAQ equity EDA | paths, regimes, tails, and dependence",
            x=0.01,
            ha="left",
        )
        add_figure_note(
            figure,
            "Source: versioned Yahoo Finance adjusted-close snapshot via yfinance. "
            f"Sample: {sample_label}. Drawdowns come directly from adjusted prices; "
            "returns are daily log returns; annualization uses 252 trading days.",
        )
        figure.tight_layout(rect=(0, 0.028, 1, 0.975))
    return figure


def build_stock_risk_map(
    log_returns: pd.DataFrame,
    *,
    sample_label: str | None = None,
) -> Figure:
    """Map annualized return against volatility with visible tail-risk cues."""
    if log_returns.empty:
        raise ValueError("log_returns must contain observations")
    _require_finite_observations(log_returns, "log_returns")
    observed = log_returns.copy()
    sample_label = _observed_sample_label(
        observed,
        sample_label,
        "log_returns",
    )
    wealth = np.exp(observed.cumsum())
    running_peak = wealth.cummax().clip(lower=1.0)
    metrics = pd.DataFrame(
        {
            "annualized_mean_log_return": observed.mean() * 252,
            "annualized_volatility": observed.std() * np.sqrt(252),
            "maximum_drawdown": wealth.divide(running_peak).subtract(1).min(),
            "p05_loss": (-observed.quantile(0.05)).clip(lower=0),
        }
    )
    if not np.isfinite(metrics.to_numpy(dtype=float)).all():
        raise ValueError("log_returns do not produce finite risk-map metrics")

    maximum_tail_loss = metrics["p05_loss"].max()
    tail_scale = (
        metrics["p05_loss"].divide(maximum_tail_loss)
        if maximum_tail_loss > 0
        else pd.Series(0.0, index=metrics.index)
    )

    with matplotlib_style():
        figure, axis = plt.subplots(
            figsize=(7.5, 7),
            dpi=INLINE_FIGURE_DPI,
        )
        for index, (asset, row) in enumerate(metrics.iterrows()):
            axis.scatter(
                row["annualized_volatility"],
                row["annualized_mean_log_return"],
                s=150 + 320 * tail_scale.loc[asset],
                color=series_color(index),
                edgecolor=INK,
                linewidth=0.8,
                marker=series_marker(index),
                alpha=0.88,
                label=(
                    f"{asset}: max drawdown {row['maximum_drawdown']:.1%}, "
                    f"5th-percentile log-loss magnitude {row['p05_loss']:.1%}"
                ),
            )
            axis.annotate(
                str(asset),
                (row["annualized_volatility"], row["annualized_mean_log_return"]),
                xytext=(6, 7),
                textcoords="offset points",
                color=INK,
                fontsize=9,
                fontweight="semibold",
            )

        axis.axvline(
            metrics["annualized_volatility"].median(),
            color=MUTED_BLUE,
            linestyle="--",
            linewidth=1,
            label="Cross-sectional medians",
        )
        axis.axhline(
            metrics["annualized_mean_log_return"].median(),
            color=MUTED_BLUE,
            linestyle="--",
            linewidth=1,
        )
        axis.set_title("Observed mean change, variability, and downside severity")
        axis.set_xlabel("Annualized volatility (252 trading days)")
        axis.set_ylabel("Annualized mean log return")
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        axis.legend(
            frameon=False,
            fontsize=8.5,
            loc="upper left",
            bbox_to_anchor=(0, -0.17),
            borderaxespad=0,
        )
        style_axes(axis, grid_axis="both")
        figure.suptitle(
            "NASDAQ risk map | larger markers indicate a larger\n5th-percentile loss",
            x=0.01,
            ha="left",
        )
        add_figure_note(
            figure,
            "Source: versioned Yahoo Finance adjusted-close snapshot via yfinance. "
            f"Sample: {sample_label}. Descriptive evidence, not a forecast.",
        )
        figure.tight_layout(rect=(0, 0.07, 1, 0.92))
    return figure


def build_macro_eda_overview(
    macro: pd.DataFrame,
    *,
    rolling_window: int = 12,
    sample_label: str | None = None,
) -> tuple[Figure, pd.DataFrame]:
    """Build a macro EDA board that preserves units and exposes regime changes."""
    if rolling_window < 2:
        raise ValueError("rolling_window must be at least 2")
    rate_columns = [
        "banxico_target_rate",
        "cetes_28d",
        "tiie_28d",
        "mexico_inflation",
        "us_10y",
    ]
    index_columns = ["usd_mxn", "udi", "mexico_cpi"]
    _require_columns(macro, (*rate_columns, *index_columns), "macro")
    _require_finite_observations(macro[rate_columns + index_columns], "macro")
    sample_label = _observed_sample_label(macro, sample_label, "macro")

    rate_panel = macro[rate_columns].multiply(100)
    index_panel = macro[index_columns].copy()
    bases = index_panel.apply(lambda series: series.dropna().iloc[0])
    if (~np.isfinite(bases) | bases.eq(0)).any():
        invalid = bases.index[~np.isfinite(bases) | bases.eq(0)]
        raise ValueError(f"macro series cannot be rebased: {invalid.tolist()}")
    rebased_indices = index_panel.divide(bases).multiply(100)

    transformed_changes = pd.DataFrame(
        {
            "Banxico target rate change (pp)": macro["banxico_target_rate"].diff() * 100,
            "Mexico inflation change (pp)": macro["mexico_inflation"].diff() * 100,
            "US 10Y change (pp)": macro["us_10y"].diff() * 100,
            "USD/MXN percentage change": macro["usd_mxn"].pct_change(fill_method=None),
            "UDI percentage change": macro["udi"].pct_change(fill_method=None),
        },
        index=macro.index,
    )
    change_standard_deviation = transformed_changes.std().replace(0, np.nan)
    standardized_changes = transformed_changes.subtract(transformed_changes.mean()).divide(
        change_standard_deviation
    )
    rolling_correlation = (
        transformed_changes["UDI percentage change"]
        .rolling(rolling_window)
        .corr(transformed_changes["USD/MXN percentage change"])
    )

    with matplotlib_style():
        figure, axes = plt.subplots(
            4,
            1,
            figsize=(7.5, 15.5),
            dpi=INLINE_FIGURE_DPI,
            gridspec_kw={"height_ratios": [1, 1, 1.15, 1]},
        )
        rates_axis, indices_axis, change_axis, correlation_axis = axes

        _plot_panel(
            rates_axis,
            rate_panel,
            title="Rates remain in economically meaningful units",
            ylabel="Percent per year or year-over-year",
            percent_xmax=100,
        )
        _plot_panel(
            indices_axis,
            rebased_indices,
            title="Price and index paths share a comparison base",
            ylabel="Index (first observation = 100)",
        )

        heatmap = standardized_changes.T
        sns.heatmap(
            heatmap,
            cmap=FINMATH_DIVERGING_CMAP,
            center=0,
            vmin=-3,
            vmax=3,
            cbar_kws={"label": "Within-series z-score", "shrink": 0.8},
            xticklabels=False,
            linewidths=0,
            ax=change_axis,
        )
        change_axis.set_title("Regime map of one-period changes")
        tick_count = min(6, len(heatmap.columns))
        tick_positions = np.linspace(
            0,
            len(heatmap.columns) - 1,
            num=tick_count,
            dtype=int,
        )
        tick_positions = np.unique(tick_positions)
        tick_labels = [
            pd.Timestamp(heatmap.columns[position]).strftime("%b\n%Y")
            for position in tick_positions
        ]
        change_axis.set_xticks(tick_positions + 0.5, labels=tick_labels, rotation=0)
        change_axis.set_xlabel("Reference month")
        change_axis.set_ylabel("")
        style_axes(change_axis, grid_axis=None)

        positive = rolling_correlation.clip(lower=0)
        negative = rolling_correlation.clip(upper=0)
        correlation_axis.plot(
            rolling_correlation.index,
            rolling_correlation,
            color=INK,
            linewidth=1.4,
            label=f"{rolling_window}-month rolling correlation",
        )
        correlation_axis.fill_between(
            rolling_correlation.index,
            0,
            positive,
            color=TEAL,
            alpha=0.28,
            label="Positive co-movement",
        )
        correlation_axis.fill_between(
            rolling_correlation.index,
            0,
            negative,
            color=CORAL,
            alpha=0.25,
            label="Negative co-movement",
        )
        correlation_axis.set_ylim(-1.05, 1.05)
        correlation_axis.set_title("UDI and USD/MXN co-movement changes over time")
        correlation_axis.set_xlabel("Date")
        correlation_axis.set_ylabel("Rolling correlation")
        correlation_axis.legend(frameon=False, fontsize=9)
        _format_date_axis(correlation_axis)
        style_axes(correlation_axis, grid_axis="y", show_zero_line=True)

        figure.suptitle(
            "Macroeconomic EDA | native units, comparable paths, and regimes",
            x=0.01,
            ha="left",
        )
        add_figure_note(
            figure,
            "Sources: versioned Banxico SIE and DB.NOMICS snapshots. "
            f"Sample: {sample_label}. Latest-vintage exploratory data; not a point-in-time backtest.",
        )
        figure.tight_layout(rect=(0, 0.045, 1, 0.97))
    return figure, transformed_changes


_MEXICAN_OBSERVED_SERIES = ("usd_mxn", "udi")
_MEXICAN_CARRY_SERIES = (
    "cetes_28d_carry",
    "tiie_28d_carry",
    "policy_rate_carry",
)
_MEXICAN_SERIES_ORDER = _MEXICAN_OBSERVED_SERIES + _MEXICAN_CARRY_SERIES
_MEXICAN_SERIES_LABELS = {
    "usd_mxn": "USD/MXN FIX",
    "udi": "UDI",
    "cetes_28d_carry": "CETES 28-day carry",
    "tiie_28d_carry": "TIIE 28-day carry",
    "policy_rate_carry": "Policy-rate carry",
}
_MEXICAN_MACRO_COLUMNS = (
    "banxico_target_rate_pct",
    "cetes_28d_pct",
    "tiie_28d_pct",
    "mexico_inflation_pct",
)
_MEXICAN_MACRO_LABELS = {
    "banxico_target_rate_pct": "Banxico target rate",
    "cetes_28d_pct": "CETES 28-day rate",
    "tiie_28d_pct": "TIIE 28-day rate",
    "mexico_inflation_pct": "Mexico inflation",
}


def build_wfe_market_scale_overview(
    market_scale: pd.DataFrame,
    *,
    sample_label: str = "May 2026",
    verified_on: str = "2026-07-20",
) -> Figure:
    """Build WFE metric cards and one compatible derivatives comparison."""
    required_metrics = (
        "Market capitalisation",
        "Value of share trading",
        "Listed companies",
        "Number of trades",
        "Investment flows",
        "Options contracts traded",
        "Futures contracts traded",
    )
    _require_columns(
        market_scale,
        (
            "metric",
            "display_value",
            "display_unit",
            "change_percent",
            "change_basis",
        ),
        "market_scale",
    )
    display_data = market_scale.loc[
        :,
        [
            "metric",
            "display_value",
            "display_unit",
            "change_percent",
            "change_basis",
        ],
    ].copy()
    if display_data["metric"].duplicated().any():
        raise ValueError("market_scale metrics must be unique")
    missing_metrics = sorted(set(required_metrics) - set(display_data["metric"]))
    if missing_metrics:
        raise ValueError(f"market_scale is missing required metrics: {missing_metrics}")
    _require_finite_observations(
        display_data[["display_value", "change_percent"]],
        "market_scale",
    )
    if display_data["display_value"].lt(0).any():
        raise ValueError("market_scale display values must be non-negative")
    change_bases = display_data["change_basis"].dropna().astype(str).str.strip().unique()
    if len(change_bases) != 1 or not change_bases[0]:
        raise ValueError("market_scale must declare one non-empty change_basis")
    change_basis = change_bases[0]

    by_metric = display_data.set_index("metric")
    display_groups = (
        {
            "title": "Equity market size",
            "metrics": ("Market capitalisation", "Listed companies"),
            "labels": ("Market capitalisation", "Listed companies"),
            "colors": (series_color(0), series_color(1)),
        },
        {
            "title": "Equity market activity",
            "metrics": ("Value of share trading", "Number of trades"),
            "labels": ("Share trading value", "Number of trades"),
            "colors": (series_color(2), series_color(3)),
        },
        {
            "title": "Primary-market flow",
            "metrics": ("Investment flows",),
            "labels": ("Investment flows",),
            "colors": (series_color(4),),
        },
        {
            "title": "Listed derivatives activity",
            "metrics": ("Options contracts traded", "Futures contracts traded"),
            "labels": ("Options", "Futures"),
            "colors": (series_color(3), series_color(4)),
        },
    )

    with matplotlib_style():
        figure, axes = plt.subplots(
            4,
            1,
            figsize=(7.5, 12),
            dpi=INLINE_FIGURE_DPI,
            gridspec_kw={"height_ratios": [1, 1, 0.85, 1.15]},
        )
        for axis, group in zip(axes, display_groups, strict=True):
            panel = by_metric.loc[list(group["metrics"])]
            axis.set_title(group["title"])
            if group["title"] != "Listed derivatives activity":
                axis.set_xlim(0, 1)
                axis.set_ylim(0, 1)
                y_positions = np.linspace(0.68, 0.34, len(panel))
                for y, (_, row), label, color in zip(
                    y_positions,
                    panel.iterrows(),
                    group["labels"],
                    group["colors"],
                    strict=True,
                ):
                    axis.text(
                        0.05,
                        y + 0.10,
                        label,
                        ha="left",
                        va="center",
                        fontsize=10,
                        color=INK,
                    )
                    axis.text(
                        0.05,
                        y,
                        f"{float(row['display_value']):,.2f} {row['display_unit']}",
                        ha="left",
                        va="center",
                        fontsize=18,
                        fontweight="semibold",
                        color=color,
                    )
                    axis.text(
                        0.95,
                        y,
                        f"{float(row['change_percent']):+.2f}% reported",
                        ha="right",
                        va="center",
                        fontsize=11,
                        color=INK,
                    )
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)
                continue

            positions = range(len(panel))
            axis.barh(
                positions,
                panel["display_value"],
                color=group["colors"],
                height=0.58,
            )
            axis.set_yticks(positions, labels=group["labels"])
            axis.invert_yaxis()
            axis.set_xlabel("Billion contracts")
            axis.set_xlim(0, panel["display_value"].max() * 1.22)
            for position, value, change in zip(
                positions,
                panel["display_value"],
                panel["change_percent"],
                strict=True,
            ):
                axis.text(
                    value,
                    position,
                    f"  {value:,.2f} | {change:+.2f}% reported",
                    va="center",
                )

        style_axes(axes[3], grid_axis="x")
        figure.suptitle("WFE market scale snapshot | native-unit metric cards")
        add_figure_note(
            figure,
            f"Source: WFE Focus dashboard, {sample_label}; values verified "
            f"{verified_on}. Values retain source units. Percent changes reproduce "
            f"WFE labels; change basis: {change_basis}. No month-over-month or "
            "year-over-year meaning is inferred when the source does not specify it. "
            "Compare bar lengths only between options and futures.",
        )
        figure.tight_layout(rect=(0, 0.065, 1, 0.95))
    return figure


def build_mexican_market_paths(
    levels: pd.DataFrame,
    cumulative_changes: pd.DataFrame,
    *,
    sample_label: str | None = None,
) -> Figure:
    """Build distinct observed-level and cumulative-change Mexican-market views."""
    _require_columns(
        levels,
        _MEXICAN_OBSERVED_SERIES,
        "levels",
    )
    _require_columns(cumulative_changes, _MEXICAN_SERIES_ORDER, "cumulative_changes")
    _require_finite_observations(
        levels[list(_MEXICAN_OBSERVED_SERIES)],
        "levels",
    )
    _require_finite_observations(
        cumulative_changes[list(_MEXICAN_SERIES_ORDER)],
        "cumulative_changes",
    )
    level_panel = levels.loc[:, list(_MEXICAN_OBSERVED_SERIES)].copy()
    change_panel = cumulative_changes.loc[:, list(_MEXICAN_SERIES_ORDER)].copy()
    sample_label = _observed_sample_label(
        level_panel,
        sample_label,
        "levels",
    )

    observed_bases = level_panel[list(_MEXICAN_OBSERVED_SERIES)].apply(
        lambda series: series.dropna().iloc[0]
    )
    observed_levels_are_rebased = np.allclose(observed_bases, 100.0)

    with matplotlib_style():
        figure, axes = plt.subplots(
            4,
            1,
            figsize=(7.5, 14),
            dpi=INLINE_FIGURE_DPI,
            sharex=False,
        )
        for axis, column, native_unit in zip(
            axes[:2],
            _MEXICAN_OBSERVED_SERIES,
            ("MXN per USD", "MXN per UDI"),
            strict=True,
        ):
            style_index = _MEXICAN_SERIES_ORDER.index(column)
            axis.plot(
                level_panel.index,
                level_panel[column],
                color=series_color(style_index),
                linestyle=series_linestyle(style_index),
                label=_MEXICAN_SERIES_LABELS[column],
            )
            axis.set_title(f"Observed {_MEXICAN_SERIES_LABELS[column]} level")
            axis.set_ylabel(
                "Index (first observation = 100)" if observed_levels_are_rebased else native_unit
            )
            axis.legend(loc="best")
            _format_date_axis(axis)

        for column in _MEXICAN_CARRY_SERIES:
            style_index = _MEXICAN_SERIES_ORDER.index(column)
            axes[2].plot(
                change_panel.index,
                change_panel[column],
                color=series_color(style_index),
                linestyle=series_linestyle(style_index),
                label=_MEXICAN_SERIES_LABELS[column],
            )
        for column in _MEXICAN_OBSERVED_SERIES:
            style_index = _MEXICAN_SERIES_ORDER.index(column)
            axes[3].plot(
                change_panel.index,
                change_panel[column],
                color=series_color(style_index),
                linestyle=series_linestyle(style_index),
                label=_MEXICAN_SERIES_LABELS[column],
            )

        axes[2].set_title("Synthetic cumulative carry changes")
        axes[3].set_title("Observed cumulative level changes")
        for axis in axes[2:]:
            axis.set_ylabel("Simple-change equivalent")
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            axis.legend(loc="best")
            _format_date_axis(axis)
        style_axes(axes[:2], grid_axis="y")
        style_axes(axes[2:], grid_axis="y", show_zero_line=True)
        figure.suptitle("Mexican market paths | observed levels versus synthetic carry")
        add_figure_note(
            figure,
            f"Source: versioned Banxico SIE observations | {sample_label}. "
            "Observed levels are shown in their supplied scale. Synthetic carry "
            "uses the previous published annual rate, accrues every elapsed calendar "
            "day on an internal daily grid, and is sampled on joint observed-level "
            "dates. Cumulative changes are shown once.",
        )
        figure.tight_layout(rect=(0, 0.055, 1, 0.96))
    return figure


def build_mexican_market_risk(
    log_changes: pd.DataFrame,
    rolling_dispersion: pd.DataFrame,
    scatter_data: pd.DataFrame,
    *,
    rolling_window: int = 63,
    sample_label: str | None = None,
) -> Figure:
    """Build a vertical distribution and rolling-dispersion Mexican-case block."""
    if rolling_window < 2:
        raise ValueError("rolling_window must be at least 2")
    _require_columns(log_changes, ("usd_mxn",), "log_changes")
    _require_columns(
        rolling_dispersion,
        _MEXICAN_SERIES_ORDER,
        "rolling_dispersion",
    )
    _require_columns(scatter_data, _MEXICAN_OBSERVED_SERIES, "scatter_data")
    _require_finite_observations(log_changes[["usd_mxn"]], "log_changes")
    _require_finite_observations(
        rolling_dispersion[list(_MEXICAN_SERIES_ORDER)],
        "rolling_dispersion",
    )
    _require_finite_observations(
        scatter_data[list(_MEXICAN_OBSERVED_SERIES)],
        "scatter_data",
    )
    usd_mxn_log_changes = log_changes["usd_mxn"].copy()
    dispersion_panel = rolling_dispersion.loc[:, list(_MEXICAN_SERIES_ORDER)].copy()
    relationship_panel = scatter_data.loc[:, list(_MEXICAN_OBSERVED_SERIES)].copy()
    sample_label = _observed_sample_label(
        log_changes,
        sample_label,
        "log_changes",
    )

    with matplotlib_style():
        figure, axes = plt.subplots(
            4,
            1,
            figsize=(7.5, 16),
            dpi=INLINE_FIGURE_DPI,
        )
        histogram_axis, scatter_axis, observed_volatility_axis, carry_volatility_axis = axes

        histogram_axis.hist(
            usd_mxn_log_changes.dropna(),
            bins=40,
            color=series_color(0),
            alpha=0.88,
            edgecolor=PLOT_BACKGROUND,
            linewidth=0.5,
        )
        histogram_axis.axvline(
            0,
            color=series_color(4),
            linewidth=0.9,
            alpha=0.7,
        )
        histogram_axis.set_title("USD/MXN observed-interval log-change distribution")
        histogram_axis.set_xlabel("Log change between joint observations")
        histogram_axis.set_ylabel("Observations")
        histogram_axis.xaxis.set_major_formatter(PercentFormatter(xmax=1))

        scatter_axis.scatter(
            relationship_panel["usd_mxn"],
            relationship_panel["udi"],
            alpha=0.42,
            color=series_color(2),
            edgecolors="none",
        )
        scatter_axis.axvline(
            0,
            color=series_color(4),
            linewidth=0.8,
            alpha=0.6,
        )
        scatter_axis.axhline(
            0,
            color=series_color(4),
            linewidth=0.8,
            alpha=0.6,
        )
        scatter_axis.set_title("USD/MXN versus UDI observed-interval log changes")
        scatter_axis.set_xlabel("USD/MXN log change between joint observations")
        scatter_axis.set_ylabel("UDI log change between joint observations")
        scatter_axis.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        scatter_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))

        for column in _MEXICAN_OBSERVED_SERIES:
            style_index = _MEXICAN_SERIES_ORDER.index(column)
            observed_volatility_axis.plot(
                dispersion_panel.index,
                dispersion_panel[column],
                color=series_color(style_index),
                linestyle=series_linestyle(style_index),
                label=_MEXICAN_SERIES_LABELS[column],
            )
        for column in _MEXICAN_CARRY_SERIES:
            style_index = _MEXICAN_SERIES_ORDER.index(column)
            carry_volatility_axis.plot(
                dispersion_panel.index,
                dispersion_panel[column],
                color=series_color(style_index),
                linestyle=series_linestyle(style_index),
                label=_MEXICAN_SERIES_LABELS[column],
            )

        observed_volatility_axis.set_title(
            f"Observed {rolling_window}-valid-observation rolling log-change dispersion"
        )
        carry_volatility_axis.set_title(
            f"Synthetic {rolling_window}-valid-observation rolling log-change dispersion"
        )
        for axis in (observed_volatility_axis, carry_volatility_axis):
            axis.set_xlabel("Date")
            axis.set_ylabel("Standard deviation per stored change")
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            axis.legend(loc="best")
            date_locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
            axis.xaxis.set_major_locator(date_locator)
            axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(date_locator))

        style_axes(
            [histogram_axis, observed_volatility_axis, carry_volatility_axis],
            grid_axis="y",
        )
        style_axes(scatter_axis, grid_axis="both")
        figure.suptitle("Distribution and dispersion | observed variables and synthetic carry")
        add_figure_note(
            figure,
            "Sources: versioned Banxico SIE observations and documented carry "
            "transformations | "
            f"{sample_label}. Rolling standard deviations use valid stored changes; "
            "heterogeneous observation intervals are not annualized.",
        )
        figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    return figure


def build_mexican_market_dependence(
    correlations: pd.DataFrame,
    drawdowns: pd.DataFrame,
    *,
    sample_label: str | None = None,
) -> Figure:
    """Build a vertical correlation and observed-drawdown block."""
    _require_columns(correlations, _MEXICAN_SERIES_ORDER, "correlations")
    missing_rows = sorted(set(_MEXICAN_SERIES_ORDER) - set(correlations.index))
    if missing_rows:
        raise ValueError(f"correlations is missing required index rows: {missing_rows}")
    _require_columns(drawdowns, _MEXICAN_SERIES_ORDER, "drawdowns")
    correlation_panel = correlations.loc[
        list(_MEXICAN_SERIES_ORDER),
        list(_MEXICAN_SERIES_ORDER),
    ].copy()
    drawdown_panel = drawdowns.loc[:, list(_MEXICAN_SERIES_ORDER)].copy()
    sample_label = _observed_sample_label(
        drawdown_panel,
        sample_label,
        "drawdowns",
    )
    _require_finite_observations(correlation_panel, "correlations")
    _require_finite_observations(drawdown_panel, "drawdowns")
    carry_minimum_drawdowns = drawdown_panel[list(_MEXICAN_CARRY_SERIES)].min()
    carry_drawdown_note = ", ".join(
        f"{_MEXICAN_SERIES_LABELS[column]}: {value:.2%}"
        for column, value in carry_minimum_drawdowns.items()
    )

    with matplotlib_style():
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(7.5, 10.5),
            gridspec_kw={"height_ratios": [1.2, 1]},
        )
        correlation_axis, drawdown_axis = axes
        display_correlations = correlation_panel.rename(
            index=_MEXICAN_SERIES_LABELS,
            columns=_MEXICAN_SERIES_LABELS,
        )
        sns.heatmap(
            display_correlations,
            annot=True,
            fmt=".2f",
            cmap=FINMATH_DIVERGING_CMAP,
            center=0,
            vmin=-1,
            vmax=1,
            ax=correlation_axis,
            cbar_kws={"label": "Correlation coefficient", "shrink": 0.82},
        )
        correlation_axis.set_title("Observed-interval log-change correlation | fixed scale [-1, 1]")

        for column in _MEXICAN_OBSERVED_SERIES:
            style_index = _MEXICAN_SERIES_ORDER.index(column)
            drawdown_axis.plot(
                drawdown_panel.index,
                drawdown_panel[column],
                color=series_color(style_index),
                linestyle=series_linestyle(style_index),
                label=(
                    f"{_MEXICAN_SERIES_LABELS[column]} (minimum {drawdown_panel[column].min():.1%})"
                ),
            )
        drawdown_axis.set_title("Observed-level drawdown (not investment loss)")
        drawdown_axis.set_xlabel("Date")
        drawdown_axis.set_ylabel("Level drawdown")
        drawdown_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        drawdown_axis.legend(loc="lower left")
        drawdown_locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
        drawdown_axis.xaxis.set_major_locator(drawdown_locator)
        drawdown_axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(drawdown_locator))

        style_axes(correlation_axis, grid_axis=None)
        style_axes(drawdown_axis, grid_axis="y", show_zero_line=True)
        figure.suptitle("Dependence and level drawdown | not an investment-loss measure")
        add_figure_note(
            figure,
            "Sources: versioned Banxico SIE observations and documented carry "
            "transformations | "
            f"{sample_label}. Synthetic carry minimum drawdowns: "
            f"{carry_drawdown_note}. Level drawdown is a path statistic, not an "
            "investment loss; correlation is descriptive, not causal.",
        )
        figure.tight_layout(rect=(0, 0.075, 1, 0.94))
    return figure


def build_mexican_macro_context(
    macro_context: pd.DataFrame,
    *,
    sample_label: str | None = None,
) -> Figure:
    """Build the monthly Mexican rate and inflation context block."""
    _require_columns(macro_context, _MEXICAN_MACRO_COLUMNS, "macro_context")
    _require_finite_observations(
        macro_context[list(_MEXICAN_MACRO_COLUMNS)],
        "macro_context",
    )
    display_data = macro_context.loc[:, list(_MEXICAN_MACRO_COLUMNS)].copy()
    sample_label = _observed_sample_label(
        display_data,
        sample_label,
        "macro_context",
    )

    with matplotlib_style():
        figure, axis = plt.subplots(
            figsize=(7.5, 6.2),
            dpi=INLINE_FIGURE_DPI,
        )
        for style_index, column in enumerate(display_data.columns):
            axis.plot(
                display_data.index,
                display_data[column],
                color=series_color(style_index),
                linestyle=series_linestyle(style_index),
                label=_MEXICAN_MACRO_LABELS[column],
            )
        axis.set_title("Monthly Mexican rate and inflation context")
        axis.set_xlabel("Reference month")
        axis.set_ylabel("Percent")
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))
        axis.legend(loc="best", ncols=2)
        _format_date_axis(axis)
        style_axes(axis, grid_axis="y")
        figure.suptitle("Macro context | latest-vintage monthly observations")
        add_figure_note(
            figure,
            f"Sources: Banxico SIE and DB.NOMICS snapshots | {sample_label}. "
            "Release dates are not aligned to historical availability.",
        )
        figure.tight_layout(rect=(0, 0.11, 1, 0.89))
    return figure


__all__ = [
    "build_macro_eda_overview",
    "build_mexican_macro_context",
    "build_mexican_market_dependence",
    "build_mexican_market_paths",
    "build_mexican_market_risk",
    "build_stock_eda_overview",
    "build_stock_risk_map",
    "build_wfe_market_scale_overview",
]
