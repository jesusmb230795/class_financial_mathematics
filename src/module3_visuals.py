"""Publication-safe conceptual and empirical figures for Module 3."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.visual_style import (
    AMBER_DARK,
    CORAL,
    INK,
    MUTED_BLUE,
    SOFT_GRID,
    TEAL,
    add_figure_note,
    matplotlib_style,
    style_axes,
)


_HISTORY_COLUMNS = (
    "banxico_target_rate",
    "cetes_28d",
    "mexico_inflation_lag1m",
    "us_10y",
    "usd_mxn",
)
_EVALUATION_COLUMNS = (
    "Actual next-month return",
    "Ridge prediction",
    "Training-mean benchmark",
    "Zero-change benchmark",
)
_FIT_COLUMNS = ("Evaluation fit", "Updated fit")


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _require_finite(frame: pd.DataFrame, name: str) -> None:
    if frame.empty:
        raise ValueError(f"{name} must contain observations")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} must contain finite values")


def _format_dates(axis: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _wrapped_note(note: str, *, width: int = 145) -> str:
    return "\n".join(textwrap.wrap(note, width=width))


def _linear_equilibrium(
    demand_intercept: float,
    demand_slope: float,
    supply_intercept: float,
    supply_slope: float,
) -> tuple[float, float]:
    """Return quantity and price for two linear schematic curves."""

    quantity = (demand_intercept - supply_intercept) / (demand_slope + supply_slope)
    price = demand_intercept - demand_slope * quantity
    return quantity, price


def build_supply_demand_identification_figure() -> Figure:
    """Contrast a movement along demand with an outward demand shift."""

    quantity = np.linspace(0.0, 10.0, 240)
    demand_0 = 10.0 - 0.8 * quantity
    demand_1 = 12.0 - 0.8 * quantity
    supply_0 = 1.0 + 0.8 * quantity
    supply_1 = 3.0 + 0.8 * quantity

    equilibrium_a = _linear_equilibrium(10.0, 0.8, 1.0, 0.8)
    equilibrium_b = _linear_equilibrium(10.0, 0.8, 3.0, 0.8)
    equilibrium_c = _linear_equilibrium(12.0, 0.8, 1.0, 0.8)

    with matplotlib_style():
        figure, axes = plt.subplots(1, 2, figsize=(10, 5.625), sharex=True, sharey=True)
        supply_axis, demand_axis = axes

        supply_axis.plot(quantity, demand_0, color=TEAL, label="Demand D0")
        supply_axis.plot(
            quantity,
            supply_0,
            color=MUTED_BLUE,
            linestyle="--",
            label="Supply S0",
        )
        supply_axis.plot(
            quantity,
            supply_1,
            color=CORAL,
            linestyle="-.",
            label="Supply S1 (inward)",
        )
        supply_axis.set_title("Stable demand: an inward supply shift")

        demand_axis.plot(quantity, supply_0, color=MUTED_BLUE, linestyle="--", label="Supply S0")
        demand_axis.plot(quantity, demand_0, color=TEAL, label="Demand D0")
        demand_axis.plot(
            quantity,
            demand_1,
            color=AMBER_DARK,
            linestyle="-.",
            label="Demand D1 (outward)",
        )
        demand_axis.set_title("Stable supply: an outward demand shift")

        point_specs = (
            (supply_axis, equilibrium_a, "A", TEAL),
            (supply_axis, equilibrium_b, "B", CORAL),
            (demand_axis, equilibrium_a, "A", TEAL),
            (demand_axis, equilibrium_c, "C", AMBER_DARK),
        )
        for axis, (point_quantity, point_price), label, color in point_specs:
            axis.scatter(
                [point_quantity],
                [point_price],
                color=color,
                edgecolor=INK,
                marker="o" if label == "A" else "s",
                s=55,
                zorder=4,
            )
            axis.annotate(
                label,
                (point_quantity, point_price),
                xytext=(7, 7),
                textcoords="offset points",
                color=INK,
                fontsize=10,
                fontweight="semibold",
            )

        supply_axis.annotate(
            "Price rises; quantity falls",
            xy=equilibrium_b,
            xytext=equilibrium_a,
            arrowprops={"arrowstyle": "->", "color": INK, "linewidth": 1.3},
            color=INK,
            fontsize=9,
            ha="center",
            va="bottom",
        )
        demand_axis.annotate(
            "Price and quantity rise",
            xy=equilibrium_c,
            xytext=equilibrium_a,
            arrowprops={"arrowstyle": "->", "color": INK, "linewidth": 1.3},
            color=INK,
            fontsize=9,
            ha="center",
            va="bottom",
        )

        for axis in axes:
            axis.set_xlim(0, 10)
            axis.set_ylim(0, 12.5)
            axis.set_xlabel("Quantity (schematic)")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.legend(loc="lower center")
            style_axes(axis, grid_axis=None)
        supply_axis.set_ylabel("Price (schematic)")

        figure.suptitle("Why price and quantity alone do not identify the shock")
        add_figure_note(
            figure,
            "Source: book schematic, not estimated data. Curves are illustrative; "
            "identification requires a stable counterpart curve or additional evidence.",
        )
        figure.tight_layout(rect=(0, 0.12, 1, 0.92))
    return figure


def build_fx_carry_break_even_figure(
    *,
    spot_mxn_per_usd: float = 17.20,
    mxn_effective_rate: float = 0.09,
    usd_effective_rate: float = 0.05,
) -> Figure:
    """Show the future-spot level that erases an unhedged MXN carry advantage."""

    if not np.isfinite(spot_mxn_per_usd) or spot_mxn_per_usd <= 0:
        raise ValueError("spot_mxn_per_usd must be finite and positive")
    rates = np.asarray([mxn_effective_rate, usd_effective_rate], dtype=float)
    if not np.isfinite(rates).all() or (rates <= -1).any():
        raise ValueError("effective rates must be finite and greater than -1")

    break_even = spot_mxn_per_usd * (1 + mxn_effective_rate) / (1 + usd_effective_rate)
    lower_spot = 0.94 * min(spot_mxn_per_usd, break_even)
    upper_spot = 1.06 * max(spot_mxn_per_usd, break_even)
    future_spot = np.linspace(lower_spot, upper_spot, 320)
    unhedged_usd_return = (1 + mxn_effective_rate) * spot_mxn_per_usd / future_spot - 1

    with matplotlib_style():
        figure, axis = plt.subplots(figsize=(10, 5.625))
        axis.plot(
            future_spot,
            100 * unhedged_usd_return,
            color=TEAL,
            marker="o",
            markevery=40,
            label="Unhedged MXN investment",
        )
        axis.axhline(
            100 * usd_effective_rate,
            color=MUTED_BLUE,
            linestyle="--",
            label="USD alternative",
        )
        axis.axvline(
            break_even,
            color=INK,
            linestyle=":",
            linewidth=1.3,
            label="CIP / carry break-even",
        )
        axis.scatter(
            [break_even],
            [100 * usd_effective_rate],
            color=AMBER_DARK,
            edgecolor=INK,
            marker="s",
            s=65,
            zorder=4,
        )
        axis.annotate(
            f"Break-even: {break_even:.4f} MXN per USD",
            (break_even, 100 * usd_effective_rate),
            xytext=(12, 16),
            textcoords="offset points",
            color=INK,
            fontsize=9,
        )
        axis.set_title("When does MXN carry stop beating the USD alternative?")
        axis.set_xlabel("Future spot, $S_T$ (MXN per USD)")
        axis.set_ylabel("One-year USD return (%)")
        axis.legend(loc="upper right")
        style_axes(axis, grid_axis="both")
        add_figure_note(
            figure,
            "Source: book calculation. Illustrative one-year effective rates: "
            f"MXN {100 * mxn_effective_rate:.2f}%, USD {100 * usd_effective_rate:.2f}%; "
            f"spot {spot_mxn_per_usd:.2f} MXN per USD. No transaction costs, taxes, "
            "cross-currency basis, or settlement frictions; the intersection is not a forecast.",
        )
        figure.tight_layout(rect=(0, 0.14, 1, 1))
    return figure


def build_scenario_valuation_bridge_figure(
    *,
    base_cash_flow: float = 105.0,
    scenario_cash_flow: float = 102.0,
    base_rate: float = 0.06,
    base_risk_premium: float = 0.03,
    scenario_rate: float = 0.075,
    scenario_risk_premium: float = 0.04,
) -> Figure:
    """Show a cash-flow-first sequential attribution for a one-period scenario."""

    values = np.asarray(
        [
            base_cash_flow,
            scenario_cash_flow,
            base_rate,
            base_risk_premium,
            scenario_rate,
            scenario_risk_premium,
        ],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise ValueError("scenario inputs must be finite")
    if base_cash_flow <= 0 or scenario_cash_flow <= 0:
        raise ValueError("cash flows must be positive for the valuation bridge")
    base_denominator = 1 + base_rate + base_risk_premium
    scenario_denominator = 1 + scenario_rate + scenario_risk_premium
    if base_denominator <= 0 or scenario_denominator <= 0:
        raise ValueError("scenario discount denominators must be positive")

    base_value = base_cash_flow / base_denominator
    cash_flow_only_value = scenario_cash_flow / base_denominator
    scenario_value = scenario_cash_flow / scenario_denominator
    cash_flow_change = cash_flow_only_value - base_value
    discount_change = scenario_value - cash_flow_only_value

    with matplotlib_style():
        figure, axis = plt.subplots(figsize=(10, 5.625))
        positions = np.arange(4)
        axis.bar(
            positions[0],
            base_value,
            color=TEAL,
            hatch="..",
            label="Level",
        )
        axis.bar(
            positions[1],
            abs(cash_flow_change),
            bottom=min(base_value, cash_flow_only_value),
            color=CORAL if cash_flow_change < 0 else TEAL,
            hatch="//",
            label="Sequential change",
        )
        axis.bar(
            positions[2],
            abs(discount_change),
            bottom=min(cash_flow_only_value, scenario_value),
            color=CORAL if discount_change < 0 else TEAL,
            hatch="xx",
        )
        axis.bar(
            positions[3],
            scenario_value,
            color=MUTED_BLUE,
            hatch="..",
        )
        axis.hlines(base_value, positions[0] + 0.4, positions[1] - 0.4, color=INK, linewidth=0.9)
        axis.hlines(
            cash_flow_only_value,
            positions[1] + 0.4,
            positions[2] - 0.4,
            color=INK,
            linewidth=0.9,
        )
        axis.hlines(
            scenario_value,
            positions[2] + 0.4,
            positions[3] - 0.4,
            color=INK,
            linewidth=0.9,
        )

        annotations = (
            (positions[0], base_value, f"{base_value:.2f}", 6),
            (
                positions[1],
                max(base_value, cash_flow_only_value),
                f"{cash_flow_change:+.2f}",
                6,
            ),
            (
                positions[2],
                max(cash_flow_only_value, scenario_value),
                f"{discount_change:+.2f}",
                6,
            ),
            (positions[3], scenario_value, f"{scenario_value:.2f}", 6),
        )
        for x_value, y_value, label, offset in annotations:
            axis.annotate(
                label,
                (x_value, y_value),
                xytext=(0, offset),
                textcoords="offset points",
                color=INK,
                fontsize=10,
                fontweight="semibold",
                ha="center",
                va="bottom",
            )

        axis.set_xticks(
            positions,
            labels=(
                "Base value",
                "Cash-flow\nchange first",
                "Rate + premium\nchange second",
                "Sticky-inflation\nvalue",
            ),
        )
        axis.set_ylim(
            0,
            max(base_value, cash_flow_only_value, scenario_value) * 1.12,
        )
        axis.set_ylabel("Illustrative one-year value (MXN)")
        axis.set_title("What drives the one-year scenario valuation change?")
        style_axes(axis, grid_axis="y")
        add_figure_note(
            figure,
            "Source: book calculation. Sequential attribution changes cash flow first, "
            "then the combined relevant rate and risk premium. Contributions are exact "
            "for this ordering but are not causal estimates and would change under another order.",
        )
        figure.tight_layout(rect=(0, 0.14, 1, 1))
    return figure


def build_macro_fx_history_figure(
    macro_features: pd.DataFrame,
    *,
    sample_label: str,
    source_vintage_label: str,
    rebuild_label: str,
) -> Figure:
    """Show the differently scaled inputs used by the macro-to-FX case."""

    _require_columns(macro_features, _HISTORY_COLUMNS, "macro_features")
    display_series = {column: macro_features[column].dropna() for column in _HISTORY_COLUMNS}
    for column, series in display_series.items():
        _require_finite(series.to_frame(), f"macro_features[{column!r}]")

    with matplotlib_style():
        figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(
            display_series["banxico_target_rate"].index,
            100 * display_series["banxico_target_rate"],
            color=TEAL,
            linestyle="-",
            marker="o",
            markevery=max(1, len(display_series["banxico_target_rate"]) // 8),
            label="Banxico target rate",
        )
        axes[0].plot(
            display_series["cetes_28d"].index,
            100 * display_series["cetes_28d"],
            color=MUTED_BLUE,
            linestyle="--",
            marker="s",
            markevery=max(1, len(display_series["cetes_28d"]) // 8),
            label="CETES 28-day rate",
        )
        axes[0].set_title("Domestic rates used as model inputs")
        axes[0].set_ylabel("Annual rate (%)")
        axes[0].legend(loc="upper center", ncols=2)

        axes[1].plot(
            display_series["mexico_inflation_lag1m"].index,
            100 * display_series["mexico_inflation_lag1m"],
            color=AMBER_DARK,
            linestyle="-.",
            marker="^",
            markevery=max(1, len(display_series["mexico_inflation_lag1m"]) // 8),
            label="Mexico CPI YoY (lagged one month)",
        )
        axes[1].plot(
            display_series["us_10y"].index,
            100 * display_series["us_10y"],
            color=MUTED_BLUE,
            linestyle=":",
            marker="D",
            markevery=max(1, len(display_series["us_10y"]) // 8),
            label="US 10-year Treasury yield",
        )
        axes[1].set_title("Inflation availability proxy and external yield")
        axes[1].set_ylabel("Annual rate / yield (%)")
        axes[1].legend(loc="upper center", ncols=2)

        axes[2].plot(
            display_series["usd_mxn"].index,
            display_series["usd_mxn"],
            color=TEAL,
            linestyle="-",
            marker="o",
            markevery=max(1, len(display_series["usd_mxn"]) // 8),
            label="USD/MXN FIX",
        )
        axes[2].set_title("FX level used to construct momentum and the next-month target")
        axes[2].set_xlabel("Reference month-end")
        axes[2].set_ylabel("MXN per USD")
        axes[2].legend(loc="upper center")

        for axis in axes:
            axis.margins(y=0.15)
            style_axes(axis, grid_axis="y")
        _format_dates(axes[-1])
        figure.suptitle("What information enters the one-month macro-to-FX case?")
        add_figure_note(
            figure,
            _wrapped_note(
                "Sources: Banxico SIE SF61745, SF60633, and SF43718; "
                "DB.NOMICS IMF/CPI/M.MX.PCPI_IX and FED/H15/RIFLGFCY10_N.B. "
                f"Offline monthly panel | {sample_label} | source vintage "
                f"{source_vintage_label} | derived rebuild {rebuild_label}. "
                "Mexico inflation is the 12-month percent change in the CPI index. "
                "Month-end labels are period labels; the one-month lag approximates release "
                "availability and is not a real-time vintage."
            ),
        )
        figure.tight_layout(rect=(0, 0.12, 1, 0.94))
    return figure


def build_macro_fx_evaluation_figure(
    evaluation_frame: pd.DataFrame,
    mae_by_model: Mapping[str, float],
    *,
    sample_label: str,
    realization_end: str,
    source_vintage_label: str,
    rebuild_label: str,
) -> Figure:
    """Compare held-out predictions with two naive FX benchmarks."""

    _require_columns(evaluation_frame, _EVALUATION_COLUMNS, "evaluation_frame")
    _require_finite(evaluation_frame[list(_EVALUATION_COLUMNS)], "evaluation_frame")
    missing_metrics = set(_EVALUATION_COLUMNS[1:]) - set(mae_by_model)
    if missing_metrics:
        raise ValueError(f"mae_by_model is missing: {sorted(missing_metrics)}")
    if any(not np.isfinite(value) or value < 0 for value in mae_by_model.values()):
        raise ValueError("MAE values must be finite and non-negative")

    display = 100 * evaluation_frame
    model_order = list(_EVALUATION_COLUMNS[1:])
    mae_percent = [100 * mae_by_model[name] for name in model_order]

    with matplotlib_style():
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(10, 8),
            gridspec_kw={"height_ratios": [2.1, 1]},
        )
        path_axis, metric_axis = axes
        styles = {
            "Actual next-month return": (TEAL, "-", "o"),
            "Ridge prediction": (AMBER_DARK, "--", "s"),
            "Training-mean benchmark": (MUTED_BLUE, ":", "^"),
            "Zero-change benchmark": (INK, "-.", None),
        }
        for column in _EVALUATION_COLUMNS:
            color, linestyle, marker = styles[column]
            path_axis.plot(
                display.index,
                display[column],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=4,
                label=column,
            )
        path_axis.set_title("Held-out paths by forecast origin")
        path_axis.set_xlabel("Forecast origin (month-end)")
        path_axis.set_ylabel("Next-month USD/MXN log return (%)")
        style_axes(path_axis, grid_axis="y", show_zero_line=True)
        _format_dates(path_axis)

        bars = metric_axis.barh(
            model_order,
            mae_percent,
            color=[AMBER_DARK, MUTED_BLUE, INK],
            hatch=["//", "..", "xx"],
            alpha=0.88,
        )
        for bar, value in zip(bars, mae_percent, strict=True):
            metric_axis.text(
                value,
                bar.get_y() + bar.get_height() / 2,
                f" {value:.3f}%",
                color=INK,
                va="center",
                ha="left",
                fontsize=9,
            )
        metric_axis.set_title("Lower held-out MAE is better")
        metric_axis.set_xlabel("MAE (monthly log-return percentage points)")
        metric_axis.set_ylabel("")
        metric_scale = max(max(mae_percent), np.finfo(float).eps)
        metric_axis.set_xlim(0, metric_scale * 1.22)
        style_axes(metric_axis, grid_axis="x")

        figure.suptitle("Does Ridge beat transparent naive FX benchmarks?")
        figure.legend(
            *path_axis.get_legend_handles_labels(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncols=2,
            frameon=False,
        )
        add_figure_note(
            figure,
            _wrapped_note(
                "Target: Banxico SIE SF43718 monthly-end USD/MXN FIX log return. "
                f"Test origins {sample_label}; last realization {realization_end}; "
                f"source vintage {source_vintage_label}; derived rebuild {rebuild_label}. "
                "The zero-change benchmark "
                "is a random-walk level forecast. Results describe this frozen holdout only."
            ),
        )
        figure.tight_layout(rect=(0, 0.115, 1, 0.86))
    return figure


def build_macro_fx_scenario_stability_figure(
    sensitivity_bps: pd.DataFrame,
    *,
    model_mae_bps: float,
    fit_window_labels: Mapping[str, str],
    sample_label: str,
    source_vintage_label: str,
    rebuild_label: str,
) -> Figure:
    """Compare scenario deltas across estimation windows and against error scale."""

    _require_columns(sensitivity_bps, _FIT_COLUMNS, "sensitivity_bps")
    _require_finite(sensitivity_bps[list(_FIT_COLUMNS)], "sensitivity_bps")
    missing_labels = set(_FIT_COLUMNS) - set(fit_window_labels)
    if missing_labels:
        raise ValueError(f"fit_window_labels is missing: {sorted(missing_labels)}")
    if not np.isfinite(model_mae_bps) or model_mae_bps <= 0:
        raise ValueError("model_mae_bps must be finite and positive")

    largest_sensitivity = float(sensitivity_bps[list(_FIT_COLUMNS)].abs().to_numpy().max())
    positions = np.arange(len(sensitivity_bps))

    with matplotlib_style():
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(10, 8),
            gridspec_kw={"height_ratios": [2.2, 1]},
        )
        stability_axis, scale_axis = axes

        for position, (_, row) in zip(
            positions,
            sensitivity_bps.iterrows(),
            strict=True,
        ):
            stability_axis.plot(
                [row["Evaluation fit"], row["Updated fit"]],
                [position, position],
                color=SOFT_GRID,
                linewidth=2.2,
                zorder=1,
            )
        stability_axis.scatter(
            sensitivity_bps["Evaluation fit"],
            positions,
            color=TEAL,
            marker="o",
            s=55,
            label=fit_window_labels["Evaluation fit"],
            zorder=3,
        )
        stability_axis.scatter(
            sensitivity_bps["Updated fit"],
            positions,
            color=AMBER_DARK,
            marker="s",
            s=55,
            label=fit_window_labels["Updated fit"],
            zorder=3,
        )
        for position, (_, row) in zip(
            positions,
            sensitivity_bps.iterrows(),
            strict=True,
        ):
            for fit_name in _FIT_COLUMNS:
                value = float(row[fit_name])
                stability_axis.annotate(
                    f"{value:+.1f}",
                    (value, position),
                    xytext=(0, 7 if fit_name == "Updated fit" else -12),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=INK,
                )
        stability_axis.set_yticks(positions, labels=sensitivity_bps.index)
        stability_axis.invert_yaxis()
        stability_axis.set_title("Scenario deltas reverse across estimation windows")
        stability_axis.set_xlabel(
            "Change vs each fit's baseline (basis points in one-month log-return units)"
        )
        style_axes(stability_axis, grid_axis="x")
        stability_axis.axvline(0, color=INK, linewidth=0.9, alpha=0.75)

        magnitude_names = ["Ridge test MAE", "Largest absolute scenario delta"]
        magnitude_values = [model_mae_bps, largest_sensitivity]
        bars = scale_axis.barh(
            magnitude_names,
            magnitude_values,
            color=[AMBER_DARK, MUTED_BLUE],
            hatch=["//", ".."],
            alpha=0.88,
        )
        for bar, value in zip(bars, magnitude_values, strict=True):
            scale_axis.text(
                value,
                bar.get_y() + bar.get_height() / 2,
                f" {value:.1f} bp",
                color=INK,
                va="center",
                ha="left",
                fontsize=9,
            )
        if largest_sensitivity <= model_mae_bps:
            scale_title = "Scenario movement is small relative to held-out error"
        else:
            scale_title = "Scenario movement exceeds held-out error scale"
        scale_axis.set_title(scale_title)
        scale_axis.set_xlabel("Absolute magnitude (basis points in monthly log-return units)")
        scale_axis.set_xlim(0, max(model_mae_bps, largest_sensitivity) * 1.18)
        style_axes(scale_axis, grid_axis="x")

        figure.suptitle("Are the rejected model's scenario sensitivities stable?")
        figure.legend(
            *stability_axis.get_legend_handles_labels(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncols=2,
            frameon=False,
        )
        add_figure_note(
            figure,
            _wrapped_note(
                "The MAE bar is an error-scale comparison, not a prediction interval. "
                f"Offline diagnostic | {sample_label} | source vintage "
                f"{source_vintage_label} | derived rebuild {rebuild_label}. "
                "Changing fit windows is a stability diagnostic; neither fit identifies causal FX shocks."
            ),
        )
        figure.tight_layout(rect=(0, 0.11, 1, 0.86))
    return figure


__all__ = [
    "build_fx_carry_break_even_figure",
    "build_macro_fx_evaluation_figure",
    "build_macro_fx_history_figure",
    "build_macro_fx_scenario_stability_figure",
    "build_scenario_valuation_bridge_figure",
    "build_supply_demand_identification_figure",
]
