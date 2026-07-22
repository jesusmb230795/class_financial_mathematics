"""Publication-safe figures for the Module 3 macro-to-FX case."""

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


def build_macro_fx_history_figure(
    macro_features: pd.DataFrame,
    *,
    sample_label: str,
    vintage_label: str,
) -> Figure:
    """Show the differently scaled inputs used by the macro-to-FX case."""

    _require_columns(macro_features, _HISTORY_COLUMNS, "macro_features")
    display_data = macro_features[list(_HISTORY_COLUMNS)].dropna()
    _require_finite(display_data, "macro_features")

    with matplotlib_style():
        figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(
            display_data.index,
            100 * display_data["banxico_target_rate"],
            color=TEAL,
            linestyle="-",
            marker="o",
            markevery=max(1, len(display_data) // 8),
            label="Banxico target rate",
        )
        axes[0].plot(
            display_data.index,
            100 * display_data["cetes_28d"],
            color=MUTED_BLUE,
            linestyle="--",
            marker="s",
            markevery=max(1, len(display_data) // 8),
            label="CETES 28-day rate",
        )
        axes[0].set_title("Domestic rates used as model inputs")
        axes[0].set_ylabel("Annual rate (%)")
        axes[0].legend(loc="upper center", ncols=2)

        axes[1].plot(
            display_data.index,
            100 * display_data["mexico_inflation_lag1m"],
            color=AMBER_DARK,
            linestyle="-.",
            marker="^",
            markevery=max(1, len(display_data) // 8),
            label="Mexico CPI YoY (lagged one month)",
        )
        axes[1].plot(
            display_data.index,
            100 * display_data["us_10y"],
            color=MUTED_BLUE,
            linestyle=":",
            marker="D",
            markevery=max(1, len(display_data) // 8),
            label="US 10-year Treasury yield",
        )
        axes[1].set_title("Inflation availability proxy and external yield")
        axes[1].set_ylabel("Annual rate / yield (%)")
        axes[1].legend(loc="upper center", ncols=2)

        axes[2].plot(
            display_data.index,
            display_data["usd_mxn"],
            color=TEAL,
            linestyle="-",
            marker="o",
            markevery=max(1, len(display_data) // 8),
            label="USD/MXN FIX",
        )
        axes[2].set_title(
            "FX level used to construct momentum and the next-month target"
        )
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
                f"Offline monthly snapshot | {sample_label} | generated {vintage_label}. "
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
    vintage_label: str,
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
                f"offline snapshot generated {vintage_label}. The zero-change benchmark "
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
    vintage_label: str,
) -> Figure:
    """Compare scenario deltas across estimation windows and against error scale."""

    _require_columns(sensitivity_bps, _FIT_COLUMNS, "sensitivity_bps")
    _require_finite(sensitivity_bps[list(_FIT_COLUMNS)], "sensitivity_bps")
    missing_labels = set(_FIT_COLUMNS) - set(fit_window_labels)
    if missing_labels:
        raise ValueError(f"fit_window_labels is missing: {sorted(missing_labels)}")
    if not np.isfinite(model_mae_bps) or model_mae_bps <= 0:
        raise ValueError("model_mae_bps must be finite and positive")

    largest_sensitivity = float(
        sensitivity_bps[list(_FIT_COLUMNS)].abs().to_numpy().max()
    )
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
        scale_axis.set_xlabel(
            "Absolute magnitude (basis points in monthly log-return units)"
        )
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
                f"Offline diagnostic | {sample_label} | snapshot generated {vintage_label}. "
                "Changing fit windows is a stability diagnostic; neither fit identifies causal FX shocks."
            ),
        )
        figure.tight_layout(rect=(0, 0.11, 1, 0.86))
    return figure


__all__ = [
    "build_macro_fx_evaluation_figure",
    "build_macro_fx_history_figure",
    "build_macro_fx_scenario_stability_figure",
]
