"""Publication-safe, deterministic analytical figures for Module 6.

The builders in this module accept already prepared classroom or snapshot data.
They perform no acquisition, simulation, calibration, or hidden fallback so that
notebook calculations and figure semantics remain independently testable.
"""

from __future__ import annotations

from collections.abc import Sequence
from textwrap import wrap

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
    SERIES_COLORS,
    SERIES_LINESTYLES,
    SERIES_MARKERS,
    TEAL,
    add_figure_note,
    matplotlib_style,
    style_axes,
)


_LOSS_COLUMN_CANDIDATES = (
    "collateral_loss",
    "collateral_loss_amount",
    "pool_loss",
    "pool_loss_amount",
    "loss_amount",
)
_MATURITY_COLUMN_CANDIDATES = ("maturity_years", "maturity")
_OBSERVED_YIELD_COLUMN_CANDIDATES = (
    "observed_yield",
    "market_yield",
    "input_yield",
    "synthetic_zero_rate_annual",
)
_FITTED_YIELD_COLUMN_CANDIDATES = ("fitted_yield", "fitted_zero_rate_annual")
_RATE_LABELS = {
    "policy_rate": "Banxico target rate",
    "banxico_target_rate": "Banxico target rate",
    "cetes_28d": "CETES 28-day rate",
    "tiie_28d": "TIIE 28-day rate",
}
_HATCHES = ("///", "\\\\", "...", "xx", "++")


def _require_source_note(source_note: str) -> str:
    """Return a normalized, non-empty source and method note."""

    if not isinstance(source_note, str) or not source_note.strip():
        raise ValueError("source_note must be a non-empty string")
    return " ".join(source_note.split())


def _require_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Validate that a tabular input contains at least one row and column."""

    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame")
    if frame.empty or frame.shape[1] == 0:
        raise ValueError(f"{name} must contain observations and columns")
    return frame


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _finite_values(values: object, name: str) -> np.ndarray:
    """Return a finite float array or raise an actionable validation error."""

    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values") from error
    if array.size == 0:
        raise ValueError(f"{name} must contain values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def _positive_scalar(value: float, name: str) -> float:
    array = _finite_values([value], name)
    scalar = float(array[0])
    if scalar <= 0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _first_present(frame: pd.DataFrame, candidates: Sequence[str], name: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"{name} must include one of these columns: {list(candidates)}")


def _display_label(value: object) -> str:
    label = str(value)
    if label in _RATE_LABELS:
        return _RATE_LABELS[label]
    if label.upper().startswith("PC"):
        return label.upper()
    return label.replace("_", " ").strip().title()


def _format_dates(axis: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _finish_figure(
    figure: Figure,
    source_note: str,
    *,
    method_note: str,
    top: float = 0.91,
    label_clearance: float = 0.0,
) -> None:
    note = f"{_require_source_note(source_note)} | {method_note}"
    add_figure_note(figure, note)
    note_lines = wrap(
        note,
        width=96,
        break_long_words=False,
        break_on_hyphens=False,
    )
    # `add_figure_note` places 9 pt text at y=0.005. Reserve space from the
    # actual wrapped line count, plus optional clearance for rotated labels.
    note_top = 0.005 + 0.026 * max(1, len(note_lines))
    bottom = min(0.36, max(0.13, note_top + 0.03 + label_clearance))
    figure.tight_layout(rect=(0, bottom, 1, top))


def build_securitization_waterfall_figure(
    allocation: pd.DataFrame,
    pool_notional: float,
    source_note: str,
) -> Figure:
    """Plot contractual tranche losses as percentages of pool notional.

    ``allocation`` contains one collateral-loss column plus one or more tranche
    loss columns. At every row, tranche losses must reconcile exactly to the
    collateral loss, which protects the conservation-of-loss teaching contract.
    """

    frame = _require_frame(allocation, "allocation")
    notional = _positive_scalar(pool_notional, "pool_notional")
    loss_column = _first_present(frame, _LOSS_COLUMN_CANDIDATES, "allocation")
    scenario_labels: list[str] | None = None
    if {"tranche", "tranche_loss"} <= set(frame.columns):
        key_column = "scenario" if "scenario" in frame else loss_column
        if frame.duplicated([key_column, "tranche"]).any():
            raise ValueError("allocation must contain one row per scenario and tranche")
        scenario_order = list(dict.fromkeys(frame[key_column].tolist()))
        if "pool_notional" in frame:
            embedded_notional = _finite_values(
                frame["pool_notional"],
                "allocation['pool_notional']",
            )
            if not np.allclose(embedded_notional, notional, rtol=0, atol=1e-10):
                raise ValueError("allocation pool_notional must match the function argument")
        if {"attachment", "detachment"} <= set(frame.columns):
            attachment = _finite_values(frame["attachment"], "allocation['attachment']")
            detachment = _finite_values(frame["detachment"], "allocation['detachment']")
            if (attachment < 0).any() or (detachment <= attachment).any():
                raise ValueError("tranche points must satisfy 0 <= attachment < detachment")
            tranche_order = (
                frame[["tranche", "attachment"]]
                .drop_duplicates()
                .sort_values("attachment")["tranche"]
                .tolist()
            )
        else:
            tranche_order = list(dict.fromkeys(frame["tranche"].tolist()))
        losses_by_scenario = frame.groupby(key_column, sort=False)[loss_column].agg(
            ["first", "nunique"]
        )
        if (losses_by_scenario["nunique"] != 1).any():
            raise ValueError("collateral_loss must be constant within each scenario")
        loss = losses_by_scenario.loc[scenario_order, "first"].to_numpy(dtype=float)
        pivoted = frame.pivot(index=key_column, columns="tranche", values="tranche_loss")
        pivoted = pivoted.reindex(index=scenario_order, columns=tranche_order)
        if pivoted.isna().any().any():
            raise ValueError("every scenario must allocate a loss to every tranche")
        tranche_columns = [str(column) for column in pivoted.columns]
        tranche_loss = pivoted.to_numpy(dtype=float)
        scenario_labels = [str(value) for value in scenario_order]
    else:
        tranche_columns = [column for column in frame.columns if column != loss_column]
        if not tranche_columns:
            raise ValueError("allocation must contain at least one tranche-loss column")
        loss = _finite_values(frame[loss_column], f"allocation[{loss_column!r}]")
        tranche_loss = _finite_values(frame[tranche_columns], "tranche losses")
    if tranche_loss.ndim != 2:
        raise ValueError("tranche losses must form a two-dimensional table")
    if (loss < 0).any() or (tranche_loss < 0).any():
        raise ValueError("collateral and tranche losses must be non-negative")
    if (loss > notional * (1 + 1e-12)).any():
        raise ValueError("collateral loss cannot exceed pool_notional")
    tolerance = max(1e-10, notional * 1e-9)
    if not np.allclose(tranche_loss.sum(axis=1), loss, rtol=1e-9, atol=tolerance):
        raise ValueError("tranche losses must reconcile to collateral loss at every row")
    if scenario_labels is None and len(np.unique(loss)) != len(loss):
        raise ValueError("collateral loss observations must be unique")

    order = np.argsort(loss, kind="stable")
    loss_percent = 100 * loss[order] / notional
    ordered_tranches = tranche_loss[order]
    if scenario_labels is not None:
        scenario_labels = [scenario_labels[index] for index in order]

    with matplotlib_style():
        figure, axis = plt.subplots(figsize=(10, 5.625))
        lower = np.zeros_like(loss_percent)
        if scenario_labels is not None:
            positions = np.arange(len(loss_percent), dtype=float)
            for index, column in enumerate(tranche_columns):
                values = 100 * ordered_tranches[:, index] / notional
                axis.bar(
                    positions,
                    values,
                    bottom=lower,
                    color=SERIES_COLORS[index % len(SERIES_COLORS)],
                    alpha=0.78,
                    edgecolor=INK,
                    linewidth=0.6,
                    hatch=_HATCHES[index % len(_HATCHES)],
                    label=f"{_display_label(column)} allocation",
                )
                lower += values
            labels = [
                f"{_display_label(label)}\n{loss_value:.1f}% loss"
                for label, loss_value in zip(scenario_labels, loss_percent, strict=True)
            ]
            axis.set_xticks(positions, labels=labels)
            axis.set_xlabel("Scenario and collateral loss (% of pool notional)")
        else:
            for index, column in enumerate(tranche_columns):
                values = 100 * ordered_tranches[:, index] / notional
                upper = lower + values
                color = SERIES_COLORS[index % len(SERIES_COLORS)]
                axis.fill_between(
                    loss_percent,
                    lower,
                    upper,
                    color=color,
                    alpha=0.72,
                    edgecolor=INK,
                    linewidth=0.5,
                    hatch=_HATCHES[index % len(_HATCHES)],
                    label=f"{_display_label(column)} allocation",
                )
                axis.plot(
                    loss_percent,
                    upper,
                    color=INK,
                    linestyle=SERIES_LINESTYLES[index % len(SERIES_LINESTYLES)],
                    linewidth=0.8,
                )
                lower = upper
            axis.set_xlim(0, max(float(loss_percent.max()), 1.0))
            axis.set_xlabel("Collateral loss (% of pool notional)")
        axis.set_ylim(0, max(float(lower.max()) * 1.08, 1.0))
        axis.set_ylabel("Allocated tranche loss (% of pool notional)")
        axis.set_title("Which tranche absorbs each increment of collateral loss?")
        axis.legend(loc="upper left", ncols=min(3, len(tranche_columns)))
        style_axes(axis, grid_axis="both")
        figure.suptitle("Contractual securitization loss waterfall")
        _finish_figure(
            figure,
            source_note,
            method_note=(
                "Loss conservation is enforced row by row; hatching and boundary styles "
                "duplicate the color encoding."
            ),
        )
    return figure


def build_bootstrap_curve_figure(curve: pd.DataFrame, source_note: str) -> Figure:
    """Plot annual spot and one-period forward rates from a bootstrap table."""

    frame = _require_frame(curve, "curve")
    maturity_column = _first_present(frame, _MATURITY_COLUMN_CANDIDATES, "curve")
    _require_columns(frame, ("spot_rate_annual", "forward_rate_annual"), "curve")
    maturity = _finite_values(frame[maturity_column], f"curve[{maturity_column!r}]")
    spot = _finite_values(frame["spot_rate_annual"], "curve['spot_rate_annual']")
    try:
        forward = np.asarray(frame["forward_rate_annual"], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("curve['forward_rate_annual'] must be numeric") from error
    if maturity.ndim != 1 or spot.ndim != 1 or forward.ndim != 1:
        raise ValueError("curve rate columns must be one-dimensional")
    if not (len(maturity) == len(spot) == len(forward)):
        raise ValueError("curve columns must have the same length")
    if (maturity <= 0).any() or len(np.unique(maturity)) != len(maturity):
        raise ValueError("maturities must be positive and unique")
    finite_forward = np.isfinite(forward)
    if not finite_forward.any():
        raise ValueError("forward_rate_annual must contain at least one finite rate")
    first_forward = int(np.flatnonzero(finite_forward)[0])
    if first_forward > 1 or not finite_forward[first_forward:].all():
        raise ValueError("only the first forward rate may be missing")

    order = np.argsort(maturity)
    maturity = maturity[order]
    spot = spot[order]
    forward = forward[order]
    par: np.ndarray | None = None
    if "par_rate_annual" in frame:
        par = _finite_values(frame["par_rate_annual"], "curve['par_rate_annual']")
        if par.ndim != 1 or len(par) != len(maturity):
            raise ValueError("curve['par_rate_annual'] must align with curve maturities")
        par = par[order]

    with matplotlib_style():
        figure, axis = plt.subplots(figsize=(10, 5.625))
        axis.plot(
            maturity,
            100 * spot,
            color=TEAL,
            linestyle="-",
            marker="o",
            label="Effective annual spot rate",
        )
        axis.plot(
            maturity,
            100 * forward,
            color=MUTED_BLUE,
            linestyle="--",
            marker="s",
            label="Effective annual one-period forward rate",
        )
        if par is not None:
            axis.plot(
                maturity,
                100 * par,
                color=AMBER_DARK,
                linestyle="-.",
                marker="^",
                label="Nominal annual par coupon (semiannual)",
            )
        axis.set_xlabel("Maturity (years)")
        axis.set_ylabel("Rate or nominal coupon (%)")
        curve_terms = (
            "effective spot/forward rates and nominal par coupons"
            if par is not None
            else "effective spot and implied forward rates"
        )
        axis.set_title(f"How do {curve_terms} differ by maturity?")
        axis.legend(loc="best")
        style_axes(axis, grid_axis="both", show_zero_line=True)
        figure.suptitle(
            "Bootstrapped effective rates and nominal par coupons"
            if par is not None
            else "Bootstrapped effective spot and forward curves"
        )
        _finish_figure(
            figure,
            source_note,
            method_note=(
                "Spot and forward inputs are effective annual rates; "
                "the par series is a nominal annual coupon convertible semiannually. "
                "Decimal inputs are displayed as percentages; the first forward rate "
                "is undefined by construction."
                if par is not None
                else "Spot and forward inputs are effective annual rates displayed as "
                "percentages; the first forward rate is undefined by construction."
            ),
        )
    return figure


def _path_frame(frame: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray]:
    _require_frame(frame, name)
    paths = _finite_values(frame, name)
    if paths.ndim != 2 or paths.shape[0] < 2 or paths.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two times and two paths")
    time = _finite_values(frame.index, f"{name}.index")
    if time.ndim != 1 or len(time) != paths.shape[0]:
        raise ValueError(f"{name}.index must be one-dimensional")
    if (np.diff(time) <= 0).any():
        raise ValueError(f"{name}.index must be strictly increasing in years")
    return time, paths


def build_short_rate_paths_figure(
    vasicek_paths: pd.DataFrame,
    cir_paths: pd.DataFrame,
    vasicek_theta: float,
    cir_theta: float,
    source_note: str,
) -> Figure:
    """Compare simulated short-rate distributions without path clutter."""

    vasicek_time, vasicek = _path_frame(vasicek_paths, "vasicek_paths")
    cir_time, cir = _path_frame(cir_paths, "cir_paths")
    theta_vasicek = float(_finite_values([vasicek_theta], "vasicek_theta")[0])
    theta_cir = float(_finite_values([cir_theta], "cir_theta")[0])

    model_specs = (
        ("Vasicek", vasicek_time, vasicek, theta_vasicek, TEAL, "o"),
        ("CIR", cir_time, cir, theta_cir, MUTED_BLUE, "s"),
    )
    with matplotlib_style():
        figure, axes = plt.subplots(1, 2, figsize=(10, 5.625), sharey=True)
        for axis, (name, time, paths, theta, color, marker) in zip(
            axes,
            model_specs,
            strict=True,
        ):
            lower, median, upper = np.quantile(paths, (0.05, 0.50, 0.95), axis=1)
            axis.fill_between(
                time,
                100 * lower,
                100 * upper,
                color=color,
                alpha=0.22,
                hatch="///" if name == "Vasicek" else "\\\\",
                edgecolor=color,
                linewidth=0.5,
                label="Pointwise 5th-95th percentiles",
            )
            axis.plot(
                time,
                100 * median,
                color=color,
                linestyle="-",
                marker=marker,
                markevery=max(1, len(time) // 10),
                label="Path median",
            )
            axis.axhline(
                100 * theta,
                color=INK,
                linestyle="--",
                linewidth=1.2,
                label=r"Long-run mean $\theta$",
            )
            axis.set_title(f"{name}: distribution across simulated paths")
            axis.set_xlabel("Simulation horizon (years)")
            axis.legend(loc="best")
            style_axes(axis, grid_axis="both", show_zero_line=True)
        axes[0].set_ylabel("Annual short rate (%)")
        figure.suptitle("What range of short rates do the two models generate?")
        _finish_figure(
            figure,
            source_note,
            method_note=(
                "Lines show cross-path medians and bands show pointwise 5th-95th "
                "percentiles; these are model simulations, not confidence intervals."
            ),
            top=0.90,
        )
    return figure


def build_nelson_siegel_diagnostic_figure(
    curve: pd.DataFrame,
    dense_maturities: Sequence[float],
    fitted_yields: Sequence[float],
    source_note: str,
) -> Figure:
    """Compare Nelson-Siegel fitted yields with inputs and residuals."""

    frame = _require_frame(curve, "curve")
    maturity_column = _first_present(frame, _MATURITY_COLUMN_CANDIDATES, "curve")
    observed_column = _first_present(
        frame,
        _OBSERVED_YIELD_COLUMN_CANDIDATES,
        "curve",
    )
    is_synthetic_input = observed_column == "synthetic_zero_rate_annual"
    maturity = _finite_values(frame[maturity_column], f"curve[{maturity_column!r}]")
    observed = _finite_values(frame[observed_column], f"curve[{observed_column!r}]")
    dense = _finite_values(dense_maturities, "dense_maturities")
    fitted_dense = _finite_values(fitted_yields, "fitted_yields")
    if any(array.ndim != 1 for array in (maturity, observed, dense, fitted_dense)):
        raise ValueError("curve and fitted-yield inputs must be one-dimensional")
    if len(maturity) != len(observed) or len(dense) != len(fitted_dense):
        raise ValueError("maturity and yield inputs must have matching lengths")
    if (maturity <= 0).any() or (dense <= 0).any():
        raise ValueError("maturities must be positive")
    if (np.diff(dense) <= 0).any():
        raise ValueError("dense_maturities must be strictly increasing")
    if dense[0] > maturity.min() or dense[-1] < maturity.max():
        raise ValueError("dense_maturities must span every input maturity")

    fitted_column = next(
        (column for column in _FITTED_YIELD_COLUMN_CANDIDATES if column in frame),
        None,
    )
    if fitted_column is not None:
        fitted_input = _finite_values(frame[fitted_column], f"curve[{fitted_column!r}]")
    else:
        fitted_input = np.interp(maturity, dense, fitted_dense)
    if len(fitted_input) != len(maturity):
        raise ValueError("curve['fitted_yield'] must align with curve maturities")
    residual_bp = 10_000 * (observed - fitted_input)
    order = np.argsort(maturity)
    maturity = maturity[order]
    observed = observed[order]
    residual_bp = residual_bp[order]
    bar_width = max(0.08, min(0.60, float(np.min(np.diff(maturity))) * 0.55))

    with matplotlib_style():
        figure, axes = plt.subplots(1, 2, figsize=(10, 5.625))
        fit_axis, residual_axis = axes
        fit_axis.plot(
            dense,
            100 * fitted_dense,
            color=TEAL,
            linestyle="-",
            label="Nelson-Siegel fitted zero curve",
        )
        fit_axis.scatter(
            maturity,
            100 * observed,
            facecolor="white",
            edgecolor=INK,
            marker="o",
            linewidth=1.1,
            label=(
                "Synthetic zero-coupon inputs"
                if is_synthetic_input
                else "Input zero-coupon yields"
            ),
            zorder=3,
        )
        fit_axis.set_title(
            "Synthetic inputs versus smooth fitted curve"
            if is_synthetic_input
            else "Input yields versus smooth fitted curve"
        )
        fit_axis.set_xlabel("Maturity (years)")
        fit_axis.set_ylabel("Annual zero-coupon rate (%)")
        fit_axis.legend(loc="best")
        style_axes(fit_axis, grid_axis="both", show_zero_line=True)

        bars = residual_axis.bar(
            maturity,
            residual_bp,
            width=bar_width,
            color=[TEAL if value >= 0 else CORAL for value in residual_bp],
            edgecolor=INK,
            linewidth=0.6,
        )
        for bar, value in zip(bars, residual_bp, strict=True):
            bar.set_hatch("///" if value >= 0 else "\\\\")
        residual_axis.set_title(
            "Synthetic input minus fitted rate"
            if is_synthetic_input
            else "Input minus fitted yield"
        )
        residual_axis.set_xlabel("Maturity (years)")
        residual_axis.set_ylabel("Fit residual (basis points)")
        style_axes(residual_axis, grid_axis="both", show_zero_line=True)

        figure.suptitle(
            "Does the Nelson-Siegel fit preserve the synthetic input curve shape?"
            if is_synthetic_input
            else "Does the Nelson-Siegel fit preserve the input curve shape?"
        )
        _finish_figure(
            figure,
            source_note,
            method_note=(
                "Rates are annual decimals displayed as percentages; residuals equal "
                "input minus fitted rate in basis points."
            ),
            top=0.90,
        )
    return figure


def build_rate_history_figure(rate_history: pd.DataFrame, source_note: str) -> Figure:
    """Plot an explicitly aligned Mexican rate panel with display units."""

    frame = _require_frame(rate_history, "rate_history")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("rate_history must use a DatetimeIndex")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("rate_history index must be unique and increasing")
    rates = _finite_values(frame, "rate_history")
    if rates.ndim != 2:
        raise ValueError("rate_history must be two-dimensional")
    methodology_break = frame.attrs.get("methodology_break")
    break_date: pd.Timestamp | None = None
    break_label: str | None = None
    if methodology_break is not None and "tiie_28d" in frame.columns:
        if not isinstance(methodology_break, dict) or "effective_date" not in methodology_break:
            raise ValueError("methodology_break must provide an effective_date")
        try:
            break_date = pd.Timestamp(methodology_break["effective_date"])
        except (TypeError, ValueError) as error:
            raise ValueError("methodology_break effective_date must be a valid date") from error
        if pd.isna(break_date):
            raise ValueError("methodology_break effective_date must be a valid date")
        if frame.index.tz is not None and break_date.tz is None:
            break_date = break_date.tz_localize(frame.index.tz)
        if not (frame.index.min() <= break_date <= frame.index.max()):
            break_date = None
        else:
            series_id = methodology_break.get("series_id", "TIIE 28-day")
            break_label = f"TIIE methodology change ({break_date:%Y-%m-%d}; {series_id})"

    with matplotlib_style():
        figure, axis = plt.subplots(figsize=(10, 5.625))
        for index, column in enumerate(frame.columns):
            axis.plot(
                frame.index,
                100 * rates[:, index],
                color=SERIES_COLORS[index % len(SERIES_COLORS)],
                linestyle=SERIES_LINESTYLES[index % len(SERIES_LINESTYLES)],
                marker=SERIES_MARKERS[index % len(SERIES_MARKERS)],
                markevery=max(1, len(frame) // 12),
                markersize=4,
                label=_display_label(column),
            )
        axis.set_xlabel("Aligned observation date")
        axis.set_ylabel("Annual rate (%)")
        axis.set_title("How did the selected Mexican rates move over the observed sample?")
        if break_date is not None and break_label is not None:
            axis.axvline(
                break_date,
                color=INK,
                linestyle=":",
                linewidth=1.3,
                label=break_label,
            )
        axis.legend(loc="best", ncols=min(3, len(frame.columns)))
        _format_dates(axis)
        style_axes(axis, grid_axis="y", show_zero_line=True)
        figure.suptitle("Mexican interest-rate history")
        actual_sample = (
            f"Actual inclusive sample: {frame.index.min():%Y-%m-%d} to "
            f"{frame.index.max():%Y-%m-%d}."
        )
        break_note = ""
        if break_date is not None and break_label is not None:
            break_note = f" {break_label} is marked with a labeled dotted line."
        _finish_figure(
            figure,
            source_note,
            method_note=(
                f"{actual_sample} Decimal annual rates are displayed as percentages; "
                f"lines, markers, and labels duplicate color.{break_note}"
            ),
        )
    return figure


def build_rate_panel_diagnostics_figure(
    components: pd.DataFrame,
    scenarios: pd.DataFrame,
    source_note: str,
) -> Figure:
    """Pair standardized PCA loadings with categorical scenario shocks."""

    component_frame = _require_frame(components, "components")
    scenario_frame = _require_frame(scenarios, "scenarios")
    _require_columns(scenario_frame, ("base",), "scenarios")
    loading_values = _finite_values(component_frame, "components")
    scenario_values = _finite_values(scenario_frame, "scenarios")
    if loading_values.ndim != 2 or scenario_values.ndim != 2:
        raise ValueError("components and scenarios must be two-dimensional")
    if len(component_frame.index) > 5:
        raise ValueError("components should contain at most five principal components")
    component_series = list(component_frame.columns)
    scenario_series = list(scenario_frame.index)
    if set(component_series) != set(scenario_series):
        raise ValueError("component columns and scenario rows must name the same rate series")
    scenario_frame = scenario_frame.loc[component_series]
    scenario_columns = [column for column in scenario_frame.columns if column != "base"]
    if not scenario_columns:
        raise ValueError("scenarios must contain at least one shocked scenario besides base")
    shocks_bp = scenario_frame[scenario_columns].subtract(
        scenario_frame["base"],
        axis="index",
    ) * 10_000

    x = np.arange(len(component_series), dtype=float)
    with matplotlib_style():
        figure, axes = plt.subplots(1, 2, figsize=(10, 5.625))
        loading_axis, scenario_axis = axes

        component_count = len(component_frame.index)
        bar_width = min(0.24, 0.78 / component_count)
        for index, component in enumerate(component_frame.index):
            offset = (index - (component_count - 1) / 2) * bar_width
            bars = loading_axis.bar(
                x + offset,
                component_frame.loc[component].to_numpy(dtype=float),
                width=bar_width,
                color=SERIES_COLORS[index % len(SERIES_COLORS)],
                edgecolor=INK,
                linewidth=0.5,
                label=_display_label(component),
            )
            for bar in bars:
                bar.set_hatch(_HATCHES[index % len(_HATCHES)])
        loading_axis.set_title("Standardized PCA loadings by named rate")
        loading_axis.set_ylabel("Standardized loading")
        loading_axis.legend(loc="best")
        style_axes(loading_axis, grid_axis="y", show_zero_line=True)

        scenario_count = len(scenario_columns)
        offsets = np.linspace(-0.22, 0.22, scenario_count) if scenario_count > 1 else [0.0]
        for index, (column, offset) in enumerate(
            zip(scenario_columns, offsets, strict=True),
        ):
            values = shocks_bp[column].to_numpy(dtype=float)
            positions = x + offset
            color = SERIES_COLORS[index % len(SERIES_COLORS)]
            scenario_axis.vlines(
                positions,
                0,
                values,
                color=color,
                linestyle=SERIES_LINESTYLES[index % len(SERIES_LINESTYLES)],
                linewidth=1.2,
                alpha=0.85,
            )
            scenario_axis.scatter(
                positions,
                values,
                color=color,
                edgecolor=INK,
                linewidth=0.5,
                marker=SERIES_MARKERS[index % len(SERIES_MARKERS)],
                s=48,
                label=_display_label(column),
                zorder=3,
            )
        scenario_axis.set_title("Constructed shock relative to base")
        scenario_axis.set_ylabel("Rate shock (basis points)")
        scenario_axis.legend(loc="best")
        style_axes(scenario_axis, grid_axis="y", show_zero_line=True)

        labels = [_display_label(column) for column in component_series]
        for axis in axes:
            axis.set_xticks(x, labels=labels, rotation=18, ha="right")
            axis.set_xlabel("Named rate series (categorical, not curve tenors)")
        figure.suptitle("What do the statistical factors and named stress scenarios change?")
        _finish_figure(
            figure,
            source_note,
            method_note=(
                "PCA uses standardized rate changes; scenario dots show constructed "
                "level changes from base in basis points and do not connect unlike tenors."
            ),
            top=0.90,
            label_clearance=0.055,
        )
    return figure


__all__ = [
    "build_bootstrap_curve_figure",
    "build_nelson_siegel_diagnostic_figure",
    "build_rate_history_figure",
    "build_rate_panel_diagnostics_figure",
    "build_securitization_waterfall_figure",
    "build_short_rate_paths_figure",
]
