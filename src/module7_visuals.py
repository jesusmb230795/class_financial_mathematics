"""Publication-safe analytical figures for Module 7.

The builders in this module validate their financial sign and unit contracts,
copy caller-owned data, and return Matplotlib figures without displaying or
writing them.  Return and risk inputs use decimal units; figures convert those
values to percentage points for display.  VaR and Expected Shortfall use the
book's non-negative loss convention.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import norm, skewnorm

from src.visual_style import (
    AMBER_DARK,
    BACKGROUND,
    CORAL,
    INK,
    INLINE_FIGURE_DPI,
    MUTED_BLUE,
    SOFT_GRID,
    TEAL,
    add_figure_note,
    matplotlib_style,
    style_axes,
)


def _finite_scalar(
    value: Real,
    name: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    """Return one validated scalar without silently accepting booleans."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real scalar")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real scalar") from error
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if positive and normalized <= 0:
        raise ValueError(f"{name} must be positive")
    if non_negative and normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _validate_index(index: pd.Index, name: str) -> None:
    """Reject ambiguous plotting indexes."""

    if isinstance(index, pd.MultiIndex):
        raise ValueError(f"{name} must use a one-dimensional index")
    if index.has_duplicates:
        raise ValueError(f"{name} index cannot contain duplicates")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be sorted in increasing order")
    if isinstance(index, pd.DatetimeIndex) and index.hasnans:
        raise ValueError(f"{name} index cannot contain missing dates")


def _numeric_series(
    values: pd.Series | Sequence[float] | np.ndarray,
    name: str,
    *,
    minimum_observations: int = 2,
    allow_missing: bool = False,
    index: pd.Index | None = None,
) -> pd.Series:
    """Return an isolated numeric series after validating shape and finiteness."""

    if isinstance(values, pd.Series):
        isolated = values.copy(deep=True)
        _validate_index(isolated.index, name)
    else:
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if index is not None and len(index) != len(array):
            raise ValueError(f"{name} must have the same length as its plotting index")
        isolated = pd.Series(array.copy(), index=index)

    numeric = pd.to_numeric(isolated, errors="coerce")
    invalid_numeric = isolated.notna() & numeric.isna()
    if invalid_numeric.any():
        raise ValueError(f"{name} must contain numeric values")
    array = numeric.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(array).any():
        raise ValueError(f"{name} cannot contain infinite values")
    if not allow_missing and np.isnan(array).any():
        raise ValueError(f"{name} cannot contain missing values")
    if int(np.isfinite(array).sum()) < minimum_observations:
        raise ValueError(f"{name} must contain at least {minimum_observations} finite observations")
    numeric = numeric.astype(float)
    numeric.attrs = dict(getattr(values, "attrs", {}))
    return numeric


def _numeric_matrix(values: object, name: str) -> np.ndarray:
    """Return a finite, copied simulation matrix in time-by-path orientation."""

    try:
        matrix = np.asarray(values, dtype=float).copy()
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric two-dimensional array") from error
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional time-by-path array")
    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        raise ValueError(f"{name} must contain at least 3 times and 2 paths")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _source_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, Sequence):
        rendered = "; ".join(str(item).strip() for item in value if str(item).strip())
        return rendered or None
    return str(value).strip() or None


def _figure_note(*objects: object, model_note: str) -> str:
    """Build a compact provenance note from caller-supplied pandas metadata."""

    metadata: dict[str, object] = {}
    for item in objects:
        attrs = getattr(item, "attrs", None)
        if isinstance(attrs, dict):
            metadata.update(attrs)
    source = _source_value(metadata.get("source") or metadata.get("sources"))
    mode = _source_value(metadata.get("data_mode"))
    source_text = source if source is not None else "caller-supplied data"
    mode_text = f" Data mode: {mode}." if mode is not None else ""
    return f"Source: {source_text}.{mode_text} Model note: {model_note}"


def _sample_description(series: pd.Series) -> str:
    first = series.index[0]
    last = series.index[-1]
    if isinstance(series.index, pd.DatetimeIndex):
        return f"{first:%Y-%m-%d} to {last:%Y-%m-%d}; n={len(series)}"
    return f"n={len(series)} ordered observations"


def _set_horizontal_axis(axis: plt.Axes, index: pd.Index) -> None:
    axis.set_xlabel("Date" if isinstance(index, pd.DatetimeIndex) else "Observation")


def _resolve_column(frame: pd.DataFrame, requested: str, aliases: Sequence[str], name: str) -> str:
    candidates = (requested, *aliases)
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"comparison is missing {name}; tried {list(dict.fromkeys(candidates))}")


def build_option_payoff_figure(
    terminal_prices: pd.Series | Sequence[float] | np.ndarray,
    strike: float,
) -> Figure:
    """Plot expiry call and put payoffs in the same price units as the underlying."""

    prices = _numeric_series(terminal_prices, "terminal_prices")
    strike_value = _finite_scalar(strike, "strike", positive=True)
    if (prices <= 0).any():
        raise ValueError("terminal_prices must be positive")
    ordered = np.sort(prices.to_numpy(copy=True))
    call_payoff = np.maximum(ordered - strike_value, 0.0)
    put_payoff = np.maximum(strike_value - ordered, 0.0)

    with matplotlib_style():
        figure, axis = plt.subplots(figsize=(7.5, 4.5), dpi=INLINE_FIGURE_DPI)
        axis.plot(ordered, call_payoff, color=TEAL, linestyle="-", label="Call payoff")
        axis.plot(ordered, put_payoff, color=CORAL, linestyle="--", label="Put payoff")
        axis.axvline(
            strike_value,
            color=INK,
            linestyle=":",
            linewidth=1.2,
            label=f"Strike = {strike_value:g}",
        )
        axis.set_title("Expiry payoffs before the option premium")
        axis.set_xlabel("Underlying price at expiry (price units)")
        axis.set_ylabel("Payoff (same price units)")
        axis.legend(loc="upper center", ncol=3)
        style_axes(axis, grid_axis="both", show_zero_line=True)
        figure.suptitle(
            "European call and put payoff profiles",
            x=0.06,
            y=0.97,
            ha="left",
        )
        add_figure_note(
            figure,
            _figure_note(
                terminal_prices,
                model_note=(
                    "Deterministic payoff identities max(S_T-K, 0) and max(K-S_T, 0); "
                    "premium, discounting, and probability are excluded."
                ),
            ),
        )
        figure.subplots_adjust(left=0.11, right=0.98, top=0.82, bottom=0.23)
        return figure


def build_convergence_figure(
    comparison: pd.DataFrame,
    *,
    x: str = "steps",
    estimate: str = "crr_call",
    benchmark: str = "black_scholes_call",
    error: str = "error",
) -> Figure:
    """Compare a numerical option estimate with a closed-form benchmark."""

    if not isinstance(comparison, pd.DataFrame) or comparison.empty:
        raise ValueError("comparison must be a non-empty pandas DataFrame")
    frame = comparison.copy(deep=True)
    x_column = _resolve_column(frame, x, ("steps", "n_steps", "tree_steps"), "step column")
    estimate_column = _resolve_column(
        frame,
        estimate,
        ("crr_call", "binomial_price", "crr_price", "estimate", "numerical_price"),
        "estimate column",
    )
    benchmark_column = _resolve_column(
        frame,
        benchmark,
        (
            "black_scholes_call",
            "black_scholes_price",
            "benchmark",
            "closed_form_price",
        ),
        "benchmark column",
    )
    error_candidates = (error, "error", "pricing_error", "signed_error")
    error_column = next((item for item in error_candidates if item in frame.columns), None)

    selected = frame[[x_column, estimate_column, benchmark_column]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if selected.isna().any().any() or not np.isfinite(selected.to_numpy()).all():
        raise ValueError("convergence columns must contain only finite numeric values")
    if len(selected) < 2:
        raise ValueError("comparison must contain at least two convergence points")
    if (selected[x_column] <= 0).any() or selected[x_column].duplicated().any():
        raise ValueError("step values must be positive and unique")
    selected = selected.sort_values(x_column)
    if error_column is None:
        signed_error = selected[estimate_column] - selected[benchmark_column]
    else:
        signed_error = pd.to_numeric(frame.loc[selected.index, error_column], errors="coerce")
        if signed_error.isna().any() or not np.isfinite(signed_error.to_numpy()).all():
            raise ValueError("error column must contain only finite numeric values")
        implied_error = selected[estimate_column] - selected[benchmark_column]
        if not np.allclose(signed_error, implied_error, rtol=1e-8, atol=1e-10):
            raise ValueError("error must equal estimate minus benchmark")

    step_values = selected[x_column].to_numpy()
    with matplotlib_style():
        figure, (price_axis, error_axis) = plt.subplots(
            2,
            1,
            figsize=(7.5, 5.7),
            dpi=INLINE_FIGURE_DPI,
            sharex=True,
            gridspec_kw={"height_ratios": (1.5, 1)},
        )
        price_axis.plot(
            step_values,
            selected[estimate_column],
            color=TEAL,
            marker="o",
            linestyle="-",
            label="Numerical estimate",
        )
        price_axis.plot(
            step_values,
            selected[benchmark_column],
            color=INK,
            linestyle="--",
            label="Closed-form benchmark",
        )
        price_axis.set_title("Price convergence against a fixed benchmark")
        price_axis.set_ylabel("Option value (price units)")
        price_axis.legend(loc="best")
        style_axes(price_axis, grid_axis="both")

        error_axis.plot(
            step_values,
            signed_error,
            color=AMBER_DARK,
            marker="s",
            linestyle="-",
            label="Estimate − benchmark",
        )
        error_axis.axhline(0, color=INK, linestyle="--", linewidth=1.0)
        error_axis.set_xlabel("Tree steps (count)")
        error_axis.set_ylabel("Signed error\n(price units)")
        error_axis.set_title("Discretization error")
        style_axes(error_axis, grid_axis="both")
        figure.suptitle("Numerical option-pricing convergence", x=0.06, y=0.98, ha="left")
        add_figure_note(
            figure,
            _figure_note(
                comparison,
                model_note=(
                    "The benchmark is held fixed; agreement under shared assumptions is an "
                    "implementation diagnostic, not market validation."
                ),
            ),
        )
        figure.subplots_adjust(left=0.13, right=0.98, top=0.86, bottom=0.19, hspace=0.48)
        return figure


def build_implied_volatility_figure(chain: pd.DataFrame) -> Figure:
    """Plot a strike slice of implied volatility and optional synthetic truth."""

    if not isinstance(chain, pd.DataFrame) or chain.empty:
        raise ValueError("chain must be a non-empty pandas DataFrame")
    frame = chain.copy(deep=True)
    strike_column = _resolve_column(frame, "strike", ("exercise_price", "K"), "strike column")
    implied_column = _resolve_column(
        frame,
        "implied_volatility",
        ("implied_vol", "iv", "sigma_implied"),
        "implied-volatility column",
    )
    generating_column = next(
        (
            column
            for column in (
                "generating_volatility",
                "true_volatility",
                "input_volatility",
                "synthetic_volatility",
            )
            if column in frame.columns
        ),
        None,
    )
    columns = [strike_column, implied_column]
    if generating_column is not None:
        columns.append(generating_column)
    selected = frame[columns].apply(pd.to_numeric, errors="coerce")
    if selected.isna().any().any() or not np.isfinite(selected.to_numpy()).all():
        raise ValueError("strike and volatility columns must contain only finite numeric values")
    if len(selected) < 3:
        raise ValueError("chain must contain at least three strikes")
    if (selected[strike_column] <= 0).any() or selected[strike_column].duplicated().any():
        raise ValueError("strikes must be positive and unique")
    volatility_columns = [implied_column] + (
        [generating_column] if generating_column is not None else []
    )
    if (selected[volatility_columns] <= 0).any().any():
        raise ValueError("volatilities must be positive decimal values")
    selected = selected.sort_values(strike_column)

    with matplotlib_style():
        figure, axis = plt.subplots(figsize=(7.5, 4.5), dpi=INLINE_FIGURE_DPI)
        if generating_column is not None:
            axis.plot(
                selected[strike_column],
                selected[generating_column] * 100,
                color=MUTED_BLUE,
                marker="s",
                markerfacecolor=BACKGROUND,
                linestyle="--",
                label="Synthetic generating volatility",
                zorder=2,
            )
        axis.plot(
            selected[strike_column],
            selected[implied_column] * 100,
            color=TEAL,
            marker="o",
            linestyle="-",
            label="Recovered implied volatility",
            zorder=3,
        )
        axis.set_title("One-maturity strike slice")
        axis.set_xlabel("Strike (price units)")
        axis.set_ylabel("Annualized volatility (%)")
        axis.legend(loc="best")
        style_axes(axis, grid_axis="both")
        figure.suptitle("Implied-volatility skew or smile", x=0.06, y=0.97, ha="left")
        add_figure_note(
            figure,
            _figure_note(
                chain,
                model_note=(
                    "Each point is a Black-Scholes inversion at one maturity; the curve is "
                    "not a forecast of realized volatility."
                ),
            ),
        )
        figure.subplots_adjust(left=0.12, right=0.98, top=0.82, bottom=0.23)
        return figure


def build_heston_diagnostics_figure(
    heston_spots: object,
    heston_variances: object,
    gbm_paths: object,
    *,
    maturity: float = 1.0,
) -> Figure:
    """Show full-horizon simulation bands and terminal-model distributions."""

    spots = _numeric_matrix(heston_spots, "heston_spots")
    variances = _numeric_matrix(heston_variances, "heston_variances")
    gbm = _numeric_matrix(gbm_paths, "gbm_paths")
    maturity_value = _finite_scalar(maturity, "maturity", positive=True)
    if spots.shape != variances.shape:
        raise ValueError("heston_spots and heston_variances must have the same shape")
    if gbm.shape[0] != spots.shape[0]:
        raise ValueError("gbm_paths must use the same time grid as Heston paths")
    if (spots <= 0).any() or (gbm <= 0).any():
        raise ValueError("spot paths must remain positive")
    if (variances < 0).any():
        raise ValueError("heston_variances cannot be negative")

    time = np.linspace(0.0, maturity_value, spots.shape[0])
    heston_quantiles = np.quantile(spots, [0.05, 0.50, 0.95], axis=1)
    gbm_quantiles = np.quantile(gbm, [0.05, 0.50, 0.95], axis=1)
    volatility_quantiles = np.sqrt(np.quantile(variances, [0.05, 0.50, 0.95], axis=1))

    with matplotlib_style():
        figure = plt.figure(figsize=(7.5, 6.3), dpi=INLINE_FIGURE_DPI)
        grid = figure.add_gridspec(2, 2, height_ratios=(1.15, 1), hspace=0.48, wspace=0.34)
        spot_axis = figure.add_subplot(grid[0, :])
        volatility_axis = figure.add_subplot(grid[1, 0])
        terminal_axis = figure.add_subplot(grid[1, 1])

        spot_axis.fill_between(
            time,
            heston_quantiles[0],
            heston_quantiles[2],
            color=TEAL,
            alpha=0.18,
            label="Heston 5th–95th percentiles",
        )
        spot_axis.plot(
            time,
            heston_quantiles[1],
            color=TEAL,
            linestyle="-",
            label="Heston median",
        )
        spot_axis.plot(
            time,
            gbm_quantiles[1],
            color=MUTED_BLUE,
            linestyle="--",
            label="Constant-volatility GBM median",
        )
        spot_axis.set_title("Full-horizon spot-path distribution")
        spot_axis.set_xlabel("Time (years)")
        spot_axis.set_ylabel("Spot (price units)")
        spot_axis.legend(loc="best", ncol=2)
        style_axes(spot_axis, grid_axis="both")

        volatility_axis.fill_between(
            time,
            volatility_quantiles[0] * 100,
            volatility_quantiles[2] * 100,
            color=AMBER_DARK,
            alpha=0.20,
            label="5th–95th percentiles",
        )
        volatility_axis.plot(
            time,
            volatility_quantiles[1] * 100,
            color=AMBER_DARK,
            linestyle="-",
            label="Median",
        )
        volatility_axis.set_title("Instantaneous volatility")
        volatility_axis.set_xlabel("Time (years)")
        volatility_axis.set_ylabel("Volatility (%)")
        volatility_axis.legend(loc="best")
        style_axes(volatility_axis, grid_axis="both")

        pooled_min = float(min(spots[-1].min(), gbm[-1].min()))
        pooled_max = float(max(spots[-1].max(), gbm[-1].max()))
        bins = np.linspace(pooled_min, pooled_max, 36)
        terminal_axis.hist(
            gbm[-1],
            bins=bins,
            density=True,
            histtype="step",
            color=MUTED_BLUE,
            linestyle="--",
            linewidth=1.6,
            label="Constant-volatility GBM",
        )
        terminal_axis.hist(
            spots[-1],
            bins=bins,
            density=True,
            histtype="stepfilled",
            facecolor=TEAL,
            edgecolor=TEAL,
            alpha=0.24,
            linewidth=1.1,
            label="Heston",
        )
        terminal_axis.set_title("Terminal spot density")
        terminal_axis.set_xlabel("Terminal spot (price units)")
        terminal_axis.set_ylabel("Density")
        terminal_axis.legend(loc="best", fontsize=8)
        style_axes(terminal_axis, grid_axis="y")

        figure.suptitle("Heston simulation diagnostics", x=0.06, y=0.98, ha="left")
        add_figure_note(
            figure,
            "Source: seeded classroom simulations. Model note: Bands are cross-path "
            "quantiles, not confidence intervals; Heston and GBM are compared on the same "
            f"{maturity_value:g}-year grid (Heston paths={spots.shape[1]:,}; "
            f"GBM paths={gbm.shape[1]:,}).",
        )
        figure.subplots_adjust(left=0.10, right=0.98, top=0.91, bottom=0.16)
        return figure


def build_tail_risk_figure(
    returns: pd.Series | Sequence[float] | np.ndarray,
    historical_var: float,
    expected_shortfall: float,
    gaussian_var: float | None = None,
) -> Figure:
    """Plot a return sample in loss space with VaR and tail-mean Expected Shortfall."""

    observations = _numeric_series(returns, "returns", minimum_observations=10)
    var_value = _finite_scalar(historical_var, "historical_var", non_negative=True)
    es_value = _finite_scalar(expected_shortfall, "expected_shortfall", non_negative=True)
    if es_value + 1e-12 < var_value:
        raise ValueError("expected_shortfall must be at least historical_var in loss space")
    gaussian_value = (
        None
        if gaussian_var is None
        else _finite_scalar(gaussian_var, "gaussian_var", non_negative=True)
    )
    losses = -observations.to_numpy(copy=True) * 100
    var_percent = var_value * 100
    es_percent = es_value * 100
    gaussian_percent = None if gaussian_value is None else gaussian_value * 100
    bins = min(40, max(12, int(np.sqrt(len(losses)) * 2)))

    with matplotlib_style():
        figure, axis = plt.subplots(figsize=(7.5, 4.6), dpi=INLINE_FIGURE_DPI)
        counts, boundaries, patches = axis.hist(
            losses,
            bins=bins,
            density=True,
            color=MUTED_BLUE,
            edgecolor=INK,
            linewidth=0.45,
            alpha=0.30,
            label="Observed losses",
        )
        for left, right, patch in zip(boundaries[:-1], boundaries[1:], patches, strict=True):
            if right > var_percent:
                patch.set_facecolor(CORAL)
                patch.set_alpha(0.48 if left >= var_percent else 0.34)
                patch.set_hatch("//")
        axis.axvline(
            var_percent,
            color=CORAL,
            linestyle="--",
            linewidth=1.7,
            label=f"Historical VaR = {var_percent:.2f}%",
        )
        if gaussian_percent is not None:
            axis.axvline(
                gaussian_percent,
                color=INK,
                linestyle=":",
                linewidth=1.4,
                label=f"Gaussian VaR = {gaussian_percent:.2f}%",
            )
        marker_height = max(float(np.max(counts)) * 0.72, 0.01)
        axis.scatter(
            [es_percent],
            [marker_height],
            marker="D",
            s=46,
            color=TEAL,
            edgecolor=INK,
            linewidth=0.7,
            zorder=5,
            label=f"ES tail mean = {es_percent:.2f}%",
        )
        axis.annotate(
            "Mean/integral of tail loss\n(not a second cutoff)",
            xy=(es_percent, marker_height),
            xytext=(8, 18),
            textcoords="offset points",
            color=TEAL,
            fontsize=8.5,
            arrowprops={"arrowstyle": "->", "color": TEAL, "linewidth": 1.0},
        )
        axis.axvline(0, color=INK, linestyle="-", linewidth=0.8, alpha=0.6)
        axis.set_title("Observed return distribution expressed as portfolio loss")
        axis.set_xlabel("One-period portfolio loss (%; gains are negative)")
        axis.set_ylabel("Density")
        axis.legend(loc="upper left", fontsize=8.3)
        style_axes(axis, grid_axis="y")
        figure.suptitle("Value at Risk and Expected Shortfall", x=0.06, y=0.97, ha="left")
        add_figure_note(
            figure,
            _figure_note(
                returns,
                model_note=(
                    f"{_sample_description(observations)}. VaR is a positive loss quantile; "
                    "ES is the average (quantile integral) of losses in the selected tail."
                ),
            ),
        )
        figure.subplots_adjust(left=0.11, right=0.98, top=0.82, bottom=0.24)
        return figure


def build_ewma_volatility_figure(
    returns: pd.Series | Sequence[float] | np.ndarray,
    ewma_state: pd.Series | Sequence[float] | np.ndarray,
) -> Figure:
    """Plot decimal returns and their per-observation EWMA volatility state."""

    return_series = _numeric_series(returns, "returns", minimum_observations=3)
    state_series = _numeric_series(
        ewma_state,
        "ewma_state",
        minimum_observations=3,
        index=None if isinstance(ewma_state, pd.Series) else return_series.index,
    )
    if not return_series.index.equals(state_series.index):
        raise ValueError("returns and ewma_state must use the same ordered index")
    if (state_series < 0).any():
        raise ValueError("ewma_state must be non-negative")

    with matplotlib_style():
        figure, (return_axis, state_axis) = plt.subplots(
            2,
            1,
            figsize=(7.5, 5.3),
            dpi=INLINE_FIGURE_DPI,
            sharex=True,
            gridspec_kw={"height_ratios": (1, 1.15)},
        )
        return_axis.plot(
            return_series.index,
            return_series * 100,
            color=MUTED_BLUE,
            linewidth=1.0,
            label="Portfolio return",
        )
        return_axis.axhline(0, color=INK, linestyle="--", linewidth=0.9)
        return_axis.set_title("Observed portfolio returns")
        return_axis.set_ylabel("Return (%)")
        style_axes(return_axis, grid_axis="y")

        state_axis.plot(
            state_series.index,
            state_series * 100,
            color=TEAL,
            linestyle="-",
            label="EWMA volatility state",
        )
        state_axis.set_title("Conditional volatility state per observation")
        state_axis.set_ylabel("Volatility (%)")
        _set_horizontal_axis(state_axis, state_series.index)
        state_axis.legend(loc="best")
        style_axes(state_axis, grid_axis="both")
        figure.suptitle("EWMA volatility response", x=0.06, y=0.98, ha="left")
        add_figure_note(
            figure,
            _figure_note(
                returns,
                ewma_state,
                model_note=(
                    f"{_sample_description(return_series)}. Volatility is per observation and "
                    "is not annualized; the plotted state is supplied by the caller."
                ),
            ),
        )
        figure.subplots_adjust(left=0.11, right=0.98, top=0.86, bottom=0.19, hspace=0.44)
        return figure


def _binary_exception_series(
    values: pd.Series | Sequence[bool] | np.ndarray,
    expected_index: pd.Index,
) -> pd.Series:
    if isinstance(values, pd.Series):
        isolated = values.copy(deep=True)
        _validate_index(isolated.index, "exceptions")
        if not isolated.index.equals(expected_index):
            raise ValueError("exceptions must use the finite return/forecast index exactly")
    else:
        array = np.asarray(values)
        if array.ndim != 1 or len(array) != len(expected_index):
            raise ValueError("exceptions must match the finite return/forecast observations")
        isolated = pd.Series(array.copy(), index=expected_index)
    if isolated.isna().any():
        raise ValueError("exceptions cannot contain missing values")
    if pd.api.types.is_bool_dtype(isolated.dtype):
        return isolated.astype(bool)
    numeric = pd.to_numeric(isolated, errors="coerce")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        raise ValueError("exceptions must contain only boolean or binary 0/1 values")
    return numeric.astype(bool)


def build_var_backtest_figure(
    returns: pd.Series | Sequence[float] | np.ndarray,
    var_forecast: pd.Series | Sequence[float] | np.ndarray | float,
    exceptions: pd.Series | Sequence[bool] | np.ndarray,
) -> Figure:
    """Plot positive realized losses, VaR forecasts, and exact strict exceptions."""

    return_series = _numeric_series(returns, "returns", minimum_observations=3)
    if isinstance(var_forecast, Real) and not isinstance(var_forecast, bool):
        forecast_value = _finite_scalar(
            var_forecast,
            "var_forecast",
            non_negative=True,
        )
        forecast_series = pd.Series(
            forecast_value,
            index=return_series.index,
            name="var_forecast",
        )
    else:
        forecast_series = _numeric_series(
            var_forecast,
            "var_forecast",
            minimum_observations=3,
            allow_missing=True,
            index=None if isinstance(var_forecast, pd.Series) else return_series.index,
        )
        if not return_series.index.equals(forecast_series.index):
            raise ValueError("returns and var_forecast must use the same ordered index")
    finite_mask = forecast_series.notna()
    aligned_returns = return_series.loc[finite_mask]
    aligned_forecast = forecast_series.loc[finite_mask]
    if len(aligned_returns) < 3:
        raise ValueError("at least three finite return/forecast pairs are required")
    if (aligned_forecast < 0).any():
        raise ValueError("var_forecast must contain non-negative loss thresholds")
    observed_exceptions = _binary_exception_series(exceptions, aligned_returns.index)
    losses = -aligned_returns
    exact_exceptions = losses > aligned_forecast
    if not observed_exceptions.equals(exact_exceptions.astype(bool)):
        raise ValueError("exceptions must equal the strict condition loss > VaR forecast")

    loss_percent = losses * 100
    forecast_percent = aligned_forecast * 100
    exception_percent = loss_percent.loc[exact_exceptions]
    cumulative = exact_exceptions.astype(int).cumsum()

    with matplotlib_style():
        figure, (risk_axis, count_axis) = plt.subplots(
            2,
            1,
            figsize=(7.5, 5.4),
            dpi=INLINE_FIGURE_DPI,
            sharex=True,
            gridspec_kw={"height_ratios": (1.6, 1)},
        )
        risk_axis.plot(
            loss_percent.index,
            loss_percent,
            color=MUTED_BLUE,
            linewidth=0.9,
            alpha=0.8,
            label="Realized loss",
        )
        risk_axis.plot(
            forecast_percent.index,
            forecast_percent,
            color=INK,
            linestyle="--",
            linewidth=1.4,
            label="VaR forecast",
        )
        risk_axis.scatter(
            exception_percent.index,
            exception_percent,
            color=CORAL,
            marker="x",
            s=34,
            linewidth=1.5,
            label="Exception: loss > VaR",
            zorder=5,
        )
        risk_axis.set_title("Realized positive loss against the ex ante VaR threshold")
        risk_axis.set_ylabel("Loss or VaR (%)")
        risk_axis.legend(loc="best", ncol=3, fontsize=8)
        style_axes(risk_axis, grid_axis="y", show_zero_line=True)

        count_axis.step(
            cumulative.index,
            cumulative,
            where="post",
            color=TEAL,
            linewidth=1.7,
            label="Cumulative exceptions",
        )
        count_axis.scatter(
            cumulative.index[exact_exceptions],
            cumulative.loc[exact_exceptions],
            color=CORAL,
            marker="o",
            s=24,
            zorder=4,
        )
        count_axis.set_title("Exception accumulation")
        count_axis.set_ylabel("Cumulative count")
        _set_horizontal_axis(count_axis, cumulative.index)
        count_axis.set_ylim(bottom=0)
        style_axes(count_axis, grid_axis="both")
        figure.suptitle("VaR backtest timeline", x=0.06, y=0.98, ha="left")
        add_figure_note(
            figure,
            _figure_note(
                returns,
                var_forecast,
                model_note=(
                    f"{_sample_description(aligned_returns)} after removing unavailable "
                    "forecasts. Exceptions use the strict rule L_t > VaR_t; equality is not "
                    "an exception."
                ),
            ),
        )
        figure.subplots_adjust(left=0.11, right=0.98, top=0.86, bottom=0.19, hspace=0.46)
        return figure


def build_var_loss_diagram_figure() -> Figure:
    """Build the export-only right-tail loss-space VaR teaching diagram."""

    alpha = 0.01
    mean_loss = 0.0015
    loss_volatility = 0.012
    var_loss = float(norm.ppf(1 - alpha, loc=mean_loss, scale=loss_volatility))
    losses = np.linspace(mean_loss - 4.0 * loss_volatility, mean_loss + 4.3 * loss_volatility, 800)
    density = norm.pdf(losses, loc=mean_loss, scale=loss_volatility)

    with matplotlib_style():
        figure, axis = plt.subplots(figsize=(12.0, 6.75))
        axis.plot(losses * 100, density, color=INK, linewidth=2.2, label="Loss density")
        tail = losses >= var_loss
        axis.fill_between(
            losses[tail] * 100,
            density[tail],
            color=CORAL,
            alpha=0.50,
            hatch="//",
            label="Worst 1% probability mass",
        )
        axis.axvline(
            var_loss * 100,
            color=CORAL,
            linestyle="--",
            linewidth=2.0,
            label=f"99% VaR = {var_loss * 100:.2f}% loss",
        )
        axis.axvline(0, color=INK, linewidth=1.0, alpha=0.65)
        axis.annotate(
            "More severe losses →",
            xy=(losses.max() * 100, density.max() * 0.13),
            xytext=(var_loss * 100, density.max() * 0.13),
            arrowprops={"arrowstyle": "->", "color": CORAL, "linewidth": 1.5},
            color=CORAL,
            fontsize=12,
            fontweight="semibold",
            ha="left",
        )
        axis.set_title("VaR is a quantile on the right tail of the loss distribution")
        axis.set_xlabel("One-period portfolio loss (%; gains are negative)")
        axis.set_ylabel("Probability density")
        axis.legend(loc="upper right")
        style_axes(axis, grid_axis="y")
        figure.suptitle("Value at Risk in positive-loss space", x=0.07, y=0.95, ha="left")
        add_figure_note(
            figure,
            "Source: author-created deterministic normal-loss teaching model; no external "
            "data. Model note: illustrative one-period Gaussian loss with mean 0.15% and "
            "volatility 1.20%; the 1% shaded mass is the right loss tail.",
        )
        figure.subplots_adjust(left=0.08, right=0.96, top=0.80, bottom=0.20)
        return figure


def build_expected_shortfall_diagram_figure() -> Figure:
    """Build the export-only Expected Shortfall tail-mean teaching diagram."""

    alpha = 0.01
    mean_loss = 0.0015
    loss_volatility = 0.012
    z_score = float(norm.ppf(1 - alpha))
    var_loss = mean_loss + loss_volatility * z_score
    expected_shortfall = mean_loss + loss_volatility * float(norm.pdf(z_score)) / alpha
    losses = np.linspace(mean_loss - 4.0 * loss_volatility, mean_loss + 4.7 * loss_volatility, 900)
    density = norm.pdf(losses, loc=mean_loss, scale=loss_volatility)

    with matplotlib_style():
        figure, axis = plt.subplots(figsize=(12.0, 6.75))
        axis.plot(losses * 100, density, color=INK, linewidth=2.2, label="Loss density")
        tail = losses >= var_loss
        axis.fill_between(
            losses[tail] * 100,
            density[tail],
            color=CORAL,
            alpha=0.50,
            hatch="//",
            label="Worst 1% probability mass",
        )
        axis.axvline(
            var_loss * 100,
            color=CORAL,
            linestyle="--",
            linewidth=2.0,
            label=f"99% VaR = {var_loss * 100:.2f}%",
        )
        marker_y = float(norm.pdf(var_loss, loc=mean_loss, scale=loss_volatility)) * 0.42
        axis.scatter(
            [expected_shortfall * 100],
            [marker_y],
            marker="D",
            s=90,
            color=TEAL,
            edgecolor=INK,
            linewidth=0.9,
            zorder=5,
            label=f"ES tail mean = {expected_shortfall * 100:.2f}%",
        )
        axis.annotate(
            "Average of losses in the shaded tail\n(not another quantile cutoff)",
            xy=(expected_shortfall * 100, marker_y),
            xytext=(expected_shortfall * 100 - 0.25, density.max() * 0.32),
            arrowprops={"arrowstyle": "->", "color": TEAL, "linewidth": 1.5},
            color=TEAL,
            fontsize=11,
            fontweight="semibold",
            ha="right",
        )
        axis.axvline(0, color=INK, linewidth=1.0, alpha=0.65)
        axis.set_title("Expected Shortfall averages the selected loss tail")
        axis.set_xlabel("One-period portfolio loss (%; gains are negative)")
        axis.set_ylabel("Probability density")
        axis.legend(loc="upper right")
        style_axes(axis, grid_axis="y")
        figure.suptitle("Expected Shortfall beyond Value at Risk", x=0.07, y=0.95, ha="left")
        add_figure_note(
            figure,
            "Source: author-created deterministic normal-loss teaching model; no external "
            "data. Model note: ES is the conditional tail mean (equivalently, a quantile "
            "integral here), so it is shown as a point summary rather than a second threshold.",
        )
        figure.subplots_adjust(left=0.08, right=0.96, top=0.80, bottom=0.20)
        return figure


def build_skewness_tail_diagram_figure() -> Figure:
    """Build the export-only three-panel skewness and tail-orientation diagram."""

    x_values = np.linspace(-4.5, 4.5, 900)
    specifications = (
        ("Positive skew", 6.0, TEAL, "Longer right tail", "right"),
        ("Symmetric", 0.0, MUTED_BLUE, "Balanced tails", "both"),
        ("Negative skew", -6.0, CORAL, "Longer left tail", "left"),
    )

    with matplotlib_style():
        figure, axes = plt.subplots(1, 3, figsize=(12.0, 6.75), sharey=True)
        for axis, (title, shape, color, tail_label, tail_side) in zip(
            axes,
            specifications,
            strict=True,
        ):
            density = skewnorm.pdf(x_values, shape)
            median = float(skewnorm.ppf(0.5, shape))
            axis.plot(x_values, density, color=color, linewidth=2.1)
            axis.axvline(
                median,
                color=INK,
                linestyle="--",
                linewidth=1.3,
                label=f"Calculated median = {median:.2f}",
            )
            if tail_side == "right":
                boundary = float(skewnorm.ppf(0.95, shape))
                shade = x_values >= boundary
            elif tail_side == "left":
                boundary = float(skewnorm.ppf(0.05, shape))
                shade = x_values <= boundary
            else:
                lower = float(skewnorm.ppf(0.025, shape))
                upper = float(skewnorm.ppf(0.975, shape))
                left_shade = x_values <= lower
                right_shade = x_values >= upper
                axis.fill_between(
                    x_values[left_shade],
                    density[left_shade],
                    color=color,
                    alpha=0.30,
                    hatch="//",
                    label=tail_label,
                )
                axis.fill_between(
                    x_values[right_shade],
                    density[right_shade],
                    color=color,
                    alpha=0.30,
                    hatch="//",
                )
                shade = None
            if shade is not None:
                axis.fill_between(
                    x_values[shade],
                    density[shade],
                    color=color,
                    alpha=0.30,
                    hatch="//",
                    label=tail_label,
                )
            axis.set_title(title)
            axis.set_xlabel("Standardized return")
            axis.legend(loc="upper center", fontsize=8.5)
            style_axes(axis, grid_axis="y")
        axes[0].set_ylabel("Probability density")
        figure.suptitle("Skewness changes which tail dominates risk", x=0.07, y=0.95, ha="left")
        figure.text(
            0.07,
            0.885,
            "Dashed lines are distribution medians calculated from each plotted model.",
            color=INK,
            fontsize=11,
            ha="left",
        )
        add_figure_note(
            figure,
            "Source: author-created deterministic skew-normal teaching models; no external "
            "data. Model note: panels compare shape and tail orientation only; densities are "
            "standardized and do not represent empirical return estimates.",
        )
        figure.subplots_adjust(left=0.07, right=0.97, top=0.77, bottom=0.20, wspace=0.22)
        return figure


__all__ = [
    "build_convergence_figure",
    "build_ewma_volatility_figure",
    "build_expected_shortfall_diagram_figure",
    "build_heston_diagnostics_figure",
    "build_implied_volatility_figure",
    "build_option_payoff_figure",
    "build_skewness_tail_diagram_figure",
    "build_tail_risk_figure",
    "build_var_backtest_figure",
    "build_var_loss_diagram_figure",
]
