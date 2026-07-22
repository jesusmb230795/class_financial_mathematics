"""Pure Plotly figure builders shared by the interactive course notebooks.

The functions in this module do not display figures or construct widgets. That
separation keeps notebook cells short and makes the visual contracts testable.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.market_data import returns_from_prices
from src.portfolio_optimization import (
    efficient_frontier_variance,
    global_minimum_variance_weights,
    portfolio_return,
    portfolio_volatility,
    tangency_weights,
)
from src.visual_style import (
    INK,
    PLOT_BACKGROUND,
    SEMANTIC_COLORS,
    apply_plotly_style,
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
_PLOTLY_THRESHOLD_DASHES = ("dash", "dot", "dashdot", "longdash", "longdashdot")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
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


def _add_plotly_context_note(
    figure: go.Figure,
    data: pd.Series | pd.DataFrame,
    *,
    data_mode: str | None = None,
) -> None:
    """Add a metadata note below a Plotly figure when evidence is available."""
    note = _context_note(data, data_mode=data_mode)
    if not note:
        return
    figure.add_annotation(
        x=0,
        y=-0.16,
        xref="paper",
        yref="paper",
        text=escape(note),
        showarrow=False,
        align="left",
        xanchor="left",
        font={"color": INK, "size": 10},
    )
    figure.update_layout(margin_b=105)


def drawdown(cumulative_returns: pd.Series) -> pd.Series:
    """Return drawdown relative to the running wealth peak."""
    wealth_index = 1 + cumulative_returns
    running_peak = wealth_index.cummax().clip(lower=1.0)
    return wealth_index / running_peak - 1


def build_macro_dashboard(
    macro: pd.DataFrame,
    primary_series: str,
    comparison_series: str,
    *,
    normalize: bool = False,
    rolling_window: int = 12,
    data_mode: str = "offline",
) -> go.Figure:
    """Build the macro comparison and rolling-correlation dashboard."""
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
    rolling_corr = primary_changes.rolling(rolling_window).corr(comparison_changes)
    primary_label = _display_label(primary_series)
    comparison_label = _display_label(comparison_series)
    correlation_title = (
        f"{rolling_window}-month rolling correlation:<br>"
        f"{primary_label} ({primary_transformation}) vs "
        f"{comparison_label} ({comparison_transformation})"
    )
    if normalize:
        correlation_row = 2
        figure = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            subplot_titles=("Macro comparison (base = 100)", correlation_title),
        )
    else:
        correlation_row = 3
        figure = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.09,
            subplot_titles=(
                f"{primary_label} in native units",
                f"{comparison_label} in native units",
                correlation_title,
            ),
        )
    figure.add_trace(
        go.Scatter(
            x=macro.index,
            y=primary_values,
            mode="lines",
            name=primary_label,
            line={"color": SEMANTIC_COLORS["primary"], "dash": "solid"},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=macro.index,
            y=comparison_values,
            mode="lines",
            name=comparison_label,
            line={"color": SEMANTIC_COLORS["comparison"], "dash": "dash"},
        ),
        row=1 if normalize else 2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=rolling_corr.index,
            y=rolling_corr,
            mode="lines",
            name="Rolling correlation",
            line={"color": SEMANTIC_COLORS["highlight"], "dash": "dot"},
        ),
        row=correlation_row,
        col=1,
    )
    apply_plotly_style(
        figure,
        title=(f"Macro dashboard | {primary_label} vs {comparison_label} | {data_mode} data"),
        height=700 if normalize else 900,
    )
    figure.update_layout(showlegend=True)
    if normalize:
        figure.update_yaxes(title_text="Index (base = 100)", row=1, col=1)
    else:
        figure.update_yaxes(
            title_text=_native_macro_axis_title(primary_series),
            tickformat=".1%" if primary_series in _RATE_CHANGE_SERIES else None,
            row=1,
            col=1,
        )
        figure.update_yaxes(
            title_text=_native_macro_axis_title(comparison_series),
            tickformat=(".1%" if comparison_series in _RATE_CHANGE_SERIES else None),
            row=2,
            col=1,
        )
    figure.update_xaxes(title_text="Date", row=correlation_row, col=1)
    figure.update_yaxes(
        title_text="Correlation coefficient",
        range=[-1.05, 1.05],
        row=correlation_row,
        col=1,
    )
    figure.add_hline(
        y=0,
        row=correlation_row,
        col=1,
        line={"color": SEMANTIC_COLORS["reference"], "width": 1},
    )
    _add_plotly_context_note(figure, macro, data_mode=data_mode)
    return figure


def build_return_explorer(
    prices: pd.DataFrame,
    asset: str,
    *,
    return_method: str = "log",
    rolling_window: int = 63,
    data_mode: str = "offline",
) -> go.Figure:
    """Build a price, performance, drawdown, and risk exploration figure."""
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

    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.075,
        subplot_titles=(
            "Adjusted price",
            "Cumulative adjusted-close change and drawdown",
            f"{rolling_window}-trading-day rolling volatility",
            f"{return_method.title()}-return histogram",
        ),
    )
    figure.add_trace(
        go.Scatter(
            x=asset_prices.index,
            y=asset_prices,
            mode="lines",
            name="Adjusted price",
            line={"color": SEMANTIC_COLORS["primary"], "dash": "solid"},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=cumulative.index,
            y=cumulative,
            mode="lines",
            name="Cumulative change from adjusted price",
            line={"color": SEMANTIC_COLORS["primary"], "dash": "solid"},
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=cumulative.index,
            y=asset_drawdown,
            mode="lines",
            name="Adjusted-price drawdown",
            line={"color": SEMANTIC_COLORS["negative"], "dash": "dash"},
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=rolling_volatility.index,
            y=rolling_volatility,
            mode="lines",
            name=f"{rolling_window}-trading-day rolling volatility",
            line={"color": SEMANTIC_COLORS["highlight"], "dash": "dashdot"},
        ),
        row=3,
        col=1,
    )
    figure.add_trace(
        go.Histogram(
            x=asset_returns,
            nbinsx=50,
            name=f"Observed daily {return_method} returns",
            marker={
                "color": SEMANTIC_COLORS["comparison"],
                "line": {"color": PLOT_BACKGROUND, "width": 0.5},
            },
            opacity=0.85,
        ),
        row=4,
        col=1,
    )
    apply_plotly_style(
        figure,
        title=(
            f"Return explorer: {_display_label(asset)} | daily {return_label}s | {data_mode} data"
        ),
        height=1180,
    )
    figure.update_layout(showlegend=True)
    figure.update_xaxes(title_text="Date", row=1, col=1)
    figure.update_yaxes(title_text="Adjusted price (USD)", row=1, col=1)
    figure.update_xaxes(title_text="Date", row=2, col=1)
    figure.update_yaxes(
        title_text="Adjusted-close change / drawdown",
        tickformat=".1%",
        row=2,
        col=1,
    )
    figure.update_xaxes(title_text="Date", row=3, col=1)
    figure.update_yaxes(
        title_text="Annualized volatility (252 trading days/year)",
        tickformat=".1%",
        row=3,
        col=1,
    )
    figure.update_xaxes(
        title_text=f"Daily {return_label}",
        tickformat=".1%",
        row=4,
        col=1,
    )
    figure.update_yaxes(title_text="Observations", row=4, col=1)
    _add_plotly_context_note(figure, prices, data_mode=data_mode)
    return figure


def build_volatility_dashboard(
    filtered: pd.DataFrame,
    *,
    asset: str,
    persistence: float,
    return_axis_label: str = "Daily return",
    volatility_axis_label: str = "Daily volatility",
) -> go.Figure:
    """Build observed-return and filtered-volatility panels with explicit units."""
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

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            "Observed returns",
            "GARCH-filtered volatility versus EWMA volatility",
        ),
    )
    for column, label, row, color, dash in (
        (
            "return",
            f"{_display_label(asset)} return",
            1,
            SEMANTIC_COLORS["comparison"],
            "solid",
        ),
        (
            "garch_filtered_volatility",
            "GARCH-filtered volatility",
            2,
            SEMANTIC_COLORS["highlight"],
            "dash",
        ),
        (
            "ewma_volatility",
            "EWMA volatility",
            2,
            SEMANTIC_COLORS["primary"],
            "dashdot",
        ),
    ):
        figure.add_trace(
            go.Scatter(
                x=filtered.index,
                y=filtered[column],
                mode="lines",
                name=label,
                line={"color": color, "dash": dash},
            ),
            row=row,
            col=1,
        )
    apply_plotly_style(
        figure,
        title=(
            f"Observed-return volatility dashboard | {_display_label(asset)} | "
            f"alpha + beta = {persistence:.3f}"
        ),
        height=650,
    )
    figure.update_layout(showlegend=True)
    figure.update_yaxes(
        title_text=return_axis_label.strip(),
        tickformat=".1%",
        row=1,
        col=1,
    )
    figure.update_xaxes(title_text="Date", row=2, col=1)
    figure.update_yaxes(
        title_text=volatility_axis_label.strip(),
        tickformat=".1%",
        row=2,
        col=1,
    )
    figure.add_hline(
        y=0,
        row=1,
        col=1,
        line={"color": SEMANTIC_COLORS["reference"], "width": 1},
    )
    _add_plotly_context_note(figure, filtered)
    return figure


def build_efficient_frontier_dashboard(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    *,
    risk_free_rate: float,
    points: int = 80,
) -> tuple[go.Figure, pd.DataFrame]:
    """Build an efficient frontier plus its two reference portfolios."""
    target_returns = np.linspace(
        expected_returns.min() * 0.9,
        expected_returns.max() * 1.1,
        points,
    )
    frontier_volatility = np.sqrt(
        efficient_frontier_variance(
            target_returns,
            expected_returns,
            covariance,
        )
    )
    gmvp = global_minimum_variance_weights(covariance)
    tangency = tangency_weights(
        expected_returns,
        covariance,
        risk_free_rate=risk_free_rate,
    )
    gmvp_volatility = portfolio_volatility(gmvp, covariance)
    tangency_volatility = portfolio_volatility(tangency, covariance)
    tangency_return = portfolio_return(tangency, expected_returns)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frontier_volatility,
            y=target_returns,
            mode="lines",
            name="Efficient frontier",
            line={"color": SEMANTIC_COLORS["primary"], "dash": "solid"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[gmvp_volatility],
            y=[portfolio_return(gmvp, expected_returns)],
            mode="markers",
            marker={
                "color": SEMANTIC_COLORS["highlight"],
                "size": 11,
                "symbol": "circle",
            },
            name="GMVP",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[tangency_volatility],
            y=[tangency_return],
            mode="markers",
            marker={
                "color": SEMANTIC_COLORS["comparison"],
                "size": 11,
                "symbol": "diamond",
            },
            name="Tangency",
        )
    )
    apply_plotly_style(
        figure,
        title=(
            f"Efficient frontier dashboard | GMVP volatility {gmvp_volatility:.2%} | "
            f"Tangency Sharpe "
            f"{(tangency_return - risk_free_rate) / tangency_volatility:.2f}"
        ),
        height=560,
    )
    figure.update_layout(
        xaxis_title="Annualized volatility",
        yaxis_title="Expected annual return",
        showlegend=True,
    )
    figure.update_xaxes(tickformat=".1%")
    figure.update_yaxes(tickformat=".1%")
    context_data = expected_returns if _context_note(expected_returns) else covariance
    _add_plotly_context_note(figure, context_data)
    return figure, pd.DataFrame({"gmvp": gmvp, "tangency": tangency})


def build_bond_sensitivity_dashboard(
    scenario: pd.DataFrame,
    *,
    price: float,
    macaulay_duration: float,
    modified_duration: float,
    convexity: float,
) -> go.Figure:
    """Build exact and approximated bond repricing curves."""
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
    figure = go.Figure()
    for column, label, color, dash in (
        (
            "exact_price",
            "Exact repricing",
            SEMANTIC_COLORS["reference"],
            "solid",
        ),
        (
            "duration_price",
            "Duration approximation",
            SEMANTIC_COLORS["comparison"],
            "dash",
        ),
        (
            "duration_convexity_price",
            "Duration-convexity approximation",
            SEMANTIC_COLORS["primary"],
            "dashdot",
        ),
    ):
        figure.add_trace(
            go.Scatter(
                x=scenario["shock_bps"],
                y=scenario[column],
                mode="lines",
                name=label,
                line={"color": color, "dash": dash},
            )
        )
    apply_plotly_style(
        figure,
        title=(
            f"Price: {price:.2f} | Macaulay duration: {macaulay_duration:.2f} | "
            f"Modified duration: {modified_duration:.2f} | Convexity: {convexity:.2f}"
        ),
        height=480,
    )
    figure.update_layout(
        xaxis_title="Parallel yield shock, basis points",
        yaxis_title="Bond price",
        showlegend=True,
    )
    figure.add_vline(
        x=0,
        line_color=SEMANTIC_COLORS["reference"],
        line_width=1,
    )
    _add_plotly_context_note(figure, scenario)
    return figure


def build_black_scholes_dashboard(
    spot_grid: Sequence[float],
    payoff: Sequence[float],
    option_values: Sequence[float],
    *,
    current_spot: float,
    strike: float,
    title: str,
) -> go.Figure:
    """Build payoff and Black-Scholes value curves."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=spot_grid,
            y=payoff,
            mode="lines",
            name="Payoff at maturity",
            line={"color": SEMANTIC_COLORS["reference"], "dash": "dash"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=spot_grid,
            y=option_values,
            mode="lines",
            name="Black-Scholes value",
            line={"color": SEMANTIC_COLORS["primary"], "dash": "solid"},
        )
    )
    figure.add_vline(
        x=current_spot,
        line_dash="dash",
        line_color=SEMANTIC_COLORS["comparison"],
        annotation_text="Current spot",
    )
    figure.add_vline(
        x=strike,
        line_dash="dot",
        line_color=SEMANTIC_COLORS["highlight"],
        annotation_text="Strike",
    )
    apply_plotly_style(figure, title=title, height=520)
    figure.update_layout(
        xaxis_title="Underlying price",
        yaxis_title="Option value",
        showlegend=True,
    )
    return figure


def build_var_cvar_dashboard(
    returns: pd.Series,
    risk_metrics: pd.Series,
) -> go.Figure:
    """Build a loss-distribution dashboard with VaR/ES thresholds."""
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

    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=observed_returns,
            nbinsx=70,
            name="Observed returns",
            marker={
                "color": SEMANTIC_COLORS["comparison"],
                "line": {"color": PLOT_BACKGROUND, "width": 0.5},
            },
            opacity=0.8,
        )
    )
    threshold_colors = (
        SEMANTIC_COLORS["negative"],
        SEMANTIC_COLORS["highlight"],
        SEMANTIC_COLORS["primary"],
        SEMANTIC_COLORS["comparison"],
        SEMANTIC_COLORS["reference"],
    )
    for index, (metric, value) in enumerate(finite_metrics.items()):
        figure.add_vline(
            x=-value,
            line_color=threshold_colors[index % len(threshold_colors)],
            line_dash=_PLOTLY_THRESHOLD_DASHES[index % len(_PLOTLY_THRESHOLD_DASHES)],
            annotation_text=metric.replace("_", " "),
            annotation_position="top left",
        )
    apply_plotly_style(
        figure,
        title=(
            f"VaR and Expected Shortfall from official returns | lookback = {len(observed_returns)}"
        ),
        height=520,
    )
    figure.update_layout(
        xaxis_title="Daily return",
        yaxis_title="Frequency",
        showlegend=True,
    )
    figure.update_xaxes(tickformat=".2%")
    _add_plotly_context_note(figure, returns)
    return figure
