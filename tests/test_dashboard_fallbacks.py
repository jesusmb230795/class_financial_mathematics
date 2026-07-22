from __future__ import annotations

from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from src.dashboard_fallbacks import (
    build_black_scholes_dashboard_fallback,
    build_bond_sensitivity_dashboard_fallback,
    build_efficient_frontier_dashboard_fallback,
    build_macro_dashboard_fallback,
    build_return_explorer_fallback,
    build_var_cvar_dashboard_fallback,
    build_volatility_dashboard_fallback,
)
from src.dashboards import build_macro_dashboard, build_return_explorer
from src.visual_style import BACKGROUND, SEMANTIC_COLORS


def render_png(figure: Figure) -> bytes:
    """Render a figure through the same PNG format consumed by MyST-NB."""
    output = BytesIO()
    figure.savefig(output, format="png")
    return output.getvalue()


def _figure_text(figure: Figure) -> str:
    """Return figure-level copy with rendering line breaks normalized."""
    return " ".join(" ".join(text.get_text().split()) for text in figure.texts)


def _assert_publication_geometry(figure: Figure) -> None:
    """Protect the narrow, vertically stacked publication fallback contract."""
    assert figure.get_figwidth() <= 7.5
    if len(figure.axes) > 1:
        positions = [axis.get_position() for axis in figure.axes]
        centers = [position.x0 + position.width / 2 for position in positions]
        assert max(centers) - min(centers) < 0.04
        assert all(upper.y0 > lower.y1 for upper, lower in zip(positions, positions[1:]))

    output = BytesIO()
    figure.savefig(output, format="png", bbox_inches="tight")
    tight_width = int.from_bytes(output.getvalue()[16:20], "big")
    nominal_width = figure.get_figwidth() * figure.dpi
    assert tight_width <= nominal_width + figure.dpi * 0.75


def dashboard_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return deterministic inputs shared by the fallback contract tests."""
    index = pd.date_range("2024-01-01", periods=60, freq="D")
    macro = pd.DataFrame(
        {
            "banxico_target_rate": np.linspace(0.08, 0.10, len(index)),
            "mexico_inflation": np.linspace(0.05, 0.04, len(index)),
            "udi": np.linspace(7.5, 8.0, len(index)),
            "usd_mxn": 17 + np.sin(np.linspace(0, 4, len(index))),
        },
        index=index,
    )
    macro.attrs.update(
        {
            "data_mode": "snapshot",
            "sources": "Banxico SIE and DB.NOMICS test snapshot",
            "start": "2023-12-01",
            "end": "2024-12-31",
        }
    )
    prices = pd.DataFrame(
        {"asset": 100 * np.exp(np.linspace(0, 0.12, len(index)))},
        index=index,
    )
    prices.attrs.update(
        {
            "data_mode": "snapshot",
            "sources": "Classroom price test snapshot",
            "start": "2023-12-01",
            "end": "2024-12-31",
        }
    )
    filtered = pd.DataFrame(
        {
            "return": np.sin(np.linspace(0, 5, len(index))) / 100,
            "garch_filtered_volatility": np.linspace(0.014, 0.010, len(index)),
            "ewma_volatility": np.linspace(0.013, 0.011, len(index)),
        },
        index=index,
    )
    return macro, prices, filtered


def test_macro_fallback_has_stable_normalized_contract_and_png() -> None:
    macro, _, _ = dashboard_inputs()
    original = macro.copy(deep=True)

    figure = build_macro_dashboard_fallback(
        macro,
        "usd_mxn",
        "udi",
        normalize=True,
        rolling_window=6,
        data_mode="offline",
    )
    interactive = build_macro_dashboard(
        macro,
        "usd_mxn",
        "udi",
        normalize=True,
        rolling_window=6,
        data_mode="offline",
    )

    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 2
        assert figure.axes[0].get_ylabel() == "Index (base = 100)"
        assert figure.axes[1].get_ylabel() == "Correlation coefficient"
        assert figure.axes[0].lines[0].get_ydata()[0] == 100
        assert figure.axes[0].lines[1].get_ydata()[0] == 100
        assert "offline data" in figure._suptitle.get_text()
        assert figure.axes[0].lines[0].get_color() == SEMANTIC_COLORS["primary"]
        assert figure.axes[0].lines[0].get_linestyle() == "-"
        assert figure.axes[0].lines[1].get_color() == SEMANTIC_COLORS["comparison"]
        assert figure.axes[0].lines[1].get_linestyle() == "--"
        assert figure.axes[1].lines[0].get_color() == SEMANTIC_COLORS["highlight"]
        assert figure.axes[1].lines[0].get_linestyle() == ":"
        assert figure.axes[1].get_title(loc="left").count("percent change") == 2
        np.testing.assert_allclose(
            figure.axes[1].lines[0].get_ydata(),
            np.asarray(interactive.data[2].y, dtype=float),
            equal_nan=True,
        )
        figure_text = _figure_text(figure)
        assert "Sample: 2024-01-01 to 2024-02-29" in figure_text
        assert "Source: Banxico SIE and DB.NOMICS test snapshot" in figure_text
        assert figure.get_facecolor() == matplotlib.colors.to_rgba(BACKGROUND)
        _assert_publication_geometry(figure)
        assert render_png(figure).startswith(b"\x89PNG\r\n\x1a\n")
        pd.testing.assert_frame_equal(macro, original)
    finally:
        plt.close(figure)


def test_return_fallback_preserves_performance_and_risk_panels() -> None:
    _, prices, _ = dashboard_inputs()
    original = prices.copy(deep=True)

    figure = build_return_explorer_fallback(
        prices,
        "asset",
        rolling_window=5,
        data_mode="offline",
    )
    interactive = build_return_explorer(
        prices,
        "asset",
        rolling_window=5,
        data_mode="offline",
    )

    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 4
        assert [axis.get_title(loc="left") for axis in figure.axes] == [
            "Adjusted price",
            "Cumulative adjusted-close change and drawdown",
            "5-trading-day rolling volatility",
            "Log-return histogram",
        ]
        assert figure.axes[0].get_ylabel() == "Adjusted price (USD)"
        assert figure.axes[1].get_ylabel() == "Adjusted-close change / drawdown"
        assert figure.axes[2].get_ylabel() == "Annualized volatility (252 trading days/year)"
        assert figure.axes[3].get_xlabel() == "Daily log return"
        assert "daily log returns" in figure._suptitle.get_text()
        assert len(figure.axes[1].lines) == 3
        assert figure.axes[0].lines[0].get_color() == SEMANTIC_COLORS["primary"]
        assert figure.axes[1].lines[0].get_color() == SEMANTIC_COLORS["primary"]
        assert figure.axes[1].lines[1].get_color() == SEMANTIC_COLORS["negative"]
        assert figure.axes[1].lines[1].get_linestyle() == "--"
        assert figure.axes[2].lines[0].get_color() == SEMANTIC_COLORS["highlight"]
        assert figure.axes[2].lines[0].get_linestyle() == "-."
        np.testing.assert_allclose(
            figure.axes[1].lines[0].get_ydata(),
            np.asarray(interactive.data[1].y, dtype=float),
        )
        assert figure.axes[1].lines[0].get_ydata()[0] == 0
        np.testing.assert_allclose(
            figure.axes[1].lines[1].get_ydata(),
            np.asarray(interactive.data[2].y, dtype=float),
        )
        np.testing.assert_allclose(
            figure.axes[2].lines[0].get_ydata(),
            np.asarray(interactive.data[3].y, dtype=float),
            equal_nan=True,
        )
        assert any(
            "Source: Classroom price test snapshot" in text.get_text() for text in figure.texts
        )
        _assert_publication_geometry(figure)
        assert render_png(figure).startswith(b"\x89PNG\r\n\x1a\n")
        pd.testing.assert_frame_equal(prices, original)
    finally:
        plt.close(figure)


def test_volatility_fallback_has_daily_units_and_deterministic_png() -> None:
    _, _, filtered = dashboard_inputs()
    original = filtered.copy(deep=True)

    first = build_volatility_dashboard_fallback(
        filtered,
        asset="asset",
        persistence=0.98,
    )
    second = build_volatility_dashboard_fallback(
        filtered,
        asset="asset",
        persistence=0.98,
    )

    try:
        assert isinstance(first, Figure)
        assert len(first.axes) == 2
        assert first.axes[0].get_ylabel() == "Daily return"
        assert first.axes[1].get_ylabel() == "Daily volatility"
        assert len(first.axes[1].lines) == 2
        assert "alpha + beta = 0.980" in first._suptitle.get_text()
        assert first.axes[0].lines[0].get_color() == SEMANTIC_COLORS["comparison"]
        assert first.axes[1].lines[0].get_color() == SEMANTIC_COLORS["highlight"]
        assert first.axes[1].lines[0].get_linestyle() == "--"
        assert first.axes[1].lines[1].get_color() == SEMANTIC_COLORS["primary"]
        assert first.axes[1].lines[1].get_linestyle() == "-."
        _assert_publication_geometry(first)
        assert render_png(first) == render_png(second)
        pd.testing.assert_frame_equal(filtered, original)
    finally:
        plt.close(first)
        plt.close(second)


def test_bond_fallback_preserves_repricing_contract() -> None:
    scenario = pd.DataFrame(
        {
            "shock_bps": [-100, 0, 100],
            "exact_price": [104.2, 100.0, 96.1],
            "duration_price": [104.0, 100.0, 96.0],
            "duration_convexity_price": [104.2, 100.0, 96.2],
        }
    )
    original = scenario.copy(deep=True)

    figure = build_bond_sensitivity_dashboard_fallback(
        scenario,
        price=100.0,
        macaulay_duration=4.2,
        modified_duration=4.0,
        convexity=20.0,
        face_value=100.0,
        coupon_rate=0.08,
        maturity=5.0,
        ytm=0.07,
        frequency=2,
    )

    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 1
        assert figure.axes[0].get_xlabel() == "Parallel yield shock (basis points)"
        assert figure.axes[0].get_ylabel() == "Bond price (currency units)"
        assert len(figure.axes[0].lines) == 4
        assert [line.get_color() for line in figure.axes[0].lines[:3]] == [
            SEMANTIC_COLORS["reference"],
            SEMANTIC_COLORS["comparison"],
            SEMANTIC_COLORS["primary"],
        ]
        assert [line.get_linestyle() for line in figure.axes[0].lines[:3]] == [
            "-",
            "--",
            "-.",
        ]
        assert [line.get_marker() for line in figure.axes[0].lines[:3]] == [
            "o",
            "s",
            "D",
        ]
        note = _figure_text(figure)
        assert "annual coupon 8.00%" in note
        assert "maturity 5 years" in note
        assert "base annual YTM 7.00%" in note
        assert "2 payments per year" in note
        _assert_publication_geometry(figure)
        assert render_png(figure).startswith(b"\x89PNG\r\n\x1a\n")
        pd.testing.assert_frame_equal(scenario, original)
    finally:
        plt.close(figure)


def test_black_scholes_fallback_preserves_payoff_and_model_curves() -> None:
    spot_grid = np.linspace(60, 140, 9)
    payoff = np.maximum(spot_grid - 100, 0)
    option_values = payoff + 5

    figure = build_black_scholes_dashboard_fallback(
        spot_grid,
        payoff,
        option_values,
        current_spot=100,
        strike=105,
        title="Call pricing diagnostic",
    )

    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 1
        assert figure.axes[0].get_xlabel() == "Underlying price"
        assert figure.axes[0].get_ylabel() == "Option value"
        assert len(figure.axes[0].lines) == 4
        assert figure._suptitle.get_text() == "Call pricing diagnostic"
        assert figure.axes[0].lines[0].get_color() == SEMANTIC_COLORS["reference"]
        assert figure.axes[0].lines[0].get_linestyle() == "--"
        assert figure.axes[0].lines[1].get_color() == SEMANTIC_COLORS["primary"]
        assert figure.axes[0].lines[1].get_linestyle() == "-"
        assert figure.axes[0].lines[2].get_color() == SEMANTIC_COLORS["comparison"]
        assert figure.axes[0].lines[3].get_color() == SEMANTIC_COLORS["highlight"]
        assert render_png(figure).startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        plt.close(figure)


def test_macro_fallback_raw_mode_uses_separate_native_unit_panels() -> None:
    macro, _, _ = dashboard_inputs()

    figure = build_macro_dashboard_fallback(
        macro,
        "banxico_target_rate",
        "usd_mxn",
        rolling_window=6,
    )

    try:
        assert len(figure.axes) == 3
        assert [len(axis.lines) for axis in figure.axes] == [1, 1, 2]
        assert figure.axes[0].get_ylabel() == "Banxico target rate (%)"
        assert isinstance(
            figure.axes[0].yaxis.get_major_formatter(),
            PercentFormatter,
        )
        assert figure.axes[0].yaxis.get_major_formatter().xmax == 1
        assert figure.axes[1].get_ylabel() == "USD/MXN (MXN per USD)"
        assert figure.axes[2].get_ylabel() == "Correlation coefficient"
        assert figure.get_layout_engine().get()["rect"][1] == 0.065
        _assert_publication_geometry(figure)
        assert render_png(figure).startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        plt.close(figure)


def test_macro_fallback_rejects_comparing_a_series_with_itself() -> None:
    macro, _, _ = dashboard_inputs()

    with pytest.raises(ValueError, match="must be different"):
        build_macro_dashboard_fallback(
            macro,
            "banxico_target_rate",
            "banxico_target_rate",
        )


def test_macro_fallback_rejects_rebasing_rate_series() -> None:
    macro, _, _ = dashboard_inputs()

    with pytest.raises(ValueError, match="positive level series"):
        build_macro_dashboard_fallback(
            macro,
            "banxico_target_rate",
            "udi",
            normalize=True,
        )


def test_var_cvar_fallback_skips_unavailable_metric_and_renders_png() -> None:
    returns = pd.Series(
        np.linspace(-0.03, 0.025, 120),
        name="portfolio_return",
    )
    risk_metrics = pd.Series(
        {
            "historical_var": 0.025,
            "cornish_fisher_var": np.nan,
            "expected_shortfall": 0.029,
        }
    )

    figure = build_var_cvar_dashboard_fallback(returns, risk_metrics)

    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 1
        assert figure.axes[0].get_xlabel() == "Daily return"
        assert figure.axes[0].get_ylabel() == "Frequency"
        assert len(figure.axes[0].lines) == 2
        assert figure.axes[0].lines[0].get_color() == SEMANTIC_COLORS["negative"]
        assert figure.axes[0].lines[0].get_linestyle() == "--"
        assert figure.axes[0].lines[1].get_color() == SEMANTIC_COLORS["highlight"]
        assert figure.axes[0].lines[1].get_linestyle() == ":"
        assert render_png(figure).startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        plt.close(figure)


def test_efficient_frontier_fallback_returns_figure_and_reference_weights() -> None:
    assets = ["asset_a", "asset_b", "asset_c"]
    expected_returns = pd.Series([0.08, 0.11, 0.14], index=assets)
    covariance = pd.DataFrame(
        [
            [0.0225, 0.0060, 0.0045],
            [0.0060, 0.0324, 0.0081],
            [0.0045, 0.0081, 0.0484],
        ],
        index=assets,
        columns=assets,
    )

    figure, weights = build_efficient_frontier_dashboard_fallback(
        expected_returns,
        covariance,
        risk_free_rate=0.04,
        points=30,
    )

    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 1
        assert figure.axes[0].get_xlabel() == "Annualized volatility"
        assert figure.axes[0].get_ylabel() == "Expected annual return"
        assert list(weights.columns) == ["gmvp", "tangency"]
        np.testing.assert_allclose(weights.sum(), [1.0, 1.0])
        assert figure.axes[0].lines[0].get_color() == SEMANTIC_COLORS["primary"]
        assert figure.axes[0].collections[0].get_facecolors()[0].tolist() == list(
            matplotlib.colors.to_rgba(SEMANTIC_COLORS["highlight"])
        )
        assert figure.axes[0].collections[1].get_facecolors()[0].tolist() == list(
            matplotlib.colors.to_rgba(SEMANTIC_COLORS["comparison"])
        )
        assert render_png(figure).startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        plt.close(figure)
