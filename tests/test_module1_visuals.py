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

from src.module1_visuals import (
    build_macro_eda_overview,
    build_mexican_macro_context,
    build_mexican_market_dependence,
    build_mexican_market_paths,
    build_mexican_market_risk,
    build_stock_eda_overview,
    build_stock_risk_map,
    build_wfe_market_scale_overview,
)


def _render_png(figure: Figure) -> bytes:
    output = BytesIO()
    figure.savefig(output, format="png")
    return output.getvalue()


def _figure_text(figure: Figure) -> str:
    """Return figure-level copy with rendering line breaks normalized."""
    return " ".join(" ".join(text.get_text().split()) for text in figure.texts)


def _assert_publication_geometry(
    figure: Figure,
    main_axes: list | np.ndarray | tuple,
) -> None:
    """Protect the single-column, mobile-readable publication contract."""
    assert figure.get_figwidth() <= 7.5
    axes = list(main_axes)
    if len(axes) > 1:
        positions = [axis.get_position() for axis in axes]
        common_width = min(position.x1 for position in positions) - max(
            position.x0 for position in positions
        )
        assert common_width > min(position.width for position in positions) * 0.5
        assert all(upper.y0 > lower.y1 for upper, lower in zip(positions, positions[1:]))

    output = BytesIO()
    figure.savefig(output, format="png", bbox_inches="tight")
    tight_width = int.from_bytes(output.getvalue()[16:20], "big")
    nominal_width = figure.get_figwidth() * figure.dpi
    assert tight_width <= nominal_width + figure.dpi * 0.75


def _axis_titles(figure: Figure) -> set[str]:
    return {
        title
        for axis in figure.axes
        for title in (axis.get_title(), axis.get_title(loc="left"))
        if title
    }


def _stock_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2023-01-02", periods=180, freq="B")
    angles = np.linspace(0, 8, len(index))
    innovations = pd.DataFrame(
        {
            "AAPL": 0.0007 + np.sin(angles) * 0.008,
            "MSFT": 0.0005 + np.cos(angles * 0.8) * 0.007,
            "NVDA": 0.0010 + np.sin(angles * 1.4) * 0.014,
            "AMZN": 0.0004 + np.cos(angles * 1.2) * 0.011,
            "GOOGL": 0.0006 + np.sin(angles * 0.6) * 0.009,
        },
        index=index,
    )
    prices = np.exp(innovations.cumsum()).multiply(
        pd.Series([120, 220, 80, 100, 90], index=innovations.columns)
    )
    return prices, innovations


def _macro_input() -> pd.DataFrame:
    index = pd.date_range("2021-01-31", periods=54, freq="ME")
    phase = np.linspace(0, 5, len(index))
    return pd.DataFrame(
        {
            "banxico_target_rate": 0.06 + 0.025 * np.sin(phase),
            "cetes_28d": 0.058 + 0.022 * np.sin(phase + 0.15),
            "tiie_28d": 0.063 + 0.023 * np.sin(phase + 0.25),
            "mexico_inflation": 0.045 + 0.012 * np.cos(phase * 0.8),
            "us_10y": 0.025 + 0.009 * np.sin(phase * 0.7),
            "usd_mxn": 19 + 0.8 * np.sin(phase * 1.2),
            "udi": 7.0 * np.exp(np.linspace(0, 0.18, len(index))),
            "mexico_cpi": 105 * np.exp(np.linspace(0, 0.20, len(index))),
        },
        index=index,
    )


def _wfe_input() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "metric": "Market capitalisation",
                "display_value": 127.4,
                "display_unit": "USD trillions",
                "change_percent": 5.9,
            },
            {
                "metric": "Value of share trading",
                "display_value": 14.2,
                "display_unit": "USD trillions",
                "change_percent": 3.2,
            },
            {
                "metric": "Number of trades",
                "display_value": 11.8,
                "display_unit": "billion trades",
                "change_percent": 1.5,
            },
            {
                "metric": "Options contracts traded",
                "display_value": 8.4,
                "display_unit": "billion contracts",
                "change_percent": 2.0,
            },
            {
                "metric": "Futures contracts traded",
                "display_value": 3.7,
                "display_unit": "billion contracts",
                "change_percent": -1.0,
            },
            {
                "metric": "Listed companies",
                "display_value": 54_300.0,
                "display_unit": "companies",
                "change_percent": 0.8,
            },
            {
                "metric": "Investment flows",
                "display_value": 1.6,
                "display_unit": "USD billions",
                "change_percent": 4.0,
            },
        ]
    )
    frame["change_basis"] = "not specified on source dashboard"
    return frame


def _mexican_case_inputs() -> dict[str, pd.DataFrame]:
    index = pd.date_range("2023-01-02", periods=180, freq="B")
    phase = np.linspace(0, 8, len(index))
    returns = pd.DataFrame(
        {
            "usd_mxn": 0.0001 + 0.006 * np.sin(phase),
            "udi": 0.0003 + 0.0015 * np.cos(phase * 0.65),
            "cetes_28d_carry": 0.00025 + 0.00003 * np.sin(phase * 0.4),
            "tiie_28d_carry": 0.00028 + 0.00004 * np.cos(phase * 0.45),
            "policy_rate_carry": 0.00023 + 0.00005 * np.sin(phase * 0.3),
        },
        index=index,
    )
    levels = np.exp(returns.cumsum()).multiply(
        pd.Series(
            [17.0, 7.5, 100.0, 100.0, 100.0],
            index=returns.columns,
        )
    )
    cumulative_returns = np.exp(returns.cumsum()).subtract(1)
    rolling_volatility = returns.rolling(21).std()
    scatter_data = returns[["usd_mxn", "udi"]].dropna()
    correlations = returns.corr()
    wealth = cumulative_returns.add(1)
    drawdowns = wealth.divide(wealth.cummax()).subtract(1)
    drawdowns.loc[index[70:90], "cetes_28d_carry"] = np.linspace(-0.005, -0.025, 20)

    macro_index = pd.date_range("2021-01-31", periods=48, freq="ME")
    macro_phase = np.linspace(0, 5, len(macro_index))
    macro_context = pd.DataFrame(
        {
            "banxico_target_rate_pct": 6.0 + 2.5 * np.sin(macro_phase),
            "cetes_28d_pct": 5.8 + 2.2 * np.sin(macro_phase + 0.15),
            "tiie_28d_pct": 6.3 + 2.3 * np.sin(macro_phase + 0.25),
            "mexico_inflation_pct": 4.5 + 1.2 * np.cos(macro_phase * 0.8),
        },
        index=macro_index,
    )
    return {
        "levels": levels,
        "cumulative_returns": cumulative_returns,
        "returns": returns,
        "rolling_volatility": rolling_volatility,
        "scatter_data": scatter_data,
        "correlations": correlations,
        "drawdowns": drawdowns,
        "macro_context": macro_context,
    }


def test_stock_eda_overview_has_six_distinct_questions_and_preserves_inputs() -> None:
    prices, returns = _stock_inputs()
    original_prices = prices.copy(deep=True)
    original_returns = returns.copy(deep=True)

    figure = build_stock_eda_overview(prices, returns, rolling_window=21)

    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 7  # six panels plus the correlation colorbar
        titles = _axis_titles(figure)
        assert "Adjusted prices on a common base" in titles
        assert "21-trading-day correlation is not constant" in titles
        assert "21-trading-day rolling risk" in titles
        assert "Daily log-return distributions" in titles
        assert "Return dependence on a fixed [-1, 1] scale" in titles
        assert "Sample: 2023-01-02 to 2023-09-08" in _figure_text(figure)
        _assert_publication_geometry(figure, figure.axes[:6])
        assert _render_png(figure).startswith(b"\x89PNG\r\n\x1a\n")
        pd.testing.assert_frame_equal(prices, original_prices)
        pd.testing.assert_frame_equal(returns, original_returns)
    finally:
        plt.close(figure)


def test_stock_eda_drawdown_comes_from_prices_and_preserves_price_gaps() -> None:
    prices, returns = _stock_inputs()
    gap_position = 40
    prices.iloc[gap_position, prices.columns.get_loc("AAPL")] = np.nan

    figure = build_stock_eda_overview(prices, returns, rolling_window=21)

    try:
        drawdown_axis = next(
            axis
            for axis in figure.axes
            if axis.get_title(loc="left") == "Loss from each running peak"
        )
        plotted_drawdown = drawdown_axis.lines[0].get_ydata()
        expected_drawdown = prices["AAPL"].divide(prices["AAPL"].cummax()).subtract(1)
        np.testing.assert_allclose(plotted_drawdown, expected_drawdown, equal_nan=True)
        assert np.isnan(plotted_drawdown[gap_position])
    finally:
        plt.close(figure)


def test_stock_risk_map_uses_labels_shapes_and_tail_size() -> None:
    _, returns = _stock_inputs()

    figure = build_stock_risk_map(returns)

    try:
        axis = figure.axes[0]
        assert isinstance(figure, Figure)
        assert axis.get_xlabel() == "Annualized volatility (252 trading days)"
        assert axis.get_ylabel() == "Annualized mean log return"
        assert {annotation.get_text() for annotation in axis.texts} == set(returns.columns)
        assert len(axis.collections) == len(returns.columns)
        assert "larger markers" in figure._suptitle.get_text()
        _assert_publication_geometry(figure, figure.axes)
        assert _render_png(figure).startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        plt.close(figure)


def test_macro_overview_preserves_units_returns_auditable_changes_and_inputs() -> None:
    macro = _macro_input()
    original = macro.copy(deep=True)

    figure, changes = build_macro_eda_overview(macro, rolling_window=12)

    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 5  # four panels plus the regime-map colorbar
        titles = _axis_titles(figure)
        assert "Rates remain in economically meaningful units" in titles
        assert "Regime map of one-period changes" in titles
        assert "UDI and USD/MXN co-movement changes over time" in titles
        rates_axis = next(
            axis
            for axis in figure.axes
            if axis.get_title(loc="left") == "Rates remain in economically meaningful units"
        )
        assert isinstance(rates_axis.yaxis.get_major_formatter(), PercentFormatter)
        assert rates_axis.yaxis.get_major_formatter().xmax == 100
        regime_axis = next(
            axis
            for axis in figure.axes
            if axis.get_title(loc="left") == "Regime map of one-period changes"
        )
        regime_labels = [label.get_text() for label in regime_axis.get_xticklabels()]
        assert regime_labels[0] == "Jan\n2021"
        assert regime_labels[-1] == "Jun\n2025"
        assert regime_axis.get_xlabel() == "Reference month"
        assert "Sample: 2021-01-31 to 2025-06-30" in _figure_text(figure)
        assert changes.columns.tolist() == [
            "Banxico target rate change (pp)",
            "Mexico inflation change (pp)",
            "US 10Y change (pp)",
            "USD/MXN percentage change",
            "UDI percentage change",
        ]
        _assert_publication_geometry(figure, figure.axes[:4])
        assert _render_png(figure).startswith(b"\x89PNG\r\n\x1a\n")
        pd.testing.assert_frame_equal(macro, original)
    finally:
        plt.close(figure)


def test_wfe_overview_is_a_vertical_comparable_unit_contract() -> None:
    market_scale = _wfe_input()
    original = market_scale.copy(deep=True)

    figure = build_wfe_market_scale_overview(market_scale)

    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 4
        assert [len(axis.patches) for axis in figure.axes] == [0, 0, 0, 2]
        assert [axis.get_xlabel() for axis in figure.axes] == [
            "",
            "",
            "",
            "Billion contracts",
        ]
        card_text = [[text.get_text() for text in axis.texts] for axis in figure.axes[:3]]
        assert card_text == [
            [
                "Market capitalisation",
                "127.40 USD trillions",
                "+5.90% reported",
                "Listed companies",
                "54,300.00 companies",
                "+0.80% reported",
            ],
            [
                "Share trading value",
                "14.20 USD trillions",
                "+3.20% reported",
                "Number of trades",
                "11.80 billion trades",
                "+1.50% reported",
            ],
            ["Investment flows", "1.60 USD billions", "+4.00% reported"],
        ]
        assert [text.get_text() for text in figure.axes[3].texts] == [
            "  8.40 | +2.00% reported",
            "  3.70 | -1.00% reported",
        ]
        figure_text = _figure_text(figure)
        assert "change basis: not specified on source dashboard" in figure_text
        assert _axis_titles(figure) == {
            "Equity market size",
            "Equity market activity",
            "Primary-market flow",
            "Listed derivatives activity",
        }
        assert any("\n" in text.get_text() for text in figure.texts if "Source:" in text.get_text())
        _assert_publication_geometry(figure, figure.axes)
        assert _render_png(figure).startswith(b"\x89PNG\r\n\x1a\n")
        pd.testing.assert_frame_equal(market_scale, original)
    finally:
        plt.close(figure)


def test_mexican_paths_keep_observed_and_synthetic_series_separate() -> None:
    inputs = _mexican_case_inputs()
    levels = inputs["levels"]
    cumulative_returns = inputs["cumulative_returns"]
    original_levels = levels.copy(deep=True)
    original_returns = cumulative_returns.copy(deep=True)

    figure = build_mexican_market_paths(levels, cumulative_returns)

    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 4
        assert _axis_titles(figure) == {
            "Observed USD/MXN FIX level",
            "Observed UDI level",
            "Synthetic cumulative carry changes",
            "Observed cumulative level changes",
        }
        legend_sizes = [len(axis.get_legend().get_texts()) for axis in figure.axes]
        assert legend_sizes == [1, 1, 3, 2]
        assert [axis.get_ylabel() for axis in figure.axes] == [
            "MXN per USD",
            "MXN per UDI",
            "Simple-change equivalent",
            "Simple-change equivalent",
        ]
        _assert_publication_geometry(figure, figure.axes)
        assert _render_png(figure).startswith(b"\x89PNG\r\n\x1a\n")
        pd.testing.assert_frame_equal(levels, original_levels)
        pd.testing.assert_frame_equal(cumulative_returns, original_returns)
    finally:
        plt.close(figure)


def test_mexican_risk_is_vertical_and_preserves_all_inputs() -> None:
    inputs = _mexican_case_inputs()
    returns = inputs["returns"]
    rolling_volatility = inputs["rolling_volatility"]
    scatter_data = inputs["scatter_data"]
    original_returns = returns.copy(deep=True)
    original_volatility = rolling_volatility.copy(deep=True)
    original_scatter = scatter_data.copy(deep=True)

    figure = build_mexican_market_risk(
        returns,
        rolling_volatility,
        scatter_data,
        rolling_window=21,
    )

    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 4
        assert _axis_titles(figure) == {
            "USD/MXN observed-interval log-change distribution",
            "USD/MXN versus UDI observed-interval log changes",
            "Observed 21-valid-observation rolling log-change dispersion",
            "Synthetic 21-valid-observation rolling log-change dispersion",
        }
        assert figure.axes[0].get_xlabel() == "Log change between joint observations"
        assert figure.axes[1].get_xlabel() == "USD/MXN log change between joint observations"
        figure_text = _figure_text(figure)
        assert "heterogeneous observation intervals are not annualized" in figure_text
        assert figure.axes[2].get_ylabel() == "Standard deviation per stored change"
        assert figure.axes[3].get_ylabel() == "Standard deviation per stored change"
        _assert_publication_geometry(figure, figure.axes)
        assert _render_png(figure).startswith(b"\x89PNG\r\n\x1a\n")
        pd.testing.assert_frame_equal(returns, original_returns)
        pd.testing.assert_frame_equal(rolling_volatility, original_volatility)
        pd.testing.assert_frame_equal(scatter_data, original_scatter)
    finally:
        plt.close(figure)


def test_mexican_dependence_uses_fixed_scale_and_derived_carry_note() -> None:
    inputs = _mexican_case_inputs()
    correlations = inputs["correlations"]
    drawdowns = inputs["drawdowns"]
    original_correlations = correlations.copy(deep=True)
    original_drawdowns = drawdowns.copy(deep=True)

    figure = build_mexican_market_dependence(correlations, drawdowns)

    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 3  # two panels plus the heatmap colorbar
        assert _axis_titles(figure) == {
            "Observed-interval log-change correlation | fixed scale [-1, 1]",
            "Observed-level drawdown (not investment loss)",
        }
        assert figure.axes[1].get_ylabel() == "Level drawdown"
        assert figure.axes[0].collections[0].get_clim() == (-1.0, 1.0)
        assert len(figure.axes[1].get_legend().get_texts()) == 2
        figure_text = _figure_text(figure)
        assert "CETES 28-day carry: -2.50%" in figure_text
        assert "TIIE 28-day carry: 0.00%" in figure_text
        assert "Policy-rate carry: 0.00%" in figure_text
        assert "Level drawdown is a path statistic, not an investment loss" in figure_text
        _assert_publication_geometry(figure, figure.axes[:2])
        assert _render_png(figure).startswith(b"\x89PNG\r\n\x1a\n")
        pd.testing.assert_frame_equal(correlations, original_correlations)
        pd.testing.assert_frame_equal(drawdowns, original_drawdowns)
    finally:
        plt.close(figure)


def test_mexican_macro_context_is_a_single_monthly_block() -> None:
    macro_context = _mexican_case_inputs()["macro_context"]
    original = macro_context.copy(deep=True)

    figure = build_mexican_macro_context(macro_context)

    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 1
        assert "Monthly Mexican rate and inflation context" in _axis_titles(figure)
        assert len(figure.axes[0].get_legend().get_texts()) == 4
        formatter = figure.axes[0].yaxis.get_major_formatter()
        assert isinstance(formatter, PercentFormatter)
        assert formatter.xmax == 100
        _assert_publication_geometry(figure, figure.axes)
        assert _render_png(figure).startswith(b"\x89PNG\r\n\x1a\n")
        pd.testing.assert_frame_equal(macro_context, original)
    finally:
        plt.close(figure)


def test_module1_visuals_reject_invalid_inputs() -> None:
    prices, returns = _stock_inputs()
    with pytest.raises(ValueError, match="rolling_window"):
        build_stock_eda_overview(prices, returns, rolling_window=1)
    with pytest.raises(ValueError, match="at least two columns"):
        build_stock_eda_overview(prices[["AAPL"]], returns[["AAPL"]])
    with pytest.raises(ValueError, match="missing required columns"):
        build_macro_eda_overview(_macro_input().drop(columns="udi"))
    with pytest.raises(ValueError, match="missing required metrics"):
        build_wfe_market_scale_overview(_wfe_input().query("metric != 'Futures contracts traded'"))
    inputs = _mexican_case_inputs()
    with pytest.raises(ValueError, match="missing required columns"):
        build_mexican_market_dependence(
            inputs["correlations"],
            inputs["drawdowns"].drop(columns="policy_rate_carry"),
        )
