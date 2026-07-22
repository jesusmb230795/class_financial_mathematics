from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from plotly.graph_objects import Figure as PlotlyFigure

from src.dashboards import (
    build_black_scholes_dashboard,
    build_bond_sensitivity_dashboard,
    build_efficient_frontier_dashboard,
    build_macro_dashboard,
    build_return_explorer,
    build_var_cvar_dashboard,
    build_volatility_dashboard,
    drawdown,
)
from src.visual_style import (
    BACKGROUND,
    SEMANTIC_COLORS,
    SERIES_COLORS,
)


def _assert_vertical_domains(figure: PlotlyFigure, row_count: int) -> None:
    """Assert that Plotly subplots occupy distinct top-to-bottom rows."""
    domains = [
        getattr(figure.layout, "yaxis" if row == 1 else f"yaxis{row}").domain
        for row in range(1, row_count + 1)
    ]
    assert all(upper[0] > lower[1] for upper, lower in zip(domains, domains[1:]))


def test_drawdown_starts_at_zero_and_tracks_peak_loss() -> None:
    cumulative = pd.Series([0.0, 0.10, 0.05, 0.20])

    result = drawdown(cumulative)

    assert result.iloc[0] == 0
    assert result.iloc[2] == pytest.approx(1.05 / 1.10 - 1)
    assert result.iloc[-1] == 0


def test_drawdown_keeps_the_initial_wealth_base_for_a_first_period_loss() -> None:
    cumulative = pd.Series([-0.05, 0.02, -0.01])

    result = drawdown(cumulative)

    expected = pd.Series([-0.05, 0.0, 0.99 / 1.02 - 1.0])
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize("return_method", ("log", "simple"))
def test_return_explorer_performance_comes_directly_from_adjusted_prices(
    return_method: str,
) -> None:
    index = pd.date_range("2024-01-02", periods=4, freq="B")
    prices = pd.DataFrame({"asset": [100.0, 110.0, 105.0, 120.0]}, index=index)

    figure = build_return_explorer(
        prices,
        "asset",
        return_method=return_method,
        rolling_window=2,
    )

    expected_cumulative = np.array([0.0, 0.10, 0.05, 0.20])
    expected_drawdown = np.array([0.0, 0.0, 105.0 / 110.0 - 1.0, 0.0])
    np.testing.assert_allclose(figure.data[1].y, expected_cumulative)
    np.testing.assert_allclose(figure.data[2].y, expected_drawdown)
    assert tuple(figure.data[1].x) == tuple(index)
    assert figure.data[1].name == "Cumulative change from adjusted price"
    assert figure.data[2].name == "Adjusted-price drawdown"


def test_market_dashboards_have_stable_visual_contracts() -> None:
    index = pd.date_range("2024-01-01", periods=40, freq="D")
    phase = np.linspace(0, 4, len(index))
    macro = pd.DataFrame(
        {
            "banxico_target_rate": 0.09 + 0.01 * np.sin(phase),
            "mexico_inflation": np.linspace(0.05, 0.04, len(index)),
            "udi": 7.5 * np.exp(np.linspace(0, 0.08, len(index))) + 0.03 * np.cos(phase * 1.4),
            "usd_mxn": 17 + np.sin(phase),
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
        {"asset": 100 * np.exp(np.linspace(0, 0.1, len(index)))},
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
            "return": np.sin(np.linspace(0, 3, len(index))) / 100,
            "garch_filtered_volatility": np.full(len(index), 0.012),
            "ewma_volatility": np.full(len(index), 0.011),
        },
        index=index,
    )

    macro_figure = build_macro_dashboard(
        macro,
        "banxico_target_rate",
        "udi",
        rolling_window=6,
    )
    return_figure = build_return_explorer(prices, "asset", rolling_window=5)
    volatility_figure = build_volatility_dashboard(
        filtered,
        asset="asset",
        persistence=0.98,
    )

    assert len(macro_figure.data) == 3
    assert len(return_figure.data) == 5
    assert len(volatility_figure.data) == 3
    assert "offline data" in macro_figure.layout.title.text
    expected_correlation = (
        macro["banxico_target_rate"]
        .diff()
        .multiply(100)
        .rolling(6)
        .corr(macro["udi"].pct_change(fill_method=None))
    )
    np.testing.assert_allclose(
        np.asarray(macro_figure.data[2].y, dtype=float),
        expected_correlation.to_numpy(),
        equal_nan=True,
    )
    assert macro_figure.data[2].name == "Rolling correlation"
    assert any(
        "Banxico target rate (percentage-point change)" in annotation.text
        and "UDI (percent change)" in annotation.text
        for annotation in macro_figure.layout.annotations
    )
    assert macro_figure.layout.yaxis.title.text == "Banxico target rate (%)"
    assert macro_figure.layout.yaxis2.title.text == "UDI (MXN per UDI)"
    assert tuple(macro_figure.layout.yaxis3.range) == (-1.05, 1.05)
    assert macro_figure.layout.height >= 900
    _assert_vertical_domains(macro_figure, 3)
    assert any(
        "<br>" in annotation.text and "rolling correlation" in annotation.text
        for annotation in macro_figure.layout.annotations
    )

    assert macro_figure.data[0].line.color == SEMANTIC_COLORS["primary"]
    assert macro_figure.data[0].line.dash == "solid"
    assert macro_figure.data[1].line.color == SEMANTIC_COLORS["comparison"]
    assert macro_figure.data[1].line.dash == "dash"
    assert macro_figure.data[2].line.color == SEMANTIC_COLORS["highlight"]
    assert macro_figure.data[2].line.dash == "dot"

    assert return_figure.layout.showlegend is True
    assert return_figure.data[1].line.color == SEMANTIC_COLORS["primary"]
    assert return_figure.data[2].line.color == SEMANTIC_COLORS["negative"]
    assert return_figure.data[2].line.dash == "dash"
    assert return_figure.data[3].line.color == SEMANTIC_COLORS["highlight"]
    assert return_figure.layout.yaxis.title.text == "Adjusted price (USD)"
    assert return_figure.layout.yaxis2.title.text == "Adjusted-close change / drawdown"
    assert return_figure.layout.yaxis3.title.text == "Annualized volatility (252 trading days/year)"
    assert "5-trading-day" in return_figure.data[3].name
    assert return_figure.layout.xaxis4.title.text == "Daily log return"
    assert "daily log returns" in return_figure.layout.title.text
    assert [trace.yaxis for trace in return_figure.data] == [
        "y",
        "y2",
        "y2",
        "y3",
        "y4",
    ]
    assert return_figure.layout.height >= 1100
    _assert_vertical_domains(return_figure, 4)

    for figure in (macro_figure, return_figure, volatility_figure):
        template_layout = figure.layout.template.layout
        assert tuple(template_layout.colorway) == SERIES_COLORS
        assert template_layout.paper_bgcolor == BACKGROUND

    macro_note = macro_figure.layout.annotations[-1].text
    return_note = return_figure.layout.annotations[-1].text
    assert "Sample: 2024-01-01 to 2024-02-09" in macro_note
    assert "Source: Banxico SIE and DB.NOMICS test snapshot" in macro_note
    assert "Source: Classroom price test snapshot" in return_note


def test_financial_dashboards_return_figures_and_auditable_outputs() -> None:
    expected_returns = pd.Series([0.08, 0.12], index=["bond", "equity"])
    covariance = pd.DataFrame(
        [[0.01, 0.004], [0.004, 0.04]],
        index=expected_returns.index,
        columns=expected_returns.index,
    )
    frontier, weights = build_efficient_frontier_dashboard(
        expected_returns,
        covariance,
        risk_free_rate=0.04,
        points=20,
    )

    scenario = pd.DataFrame(
        {
            "shock_bps": [-100, 0, 100],
            "exact_price": [105, 100, 95],
            "duration_price": [104.8, 100, 95.2],
            "duration_convexity_price": [105, 100, 95],
        }
    )
    bond = build_bond_sensitivity_dashboard(
        scenario,
        price=100,
        macaulay_duration=4.5,
        modified_duration=4.2,
        convexity=22,
        face_value=100,
        coupon_rate=0.08,
        maturity=5,
        ytm=0.07,
        frequency=2,
    )
    options = build_black_scholes_dashboard(
        [80, 100, 120],
        [0, 0, 20],
        [2, 10, 24],
        current_spot=100,
        strike=100,
        title="Call contract",
    )
    risk = build_var_cvar_dashboard(
        pd.Series([-0.02, -0.01, 0.0, 0.01]),
        pd.Series({"historical_var": 0.02, "expected_shortfall": 0.025}),
    )

    assert len(frontier.data) == 3
    assert weights.columns.tolist() == ["gmvp", "tangency"]
    assert np.allclose(weights.sum(axis=0), 1)
    assert len(bond.data) == 3
    assert len(options.data) == 2
    assert len(risk.layout.shapes) == 1
    assert frontier.data[0].line.color == SEMANTIC_COLORS["primary"]
    assert frontier.data[1].marker.color == SEMANTIC_COLORS["highlight"]
    assert frontier.data[2].marker.color == SEMANTIC_COLORS["comparison"]
    assert [trace.line.color for trace in bond.data] == [
        SEMANTIC_COLORS["reference"],
        SEMANTIC_COLORS["comparison"],
        SEMANTIC_COLORS["primary"],
    ]
    assert [trace.marker.symbol for trace in bond.data] == ["circle", "square", "diamond"]
    assert bond.layout.xaxis.title.text == "Parallel yield shock (basis points)"
    assert bond.layout.yaxis.title.text == "Bond price (currency units)"
    assert "Macaulay duration 4.50 years" in bond.layout.title.text
    assert "annual coupon 8.00%" in bond.layout.annotations[-1].text
    assert "base annual YTM 7.00%" in bond.layout.annotations[-1].text
    assert options.data[0].line.color == SEMANTIC_COLORS["reference"]
    assert options.data[1].line.color == SEMANTIC_COLORS["primary"]
    assert len(options.layout.shapes) == 1
    assert options.layout.shapes[0].line.color == SEMANTIC_COLORS["highlight"]
    assert risk.data[0].marker.color == SEMANTIC_COLORS["comparison"]
    assert risk.data[0].x[0] == pytest.approx(0.02)
    assert risk.layout.xaxis.title.text == "One-period loss"
    assert risk.layout.xaxis2.title.text == "Non-negative loss estimate"
    assert risk.layout.shapes[0].line.color == SEMANTIC_COLORS["highlight"]
    assert risk.data[1].name == "ES tail mean"
    assert risk.data[1].marker.symbol == "diamond"
    assert risk.data[1].marker.color == SEMANTIC_COLORS["negative"]
    assert risk.data[1].x[0] == pytest.approx(0.025)
    assert risk.data[1].y[0] == 0
    assert risk.data[2].marker.symbol == "diamond"


def test_macro_raw_mode_separates_incompatible_native_units() -> None:
    index = pd.date_range("2024-01-31", periods=18, freq="ME")
    macro = pd.DataFrame(
        {
            "banxico_target_rate": np.linspace(0.11, 0.08, len(index)),
            "usd_mxn": np.linspace(17.0, 19.5, len(index)),
        },
        index=index,
    )

    figure = build_macro_dashboard(
        macro,
        "banxico_target_rate",
        "usd_mxn",
        rolling_window=6,
    )

    assert len(figure.data) == 3
    assert figure.data[0].yaxis == "y"
    assert figure.data[1].yaxis == "y2"
    assert figure.data[2].yaxis == "y3"
    assert "Banxico target rate (%)" == figure.layout.yaxis.title.text
    assert figure.layout.yaxis.tickformat == ".1%"
    assert "USD/MXN (MXN per USD)" == figure.layout.yaxis2.title.text
    assert tuple(figure.layout.yaxis3.range) == (-1.05, 1.05)


def test_macro_dashboard_rejects_comparing_a_series_with_itself() -> None:
    index = pd.date_range("2024-01-31", periods=12, freq="ME")
    macro = pd.DataFrame(
        {"banxico_target_rate": np.linspace(0.11, 0.08, len(index))},
        index=index,
    )

    with pytest.raises(ValueError, match="must be different"):
        build_macro_dashboard(
            macro,
            "banxico_target_rate",
            "banxico_target_rate",
        )


def test_macro_dashboard_rejects_rebasing_rate_series() -> None:
    index = pd.date_range("2024-01-31", periods=12, freq="ME")
    macro = pd.DataFrame(
        {
            "banxico_target_rate": np.linspace(0.11, 0.08, len(index)),
            "udi": np.linspace(7.5, 8.0, len(index)),
        },
        index=index,
    )

    with pytest.raises(ValueError, match="positive level series"):
        build_macro_dashboard(
            macro,
            "banxico_target_rate",
            "udi",
            normalize=True,
        )


@pytest.mark.parametrize(
    "relative_path",
    ("src/dashboards.py", "src/dashboard_fallbacks.py"),
)
def test_dashboard_builders_do_not_define_local_hex_colors_or_templates(
    relative_path: str,
) -> None:
    source = Path(relative_path).read_text(encoding="utf-8")

    assert re.search(r"#[0-9A-Fa-f]{6}", source) is None
    assert "plotly_white" not in source
