from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.figure import Figure
from PIL import Image

from scripts.generate_module7_risk_diagrams import ASSETS, CANVAS_SIZE, FIGURE_SIZE
from src.module7_visuals import (
    build_convergence_figure,
    build_ewma_volatility_figure,
    build_heston_diagnostics_figure,
    build_implied_volatility_figure,
    build_option_payoff_figure,
    build_tail_risk_figure,
    build_var_backtest_figure,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ASSET_IDS = {
    "risk-var-tail-loss",
    "risk-cvar-expected-shortfall",
    "risk-skewness-tail-orientation",
}


def _assert_inline_size(figure: Figure) -> None:
    width, height = figure.get_size_inches()
    assert width <= 7.5
    assert height <= 7.5


def _all_figure_text(figure: Figure) -> str:
    return " ".join(
        artist.get_text().replace("\n", " ")
        for artist in figure.findobj()
        if hasattr(artist, "get_text") and artist.get_text()
    )


def test_option_payoff_figure_is_unit_explicit_and_does_not_mutate_inputs() -> None:
    terminal_prices = pd.Series([80.0, 90.0, 100.0, 110.0, 120.0], name="terminal_price")
    original = terminal_prices.copy(deep=True)

    figure = build_option_payoff_figure(terminal_prices, 100.0)

    try:
        assert isinstance(figure, Figure)
        _assert_inline_size(figure)
        axis = figure.axes[0]
        np.testing.assert_allclose(axis.lines[0].get_ydata(), [0, 0, 0, 10, 20])
        np.testing.assert_allclose(axis.lines[1].get_ydata(), [20, 10, 0, 0, 0])
        assert axis.get_xlabel() == "Underlying price at expiry (price units)"
        assert axis.get_ylabel() == "Payoff (same price units)"
        assert "premium, discounting, and probability are excluded" in _all_figure_text(figure)
        pd.testing.assert_series_equal(terminal_prices, original)
    finally:
        plt.close(figure)


def test_convergence_figure_accepts_lesson_aliases_and_reconciles_error() -> None:
    comparison = pd.DataFrame(
        {
            "steps": [5, 10, 25, 50],
            "binomial_price": [9.6, 9.8, 9.91, 9.96],
            "black_scholes_price": [10.0, 10.0, 10.0, 10.0],
            "pricing_error": [-0.4, -0.2, -0.09, -0.04],
        }
    )
    original = comparison.copy(deep=True)

    figure = build_convergence_figure(comparison)

    try:
        _assert_inline_size(figure)
        assert len(figure.axes) == 2
        price_axis, error_axis = figure.axes
        np.testing.assert_allclose(price_axis.lines[0].get_ydata(), comparison["binomial_price"])
        np.testing.assert_allclose(error_axis.lines[0].get_ydata(), comparison["pricing_error"])
        assert error_axis.get_xlabel() == "Tree steps (count)"
        assert "implementation diagnostic, not market validation" in _all_figure_text(figure)
        pd.testing.assert_frame_equal(comparison, original)
    finally:
        plt.close(figure)


def test_convergence_figure_rejects_an_unreconciled_error_column() -> None:
    comparison = pd.DataFrame(
        {
            "steps": [5, 10],
            "binomial_price": [9.5, 9.8],
            "black_scholes_price": [10.0, 10.0],
            "pricing_error": [0.5, 0.2],
        }
    )

    with pytest.raises(ValueError, match="estimate minus benchmark"):
        build_convergence_figure(comparison)


def test_implied_volatility_figure_separates_input_and_recovered_series() -> None:
    chain = pd.DataFrame(
        {
            "strike": [80.0, 90.0, 100.0, 110.0, 120.0],
            "generating_volatility": [0.28, 0.25, 0.23, 0.22, 0.225],
            "implied_volatility": [0.28, 0.25, 0.23, 0.22, 0.225],
        }
    )
    original = chain.copy(deep=True)

    figure = build_implied_volatility_figure(chain)

    try:
        _assert_inline_size(figure)
        axis = figure.axes[0]
        assert [line.get_linestyle() for line in axis.lines] == ["--", "-"]
        assert [line.get_marker() for line in axis.lines] == ["s", "o"]
        assert axis.get_ylabel() == "Annualized volatility (%)"
        np.testing.assert_allclose(axis.lines[-1].get_ydata(), chain["implied_volatility"] * 100)
        assert "not a forecast of realized volatility" in _all_figure_text(figure)
        pd.testing.assert_frame_equal(chain, original)
    finally:
        plt.close(figure)


def test_heston_diagnostics_use_the_complete_horizon_and_quantile_bands() -> None:
    time = np.linspace(0.0, 1.0, 9)
    path_offsets = np.array([-0.08, -0.03, 0.02, 0.07])
    heston_spots = 100 * np.exp(time[:, None] * (0.03 + path_offsets[None, :]))
    gbm_paths = 100 * np.exp(time[:, None] * (0.04 + path_offsets[None, :] / 2))
    heston_variances = np.square(
        0.20 + time[:, None] * np.array([0.02, -0.01, 0.04, -0.02])[None, :]
    )
    original_spots = heston_spots.copy()

    figure = build_heston_diagnostics_figure(
        heston_spots,
        heston_variances,
        gbm_paths,
        maturity=1.0,
    )

    try:
        _assert_inline_size(figure)
        assert len(figure.axes) == 3
        spot_axis, volatility_axis, terminal_axis = figure.axes
        np.testing.assert_allclose(spot_axis.lines[0].get_xdata(), time)
        assert spot_axis.lines[0].get_xdata()[-1] == pytest.approx(1.0)
        assert any(isinstance(item, PolyCollection) for item in spot_axis.collections)
        assert any(isinstance(item, PolyCollection) for item in volatility_axis.collections)
        assert terminal_axis.get_xlabel() == "Terminal spot (price units)"
        assert "not confidence intervals" in _all_figure_text(figure)
        np.testing.assert_array_equal(heston_spots, original_spots)
    finally:
        plt.close(figure)


def test_heston_diagnostics_reject_non_finite_paths_and_misaligned_time_grids() -> None:
    valid = np.ones((5, 3)) * 100
    variances = np.ones((5, 3)) * 0.04
    invalid = valid.copy()
    invalid[-1, -1] = np.nan

    with pytest.raises(ValueError, match="finite"):
        build_heston_diagnostics_figure(invalid, variances, valid)
    with pytest.raises(ValueError, match="same time grid"):
        build_heston_diagnostics_figure(valid, variances, np.ones((4, 3)) * 100)


def test_tail_risk_figure_uses_loss_space_and_marks_es_as_a_tail_mean() -> None:
    returns = pd.Series(
        [-0.055, -0.040, -0.030, -0.022, -0.014, -0.008, -0.004, 0.0, 0.003, 0.006] * 4,
        name="portfolio_simple_return",
    )
    original = returns.copy(deep=True)

    figure = build_tail_risk_figure(
        returns,
        historical_var=0.04,
        expected_shortfall=0.05,
        gaussian_var=0.037,
    )

    try:
        _assert_inline_size(figure)
        axis = figure.axes[0]
        assert axis.get_xlabel() == "One-period portfolio loss (%; gains are negative)"
        assert any(isinstance(item, PathCollection) for item in axis.collections)
        text = _all_figure_text(figure)
        assert "Mean/integral of tail loss" in text
        assert "not a second cutoff" in text
        assert "ES is the average (quantile integral)" in text
        pd.testing.assert_series_equal(returns, original)
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    ("historical_var", "expected_shortfall", "message"),
    [
        (0.05, 0.04, "at least historical_var"),
        (-0.01, 0.04, "non-negative"),
        (0.03, np.inf, "finite"),
    ],
)
def test_tail_risk_figure_rejects_invalid_loss_metrics(
    historical_var: float,
    expected_shortfall: float,
    message: str,
) -> None:
    returns = pd.Series(np.linspace(-0.05, 0.04, 30))

    with pytest.raises(ValueError, match=message):
        build_tail_risk_figure(returns, historical_var, expected_shortfall)


def test_ewma_figure_preserves_per_observation_units_and_exact_index() -> None:
    index = pd.date_range("2026-01-02", periods=8, freq="B")
    returns = pd.Series(
        [0.01, -0.02, 0.005, -0.03, 0.008, 0.002, -0.015, 0.012],
        index=index,
    )
    state = pd.Series(
        [0.018, 0.0178, 0.0180, 0.0176, 0.0192, 0.0188, 0.0183, 0.0182],
        index=index,
    )
    original_state = state.copy(deep=True)

    figure = build_ewma_volatility_figure(returns, state)

    try:
        _assert_inline_size(figure)
        return_axis, state_axis = figure.axes
        np.testing.assert_allclose(return_axis.lines[0].get_ydata(), returns * 100)
        np.testing.assert_allclose(state_axis.lines[0].get_ydata(), state * 100)
        assert state_axis.get_xlabel() == "Date"
        assert "not annualized" in _all_figure_text(figure)
        pd.testing.assert_series_equal(state, original_state)
    finally:
        plt.close(figure)


def test_ewma_figure_rejects_a_different_index() -> None:
    returns = pd.Series([0.01, -0.02, 0.03], index=[0, 1, 2])
    state = pd.Series([0.02, 0.02, 0.021], index=[1, 2, 3])

    with pytest.raises(ValueError, match="same ordered index"):
        build_ewma_volatility_figure(returns, state)


def test_var_backtest_marks_exact_strict_exceptions_and_accumulates_them() -> None:
    index = pd.date_range("2026-01-02", periods=6, freq="B")
    returns = pd.Series([-0.01, -0.04, 0.005, -0.03, -0.005, -0.06], index=index)
    forecast = pd.Series([np.nan, np.nan, 0.02, 0.02, 0.02, 0.05], index=index)
    valid_index = index[2:]
    exceptions = pd.Series([False, True, False, True], index=valid_index, name="exception")
    original_forecast = forecast.copy(deep=True)

    figure = build_var_backtest_figure(returns, forecast, exceptions)

    try:
        _assert_inline_size(figure)
        risk_axis, count_axis = figure.axes
        exception_points = next(
            collection
            for collection in risk_axis.collections
            if isinstance(collection, PathCollection)
        )
        assert len(exception_points.get_offsets()) == 2
        np.testing.assert_array_equal(count_axis.lines[0].get_ydata(), [0, 1, 1, 2])
        assert risk_axis.get_ylabel() == "Loss or VaR (%)"
        assert "strict rule L_t > VaR_t" in _all_figure_text(figure)
        pd.testing.assert_series_equal(forecast, original_forecast)
    finally:
        plt.close(figure)


def test_var_backtest_rejects_equality_as_an_exception() -> None:
    returns = pd.Series([-0.01, -0.02, -0.03])
    forecast = pd.Series([0.01, 0.02, 0.025])
    incorrect = pd.Series([True, False, True])

    with pytest.raises(ValueError, match="strict condition"):
        build_var_backtest_figure(returns, forecast, incorrect)


def test_static_risk_assets_are_deterministic_rgb_exports_with_complete_manifest() -> None:
    assert set(ASSETS) == STATIC_ASSET_IDS
    assert FIGURE_SIZE == (12.0, 6.75)
    manifest = json.loads(
        (PROJECT_ROOT / "img/generated/visual-assets/module-07-risk.json").read_text(
            encoding="utf-8"
        )
    )
    assets = {asset["id"]: asset for asset in manifest["assets"]}
    assert set(assets) == STATIC_ASSET_IDS

    for asset_id, asset in assets.items():
        assert asset["status"] == "approved"
        assert asset["production_method"] == "deterministic_matplotlib"
        assert asset["generator_script"].endswith(f"--asset {asset_id}")
        assert "no external data" in asset["source"]
        assert asset["alt_text"].strip()
        assert asset["production_notes"]
        assert "legacy_sources" not in asset
        with Image.open(PROJECT_ROOT / asset["target_path"]) as image:
            assert image.size == CANVAS_SIZE
            assert image.mode == "RGB"
