from __future__ import annotations

from io import BytesIO
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from scripts.generate_module_concept_maps import render_m2_time_series_bridge
from src.module2_visuals import (
    build_acf_pacf_figure,
    build_forecast_comparison,
    build_innovation_tail_risk_figure,
    build_level_return_diagnostics,
    build_residual_diagnostics,
    build_volatility_comparison,
)
from src.visual_style import INLINE_FIGURE_DPI, series_color, series_linestyle


def _dated_inputs() -> tuple[pd.Series, pd.Series]:
    dates = pd.date_range("2024-01-02", periods=180, freq="B")
    phase = np.linspace(0, 12, len(dates))
    returns = pd.Series(
        0.0002 + 0.006 * np.sin(phase) + 0.002 * np.cos(phase * 1.7),
        index=dates,
        name="usd_mxn_log_return",
    )
    levels = pd.Series(
        17.1 * np.exp(returns.cumsum()),
        index=dates,
        name="usd_mxn_level",
    )
    return levels, returns


def _residuals() -> pd.Series:
    dates = pd.date_range("2023-02-01", periods=160, freq="B")
    phase = np.linspace(0, 18, len(dates))
    return pd.Series(
        0.004 * np.sin(phase) + 0.003 * np.cos(phase * 2.1),
        index=dates,
        name="arima_residual",
    )


def _render_png(figure: Figure) -> bytes:
    output = BytesIO()
    figure.savefig(output, format="png", dpi=INLINE_FIGURE_DPI)
    return output.getvalue()


def _note(figure: Figure) -> str:
    return " ".join(text.get_text().replace("\n", " ") for text in figure.texts)


def _assert_publication_geometry(figure: Figure) -> None:
    """Protect the narrow, vertically stacked Module 2 figure contract."""
    assert figure.get_figwidth() <= 7.5
    if len(figure.axes) > 1:
        positions = [axis.get_position() for axis in figure.axes]
        centers = [position.x0 + position.width / 2 for position in positions]
        assert max(centers) - min(centers) < 0.04
        assert all(
            upper.y0 > lower.y1
            for upper, lower in zip(positions, positions[1:])
        )

    output = BytesIO()
    figure.savefig(output, format="png", bbox_inches="tight")
    tight_width = int.from_bytes(output.getvalue()[16:20], "big")
    nominal_width = figure.get_figwidth() * figure.dpi
    assert tight_width <= nominal_width + figure.dpi * 0.75


def test_module2_concept_map_separates_modeling_spine_from_market_context() -> None:
    figure = render_m2_time_series_bridge()

    try:
        assert tuple(figure.get_size_inches()) == pytest.approx((16, 9))
        visible_text = " ".join(
            artist.get_text().replace("\n", " ")
            for artist in figure.axes[0].texts
            if artist.get_visible()
        )
        for fragment in (
            "MODELING SPINE",
            "Price level",
            "Return transform",
            "Diagnostics",
            "Mean & variance models",
            "Forecast & risk",
            "MARKET CONTEXT — NOT AUTOMATIC MODEL INPUTS",
            "Volume",
            "Liquidity",
            "Interpretation & limits",
        ):
            assert fragment in visible_text

        arrows = [
            artist
            for artist in figure.axes[0].patches
            if isinstance(artist, FancyArrowPatch)
        ]
        assert len(arrows) == 6
        assert sum(arrow.get_linestyle() != "solid" for arrow in arrows) == 2
    finally:
        plt.close(figure)


def test_level_return_diagnostics_has_units_dates_sample_note_and_no_mutation() -> None:
    levels, returns = _dated_inputs()
    original_levels = levels.copy(deep=True)
    original_returns = returns.copy(deep=True)

    figure = build_level_return_diagnostics(
        levels,
        returns,
        level_label="USD/MXN FIX",
        level_unit="MXN per USD",
        rolling_window=21,
        source="Banxico SIE official snapshot",
        data_mode="offline snapshot",
    )

    try:
        assert isinstance(figure, Figure)
        assert figure.dpi == INLINE_FIGURE_DPI
        assert len(figure.axes) == 4
        assert [axis.get_title(loc="left") for axis in figure.axes] == [
            "USD/MXN FIX level",
            "Log return observations",
            "21-observation rolling sample mean",
            "21-observation rolling sample volatility",
        ]
        assert [axis.get_ylabel() for axis in figure.axes] == [
            "Level (MXN per USD)",
            "Log return (%)",
            "Rolling mean (%)",
            "Per-observation volatility (%)",
        ]
        assert [axis.get_xlabel() for axis in figure.axes] == ["", "", "", "Date"]
        note = _note(figure)
        assert "2024-01-02 to 2024-09-09" in note
        assert "Banxico SIE official snapshot" in note
        assert "Data mode: offline snapshot" in note
        assert "in-sample descriptions, not forecasts" in note
        assert "volatility is not annualized" in note
        _assert_publication_geometry(figure)
        pd.testing.assert_series_equal(levels, original_levels)
        pd.testing.assert_series_equal(returns, original_returns)
    finally:
        plt.close(figure)


def test_acf_pacf_uses_explicit_95_percent_bands_and_metadata() -> None:
    _, returns = _dated_inputs()

    with (
        patch("src.module2_visuals.plot_acf", wraps=plot_acf) as acf_spy,
        patch("src.module2_visuals.plot_pacf", wraps=plot_pacf) as pacf_spy,
    ):
        figure = build_acf_pacf_figure(
            returns,
            title="USD/MXN return dependence",
            lags=25,
            source="Banxico",
            data_mode="snapshot",
        )

    try:
        assert len(figure.axes) == 2
        assert acf_spy.call_args.kwargs["alpha"] == pytest.approx(0.05)
        assert acf_spy.call_args.kwargs["bartlett_confint"] is False
        assert acf_spy.call_args.kwargs["zero"] is False
        assert pacf_spy.call_args.kwargs["alpha"] == pytest.approx(0.05)
        assert pacf_spy.call_args.kwargs["zero"] is False
        assert [axis.get_title(loc="left") for axis in figure.axes] == [
            "Autocorrelation function (95% bands)",
            "Partial autocorrelation function (95% bands)",
        ]
        assert all(axis.collections for axis in figure.axes)
        for axis in figure.axes:
            lower, upper = axis.get_ylim()
            assert lower == pytest.approx(-upper)
            assert 0.15 <= upper <= 1.0
        note = _note(figure)
        assert "approximate 95% confidence bands" in note
        assert "white-noise standard-error reference" in note
        assert "alpha=0.05" in note
        assert "alpha=0.5" not in note
        assert "Source: Banxico" in note
        _assert_publication_geometry(figure)
    finally:
        plt.close(figure)


def test_residual_diagnostics_has_four_distinct_sample_checks() -> None:
    residuals = _residuals()
    original = residuals.copy(deep=True)

    figure = build_residual_diagnostics(
        residuals,
        title="ARIMA residual diagnostics",
        residual_unit="decimal log return / FIX publication interval",
        lags=20,
        source="Fitted ARIMA on Banxico snapshot",
        data_mode="offline snapshot",
    )

    try:
        assert len(figure.axes) == 4
        assert [axis.get_title(loc="left") for axis in figure.axes] == [
            "Residual path",
            "Residual ACF (95% bands)",
            "Squared-residual ACF (95% bands)",
            "Normal QQ plot",
        ]
        assert figure.axes[0].get_xlabel() == "Date"
        assert figure.axes[3].get_xlabel() == "Theoretical normal quantile"
        assert figure.axes[0].get_ylabel() == (
            "Residual\n(decimal log return / FIX publication interval)"
        )
        assert figure.axes[3].get_ylabel() == (
            "Ordered residual\n(decimal log return / FIX publication interval)"
        )
        assert len(figure.axes[3].collections) == 1
        for axis in figure.axes[1:3]:
            lower, upper = axis.get_ylim()
            assert lower == pytest.approx(-upper)
            assert 0.15 <= upper <= 1.0
        assert "not proof of white noise or Gaussian innovations" in _note(figure)
        assert "Residual unit: decimal log return / FIX publication interval" in _note(
            figure
        )
        assert "squared residuals screen for remaining variance dependence" in _note(
            figure
        )
        _assert_publication_geometry(figure)
        pd.testing.assert_series_equal(residuals, original)
    finally:
        plt.close(figure)


def test_volatility_comparison_converts_units_and_pairs_color_with_line_style() -> None:
    _, returns = _dated_inputs()
    paths = pd.DataFrame(
        {
            "ARCH(3)": 0.007 + 0.001 * np.sin(np.linspace(0, 6, len(returns))),
            "GARCH(1,1)": 0.008 + 0.001 * np.cos(np.linspace(0, 6, len(returns))),
        },
        index=returns.index,
    )
    original_returns = returns.copy(deep=True)
    original_paths = paths.copy(deep=True)

    figure = build_volatility_comparison(
        returns,
        paths,
        title="ARCH and GARCH volatility comparison",
        return_scale="decimal",
        volatility_scale="decimal",
        annualization_factor=252,
        source="Banxico SIE official snapshot",
        data_mode="snapshot",
    )

    try:
        assert len(figure.axes) == 2
        assert figure.axes[0].get_ylabel() == "Return (%)"
        assert figure.axes[1].get_ylabel() == "Annualized volatility (%)"
        assert np.allclose(
            figure.axes[0].lines[0].get_ydata(),
            returns.to_numpy() * 100,
        )
        volatility_lines = [
            line
            for line in figure.axes[1].lines
            if line.get_label() in {"ARCH(3)", "GARCH(1,1)"}
        ]
        assert [line.get_color() for line in volatility_lines] == [
            series_color(0),
            series_color(1),
        ]
        assert [line.get_linestyle() for line in volatility_lines] == [
            series_linestyle(0),
            series_linestyle(1),
        ]
        assert len({line.get_marker() for line in volatility_lines}) == 2
        assert np.allclose(
            volatility_lines[0].get_ydata(),
            paths["ARCH(3)"].to_numpy() * np.sqrt(252) * 100,
        )
        note = _note(figure)
        assert "volatility annualized with sqrt(252)" in note
        assert "Paths are in-sample estimates, not forecasts" in note
        _assert_publication_geometry(figure)
        pd.testing.assert_series_equal(returns, original_returns)
        pd.testing.assert_frame_equal(paths, original_paths)
    finally:
        plt.close(figure)


def test_forecast_comparison_uses_one_evaluation_sample_and_cumulative_error() -> None:
    _, observed = _dated_inputs()
    observed = observed.tail(40).copy()
    forecasts = pd.DataFrame(
        {
            "ARIMA fixed-origin forecast": observed.shift(1).fillna(0.0),
            "Training-mean benchmark": np.repeat(
                observed.iloc[:-1].mean(),
                len(observed),
            ),
        },
        index=observed.index,
    )
    prediction_interval = pd.DataFrame(
        {
            "lower": forecasts["ARIMA fixed-origin forecast"] - 0.012,
            "upper": forecasts["ARIMA fixed-origin forecast"] + 0.012,
        },
        index=observed.index,
    )
    original_observed = observed.copy(deep=True)
    original_forecasts = forecasts.copy(deep=True)

    figure = build_forecast_comparison(
        observed,
        forecasts,
        title="Chronological evaluation comparison",
        scale="decimal",
        prediction_interval=prediction_interval,
        source="Deterministic test fixture",
        data_mode="synthetic",
    )

    try:
        assert len(figure.axes) == 2
        assert [axis.get_title(loc="left") for axis in figure.axes] == [
            "Observed evaluation returns and fixed forecasts",
            "Cumulative absolute forecast error",
        ]
        assert figure.axes[0].get_ylabel() == "Return (%)"
        assert len(figure.axes[0].collections) == 1
        assert (
            figure.axes[1].get_ylabel()
            == "Cumulative absolute error (percentage points)"
        )
        assert np.allclose(
            figure.axes[0].lines[0].get_ydata(),
            observed.to_numpy() * 100,
        )
        expected_final_error = (
            forecasts.sub(observed, axis=0).abs().sum().multiply(100)
        )
        error_lines = {
            line.get_label(): line for line in figure.axes[1].lines
        }
        for label, expected in expected_final_error.items():
            assert error_lines[label].get_ydata()[-1] == pytest.approx(expected)
        note = _note(figure)
        assert "lower is better on this shared evaluation sample" in note
        assert "not trading profit or loss" in note
        assert "model-based interval" in note
        _assert_publication_geometry(figure)
        pd.testing.assert_series_equal(observed, original_observed)
        pd.testing.assert_frame_equal(forecasts, original_forecasts)
    finally:
        plt.close(figure)


def test_innovation_tail_figure_standardizes_tails_and_separates_loss_scales() -> None:
    risk_measures = pd.DataFrame(
        {
            "log_return_loss_threshold_pct": [1.57, 1.75],
            "simple_return_loss_pct": [1.56, 1.74],
        },
        index=["Gaussian GARCH", "Student-t GARCH"],
    )
    original = risk_measures.copy(deep=True)

    figure = build_innovation_tail_risk_figure(
        risk_measures,
        alpha=0.01,
        degrees_of_freedom=7.5,
        source="Deterministic risk fixture",
        data_mode="synthetic",
    )

    try:
        assert [axis.get_title(loc="left") for axis in figure.axes] == [
            "Standardized innovation distributions and left tails",
            "One-step VaR on log-return and simple-loss scales",
        ]
        assert figure.axes[0].get_xlabel() == "Standardized innovation"
        assert figure.axes[1].get_ylabel() == "Positive loss magnitude (%)"
        assert len(figure.axes[0].collections) == 2
        assert len(figure.axes[1].patches) == 4
        assert any(
            "Student-t 1% quantile" in line.get_label()
            for line in figure.axes[0].lines
        )
        assert "scaled to unit variance" in _note(figure)
        _assert_publication_geometry(figure)
        pd.testing.assert_frame_equal(risk_measures, original)
    finally:
        plt.close(figure)


def test_rendered_png_is_deterministic() -> None:
    _, returns = _dated_inputs()
    first = build_acf_pacf_figure(returns, title="Deterministic diagnostic", lags=20)
    second = build_acf_pacf_figure(returns, title="Deterministic diagnostic", lags=20)

    try:
        first_png = _render_png(first)
        second_png = _render_png(second)
        assert first_png.startswith(b"\x89PNG\r\n\x1a\n")
        assert first_png == second_png
    finally:
        plt.close(first)
        plt.close(second)


def test_level_return_diagnostics_rejects_invalid_inputs() -> None:
    levels, returns = _dated_inputs()
    with pytest.raises(ValueError, match="rolling_window must be at least 2"):
        build_level_return_diagnostics(
            levels,
            returns,
            level_label="USD/MXN",
            level_unit="MXN per USD",
            rolling_window=1,
        )
    with pytest.raises(ValueError, match="DatetimeIndex"):
        build_level_return_diagnostics(
            levels.reset_index(drop=True),
            returns,
            level_label="USD/MXN",
            level_unit="MXN per USD",
        )
    invalid_returns = returns.copy()
    invalid_returns.iloc[10] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        build_level_return_diagnostics(
            levels,
            invalid_returns,
            level_label="USD/MXN",
            level_unit="MXN per USD",
        )


def test_correlation_and_residual_builders_reject_unusable_lags_and_series() -> None:
    _, returns = _dated_inputs()
    with pytest.raises(ValueError, match="less than half"):
        build_acf_pacf_figure(returns.iloc[:40], title="Too many lags", lags=20)
    with pytest.raises(ValueError, match="must vary"):
        build_acf_pacf_figure(
            pd.Series(np.ones(80), index=returns.index[:80]),
            title="Constant",
            lags=10,
        )
    with pytest.raises(ValueError, match="lags must be at least 1"):
        build_residual_diagnostics(_residuals(), title="Residuals", lags=0)


def test_volatility_comparison_rejects_scale_sign_alignment_and_factor_errors() -> None:
    _, returns = _dated_inputs()
    valid_path = pd.Series(0.01, index=returns.index, name="GARCH")

    with pytest.raises(ValueError, match="return_scale"):
        build_volatility_comparison(
            returns,
            {"GARCH": valid_path},
            title="Invalid scale",
            return_scale="basis-points",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="negative"):
        build_volatility_comparison(
            returns,
            {"GARCH": valid_path.where(valid_path.index != valid_path.index[5], -0.01)},
            title="Negative volatility",
        )
    with pytest.raises(ValueError, match="same dated index"):
        build_volatility_comparison(
            returns,
            {"GARCH": valid_path.iloc[1:]},
            title="Misaligned",
        )
    with pytest.raises(ValueError, match="annualization_factor"):
        build_volatility_comparison(
            returns,
            {"GARCH": valid_path},
            title="Invalid factor",
            annualization_factor=0,
        )


def test_forecast_comparison_rejects_missing_or_misaligned_evaluation_values() -> None:
    _, observed = _dated_inputs()
    forecasts = pd.DataFrame(
        {"benchmark": observed.shift(1)},
        index=observed.index,
    )
    with pytest.raises(ValueError, match="cannot contain missing"):
        build_forecast_comparison(
            observed,
            forecasts,
            title="Missing forecast",
        )
    with pytest.raises(ValueError, match="same dated index"):
        build_forecast_comparison(
            observed,
            forecasts.dropna(),
            title="Misaligned forecast",
        )
    invalid_interval = pd.DataFrame(
        {
            "lower": observed + 0.01,
            "upper": observed - 0.01,
        },
        index=observed.index,
    )
    with pytest.raises(ValueError, match="lower values cannot exceed"):
        build_forecast_comparison(
            observed,
            pd.DataFrame({"benchmark": observed}, index=observed.index),
            title="Reversed interval",
            prediction_interval=invalid_interval,
        )


def test_innovation_tail_figure_rejects_invalid_distribution_or_risk_units() -> None:
    valid = pd.DataFrame(
        {
            "log_return_loss_threshold_pct": [1.5],
            "simple_return_loss_pct": [1.4],
        },
        index=["model"],
    )
    with pytest.raises(ValueError, match="greater than 2"):
        build_innovation_tail_risk_figure(
            valid,
            alpha=0.01,
            degrees_of_freedom=2,
        )
    invalid = valid.copy()
    invalid.iloc[0, 1] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        build_innovation_tail_risk_figure(
            invalid,
            alpha=0.01,
            degrees_of_freedom=7,
        )
