from __future__ import annotations

from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.module3_visuals import (
    build_macro_fx_evaluation_figure,
    build_macro_fx_history_figure,
    build_macro_fx_scenario_stability_figure,
)
from src.visual_style import AMBER_DARK, MUTED_BLUE, TEAL


def _render_png(figure: Figure) -> bytes:
    output = BytesIO()
    figure.savefig(output, format="png")
    return output.getvalue()


def _history_input() -> pd.DataFrame:
    index = pd.date_range("2022-01-31", periods=24, freq="ME")
    phase = np.linspace(0, 4, len(index))
    return pd.DataFrame(
        {
            "banxico_target_rate": 0.08 + 0.02 * np.sin(phase),
            "cetes_28d": 0.079 + 0.018 * np.sin(phase + 0.1),
            "mexico_inflation_lag1m": 0.045 + 0.01 * np.cos(phase),
            "us_10y": 0.03 + 0.008 * np.sin(phase * 0.7),
            "usd_mxn": 18.5 + 0.8 * np.sin(phase * 1.2),
        },
        index=index,
    )


def test_history_figure_preserves_units_styles_and_source_note() -> None:
    figure = build_macro_fx_history_figure(
        _history_input(),
        sample_label="2022-01-31 to 2023-12-31",
        vintage_label="2026-06-07",
    )

    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 3
        assert [axis.get_ylabel() for axis in figure.axes] == [
            "Annual rate (%)",
            "Annual rate / yield (%)",
            "MXN per USD",
        ]
        assert figure.axes[0].lines[0].get_color() == TEAL
        assert figure.axes[0].lines[0].get_linestyle() == "-"
        assert figure.axes[0].lines[1].get_color() == MUTED_BLUE
        assert figure.axes[0].lines[1].get_linestyle() == "--"
        assert figure.axes[1].lines[0].get_color() == AMBER_DARK
        assert any("SF61745" in text.get_text() for text in figure.texts)
        assert _render_png(figure).startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        plt.close(figure)


def test_evaluation_figure_answers_benchmark_question_in_two_panels() -> None:
    index = pd.date_range("2024-01-31", periods=8, freq="ME")
    actual = np.array([0.01, -0.02, 0.015, -0.01, 0.005, -0.025, 0.012, -0.006])
    frame = pd.DataFrame(
        {
            "Actual next-month return": actual,
            "Ridge prediction": 0.7 * actual,
            "Training-mean benchmark": np.repeat(actual[:5].mean(), len(actual)),
            "Zero-change benchmark": np.zeros(len(actual)),
        },
        index=index,
    )
    mae = {
        "Ridge prediction": 0.012,
        "Training-mean benchmark": 0.010,
        "Zero-change benchmark": 0.011,
    }

    figure = build_macro_fx_evaluation_figure(
        frame,
        mae,
        sample_label="2024-01-31 to 2024-08-31",
        realization_end="2024-09-30",
        vintage_label="2026-06-07",
    )

    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 2
        assert figure._suptitle.get_text() == (
            "Does Ridge beat transparent naive FX benchmarks?"
        )
        assert figure.axes[0].get_ylabel() == "Next-month USD/MXN log return (%)"
        assert figure.axes[1].get_xlabel() == (
            "MAE (monthly log-return percentage points)"
        )
        assert figure.axes[0].lines[0].get_color() == TEAL
        assert figure.axes[0].lines[1].get_color() == AMBER_DARK
        assert len(figure.axes[1].patches) == 3
        assert any("last realization" in text.get_text() for text in figure.texts)
        assert _render_png(figure).startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        plt.close(figure)


def test_scenario_figure_shows_fit_instability_and_error_scale() -> None:
    sensitivity = pd.DataFrame(
        {
            "Evaluation fit": [-33.1, -15.5, 33.1],
            "Updated fit": [46.7, 26.5, -46.7],
        },
        index=[
            "Sticky inflation and domestic tightening",
            "Global long-yield shock",
            "Disinflation and domestic easing",
        ],
    )

    figure = build_macro_fx_scenario_stability_figure(
        sensitivity,
        model_mae_bps=239.4,
        fit_window_labels={
            "Evaluation fit": "Evaluation fit (through 2023-10)",
            "Updated fit": "Updated fit (through 2025-05)",
        },
        sample_label="scenario origin 2025-06-30",
        vintage_label="2026-06-07",
    )

    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 2
        assert figure._suptitle.get_text() == (
            "Are the rejected model's scenario sensitivities stable?"
        )
        assert "Change vs each fit's baseline" in figure.axes[0].get_xlabel()
        assert figure.axes[0].collections[0].get_facecolors()[0].tolist() == list(
            matplotlib.colors.to_rgba(TEAL)
        )
        assert figure.axes[0].collections[1].get_facecolors()[0].tolist() == list(
            matplotlib.colors.to_rgba(AMBER_DARK)
        )
        assert len(figure.axes[1].patches) == 2
        assert any("not a prediction interval" in text.get_text() for text in figure.texts)
        assert _render_png(figure).startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        plt.close(figure)


def test_scenario_figure_does_not_clip_sensitivity_larger_than_mae() -> None:
    sensitivity = pd.DataFrame(
        {"Evaluation fit": [-120.0], "Updated fit": [80.0]},
        index=["Stress"],
    )
    figure = build_macro_fx_scenario_stability_figure(
        sensitivity,
        model_mae_bps=50.0,
        fit_window_labels={
            "Evaluation fit": "Evaluation fit (training window)",
            "Updated fit": "Updated fit (expanded window)",
        },
        sample_label="scenario origin 2025-06-30",
        vintage_label="2026-06-07",
    )

    try:
        assert figure.axes[1].get_xlim()[1] > 120.0
        assert figure.axes[1].get_title(loc="left") == (
            "Scenario movement exceeds held-out error scale"
        )
    finally:
        plt.close(figure)
