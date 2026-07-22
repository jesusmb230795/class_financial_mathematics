from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from PIL import Image

from scripts.generate_module_concept_maps import RENDERERS, save_rgb
from src.module5_visuals import (
    DCF_VALUE_PER_SHARE,
    FORECAST_FCFF_ANATOMY,
    PROJECT_CASH_FLOWS,
    PROJECT_IRR,
    PROJECT_NPV,
    TARGET_DEBT_WEIGHTS,
    TARGET_EQUITY_COST,
    TARGET_LEVERED_BETA,
    TARGET_WACC,
    build_capital_structure_wacc_figure,
    build_dcf_value_sensitivity_figure,
    build_fcff_forecast_waterfall_figure,
    build_project_cash_flow_npv_profile_figure,
    calculate_capital_structure_sensitivity,
    calculate_dcf_value_per_share,
    calculate_fcff_anatomy,
    calculate_project_irr,
    calculate_project_npv,
)
from src.visual_style import AMBER_DARK, BACKGROUND, EXPORT_FIGURE_DPI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORTED_ASSETS = (
    "m5-capital-structure-wacc-sensitivity",
    "m5-dcf-value-sensitivity",
    "m5-project-cash-flow-npv-profile",
    "m5-fcff-forecast-waterfall",
)
ASSET_BUILDERS = {
    "m5-corporate-value-creation-valuation-map": RENDERERS[
        "m5-corporate-value-creation-valuation-map"
    ],
    "m5-capital-structure-wacc-sensitivity": build_capital_structure_wacc_figure,
    "m5-dcf-value-sensitivity": build_dcf_value_sensitivity_figure,
    "m5-project-cash-flow-npv-profile": build_project_cash_flow_npv_profile_figure,
    "m5-fcff-forecast-waterfall": build_fcff_forecast_waterfall_figure,
}


def test_module5_concept_map_states_discounting_and_remains_legible() -> None:
    figure = RENDERERS["m5-corporate-value-creation-valuation-map"]()

    try:
        visible_text = [
            artist
            for artist in figure.findobj()
            if hasattr(artist, "get_text")
            and hasattr(artist, "get_fontsize")
            and artist.get_visible()
            and artist.get_text().strip()
        ]
        combined = " ".join(artist.get_text().replace("\n", " ") for artist in visible_text)
        assert "Discount FCFF at WACC" in combined
        assert "Discount FCFE at cost of equity" in combined
        assert "FCFF + WACC" not in combined
        assert "ENTERPRISE VALUE ROUTE" in combined
        assert "FIRM VALUE ROUTE" not in combined
        assert min(artist.get_fontsize() for artist in visible_text) >= 18

        axis = figure.axes[0]
        for label in ("WACC", "Cost of equity"):
            title = next(text for text in axis.texts if text.get_text() == label)
            x_position, y_position = title.get_position()
            box = next(
                patch
                for patch in axis.patches
                if isinstance(patch, FancyBboxPatch)
                and patch.get_x() <= x_position <= patch.get_x() + patch.get_width()
                and patch.get_y() <= y_position <= patch.get_y() + patch.get_height()
            )
            assert to_rgba(box.get_edgecolor()) == to_rgba(AMBER_DARK)
    finally:
        plt.close(figure)


def test_capital_structure_figure_recomputes_and_labels_required_returns() -> None:
    np.testing.assert_allclose(TARGET_LEVERED_BETA, [0.95, 1.10, 1.29090909, 1.53333333])
    np.testing.assert_allclose(TARGET_EQUITY_COST, [0.09225, 0.10050, 0.11100, 0.12433333])
    np.testing.assert_allclose(TARGET_WACC, [0.08280, 0.08325, 0.08805, 0.0992625])

    figure = build_capital_structure_wacc_figure()

    try:
        assert isinstance(figure, Figure)
        axis = figure.axes[0]
        np.testing.assert_allclose(axis.lines[0].get_xdata(), TARGET_DEBT_WEIGHTS * 100)
        np.testing.assert_allclose(axis.lines[0].get_ydata(), TARGET_WACC * 100)
        assert [line.get_marker() for line in axis.lines[:3]] == ["o", "s", "^"]
        assert len({line.get_linestyle() for line in axis.lines[:3]}) == 3
        assert axis.get_xlabel() == "Target debt / capital (%)"
        assert axis.get_ylabel() == "Effective annual USD rate (%)"
        assert any("simulated teaching scenarios" in text.get_text() for text in figure.texts)
        np.testing.assert_allclose(figure.get_size_inches(), [10.0, 5.625])
    finally:
        plt.close(figure)


def test_dcf_sensitivity_figure_preserves_matrix_values_and_base_case() -> None:
    np.testing.assert_allclose(
        np.round(DCF_VALUE_PER_SHARE, 2),
        [
            [11.67, 14.38, 18.46],
            [9.50, 11.40, 14.07],
            [7.87, 9.27, 11.14],
        ],
        rtol=0,
        atol=0,
    )
    figure = build_dcf_value_sensitivity_figure()

    try:
        assert isinstance(figure, Figure)
        axis = figure.axes[0]
        np.testing.assert_allclose(axis.images[0].get_array(), DCF_VALUE_PER_SHARE)
        np.testing.assert_array_equal(axis.get_xticks(), [0, 1, 2])
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        rendered_tick_centers = [
            tick.get_window_extent(renderer).x0 + tick.get_window_extent(renderer).width / 2
            for tick in axis.get_xticklabels()
        ]
        expected_column_centers = [axis.transData.transform((column, 0))[0] for column in range(3)]
        np.testing.assert_allclose(rendered_tick_centers, expected_column_centers)
        assert axis.get_xlabel() == "Perpetual nominal growth, $g$"
        assert axis.get_ylabel() == "Effective annual USD WACC"
        annotations = {text.get_text() for text in axis.texts}
        assert {"11.67", "11.40", "11.14"} <= annotations
        assert any("not a probability distribution" in text.get_text() for text in figure.texts)
        np.testing.assert_allclose(figure.get_size_inches(), [10.0, 5.625])
    finally:
        plt.close(figure)


def test_project_timeline_and_npv_profile_preserve_financial_contracts() -> None:
    np.testing.assert_allclose(PROJECT_CASH_FLOWS, [-110, 28, 28, 28, 28, 46])
    assert PROJECT_NPV == pytest.approx(7.3186133585)
    assert PROJECT_IRR == pytest.approx(0.1239131295)

    figure = build_project_cash_flow_npv_profile_figure()

    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 2
        cash_axis, npv_axis = figure.axes
        assert [patch.get_height() for patch in cash_axis.patches] == pytest.approx(
            [-110, 28, 28, 28, 28, 28, 18]
        )
        assert {patch.get_hatch() for patch in cash_axis.patches} == {"//", "..", "xx"}
        assert cash_axis.get_ylabel() == "Incremental cash flow (USD millions)"
        profile = next(line for line in npv_axis.lines if line.get_label() == "NPV profile")
        assert np.interp(10.0, profile.get_xdata(), profile.get_ydata()) == pytest.approx(
            PROJECT_NPV
        )
        annotations = {text.get_text() for text in cash_axis.texts + npv_axis.texts}
        assert any("4.00 years = end-of-year cash flows" in text for text in annotations)
        assert any("3.93 years = uniform-within-year approximation" in text for text in annotations)
        assert any("IRR = 12.39%" in text for text in annotations)
        assert npv_axis.get_xlabel() == "Effective annual USD required return (%)"
        assert any(
            "Cash flows (USD m)" in text.get_text().replace("\n", " ") for text in figure.texts
        )
        np.testing.assert_allclose(figure.get_size_inches(), [10.0, 5.625])
    finally:
        plt.close(figure)


def test_fcff_waterfall_reconciles_revenue_nopat_and_reinvestment() -> None:
    values = FORECAST_FCFF_ANATOMY
    assert values.gross_profit == pytest.approx(18.648)
    assert values.ebitda == pytest.approx(8.648)
    assert values.ebit == pytest.approx(6.648)
    assert values.cash_tax == pytest.approx(1.662)
    assert values.nopat == pytest.approx(4.986)
    assert values.fcff == pytest.approx(3.886)

    figure = build_fcff_forecast_waterfall_figure()

    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 2
        profit_axis, cash_axis = figure.axes
        assert len(profit_axis.patches) == 9
        assert len(cash_axis.patches) == 5
        assert [patch.get_height() for patch in cash_axis.patches] == pytest.approx(
            [4.986, 2.000, 2.500, 0.600, 3.886]
        )
        assert {patch.get_hatch() for patch in cash_axis.patches} == {"..", "xx", "//"}
        assert {text.get_text() for text in cash_axis.texts} >= {
            "4.986",
            "+2.000",
            "−2.500",
            "−0.600",
            "3.886",
        }
        assert profit_axis.get_ylabel() == "USD millions (annual simulation)"
        assert any("Revenue 26.640" in text.get_text() for text in figure.texts)
        np.testing.assert_allclose(figure.get_size_inches(), [10.0, 5.625])
    finally:
        plt.close(figure)


def test_financial_calculators_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        calculate_capital_structure_sensitivity(
            [0.20, 0.40],
            [0.06],
            risk_free_rate=0.04,
            equity_risk_premium=0.055,
            marginal_tax_rate=0.25,
            unlevered_beta=0.80,
        )
    with pytest.raises(ValueError, match="interval"):
        calculate_capital_structure_sensitivity(
            [0.20, 1.00],
            [0.06, 0.08],
            risk_free_rate=0.04,
            equity_risk_premium=0.055,
            marginal_tax_rate=0.25,
            unlevered_beta=0.80,
        )
    with pytest.raises(ValueError, match="greater than perpetual growth"):
        calculate_dcf_value_per_share(
            0.03,
            0.03,
            fcff=[80, 88, 95],
            debt=400,
            nci=20,
            excess_cash=80,
            diluted_shares=100,
        )
    with pytest.raises(ValueError, match="diluted_shares must be positive"):
        calculate_dcf_value_per_share(
            0.09,
            0.03,
            fcff=[80, 88, 95],
            debt=400,
            nci=20,
            excess_cash=80,
            diluted_shares=0,
        )
    with pytest.raises(ValueError, match="greater than -1"):
        calculate_project_npv([-110, 120], -1.0)
    with pytest.raises(ValueError, match="exactly one sign change"):
        calculate_project_irr([-100, 230, -132])
    with pytest.raises(ValueError, match="non-negative"):
        calculate_fcff_anatomy(
            revenue=26.640,
            service_variable_cost=-1.0,
            fixed_cash_operating_expense=10.0,
            depreciation_amortization=2.0,
            cash_tax_rate=0.25,
            capital_expenditures=2.5,
            change_in_operating_working_capital=0.6,
        )


def test_fcff_anatomy_requires_evidence_before_recognizing_tax_loss_benefit() -> None:
    inputs = {
        "revenue": 8.0,
        "service_variable_cost": 4.0,
        "fixed_cash_operating_expense": 5.0,
        "depreciation_amortization": 1.0,
        "cash_tax_rate": 0.25,
        "capital_expenditures": 1.0,
        "change_in_operating_working_capital": 0.0,
    }

    conservative = calculate_fcff_anatomy(**inputs)
    usable_loss = calculate_fcff_anatomy(
        **inputs,
        recognize_immediate_tax_loss_benefit=True,
    )

    assert conservative.ebit == pytest.approx(-2.0)
    assert conservative.cash_tax == pytest.approx(0.0)
    assert conservative.nopat == pytest.approx(-2.0)
    assert usable_loss.cash_tax == pytest.approx(-0.5)
    assert usable_loss.nopat == pytest.approx(-1.5)


def test_fcff_anatomy_accepts_a_working_capital_release_with_explicit_sign() -> None:
    result = calculate_fcff_anatomy(
        revenue=10.0,
        service_variable_cost=3.0,
        fixed_cash_operating_expense=2.0,
        depreciation_amortization=1.0,
        cash_tax_rate=0.25,
        capital_expenditures=1.5,
        change_in_operating_working_capital=-2.0,
    )

    assert result.nopat == pytest.approx(3.0)
    assert result.fcff == pytest.approx(4.5)
    assert result.change_in_operating_working_capital == pytest.approx(-2.0)


@pytest.mark.parametrize("asset_id", EXPORTED_ASSETS)
def test_module5_exported_pngs_are_exact_size_and_rgb(asset_id: str) -> None:
    path = PROJECT_ROOT / "img" / "generated" / f"{asset_id}.png"

    with Image.open(path) as image:
        assert image.size == (1600, 900)
        assert image.mode == "RGB"


@pytest.mark.parametrize(("asset_id", "builder"), ASSET_BUILDERS.items())
def test_module5_exported_pixels_match_current_builder(
    asset_id: str,
    builder: Callable[[], Figure],
    tmp_path: Path,
) -> None:
    figure = builder()
    rendered_path = tmp_path / f"{asset_id}.png"
    if asset_id == "m5-corporate-value-creation-valuation-map":
        save_rgb(figure, rendered_path)
    else:
        try:
            figure.savefig(
                rendered_path,
                dpi=EXPORT_FIGURE_DPI,
                facecolor=BACKGROUND,
                edgecolor="none",
                bbox_inches=None,
                pad_inches=0,
            )
        finally:
            plt.close(figure)

    exported_path = PROJECT_ROOT / "img" / "generated" / f"{asset_id}.png"
    with Image.open(rendered_path) as rendered, Image.open(exported_path) as exported:
        assert rendered.size == exported.size
        np.testing.assert_array_equal(
            np.asarray(rendered.convert("RGB")),
            np.asarray(exported.convert("RGB")),
        )
