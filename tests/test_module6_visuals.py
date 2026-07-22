from __future__ import annotations

from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure

from src.module6_visuals import (
    build_bootstrap_curve_figure,
    build_nelson_siegel_diagnostic_figure,
    build_rate_history_figure,
    build_rate_panel_diagnostics_figure,
    build_securitization_waterfall_figure,
    build_short_rate_paths_figure,
)


SOURCE_NOTE = (
    "Source: deterministic test fixture. Synthetic classroom inputs; annual decimal rates; "
    "offline mode."
)


def _render_png(figure: Figure) -> bytes:
    output = BytesIO()
    figure.savefig(output, format="png")
    return output.getvalue()


def _assert_note_clears_axes(figure: Figure) -> None:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    note = min(figure.texts, key=lambda text: text.get_position()[1])
    note_top = note.get_window_extent(renderer).y1
    axes_bottom = min(axis.get_window_extent(renderer).y0 for axis in figure.axes)
    lower_labels = [
        artist
        for axis in figure.axes
        for artist in (axis.xaxis.label, *axis.get_xticklabels())
        if artist.get_visible() and artist.get_text()
    ]
    label_bottom = min(artist.get_window_extent(renderer).y0 for artist in lower_labels)
    assert note_top + 4 <= axes_bottom
    assert note_top + 4 <= label_bottom


def _long_waterfall_fixture() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    specifications = {
        "base": (3.0, {"equity": 3.0, "mezzanine": 0.0, "senior": 0.0}),
        "severe": (9.0, {"equity": 4.0, "mezzanine": 5.0, "senior": 0.0}),
        "extreme": (15.0, {"equity": 4.0, "mezzanine": 6.0, "senior": 5.0}),
    }
    tranche_points = {
        "equity": (0.0, 4.0),
        "mezzanine": (4.0, 10.0),
        "senior": (10.0, 100.0),
    }
    for scenario, (collateral_loss, losses) in specifications.items():
        for tranche, tranche_loss in losses.items():
            attachment, detachment = tranche_points[tranche]
            records.append(
                {
                    "scenario": scenario,
                    "default_rate": collateral_loss / 100,
                    "recovery_rate": 0.0,
                    "collateral_loss": collateral_loss,
                    "tranche": tranche,
                    "attachment": attachment,
                    "detachment": detachment,
                    "tranche_loss": tranche_loss,
                    "pool_notional": 100.0,
                }
            )
    return pd.DataFrame(records)


def test_waterfall_figure_accepts_long_allocations_and_conserves_loss() -> None:
    allocation = _long_waterfall_fixture()
    original = allocation.copy(deep=True)

    figure = build_securitization_waterfall_figure(allocation, 100.0, SOURCE_NOTE)

    try:
        assert isinstance(figure, Figure)
        axis = figure.axes[0]
        assert axis.get_ylabel() == "Allocated tranche loss (% of pool notional)"
        assert "Scenario and collateral loss" in axis.get_xlabel()
        assert len(axis.patches) == 9
        assert {patch.get_hatch() for patch in axis.patches} == {"///", "\\\\", "..."}
        assert [tick.get_text().splitlines()[-1] for tick in axis.get_xticklabels()] == [
            "3.0% loss",
            "9.0% loss",
            "15.0% loss",
        ]
        assert any("Loss conservation" in text.get_text() for text in figure.texts)
        assert _render_png(figure).startswith(b"\x89PNG\r\n\x1a\n")
        pd.testing.assert_frame_equal(allocation, original)
    finally:
        plt.close(figure)


def test_waterfall_figure_rejects_unallocated_collateral_loss() -> None:
    allocation = _long_waterfall_fixture()
    allocation.loc[
        (allocation["scenario"] == "severe") & (allocation["tranche"] == "mezzanine"),
        "tranche_loss",
    ] = 4.5

    with pytest.raises(ValueError, match="reconcile"):
        build_securitization_waterfall_figure(allocation, 100.0, SOURCE_NOTE)


def test_bootstrap_figure_labels_annual_rates_and_preserves_first_missing_forward() -> None:
    curve = pd.DataFrame(
        {
            "maturity": [0.5, 1.0, 1.5, 2.0],
            "discount_factor": [0.97, 0.94, 0.91, 0.88],
            "spot_rate_annual": [0.062, 0.064, 0.066, 0.068],
            "forward_rate_annual": [np.nan, 0.066, 0.070, 0.074],
            "par_rate_annual": [0.062, 0.063, 0.065, 0.067],
        }
    )

    figure = build_bootstrap_curve_figure(curve, SOURCE_NOTE)

    try:
        axis = figure.axes[0]
        assert axis.get_xlabel() == "Maturity (years)"
        assert axis.get_ylabel() == "Rate or nominal coupon (%)"
        np.testing.assert_allclose(axis.lines[0].get_ydata(), [6.2, 6.4, 6.6, 6.8])
        assert np.isnan(axis.lines[1].get_ydata()[0])
        np.testing.assert_allclose(axis.lines[2].get_ydata(), [6.2, 6.3, 6.5, 6.7])
        assert axis.lines[0].get_marker() == "o"
        assert axis.lines[1].get_linestyle() == "--"
        assert axis.lines[2].get_marker() == "^"
        assert axis.get_title(loc="left") == (
            "How do effective spot/forward rates and nominal par coupons differ by maturity?"
        )
        assert figure._suptitle.get_text() == (
            "Bootstrapped effective rates and nominal par coupons"
        )
        assert axis.get_legend_handles_labels()[1] == [
            "Effective annual spot rate",
            "Effective annual one-period forward rate",
            "Nominal annual par coupon (semiannual)",
        ]
        figure_note = " ".join(" ".join(text.get_text().split()) for text in figure.texts)
        assert "Spot and forward inputs are effective annual rates" in figure_note
        assert "nominal annual coupon convertible semiannually" in figure_note
        assert "annual-compounding inputs" not in figure_note
    finally:
        plt.close(figure)


def test_bootstrap_figure_rejects_an_internal_forward_gap() -> None:
    curve = pd.DataFrame(
        {
            "maturity": [0.5, 1.0, 1.5],
            "spot_rate_annual": [0.06, 0.065, 0.07],
            "forward_rate_annual": [np.nan, 0.07, np.nan],
        }
    )

    with pytest.raises(ValueError, match="only the first forward"):
        build_bootstrap_curve_figure(curve, SOURCE_NOTE)


def test_short_rate_figure_shows_path_quantiles_and_long_run_means() -> None:
    index = pd.Index([0.0, 0.5, 1.0, 1.5], name="years")
    vasicek = pd.DataFrame(
        [
            [0.08, 0.08, 0.08, 0.08],
            [0.07, 0.08, 0.09, 0.10],
            [0.06, 0.07, 0.08, 0.09],
            [0.05, 0.06, 0.07, 0.08],
        ],
        index=index,
    )
    cir = vasicek.clip(lower=0).add(0.005)

    figure = build_short_rate_paths_figure(vasicek, cir, 0.06, 0.065, SOURCE_NOTE)

    try:
        assert len(figure.axes) == 2
        assert figure.axes[0].get_ylabel() == "Annual short rate (%)"
        np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), [8.0, 8.5, 7.5, 6.5])
        assert figure.axes[0].lines[1].get_ydata()[0] == pytest.approx(6.0)
        assert figure.axes[1].lines[1].get_ydata()[0] == pytest.approx(6.5)
        assert len(figure.axes[0].collections) >= 1
        assert any("not confidence intervals" in text.get_text() for text in figure.texts)
        assert _render_png(figure).startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        plt.close(figure)


def test_nelson_siegel_figure_accepts_module6_column_names_and_computes_residuals() -> None:
    curve = pd.DataFrame(
        {
            "maturity_years": [0.25, 1.0, 5.0],
            "synthetic_zero_rate_annual": [0.1010, 0.0960, 0.0860],
            "fitted_zero_rate_annual": [0.1008, 0.0963, 0.0859],
            "residual_bp": [2.0, -3.0, 1.0],
        }
    )
    dense_maturities = np.linspace(0.25, 5.0, 40)
    fitted_yields = np.interp(
        dense_maturities,
        curve["maturity_years"],
        curve["fitted_zero_rate_annual"],
    )

    figure = build_nelson_siegel_diagnostic_figure(
        curve,
        dense_maturities,
        fitted_yields,
        SOURCE_NOTE,
    )

    try:
        fit_axis, residual_axis = figure.axes
        assert fit_axis.get_ylabel() == "Annual zero-coupon rate (%)"
        assert fit_axis.get_legend_handles_labels()[1][-1] == "Synthetic zero-coupon inputs"
        assert residual_axis.get_title(loc="left") == "Synthetic input minus fitted rate"
        assert residual_axis.get_ylabel() == "Fit residual (basis points)"
        np.testing.assert_allclose(
            [patch.get_height() for patch in residual_axis.patches],
            [2.0, -3.0, 1.0],
        )
        assert [patch.get_hatch() for patch in residual_axis.patches] == [
            "///",
            "\\\\",
            "///",
        ]
        figure_note = " ".join(" ".join(text.get_text().split()) for text in figure.texts)
        assert "input minus fitted rate" in figure_note
    finally:
        plt.close(figure)


def test_rate_history_figure_uses_percent_units_human_labels_and_actual_sample() -> None:
    index = pd.date_range("2024-01-05", periods=5, freq="W-FRI")
    history = pd.DataFrame(
        {
            "policy_rate": [0.11, 0.11, 0.11, 0.1075, 0.1075],
            "cetes_28d": [0.109, 0.108, 0.107, 0.106, 0.105],
            "tiie_28d": [0.112, 0.111, 0.110, 0.109, 0.108],
        },
        index=index,
    )
    history.attrs["methodology_break"] = {
        "effective_date": "2024-01-19",
        "series_id": "SF60648",
    }

    figure = build_rate_history_figure(history, SOURCE_NOTE)

    try:
        axis = figure.axes[0]
        assert axis.get_xlabel() == "Aligned observation date"
        assert axis.get_ylabel() == "Annual rate (%)"
        assert [line.get_label() for line in axis.lines[:3]] == [
            "Banxico target rate",
            "CETES 28-day rate",
            "TIIE 28-day rate",
        ]
        assert axis.lines[3].get_label() == (
            "TIIE methodology change (2024-01-19; SF60648)"
        )
        assert axis.lines[3].get_linestyle() == ":"
        np.testing.assert_allclose(axis.lines[0].get_ydata(), np.asarray(history.iloc[:, 0]) * 100)
        note = " ".join(text.get_text() for text in figure.texts)
        assert "2024-01-05 to 2024-02-02" in note
        assert "lines, markers, and labels duplicate color" in note
        assert "labeled dotted line" in note
    finally:
        plt.close(figure)


def test_rate_panel_diagnostics_plot_scenario_changes_as_unconnected_basis_point_dots() -> None:
    components = pd.DataFrame(
        {
            "policy_rate": [0.65, -0.72, 0.15],
            "cetes_28d": [0.70, 0.05, -0.70],
            "tiie_28d": [0.63, 0.69, 0.70],
        },
        index=["PC1", "PC2", "PC3"],
    )
    base = pd.Series({"policy_rate": 0.10, "cetes_28d": 0.095, "tiie_28d": 0.105})
    scenarios = pd.DataFrame(
        {
            "base": base,
            "common_up_100bp": base + 0.0100,
            "policy_led_tightening": base + pd.Series(
                {"policy_rate": 0.0100, "cetes_28d": 0.0060, "tiie_28d": 0.0040}
            ),
            "interbank_spread_widening": base + pd.Series(
                {"policy_rate": 0.0, "cetes_28d": 0.0, "tiie_28d": 0.0075}
            ),
        }
    )

    figure = build_rate_panel_diagnostics_figure(components, scenarios, SOURCE_NOTE)

    try:
        loading_axis, scenario_axis = figure.axes
        assert loading_axis.get_ylabel() == "Standardized loading"
        assert scenario_axis.get_ylabel() == "Rate shock (basis points)"
        assert len(scenario_axis.lines) == 1  # zero reference only; no categorical connections
        scatter_collections = [
            collection
            for collection in scenario_axis.collections
            if isinstance(collection, PathCollection)
        ]
        assert len(scatter_collections) == 3
        np.testing.assert_allclose(scatter_collections[0].get_offsets()[:, 1], [100, 100, 100])
        np.testing.assert_allclose(scatter_collections[1].get_offsets()[:, 1], [100, 60, 40])
        np.testing.assert_allclose(scatter_collections[2].get_offsets()[:, 1], [0, 0, 75])
        assert all(patch.get_hatch() for patch in loading_axis.patches)
        assert any("do not connect unlike tenors" in text.get_text() for text in figure.texts)
    finally:
        plt.close(figure)


def test_long_source_notes_do_not_overlap_axes_or_rotated_labels() -> None:
    long_note = (
        "Source: Banco de México official snapshot; three named series; weekly alignment; "
        "actual inclusive sample 2018-01-05 to 2026-06-05; snapshot vintage and retrieval "
        "timestamp recorded; redistribution rights have not been independently verified; "
        "author-created scenarios are deterministic and carry no probability."
    )
    history = pd.DataFrame(
        {"cetes_28d": [0.10, 0.09, 0.08]},
        index=pd.date_range("2024-01-05", periods=3, freq="W-FRI"),
    )
    components = pd.DataFrame(
        {"policy_rate": [0.7], "cetes_28d": [0.6], "tiie_28d": [0.5]},
        index=["PC1"],
    )
    base = pd.Series({"policy_rate": 0.10, "cetes_28d": 0.09, "tiie_28d": 0.11})
    scenarios = pd.DataFrame({"base": base, "common_up_100bp": base + 0.01})
    curve = pd.DataFrame(
        {
            "maturity": [0.5, 1.0],
            "spot_rate_annual": [0.06, 0.065],
            "forward_rate_annual": [np.nan, 0.07],
            "par_rate_annual": [0.06, 0.064],
        }
    )

    figures = (
        build_rate_history_figure(history, long_note),
        build_rate_panel_diagnostics_figure(components, scenarios, long_note),
        build_bootstrap_curve_figure(curve, long_note),
    )
    try:
        for figure in figures:
            _assert_note_clears_axes(figure)
    finally:
        for figure in figures:
            plt.close(figure)


def test_module6_visuals_reject_missing_provenance_and_non_finite_history() -> None:
    curve = pd.DataFrame(
        {
            "maturity": [0.5, 1.0],
            "spot_rate_annual": [0.06, 0.065],
            "forward_rate_annual": [np.nan, 0.07],
        }
    )
    history = pd.DataFrame(
        {"cetes_28d": [0.10, np.nan]},
        index=pd.date_range("2024-01-05", periods=2, freq="W-FRI"),
    )

    with pytest.raises(ValueError, match="source_note"):
        build_bootstrap_curve_figure(curve, " ")
    with pytest.raises(ValueError, match="finite"):
        build_rate_history_figure(history, SOURCE_NOTE)
