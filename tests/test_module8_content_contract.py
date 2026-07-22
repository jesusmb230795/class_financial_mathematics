from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from PIL import Image

from src.module8_visuals import (
    LIQUIDITY_CASE,
    LP_WATERFALL_CASE,
    build_liquidity_budget_figure,
    build_lp_waterfall_figure,
    calculate_liquidity_budget,
    calculate_lp_waterfall,
)
from src.visual_style import BACKGROUND, EXPORT_FIGURE_DPI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COURSE_DIR = PROJECT_ROOT / "notebooks" / "course"
MODULE8_SOURCES = [
    PROJECT_ROOT / "chapters" / "08-alternative-investments.md",
    COURSE_DIR / "8.1.private_capital_structures_and_economics.md",
    COURSE_DIR / "8.2.private_company_valuation_and_due_diligence.md",
    COURSE_DIR / "8.3.real_assets_and_commodities.md",
    COURSE_DIR / "8.4.hedge_funds_digital_assets_and_liquidity.md",
    COURSE_DIR / "8.5.alternatives_portfolio_role_and_mexico_case.md",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _citation_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for group in re.findall(r"\{cite\}`([^`]+)`", text):
        keys.update(key.strip() for key in group.split(",") if key.strip())
    return keys


def test_module8_lessons_keep_the_published_learning_sequence() -> None:
    for source in MODULE8_SOURCES:
        text = _read(source)
        assert "## Learning objectives" in text
        assert "## Prerequisites" in text
        assert "## Sources and further reading" in text
        assert "## Handoff" in text

    for lesson in MODULE8_SOURCES[1:]:
        text = _read(lesson)
        assert "## Checks and limitations" in text
        assert "## Practice" in text


def test_module8_citations_are_dense_traceable_and_resolve() -> None:
    cited_by_source = {source: _citation_keys(_read(source)) for source in MODULE8_SOURCES}
    all_cited = set().union(*cited_by_source.values())
    bibliography = _read(PROJECT_ROOT / "references.bib")
    defined = set(re.findall(r"@\w+\{([^,]+),", bibliography))

    assert all(len(keys) >= 3 for keys in cited_by_source.values())
    assert len(all_cited) >= 15
    assert all_cited <= defined


def test_module8_core_notation_and_equations_are_explicit() -> None:
    overview = _read(MODULE8_SOURCES[0])
    private_capital = _read(MODULE8_SOURCES[1])
    valuation = _read(MODULE8_SOURCES[2])
    real_assets = _read(MODULE8_SOURCES[3])
    hedge_funds = _read(MODULE8_SOURCES[4])
    portfolio = _read(MODULE8_SOURCES[5])

    assert r"\tau_i=\frac{d_i-d_0}{365}" in overview
    assert r"P(0,t)=\prod_{j=1}^{t}\frac{1}{1+k_j}" in overview
    assert r"H_{\mathrm{compound}}=C\left[(1+h)^\tau-1\right]" in private_capital
    assert r"Carry_{GP}=cR" in private_capital
    assert r"\prod_{j=1}^{t}(1+WACC_j)" in valuation
    assert r"\sum_{s=1}^{S}p_s=1" in valuation
    assert r"\frac{\partial V_{\text{property}}}{\partial k}" in real_assets
    assert r"\Delta_F^{\mathrm{norm}}=\frac{S_T-F_0}{F_0}" in real_assets
    assert r"E_g=\frac{\sum_i L_i+\sum_j S_j}{NAV}" in hedge_funds
    assert r"R_{net}=\frac{V_1}{V_0}-1" in hedge_funds
    assert r"LCR=\frac{L-U}{N+B}" in portfolio
    assert r"H=(L-U)-q(N+B)" in portfolio
    assert r"D(0,t)=\prod_{j=1}^{t}\frac{1}{1+WACC_j}" in valuation
    assert r"E=EV-D-O+C+A_{NO}" in valuation
    assert r"L_1=L_0-\sum_i(1-\lambda_i)A_i" in portfolio
    assert "effective one-period weighted-average cost of capital" in valuation
    assert "mutually exclusive" in valuation
    assert "zero stressed one-year liquidity credit" in portfolio
    assert "neither a return on posted margin nor a fully" in real_assets
    assert "N=80+60" not in portfolio


def test_lp_waterfall_reconciles_and_handles_insufficient_proceeds() -> None:
    case = LP_WATERFALL_CASE
    assert case.preferred_return == pytest.approx(2.88)
    assert case.lp_residual_profit == pytest.approx(4.096)
    assert case.gp_carry == pytest.approx(1.024)
    assert case.lp_total == pytest.approx(18.976)
    assert case.lp_total + case.gp_carry == pytest.approx(case.total_proceeds)

    loss_case = calculate_lp_waterfall(
        contributed_capital=12,
        total_proceeds=9,
        preferred_rate=0.08,
        years=3,
        carry_rate=0.20,
        compound_preference=False,
    )
    assert loss_case.return_of_capital == pytest.approx(9)
    assert loss_case.preferred_return == pytest.approx(0)
    assert loss_case.gp_carry == pytest.approx(0)
    assert loss_case.lp_total == pytest.approx(9)

    compound_case = calculate_lp_waterfall(
        contributed_capital=12,
        total_proceeds=20,
        preferred_rate=0.08,
        years=3,
        carry_rate=0.20,
        compound_preference=True,
    )
    assert compound_case.preferred_return == pytest.approx(12 * ((1.08**3) - 1))
    assert compound_case.lp_total + compound_case.gp_carry == pytest.approx(20)

    with pytest.raises(ValueError, match="carry_rate"):
        calculate_lp_waterfall(
            contributed_capital=12,
            total_proceeds=20,
            preferred_rate=0.08,
            years=3,
            carry_rate=1.01,
            compound_preference=False,
        )


def test_liquidity_budget_counts_commitments_and_buffer_once() -> None:
    case = LIQUIDITY_CASE
    assert case.proposed_allocations == pytest.approx(130)
    assert case.post_allocation_liquid_assets == pytest.approx(720)
    assert case.available_after_commitments == pytest.approx(670)
    assert case.denominator_need == pytest.approx(140)
    assert case.required_liquid_assets == pytest.approx(420)
    assert case.liquidity_headroom == pytest.approx(250)
    assert case.lcr == pytest.approx(670 / 140)

    with pytest.raises(ValueError, match="must not exceed"):
        calculate_liquidity_budget(
            opening_liquid_assets=100,
            proposed_allocations=(90,),
            unfunded_commitments=20,
            planned_cash_need=10,
            policy_buffer=5,
            minimum_lcr=2,
        )


def test_module8_figures_preserve_the_financial_contracts() -> None:
    waterfall = build_lp_waterfall_figure()
    liquidity = build_liquidity_budget_figure()
    try:
        waterfall_axis = waterfall.axes[0]
        assert [patch.get_width() for patch in waterfall_axis.patches] == pytest.approx(
            [12.0, 2.88, 4.096, 1.024]
        )
        assert {patch.get_hatch() for patch in waterfall_axis.patches} == {
            "//",
            "..",
            "xx",
            "\\\\",
        }
        assert any("simulated teaching assumptions" in text.get_text() for text in waterfall.texts)

        liquidity_axis = liquidity.axes[0]
        assert [patch.get_width() for patch in liquidity_axis.patches] == pytest.approx(
            [850, 720, 670, 420]
        )
        assert any("LCR 4.79" in text.get_text() for text in liquidity.texts)
        assert any("Policy headroom = MXN 250m" in text.get_text() for text in liquidity_axis.texts)
        assert tuple(waterfall.get_size_inches()) == pytest.approx((10.0, 5.625))
        assert tuple(liquidity.get_size_inches()) == pytest.approx((10.0, 5.625))
    finally:
        plt.close(waterfall)
        plt.close(liquidity)


def test_module8_visual_assets_are_registered_referenced_and_present() -> None:
    manifest = json.loads(
        _read(PROJECT_ROOT / "img" / "generated" / "visual-assets" / "module-08.json")
    )
    assets = {asset["id"]: asset for asset in manifest["assets"]}
    assert set(assets) == {
        "m8-alternatives-asset-vehicle-strategy-map",
        "m8-lp-waterfall-economics",
        "m8-liquidity-budget-coverage",
    }

    for asset in assets.values():
        target = PROJECT_ROOT / asset["target_path"]
        source = PROJECT_ROOT / asset["used_by"][0]["file"]
        assert asset["status"] == "approved"
        assert target.is_file()
        assert asset["used_by"][0]["target_markdown"] in _read(source)
        assert asset["alt_text"] in _read(source)
        with Image.open(target) as image:
            expected = tuple(int(value) for value in asset["recommended_size"].split("x"))
            assert image.size == expected
            assert image.mode == "RGB"


@pytest.mark.parametrize(
    ("asset_id", "builder"),
    (
        ("m8-lp-waterfall-economics", build_lp_waterfall_figure),
        ("m8-liquidity-budget-coverage", build_liquidity_budget_figure),
    ),
)
def test_module8_exported_pixels_match_current_builder(
    asset_id: str,
    builder: Callable[[], Figure],
    tmp_path: Path,
) -> None:
    figure = builder()
    rendered_path = tmp_path / f"{asset_id}.png"
    try:
        figure.savefig(
            rendered_path,
            dpi=EXPORT_FIGURE_DPI,
            facecolor=BACKGROUND,
            bbox_inches=None,
            pad_inches=0,
        )
    finally:
        plt.close(figure)

    exported_path = PROJECT_ROOT / "img" / "generated" / f"{asset_id}.png"
    with Image.open(rendered_path) as rendered, Image.open(exported_path) as exported:
        assert rendered.size == exported.size == (1600, 900)
        np.testing.assert_array_equal(
            np.asarray(rendered.convert("RGB")),
            np.asarray(exported.convert("RGB")),
        )
