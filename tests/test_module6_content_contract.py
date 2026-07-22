from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COURSE_DIR = PROJECT_ROOT / "notebooks" / "course"
MODULE6_SOURCES = sorted(COURSE_DIR.glob("6.*.py"))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _citation_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for group in re.findall(r"\{cite\}`([^`]+)`", text):
        keys.update(key.strip() for key in group.split(",") if key.strip())
    return keys


def test_module6_has_ten_canonical_lessons_with_transfer_sections() -> None:
    assert len(MODULE6_SOURCES) == 10

    for source in MODULE6_SOURCES:
        text = _read(source)
        assert "Module: Fixed Income, Credit, and Term Structure" in text
        assert "## Learning objectives" in text
        assert "## Prerequisites" in text
        assert "## Handoff" in text
        assert "## Model limitations" in text or "## Limitations" in text


def test_module6_lessons_have_multiple_traceable_sources() -> None:
    for source in MODULE6_SOURCES:
        citation_keys = _citation_keys(_read(source))
        assert len(citation_keys) >= 2, (
            f"{source.name} must cite at least two distinct methodological or official sources"
        )


def test_module6_programmatic_figures_have_semantic_alt_text() -> None:
    figure_sources = {
        "6.2.interactive_bond_sensitivity.py": 1,
        "6.5.credit_spreads_securitized_products.py": 1,
        "6.6.yield_curve_bootstrapping.py": 1,
        "6.7.short_rate_models.py": 1,
        "6.8.nelson_siegel_curve_fitting.py": 1,
        "6.9.rate_panel_pca_and_scenarios.py": 2,
        "6.10.short_rate_calibration_lab.py": 2,
    }
    for filename, expected_count in figure_sources.items():
        source = _read(COURSE_DIR / filename)
        assert source.count('mystnb={"image": {"alt":') >= expected_count


def test_module6_foundational_equations_match_the_implemented_contracts() -> None:
    pricing = _read(COURSE_DIR / "6.1.bond_pricing_duration_convexity.py")
    immunization = _read(COURSE_DIR / "6.4.ytm_dv01_and_immunization_lab.py")
    credit = _read(COURSE_DIR / "6.5.credit_spreads_securitized_products.py")
    bootstrap = _read(COURSE_DIR / "6.6.yield_curve_bootstrapping.py")
    short_rate = _read(COURSE_DIR / "6.7.short_rate_models.py")
    nelson_siegel = _read(COURSE_DIR / "6.8.nelson_siegel_curve_fitting.py")

    assert r"\frac{FC/m}" in pricing
    assert "coupon_bond_cash_flows" in pricing
    assert "positive price-change magnitude" in immunization
    assert r"\Delta y=+1\text{ bp}\Longrightarrow\Delta P\approx-\operatorname{DV01}" in immunization
    assert r"L_j(L)=\min" in credit
    assert "tranche_loss(" in credit
    assert "nominal annual, convertible" in credit
    assert "same valuation and settlement date" in credit
    assert "par_rate_annual" in bootstrap
    assert "price_error" in bootstrap
    assert r"\mathbb E_t^{\mathbb Q}" in short_rate
    assert "simulate_vasicek_exact(" in short_rate
    assert "synthetic_zero_rate_annual" in nelson_siegel
    assert r"\arg\min" in nelson_siegel
    assert r"y_{\mathrm{NSS}}" in nelson_siegel


def test_module6_empirical_clock_and_pca_transformation_are_explicit() -> None:
    pca_lesson = _read(COURSE_DIR / "6.9.rate_panel_pca_and_scenarios.py")
    calibration_lesson = _read(COURSE_DIR / "6.10.short_rate_calibration_lab.py")

    for lesson in (pca_lesson, calibration_lesson):
        assert 'frequency="weekly"' in lesson
        assert "empty week" in lesson
        assert "source_vintage" in lesson

    assert r"S^{-1}" in pca_lesson
    assert r"\mu_{\Delta r}" in pca_lesson
    assert "standardized weekly decimal-rate" in pca_lesson
    assert "rights have not been independently verified" in pca_lesson
    assert "rights have not been independently verified" in calibration_lesson
    assert "WEEKLY_DT = 1 / 52" in calibration_lesson
    assert "1 / 252" not in calibration_lesson
    assert "CIR parameters are hypothetical" in calibration_lesson
    assert "ACT/360 return-yield" in pca_lesson
    assert "ACT/360 return-yield" in calibration_lesson
    assert "does not make" in pca_lesson
    assert "does not make" in calibration_lesson

    sensitivity = _read(COURSE_DIR / "6.2.interactive_bond_sensitivity.py")
    assert "coupon_bond_cash_flows" in sensitivity


def test_module6_analytical_figures_use_shared_builders() -> None:
    for source in MODULE6_SOURCES:
        if source.name.startswith(("6.5.", "6.6.", "6.7.", "6.8.", "6.9.", "6.10.")):
            assert "import matplotlib.pyplot as plt" not in _read(source)


def test_module6_overview_separates_published_core_from_advanced_targets() -> None:
    overview = _read(PROJECT_ROOT / "chapters" / "06-fixed-income-credit-term-structure.md")
    roadmap = _read(PROJECT_ROOT / "chapters" / "course-roadmap.md")
    plan = _read(PROJECT_ROOT / "content" / "plan.json")

    assert "Quantitative Methods and Financial Time Series" in overview
    assert "Redistribution rights" in overview
    assert "Published core coverage:" in roadmap
    assert "Advanced expansion targets:" in roadmap
    assert "bond pricing, annuities" not in roadmap
    assert "homogeneous-tenor curve PCA" in plan
    assert "owner-authorized review" in plan
