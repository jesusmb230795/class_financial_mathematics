from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COURSE_DIR = PROJECT_ROOT / "notebooks" / "course"
MODULE3_LESSONS = [
    COURSE_DIR / "3.1.economic_foundations.md",
    COURSE_DIR / "3.2.macro_indicators_policy.md",
    COURSE_DIR / "3.3.currency_parity_fx.md",
    COURSE_DIR / "3.4.macro_scenarios_cme.md",
    COURSE_DIR / "3.5.applied_macro_fx_case.py",
]
MODULE3_OVERVIEW = PROJECT_ROOT / "chapters/03-economics-macro-currency.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_module3_lessons_preserve_the_published_teaching_sequence() -> None:
    for lesson in MODULE3_LESSONS:
        text = _read(lesson)
        assert "## Learning objectives" in text
        assert "## Prerequisites" in text
        assert "## Handoff" in text

    overview = _read(MODULE3_OVERVIEW)
    normalized_overview = " ".join(overview.split())
    assert "Each lesson ends with an assessment" not in overview
    assert (
        "numerical output, a mechanism statement, and a limitation"
        in normalized_overview
    )


def test_module3_inline_math_uses_the_enabled_myst_delimiter() -> None:
    for source in [MODULE3_OVERVIEW, *MODULE3_LESSONS]:
        text = _read(source)
        assert r"\(" not in text
        assert r"\)" not in text

    case = _read(COURSE_DIR / "3.5.applied_macro_fx_case.py")
    assert r"$S_{\mathrm{MXN/USD}}$" in case


def test_currency_lesson_uses_one_coherent_quote_and_rate_example() -> None:
    currency_lesson = _read(COURSE_DIR / "3.3.currency_parity_fx.md")

    assert "MXN per USD" in currency_lesson
    assert r"S_t=17.20" in currency_lesson
    assert r"i_{\mathrm{MXN}}=9.00\%" in currency_lesson
    assert r"i_{\mathrm{USD}}=5.00\%" in currency_lesson
    assert r"17.20\frac{1.09}{1.05}" in currency_lesson
    assert "17.8552" in currency_lesson
    assert "18.40" not in currency_lesson


def test_macro_case_distinguishes_source_vintage_from_derived_rebuild() -> None:
    overview = _read(MODULE3_OVERVIEW)
    case = _read(COURSE_DIR / "3.5.applied_macro_fx_case.py")

    assert "retrieved on 2026-06-07" in overview
    assert "rebuilt from those versioned sources on 2026-07-21" in overview
    assert 'metadata["generated_at"]' in case
    assert 'metadata["derived_generated_at"]' in case
    assert "Source retrieval vintage (UTC)" in case
    assert "Derived panel rebuild (UTC)" in case
    assert "snapshot generated" not in case


def test_module3_lesson_figures_are_registered_and_referenced() -> None:
    manifest = json.loads(
        (PROJECT_ROOT / "img/generated/visual-assets/module-03.json").read_text(encoding="utf-8")
    )
    assets = {asset["id"]: asset for asset in manifest["assets"]}
    expected = {
        "m3-supply-demand-identification": COURSE_DIR / "3.1.economic_foundations.md",
        "m3-fx-carry-break-even": COURSE_DIR / "3.3.currency_parity_fx.md",
        "m3-scenario-valuation-bridge": COURSE_DIR / "3.4.macro_scenarios_cme.md",
    }

    for asset_id, lesson in expected.items():
        asset = assets[asset_id]
        assert asset["status"] == "approved"
        assert (PROJECT_ROOT / asset["target_path"]).is_file()
        target_markdown = asset["used_by"][0]["target_markdown"]
        assert target_markdown in _read(lesson)
