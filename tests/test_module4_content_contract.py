from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COURSE_DIR = PROJECT_ROOT / "notebooks" / "course"
MODULE4_SOURCES = [
    PROJECT_ROOT / "chapters" / "04-financial-statements-modeling.md",
    COURSE_DIR / "4.1.analysis_framework_and_statement_linkages.md",
    COURSE_DIR / "4.2.ratios_dupont_and_capital_efficiency.md",
    COURSE_DIR / "4.3.accounting_quality_and_normalization.md",
    COURSE_DIR / "4.4.specialized_entities_and_consolidation.md",
    COURSE_DIR / "4.5.three_statement_model_and_review.md",
]
VISUAL_REFERENCES = {
    "m4-three-statement-evidence-model-map": MODULE4_SOURCES[0],
    "m4-roe-roic-driver-map": MODULE4_SOURCES[2],
    "m4-normalized-ebit-bridge": MODULE4_SOURCES[3],
    "m4-entity-perimeter-map": MODULE4_SOURCES[4],
    "m4-forecast-cash-reconciliation": MODULE4_SOURCES[5],
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_module4_lessons_preserve_objectives_prerequisites_and_handoffs() -> None:
    for lesson in MODULE4_SOURCES:
        text = _read(lesson)
        assert "## Learning objectives" in text
        assert "## Prerequisites" in text
        assert "## Handoff" in text


def test_module4_uses_renderable_inline_math_delimiters() -> None:
    combined = "\n".join(_read(path) for path in MODULE4_SOURCES)

    assert r"\(" not in combined
    assert r"\)" not in combined
    assert combined.count("$") % 2 == 0


def test_module4_forecast_labels_closing_day_proxies_and_cash_scope() -> None:
    overview = _read(MODULE4_SOURCES[0])
    forecast = _read(MODULE4_SOURCES[5])
    normalized_forecast = " ".join(forecast.split())

    assert "closing-balance day proxies" in forecast
    assert r"B_t" in forecast
    assert r"2\left(\text{Flow}_t\times\frac{\text{Days}_t}{365}\right)-B_{t-1}" in forecast
    assert "mechanical closing-balance sensitivity" in normalized_forecast
    assert "exchange-rate effect on cash is zero" in normalized_forecast
    assert "cash scopes are identical" in normalized_forecast
    assert "cash-and-cash-equivalents scope" in overview
    assert "fixed-asset, and debt schedules also feed the cash bridge" in overview
    assert "cash interest paid equals interest expense" in normalized_forecast
    assert "all interest paid is classified in CFO" in normalized_forecast


def test_module4_financial_boundaries_and_notation_are_explicit() -> None:
    overview = _read(MODULE4_SOURCES[0])
    statement = _read(MODULE4_SOURCES[1])
    ratios = _read(MODULE4_SOURCES[2])
    normalization = _read(MODULE4_SOURCES[3])
    forecast = _read(MODULE4_SOURCES[5])

    assert "total equity, not its retained-earnings component" in statement
    assert r"\mathrm{RE}_t" in statement
    assert r"\mathrm{Equity}_t" in statement
    assert r"\Delta\mathrm{Scope\ adjustments}_t" in statement

    assert "Net credit sales" in ratios
    assert "Credit purchases" in ratios
    assert r"\text{Purchases}_t" in ratios
    assert "revenue- and COGS-based proxies" in ratios
    assert "using unrounded components gives 74.3 days" in ratios
    assert r"1.307\times" in ratios

    assert "Cash investment in operating assets" in overview
    assert r"\mathrm{Reported\ NOPAT}" in normalization
    assert "Operating taxes attributable to EBIT" in normalization
    assert r"\mathrm{FCFE}_t" in normalization
    assert "Owner earnings is therefore not automatically FCFF" in normalization

    assert r"\mathrm{EBT}" in forecast
    assert r"\mathrm{Balance\ check}_t" in forecast
    assert r"\mathrm{Cash\ check}_t" in forecast
    assert "−3.0%" in forecast
    assert "**unmitigated**" in forecast
    assert "**mitigated**" in forecast
    assert r"\Delta\mathrm{DSO}^{\mathrm{close}}_t" in forecast


def test_module4_simulations_and_figure_sources_are_disclosed() -> None:
    for source in MODULE4_SOURCES:
        text = _read(source)
        for figure_block in re.findall(r"```\{figure\}.*?```", text, flags=re.DOTALL):
            assert "Source:" in figure_block

    for lesson in MODULE4_SOURCES[1:]:
        normalized = " ".join(_read(lesson).split()).lower()
        assert "simulation" in normalized
        assert "not represent" in normalized or "no real issuer" in normalized


def test_module4_uses_the_correct_standard_owners_and_effective_dates() -> None:
    statement_lesson = _read(MODULE4_SOURCES[1])
    normalized_statement_lesson = " ".join(statement_lesson.split())
    specialized_lesson = _read(MODULE4_SOURCES[4])

    for citation_key in (
        "iaasbISA705Revised2015",
        "iaasbISA706Revised2015",
        "iaasbISA570Revised2015",
        "iaasbISA570Revised2024",
    ):
        assert citation_key in statement_lesson
    assert "periods beginning on or after 15 December 2026" in normalized_statement_lesson
    assert "emphasis-of-matter paragraph does not modify the opinion" in normalized_statement_lesson
    assert (
        "scope limitation may lead to a qualified opinion or disclaimer"
        in normalized_statement_lesson
    )

    assert "ifrs9FinancialInstruments2014" in specialized_lesson
    assert "ifrs11JointArrangements2011" in specialized_lesson
    assert "ifrs3BusinessCombinations2008" in specialized_lesson
    assert "full-goodwill" in specialized_lesson
    assert "proportionate share" in specialized_lesson


def test_module4_worked_forecast_reconciles_numerically() -> None:
    forecast = _read(MODULE4_SOURCES[5])
    revenue = 1_000 * 1.08
    cogs = revenue * (1 - 0.40)
    ebit = revenue * 0.16
    net_income = (ebit - 25) * (1 - 0.25)
    receivables = revenue * 51.1 / 365
    inventory = cogs * 91.25 / 365
    payables = cogs * 63.875 / 365
    opening_owc = 140 + 150 - 105
    closing_owc = receivables + inventory - payables
    delta_owc = closing_owc - opening_owc
    cfo = net_income + 50 - delta_owc
    ending_cash = 60 + cfo - 70 - 20 - 30
    fcff = ebit * (1 - 0.25) + 50 - 70 - delta_owc
    fcff_from_cfo = cfo + 25 * (1 - 0.25) - 70
    closing_assets = ending_cash + receivables + inventory + (440 + 70 - 50)
    closing_equity = 455 + net_income - 30
    closing_liabilities_and_equity = payables + (230 - 20) + closing_equity
    dso_cash_use = revenue / 365 * (60 - 51.1)

    assert revenue == pytest.approx(1_080)
    assert cogs == pytest.approx(648)
    assert ebit == pytest.approx(172.8)
    assert net_income == pytest.approx(110.85)
    assert delta_owc == pytest.approx(14.8)
    assert cfo == pytest.approx(146.05)
    assert ending_cash == pytest.approx(86.05)
    assert fcff == pytest.approx(94.8)
    assert fcff_from_cfo == pytest.approx(fcff)
    assert closing_assets == pytest.approx(859.25)
    assert closing_assets == pytest.approx(closing_liabilities_and_equity)
    assert dso_cash_use == pytest.approx(26.3342466)
    assert "USD 26.33m" in forecast


def test_module4_statement_ratio_and_normalization_examples_recompute() -> None:
    statement_lesson = _read(MODULE4_SOURCES[1])
    ratio_lesson = _read(MODULE4_SOURCES[2])
    normalization_lesson = _read(MODULE4_SOURCES[3])

    cfo = 90 + 50 - 20 - 10 + 5
    cash_change = cfo - 90 - 20 - 25
    roe = 90 / ((390 + 455) / 2)
    nopat = 150 * (1 - 0.25)
    roic = nopat / (((250 + 390 - 80) + (230 + 455 - 60)) / 2)
    ccc = 130 / 1_000 * 365 + 145 / 600 * 365 - 102.5 / 600 * 365
    purchases = 600 + (150 - 140)
    purchase_dpo = 102.5 / purchases * 365
    mixed_ccc = 130 / 1_000 * 365 + 145 / 600 * 365 - purchase_dpo
    closing_equity = 390 + 90 - 25
    normalized_ebit = 120 + 18 - 10 - 6
    normalized_nopat = normalized_ebit * (1 - 0.25)
    owner_earnings = 78 + 30 - 24 - 12

    assert cfo == pytest.approx(115)
    assert cash_change == pytest.approx(-20)
    assert roe == pytest.approx(0.21301775)
    assert roic == pytest.approx(0.18987342)
    assert round(ccc, 1) == pytest.approx(73.3)
    assert purchases == pytest.approx(610)
    assert round(purchase_dpo, 1) == pytest.approx(61.3)
    assert round(mixed_ccc, 1) == pytest.approx(74.3)
    assert closing_equity == pytest.approx(455)
    assert normalized_ebit == pytest.approx(122)
    assert normalized_nopat == pytest.approx(91.5)
    assert owner_earnings == pytest.approx(72)

    assert "&=115." in statement_lesson
    assert "&=-20." in statement_lesson
    assert r"=21.30\%." in ratio_lesson
    assert r"=18.99\%." in ratio_lesson
    assert "73.3 days" in ratio_lesson
    assert r"74.4\text{ days}" in ratio_lesson
    assert "unrounded components gives 74.3 days" in ratio_lesson
    assert r"\mathrm{CCC}^{\mathrm{mixed}}" in ratio_lesson
    assert r"\text{Normalized EBIT}" in normalization_lesson
    assert "&=91.5." in normalization_lesson
    assert r"\text{Owner earnings}=78+30-24-12=72." in normalization_lesson


def test_module4_specialized_entity_examples_recompute() -> None:
    specialized = _read(MODULE4_SOURCES[4])

    combined_ratio = (310 + 125) / 500
    underwriting_result = 500 * (1 - combined_ratio)
    reported_fx_growth = (110 * 0.045) / (100 * 0.050) - 1
    constant_currency_growth = 110 / 100 - 1
    fx_operating_component = (110 - 100) * 0.050
    fx_translation_component = 110 * (0.045 - 0.050)
    fx_total_change = 110 * 0.045 - 100 * 0.050
    consolidated_revenue = 800 + 200 - 50
    consolidated_ebit = 120 + 30 - 10
    adjusted_subsidiary_income = 21 - 10 * (1 - 0.25)
    nci_income = (1 - 0.80) * adjusted_subsidiary_income
    full_goodwill = 160 + 40 - 170
    proportionate_nci = (1 - 0.80) * 170
    partial_goodwill = 160 + proportionate_nci - 170

    assert combined_ratio == pytest.approx(0.87)
    assert underwriting_result == pytest.approx(65)
    assert reported_fx_growth == pytest.approx(-0.01)
    assert constant_currency_growth == pytest.approx(0.10)
    assert fx_operating_component == pytest.approx(0.50)
    assert fx_translation_component == pytest.approx(-0.55)
    assert fx_total_change == pytest.approx(-0.05)
    assert fx_operating_component + fx_translation_component == pytest.approx(
        fx_total_change
    )
    assert consolidated_revenue == pytest.approx(950)
    assert consolidated_ebit == pytest.approx(140)
    assert adjusted_subsidiary_income == pytest.approx(13.5)
    assert nci_income == pytest.approx(2.7)
    assert full_goodwill == pytest.approx(30)
    assert proportionate_nci == pytest.approx(34)
    assert partial_goodwill == pytest.approx(24)

    for displayed_result in (
        r"&=-1.0\%.",
        r"&=10.0\%.",
        r"\text{Revenue}=800+200-50=950",
        r"\text{EBIT}=120+30-10=140",
        r"=2.7\text{ USD m}",
        "combined ratio is 87%",
        "Goodwill is $160+40-170=30$ USD m",
        r"\text{Partial goodwill}&=160+34-170=24",
        r"\text{Full goodwill}&=160+40-170=30",
    ):
        assert displayed_result in specialized


def test_module4_citation_keys_resolve_in_the_canonical_bibliography() -> None:
    cited: set[str] = set()
    for source in MODULE4_SOURCES:
        for group in re.findall(r"\{cite\}`([^`]+)`", _read(source)):
            cited.update(key.strip() for key in group.split(","))

    bibliography = _read(PROJECT_ROOT / "references.bib")
    defined = set(re.findall(r"@\w+\{([^,]+),", bibliography))

    assert cited
    assert cited <= defined


def test_module4_visual_assets_are_registered_referenced_and_present() -> None:
    manifest_path = PROJECT_ROOT / "img" / "generated" / "visual-assets" / "module-04.json"
    manifest = json.loads(_read(manifest_path))
    assets = {asset["id"]: asset for asset in manifest["assets"]}

    assert set(assets) == set(VISUAL_REFERENCES)
    for asset_id, source in VISUAL_REFERENCES.items():
        asset = assets[asset_id]
        target = PROJECT_ROOT / asset["target_path"]
        assert asset["status"] == "approved"
        assert target.is_file()
        assert asset["used_by"][0]["target_markdown"] in _read(source)
        assert asset["alt_text"] in _read(source)


def test_module4_has_one_canonical_plan_and_consistent_final_lesson_title() -> None:
    plan_names = sorted(path.name for path in (PROJECT_ROOT / "content").glob("plan*.json"))
    toc = _read(PROJECT_ROOT / "_toc.yml")
    final_lesson = _read(MODULE4_SOURCES[5])

    assert plan_names == ["plan.json"]
    assert "title: Three-Statement Model and Review Memo" in toc
    assert final_lesson.startswith("# Three-Statement Model and Review Memo\n")
