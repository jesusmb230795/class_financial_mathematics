from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COURSE_DIR = PROJECT_ROOT / "notebooks" / "course"
MODULE5_SOURCES = [
    PROJECT_ROOT / "chapters" / "05-corporate-equity-valuation.md",
    COURSE_DIR / "5.1.corporate_decisions_and_capital_budgeting.md",
    COURSE_DIR / "5.2.capital_structure_and_wacc.md",
    COURSE_DIR / "5.3.business_model_and_forecasting.md",
    COURSE_DIR / "5.4.equity_valuation_frameworks.md",
    COURSE_DIR / "5.5.investment_thesis_and_valuation_memo.md",
]
VISUAL_REFERENCES = {
    "m5-corporate-value-creation-valuation-map": MODULE5_SOURCES[0],
    "m5-project-cash-flow-npv-profile": MODULE5_SOURCES[1],
    "m5-capital-structure-wacc-sensitivity": MODULE5_SOURCES[2],
    "m5-fcff-forecast-waterfall": MODULE5_SOURCES[3],
    "m5-dcf-value-sensitivity": MODULE5_SOURCES[4],
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _npv(rate: float, cash_flows: list[float]) -> float:
    return sum(cash_flow / (1 + rate) ** period for period, cash_flow in enumerate(cash_flows))


def _irr(cash_flows: list[float]) -> float:
    lower, upper = -0.99, 1.0
    for _ in range(200):
        midpoint = (lower + upper) / 2
        if _npv(lower, cash_flows) * _npv(midpoint, cash_flows) <= 0:
            upper = midpoint
        else:
            lower = midpoint
    return (lower + upper) / 2


def test_module5_lessons_preserve_objectives_prerequisites_and_handoffs() -> None:
    for source in MODULE5_SOURCES:
        text = _read(source)
        assert "## Learning objectives" in text
        assert "## Prerequisites" in text
        assert "## Handoff" in text


def test_module5_uses_renderable_inline_math_delimiters() -> None:
    combined = "\n".join(_read(path) for path in MODULE5_SOURCES)

    assert r"\(" not in combined
    assert r"\)" not in combined
    assert combined.count("$") % 2 == 0


def test_module5_capital_budgeting_examples_recompute() -> None:
    lesson = _read(MODULE5_SOURCES[1])
    worked = [-110, 28, 28, 28, 28, 46]
    practice = [-85, 26, 26, 26, 40]

    assert _npv(0.10, worked) == pytest.approx(7.31861336)
    assert _irr(worked) == pytest.approx(0.12391313)
    assert _npv(0.09, practice) == pytest.approx(9.15066976)
    assert _irr(practice) == pytest.approx(0.13487116)
    for displayed_result in ("7.32", "12.39%", "9.15", "13.49%"):
        assert displayed_result in lesson
    assert "discrete payback is 4.00 years" in lesson
    assert "discrete payback of 4.00 years" in lesson
    assert "payback is about 3.18 years" not in lesson
    assert "sole outflow is" in lesson
    assert r"PI_{\mathrm{gross}}" in lesson
    assert "IRR itself does not assume" in lesson


def test_module5_wacc_scenarios_recompute_with_market_value_weights() -> None:
    lesson = _read(MODULE5_SOURCES[2])
    risk_free_rate = 0.04
    equity_risk_premium = 0.055
    tax_rate = 0.25
    unlevered_beta = 0.80
    scenarios = (
        (0.20, 0.060, 0.9500, 0.09225, 0.08280),
        (1 / 3, 0.065, 1.1000, 0.10050, 0.08325),
        (0.45, 0.080, 1.2909, 0.11100, 0.08805),
        (0.55, 0.105, 1.5333, 0.12433, 0.09926),
    )

    for debt_weight, debt_cost, expected_beta, expected_equity_cost, expected_wacc in scenarios:
        debt_to_equity = debt_weight / (1 - debt_weight)
        levered_beta = unlevered_beta * (1 + (1 - tax_rate) * debt_to_equity)
        equity_cost = risk_free_rate + levered_beta * equity_risk_premium
        wacc = (1 - debt_weight) * equity_cost + debt_weight * debt_cost * (1 - tax_rate)
        assert levered_beta == pytest.approx(expected_beta, abs=5e-5)
        assert equity_cost == pytest.approx(expected_equity_cost, abs=5e-5)
        assert wacc == pytest.approx(expected_wacc, abs=5e-5)

    for displayed_row in (
        "| 20% | 0.25× | 6.0% | 0.95 | 9.23% | 8.28% |",
        "| 33% | 0.50× | 6.5% | 1.10 | 10.05% | 8.33% |",
        "| 45% | 0.82× | 8.0% | 1.29 | 11.10% | 8.81% |",
        "| 55% | 1.22× | 10.5% | 1.53 | 12.43% | 9.93% |",
    ):
        assert displayed_row in lesson

    assert r"\tau_{\mathrm{shield}}" in lesson
    assert r"\mathrm{ERP}" in lesson
    assert "$T$" not in lesson
    assert r"+\mathrm{PV}(\text{financing subsidies})" in lesson
    assert r"-\mathrm{PV}(\text{issuance, distress, and recapture costs})" in lesson
    assert r"\mathrm{APV}_{\mathrm{issuer}}" in lesson
    assert r"=V_U+\mathrm{PV}(\text{financing effects})" in lesson
    assert "distress, subsidy, or other incremental effects" not in lesson


def test_module5_forecast_examples_recompute_and_define_churn_rate() -> None:
    lesson = _read(MODULE5_SOURCES[3])
    ending_customers = 100_000 - 8_000 + 30_000
    average_customers = (100_000 + ending_customers) / 2
    revenue = average_customers * 240 / 1_000_000
    gross_profit = revenue * 0.70
    ebit = gross_profit - 10.0 - 2.0
    nopat = ebit * (1 - 0.25)
    fcff = nopat + 2.0 - 2.5 - 0.6

    assert ending_customers == 122_000
    assert average_customers == 111_000
    assert revenue == pytest.approx(26.640)
    assert gross_profit == pytest.approx(18.648)
    assert ebit == pytest.approx(6.648)
    assert nopat == pytest.approx(4.986)
    assert fcff == pytest.approx(3.886)
    assert r"c=\frac{C_{\mathrm{churn}}}{C_0}" in lesson
    assert r"\frac{P_1Q_1-P_0Q_0}{P_0Q_0}" in lesson
    assert r"\tau_{\mathrm{op}}" in lesson
    assert r"\tau_{\mathrm{op}}\max(\mathrm{EBIT}_t,0)" in lesson
    assert "explicit tax-loss schedule" in lesson
    for displayed_row in (
        "| Downside | 24.4400 | 3.8860 | 1.8145 |",
        "| Base | 26.6400 | 6.6480 | 3.8860 |",
        "| Upside | 28.0525 | 7.4978 | 4.1234 |",
    ):
        assert displayed_row in lesson
    assert "At 12% churn, 6,000 customers are lost" in lesson


def test_module5_dcf_examples_and_sensitivity_recompute() -> None:
    lesson = _read(MODULE5_SOURCES[4])
    fcff = [80.0, 88.0, 95.0]
    debt, nci, cash, shares = 400.0, 20.0, 80.0, 100.0

    for wacc, expected_values in (
        (0.08, (11.67, 14.38, 18.46)),
        (0.09, (9.50, 11.40, 14.07)),
        (0.10, (7.87, 9.27, 11.14)),
    ):
        actual_values = []
        for growth in (0.02, 0.03, 0.04):
            explicit_value = sum(
                value / (1 + wacc) ** period for period, value in enumerate(fcff, 1)
            )
            terminal_value = fcff[-1] * (1 + growth) / (wacc - growth)
            enterprise_value = explicit_value + terminal_value / (1 + wacc) ** 3
            actual_values.append((enterprise_value - debt - nci + cash) / shares)
        assert actual_values == pytest.approx(expected_values, abs=0.005)

    for displayed_row in (
        "| 8.0% | 11.67 | 14.38 | 18.46 |",
        "| 9.0% | 9.50 | 11.40 | 14.07 |",
        "| 10.0% | 7.87 | 9.27 | 11.14 |",
    ):
        assert displayed_row in lesson

    assert r"\text{Reinvestment rate}_{N+1}=\frac{g}{\mathrm{RONIC}_{N+1}}" in lesson
    assert (
        r"\mathrm{FCFF}_{N+1}"
        "\n"
        r"=\mathrm{NOPAT}_{N+1}"
        "\n"
        r"\left(1-\frac{g}{\mathrm{RONIC}_{N+1}}\right)"
    ) in lesson
    assert r"\mathrm{FCFE}_t=\mathrm{FCFF}_t" in lesson
    assert r"\mathrm{RI}_t=\mathrm{NI}_t^{\text{common}}-k_eB_{t-1}" in lesson
    assert r"\mathrm{ROE}_t^{\text{beg}}" in lesson
    assert "average-equity ROE" in lesson
    assert "do not subtract its assigned debt a" in lesson
    assert r"\tau_{\mathrm{op}}" in lesson
    assert r"\tau_{\mathrm{shield},t}" in lesson
    sotp_common_equity = 900 + 260 - 80 + 40 - 300 - 20 + 60
    assert sotp_common_equity == pytest.approx(860.0)
    assert sotp_common_equity / 100 == pytest.approx(8.60)
    assert r"900+260-80+40-300-20+60=860" in lesson
    assert "USD 8.60 per share" in lesson


def test_module5_memo_scenario_bridge_recomputes_per_share_values() -> None:
    memo = _read(MODULE5_SOURCES[5])
    bridge = 620 + 20 - 120
    cases = (
        ([92.0, 113.9, 135.8, 157.8, 168.3], 0.10, 0.025),
        ([105.0, 130.0, 155.0, 180.0, 192.0], 0.09, 0.03),
        ([108.0, 133.7, 159.4, 185.1, 197.4], 0.085, 0.035),
    )
    enterprise_values = []
    for fcff, wacc, growth in cases:
        explicit = sum(value / (1 + wacc) ** period for period, value in enumerate(fcff, 1))
        terminal = fcff[-1] * (1 + growth) / (wacc - growth)
        enterprise_values.append(explicit + terminal / (1 + wacc) ** 5)

    assert enterprise_values == pytest.approx([1_920.26, 2_719.91, 3_320.25], abs=0.005)
    assert [
        ((enterprise_value - bridge) / 100) for enterprise_value in enterprise_values
    ] == pytest.approx([14.0, 22.0, 28.0], abs=0.005)
    assert 94 + 3 + 2 + 1 == 100
    assert 137.944 + 90.000 == pytest.approx(227.944)
    assert 103.46 + 90.00 - 100.00 - (-11.54) == pytest.approx(105.00)
    multiple_values = [((227.944 * multiple) - bridge) / 100 for multiple in (10.7, 12.4)]
    assert multiple_values == pytest.approx([19.19, 23.07], abs=0.005)
    residual_income_values = [
        12 + residual_income / (0.10 - 0.02) for residual_income in (0.64, 0.80)
    ]
    assert residual_income_values == pytest.approx([20.0, 22.0])
    assert [residual_income + 0.10 * 12 for residual_income in (0.64, 0.80)] == pytest.approx(
        [1.84, 2.00]
    )
    assert "| Downside | USD 1,920.26m | USD 1,400.26m | USD 14.00 |" in memo
    assert "| Base | USD 2,719.91m | USD 2,199.91m | USD 22.00 |" in memo
    assert "| Upside | USD 3,320.25m | USD 2,800.25m | USD 28.00 |" in memo
    assert "| Interest-bearing debt | USD 620m |" in memo
    assert "| Excess cash | USD 120m |" in memo
    assert "| Non-controlling interests (NCI) | USD 20m |" in memo
    assert r"\frac{\mathrm{EV}-620-20+120}{100}" in memo
    assert r"\mathrm{EV}_{0,s}" in memo
    assert "6.11%" in memo
    assert "The source register for this worked example" in memo
    assert r"\mathrm{CCC}=\mathrm{DSO}+\mathrm{DIO}-\mathrm{DPO}" in memo
    assert r"103.46+90.00-100.00-(-11.54)" in memo
    assert r"\mathrm{NI}_1^{\mathrm{common}}" in memo
    assert "| EBITDA | 227.94 |" in memo
    for displayed_row in (
        "| Downside | USD 1,920.26m | USD 1,400.26m | USD 14.00 |",
        "| Base | USD 2,719.91m | USD 2,199.91m | USD 22.00 |",
        "| Upside | USD 3,320.25m | USD 2,800.25m | USD 28.00 |",
    ):
        assert displayed_row in memo
    assert r"N_{\mathrm{diluted}}" in memo
    assert "net incremental share count" in memo
    assert r"B_0=12(100)=1{,}200" in memo


def test_module5_citations_are_dense_traceable_and_resolve() -> None:
    cited_by_source: dict[Path, set[str]] = {}
    all_cited: set[str] = set()
    for source in MODULE5_SOURCES:
        cited: set[str] = set()
        for group in re.findall(r"\{cite\}`([^`]+)`", _read(source)):
            cited.update(key.strip() for key in group.split(","))
        cited_by_source[source] = cited
        all_cited.update(cited)

    bibliography = _read(PROJECT_ROOT / "references.bib")
    defined = set(re.findall(r"@\w+\{([^,]+),", bibliography))
    all_defined = re.findall(r"@\w+\{([^,]+),", bibliography)

    assert all(len(cited) >= 2 for cited in cited_by_source.values())
    assert len(all_cited) >= 30
    assert all_cited <= defined
    assert len(all_defined) == len({key.casefold() for key in all_defined})


def test_module5_visual_assets_are_registered_referenced_and_present() -> None:
    manifest_path = PROJECT_ROOT / "img/generated/visual-assets/module-05.json"
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
        assert any(
            "no external image or third-party data" in note for note in asset["production_notes"]
        )
