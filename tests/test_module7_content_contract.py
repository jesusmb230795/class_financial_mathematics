from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COURSE_DIR = PROJECT_ROOT / "notebooks" / "course"
MODULE7_LESSONS = sorted(COURSE_DIR.glob("7.*.py"))
MODULE7_OVERVIEW = PROJECT_ROOT / "chapters/07-derivatives-risk-management.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_module7_has_twelve_complete_published_lesson_sources() -> None:
    assert len(MODULE7_LESSONS) == 12
    assert [path.name.split(".", 2)[1] for path in MODULE7_LESSONS] == [
        "1",
        "10",
        "11",
        "12",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]

    for lesson in MODULE7_LESSONS:
        text = _read(lesson)
        assert "## Lesson summary" in text, lesson
        assert "## Learning objectives" in text, lesson
        assert "## Prerequisites" in text, lesson
        assert "## Handoff" in text, lesson
        assert "## Model limitations" in text or "## Limitations" in text, lesson
        assert "{cite" in text, lesson


def test_module7_math_uses_enabled_delimiters_and_canonical_symbols() -> None:
    combined = "\n".join(_read(path) for path in MODULE7_LESSONS)

    assert r"\(" not in combined
    assert r"\)" not in combined
    assert r"\rho_{S,v}" in _read(COURSE_DIR / "7.7.stochastic_volatility_heston_lab.py")
    assert r"n_H^*=-\frac{S_P}{S_H}" in _read(COURSE_DIR / "7.8.hedging_risk_governance.py")
    assert r"\int_0^\alpha Q_u(R)\,du" in _read(COURSE_DIR / "7.10.downside_risk_var_methods.py")
    assert "Hull2018" not in combined
    assert "cmeFutures`" not in combined
    assert "baselMarketRiskFramework2019" not in combined


def test_module7_uses_shared_finance_and_visual_builders() -> None:
    combined = "\n".join(_read(path) for path in MODULE7_LESSONS)

    assert "def black_scholes_price" not in combined
    assert "def expected_shortfall" not in combined
    assert "plt.subplots(" not in combined
    assert combined.count('mystnb={"image": {"alt":') >= 10
    for builder in (
        "build_option_payoff_figure",
        "build_convergence_figure",
        "build_implied_volatility_figure",
        "build_heston_diagnostics_figure",
        "build_tail_risk_figure",
        "build_ewma_volatility_figure",
        "build_var_backtest_figure",
    ):
        assert builder in combined


def test_portfolio_risk_lessons_use_one_observed_simple_return_contract() -> None:
    risk_lessons = [
        COURSE_DIR / "7.9.value_at_risk_foundations.py",
        COURSE_DIR / "7.10.downside_risk_var_methods.py",
        COURSE_DIR / "7.11.var_backtesting_and_stress_testing.py",
        COURSE_DIR / "7.12.interactive_var_cvar_simulator.py",
    ]
    combined = "\n".join(_read(path) for path in risk_lessons)

    assert "official_price_panel" not in combined
    assert combined.count("nasdaq_stock_price_panel") >= 8
    assert combined.count('method="simple"') == 4
    assert "log_return" not in combined
    assert "redistribution rights not independently verified" in combined

    for lesson_number in ("7.10", "7.11", "7.12"):
        path = next(path for path in risk_lessons if path.name.startswith(lesson_number))
        text = _read(path)
        assert "rebalanced_portfolio_returns" in text
        assert all(
            f'"{ticker}": 0.20' in text for ticker in ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL")
        )


def test_module7_primary_reference_set_is_present_and_used() -> None:
    bibliography = _read(PROJECT_ROOT / "references.bib")
    publication_text = "\n".join(
        [_read(MODULE7_OVERVIEW), *(_read(path) for path in MODULE7_LESSONS)]
    )
    expected_keys = {
        "blackScholes1973",
        "merton1973",
        "coxRossRubinstein1979",
        "boyle1977",
        "breedenLitzenberger1978",
        "leisenReimer1996",
        "kemnaVorst1990",
        "broadieGlassermanKou1997",
        "heston1993",
        "acerbiTasche2002",
        "rockafellarUryasev2002",
        "riskMetrics1996",
        "basel1996MarketRiskAmendment",
    }

    for key in expected_keys:
        assert re.search(rf"@[a-z]+\{{{re.escape(key)},", bibliography, flags=re.IGNORECASE)
        assert key in publication_text


def test_module7_static_risk_assets_match_their_registered_markdown() -> None:
    manifest = json.loads(
        (PROJECT_ROOT / "img/generated/visual-assets/module-07-risk.json").read_text(
            encoding="utf-8"
        )
    )
    foundation = _read(COURSE_DIR / "7.9.value_at_risk_foundations.py")

    assert len(manifest["assets"]) == 3
    for asset in manifest["assets"]:
        assert asset["status"] == "approved"
        assert asset["production_method"] == "deterministic_matplotlib"
        assert asset["generator_script"].startswith(
            "scripts/generate_module7_risk_diagrams.py --asset "
        )
        assert (PROJECT_ROOT / asset["target_path"]).is_file()
        assert asset["used_by"][0]["target_markdown"] in foundation
        assert "legacy_sources" not in asset
