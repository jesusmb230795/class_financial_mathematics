from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COURSE_DIR = PROJECT_ROOT / "notebooks" / "course"
MODULE2_SOURCES = sorted(COURSE_DIR.glob("2.*.py"))
EMPIRICAL_SOURCES = [
    COURSE_DIR / "2.1.time_series_1.py",
    COURSE_DIR / "2.3.time_series_diagnostics_and_volatility_extensions.py",
    COURSE_DIR / "2.4.arima_diagnostic_workflow.py",
    COURSE_DIR / "2.2.time_series_2.py",
    COURSE_DIR / "2.5.garch_volatility_risk_workflow.py",
    COURSE_DIR / "2.6.interactive_volatility_garch_dashboard.py",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_module2_has_seven_canonical_lessons_with_assessment_and_handoff() -> None:
    assert len(MODULE2_SOURCES) == 7

    for source in MODULE2_SOURCES:
        text = _read(source)
        assert "## Learning objectives" in text
        assert "## Prerequisites" in text
        assert 'tags=["exercise"]' in text
        assert 'tags=["solution"]' in text
        assert "## Handoff" in text


def test_module2_empirical_lessons_preserve_provider_observation_dates() -> None:
    for source in EMPIRICAL_SOURCES:
        text = _read(source)
        assert "banxico_daily_panel" in text
        assert "official_price_panel" not in text
        assert ".ffill(" not in text
        assert "daily log-return" not in text.lower()
        assert "long_run_daily" not in text


def test_module2_retired_math_and_visual_patterns_do_not_return() -> None:
    text = "\n".join(
        [
            _read(PROJECT_ROOT / "chapters" / "02-time-series.md"),
            *(_read(source) for source in MODULE2_SOURCES),
        ]
    )

    assert r"\(" not in text
    assert r"\)" not in text
    assert "alpha=0.5" not in text
    assert "Adj Close" not in text
    for retired_asset in (
        "ts-stationarity-properties.png",
        "ts-volatility-from-prices.png",
        "ts-sample-volatility-estimators.png",
    ):
        assert retired_asset not in text


def test_module2_var_equation_uses_the_positive_loss_floor() -> None:
    risk_lesson = _read(COURSE_DIR / "2.5.garch_volatility_risk_workflow.py")

    assert r"\max\left\{0,-\left(" in risk_lesson
    assert "long USD / short MXN" in risk_lesson
    assert '"diagnostic_status"' in risk_lesson


def test_module2_dashboard_labels_the_provider_interval_horizon() -> None:
    dashboard_lesson = _read(
        COURSE_DIR / "2.6.interactive_volatility_garch_dashboard.py"
    )

    assert "Return per FIX publication interval" in dashboard_lesson
    assert "Volatility per FIX publication interval" in dashboard_lesson
