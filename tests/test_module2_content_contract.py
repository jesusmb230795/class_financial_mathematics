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


def test_module2_has_seven_canonical_lessons_with_objectives_and_handoff() -> None:
    assert len(MODULE2_SOURCES) == 7

    for source in MODULE2_SOURCES:
        text = _read(source)
        assert "## Learning objectives" in text
        assert "## Prerequisites" in text
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


def test_module2_var_distinguishes_log_threshold_from_exact_simple_loss() -> None:
    risk_lesson = _read(COURSE_DIR / "2.5.garch_volatility_risk_workflow.py")

    assert r"\operatorname{VaR}^{(g)}" in risk_lesson
    assert r"\operatorname{VaR}^{(R)}" in risk_lesson
    assert r"\max\left\{0,1-" in risk_lesson
    assert "simple_loss_from_log_return(" in risk_lesson
    assert "normal_log_return_quantile_pct / 100" in risk_lesson
    assert "student_t_log_return_quantile_pct / 100" in risk_lesson
    assert "one_step_var_log_return_loss_threshold_pct" in risk_lesson
    assert "one_step_var_simple_return_loss_pct" in risk_lesson
    assert "long USD / short MXN" in risk_lesson
    assert '"overall_diagnostic_status"' in risk_lesson
    assert '"failed_diagnostic_count"' in risk_lesson
    assert '"diagnostic_status"' in risk_lesson


def test_module2_dashboard_labels_the_provider_interval_horizon() -> None:
    dashboard_lesson = _read(
        COURSE_DIR / "2.6.interactive_volatility_garch_dashboard.py"
    )

    assert "Log return per FIX publication interval (%)" in dashboard_lesson
    assert "Volatility per FIX publication interval" in dashboard_lesson
    assert r"\alpha g_{t-1}^2" in dashboard_lesson


def test_module2_arima_keeps_selection_training_only_and_uses_materiality() -> None:
    arima_lesson = _read(COURSE_DIR / "2.4.arima_diagnostic_workflow.py")

    split_position = arima_lesson.index("training_returns = returns.iloc[:split_index]")
    adf_position = arima_lesson.index(
        'return_adf = adf_report(training_returns, regression="c")'
    )
    search_position = arima_lesson.index(
        "candidate_results = arima_order_search(\n    training_returns,"
    )
    evaluation_position = arima_lesson.index(
        "training_forecast_result = training_model.get_forecast"
    )
    assert split_position < adf_position < search_position < evaluation_position
    assert "training_returns.index.max() < testing_returns.index.min()" in arima_lesson
    assert "MATERIAL_RELATIVE_MAE_IMPROVEMENT" in arima_lesson
    assert "same_constant_forecast_class" in arima_lesson
    assert "zero-return benchmark" in arima_lesson
    assert "arima_beats_benchmark" not in arima_lesson


def test_module2_programmatic_figures_have_semantic_alt_text() -> None:
    figure_sources = {
        "2.0.quantitative_foundations.py": 1,
        "2.1.time_series_1.py": 2,
        "2.2.time_series_2.py": 1,
        "2.3.time_series_diagnostics_and_volatility_extensions.py": 2,
        "2.4.arima_diagnostic_workflow.py": 2,
        "2.5.garch_volatility_risk_workflow.py": 2,
        "2.6.interactive_volatility_garch_dashboard.py": 1,
    }
    for filename, minimum_count in figure_sources.items():
        source = _read(COURSE_DIR / filename)
        assert source.count('mystnb={"image": {"alt":') >= minimum_count
