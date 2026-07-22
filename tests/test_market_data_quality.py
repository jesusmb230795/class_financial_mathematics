from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.market_data_quality import (
    align_to_business_calendar,
    annualized_log_change_from_levels,
    annualized_volatility,
    banxico_series_catalog,
    data_quality_report,
    hampel_outlier_flags,
    log_returns,
    simple_returns,
)


def test_simple_returns_preserve_series_type_and_do_not_bridge_missing_levels() -> None:
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    prices = pd.Series([100.0, np.nan, 110.0, 121.0], index=dates, name="asset")

    returns = simple_returns(prices)

    expected = pd.Series([0.1], index=dates[-1:], name="asset")
    pd.testing.assert_series_equal(returns, expected)


def test_simple_returns_keep_dataframe_rows_with_at_least_one_observed_return() -> None:
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    prices = pd.DataFrame(
        {
            "gapped": [100.0, np.nan, 110.0],
            "complete": [50.0, 55.0, 60.5],
        },
        index=dates,
    )

    returns = simple_returns(prices)

    assert returns.index.equals(dates[1:])
    assert returns["gapped"].isna().all()
    assert returns["complete"].tolist() == pytest.approx([0.1, 0.1])


def test_log_returns_reject_non_positive_levels_instead_of_returning_infinity() -> None:
    prices = pd.Series([100.0, 0.0, 105.0], index=pd.date_range("2026-01-01", periods=3))

    with pytest.raises(ValueError, match="strictly positive"):
        log_returns(prices)


def test_annualized_log_change_uses_elapsed_calendar_time() -> None:
    levels = pd.Series(
        [100.0, np.e * 100.0],
        index=pd.to_datetime(["2024-01-01", "2024-12-31"]),
    )

    result = annualized_log_change_from_levels(levels, days_per_year=365.0)

    assert result == pytest.approx(1.0)


@pytest.mark.parametrize(
    "series, message",
    [
        (pd.Series([100.0], index=pd.to_datetime(["2026-01-01"])), "at least two"),
        (
            pd.Series([100.0, -1.0], index=pd.to_datetime(["2026-01-01", "2026-01-02"])),
            "strictly positive",
        ),
    ],
)
def test_annualized_log_change_validates_observed_levels(
    series: pd.Series,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        annualized_log_change_from_levels(series)


def test_quality_report_separates_structural_and_value_failures() -> None:
    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-02", "2026-01-04", "2026-01-05"])
    data = pd.DataFrame(
        {
            "numeric": [1.0, 1.0, np.inf, np.nan, 2.0],
            "mixed": ["1", "bad", "3", "4", "5"],
        },
        index=dates,
    )

    report = data_quality_report(data, expected_frequency="D")

    assert report.loc["numeric", "non_finite"] == 1
    assert report.loc["numeric", "missing"] == 1
    assert report.loc["mixed", "non_numeric"] == 1
    assert report["duplicate_timestamp_rows"].eq(2).all()
    assert report.loc["numeric", "longest_constant_run"] == 2
    assert report["missing_index_dates"].eq(1).all()
    assert report.loc["numeric", "missing_expected_observations"] == 2
    assert report.loc["mixed", "missing_expected_observations"] == 1
    assert report["calendar_checked"].all()
    assert report["expected_frequency"].eq("D").all()


def test_quality_report_does_not_claim_calendar_gaps_without_a_calendar() -> None:
    series = pd.Series(
        [1.0, 2.0],
        index=pd.to_datetime(["2026-01-01", "2026-01-03"]),
    )

    report = data_quality_report(series)

    assert not report.loc["value", "calendar_checked"]
    assert pd.isna(report.loc["value", "missing_index_dates"])
    assert pd.isna(report.loc["value", "missing_expected_observations"])


def test_business_calendar_alignment_does_not_fill_by_default() -> None:
    series = pd.Series(
        [1.0, 3.0],
        index=pd.to_datetime(["2026-01-01", "2026-01-05"]),
    )

    aligned = align_to_business_calendar(series)

    assert pd.isna(aligned.loc[pd.Timestamp("2026-01-02")])
    assert aligned.loc[pd.Timestamp("2026-01-05")] == 3.0


@pytest.mark.parametrize("periods_per_year", [0, -1])
def test_annualized_volatility_rejects_nonpositive_periods_per_year(
    periods_per_year: int,
) -> None:
    returns = pd.Series([0.01, -0.01, 0.02])

    with pytest.raises(ValueError, match="periods_per_year must be positive"):
        annualized_volatility(returns, periods_per_year=periods_per_year)


def test_banxico_catalog_marks_tiie_break_and_cetes_as_sovereign_reference() -> None:
    catalog = banxico_series_catalog().set_index("series")

    assert catalog.loc["SF60633", "course_use"] == "short sovereign-rate reference"
    assert catalog.loc["SF60648", "methodology_break"] == "2025-01-01"
    assert "not homogeneous" in catalog.loc["SF60648", "methodology_note"]


def test_hampel_flags_match_mad_computed_inside_each_rolling_window() -> None:
    series = pd.Series([1.0, 2.0, 100.0, 4.0, 5.0])

    flags = hampel_outlier_flags(series, window=5, n_sigmas=3.0)

    expected = pd.Series([False, False, True, False, False])
    pd.testing.assert_series_equal(flags, expected)
