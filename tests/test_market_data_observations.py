from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.market_data as market_data
from src.market_data import banxico_daily_panel, official_price_panel, returns_from_prices


def _observed_source_and_intersection() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2026-01-01", periods=5, freq="D", name="date")
    source = pd.DataFrame(
        {
            "usd_mxn": [np.nan, 20.0, 20.0, np.nan, 21.0],
            "udi": [7.0, 7.1, 7.2, 7.3, 7.4],
            "cetes_28d": [8.0, 8.0, np.nan, np.nan, 8.1],
            "tiie_28d": [9.0, 9.0, np.nan, np.nan, 9.1],
            "policy_rate": [7.0, 7.0, np.nan, np.nan, 7.1],
        },
        index=dates,
    )
    observed_levels = source[["usd_mxn", "udi"]].dropna(how="any")
    intersection = observed_levels.assign(
        cetes_28d_carry=[100.0, 100.1, 100.4],
        tiie_28d_carry=[100.0, 100.2, 100.5],
        policy_rate_carry=[100.0, 100.1, 100.3],
    )
    return source, intersection


@pytest.fixture
def synthetic_snapshots(monkeypatch: pytest.MonkeyPatch) -> tuple[pd.DataFrame, pd.DataFrame]:
    source, intersection = _observed_source_and_intersection()

    def fake_read_snapshot(path: str | Path) -> pd.DataFrame:
        if Path(path) == market_data.BANXICO_DAILY_PATH:
            return source.copy()
        if Path(path) == market_data.OFFICIAL_PRICE_PANEL_PATH:
            return intersection.copy()
        raise AssertionError(f"unexpected snapshot path: {path}")

    monkeypatch.setattr(market_data, "_read_snapshot", fake_read_snapshot)
    monkeypatch.setattr(
        market_data,
        "SNAPSHOT_METADATA_PATH",
        Path("nonexistent-test-metadata.json"),
    )
    return source, intersection


def test_market_level_panel_is_the_joint_observed_intersection_without_fill(
    synthetic_snapshots: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    source, expected_intersection = synthetic_snapshots

    observed = banxico_daily_panel(start="2026-01-01", end="2026-01-05")
    panel = official_price_panel(start="2026-01-01", end="2026-01-05")

    expected_dates = source[["usd_mxn", "udi"]].dropna(how="any").index
    assert panel.index.equals(expected_dates)
    pd.testing.assert_series_equal(panel["usd_mxn"], expected_intersection["usd_mxn"])
    assert pd.Timestamp("2026-01-01") in observed.index
    assert pd.Timestamp("2026-01-01") not in panel.index
    assert pd.Timestamp("2026-01-04") not in panel.index
    assert observed.attrs["observation_policy"].startswith("provider-dated")
    assert panel.attrs["observation_policy"].endswith("never forward-fill levels")


def test_market_level_returns_match_consecutive_observed_fix_returns(
    synthetic_snapshots: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    source, _ = synthetic_snapshots
    panel = official_price_panel(start="2026-01-01", end="2026-01-05")

    observed_returns = returns_from_prices(source["usd_mxn"].dropna(), method="simple")
    panel_returns = returns_from_prices(panel["usd_mxn"], method="simple")

    pd.testing.assert_series_equal(panel_returns, observed_returns)
    assert panel_returns.index.tolist() == [
        pd.Timestamp("2026-01-03"),
        pd.Timestamp("2026-01-05"),
    ]
