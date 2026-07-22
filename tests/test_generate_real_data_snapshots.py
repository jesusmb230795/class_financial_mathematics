from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.generate_real_data_snapshots as snapshots
from scripts.generate_real_data_snapshots import build_price_panel, carry_index


def test_carry_index_uses_previous_rate_and_elapsed_calendar_days() -> None:
    rates = pd.Series(
        [36.0, 72.0, 10.0],
        index=pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-04"]),
        name="rate",
    )

    result = carry_index(rates)

    expected = pd.Series(
        [100.0, 100.2001, 100.4005002],
        index=rates.index,
        name="rate",
    )
    pd.testing.assert_series_equal(result, expected)


def test_carry_index_uses_last_published_rate_across_missing_observations() -> None:
    rates = pd.Series(
        [36.0, np.nan, 72.0],
        index=pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-06"]),
    )

    result = carry_index(rates)

    assert result.iloc[0] == 100.0
    assert result.iloc[1] == pytest.approx(100.2001)
    assert result.iloc[2] == pytest.approx(100.5010010005)


def test_build_price_panel_never_fills_observed_market_levels() -> None:
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    banxico = pd.DataFrame(
        {
            "usd_mxn": [20.0, np.nan, 21.0, 22.0],
            "udi": [7.0, 7.1, 7.2, 7.3],
            "cetes_28d": [36.0, 72.0, np.nan, np.nan],
            "tiie_28d": [36.0, 72.0, np.nan, np.nan],
            "policy_rate": [36.0, 72.0, np.nan, np.nan],
        },
        index=dates,
    )

    panel = build_price_panel(banxico)

    assert dates[1] not in panel.index
    assert panel.loc[dates[2], "usd_mxn"] == 21.0
    assert panel.loc[dates[2], "cetes_28d_carry"] == pytest.approx(100.3002)
    assert panel.loc[dates[3], "cetes_28d_carry"] == pytest.approx(100.5008004)
    assert panel.notna().all().all()


def test_build_price_panel_rejects_sources_without_jointly_observed_levels() -> None:
    dates = pd.date_range("2026-01-01", periods=2, freq="D")
    banxico = pd.DataFrame(
        {
            "usd_mxn": [np.nan, np.nan],
            "udi": [7.0, 7.1],
            "cetes_28d": [8.0, 8.0],
            "tiie_28d": [9.0, 9.0],
            "policy_rate": [7.0, 7.0],
        },
        index=dates,
    )

    with pytest.raises(ValueError, match="empty after observed-level alignment"):
        build_price_panel(banxico)


def _versioned_source_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2020-01-31", periods=14, freq="ME")
    banxico = pd.DataFrame(
        {
            "usd_mxn": np.linspace(18.0, 20.0, len(dates)),
            "cetes_28d": 8.0,
            "tiie_28d": 9.0,
            "policy_rate": 7.0,
            "udi": np.linspace(6.0, 7.0, len(dates)),
        },
        index=dates,
    ).rename_axis("date")
    dbnomics = pd.DataFrame(
        {
            "mexico_cpi": np.linspace(100.0, 115.0, len(dates)),
            "us_10y": 4.0,
        },
        index=dates,
    ).rename_axis("date")
    nasdaq = pd.DataFrame(
        {ticker: [100.0, 101.0] for ticker in snapshots.DEFAULT_NASDAQ_STOCK_TICKERS},
        index=pd.to_datetime(["2021-01-04", "2021-01-05"]),
    ).rename_axis("date")
    return banxico, dbnomics, nasdaq


def test_derived_only_rebuilds_from_versioned_csvs_without_touching_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    monkeypatch.setattr(snapshots, "SNAPSHOT_DIR", snapshot_dir)
    paths = snapshots._snapshot_paths()
    banxico, dbnomics, nasdaq = _versioned_source_frames()
    banxico.to_csv(paths["banxico_daily"], date_format="%Y-%m-%d")
    dbnomics.to_csv(paths["dbnomics_daily"], date_format="%Y-%m-%d")
    nasdaq.to_csv(paths["nasdaq_stock_panel"], date_format="%Y-%m-%d")
    vintage = "2026-06-07T03:26:53+00:00"
    (snapshot_dir / "metadata.json").write_text(
        json.dumps(
            {
                "generated_at": vintage,
                "snapshot_start": "2020-01-31",
                "snapshot_end": "2021-02-28",
                "nasdaq_stock_snapshot_generated_at": "2026-06-07T04:55:41+00:00",
            }
        ),
        encoding="utf-8",
    )
    source_bytes = {
        key: paths[key].read_bytes()
        for key in ("banxico_daily", "dbnomics_daily", "nasdaq_stock_panel")
    }

    def unexpected_network_call(*args, **kwargs):
        raise AssertionError("--derived-only must not call a network fetcher")

    monkeypatch.setattr(snapshots, "fetch_banxico_panel", unexpected_network_call)
    monkeypatch.setattr(snapshots, "fetch_dbnomics_panel", unexpected_network_call)
    monkeypatch.setattr(snapshots, "fetch_nasdaq_stock_panel", unexpected_network_call)

    result = snapshots.main(["--derived-only"])

    assert result == 0
    for key, original_bytes in source_bytes.items():
        assert paths[key].read_bytes() == original_bytes
    assert paths["official_price_panel"].exists()
    assert paths["official_macro_panel"].exists()

    metadata = json.loads((snapshot_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["generated_at"] == vintage
    assert metadata["derived_generated_at"] != vintage
    assert metadata["rebuilt_from_versioned_sources"] is True
    assert metadata["generation_mode"] == "derived-only"
    assert set(metadata["row_counts"]) == set(snapshots.OUTPUT_FILENAMES)
    assert set(metadata["columns"]) == set(snapshots.OUTPUT_FILENAMES)
    assert set(metadata["output_coverage"]) == set(snapshots.OUTPUT_FILENAMES)
    assert set(metadata["sha256"]) == set(snapshots.OUTPUT_FILENAMES)
    assert metadata["carry_index_methodology"]["formula"].startswith("I_d = I_(d-1)")
    assert "product" in metadata["carry_index_methodology"]["stored_interval_formula"]
    assert "internal daily grid" in metadata["carry_index_methodology"]["calendar_day_policy"]
    assert metadata["carry_index_methodology"]["level_observation_policy"].startswith(
        "USD/MXN and UDI levels are never forward-filled"
    )
    assert metadata["methodology_breaks"]["tiie_28d"]["effective_date"] == "2025-01-01"
    assert metadata["methodology_breaks"]["tiie_28d"]["series_id"] == "SF60648"
