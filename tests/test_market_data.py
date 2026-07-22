from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest

import src.market_data as market_data
from src.market_data import (
    MarketDataClient,
    dashboard_data_inventory,
    live_macro_dashboard_panel,
    live_nasdaq_stock_prices,
    live_return_dashboard_prices,
    mexican_market_level_panel,
    nasdaq_stock_price_panel,
    official_macro_panel,
    official_price_panel,
    rate_to_decimal,
    returns_from_prices,
    wfe_equity_market_scale_snapshot,
)


@dataclass
class PassthroughCache:
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_dataframe(
        self,
        namespace: str,
        key: str,
        fetcher,
        **kwargs,
    ) -> pd.DataFrame:
        del namespace, key
        self.metadata = kwargs["metadata"]
        return fetcher()


def test_rate_conversion_requires_an_explicit_source_unit() -> None:
    rate = pd.Series([0.5, 5.0])

    percentage_points = rate_to_decimal(rate, source_unit="percentage_points")
    decimals = rate_to_decimal(rate, source_unit="decimal")

    assert percentage_points.tolist() == pytest.approx([0.005, 0.05])
    assert decimals.tolist() == pytest.approx([0.5, 5.0])
    with pytest.raises(ValueError, match="source_unit"):
        rate_to_decimal(rate, source_unit="guess")


def test_yahoo_price_end_date_is_inclusive_at_the_public_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_download(*args, **kwargs) -> pd.DataFrame:
        captured.update(kwargs)
        return pd.DataFrame(
            {"Close": [101.0]},
            index=pd.to_datetime(["2026-01-31"]),
        )

    monkeypatch.setattr(market_data.yf, "download", fake_download)
    cache = PassthroughCache()
    client = MarketDataClient(cache=cache)

    prices = client.yahoo_prices("TEST", start="2026-01-01", end="2026-01-31")

    assert captured["end"] == "2026-02-01"
    assert cache.metadata["end"] == "2026-01-31"
    assert cache.metadata["end_inclusive"] is True
    assert prices.index[-1] == pd.Timestamp("2026-01-31")


def test_official_price_panel_is_a_compatible_alias_for_semantic_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = pd.DataFrame(
        {"usd_mxn": [20.0], "udi": [8.0]},
        index=pd.to_datetime(["2026-01-02"]),
    )
    monkeypatch.setattr(market_data, "_read_snapshot", lambda path: expected.copy())

    result = official_price_panel(start="2026-01-01", end="2026-01-31")

    pd.testing.assert_frame_equal(result, expected.rename_axis("date"))
    assert result.attrs["panel_semantics"].startswith("jointly observed USD/MXN")
    assert result.attrs["observation_policy"].endswith("never forward-fill levels")


@pytest.mark.parametrize(
    ("loader", "message"),
    [
        (mexican_market_level_panel, "Mexican market level panel"),
        (official_macro_panel, "Macro snapshot panel"),
        (nasdaq_stock_price_panel, "NASDAQ stock panel"),
    ],
)
def test_snapshot_loaders_reject_requested_windows_without_observations(
    loader,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside_window = pd.DataFrame(
        {"value": [1.0]},
        index=pd.to_datetime(["2020-01-01"]),
    )
    monkeypatch.setattr(market_data, "_read_snapshot", lambda path: outside_window.copy())

    with pytest.raises(ValueError, match=message):
        loader(start="2026-01-01", end="2026-01-31")


def test_return_construction_preserves_source_provenance_and_declares_method() -> None:
    prices = pd.DataFrame(
        {"A": [100.0, 102.0, 101.0]},
        index=pd.date_range("2026-01-01", periods=3),
    )
    prices.attrs = {
        "sources": "versioned adjusted-close fixture",
        "currency": "USD",
    }

    returns = returns_from_prices(prices, method="simple")

    assert returns.attrs["sources"] == prices.attrs["sources"]
    assert returns.attrs["currency"] == "USD"
    assert returns.attrs["return_method"] == "simple"
    assert returns.attrs["return_construction"] == "P_t / P_{t-1} - 1"


class FakeProvider:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls: list[dict[str, Any]] = []

    def fetch_series_group(self, series, **kwargs) -> pd.DataFrame:
        self.calls.append({"series": series, **kwargs})
        return self.frame.copy()


class FakeMacroClient:
    def __init__(self, banxico: pd.DataFrame, dbnomics: pd.DataFrame) -> None:
        self.banxico = FakeProvider(banxico)
        self.dbnomics = FakeProvider(dbnomics)


def test_live_macro_has_offline_schema_and_fetches_cpi_lookback() -> None:
    monthly_dates = pd.date_range("2020-01-31", "2021-02-28", freq="ME")
    dbnomics = pd.DataFrame(
        {
            "mexico_cpi": np.arange(100.0, 100.0 + len(monthly_dates)),
            "us_10y": 4.0,
        },
        index=monthly_dates,
    )
    banxico_dates = pd.to_datetime(["2021-01-31", "2021-02-28"])
    banxico = pd.DataFrame(
        {
            "SF61745": [5.0, 5.5],
            "SF60633": [9.0, 9.5],
            "SF60648": [10.0, 10.5],
            "SF43718": [20.0, np.nan],
            "SP68257": [7.0, 7.1],
        },
        index=banxico_dates,
    )
    client = FakeMacroClient(banxico, dbnomics)

    result = live_macro_dashboard_panel(
        start="2021-01-01",
        end="2021-02-28",
        client=client,
    )

    assert list(result.columns) == list(market_data.OFFICIAL_MACRO_COLUMNS)
    assert result.index.tolist() == [pd.Timestamp("2021-01-31")]
    assert result.loc[pd.Timestamp("2021-01-31"), "banxico_target_rate"] == pytest.approx(0.05)
    assert result.loc[pd.Timestamp("2021-01-31"), "cetes_28d"] == pytest.approx(0.09)
    assert client.dbnomics.calls[0]["start"] == "2020-01-01"
    assert result.attrs["observation_policy"].endswith("no forward filling")


class FakeYahooClient:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def yahoo_prices(self, *args, **kwargs) -> pd.DataFrame:
        return self.frame.copy()


def test_live_return_prices_preserve_provider_missing_values() -> None:
    dates = pd.date_range("2026-01-01", periods=2)
    raw = pd.DataFrame(
        {"AAA": [100.0, np.nan], "BBB": [50.0, 51.0]},
        index=dates,
    )

    result = live_return_dashboard_prices(
        tickers={"Asset A": "AAA", "Asset B": "BBB"},
        client=FakeYahooClient(raw),
    )

    assert pd.isna(result.loc[dates[1], "Asset A"])
    assert result.loc[dates[1], "Asset B"] == 51.0
    assert result.attrs["observation_policy"].endswith("no forward filling")


def test_live_nasdaq_prices_preserve_provider_missing_values() -> None:
    dates = pd.date_range("2026-01-01", periods=2)
    raw = pd.DataFrame(
        {"AAA": [100.0, np.nan], "BBB": [50.0, 51.0]},
        index=dates,
    )

    result = live_nasdaq_stock_prices(
        tickers={"AAA": "Company A", "BBB": "Company B"},
        client=FakeYahooClient(raw),
    )

    assert pd.isna(result.loc[dates[1], "AAA"])
    assert result.loc[dates[1], "BBB"] == 51.0


def test_dashboard_inventory_does_not_recommend_mixed_levels_for_portfolios() -> None:
    inventory = dashboard_data_inventory().set_index("dashboard")

    publication_input = inventory.loc["Risk and portfolio dashboards", "publication_input"]

    assert publication_input != "official_price_panel.csv"
    assert "asset-return matrix" in publication_input


def test_wfe_snapshot_records_a_current_verification_date_not_a_retrieval_claim() -> None:
    snapshot = wfe_equity_market_scale_snapshot()

    assert snapshot["verified_on"].eq("2026-07-20").all()
    assert "retrieved_on" not in snapshot.columns
    assert snapshot["source_snapshot"].eq("WFE Focus dashboard, May 2026").all()
    assert snapshot["change_basis"].eq("not specified on source dashboard").all()
