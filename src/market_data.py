"""Reusable market data access and classroom data helpers."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.banxico import BanxicoClient
from src.cache import DataCache, default_cache, stable_cache_key
from src.dbnomics import DBNOMICS_MACRO_SERIES, DBnomicsClient
from src.market_data_quality import log_returns, simple_returns

DEFAULT_LIVE_START = "2020-01-01"
DEFAULT_LIVE_END = "2024-12-31"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshots"
BANXICO_DAILY_PATH = SNAPSHOT_DIR / "banxico_daily.csv"
OFFICIAL_PRICE_PANEL_PATH = SNAPSHOT_DIR / "official_price_panel.csv"
OFFICIAL_MACRO_PANEL_PATH = SNAPSHOT_DIR / "official_macro_panel.csv"
NASDAQ_STOCK_PANEL_PATH = SNAPSHOT_DIR / "nasdaq_stock_panel.csv"
SNAPSHOT_METADATA_PATH = SNAPSHOT_DIR / "metadata.json"

DEFAULT_NASDAQ_STOCK_TICKERS = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com, Inc.",
    "GOOGL": "Alphabet Inc. Class A",
}

DEFAULT_RETURN_DASHBOARD_TICKERS = {ticker: ticker for ticker in DEFAULT_NASDAQ_STOCK_TICKERS}

LIVE_MACRO_BANXICO_SERIES = {
    "banxico_target_rate": "SF61745",
    "cetes_28d": "SF60633",
    "tiie_28d": "SF60648",
    "usd_mxn": "SF43718",
    "udi": "SP68257",
}

OFFICIAL_MACRO_COLUMNS = (
    "banxico_target_rate",
    "cetes_28d",
    "tiie_28d",
    "usd_mxn",
    "udi",
    "mexico_cpi",
    "mexico_inflation",
    "us_10y",
)


class MarketDataClient:
    """Facade for Yahoo Finance, DB.NOMICS, and Banxico data access."""

    def __init__(self, cache: DataCache | None = None) -> None:
        self.cache = cache or default_cache()
        self.banxico = BanxicoClient(cache=self.cache)
        self.dbnomics = DBnomicsClient(cache=self.cache)

    def yahoo_prices(
        self,
        tickers: str | list[str],
        start: str,
        end: str,
        field: str = "Close",
        ttl: timedelta | None = timedelta(days=1),
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch adjusted Yahoo Finance prices over an inclusive date interval."""
        ticker_list = [tickers] if isinstance(tickers, str) else list(tickers)
        key = stable_cache_key("yahoo", ticker_list, start, end, field)
        end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        def fetcher() -> pd.DataFrame:
            data = yf.download(
                ticker_list,
                start=start,
                end=end_exclusive,
                auto_adjust=True,
                progress=False,
                group_by="column",
            )
            if isinstance(data.columns, pd.MultiIndex):
                prices = data[field]
            else:
                prices = data[[field]].rename(columns={field: ticker_list[0]})
            return prices.dropna(how="all").rename_axis("date")

        return self.cache.get_dataframe(
            "yahoo",
            key,
            fetcher,
            ttl=ttl,
            force_refresh=force_refresh,
            metadata={
                "provider": "yahoo",
                "tickers": ticker_list,
                "start": start,
                "end": end,
                "end_inclusive": True,
                "field": field,
            },
        )


def rate_to_decimal(series: pd.Series, *, source_unit: str) -> pd.Series:
    """Convert a rate series to decimals under an explicit source-unit contract."""
    numeric = pd.to_numeric(series, errors="coerce")
    if source_unit == "percentage_points":
        return numeric / 100
    if source_unit == "decimal":
        return numeric
    raise ValueError("source_unit must be 'percentage_points' or 'decimal'")


def _read_snapshot(path: str | Path) -> pd.DataFrame:
    snapshot_path = Path(path)
    if not snapshot_path.exists():
        display_path = (
            snapshot_path.relative_to(PROJECT_ROOT)
            if snapshot_path.is_relative_to(PROJECT_ROOT)
            else snapshot_path
        )
        raise FileNotFoundError(
            f"Snapshot not found: {display_path}. Run "
            "`PYTHONPATH=$PWD uv run python scripts/generate_real_data_snapshots.py` "
            "with valid Banxico and DB.NOMICS credentials."
        )
    return pd.read_csv(snapshot_path, index_col="date", parse_dates=True).sort_index()


def _slice_dates(
    frame: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    start_date = pd.to_datetime(start) if start else frame.index.min()
    end_date = pd.to_datetime(end) if end else frame.index.max()
    return frame.loc[start_date:end_date]


def banxico_daily_panel(
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load provider-dated Banxico observations without calendar filling.

    This source-layer surface preserves every date present in the committed
    Banxico extract. Individual columns can contain missing values because the
    series follow different publication calendars. By contrast,
    :func:`mexican_market_level_panel` is a derived analytical surface that
    retains the joint USD/MXN and UDI observation dates and adds synthetic carry
    indexes.
    """
    panel = _read_snapshot(BANXICO_DAILY_PATH)
    if start is not None or end is not None:
        panel = _slice_dates(panel, start=start, end=end)

    metadata: dict[str, object] = {}
    if SNAPSHOT_METADATA_PATH.exists():
        metadata = json.loads(SNAPSHOT_METADATA_PATH.read_text(encoding="utf-8"))

    panel.attrs["data_mode"] = "snapshot"
    panel.attrs["sources"] = "Banxico SIE official snapshot"
    panel.attrs["observation_policy"] = (
        "provider-dated observations; no calendar reindexing or forward fill"
    )
    panel.attrs["snapshot_generated_at"] = metadata.get("generated_at")
    panel.attrs["series_ids"] = metadata.get("banxico_series", {})
    if not panel.empty:
        panel.attrs["start"] = panel.index.min().strftime("%Y-%m-%d")
        panel.attrs["end"] = panel.index.max().strftime("%Y-%m-%d")
    return panel.rename_axis("date")


def mexican_market_level_panel(
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load observed Mexican market levels plus synthetic short-rate carry indexes."""
    panel = _read_snapshot(OFFICIAL_PRICE_PANEL_PATH)
    if start is not None or end is not None:
        panel = _slice_dates(panel, start=start, end=end)
    if panel.empty:
        raise ValueError("Mexican market level panel has no observations in the requested window.")
    panel.attrs["data_mode"] = "snapshot"
    panel.attrs["sources"] = (
        "Versioned Banxico SIE observations and documented synthetic carry transformations"
    )
    panel.attrs["panel_semantics"] = (
        "jointly observed USD/MXN and UDI levels with synthetic short-rate carry indexes"
    )
    panel.attrs["observation_policy"] = (
        "retain dates where both USD/MXN and UDI were observed; never forward-fill levels"
    )
    panel.attrs["start"] = panel.index.min().strftime("%Y-%m-%d")
    panel.attrs["end"] = panel.index.max().strftime("%Y-%m-%d")
    return panel.rename_axis("date")


def official_price_panel(
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Compatibility alias for :func:`mexican_market_level_panel`."""
    return mexican_market_level_panel(start=start, end=end)


def official_macro_panel(
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load the versioned Banxico and DB.NOMICS macro panel used by the book."""
    panel = _read_snapshot(OFFICIAL_MACRO_PANEL_PATH)
    if start is not None or end is not None:
        panel = _slice_dates(panel, start=start, end=end)
    if panel.empty:
        raise ValueError("Macro snapshot panel has no observations in the requested window.")
    panel.attrs["data_mode"] = "snapshot"
    panel.attrs["sources"] = "Versioned Banxico SIE and DB.NOMICS snapshots"
    panel.attrs["start"] = panel.index.min().strftime("%Y-%m-%d")
    panel.attrs["end"] = panel.index.max().strftime("%Y-%m-%d")
    return panel.rename_axis("date")


def nasdaq_stock_price_panel(
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load the versioned NASDAQ equity panel used for stock EDA lessons."""
    panel = _read_snapshot(NASDAQ_STOCK_PANEL_PATH)
    if start is not None or end is not None:
        panel = _slice_dates(panel, start=start, end=end)
    if panel.empty:
        raise ValueError("NASDAQ stock panel has no observations in the requested window.")
    panel.attrs["data_mode"] = "snapshot"
    panel.attrs["sources"] = "Versioned Yahoo-derived adjusted closes via yfinance"
    panel.attrs["tickers"] = DEFAULT_NASDAQ_STOCK_TICKERS
    panel.attrs["currency"] = "USD"
    panel.attrs["field"] = "provider-adjusted close"
    panel.attrs["frequency"] = "US trading days"
    panel.attrs["start"] = panel.index.min().strftime("%Y-%m-%d")
    panel.attrs["end"] = panel.index.max().strftime("%Y-%m-%d")
    return panel.rename_axis("date")


def live_macro_dashboard_panel(
    start: str = DEFAULT_LIVE_START,
    end: str = DEFAULT_LIVE_END,
    client: MarketDataClient | None = None,
    ttl: timedelta | None = timedelta(days=1),
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch a live macro dashboard panel from Banxico and DB.NOMICS.

    Returned columns match the versioned official macro snapshot so dashboard
    code can switch between reproducible and live real-data inputs without
    changing visualization logic.
    """
    client = client or MarketDataClient()
    requested_start = pd.Timestamp(start)
    requested_end = pd.Timestamp(end)
    if requested_start > requested_end:
        raise ValueError("start must be on or before end")
    cpi_lookback_start = (
        requested_start.to_period("M").start_time - pd.DateOffset(months=12)
    ).strftime("%Y-%m-%d")

    banxico = client.banxico.fetch_series_group(
        list(LIVE_MACRO_BANXICO_SERIES.values()),
        start=start,
        end=end,
        ttl=ttl,
        force_refresh=force_refresh,
    )
    dbnomics = client.dbnomics.fetch_series_group(
        DBNOMICS_MACRO_SERIES,
        start=cpi_lookback_start,
        end=end,
        ttl=ttl,
        force_refresh=force_refresh,
    )
    banxico = banxico.rename(
        columns={series_id: label for label, series_id in LIVE_MACRO_BANXICO_SERIES.items()}
    )
    banxico_monthly = banxico.resample("ME").last()
    dbnomics_monthly = dbnomics.resample("ME").last()
    cpi = dbnomics_monthly["mexico_cpi"]
    macro = pd.DataFrame(
        {
            "banxico_target_rate": rate_to_decimal(
                banxico_monthly["banxico_target_rate"], source_unit="percentage_points"
            ),
            "cetes_28d": rate_to_decimal(
                banxico_monthly["cetes_28d"], source_unit="percentage_points"
            ),
            "tiie_28d": rate_to_decimal(
                banxico_monthly["tiie_28d"], source_unit="percentage_points"
            ),
            "usd_mxn": banxico_monthly["usd_mxn"],
            "udi": banxico_monthly["udi"],
            "mexico_cpi": cpi,
            "mexico_inflation": cpi.pct_change(12, fill_method=None),
            "us_10y": rate_to_decimal(dbnomics_monthly["us_10y"], source_unit="percentage_points"),
        }
    )
    macro = macro.loc[requested_start:requested_end, list(OFFICIAL_MACRO_COLUMNS)]
    macro = macro.dropna(how="any").rename_axis("date")
    if macro.empty:
        raise ValueError("Live macro dashboard panel is empty after alignment and cleaning.")
    macro.attrs["data_mode"] = "live"
    macro.attrs["sources"] = "Banxico SIE and DB.NOMICS"
    macro.attrs["observation_policy"] = "monthly last observations; no forward filling"
    macro.attrs["start"] = macro.index.min().strftime("%Y-%m-%d")
    macro.attrs["end"] = macro.index.max().strftime("%Y-%m-%d")
    return macro


def live_return_dashboard_prices(
    start: str = DEFAULT_LIVE_START,
    end: str = DEFAULT_LIVE_END,
    tickers: dict[str, str] | None = None,
    client: MarketDataClient | None = None,
    ttl: timedelta | None = timedelta(days=1),
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch a live public-market price panel for the return explorer."""
    client = client or MarketDataClient()
    ticker_map = tickers or DEFAULT_RETURN_DASHBOARD_TICKERS
    raw_prices = client.yahoo_prices(
        list(ticker_map.values()),
        start=start,
        end=end,
        ttl=ttl,
        force_refresh=force_refresh,
    )
    rename_map = {ticker: label for label, ticker in ticker_map.items()}
    prices = raw_prices.rename(columns=rename_map).sort_index()
    prices = prices.reindex(columns=list(ticker_map.keys())).dropna(axis=1, how="all")
    prices = prices.dropna(how="all")
    if prices.empty:
        raise ValueError("Live return dashboard price panel is empty after provider fetch.")
    prices.attrs["data_mode"] = "live"
    prices.attrs["sources"] = "Yahoo Finance public market prices"
    prices.attrs["tickers"] = ticker_map
    prices.attrs["observation_policy"] = "provider observations; no forward filling"
    prices.attrs["start"] = start
    prices.attrs["end"] = end
    return prices.rename_axis("date")


def live_nasdaq_stock_prices(
    start: str = DEFAULT_LIVE_START,
    end: str = DEFAULT_LIVE_END,
    tickers: dict[str, str] | None = None,
    client: MarketDataClient | None = None,
    ttl: timedelta | None = timedelta(days=1),
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch a live NASDAQ stock panel from Yahoo Finance for local exploration."""
    client = client or MarketDataClient()
    ticker_map = tickers or DEFAULT_NASDAQ_STOCK_TICKERS
    raw_prices = client.yahoo_prices(
        list(ticker_map.keys()),
        start=start,
        end=end,
        ttl=ttl,
        force_refresh=force_refresh,
    )
    prices = raw_prices.reindex(columns=list(ticker_map.keys())).dropna(axis=1, how="all")
    prices = prices.dropna(how="all")
    if prices.empty:
        raise ValueError("Live NASDAQ stock price panel is empty after provider fetch.")
    prices.attrs["data_mode"] = "live"
    prices.attrs["sources"] = "Yahoo Finance public market prices via yfinance"
    prices.attrs["tickers"] = ticker_map
    prices.attrs["observation_policy"] = "provider observations; no forward filling"
    prices.attrs["start"] = start
    prices.attrs["end"] = end
    return prices.rename_axis("date")


def macro_dashboard_panel(
    data_mode: str = "offline",
    start: str = DEFAULT_LIVE_START,
    end: str = DEFAULT_LIVE_END,
    **live_kwargs,
) -> pd.DataFrame:
    """Return a macro dashboard panel from official snapshots or live providers."""
    normalized_mode = data_mode.lower()
    if normalized_mode in {"offline", "snapshot"}:
        return official_macro_panel(start=start, end=end)
    if normalized_mode == "live":
        return live_macro_dashboard_panel(start=start, end=end, **live_kwargs)
    raise ValueError("data_mode must be 'offline', 'snapshot', or 'live'")


def return_dashboard_price_panel(
    data_mode: str = "offline",
    start: str = DEFAULT_LIVE_START,
    end: str = DEFAULT_LIVE_END,
    **live_kwargs,
) -> pd.DataFrame:
    """Return a return-dashboard price panel from real snapshots or live providers."""
    normalized_mode = data_mode.lower()
    if normalized_mode in {"offline", "snapshot"}:
        return nasdaq_stock_price_panel(start=start, end=end)
    if normalized_mode == "live":
        return live_return_dashboard_prices(start=start, end=end, **live_kwargs)
    raise ValueError("data_mode must be 'offline', 'snapshot', or 'live'")


def returns_from_prices(
    prices: pd.Series | pd.DataFrame,
    method: str = "log",
) -> pd.Series | pd.DataFrame:
    """Compute simple or log returns from prices."""
    if method == "log":
        result = log_returns(prices)
    elif method == "simple":
        result = simple_returns(prices)
    else:
        raise ValueError("method must be 'log' or 'simple'")
    result.attrs = dict(prices.attrs)
    result.attrs["return_method"] = method
    result.attrs["return_construction"] = (
        "log(P_t / P_{t-1})" if method == "log" else "P_t / P_{t-1} - 1"
    )
    return result


def align_time_series(
    frames: dict[str, pd.Series | pd.DataFrame],
    frequency: str = "B",
    fill_method: str | None = None,
) -> pd.DataFrame:
    """Align named time series without filling unless a method is explicit."""
    normalized = []
    for name, data in frames.items():
        frame = data.to_frame(name=name) if isinstance(data, pd.Series) else data.copy()
        normalized.append(frame)
    combined = pd.concat(normalized, axis=1).sort_index()
    calendar = pd.date_range(combined.index.min(), combined.index.max(), freq=frequency)
    aligned = combined.reindex(calendar)
    if fill_method is None:
        return aligned
    if fill_method not in {"ffill", "bfill"}:
        raise ValueError("fill_method must be 'ffill', 'bfill', or None")
    return getattr(aligned, fill_method)()


def wfe_equity_market_scale_snapshot() -> pd.DataFrame:
    """Return a static WFE global market scale snapshot for publication builds.

    Values are copied from the World Federation of Exchanges Focus dashboard
    for May 2026. They are stored in code instead of fetched live so the book
    remains reproducible without network access.
    """
    source_url = "https://focus.world-exchanges.org/issue/may-2026/dashboard"
    source_snapshot = "WFE Focus dashboard, May 2026"
    verified_on = "2026-07-20"
    rows = [
        {
            "metric": "Market capitalisation",
            "category": "equity market scale",
            "reported_value": 149_199_047.91,
            "reported_unit": "USD millions",
            "display_value": 149.20,
            "display_unit": "USD trillions",
            "change_percent": 5.93,
        },
        {
            "metric": "Value of share trading",
            "category": "equity market activity",
            "reported_value": 25_186_065.32,
            "reported_unit": "USD millions",
            "display_value": 25.19,
            "display_unit": "USD trillions",
            "change_percent": 25.07,
        },
        {
            "metric": "Listed companies",
            "category": "market breadth",
            "reported_value": 60_055,
            "reported_unit": "domestic and foreign companies",
            "display_value": 60.06,
            "display_unit": "thousand companies",
            "change_percent": 0.15,
        },
        {
            "metric": "Number of trades",
            "category": "equity market activity",
            "reported_value": 6_948_777.81,
            "reported_unit": "thousand trades",
            "display_value": 6.95,
            "display_unit": "billion trades",
            "change_percent": 27.79,
        },
        {
            "metric": "Investment flows",
            "category": "primary market flow",
            "reported_value": 14_945.69,
            "reported_unit": "USD millions",
            "display_value": 14.95,
            "display_unit": "USD billions",
            "change_percent": 30.82,
        },
        {
            "metric": "Options contracts traded",
            "category": "derivatives activity",
            "reported_value": 6_732_597_316,
            "reported_unit": "contracts",
            "display_value": 6.73,
            "display_unit": "billion contracts",
            "change_percent": 8.75,
        },
        {
            "metric": "Futures contracts traded",
            "category": "derivatives activity",
            "reported_value": 3_367_804_629,
            "reported_unit": "contracts",
            "display_value": 3.37,
            "display_unit": "billion contracts",
            "change_percent": 45.10,
        },
    ]
    snapshot = pd.DataFrame(rows)
    snapshot["source_snapshot"] = source_snapshot
    snapshot["source_url"] = source_url
    snapshot["verified_on"] = verified_on
    snapshot["change_basis"] = "not specified on source dashboard"
    return snapshot


def dashboard_data_inventory() -> pd.DataFrame:
    """Return the publication-provider map for the interactive dashboards."""
    return pd.DataFrame(
        [
            {
                "dashboard": "Macro dashboard",
                "primary_sources": "Banxico SIE and DB.NOMICS",
                "publication_input": "official_macro_panel.csv",
                "provider_notes": "Publication builds read versioned Banxico and DB.NOMICS snapshots.",
            },
            {
                "dashboard": "Return explorer",
                "primary_sources": "Yahoo Finance through yfinance",
                "publication_input": "nasdaq_stock_panel.csv",
                "provider_notes": "Use Yahoo Finance adjusted-close snapshots for reproducible NASDAQ stock return examples.",
            },
            {
                "dashboard": "NASDAQ equity EDA",
                "primary_sources": "Yahoo Finance through yfinance",
                "publication_input": "nasdaq_stock_panel.csv",
                "provider_notes": "Sufficient for classroom EDA of daily adjusted closes; not sufficient as an official, redistribution, or trading feed.",
            },
            {
                "dashboard": "Risk and portfolio dashboards",
                "primary_sources": "Clean return matrix from documented market providers",
                "publication_input": "module-specific documented asset-return matrix",
                "provider_notes": "Do not treat mixed market levels and synthetic carry indexes as a single asset universe.",
            },
        ]
    )
