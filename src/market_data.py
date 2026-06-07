"""Reusable market data access and classroom data helpers."""

from __future__ import annotations

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
OFFICIAL_PRICE_PANEL_PATH = SNAPSHOT_DIR / "official_price_panel.csv"
OFFICIAL_MACRO_PANEL_PATH = SNAPSHOT_DIR / "official_macro_panel.csv"
NASDAQ_STOCK_PANEL_PATH = SNAPSHOT_DIR / "nasdaq_stock_panel.csv"

DEFAULT_NASDAQ_STOCK_TICKERS = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com, Inc.",
    "GOOGL": "Alphabet Inc. Class A",
}

DEFAULT_RETURN_DASHBOARD_TICKERS = {ticker: ticker for ticker in DEFAULT_NASDAQ_STOCK_TICKERS}


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
        """Fetch adjusted Yahoo Finance prices with local caching."""
        ticker_list = [tickers] if isinstance(tickers, str) else list(tickers)
        key = stable_cache_key("yahoo", ticker_list, start, end, field)

        def fetcher() -> pd.DataFrame:
            data = yf.download(
                ticker_list,
                start=start,
                end=end,
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
                "field": field,
            },
        )


def _percent_to_decimal(series: pd.Series) -> pd.Series:
    """Convert percentage-point series to decimals when needed."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().abs().max() > 1:
        return numeric / 100
    return numeric


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


def official_price_panel(
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load the versioned official price-like panel used by the published book."""
    panel = _read_snapshot(OFFICIAL_PRICE_PANEL_PATH)
    if start is not None or end is not None:
        panel = _slice_dates(panel, start=start, end=end)
    panel.attrs["data_mode"] = "snapshot"
    panel.attrs["sources"] = "Banxico SIE official snapshot"
    panel.attrs["start"] = panel.index.min().strftime("%Y-%m-%d")
    panel.attrs["end"] = panel.index.max().strftime("%Y-%m-%d")
    return panel.rename_axis("date")


def official_macro_panel(
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load the versioned official macro panel used by the published book."""
    panel = _read_snapshot(OFFICIAL_MACRO_PANEL_PATH)
    if start is not None or end is not None:
        panel = _slice_dates(panel, start=start, end=end)
    panel.attrs["data_mode"] = "snapshot"
    panel.attrs["sources"] = "Banxico SIE and DB.NOMICS official snapshots"
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
    panel.attrs["data_mode"] = "snapshot"
    panel.attrs["sources"] = "Yahoo Finance via yfinance snapshot"
    panel.attrs["tickers"] = DEFAULT_NASDAQ_STOCK_TICKERS
    panel.attrs["start"] = panel.index.min().strftime("%Y-%m-%d")
    panel.attrs["end"] = panel.index.max().strftime("%Y-%m-%d")
    return panel.rename_axis("date")


def live_macro_dashboard_panel(
    start: str = DEFAULT_LIVE_START,
    end: str = DEFAULT_LIVE_END,
    client: MarketDataClient | None = None,
    market_ticker: str = "^MXX",
    ttl: timedelta | None = timedelta(days=1),
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch a live macro dashboard panel from Banxico, DB.NOMICS, and public prices.

    Returned columns match the versioned official macro snapshot so dashboard
    code can switch between reproducible and live real-data inputs without
    changing visualization logic.
    """
    client = client or MarketDataClient()

    banxico = client.banxico.fetch_series_group(
        ["SF61745", "SF43718"],
        start=start,
        end=end,
        ttl=ttl,
        force_refresh=force_refresh,
    )
    dbnomics = client.dbnomics.fetch_series_group(
        DBNOMICS_MACRO_SERIES,
        start=start,
        end=end,
        ttl=ttl,
        force_refresh=force_refresh,
    )
    market_prices = client.yahoo_prices(
        market_ticker,
        start=start,
        end=end,
        ttl=ttl,
        force_refresh=force_refresh,
    )

    cpi = dbnomics["mexico_cpi"].resample("ME").last()
    macro = pd.DataFrame(
        {
            "banxico_target_rate": _percent_to_decimal(banxico["SF61745"]).resample("ME").last(),
            "mexico_inflation": cpi.pct_change(12, fill_method=None),
            "usd_mxn": banxico["SF43718"].resample("ME").last(),
            "us_10y": _percent_to_decimal(dbnomics["us_10y"]).resample("ME").last(),
            "ipc_index": market_prices.iloc[:, 0].resample("ME").last(),
        }
    )
    macro = macro.ffill().dropna().rename_axis("date")
    if macro.empty:
        raise ValueError("Live macro dashboard panel is empty after alignment and cleaning.")
    macro.attrs["data_mode"] = "live"
    macro.attrs["sources"] = "Banxico SIE, DB.NOMICS, Yahoo Finance"
    macro.attrs["start"] = start
    macro.attrs["end"] = end
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
    prices = prices.dropna(how="all").ffill()
    if prices.empty:
        raise ValueError("Live return dashboard price panel is empty after provider fetch.")
    prices.attrs["data_mode"] = "live"
    prices.attrs["sources"] = "Yahoo Finance public market prices"
    prices.attrs["tickers"] = ticker_map
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
    prices = prices.dropna(how="all").ffill()
    if prices.empty:
        raise ValueError("Live NASDAQ stock price panel is empty after provider fetch.")
    prices.attrs["data_mode"] = "live"
    prices.attrs["sources"] = "Yahoo Finance public market prices via yfinance"
    prices.attrs["tickers"] = ticker_map
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
        return log_returns(prices)
    if method == "simple":
        return simple_returns(prices)
    raise ValueError("method must be 'log' or 'simple'")


def align_time_series(
    frames: dict[str, pd.Series | pd.DataFrame],
    frequency: str = "B",
    fill_method: str | None = "ffill",
) -> pd.DataFrame:
    """Align named time series to a common calendar."""
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
    retrieved_on = "2026-06-06"
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
    snapshot["retrieved_on"] = retrieved_on
    return snapshot


def dashboard_data_inventory() -> pd.DataFrame:
    """Return the recommended provider map for the interactive dashboards."""
    return pd.DataFrame(
        [
            {
                "dashboard": "Macro dashboard",
                "primary_sources": "Banxico SIE, DB.NOMICS, INEGI API, World Bank",
                "publication_input": "official_macro_panel.csv",
                "provider_notes": "Publication builds read versioned Banxico and DB.NOMICS snapshots.",
            },
            {
                "dashboard": "Return explorer",
                "primary_sources": "Finnhub, EODHD, Alpha Vantage, FMP, Yahoo Finance",
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
                "publication_input": "official_price_panel.csv",
                "provider_notes": "Cache provider extracts before modeling risk or portfolios.",
            },
        ]
    )
