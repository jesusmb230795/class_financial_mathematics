"""Reusable market data access and classroom data helpers."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from src.banxico import BanxicoClient
from src.cache import DataCache, default_cache, stable_cache_key
from src.fred import FredClient
from src.market_data_quality import log_returns, simple_returns

DEFAULT_LIVE_START = "2020-01-01"
DEFAULT_LIVE_END = "2024-12-31"

DEFAULT_RETURN_DASHBOARD_TICKERS = {
    "mexican_equity_index": "^MXX",
    "mexico_etf_usd": "EWW",
    "global_equity": "SPY",
    "usd_mxn": "MXN=X",
}


class MarketDataClient:
    """Facade for Yahoo Finance, FRED, and Banxico data access."""

    def __init__(self, cache: DataCache | None = None) -> None:
        self.cache = cache or default_cache()
        self.banxico = BanxicoClient(cache=self.cache)
        self.fred = FredClient(cache=self.cache)

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


def live_macro_dashboard_panel(
    start: str = DEFAULT_LIVE_START,
    end: str = DEFAULT_LIVE_END,
    client: MarketDataClient | None = None,
    market_ticker: str = "^MXX",
    ttl: timedelta | None = timedelta(days=1),
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch a live macro dashboard panel from Banxico, FRED, and public prices.

    Returned columns match `synthetic_macro_panel` so dashboard code can switch
    between offline and live data without changing visualization logic.
    """
    client = client or MarketDataClient()

    banxico = client.banxico.fetch_series_group(
        ["SF61745", "SF43718"],
        start=start,
        end=end,
        ttl=ttl,
        force_refresh=force_refresh,
    )
    fred = client.fred.fetch_series_group(
        ["MEXCPALTT01IXNBM", "DGS10"],
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

    cpi = fred["MEXCPALTT01IXNBM"].resample("ME").last()
    macro = pd.DataFrame(
        {
            "banxico_target_rate": _percent_to_decimal(banxico["SF61745"]).resample("ME").last(),
            "mexico_inflation": cpi.pct_change(12, fill_method=None),
            "usd_mxn": banxico["SF43718"].resample("ME").last(),
            "us_10y": _percent_to_decimal(fred["DGS10"]).resample("ME").last(),
            "ipc_index": market_prices.iloc[:, 0].resample("ME").last(),
        }
    )
    macro = macro.ffill().dropna().rename_axis("date")
    if macro.empty:
        raise ValueError("Live macro dashboard panel is empty after alignment and cleaning.")
    macro.attrs["data_mode"] = "live"
    macro.attrs["sources"] = "Banxico SIE, FRED, Yahoo Finance"
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


def macro_dashboard_panel(
    data_mode: str = "offline",
    start: str = DEFAULT_LIVE_START,
    end: str = DEFAULT_LIVE_END,
    periods: int = 96,
    seed: int = 2027,
    **live_kwargs,
) -> pd.DataFrame:
    """Return a macro dashboard panel for offline or live mode."""
    normalized_mode = data_mode.lower()
    if normalized_mode == "offline":
        panel = synthetic_macro_panel(periods=periods, seed=seed)
        panel.attrs["data_mode"] = "offline"
        panel.attrs["sources"] = "synthetic_macro_panel"
        return panel
    if normalized_mode == "live":
        return live_macro_dashboard_panel(start=start, end=end, **live_kwargs)
    raise ValueError("data_mode must be 'offline' or 'live'")


def return_dashboard_price_panel(
    data_mode: str = "offline",
    start: str = DEFAULT_LIVE_START,
    end: str = DEFAULT_LIVE_END,
    periods: int = 756,
    seed: int = 2026,
    **live_kwargs,
) -> pd.DataFrame:
    """Return a return-dashboard price panel for offline or live mode."""
    normalized_mode = data_mode.lower()
    if normalized_mode == "offline":
        panel = synthetic_price_panel(periods=periods, seed=seed)
        panel.attrs["data_mode"] = "offline"
        panel.attrs["sources"] = "synthetic_price_panel"
        return panel
    if normalized_mode == "live":
        return live_return_dashboard_prices(start=start, end=end, **live_kwargs)
    raise ValueError("data_mode must be 'offline' or 'live'")


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


def synthetic_price_panel(
    periods: int = 756,
    seed: int = 2026,
) -> pd.DataFrame:
    """Create a reproducible multi-asset price panel for classroom dashboards."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=periods)
    assets = ["mexican_equity", "global_equity", "mxn_bond", "usd_mxn"]
    mean = np.array([0.00035, 0.00028, 0.00010, 0.00005])
    covariance = np.array(
        [
            [0.00016, 0.00008, 0.00001, 0.00003],
            [0.00008, 0.00012, 0.00002, 0.00002],
            [0.00001, 0.00002, 0.00002, -0.00001],
            [0.00003, 0.00002, -0.00001, 0.00008],
        ]
    )
    returns = rng.multivariate_normal(mean, covariance, size=periods)
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=assets).rename_axis("date")


def synthetic_macro_panel(
    periods: int = 96,
    seed: int = 2027,
) -> pd.DataFrame:
    """Create a reproducible macro panel resembling Banxico and FRED data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-31", periods=periods, freq="ME")
    target_rate = 0.075 + np.cumsum(rng.normal(0.0003, 0.0020, periods))
    inflation = 0.045 + 0.60 * (target_rate - target_rate.mean()) + rng.normal(0, 0.004, periods)
    usd_mxn = 19.5 + np.cumsum(rng.normal(0.02, 0.20, periods))
    us_10y = 0.028 + np.cumsum(rng.normal(0.0001, 0.0015, periods))
    ipc_index = 100 * np.exp(np.cumsum(rng.normal(0.006, 0.035, periods)))

    return pd.DataFrame(
        {
            "banxico_target_rate": target_rate,
            "mexico_inflation": inflation,
            "usd_mxn": usd_mxn,
            "us_10y": us_10y,
            "ipc_index": ipc_index,
        },
        index=dates,
    ).rename_axis("date")


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
                "primary_sources": "Banxico SIE, FRED, INEGI API, World Bank, DBnomics",
                "offline_fallback": "synthetic_macro_panel",
                "provider_notes": "Prefer official Mexico sources and documented macro APIs.",
            },
            {
                "dashboard": "Return explorer",
                "primary_sources": "Finnhub, EODHD, Alpha Vantage, FMP, Yahoo Finance",
                "offline_fallback": "synthetic_price_panel",
                "provider_notes": "Use Yahoo Finance as a convenience fallback, not the only source.",
            },
            {
                "dashboard": "Risk and portfolio dashboards",
                "primary_sources": "Clean return matrix from documented market providers",
                "offline_fallback": "synthetic_price_panel",
                "provider_notes": "Cache provider extracts before modeling risk or portfolios.",
            },
        ]
    )
