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


def dashboard_data_inventory() -> pd.DataFrame:
    """Return the recommended provider map for the interactive dashboards."""
    return pd.DataFrame(
        [
            {
                "dashboard": "Macro dashboard",
                "primary_sources": "Banxico SIE, FRED",
                "offline_fallback": "synthetic_macro_panel",
            },
            {
                "dashboard": "Return explorer",
                "primary_sources": "Yahoo Finance or instructor data",
                "offline_fallback": "synthetic_price_panel",
            },
            {
                "dashboard": "Risk and portfolio dashboards",
                "primary_sources": "Clean return matrix",
                "offline_fallback": "synthetic_price_panel",
            },
        ]
    )
