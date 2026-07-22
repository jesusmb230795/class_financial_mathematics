"""FRED data client helpers."""

from __future__ import annotations

import os
from datetime import timedelta

import pandas as pd

from src.cache import DataCache, default_cache, stable_cache_key


def fred_series_catalog() -> pd.DataFrame:
    """Return a compact FRED catalog used in the course."""
    return pd.DataFrame(
        [
            {
                "series": "INTGSTMXM193N",
                "meaning": "Mexican government securities and treasury bills rate",
                "course_use": "risk-free-rate proxy discussion",
            },
            {
                "series": "MEXCPALTT01IXNBM",
                "meaning": "Mexico consumer price index",
                "course_use": "inflation and real return context",
            },
            {
                "series": "DEXMXUS",
                "meaning": "Mexican pesos to one U.S. dollar",
                "course_use": "FX risk context",
            },
            {
                "series": "DGS10",
                "meaning": "10-year U.S. Treasury yield",
                "course_use": "global rates comparison",
            },
        ]
    )


class FredClient:
    """Small FRED client with cache support.

    The client uses `fredapi` when `FRED_API_KEY` is available and falls back to
    `pandas_datareader` for public FRED series otherwise.
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache: DataCache | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        self.cache = cache or default_cache()

    def _fetch_with_fredapi(self, series_id: str, start: str, end: str) -> pd.DataFrame:
        from fredapi import Fred

        fred = Fred(api_key=self.api_key)
        series = fred.get_series(series_id, observation_start=start, observation_end=end)
        return series.rename("value").to_frame().rename_axis("date")

    def _fetch_with_datareader(self, series_id: str, start: str, end: str) -> pd.DataFrame:
        from pandas_datareader import data as web

        frame = web.DataReader(series_id, "fred", start, end)
        return frame.rename(columns={series_id: "value"}).rename_axis("date")

    def fetch_series(
        self,
        series_id: str,
        start: str,
        end: str,
        ttl: timedelta | None = timedelta(days=1),
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch one FRED series and return a DataFrame indexed by date."""
        key = stable_cache_key("fred", series_id, start, end)

        def fetcher() -> pd.DataFrame:
            if self.api_key:
                return self._fetch_with_fredapi(series_id, start, end)
            return self._fetch_with_datareader(series_id, start, end)

        return self.cache.get_dataframe(
            "fred",
            key,
            fetcher,
            ttl=ttl,
            force_refresh=force_refresh,
            metadata={"provider": "fred", "series_id": series_id, "start": start, "end": end},
        )

    def fetch_series_group(
        self,
        series_ids: list[str],
        start: str,
        end: str,
        ttl: timedelta | None = timedelta(days=1),
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch multiple FRED series and return a wide value matrix."""
        frames = []
        for series_id in series_ids:
            frame = self.fetch_series(
                series_id,
                start=start,
                end=end,
                ttl=ttl,
                force_refresh=force_refresh,
            )
            frames.append(frame["value"].rename(series_id))
        return pd.concat(frames, axis=1).sort_index()
