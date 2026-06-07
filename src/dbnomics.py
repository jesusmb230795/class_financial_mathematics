"""DB.NOMICS data client helpers."""

from __future__ import annotations

import os
from datetime import timedelta
from urllib.parse import quote

import pandas as pd
import requests

from src.cache import DataCache, default_cache, stable_cache_key

DBNOMICS_API_BASE_URL = "https://api.db.nomics.world/v22"

DBNOMICS_MACRO_SERIES = {
    "mexico_cpi": "IMF/CPI/M.MX.PCPI_IX",
    "us_10y": "FED/H15/RIFLGFCY10_N.B",
}


def dbnomics_series_catalog() -> pd.DataFrame:
    """Return a compact DB.NOMICS catalog used in the course."""
    return pd.DataFrame(
        [
            {
                "series": DBNOMICS_MACRO_SERIES["mexico_cpi"],
                "provider": "IMF",
                "dataset": "Consumer Price Index",
                "meaning": "Mexico consumer price index, all items",
                "course_use": "inflation and real return context",
            },
            {
                "series": DBNOMICS_MACRO_SERIES["us_10y"],
                "provider": "FED",
                "dataset": "Selected Interest Rates",
                "meaning": "10-year U.S. Treasury nominal constant maturity yield",
                "course_use": "global rates comparison",
            },
        ]
    )


class DBnomicsClient:
    """Small DB.NOMICS API client with cache support."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base_url: str | None = None,
        cache: DataCache | None = None,
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key or os.getenv("DBNOMICS_API_KEY")
        self.api_base_url = (
            api_base_url or os.getenv("DBNOMICS_API_BASE_URL") or DBNOMICS_API_BASE_URL
        ).rstrip("/")
        self.cache = cache or default_cache()
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _series_url(self, series_path: str) -> str:
        parts = series_path.split("/", 2)
        if len(parts) != 3:
            raise ValueError(
                "DB.NOMICS series paths must use 'PROVIDER/DATASET/SERIES' format, "
                f"got {series_path!r}."
            )
        provider_code, dataset_code, series_code = parts
        quoted = [quote(part, safe="") for part in (provider_code, dataset_code, series_code)]
        return f"{self.api_base_url}/series/{quoted[0]}/{quoted[1]}/{quoted[2]}"

    def _fetch_from_api(self, series_path: str, start: str, end: str) -> pd.DataFrame:
        response = requests.get(
            self._series_url(series_path),
            params={"observations": 1, "metadata": 0},
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        errors = payload.get("errors") or []
        if errors:
            raise ValueError(f"DB.NOMICS returned errors for {series_path}: {errors}")

        docs = payload.get("series", {}).get("docs", [])
        if not docs:
            raise ValueError(f"DB.NOMICS returned no series for {series_path}.")

        doc = docs[0]
        dates = doc.get("period_start_day") or doc.get("period")
        values = doc.get("value")
        if dates is None or values is None:
            raise ValueError(f"DB.NOMICS response for {series_path} does not include observations.")

        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(dates, errors="coerce"),
                "value": pd.to_numeric(pd.Series(values), errors="coerce"),
            }
        ).dropna(subset=["date"])
        frame = frame.set_index("date").sort_index()
        frame = frame.loc[pd.to_datetime(start) : pd.to_datetime(end)]
        frame.attrs["series_path"] = series_path
        frame.attrs["series_name"] = doc.get("series_name")
        return frame

    def fetch_series(
        self,
        series_path: str,
        start: str,
        end: str,
        ttl: timedelta | None = timedelta(days=1),
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch one DB.NOMICS series and return a DataFrame indexed by date."""
        key = stable_cache_key("dbnomics", series_path, start, end)
        return self.cache.get_dataframe(
            "dbnomics",
            key,
            lambda: self._fetch_from_api(series_path, start, end),
            ttl=ttl,
            force_refresh=force_refresh,
            metadata={
                "provider": "dbnomics",
                "series_path": series_path,
                "start": start,
                "end": end,
            },
        )

    def fetch_series_group(
        self,
        series_paths: dict[str, str] | list[str],
        start: str,
        end: str,
        ttl: timedelta | None = timedelta(days=1),
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch multiple DB.NOMICS series and return a wide value matrix."""
        items = (
            series_paths.items()
            if isinstance(series_paths, dict)
            else ((path, path) for path in series_paths)
        )
        frames = []
        for label, series_path in items:
            frame = self.fetch_series(
                series_path,
                start=start,
                end=end,
                ttl=ttl,
                force_refresh=force_refresh,
            )
            frames.append(frame["value"].rename(label))
        return pd.concat(frames, axis=1).sort_index()
