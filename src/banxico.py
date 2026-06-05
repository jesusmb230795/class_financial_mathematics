"""Banxico SIE client helpers."""

from __future__ import annotations

import json
import os
from datetime import timedelta
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from src.cache import DataCache, default_cache, stable_cache_key
from src.market_data_quality import banxico_series_catalog

BANXICO_BASE_URL = "https://www.banxico.org.mx/SieAPIRest/service/v1/series"


def _normalize_banxico_value(value: str | int | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    normalized = value.replace(",", "").strip()
    if normalized in {"", "N/E", "N/A"}:
        return None
    return float(normalized)


def parse_banxico_response(payload: dict, series_id: str | None = None) -> pd.DataFrame:
    """Parse a Banxico SIE response into a tidy DataFrame."""
    series_payload = payload.get("bmx", {}).get("series", [])
    rows = []
    for series in series_payload:
        current_id = series.get("idSerie", series_id)
        for item in series.get("datos", []):
            rows.append(
                {
                    "date": pd.to_datetime(item.get("fecha"), dayfirst=True),
                    "series": current_id,
                    "value": _normalize_banxico_value(item.get("dato")),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["series", "value"]).rename_axis("date")
    return frame.sort_values(["date", "series"]).set_index("date")


class BanxicoClient:
    """Minimal Banxico SIE API client with local caching."""

    def __init__(
        self,
        token: str | None = None,
        cache: DataCache | None = None,
        base_url: str = BANXICO_BASE_URL,
        timeout: int = 30,
    ) -> None:
        self.token = token or os.getenv("BANXICO_TOKEN") or os.getenv("BANXICO_API_TOKEN")
        self.cache = cache or default_cache()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _require_token(self) -> str:
        if not self.token:
            raise ValueError(
                "Banxico requests require BANXICO_TOKEN or BANXICO_API_TOKEN in the environment."
            )
        return self.token

    def _request_json(self, url: str) -> dict:
        request = Request(url, headers={"Bmx-Token": self._require_token()})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"Banxico request failed with HTTP {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"Banxico request failed: {exc.reason}") from exc

    def fetch_series(
        self,
        series_id: str,
        start: str,
        end: str,
        ttl: timedelta | None = timedelta(days=1),
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch one Banxico series and return a DataFrame indexed by date."""
        key = stable_cache_key("banxico", series_id, start, end)
        url = f"{self.base_url}/{series_id}/datos/{start}/{end}"

        def fetcher() -> pd.DataFrame:
            payload = self._request_json(url)
            return parse_banxico_response(payload, series_id=series_id)

        return self.cache.get_dataframe(
            "banxico",
            key,
            fetcher,
            ttl=ttl,
            force_refresh=force_refresh,
            metadata={"provider": "banxico", "series_id": series_id, "start": start, "end": end},
        )

    def fetch_series_group(
        self,
        series_ids: list[str],
        start: str,
        end: str,
        ttl: timedelta | None = timedelta(days=1),
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch multiple Banxico series and return a wide value matrix."""
        frames = [
            self.fetch_series(
                series_id,
                start=start,
                end=end,
                ttl=ttl,
                force_refresh=force_refresh,
            )
            for series_id in series_ids
        ]
        tidy = pd.concat(frames).reset_index()
        return tidy.pivot(index="date", columns="series", values="value").sort_index()


def example_banxico_catalog() -> pd.DataFrame:
    """Return the compact Banxico catalog used in the course."""
    return banxico_series_catalog()
