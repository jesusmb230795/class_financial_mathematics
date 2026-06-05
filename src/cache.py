"""Small file-cache utilities for course data fetchers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_CACHE_DIR = Path(".data-cache")


def stable_cache_key(*parts: Any) -> str:
    """Build a stable short cache key from arbitrary JSON-like parts."""
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


class DataCache:
    """Filesystem-backed cache for small tabular and JSON data."""

    def __init__(self, base_dir: str | Path = DEFAULT_CACHE_DIR) -> None:
        self.base_dir = Path(base_dir)

    def _namespace_dir(self, namespace: str) -> Path:
        path = self.base_dir / namespace
        path.mkdir(parents=True, exist_ok=True)
        return path

    def path(self, namespace: str, key: str, suffix: str) -> Path:
        """Return the path for a cache item."""
        clean_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        return self._namespace_dir(namespace) / f"{key}{clean_suffix}"

    def metadata_path(self, namespace: str, key: str) -> Path:
        """Return the metadata path for a cache item."""
        return self.path(namespace, key, ".metadata.json")

    def is_fresh(
        self,
        namespace: str,
        key: str,
        suffix: str,
        ttl: timedelta | None = None,
    ) -> bool:
        """Return whether a cache item exists and is within its TTL."""
        data_path = self.path(namespace, key, suffix)
        if not data_path.exists():
            return False
        if ttl is None:
            return True
        modified_at = datetime.fromtimestamp(data_path.stat().st_mtime, tz=timezone.utc)
        return datetime.now(timezone.utc) - modified_at <= ttl

    def read_dataframe(
        self,
        namespace: str,
        key: str,
        suffix: str = ".parquet",
    ) -> pd.DataFrame:
        """Read a cached DataFrame."""
        data_path = self.path(namespace, key, suffix)
        if suffix.endswith("csv"):
            return pd.read_csv(data_path, index_col=0, parse_dates=True)
        return pd.read_parquet(data_path)

    def write_dataframe(
        self,
        namespace: str,
        key: str,
        frame: pd.DataFrame,
        suffix: str = ".parquet",
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Write a DataFrame and optional metadata to cache."""
        data_path = self.path(namespace, key, suffix)
        if suffix.endswith("csv"):
            frame.to_csv(data_path)
        else:
            frame.to_parquet(data_path)

        if metadata is not None:
            metadata_payload = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                **metadata,
            }
            self.metadata_path(namespace, key).write_text(
                json.dumps(metadata_payload, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
        return data_path

    def read_json(self, namespace: str, key: str) -> dict[str, Any]:
        """Read a cached JSON object."""
        return json.loads(self.path(namespace, key, ".json").read_text(encoding="utf-8"))

    def write_json(
        self,
        namespace: str,
        key: str,
        payload: dict[str, Any],
    ) -> Path:
        """Write a JSON object to cache."""
        data_path = self.path(namespace, key, ".json")
        data_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return data_path

    def get_dataframe(
        self,
        namespace: str,
        key: str,
        fetcher: Callable[[], pd.DataFrame],
        ttl: timedelta | None = None,
        force_refresh: bool = False,
        suffix: str = ".parquet",
        metadata: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Return a cached DataFrame or call `fetcher` and cache its output."""
        if not force_refresh and self.is_fresh(namespace, key, suffix, ttl=ttl):
            return self.read_dataframe(namespace, key, suffix=suffix)
        frame = fetcher()
        self.write_dataframe(namespace, key, frame, suffix=suffix, metadata=metadata)
        return frame


def default_cache() -> DataCache:
    """Return the default project cache."""
    return DataCache(DEFAULT_CACHE_DIR)
