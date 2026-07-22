"""Generate versioned real-data snapshots for the published book.

The default mode refreshes official macro-financial series from Banxico SIE,
DB.NOMICS, and Yahoo Finance. ``--derived-only`` performs no network calls and
rebuilds only the two derived classroom panels from already-versioned source
CSVs. Publication builds read the snapshots without live calls or secrets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.banxico import BanxicoClient
from src.dbnomics import DBnomicsClient
from src.market_data import DEFAULT_NASDAQ_STOCK_TICKERS, rate_to_decimal

SNAPSHOT_DIR = Path("data/snapshots")
SNAPSHOT_START = "2018-01-01"
SNAPSHOT_END = "2026-06-05"
NASDAQ_SNAPSHOT_START = "2021-01-01"
MODULE_1_ANALYSIS_START = "2021-01-01"
MODULE_1_ANALYSIS_END = "2025-06-30"

BANXICO_SERIES = {
    "usd_mxn": "SF43718",
    "cetes_28d": "SF60633",
    "tiie_28d": "SF60648",
    "policy_rate": "SF61745",
    "udi": "SP68257",
}

DBNOMICS_SERIES = {
    "mexico_cpi": "IMF/CPI/M.MX.PCPI_IX",
    "us_10y": "FED/H15/RIFLGFCY10_N.B",
}

OUTPUT_FILENAMES = {
    "banxico_daily": "banxico_daily.csv",
    "dbnomics_daily": "dbnomics_daily.csv",
    "nasdaq_stock_panel": "nasdaq_stock_panel.csv",
    "official_price_panel": "official_price_panel.csv",
    "official_macro_panel": "official_macro_panel.csv",
}

PRICE_PANEL_COLUMNS = [
    "usd_mxn",
    "udi",
    "cetes_28d_carry",
    "tiie_28d_carry",
    "policy_rate_carry",
]

MACRO_PANEL_COLUMNS = [
    "banxico_target_rate",
    "cetes_28d",
    "tiie_28d",
    "usd_mxn",
    "udi",
    "mexico_cpi",
    "mexico_inflation",
    "us_10y",
]


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE pairs from .env without printing secrets."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def fetch_banxico_panel(start: str, end: str) -> pd.DataFrame:
    """Fetch the provider-dated Banxico source panel."""
    client = BanxicoClient()
    panel = client.fetch_series_group(
        list(BANXICO_SERIES.values()),
        start=start,
        end=end,
        ttl=None,
        force_refresh=True,
    )
    rename_map = {series_id: label for label, series_id in BANXICO_SERIES.items()}
    return panel.rename(columns=rename_map).sort_index()


def fetch_dbnomics_panel(start: str, end: str) -> pd.DataFrame:
    """Fetch the provider-dated DB.NOMICS source panel."""
    client = DBnomicsClient()
    return client.fetch_series_group(
        DBNOMICS_SERIES,
        start=start,
        end=end,
        ttl=None,
        force_refresh=True,
    )


def fetch_nasdaq_stock_panel(start: str, end: str) -> pd.DataFrame:
    """Fetch adjusted daily closes for selected NASDAQ-listed equities."""
    tickers = list(DEFAULT_NASDAQ_STOCK_TICKERS)
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data = yf.download(
        tickers,
        start=start,
        end=end_exclusive,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    prices = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    prices = prices.reindex(columns=tickers).sort_index().dropna(how="all")
    if prices.empty:
        raise ValueError("NASDAQ stock snapshot is empty after Yahoo Finance download.")
    return prices.rename_axis("date")


def _validate_frame(
    frame: pd.DataFrame,
    *,
    label: str,
    required_columns: Sequence[str],
) -> None:
    if frame.empty:
        raise ValueError(f"{label} must not be empty")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{label} must use a DatetimeIndex")
    if frame.index.hasnans:
        raise ValueError(f"{label} contains invalid dates")
    if frame.index.has_duplicates:
        raise ValueError(f"{label} contains duplicate dates")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{label} dates must be sorted in increasing order")
    missing_columns = sorted(set(required_columns).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"{label} is missing required columns: {missing_columns}")


def carry_index(
    rate_percent: pd.Series,
    *,
    day_count_basis: int = 360,
    base_level: float = 100.0,
) -> pd.Series:
    """Compound the previous published annual rate once per elapsed calendar day.

    At an observation date ``t``, the index uses the rate known at ``t-1`` and
    raises its one-day accrual factor to
    ``delta_days = (t - (t-1)).days``. Missing rates after the first publication
    carry the last published rate. ``build_price_panel`` supplies a daily index,
    so intervening rate publications are incorporated before the result is
    sampled on joint observed-level dates.
    """
    if not isinstance(rate_percent, pd.Series):
        raise TypeError("rate_percent must be a pandas Series")
    if not isinstance(rate_percent.index, pd.DatetimeIndex):
        raise TypeError("rate_percent must use a DatetimeIndex")
    if rate_percent.index.has_duplicates:
        raise ValueError("rate_percent contains duplicate dates")
    if not rate_percent.index.is_monotonic_increasing:
        raise ValueError("rate_percent dates must be sorted in increasing order")
    if day_count_basis <= 0:
        raise ValueError("day_count_basis must be positive")
    if base_level <= 0:
        raise ValueError("base_level must be positive")

    rates = pd.to_numeric(rate_percent, errors="coerce")
    first_valid_date = rates.first_valid_index()
    if first_valid_date is None:
        raise ValueError("rate_percent must contain at least one numeric observation")

    effective_rates = rates.loc[first_valid_date:].ffill()
    delta_days = effective_rates.index.to_series().diff().dt.days.astype(float)
    daily_factors = 1 + (effective_rates.shift(1) / 100) / day_count_basis
    accrual_factors = daily_factors.pow(delta_days)
    accrual_factors.iloc[0] = 1.0
    if (accrual_factors <= 0).any():
        raise ValueError("rate path implies a non-positive carry accrual factor")

    values = base_level * accrual_factors.cumprod()
    values.name = rate_percent.name
    return values


def build_price_panel(banxico: pd.DataFrame) -> pd.DataFrame:
    """Build observed USD/MXN and UDI levels with calendar-day carry indexes.

    Market levels are never forward-filled. Rows are retained only when both
    USD/MXN and UDI were observed. Carry is accrued daily using the last
    published rate and then sampled on those same observed-level dates.
    """
    _validate_frame(
        banxico,
        label="banxico",
        required_columns=list(BANXICO_SERIES),
    )
    calendar = pd.date_range(banxico.index.min(), banxico.index.max(), freq="D")
    calendar_rates = banxico[["cetes_28d", "tiie_28d", "policy_rate"]].reindex(calendar)

    carry = pd.DataFrame(index=calendar)
    for rate_column, carry_column in (
        ("cetes_28d", "cetes_28d_carry"),
        ("tiie_28d", "tiie_28d_carry"),
        ("policy_rate", "policy_rate_carry"),
    ):
        carry[carry_column] = carry_index(calendar_rates[rate_column])

    panel = banxico[["usd_mxn", "udi"]].join(carry, how="left")
    panel = panel.loc[:, PRICE_PANEL_COLUMNS].dropna(how="any")
    if panel.empty:
        raise ValueError("official price panel is empty after observed-level alignment")
    return panel.rename_axis("date")


def build_macro_panel(banxico: pd.DataFrame, dbnomics: pd.DataFrame) -> pd.DataFrame:
    """Build the publication macro schema without calendar filling."""
    _validate_frame(
        banxico,
        label="banxico",
        required_columns=list(BANXICO_SERIES),
    )
    _validate_frame(
        dbnomics,
        label="dbnomics",
        required_columns=list(DBNOMICS_SERIES),
    )
    banxico_monthly = banxico.resample("ME").last()
    dbnomics_monthly = dbnomics.resample("ME").last()
    cpi = dbnomics_monthly["mexico_cpi"]
    macro = pd.DataFrame(
        {
            "banxico_target_rate": rate_to_decimal(
                banxico_monthly["policy_rate"], source_unit="percentage_points"
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
    macro = macro.loc[:, MACRO_PANEL_COLUMNS].dropna(how="any")
    if macro.empty:
        raise ValueError("official macro panel is empty after monthly alignment")
    return macro.rename_axis("date")


def _snapshot_paths() -> dict[str, Path]:
    return {key: SNAPSHOT_DIR / filename for key, filename in OUTPUT_FILENAMES.items()}


def read_snapshot(
    path: Path,
    *,
    label: str,
    required_columns: Sequence[str],
) -> pd.DataFrame:
    """Read and validate a versioned source snapshot."""
    if not path.exists():
        raise FileNotFoundError(f"required versioned source snapshot not found: {path}")
    frame = pd.read_csv(path, index_col="date", parse_dates=True).sort_index()
    _validate_frame(frame, label=label, required_columns=required_columns)
    return frame


def write_snapshot(frame: pd.DataFrame, path: Path) -> None:
    """Write one CSV atomically so a failed process cannot leave a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary_path, index=True, date_format="%Y-%m-%d")
    temporary_path.replace(path)


def _write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _read_existing_metadata(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(
            "--derived-only requires the versioned metadata.json so source vintage is preserved"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("generated_at"):
        raise ValueError("versioned metadata.json must contain the source vintage 'generated_at'")
    return payload


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def _coverage(frame: pd.DataFrame) -> dict[str, str]:
    return {
        "start": frame.index.min().strftime("%Y-%m-%d"),
        "end": frame.index.max().strftime("%Y-%m-%d"),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_metadata(
    frames: dict[str, pd.DataFrame],
    *,
    source_generated_at: str,
    derived_generated_at: str,
    rebuilt_from_versioned_sources: bool,
    previous_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build complete provenance, schema, coverage, and transformation metadata."""
    paths = _snapshot_paths()
    previous_metadata = previous_metadata or {}
    metadata: dict[str, object] = dict(previous_metadata)
    metadata.update(
        {
            "generated_at": source_generated_at,
            "derived_generated_at": derived_generated_at,
            "rebuilt_from_versioned_sources": rebuilt_from_versioned_sources,
            "generation_mode": (
                "derived-only" if rebuilt_from_versioned_sources else "network-refresh"
            ),
            "snapshot_start": previous_metadata.get("snapshot_start", SNAPSHOT_START),
            "snapshot_end": previous_metadata.get("snapshot_end", SNAPSHOT_END),
            "nasdaq_stock_snapshot_start": previous_metadata.get(
                "nasdaq_stock_snapshot_start", NASDAQ_SNAPSHOT_START
            ),
            "nasdaq_stock_snapshot_end": previous_metadata.get(
                "nasdaq_stock_snapshot_end", SNAPSHOT_END
            ),
            "nasdaq_stock_snapshot_generated_at": previous_metadata.get(
                "nasdaq_stock_snapshot_generated_at", source_generated_at
            ),
            "module_1_analysis_window": {
                "start": MODULE_1_ANALYSIS_START,
                "end": MODULE_1_ANALYSIS_END,
                "inclusive": True,
                "note": (
                    "Module 1 slices wider source snapshots to this requested window; "
                    "actual boundaries follow each source calendar and frequency."
                ),
            },
            "banxico_series": BANXICO_SERIES,
            "dbnomics_series": DBNOMICS_SERIES,
            "nasdaq_stock_tickers": DEFAULT_NASDAQ_STOCK_TICKERS,
            "outputs": {key: _display_path(path) for key, path in paths.items()},
            "row_counts": {key: int(len(frame)) for key, frame in frames.items()},
            "columns": {key: list(frame.columns) for key, frame in frames.items()},
            "output_coverage": {key: _coverage(frame) for key, frame in frames.items()},
            "sha256": {key: _sha256(path) for key, path in paths.items()},
            "carry_index_methodology": {
                "base_level": 100.0,
                "day_count_basis": 360,
                "formula": "I_d = I_(d-1) * (1 + (y_(d-1) / 100) / 360)",
                "stored_interval_formula": (
                    "I_t / I_s = product from d=s+1 to t of (1 + (y_(d-1) / 100) / 360)"
                ),
                "calendar_day_policy": (
                    "carry is compounded once per elapsed calendar day on an internal "
                    "daily grid, then sampled on joint USD/MXN and UDI observation dates"
                ),
                "missing_rate_policy": (
                    "use the last published annualized rate after the first valid observation"
                ),
                "level_observation_policy": (
                    "USD/MXN and UDI levels are never forward-filled; retain dates where both "
                    "levels were observed"
                ),
                "scope_limit": (
                    "synthetic classroom indexes; not observed prices, total-return indexes, "
                    "or investable strategies"
                ),
            },
            "methodology_breaks": {
                "tiie_28d": {
                    "effective_date": "2025-01-01",
                    "series_id": BANXICO_SERIES["tiie_28d"],
                    "note": (
                        "Banco de Mexico changed the 28-day TIIE methodology from "
                        "submitted bank quotes to a market-transaction-based method. "
                        "The synthetic carry path preserves, rather than removes, this break."
                    ),
                    "source": (
                        "https://www.banxico.org.mx/SieInternet/"
                        "consultarDirectorioInternetAction.do?accion=consultarCuadro"
                        "&idCuadro=CF111&locale=en"
                    ),
                }
            },
            "observation_policies": {
                "banxico_daily": "provider-dated observations; no calendar filling",
                "dbnomics_daily": "provider-dated observations; no calendar filling",
                "nasdaq_stock_panel": "provider trading dates; no forward filling",
                "official_price_panel": (
                    "observed USD/MXN and UDI intersection plus derived carry indexes"
                ),
                "official_macro_panel": "monthly last observations; no forward filling",
            },
            "notes": [
                "CSV snapshots are committed for reproducible publication builds.",
                "Banxico and DB.NOMICS source rates are percentage points; macro rates are decimals.",
                "Carry compounds the previous known rate once per calendar day under a 360-day convention.",
                "NASDAQ adjusted closes come from Yahoo Finance through yfinance for classroom EDA.",
                "Snapshot provenance does not itself establish redistribution or downstream-use rights.",
            ],
        }
    )
    return metadata


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--derived-only",
        action="store_true",
        help=(
            "Rebuild derived panels from versioned CSVs without network calls or source rewrites."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Refresh source and derived snapshots, or safely rebuild only derived panels."""
    args = _parse_args(argv)
    paths = _snapshot_paths()
    metadata_path = SNAPSHOT_DIR / "metadata.json"
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    previous_metadata: dict[str, object] = {}
    if args.derived_only:
        previous_metadata = _read_existing_metadata(metadata_path)
        banxico = read_snapshot(
            paths["banxico_daily"],
            label="banxico_daily",
            required_columns=list(BANXICO_SERIES),
        )
        dbnomics = read_snapshot(
            paths["dbnomics_daily"],
            label="dbnomics_daily",
            required_columns=list(DBNOMICS_SERIES),
        )
        nasdaq_stock_panel = read_snapshot(
            paths["nasdaq_stock_panel"],
            label="nasdaq_stock_panel",
            required_columns=list(DEFAULT_NASDAQ_STOCK_TICKERS),
        )
    else:
        load_dotenv()
        banxico = fetch_banxico_panel(SNAPSHOT_START, SNAPSHOT_END)
        dbnomics = fetch_dbnomics_panel(SNAPSHOT_START, SNAPSHOT_END)
        nasdaq_stock_panel = fetch_nasdaq_stock_panel(NASDAQ_SNAPSHOT_START, SNAPSHOT_END)

    price_panel = build_price_panel(banxico)
    macro_panel = build_macro_panel(banxico, dbnomics)
    _validate_frame(
        price_panel,
        label="official_price_panel",
        required_columns=PRICE_PANEL_COLUMNS,
    )
    _validate_frame(
        macro_panel,
        label="official_macro_panel",
        required_columns=MACRO_PANEL_COLUMNS,
    )

    if not args.derived_only:
        write_snapshot(banxico.rename_axis("date"), paths["banxico_daily"])
        write_snapshot(dbnomics.rename_axis("date"), paths["dbnomics_daily"])
        write_snapshot(nasdaq_stock_panel.rename_axis("date"), paths["nasdaq_stock_panel"])
    write_snapshot(price_panel, paths["official_price_panel"])
    write_snapshot(macro_panel, paths["official_macro_panel"])

    frames = {
        "banxico_daily": banxico,
        "dbnomics_daily": dbnomics,
        "nasdaq_stock_panel": nasdaq_stock_panel,
        "official_price_panel": price_panel,
        "official_macro_panel": macro_panel,
    }
    rebuilt_at = datetime.now(timezone.utc).isoformat()
    source_generated_at = (
        str(previous_metadata["generated_at"]) if args.derived_only else rebuilt_at
    )
    metadata = build_metadata(
        frames,
        source_generated_at=source_generated_at,
        derived_generated_at=rebuilt_at,
        rebuilt_from_versioned_sources=args.derived_only,
        previous_metadata=previous_metadata,
    )
    _write_json(metadata, metadata_path)

    print(json.dumps(metadata["row_counts"], indent=2, sort_keys=True))
    print(json.dumps(metadata["columns"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
