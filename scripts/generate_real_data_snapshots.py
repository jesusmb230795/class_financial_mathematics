"""Generate versioned real-data snapshots for the published book.

The script fetches official macro-financial series from Banxico SIE and
DB.NOMICS, then writes classroom-safe CSV snapshots. Publication builds read
the snapshots so the book can render without live network calls or secrets.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.banxico import BanxicoClient
from src.dbnomics import DBnomicsClient

SNAPSHOT_DIR = Path("data/snapshots")
SNAPSHOT_START = "2018-01-01"
SNAPSHOT_END = "2026-06-05"

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
    client = DBnomicsClient()
    return client.fetch_series_group(
        DBNOMICS_SERIES,
        start=start,
        end=end,
        ttl=None,
        force_refresh=True,
    )


def carry_index(rate_percent: pd.Series, periods_per_year: int = 360) -> pd.Series:
    daily_accrual = pd.to_numeric(rate_percent, errors="coerce").ffill() / 100 / periods_per_year
    return (100 * (1 + daily_accrual).cumprod()).rename(rate_percent.name)


def build_price_panel(banxico: pd.DataFrame) -> pd.DataFrame:
    daily = banxico.reindex(pd.bdate_range(banxico.index.min(), banxico.index.max())).ffill()
    panel = pd.DataFrame(index=daily.index)
    panel["usd_mxn"] = daily["usd_mxn"]
    panel["udi"] = daily["udi"]
    panel["cetes_28d_carry"] = carry_index(daily["cetes_28d"])
    panel["tiie_28d_carry"] = carry_index(daily["tiie_28d"])
    panel["policy_rate_carry"] = carry_index(daily["policy_rate"])
    return panel.dropna().rename_axis("date")


def build_macro_panel(banxico: pd.DataFrame, dbnomics: pd.DataFrame) -> pd.DataFrame:
    banxico_monthly = banxico.resample("ME").last()
    dbnomics_monthly = dbnomics.resample("ME").last()
    macro = pd.DataFrame(
        {
            "banxico_target_rate": banxico_monthly["policy_rate"] / 100,
            "cetes_28d": banxico_monthly["cetes_28d"] / 100,
            "tiie_28d": banxico_monthly["tiie_28d"] / 100,
            "usd_mxn": banxico_monthly["usd_mxn"],
            "udi": banxico_monthly["udi"],
            "mexico_cpi": dbnomics_monthly["mexico_cpi"],
            "mexico_inflation": dbnomics_monthly["mexico_cpi"].pct_change(12, fill_method=None),
            "us_10y": dbnomics_monthly["us_10y"] / 100,
        }
    )
    return macro.dropna(subset=["mexico_inflation", "us_10y"]).rename_axis("date")


def write_snapshot(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=True, date_format="%Y-%m-%d")


def main() -> None:
    load_dotenv()
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    banxico = fetch_banxico_panel(SNAPSHOT_START, SNAPSHOT_END)
    dbnomics = fetch_dbnomics_panel(SNAPSHOT_START, SNAPSHOT_END)
    price_panel = build_price_panel(banxico)
    macro_panel = build_macro_panel(banxico, dbnomics)

    write_snapshot(banxico.rename_axis("date"), SNAPSHOT_DIR / "banxico_daily.csv")
    write_snapshot(dbnomics.rename_axis("date"), SNAPSHOT_DIR / "dbnomics_daily.csv")
    write_snapshot(price_panel, SNAPSHOT_DIR / "official_price_panel.csv")
    write_snapshot(macro_panel, SNAPSHOT_DIR / "official_macro_panel.csv")

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_start": SNAPSHOT_START,
        "snapshot_end": SNAPSHOT_END,
        "banxico_series": BANXICO_SERIES,
        "dbnomics_series": DBNOMICS_SERIES,
        "outputs": {
            "banxico_daily": str(SNAPSHOT_DIR / "banxico_daily.csv"),
            "dbnomics_daily": str(SNAPSHOT_DIR / "dbnomics_daily.csv"),
            "official_price_panel": str(SNAPSHOT_DIR / "official_price_panel.csv"),
            "official_macro_panel": str(SNAPSHOT_DIR / "official_macro_panel.csv"),
        },
        "row_counts": {
            "banxico_daily": int(len(banxico)),
            "dbnomics_daily": int(len(dbnomics)),
            "official_price_panel": int(len(price_panel)),
            "official_macro_panel": int(len(macro_panel)),
        },
        "columns": {
            "banxico_daily": list(banxico.columns),
            "dbnomics_daily": list(dbnomics.columns),
            "official_price_panel": list(price_panel.columns),
            "official_macro_panel": list(macro_panel.columns),
        },
        "notes": [
            "CSV snapshots are committed for reproducible publication builds.",
            "Banxico rates are percentage-point source series; macro snapshot stores them as decimals.",
            "Carry indexes compound official annualized rates with a 360-day convention for classroom return examples.",
        ],
    }
    (SNAPSHOT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(metadata["row_counts"], indent=2, sort_keys=True))
    print(json.dumps(metadata["columns"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
