"""Compatibility entry point for canonical notebook-pair synchronization.

Notebook normalization used to edit ``.ipynb`` files directly. Percent-format
``.py`` files are now authoritative, so direct callers are delegated to the
same deterministic synchronizer used by ``make sync-jupytext``.
"""

from __future__ import annotations

from pathlib import Path

import nbformat

from notebook_manifest import NotebookEntry, entries


KERNELSPEC = {
    "display_name": "Python 3 (ipykernel)",
    "language": "python",
    "name": "python3",
}
LANGUAGE_INFO = {
    "codemirror_mode": {"name": "ipython", "version": 3},
    "file_extension": ".py",
    "mimetype": "text/x-python",
    "name": "python",
    "nbconvert_exporter": "python",
    "pygments_lexer": "ipython3",
}
JUPYTEXT_METADATA = {
    "formats": "ipynb,py:percent",
    "text_representation": {
        "extension": ".py",
        "format_name": "percent",
        "format_version": "1.3",
        "jupytext_version": "1.19.3",
    },
}


def source_text(cell: nbformat.NotebookNode) -> str:
    """Return a cell source as text."""
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def replace_text_in_code(notebook: nbformat.NotebookNode, old: str, new: str) -> None:
    """Replace an exact code fragment wherever it occurs."""
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = source_text(cell)
        if old in source:
            cell.source = source.replace(old, new)


def compact_time_series_one(notebook: nbformat.NotebookNode) -> None:
    """Keep the first lesson focused and leave ARIMA/GARCH to later notebooks."""
    if len(notebook.cells) <= 60:
        return

    original = notebook.cells
    notebook.cells = original[:22] + original[61:78] + [original[130]]
    notebook.cells[0].source = """# Financial Time Series I

Module: Quantitative Methods and Financial Time Series

## Lesson summary

This notebook introduces financial time series through observed prices, returns,
white-noise diagnostics, and the random-walk benchmark. Detailed ARIMA and
volatility workflows are developed in the later notebooks of Module 2.

## Learning objectives
- Convert price data into time-indexed financial series.
- Distinguish persistent price levels from more stable differences and returns.
- Interpret white-noise diagnostics for a fitted mean model.
- Explain why random-walk-like levels require transformation before modeling.

## Lesson flow
1. Load and prepare a versioned official Banxico price-like panel.
2. Create a focused USD/MXN level and log-return series.
3. Evaluate residuals against the white-noise benchmark.
4. Compare levels, first differences, and log returns.
"""

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = source_text(cell)
        if source.startswith("import pandas as pd"):
            cell.source = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scs
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.arima.model import ARIMA

from src.market_data import official_price_panel

pd.set_option("display.max_columns", 80)
"""
        if source.startswith("observed_returns ="):
            cell.source = """observed_returns = (
    apple_data.set_index("date")["log_returns"]
    .dropna()
    .rename("usd_mxn_log_return")
)

baseline_model = ARIMA(
    observed_returns.to_numpy(),
    order=(1, 0, 1),
    trend="n",
).fit()
residual_series = pd.Series(
    baseline_model.resid,
    index=observed_returns.index,
    name="arima_residual",
)

_ = tsplot(residual_series, lags=30)
"""


def fix_known_notebook_issues(
    notebook: nbformat.NotebookNode,
    path: Path,
) -> None:
    """Apply evidence-backed fixes found during the notebook audit."""
    if path.name == "2.1.time_series_1.ipynb":
        compact_time_series_one(notebook)

    replace_text_in_code(
        notebook,
        'display(HTML(fig.to_html(include_plotlyjs="cdn", full_html=False)))',
        "display(fig)",
    )
    replace_text_in_code(
        notebook,
        "df_lags.corr().applymap(\"{:.2f}\".format)",
        'df_lags.corr().map("{:.2f}".format)',
    )
    replace_text_in_code(
        notebook,
        "log_returns.hist(ax=axes[1, 0], bins=40)",
        "log_returns.plot.hist(ax=axes[1, 0], bins=40, alpha=0.45)",
    )
    replace_text_in_code(
        notebook,
        'warnings.filterwarnings("ignore")',
        "# Warnings remain visible so model and API issues cannot pass silently.",
    )
    replace_text_in_code(
        notebook,
        'best = candidate_results.dropna(subset=["aic"]).iloc[0]',
        """best = (
    candidate_results.loc[candidate_results["converged"]]
    .dropna(subset=["aic"])
    .iloc[0]
)""",
    )
    replace_text_in_code(
        notebook,
        """model = ARIMA(
    returns,
    order=order,""",
        """model = ARIMA(
    returns.to_numpy(),
    order=order,""",
    )

    all_sources = "\n".join(source_text(cell) for cell in notebook.cells)
    if "HTML(" not in all_sources:
        replace_text_in_code(
            notebook,
            "from IPython.display import HTML, display",
            "from IPython.display import display",
        )


def add_cell_tags(
    notebook: nbformat.NotebookNode,
    status: str,
) -> None:
    """Tag setup, live-data, interactive, and legacy cells."""
    first_code_seen = False
    for cell in notebook.cells:
        metadata = cell.setdefault("metadata", {})
        tags = list(dict.fromkeys(metadata.get("tags", [])))
        source = source_text(cell)

        if cell.cell_type == "code" and not first_code_seen:
            first_code_seen = True
            tags.extend(tag for tag in ("setup", "hide-input") if tag not in tags)
        if cell.cell_type == "code" and "interact(" in source and "interactive" not in tags:
            tags.append("interactive")
        if (
            cell.cell_type == "code"
            and "DATA_MODE" in source
            and "live" in source.lower()
            and "live-data" not in tags
        ):
            tags.append("live-data")
        if status == "legacy" and "legacy" not in tags:
            tags.append("legacy")

        if tags:
            metadata["tags"] = tags


def normalize_entry(entry: NotebookEntry) -> bool:
    """Normalize one notebook and return whether it changed."""
    if entry.path.suffix != ".ipynb":
        return False

    before = entry.path.read_text()
    notebook = nbformat.read(entry.path, as_version=4)

    fix_known_notebook_issues(notebook, entry.path)
    notebook.metadata["kernelspec"] = KERNELSPEC
    notebook.metadata["language_info"] = LANGUAGE_INFO
    notebook.metadata["jupytext"] = JUPYTEXT_METADATA
    notebook.metadata["finmath"] = {
        "status": entry.status,
        "data_mode": entry.data_mode,
        "live_capable": entry.live_capable,
        "timeout_seconds": entry.timeout_seconds,
    }

    add_cell_tags(notebook, entry.status)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []

    nbformat.validate(notebook)
    rendered = nbformat.writes(notebook)
    if rendered != before:
        entry.path.write_text(rendered)
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    """Preserve the old command while enforcing canonical ``.py`` authority."""
    from sync_notebook_pairs import main as sync_main

    return sync_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
