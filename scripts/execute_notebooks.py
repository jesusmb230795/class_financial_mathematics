"""Execute notebook inventories and fail on hidden semantic quality issues."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from notebook_manifest import REPO_ROOT, NotebookEntry, entries, parse_statuses
from validate_notebooks import FAILURE_TEXT_PATTERNS


DEFAULT_OUTPUT_DIR = Path("/private/tmp/class_financial_mathematics_notebooks")
DEFAULT_REPORT = REPO_ROOT / "_build" / "reports" / "notebook-execution.json"


def source_text(value: str | list[str]) -> str:
    """Normalize a notebook text field."""
    return "".join(value) if isinstance(value, list) else value


def execution_seconds(cell: nbformat.NotebookNode) -> float:
    """Return recorded iopub execution duration for one cell."""
    execution = cell.get("metadata", {}).get("execution", {})
    started = execution.get("iopub.execute_input")
    finished = execution.get("iopub.status.idle")
    if not started or not finished:
        return 0.0
    try:
        start = datetime.fromisoformat(started.replace("Z", "+00:00"))
        end = datetime.fromisoformat(finished.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    return max(0.0, (end - start).total_seconds())


def inspect_outputs(
    notebook: nbformat.NotebookNode,
    entry: NotebookEntry,
) -> tuple[list[str], dict]:
    """Inspect executed outputs for stderr, error text, size, and timing."""
    issues = []
    output_bytes = 0
    stderr_cells = []
    failure_cells = []
    max_cell_seconds = 0.0
    max_cell_index = None

    for index, cell in enumerate(notebook.cells, start=1):
        duration = execution_seconds(cell)
        if duration > max_cell_seconds:
            max_cell_seconds = duration
            max_cell_index = index

        for output in cell.get("outputs", []):
            output_bytes += len(
                json.dumps(output, ensure_ascii=False, default=str).encode()
            )
            output_type = output.get("output_type")
            if output_type == "error":
                issues.append(
                    f"cell {index} returned {output.get('ename')}: "
                    f"{output.get('evalue')}"
                )

            fragments = []
            if "text" in output:
                fragments.append(source_text(output["text"]))
            if "evalue" in output:
                fragments.append(str(output["evalue"]))
            for value in output.get("data", {}).values():
                if isinstance(value, (str, list)):
                    fragments.append(source_text(value))
            output_text = "\n".join(fragments)

            if output_type == "stream" and output.get("name") == "stderr":
                stderr_cells.append(index)
            if any(pattern in output_text for pattern in FAILURE_TEXT_PATTERNS):
                failure_cells.append(index)

    if stderr_cells:
        issues.append(f"unexpected stderr in cells {sorted(set(stderr_cells))}")
    if failure_cells:
        issues.append(
            f"failure-like output text in cells {sorted(set(failure_cells))}"
        )
    if output_bytes > entry.max_output_bytes:
        issues.append(
            f"outputs use {output_bytes} bytes; budget is {entry.max_output_bytes}"
        )

    metrics = {
        "output_bytes": output_bytes,
        "stderr_cells": sorted(set(stderr_cells)),
        "failure_text_cells": sorted(set(failure_cells)),
        "max_cell_seconds": round(max_cell_seconds, 3),
        "max_cell_index": max_cell_index,
    }
    return issues, metrics


def execute_entry(entry: NotebookEntry, output_dir: Path) -> dict:
    """Execute one notebook and return its structured result."""
    started = time.perf_counter()
    notebook = nbformat.read(entry.path, as_version=4)
    os.environ["DATA_MODE"] = entry.data_mode
    os.environ.setdefault("RUN_INTERACTIVE_WIDGETS", "0")

    try:
        client = NotebookClient(
            notebook,
            timeout=entry.timeout_seconds,
            kernel_name=entry.kernel,
            allow_errors=False,
            record_timing=True,
        )
        executed = client.execute(cwd=str(REPO_ROOT))
        issues, metrics = inspect_outputs(executed, entry)
        destination = output_dir / entry.status / entry.path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(executed, destination)
        status = "passed" if not issues else "failed"
        error = None
    except Exception as exc:  # execution errors become structured evidence
        issues = [f"{type(exc).__name__}: {exc}"]
        metrics = {
            "output_bytes": 0,
            "stderr_cells": [],
            "failure_text_cells": [],
            "max_cell_seconds": 0.0,
            "max_cell_index": None,
        }
        status = "failed"
        error = str(exc)

    return {
        "path": entry.relative_path,
        "status": status,
        "issues": issues,
        "error": error,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        **metrics,
    }


def main(argv: list[str] | None = None) -> int:
    """Execute a selected manifest inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status",
        default="published,labs",
        help="Comma-separated manifest statuses",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    statuses = parse_statuses(args.status)
    selected = [
        entry for entry in entries() if entry.status in statuses and entry.execute
    ]
    results = []
    for index, entry in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {entry.relative_path}", flush=True)
        result = execute_entry(entry, args.output_dir)
        results.append(result)
        if result["status"] == "failed":
            for issue in result["issues"]:
                print(f"  FAILED: {issue}", flush=True)

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "statuses": sorted(statuses),
        "total": len(results),
        "passed": sum(result["status"] == "passed" for result in results),
        "failed": sum(result["status"] == "failed" for result in results),
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        f"Notebook execution: {report['passed']} passed, "
        f"{report['failed']} failed; report={args.report}"
    )
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
