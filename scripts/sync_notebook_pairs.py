"""Regenerate manifest-listed notebooks from canonical Jupytext percent sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from collections import defaultdict, deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jupytext
import nbformat


STATUS_ORDER = ("published", "labs", "legacy")
AUTHORITATIVE_NOTEBOOK_METADATA = {
    "finmath",
    "jupytext",
    "kernelspec",
    "language_info",
}
KERNELSPEC = {
    "display_name": "Python 3 (ipykernel)",
    "language": "python",
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
JUPYTEXT_CELL_METADATA = {"lines_to_next_cell"}


@dataclass(frozen=True)
class NotebookEntry:
    """One executable notebook record resolved from the manifest."""

    relative_path: str
    status: str
    kernel: str
    timeout_seconds: int
    max_output_bytes: int
    data_mode: str
    execute: bool
    live_capable: bool


@dataclass(frozen=True)
class PairSyncResult:
    """Result of synchronizing one notebook pair."""

    relative_path: str
    changed: bool


@dataclass(frozen=True)
class SyncReport:
    """Aggregate synchronization report."""

    results: tuple[PairSyncResult, ...]

    @property
    def changed_paths(self) -> tuple[str, ...]:
        """Return manifest paths written during this run."""
        return tuple(result.relative_path for result in self.results if result.changed)

    @property
    def unchanged_paths(self) -> tuple[str, ...]:
        """Return manifest paths already synchronized."""
        return tuple(result.relative_path for result in self.results if not result.changed)


def source_text(cell: nbformat.NotebookNode) -> str:
    """Return cell source as text."""
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def normalize_source(cell: nbformat.NotebookNode) -> str:
    """Normalize serialization-only line-ending differences."""
    source = source_text(cell).replace("\r\n", "\n").replace("\r", "\n")
    return source.rstrip("\n")


def cell_fingerprint(cell: nbformat.NotebookNode) -> tuple[str, str]:
    """Return the identity used to carry an existing cell id forward."""
    return cell.cell_type, normalize_source(cell)


def deterministic_cell_id(
    relative_path: str,
    cell: nbformat.NotebookNode,
    occurrence: int,
) -> str:
    """Generate a stable UUID5 id for a canonical cell."""
    source_digest = hashlib.sha256(normalize_source(cell).encode("utf-8")).hexdigest()
    name = (
        f"finmath-notebook-cell://{relative_path}"
        f"#{cell.cell_type}:{source_digest}:{occurrence}"
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Merge metadata recursively, with canonical overlay values taking precedence."""
    merged = deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(merged.get(key), Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def contract_metadata(
    entry: NotebookEntry,
    existing: nbformat.NotebookNode | None,
    canonical: nbformat.NotebookNode,
) -> dict[str, Any]:
    """Preserve useful notebook metadata and impose the publication contract."""
    existing_metadata = existing.metadata if existing is not None else {}
    metadata = deepcopy(dict(existing_metadata))
    canonical_metadata = {
        key: value
        for key, value in canonical.metadata.items()
        if key not in AUTHORITATIVE_NOTEBOOK_METADATA
    }
    metadata = deep_merge(metadata, canonical_metadata)
    metadata["kernelspec"] = {**KERNELSPEC, "name": entry.kernel}
    metadata["language_info"] = deepcopy(LANGUAGE_INFO)
    metadata["jupytext"] = deepcopy(JUPYTEXT_METADATA)
    metadata["finmath"] = {
        "status": entry.status,
        "data_mode": entry.data_mode,
        "live_capable": entry.live_capable,
        "timeout_seconds": entry.timeout_seconds,
    }
    return metadata


def existing_cells_by_fingerprint(
    existing: nbformat.NotebookNode | None,
) -> dict[tuple[str, str], deque[nbformat.NotebookNode]]:
    """Index existing cells so ids survive canonical cell reordering."""
    indexed: dict[tuple[str, str], deque[nbformat.NotebookNode]] = defaultdict(deque)
    if existing is None:
        return indexed
    for cell in existing.cells:
        indexed[cell_fingerprint(cell)].append(cell)
    return indexed


def canonical_cells(
    relative_path: str,
    canonical: nbformat.NotebookNode,
    existing: nbformat.NotebookNode | None,
) -> list[nbformat.NotebookNode]:
    """Build clean canonical cells with stable identifiers."""
    matches = existing_cells_by_fingerprint(existing)
    occurrences: dict[tuple[str, str], int] = defaultdict(int)
    used_ids: set[str] = set()
    cells = []

    for canonical_cell in canonical.cells:
        cell = deepcopy(canonical_cell)
        for key in JUPYTEXT_CELL_METADATA:
            cell.metadata.pop(key, None)

        fingerprint = cell_fingerprint(cell)
        occurrence = occurrences[fingerprint]
        occurrences[fingerprint] += 1
        matching_cell = matches[fingerprint].popleft() if matches[fingerprint] else None

        existing_id = matching_cell.get("id") if matching_cell is not None else None
        if isinstance(existing_id, str) and existing_id and existing_id not in used_ids:
            cell["id"] = existing_id
        else:
            cell["id"] = deterministic_cell_id(relative_path, cell, occurrence)
        used_ids.add(cell["id"])

        if matching_cell is not None and "attachments" in matching_cell:
            cell["attachments"] = deepcopy(matching_cell["attachments"])
        if cell.cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    return cells


def regenerate_notebook(
    entry: NotebookEntry,
    canonical: nbformat.NotebookNode,
    existing: nbformat.NotebookNode | None,
) -> nbformat.NotebookNode:
    """Create a deterministic notebook from canonical Python and manifest state."""
    notebook = nbformat.v4.new_notebook()
    notebook.metadata = contract_metadata(entry, existing, canonical)
    notebook.cells = canonical_cells(entry.relative_path, canonical, existing)
    notebook.nbformat = 4
    notebook.nbformat_minor = 5
    nbformat.validate(notebook)
    return notebook


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load and minimally validate a notebook manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{manifest_path}: schema_version must be 1")
    if not isinstance(manifest.get("defaults"), Mapping):
        raise ValueError(f"{manifest_path}: defaults must be an object")
    return manifest


def parse_statuses(raw: str) -> set[str]:
    """Parse and validate a comma-separated status selection."""
    statuses = {status.strip() for status in raw.split(",") if status.strip()}
    unknown = statuses - set(STATUS_ORDER)
    if unknown:
        raise ValueError(f"unknown notebook statuses: {sorted(unknown)}")
    if not statuses:
        raise ValueError("at least one notebook status is required")
    return statuses


def safe_relative_path(raw_path: str) -> str:
    """Reject absolute or parent-traversing manifest paths."""
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe notebook manifest path: {raw_path!r}")
    return path.as_posix()


def manifest_entries(
    manifest: Mapping[str, Any],
    statuses: set[str],
) -> list[NotebookEntry]:
    """Resolve selected `.ipynb` records while preserving manifest order."""
    defaults = manifest["defaults"]
    overrides = manifest.get("overrides", {})
    live_capable = set(manifest.get("live_capable", []))
    resolved = []
    seen = set()

    for status in STATUS_ORDER:
        if status not in statuses:
            continue
        raw_entries = manifest.get(status, [])
        if not isinstance(raw_entries, list):
            raise ValueError(f"manifest status {status!r} must be a list")
        for raw_path in raw_entries:
            if not isinstance(raw_path, str):
                raise ValueError(f"manifest status {status!r} contains a non-string path")
            relative_path = safe_relative_path(raw_path)
            if Path(relative_path).suffix != ".ipynb":
                continue
            if relative_path in seen:
                raise ValueError(f"duplicate notebook manifest path: {relative_path}")
            seen.add(relative_path)

            settings = {**defaults, **overrides.get(relative_path, {})}
            execute = settings.get("execute", status != "legacy")
            resolved.append(
                NotebookEntry(
                    relative_path=relative_path,
                    status=status,
                    kernel=str(settings["kernel"]),
                    timeout_seconds=int(settings["timeout_seconds"]),
                    max_output_bytes=int(settings["max_output_bytes"]),
                    data_mode=str(settings["data_mode"]),
                    execute=bool(execute),
                    live_capable=relative_path in live_capable,
                )
            )
    return resolved


def resolve_manifest_path(root: Path, manifest_path: Path | None) -> Path:
    """Resolve an injectable manifest path relative to the selected root."""
    if manifest_path is None:
        return root / "notebooks" / "manifest.json"
    if manifest_path.is_absolute():
        return manifest_path
    return root / manifest_path


def sync_entry(root: Path, entry: NotebookEntry) -> PairSyncResult:
    """Synchronize one entry from `.py` to `.ipynb`, without consulting mtimes."""
    ipynb_path = root / entry.relative_path
    py_path = ipynb_path.with_suffix(".py")
    if not py_path.exists():
        raise FileNotFoundError(
            f"{entry.relative_path}: canonical percent source is missing: "
            f"{py_path.relative_to(root)}"
        )

    canonical = jupytext.read(py_path, fmt="py:percent")
    existing = nbformat.read(ipynb_path, as_version=4) if ipynb_path.exists() else None
    notebook = regenerate_notebook(entry, canonical, existing)
    rendered = nbformat.writes(notebook, version=4)
    before = ipynb_path.read_text(encoding="utf-8") if ipynb_path.exists() else None
    changed = rendered != before
    if changed:
        ipynb_path.parent.mkdir(parents=True, exist_ok=True)
        ipynb_path.write_text(rendered, encoding="utf-8")
    return PairSyncResult(relative_path=entry.relative_path, changed=changed)


def sync_notebook_pairs(
    *,
    root: Path,
    manifest_path: Path | None = None,
    statuses: set[str] | None = None,
) -> SyncReport:
    """Synchronize selected manifest entries and return an auditable report."""
    root = root.resolve()
    selected_statuses = set(STATUS_ORDER) if statuses is None else set(statuses)
    if not selected_statuses:
        raise ValueError("at least one notebook status is required")
    unknown = selected_statuses - set(STATUS_ORDER)
    if unknown:
        raise ValueError(f"unknown notebook statuses: {sorted(unknown)}")

    resolved_manifest_path = resolve_manifest_path(root, manifest_path)
    manifest = load_manifest(resolved_manifest_path)
    entries = manifest_entries(manifest, selected_statuses)
    results = tuple(sync_entry(root, entry) for entry in entries)
    return SyncReport(results=results)


def main(
    argv: list[str] | None = None,
    *,
    root: Path | None = None,
    manifest_path: Path | None = None,
) -> int:
    """Run deterministic notebook-pair synchronization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status",
        default="published,labs,legacy",
        help="Comma-separated manifest statuses to synchronize",
    )
    args = parser.parse_args(argv)

    selected_root = Path(__file__).resolve().parents[1] if root is None else root
    try:
        report = sync_notebook_pairs(
            root=selected_root,
            manifest_path=manifest_path,
            statuses=parse_statuses(args.status),
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Notebook pair synchronization failed: {exc}", file=sys.stderr)
        return 2

    print(
        f"Synchronized {len(report.results)} notebook pairs; "
        f"changed {len(report.changed_paths)}"
    )
    for relative_path in report.changed_paths:
        print(f"  {relative_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
