"""Read and validate the canonical notebook inventory."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "notebooks" / "manifest.json"
CONTENT_SUFFIXES = {".ipynb", ".md"}
STATUS_DIRECTORIES = {
    "published": "course",
    "labs": "labs",
    "legacy": "legacy",
}


@dataclass(frozen=True)
class NotebookEntry:
    """One resolved notebook record."""

    path: Path
    status: str
    kernel: str
    timeout_seconds: int
    max_output_bytes: int
    data_mode: str
    execute: bool
    live_capable: bool

    @property
    def relative_path(self) -> str:
        return self.path.relative_to(REPO_ROOT).as_posix()


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    """Load the versioned JSON manifest."""
    return json.loads(path.read_text())


def entries(path: Path = MANIFEST_PATH) -> list[NotebookEntry]:
    """Return all manifest records with defaults and overrides resolved."""
    manifest = load_manifest(path)
    defaults = manifest["defaults"]
    live_capable = set(manifest.get("live_capable", []))
    overrides = manifest.get("overrides", {})
    resolved = []

    for status in STATUS_DIRECTORIES:
        for relative_path in manifest.get(status, []):
            settings = {**defaults, **overrides.get(relative_path, {})}
            notebook_path = REPO_ROOT / relative_path
            execute = settings.get(
                "execute",
                notebook_path.suffix == ".ipynb" and status != "legacy",
            )
            resolved.append(
                NotebookEntry(
                    path=notebook_path,
                    status=status,
                    kernel=settings["kernel"],
                    timeout_seconds=int(settings["timeout_seconds"]),
                    max_output_bytes=int(settings["max_output_bytes"]),
                    data_mode=settings["data_mode"],
                    execute=bool(execute),
                    live_capable=relative_path in live_capable,
                )
            )
    return resolved


def parse_statuses(raw: str) -> set[str]:
    """Parse a comma-separated status selection."""
    statuses = {value.strip() for value in raw.split(",") if value.strip()}
    unknown = statuses - set(STATUS_DIRECTORIES)
    if unknown:
        raise ValueError(f"Unknown notebook statuses: {sorted(unknown)}")
    return statuses


def selected_entries(statuses: set[str], executable: bool) -> list[NotebookEntry]:
    """Filter manifest entries while preserving manifest order."""
    return [
        entry
        for entry in entries()
        if entry.status in statuses and (not executable or entry.execute)
    ]


def toc_notebook_paths() -> list[str]:
    """Extract notebook content paths from the Jupyter Book TOC."""
    toc_text = (REPO_ROOT / "_toc.yml").read_text()
    return re.findall(r"^\s*-\s+file:\s+(notebooks/[^\s]+)\s*$", toc_text, re.MULTILINE)


def validation_errors() -> list[str]:
    """Validate inventory completeness, paths, numbering, and TOC parity."""
    manifest = load_manifest()
    resolved = entries()
    errors = []

    if manifest.get("schema_version") != 1:
        errors.append("notebooks/manifest.json: schema_version must be 1")

    relative_paths = [entry.relative_path for entry in resolved]
    duplicates = sorted({path for path in relative_paths if relative_paths.count(path) > 1})
    if duplicates:
        errors.append(f"Manifest contains duplicate paths: {duplicates}")

    for entry in resolved:
        if not entry.path.exists():
            errors.append(f"Manifest path does not exist: {entry.relative_path}")
            continue
        expected_directory = STATUS_DIRECTORIES[entry.status]
        if entry.path.parent.name != expected_directory:
            errors.append(
                f"{entry.relative_path}: status {entry.status!r} requires "
                f"notebooks/{expected_directory}/"
            )
        if entry.path.suffix not in CONTENT_SUFFIXES:
            errors.append(f"{entry.relative_path}: unsupported content suffix")
        if entry.execute and entry.path.suffix != ".ipynb":
            errors.append(f"{entry.relative_path}: only .ipynb sources may execute")

    discovered = {
        path.relative_to(REPO_ROOT).as_posix()
        for directory in STATUS_DIRECTORIES.values()
        for path in (REPO_ROOT / "notebooks" / directory).iterdir()
        if path.is_file() and path.suffix in CONTENT_SUFFIXES
    }
    manifested = set(relative_paths)
    for path in sorted(discovered - manifested):
        errors.append(f"Notebook source is missing from manifest: {path}")
    for path in sorted(manifested - discovered):
        errors.append(f"Manifest path is not in a notebook status directory: {path}")

    prefixes: dict[str, list[str]] = {}
    for entry in resolved:
        match = re.match(r"(\d+\.\d+)\.", entry.path.name)
        if match:
            prefixes.setdefault(match.group(1), []).append(entry.relative_path)
    for prefix, paths in sorted(prefixes.items()):
        if len(paths) > 1:
            errors.append(f"Duplicate lesson prefix {prefix}: {paths}")

    published_stems = {
        str(Path(entry.relative_path).with_suffix(""))
        for entry in resolved
        if entry.status == "published"
    }
    toc_paths = set(toc_notebook_paths())
    for path in sorted(published_stems - toc_paths):
        errors.append(f"Published manifest entry is missing from _toc.yml: {path}")
    for path in sorted(toc_paths - published_stems):
        errors.append(f"_toc.yml notebook entry is not published in manifest: {path}")

    return errors


def command_paths(args: argparse.Namespace) -> int:
    """Print selected paths for Makefile and shell use."""
    statuses = parse_statuses(args.status)
    paths = [
        entry.relative_path
        for entry in selected_entries(statuses, executable=args.executable)
    ]
    print(" ".join(paths))
    return 0


def command_validate(_: argparse.Namespace) -> int:
    """Validate the manifest and report actionable errors."""
    errors = validation_errors()
    if errors:
        print("Notebook manifest validation failed:")
        for error in errors:
            print(f"  {error}")
        return 1
    print(f"Notebook manifest is valid: {len(entries())} sources")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    paths_parser = subparsers.add_parser("paths", help="Print selected source paths")
    paths_parser.add_argument(
        "--status",
        default="published,labs,legacy",
        help="Comma-separated statuses",
    )
    paths_parser.add_argument(
        "--executable",
        action="store_true",
        help="Return only entries configured for execution",
    )
    paths_parser.set_defaults(handler=command_paths)

    validate_parser = subparsers.add_parser("validate", help="Validate the manifest")
    validate_parser.set_defaults(handler=command_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the manifest CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Notebook manifest error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
