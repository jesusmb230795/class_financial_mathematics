from __future__ import annotations

import json
import os
import uuid
from copy import deepcopy
from pathlib import Path

import jupytext
import nbformat
import pytest

from scripts.sync_notebook_pairs import (
    JUPYTEXT_METADATA,
    LANGUAGE_INFO,
    main,
    parse_statuses,
    sync_notebook_pairs,
)


DEFAULTS = {
    "kernel": "python3",
    "timeout_seconds": 120,
    "max_output_bytes": 1_048_576,
    "data_mode": "offline",
}


def canonical_notebook(
    *,
    code: str = "value = 2\n",
) -> nbformat.NotebookNode:
    notebook = nbformat.v4.new_notebook(
        metadata={
            "jupytext": deepcopy(JUPYTEXT_METADATA),
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
        }
    )
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# Canonical lesson\n\nCanonical narrative.",
            metadata={"tags": ["lesson"]},
        ),
        nbformat.v4.new_code_cell(
            code,
            metadata={"tags": ["setup", "hide-input"]},
        ),
    ]
    return notebook


def write_percent_source(
    root: Path,
    relative_ipynb: str,
    notebook: nbformat.NotebookNode,
) -> Path:
    py_path = (root / relative_ipynb).with_suffix(".py")
    py_path.parent.mkdir(parents=True, exist_ok=True)
    py_path.write_text(jupytext.writes(notebook, fmt="py:percent"), encoding="utf-8")
    return py_path


def write_manifest(
    root: Path,
    *,
    published: list[str] | None = None,
    labs: list[str] | None = None,
    legacy: list[str] | None = None,
    live_capable: list[str] | None = None,
    overrides: dict[str, dict] | None = None,
) -> Path:
    manifest = {
        "schema_version": 1,
        "defaults": DEFAULTS,
        "published": published or [],
        "labs": labs or [],
        "legacy": legacy or [],
        "live_capable": live_capable or [],
        "overrides": overrides or {},
    }
    path = root / "notebooks" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_sync_uses_python_as_authority_and_is_idempotent(tmp_path: Path) -> None:
    relative_path = "notebooks/course/lesson.ipynb"
    canonical = canonical_notebook()
    py_path = write_percent_source(tmp_path, relative_path, canonical)

    existing = canonical_notebook(code="value = 1\n")
    existing.metadata = {
        "custom": {"preserve": True},
        "finmath": {"status": "stale"},
        "jupytext": {"formats": "stale"},
        "kernelspec": {"name": "stale"},
        "language_info": {"name": "stale"},
    }
    existing.cells[0]["id"] = "preserved-narrative-id"
    existing.cells[1]["id"] = "discarded-code-id"
    existing.cells[1]["execution_count"] = 9
    existing.cells[1]["outputs"] = [
        nbformat.v4.new_output("stream", name="stdout", text="stale\n")
    ]
    ipynb_path = tmp_path / relative_path
    nbformat.write(existing, ipynb_path)
    manifest_path = write_manifest(
        tmp_path,
        published=[relative_path],
        live_capable=[relative_path],
    )

    os.utime(py_path, (1, 1))
    os.utime(ipynb_path, (2_000_000_000, 2_000_000_000))
    first_report = sync_notebook_pairs(
        root=tmp_path,
        manifest_path=manifest_path,
        statuses={"published"},
    )

    synchronized = nbformat.read(ipynb_path, as_version=4)
    first_render = ipynb_path.read_text(encoding="utf-8")
    generated_code_id = synchronized.cells[1].id

    assert first_report.changed_paths == (relative_path,)
    assert synchronized.cells[1].source == "value = 2"
    assert synchronized.cells[0].id == "preserved-narrative-id"
    assert synchronized.cells[1].id != "discarded-code-id"
    assert uuid.UUID(generated_code_id).version == 5
    assert synchronized.cells[1].execution_count is None
    assert synchronized.cells[1].outputs == []
    assert synchronized.metadata.custom == {"preserve": True}
    assert synchronized.metadata.kernelspec.name == "python3"
    assert synchronized.metadata.language_info == LANGUAGE_INFO
    assert synchronized.metadata.jupytext == JUPYTEXT_METADATA
    assert synchronized.metadata.finmath == {
        "status": "published",
        "data_mode": "offline",
        "live_capable": True,
        "timeout_seconds": 120,
    }

    second_report = sync_notebook_pairs(
        root=tmp_path,
        manifest_path=manifest_path,
        statuses={"published"},
    )

    assert second_report.changed_paths == ()
    assert second_report.unchanged_paths == (relative_path,)
    assert ipynb_path.read_text(encoding="utf-8") == first_render
    assert nbformat.read(ipynb_path, as_version=4).cells[1].id == generated_code_id


def test_sync_preserves_ids_by_type_and_source_across_reordering(tmp_path: Path) -> None:
    relative_path = "notebooks/labs/reordered.ipynb"
    canonical = canonical_notebook()
    canonical.cells = [canonical.cells[1], canonical.cells[0]]
    write_percent_source(tmp_path, relative_path, canonical)

    existing = canonical_notebook()
    existing.cells[0]["id"] = "markdown-id"
    existing.cells[1]["id"] = "code-id"
    ipynb_path = tmp_path / relative_path
    ipynb_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(existing, ipynb_path)
    manifest_path = write_manifest(tmp_path, labs=[relative_path])

    sync_notebook_pairs(
        root=tmp_path,
        manifest_path=manifest_path,
        statuses={"labs"},
    )
    synchronized = nbformat.read(ipynb_path, as_version=4)

    assert synchronized.cells[0].cell_type == "code"
    assert synchronized.cells[0].id == "code-id"
    assert synchronized.cells[1].cell_type == "markdown"
    assert synchronized.cells[1].id == "markdown-id"


def test_sync_creates_missing_ipynb_with_deterministic_uuid5_ids(tmp_path: Path) -> None:
    relative_path = "notebooks/legacy/new.ipynb"
    write_percent_source(tmp_path, relative_path, canonical_notebook())
    manifest_path = write_manifest(
        tmp_path,
        legacy=[relative_path],
        overrides={relative_path: {"data_mode": "live", "execute": False}},
    )

    report = sync_notebook_pairs(
        root=tmp_path,
        manifest_path=manifest_path,
        statuses={"legacy"},
    )
    notebook = nbformat.read(tmp_path / relative_path, as_version=4)
    first_ids = [cell.id for cell in notebook.cells]

    assert report.changed_paths == (relative_path,)
    assert all(uuid.UUID(cell_id).version == 5 for cell_id in first_ids)
    assert notebook.metadata.finmath.data_mode == "live"

    second_report = sync_notebook_pairs(
        root=tmp_path,
        manifest_path=manifest_path,
        statuses={"legacy"},
    )
    assert second_report.changed_paths == ()
    assert [cell.id for cell in nbformat.read(tmp_path / relative_path, 4).cells] == first_ids


def test_cli_status_syncs_only_selected_entries_and_reports_changes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    published_path = "notebooks/course/published.ipynb"
    lab_path = "notebooks/labs/lab.ipynb"
    write_percent_source(tmp_path, published_path, canonical_notebook())
    write_percent_source(tmp_path, lab_path, canonical_notebook())
    manifest_path = write_manifest(
        tmp_path,
        published=[published_path],
        labs=[lab_path],
    )

    result = main(
        ["--status", "labs"],
        root=tmp_path,
        manifest_path=manifest_path,
    )
    output = capsys.readouterr().out

    assert result == 0
    assert not (tmp_path / published_path).exists()
    assert (tmp_path / lab_path).exists()
    assert "Synchronized 1 notebook pairs; changed 1" in output
    assert lab_path in output
    assert published_path not in output


def test_manifest_markdown_entries_are_not_treated_as_pairs(tmp_path: Path) -> None:
    markdown_path = "notebooks/course/narrative.md"
    (tmp_path / markdown_path).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / markdown_path).write_text("# Narrative", encoding="utf-8")
    manifest_path = write_manifest(tmp_path, published=[markdown_path])

    report = sync_notebook_pairs(
        root=tmp_path,
        manifest_path=manifest_path,
        statuses={"published"},
    )

    assert report.results == ()


def test_status_and_manifest_paths_are_validated() -> None:
    assert parse_statuses("published,labs") == {"published", "labs"}
    with pytest.raises(ValueError, match="unknown notebook statuses"):
        parse_statuses("published,unknown")
    with pytest.raises(ValueError, match="at least one notebook status"):
        sync_notebook_pairs(root=Path.cwd(), statuses=set())
