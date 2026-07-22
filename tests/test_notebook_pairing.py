from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import jupytext
import nbformat

from scripts.validate_notebooks import validate_ipynb, validate_jupytext_pair


TEXT_REPRESENTATION = {
    "extension": ".py",
    "format_name": "percent",
    "format_version": "1.3",
    "jupytext_version": "1.19.3",
}


def example_notebook() -> nbformat.NotebookNode:
    notebook = nbformat.v4.new_notebook(
        metadata={
            "finmath": {"status": "published"},
            "jupytext": {
                "formats": "ipynb,py:percent",
                "text_representation": deepcopy(TEXT_REPRESENTATION),
            },
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        }
    )
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# Pairing Test\n\n"
            "## Lesson summary\n\n"
            "A paired notebook fixture.\n\n"
            "## Learning objectives\n\n"
            "- Preserve canonical cell content."
        ),
        nbformat.v4.new_code_cell(
            "value = 1\n",
            metadata={"tags": ["setup", "hide-input"]},
        ),
        nbformat.v4.new_markdown_cell(
            "## Checkpoint exercise\n\nChange `value`.",
            metadata={"tags": ["exercise"]},
        ),
        nbformat.v4.new_markdown_cell(
            "```{dropdown} Suggested answer\nUse `value = 2`.\n```",
            metadata={"tags": ["solution"]},
        ),
    ]
    return notebook


def write_pair(tmp_path: Path) -> tuple[Path, nbformat.NotebookNode]:
    notebook = example_notebook()
    ipynb_path = tmp_path / "lesson.ipynb"
    py_path = ipynb_path.with_suffix(".py")
    nbformat.write(notebook, ipynb_path)
    py_path.write_text(jupytext.writes(notebook, fmt="py:percent"))
    return ipynb_path, notebook


def test_matching_percent_pair_passes_full_notebook_validation(tmp_path: Path) -> None:
    ipynb_path, _ = write_pair(tmp_path)

    assert validate_ipynb(ipynb_path) == []


def test_pair_validation_reports_source_tags_type_and_metadata_drift(
    tmp_path: Path,
) -> None:
    ipynb_path, notebook = write_pair(tmp_path)
    notebook.cells[0] = nbformat.v4.new_raw_cell(notebook.cells[0].source)
    notebook.cells[1].source = "value = 2\n"
    notebook.cells[2].metadata.tags = ["exercise", "hide-input"]
    notebook.cells[3].metadata.slideshow = {"slide_type": "slide"}

    errors = validate_jupytext_pair(ipynb_path, notebook)
    message = "\n".join(errors)

    assert "cell 1 type" in message
    assert "cell 2 source differs" in message
    assert "cell 3 tags" in message
    assert "cell 4 relevant metadata differs" in message
    assert "regenerate the .ipynb from the canonical .py source" in message
    assert "uv run jupytext --to ipynb --output" in message


def test_pair_validation_reports_cell_count_drift(tmp_path: Path) -> None:
    ipynb_path, notebook = write_pair(tmp_path)
    notebook.cells.pop()

    errors = validate_jupytext_pair(ipynb_path, notebook)

    assert any("cell count 3 != canonical 4" in error for error in errors)


def test_ipynb_text_representation_is_optional(tmp_path: Path) -> None:
    ipynb_path, notebook = write_pair(tmp_path)
    del notebook.metadata.jupytext["text_representation"]

    errors = validate_jupytext_pair(ipynb_path, notebook)

    assert errors == []


def test_text_representation_core_fields_are_strict(tmp_path: Path) -> None:
    ipynb_path, notebook = write_pair(tmp_path)
    notebook.metadata.jupytext.text_representation["format_version"] = "1.2"
    notebook.metadata.jupytext.text_representation["jupytext_version"] = ""

    errors = validate_jupytext_pair(ipynb_path, notebook)
    message = "\n".join(errors)

    assert "text_representation.format_version must be '1.3'" in message
    assert "text_representation.jupytext_version must be a non-empty string" in message
