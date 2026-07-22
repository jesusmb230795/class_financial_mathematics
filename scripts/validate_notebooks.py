"""Validate notebook structure, metadata, semantics, and authoring policy."""

from __future__ import annotations

import ast
import re
import shlex
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jupytext
import nbformat
from IPython.core.inputtransformer2 import TransformerManager
from nbformat.validator import NotebookValidationError


ALLOWED_TAGS = {
    "exercise",
    "hide-cell",
    "hide-input",
    "hide-output",
    "interactive",
    "legacy",
    "live-data",
    "long-running",
    "raises-exception",
    "remove-cell",
    "remove-input",
    "remove-output",
    "setup",
    "skip-execution",
    "solution",
}
REQUIRED_HEADINGS = {
    "# ",
    "## Lesson summary",
    "## Learning objectives",
}
FORBIDDEN_CODE_PATTERNS = {
    r"(?m)^\s*(?:%pip|!pip|pip\s+(?:install|freeze)\b)": (
        "manage dependencies with uv outside notebook cells"
    ),
    r"warnings\.filterwarnings\(\s*[\"']ignore[\"']": (
        "do not suppress every warning; handle expected warnings explicitly"
    ),
    r'include_plotlyjs\s*=\s*[\"\']cdn[\"\']': (
        "use Plotly MIME output instead of a CDN-dependent HTML fragment"
    ),
}
FAILURE_TEXT_PATTERNS = (
    "No module named",
    "ModuleNotFoundError",
    "ImportError",
    "command not found",
    "Traceback (most recent call last)",
)
EXPECTED_JUPYTEXT_FORMATS = "ipynb,py:percent"
EXPECTED_TEXT_REPRESENTATION = {
    "extension": ".py",
    "format_name": "percent",
    "format_version": "1.3",
}
IGNORED_PAIR_CELL_METADATA = {
    "lines_to_next_cell",
}


def source_text(cell: nbformat.NotebookNode) -> str:
    """Return cell source as plain text."""
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def notebook_status(notebook: nbformat.NotebookNode) -> str:
    """Return the normalized repository status."""
    return str(notebook.metadata.get("finmath", {}).get("status", ""))


def normalize_cell_source(cell: nbformat.NotebookNode) -> str:
    """Normalize line endings and a serialization-only final newline."""
    source = source_text(cell).replace("\r\n", "\n").replace("\r", "\n")
    return source.rstrip("\n")


def normalize_metadata_value(value: Any) -> Any:
    """Return a deterministic plain-Python representation of metadata."""
    if isinstance(value, Mapping):
        return {
            str(key): normalize_metadata_value(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [normalize_metadata_value(item) for item in value]
    return value


def cell_tags(cell: nbformat.NotebookNode) -> tuple[str, ...]:
    """Return cell tags in semantic rather than serialized order."""
    return tuple(sorted(str(tag) for tag in cell.get("metadata", {}).get("tags", [])))


def relevant_cell_metadata(cell: nbformat.NotebookNode) -> dict[str, Any]:
    """Return authored cell metadata, excluding Jupytext serialization details."""
    metadata = cell.get("metadata", {})
    return {
        str(key): normalize_metadata_value(value)
        for key, value in metadata.items()
        if key != "tags" and key not in IGNORED_PAIR_CELL_METADATA
    }


def regeneration_hint(ipynb_path: Path, py_path: Path) -> str:
    """Return an explicit command that regenerates the notebook from canonical Python."""
    command = " ".join(
        (
            "uv run jupytext --to ipynb --output",
            shlex.quote(str(ipynb_path)),
            shlex.quote(str(py_path)),
        )
    )
    return f"regenerate the .ipynb from the canonical .py source with `{command}`"


def validate_jupytext_metadata(
    path: Path,
    notebook: nbformat.NotebookNode,
    *,
    hint: str,
    require_text_representation: bool,
) -> list[str]:
    """Validate the percent-pair declaration on one notebook representation."""
    errors = []
    metadata = notebook.metadata.get("jupytext", {})
    if metadata.get("formats") != EXPECTED_JUPYTEXT_FORMATS:
        errors.append(
            f"{path}: metadata.jupytext.formats must be {EXPECTED_JUPYTEXT_FORMATS!r}; {hint}"
        )

    representation = metadata.get("text_representation")
    if not isinstance(representation, Mapping):
        if require_text_representation:
            errors.append(
                f"{path}: metadata.jupytext.text_representation is required; {hint}"
            )
        return errors

    for key, expected_value in EXPECTED_TEXT_REPRESENTATION.items():
        if representation.get(key) != expected_value:
            errors.append(
                f"{path}: metadata.jupytext.text_representation.{key} "
                f"must be {expected_value!r}; {hint}"
            )

    jupytext_version = representation.get("jupytext_version")
    if not isinstance(jupytext_version, str) or not jupytext_version.strip():
        errors.append(
            f"{path}: metadata.jupytext.text_representation.jupytext_version "
            f"must be a non-empty string; {hint}"
        )
    return errors


def validate_jupytext_pair(
    path: Path,
    notebook: nbformat.NotebookNode,
) -> list[str]:
    """Validate strict cell parity against the canonical percent-format Python source."""
    py_path = path.with_suffix(".py")
    hint = regeneration_hint(path, py_path)
    errors = validate_jupytext_metadata(
        path,
        notebook,
        hint=hint,
        require_text_representation=False,
    )

    if not py_path.exists():
        errors.append(f"{path}: paired Jupytext .py source is missing; {hint}")
        return errors

    try:
        canonical = jupytext.read(py_path)
        nbformat.validate(canonical)
    except NotebookValidationError as exc:
        errors.append(f"{py_path}: invalid canonical Jupytext notebook schema: {exc}")
        return errors
    except Exception as exc:
        errors.append(f"{py_path}: cannot read canonical Jupytext source: {exc}")
        return errors

    errors.extend(
        validate_jupytext_metadata(
            py_path,
            canonical,
            hint=hint,
            require_text_representation=True,
        )
    )

    differences = []
    if len(notebook.cells) != len(canonical.cells):
        differences.append(
            f"cell count {len(notebook.cells)} != canonical {len(canonical.cells)}"
        )

    for index, (ipynb_cell, py_cell) in enumerate(
        zip(notebook.cells, canonical.cells, strict=False),
        start=1,
    ):
        if ipynb_cell.cell_type != py_cell.cell_type:
            differences.append(
                f"cell {index} type {ipynb_cell.cell_type!r} "
                f"!= canonical {py_cell.cell_type!r}"
            )
        if normalize_cell_source(ipynb_cell) != normalize_cell_source(py_cell):
            differences.append(f"cell {index} source differs")
        if cell_tags(ipynb_cell) != cell_tags(py_cell):
            differences.append(
                f"cell {index} tags {cell_tags(ipynb_cell)!r} "
                f"!= canonical {cell_tags(py_cell)!r}"
            )
        if relevant_cell_metadata(ipynb_cell) != relevant_cell_metadata(py_cell):
            differences.append(f"cell {index} relevant metadata differs")

    if differences:
        errors.append(
            f"{path}: Jupytext pair differs from canonical {py_path}: "
            f"{'; '.join(differences)}; {hint}"
        )
    return errors


def validate_code_cell(path: Path, index: int, source: str) -> list[str]:
    """Validate Python syntax after translating supported IPython magics."""
    errors = []
    for pattern, message in FORBIDDEN_CODE_PATTERNS.items():
        if re.search(pattern, source):
            errors.append(f"{path}: code cell {index}: {message}")

    try:
        transformed = TransformerManager().transform_cell(source)
        ast.parse(transformed)
    except (SyntaxError, ValueError) as exc:
        errors.append(f"{path}: code cell {index} has invalid Python/IPython syntax: {exc}")

    if len(source.splitlines()) > 120:
        errors.append(
            f"{path}: code cell {index} exceeds 120 lines; move reusable logic to src/"
        )
    return errors


def validate_ipynb(path: Path) -> list[str]:
    """Validate an `.ipynb` file and return error messages."""
    errors = []
    try:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
    except NotebookValidationError as exc:
        return [f"{path}: invalid notebook schema: {exc}"]
    except Exception as exc:
        return [f"{path}: cannot read notebook: {exc}"]

    status = notebook_status(notebook)
    if status not in {"published", "labs", "legacy"}:
        errors.append(f"{path}: missing or invalid metadata.finmath.status")

    kernelspec = notebook.metadata.get("kernelspec", {})
    if kernelspec.get("name") != "python3":
        errors.append(f"{path}: kernelspec.name must be python3")
    if notebook.metadata.get("language_info", {}).get("name") != "python":
        errors.append(f"{path}: language_info.name must be python")

    errors.extend(validate_jupytext_pair(path, notebook))

    markdown = "\n".join(
        source_text(cell) for cell in notebook.cells if cell.cell_type == "markdown"
    )
    for heading in REQUIRED_HEADINGS:
        if heading == "# ":
            if not re.search(r"(?m)^#\s+\S", markdown):
                errors.append(f"{path}: missing a level-one lesson title")
        elif heading.lower() not in markdown.lower():
            errors.append(f"{path}: missing required heading {heading!r}")

    notebook_tags = set()
    first_code_tags = set()
    first_code_seen = False
    for index, cell in enumerate(notebook.cells, start=1):
        tags = set(cell.get("metadata", {}).get("tags", []))
        notebook_tags.update(tags)
        unknown_tags = tags - ALLOWED_TAGS
        if unknown_tags:
            errors.append(f"{path}: cell {index} has unknown tags {sorted(unknown_tags)}")

        if cell.cell_type != "code":
            continue
        if not first_code_seen:
            first_code_seen = True
            first_code_tags = tags
        if cell.get("execution_count") is not None:
            errors.append(f"{path}: code cell {index} has execution_count")
        if cell.get("outputs"):
            errors.append(f"{path}: code cell {index} has saved outputs")
        errors.extend(validate_code_cell(path, index, source_text(cell)))

    if first_code_seen and not {"setup", "hide-input"}.issubset(first_code_tags):
        errors.append(f"{path}: first code cell must use setup and hide-input tags")
    if status != "legacy":
        for required_tag in ("exercise", "solution"):
            if required_tag not in notebook_tags:
                errors.append(f"{path}: missing required {required_tag!r} cell tag")
    elif "legacy" not in notebook_tags:
        errors.append(f"{path}: legacy notebook cells must be tagged legacy")

    return errors


def validate_myst(path: Path) -> list[str]:
    """Validate a Markdown source file against the repository format policy."""
    errors = []
    text = path.read_text()
    if "```{code-cell}" in text:
        errors.append(
            f"{path}: executable Python cells belong in .ipynb sources; "
            "keep .md files for narrative content and static examples"
        )

    try:
        notebook = jupytext.read(path)
        nbformat.validate(notebook)
    except NotebookValidationError as exc:
        return [f"{path}: invalid MyST notebook schema: {exc}"]
    except Exception as exc:
        return [f"{path}: cannot read as MyST notebook: {exc}"]
    return errors


def main(argv: list[str]) -> int:
    """Validate all supplied notebook sources."""
    errors = []
    for raw_path in argv:
        path = Path(raw_path)
        if not path.exists():
            errors.append(f"{path}: source path does not exist")
        elif path.suffix == ".ipynb":
            errors.extend(validate_ipynb(path))
        elif path.suffix == ".md":
            errors.extend(validate_myst(path))

    if errors:
        print("Notebook validation failed:")
        for error in errors:
            print(f"  {error}")
        return 1
    print(f"Notebook validation passed: {len(argv)} sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
