"""Validate Jupyter notebooks and Markdown notebook sources."""

from __future__ import annotations

import sys
from pathlib import Path

import jupytext
import nbformat
from nbformat.validator import NotebookValidationError


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

    for index, cell in enumerate(notebook.cells, start=1):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            errors.append(f"{path}: code cell {index} has execution_count")
        if cell.get("outputs"):
            errors.append(f"{path}: code cell {index} has saved outputs")

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
    errors = []
    for raw_path in argv:
        path = Path(raw_path)
        if not path.exists():
            continue
        if path.suffix == ".ipynb":
            errors.extend(validate_ipynb(path))
        elif path.suffix == ".md":
            errors.extend(validate_myst(path))

    if errors:
        print("Notebook validation failed:")
        for error in errors:
            print(f"  {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
