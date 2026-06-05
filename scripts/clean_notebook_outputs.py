"""Clear output cells and execution counts from Jupyter notebooks."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat


def clean_notebook(path: Path) -> bool:
    """Clean one notebook and return whether it changed."""
    notebook = nbformat.read(path, as_version=4)
    changed = False

    for cell in notebook.cells:
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True

    if changed:
        nbformat.write(notebook, path)
    return changed


def main(argv: list[str]) -> int:
    changed_paths = []
    for raw_path in argv:
        path = Path(raw_path)
        if path.suffix != ".ipynb" or not path.exists():
            continue
        if clean_notebook(path):
            changed_paths.append(path)

    if changed_paths:
        print("Cleaned notebook outputs:")
        for path in changed_paths:
            print(f"  {path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
