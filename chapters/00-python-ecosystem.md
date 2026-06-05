# Setup and Python Ecosystem

This module defines the reproducibility contract for the course. It prepares the working environment, explains the repository structure, and gives students a stable workflow for running notebooks and building the Jupyter Book.

## Expected outcome

By the end of this module, students should be able to:

- clone the repository;
- install Python 3.12 or newer with `pyenv`;
- create the project environment with `uv`;
- open JupyterLab;
- run the class notebooks;
- understand the basic structure of the Jupyter Book;
- handle API credentials without committing secrets;
- validate the book build before sharing changes.

## Included notes

- reproducible computational finance stack;
- environment validation lab;
- initial repository setup;
- classroom environment setup.

## Technical decisions

| Area | Course decision | Reason |
| --- | --- | --- |
| Python runtime | Python 3.12 or newer through `pyenv` | Keeps the interpreter explicit and modern |
| Dependency manager | `uv` with `pyproject.toml` and `uv.lock` | Provides fast, deterministic environment setup |
| Notebook interface | JupyterLab launched with `uv run` | Prevents accidental use of the wrong Python kernel |
| Book build | Jupyter Book with notebook execution disabled | Avoids build failures from APIs, credentials, and long simulations |
| Data credentials | `.env` or shell variables, with `.env.example` as template | Keeps tokens out of Git history |
| Deep research drafts | `chapters/DR_*.md`, excluded from the build | Allows technical research to be synthesized before publication |

## Recommended next improvements

- add an optional CI/CD workflow for GitHub Pages;
- add notebook output stripping with `pre-commit` and `nbstripout`;
- create reusable data fetchers in `src/` for Banxico, FRED, and Yahoo Finance;
- add a static-output publication policy for interactive Plotly and `ipywidgets` notebooks.

## Class sequence

1. Start with the repository structure and explain where notebooks, images, source helpers, and book configuration live.
2. Install or select Python 3.12 or newer with `pyenv`.
3. Create the project environment and install dependencies with `uv sync`.
4. Launch JupyterLab with `uv run jupyter lab`.
5. Run the environment validation lab.
6. Review API credential handling with `.env.example`.
7. Build the book with `make book`.
8. Record package versions before moving into market data notebooks.

## In-class practice

Students should reproduce the `pyenv` and `uv` setup locally, launch JupyterLab from the project environment, and compare their package versions against the instructor environment. Any mismatch that affects execution should be recorded before the data modules begin.

## Module checkpoint

The checkpoint is a short environment report with Python version, operating system, package versions, and the `uv` commands used to launch JupyterLab and build the book.

## Base command

```bash
pyenv install 3.12.12
pyenv local 3.12.12
uv sync
uv run jupyter lab
```

## Validation command

```bash
git diff --check
make book
```
