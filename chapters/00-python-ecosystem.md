# Setup and Python Ecosystem

This module defines the reproducibility contract for the course. It prepares the working environment, explains the repository structure, and gives students a stable workflow for running notebooks and building the Jupyter Book {cite}`kluyver2016jupyter,jupyterbook2025`.

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
| Python runtime | Python 3.12 or newer through `pyenv` {cite}`pyenv2025` | Keeps the interpreter explicit and modern |
| Dependency manager | `uv` with `pyproject.toml` and `uv.lock` {cite}`uv2025` | Provides fast, deterministic environment setup |
| Notebook interface | JupyterLab launched with `uv run` | Prevents accidental use of the wrong Python kernel |
| Book build | `make book` with cached execution for reproducible notebooks | Publishes reader-facing outputs without saving generated notebook results in source files |
| Data credentials | `.env` or shell variables, with `.env.example` as template | Keeps tokens out of Git history |

## Reading sequence

1. Start with the repository structure and explain where notebooks, images, source helpers, and book configuration live.
2. Install or select Python 3.12 or newer with `pyenv`.
3. Create the project environment and install dependencies with `uv sync`.
4. Launch JupyterLab with `uv run jupyter lab`.
5. Run the environment validation lab.
6. Review API credential handling with `.env.example`.
7. Build the publication book with `make book`.
8. Record package versions before moving into market data notebooks.

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
