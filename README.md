# Financial Mathematics Class

Repository of the financial mathematics class for students of the actuarial sciences degree at UNAM

## Jupyter Book

This repository is organized as a Jupyter Book.

### Local setup

```bash
pyenv install 3.12.12
pyenv local 3.12.12
uv sync
```

If `3.12.12` is not available in your local pyenv build list, install the latest listed `3.12.x` release and update `.python-version` to that exact version.

### Open JupyterLab

```bash
uv run jupyter lab
```

The same command is available through:

```bash
make lab
```

### Build the book

```bash
uv run jupyter-book build .
```

The same command is available through:

```bash
make book
```

The generated site is written to `_build/html`.

Notebook execution is disabled during the book build because several notebooks depend on external data providers. Open the notebooks in JupyterLab when you want to run them interactively.

### Data access and cache

Reusable provider logic lives in `src/`:

- `src/banxico.py` for Banxico SIE;
- `src/fred.py` for FRED;
- `src/market_data.py` for Yahoo Finance, provider facades, returns, alignment, and classroom panels;
- `src/cache.py` for local tabular and JSON caching.

Downloaded data is cached under `.data-cache/`, which is ignored by Git and excluded from the Jupyter Book build.

### API credentials

Some future notebooks may use Banxico or FRED data. Copy the template and fill values locally only when needed:

```bash
cp .env.example .env
```

Never commit real API tokens. `.env` is ignored by Git.

### Validation

Before sharing changes, run:

```bash
git diff --check
make book
```

For curated MyST notebooks that should execute without external data, run:

```bash
make check-curated-notebooks
```

This writes temporary executed notebooks to `/private/tmp/class_financial_mathematics_notebooks`.

### Pre-commit hooks

Install the local hooks with:

```bash
make install-pre-commit
```

Run all hooks manually with:

```bash
make pre-commit
```

The notebook hooks clean `.ipynb` outputs and execution counts, then validate `.ipynb` and MyST notebook sources under `notebooks/class/`.

## References

- <https://es.coursera.org/specializations/python-3-programming>
- <https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/video_galleries/video-lectures/>
