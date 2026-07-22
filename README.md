# Financial Mathematics Class

Repository of the financial mathematics class for students of the actuarial sciences degree at UNAM

## Jupyter Book

This repository is organized as a Jupyter Book.

### Repository scope

Keep this repository focused on the buildable book and its reproducible support code:

- `_config.yml`, `_config.outputs.yml`, and `_toc.yml` define the book build and navigation.
- `intro.md`, the pages from `chapters/` and `notebooks/course/` listed in both
  TOCs, `references.md`, and `references.bib` are the published book sources.
  Files under those directories that are absent from both TOCs remain reviewed
  source material; their directory alone does not make them canonical or
  published.
- `content/plan.json`, `content/debt.json`, `content/incoming/`, and
  `content/templates/` form the editorial intake layer for new text.
- `notebooks/labs/` is reserved for future reviewed-but-unpromoted executable
  work. `notebooks/legacy/` and `chapters/legacy/` preserve superseded sources
  that are not part of the published navigation.
- `src/` contains reusable finance, data, and modeling helpers used by notebooks.
- `scripts/` and `img/generated/` support validation and module-scoped visual assets.
- `.env.example`, `pyproject.toml`, `uv.lock`, and `Makefile` support local setup and reproducible builds.

Instructor-only assessment files, private readings, obsolete PDFs, raw research drafts, local API keys, downloaded datasets, and generated build/cache directories should stay outside the repository or in a private companion repository. A public version can keep the book source, reproducible notebooks, code helpers, generated images, and citation files as long as no licensed readings or credentials are included.

### Source format policy

Use Markdown first for the book surface. Chapter introductions, roadmaps, glossaries, references, and narrative-only notebook pages should stay as `.md`.

Use `.ipynb` when a page contains executable Python cells intended to produce rendered outputs such as tables, plots, dashboards, or numerical diagnostics. Notebook sources should not commit saved outputs or execution counts; publication builds execute the notebooks and cache outputs under `_build/.jupyter_cache/`.

Executable notebook pairs use the percent-format `.py` file as the editable
source and the `.ipynb` file as the generated Jupyter surface. After changing a
paired source, regenerate and validate the pair before publication.

### Add new book text

The book accepts unreviewed text through a staging layer so drafts cannot enter
the published TOC accidentally. Inspect the module gaps and create an intake
file with:

```bash
make content-status
make new-content \
  MODULE=04 \
  SLUG=financial-statement-architecture \
  TITLE="Financial Statement Architecture" \
  TYPE=lesson \
  LANGUAGE=es
```

Paste the source under `## Draft text` in the generated file and run:

```bash
make check-book-content
```

Incoming text may be English or Spanish. The current publication surface is
English; language normalization, citations, notation, data provenance, and
module placement are promotion gates. The complete workflow and templates are
in `content/README.md`; module status and target paths live only in
`content/plan.json`.

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
make book
```

The generated site is written to `_build/html`.

The `make book` publication build uses `_config.outputs.yml`, executes reproducible notebooks, and caches rendered outputs under `_build/.jupyter_cache/`. It disables live widgets during execution and renders each dashboard's static default view from versioned data snapshots. Notebook sources remain free of saved outputs.

For a faster structural build without notebook execution, run:

```bash
make book-static
```

Legacy notebooks that require live market data, DB.NOMICS keys, or local `api_keys` modules are excluded from build-time execution until they are converted to reproducible real-data snapshots or approved live-data paths.

### Review the local Jupyter Book site

Build the rendered-output version first:

```bash
make clean-book-all
make book
```

Then serve the generated HTML from localhost:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory _build/html
```

Open the book at:

```text
http://127.0.0.1:8000/index.html
```

If port `8000` is already in use, choose another local port:

```bash
python3 -m http.server 8001 --bind 127.0.0.1 --directory _build/html
```

Use the full publication gate before uploading or deploying:

```bash
make publish-check
```

That gate checks the dependency lock, editorial plan, TOC parity, visual
assets, notebook manifest and Jupytext pairs, lint, tests, and fresh notebook
execution. It then performs a clean rendered-output build with warnings treated
as errors, creates `.nojekyll`, and validates local HTML links.

### Publish on GitHub Pages

GitHub Pages with GitHub Actions is the recommended hosting path for this book. The repository already contains `.github/workflows/deploy-book.yml`, which installs the locked environment, runs the publication gate, builds `_build/html`, preserves Jupyter Book static assets with `.nojekyll`, and deploys the generated site through the official Pages artifact flow.

Before the first deployment, configure the repository on GitHub:

1. Open **Settings > Pages**.
2. Set **Build and deployment > Source** to **GitHub Actions**.
3. Push or merge the prepared publication commit to `main`, or run the workflow manually from the **Actions** tab.

No generated HTML needs to be committed. The workflow builds the site from source and uploads `_build/html` as the Pages artifact.

### Data access and cache

The published book should use real data. Publication builds should read committed, versioned snapshots generated from real providers rather than fabricated panels. Model simulation can still appear in later methodological lessons when simulation itself is the topic, but it should not be used as a substitute for observed macro, price, return, rate, or dashboard data.

Reusable provider logic lives in `src/`:

- `src/banxico.py` for Banxico SIE;
- `src/dbnomics.py` for DB.NOMICS macro series;
- `src/market_data.py` for Yahoo Finance, provider facades, returns, alignment, and official snapshot panels;
- `data/snapshots/nasdaq_stock_panel.csv` for reproducible NASDAQ stock EDA using Yahoo Finance adjusted daily closes;
- `src/cache.py` for local tabular and JSON caching.

Recommended real-data sources by use case:

- Mexico official data: INEGI API and Banxico SIE;
- US macro and rates: DB.NOMICS;
- global macro comparisons: World Bank Open Data, DB.NOMICS, IMF WEO, and OECD Data;
- market data: Finnhub for generous free-tier prototypes, Alpha Vantage for technical indicators, EODHD for global end-of-day history, Financial Modeling Prep for fundamentals, and Yahoo Finance as a convenient educational source for quick NASDAQ stock EDA snapshots.

Downloaded data is cached under `.data-cache/`, which is ignored by Git and excluded from the Jupyter Book build.
Publication snapshots are stored under `data/snapshots/` and can be regenerated with valid local credentials:

```bash
PYTHONPATH=$PWD uv run python scripts/generate_real_data_snapshots.py
```

### API credentials

Live refreshes and instructor-approved provider checks may use Banxico or DB.NOMICS credentials. Copy the template and fill values locally only when needed:

```bash
cp .env.example .env
```

Never commit real API tokens. `.env` is ignored by Git.

### Validation

Before sharing changes or publishing the book, run:

```bash
git diff --check
make check-book-content
make visual-assets-sync
make book
```

For the full publication gate used by GitHub Actions, run:

```bash
make clean-book-all
make publish-check
```

For a focused fresh execution check of the executable notebooks currently included in the published table of contents, run:

```bash
make check-published-notebooks
```

For an optional wider check of executable notebooks that are still outside the published navigation, run:

```bash
make check-curated-notebooks
```

Notebook execution checks write temporary executed notebooks to `/private/tmp/class_financial_mathematics_notebooks`.

### Maintain and extend the complete book

The published navigation covers all 11 canonical modules, from setup through
the capstone. New text should deepen an existing module or enter through the
documented expansion track; it should not create a parallel numbering scheme.
Legacy sources are retained only for provenance and are never an alternate
student route.

Use these sources instead of keeping a second backlog in the README:

- `chapters/course-roadmap.md` for the student-facing curriculum;
- `content/plan.json` for exact readiness, maintenance needs, source ownership,
  and the next useful text;
- `content/debt.json` for prioritized unresolved consolidation, coverage,
  pedagogy, navigation, and platform work;
- `make content-status` for a live repository report;
- `content/README.md` for intake, review, integration, and promotion.

### Pre-commit hooks

Install the local hooks with:

```bash
make install-pre-commit
```

Run all hooks manually with:

```bash
make pre-commit
```

The notebook hook regenerates `.ipynb` files from the canonical percent-format
`.py` sources, cleaning outputs and execution counts in the process. The
remaining hooks validate notebook structure and enforce the editorial and TOC
contracts for `notebooks/course/`, `notebooks/labs/`, and `notebooks/legacy/`.

## License

This project is licensed under the MIT License. See `LICENSE`.

## References

- <https://es.coursera.org/specializations/python-3-programming>
- <https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/video_galleries/video-lectures/>
