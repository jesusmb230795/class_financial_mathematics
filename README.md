# Financial Mathematics Class

Repository of the financial mathematics class for students of the actuarial sciences degree at UNAM

## Jupyter Book

This repository is organized as a Jupyter Book.

### Repository scope

Keep this repository focused on the buildable book and its reproducible support code:

- `_config.yml`, `_config.outputs.yml`, and `_toc.yml` define the book build and navigation.
- `intro.md`, `chapters/`, `notebooks/class/`, `references.md`, and `references.bib` are book content.
- `src/` contains reusable finance, data, and modeling helpers used by notebooks.
- `scripts/` and `img/generated/` support validation and visual assets.
- `.env.example`, `pyproject.toml`, `uv.lock`, and `Makefile` support local setup and reproducible builds.

Instructor-only assessment files, private readings, obsolete PDFs, raw research drafts, local API keys, downloaded datasets, and generated build/cache directories should stay outside the repository or in a private companion repository. A public version can keep the book source, reproducible notebooks, code helpers, generated images, and citation files as long as no licensed readings or credentials are included.

### Source format policy

Use Markdown first for the book surface. Chapter introductions, roadmaps, glossaries, references, and narrative-only notebook pages should stay as `.md`.

Use `.ipynb` when a page contains executable Python cells intended to produce rendered outputs such as tables, plots, dashboards, or numerical diagnostics. Notebook sources should not commit saved outputs or execution counts; publication builds execute the notebooks and cache outputs under `_build/.jupyter_cache/`.

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

The `make book` publication build uses `_config.outputs.yml`, executes reproducible notebooks, and caches rendered outputs under `_build/.jupyter_cache/`. It disables live widgets during execution and renders each dashboard's static fallback view for publication. Notebook sources remain free of saved outputs.

For a faster structural build without notebook execution, run:

```bash
make book-static
```

Legacy notebooks that require live market data, FRED keys, or local `api_keys` modules are excluded from build-time execution until they are converted to reproducible classroom datasets.

### Data access and cache

The book should use real data whenever a reproducible, classroom-safe source is available. Prefer official sources for macroeconomic and Mexico-specific series, documented APIs with stable free tiers for market data, and synthetic or instructor-provided fallbacks only when live access would require secrets or unstable network calls during publication builds.

Reusable provider logic lives in `src/`:

- `src/banxico.py` for Banxico SIE;
- `src/fred.py` for FRED;
- `src/market_data.py` for Yahoo Finance, provider facades, returns, alignment, and classroom panels;
- `src/cache.py` for local tabular and JSON caching.

Recommended real-data sources by use case:

- Mexico official data: INEGI API and Banxico SIE;
- US macro and rates: FRED;
- global macro comparisons: World Bank Open Data, DBnomics, IMF WEO, and OECD Data;
- market data: Finnhub for generous free-tier prototypes, Alpha Vantage for technical indicators, EODHD for global end-of-day history, Financial Modeling Prep for fundamentals, and Yahoo Finance as a convenient educational fallback.

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
make visual-assets-sync
make book
```

For a focused execution check of curated executable notebooks that should run without external data, run:

```bash
make check-curated-notebooks
```

This writes temporary executed notebooks to `/private/tmp/class_financial_mathematics_notebooks`.

### Recommended next improvements

Student-facing module pages should not carry repository maintenance backlog. Keep remaining improvement items here instead.

#### Publication and workflow

- Add an optional CI/CD workflow for GitHub Pages using `uv sync --locked`, `make book`, and Pages artifact upload.
- Add notebook output stripping with `pre-commit` and `nbstripout`.
- Add a lightweight validation checklist for final publication: `git diff --check`, `make visual-assets-sync`, `make book-static`, `make book`, and `make check-curated-notebooks`.

#### Data and dashboards

- Live-data variants for dashboards using Banxico, FRED, and public market sources: add opt-in dashboard panels while keeping deterministic offline defaults for publication builds.
- Add INEGI, World Bank, and DBnomics examples for Mexico, LATAM, and multi-provider macro comparisons.
- Extend the live dashboard provider map beyond Banxico, FRED, and Yahoo Finance into instructor-approved APIs such as Finnhub, Alpha Vantage, EODHD, FMP, INEGI, World Bank, or DBnomics.
- Add provider-specific rate-limit, cache-expiration, and credential-handling notes for live-data notebooks.

#### Module 1 - Markets and Data

- Add a short executable example that compares official macro data, public market prices, and cached classroom panels under the same schema.
- Add a compact Mexican market-infrastructure glossary covering BMV, BIVA, MexDer, CNBV, PIP, Valmer, Indeval, and Asigna.
- Add a dashboard-ready data dictionary template for source, unit, frequency, calendar, currency, license note, and quality flags.

#### Module 2 - Financial Time Series

- Clean repeated dependencies across legacy notebooks.
- Add forecast evaluation examples with train/test splits and rolling-origin validation.
- Add exogenous macro variables or regime-break examples for Mexican rates, FX, or equity returns.
- Connect conditional volatility forecasts with dynamic VaR backtesting.

#### Module 3 - Market Risk

- Add Monte Carlo VaR and Expected Shortfall examples.
- Turn FRTB liquidity-horizon context into a full numerical example.
- Connect tail-risk metrics with mean-variance portfolio construction.

#### Module 4 - Modern Portfolio Theory

- Add constrained optimization.
- Add rebalancing backtesting with turnover penalties.
- Add robust optimization examples with uncertainty sets.

#### Module 5 - Fixed Income

- Add calendar-aware coupon schedule generation from actual settlement dates.
- Connect Banxico UDI and CETES series through the data layer for optional live-data runs.
- Extend the bond dashboard with clean price, dirty price, accrued interest, and settlement controls.
- Add a liability-driven portfolio optimization example after immunization.

#### Module 6 - Term Structure

- Add Banxico and Mexican yield curve data sources.
- Turn monotone-convex interpolation into an executable lab.
- Add a Nelson-Siegel-Svensson notebook that uses the existing term-structure helper.
- Add discount-factor distribution summaries from simulated short-rate paths.
- Connect PCA scenarios to fixed-income portfolio valuation and liabilities.

#### Module 7 - Derivatives

- Add strategy payoff diagrams for spreads, straddles, collars, and covered calls.
- Connect optional live option-chain data through `src/market_data.py`.
- Add an implied-volatility surface fitting lab with arbitrage checks.
- Extend Heston from simulation to stable characteristic-function pricing.
- Add local-volatility and SVI/SSVI surface concepts after implied volatility is stable.
- Add option portfolio hedging and P&L attribution examples.

### Pre-commit hooks

Install the local hooks with:

```bash
make install-pre-commit
```

Run all hooks manually with:

```bash
make pre-commit
```

The notebook hooks clean `.ipynb` outputs and execution counts, then validate `.ipynb` files and enforce that Markdown sources under `notebooks/class/` remain narrative-only.

## License

This project is licensed under the MIT License. See `LICENSE`.

## References

- <https://es.coursera.org/specializations/python-3-programming>
- <https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/video_galleries/video-lectures/>
