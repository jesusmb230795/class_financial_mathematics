# Jupyter Book 2 migration track

The production site remains on Jupyter Book 1 until the Jupyter Book 2 build
meets the same acceptance contract. A parallel `myst.yml` is versioned so the
new engine can be evaluated without replacing `_config.yml`, `_toc.yml`, or the
current GitHub Pages workflow.

## Why the migration is isolated

Jupyter Book 2 uses the MyST Document Engine and consolidates book
configuration and navigation into `myst.yml`. It also changes generated URLs,
theme behavior, extension support, and build output. The migration therefore
must not silently replace the stable Sphinx-based publication.

## Evaluation command

Run the pinned evaluator in a disposable checkout or branch:

```bash
make book-v2-evaluate
```

The target uses Jupyter Book 2.1.6 independently of the locked Jupyter Book 1
environment. It builds the static HTML candidate in strict CI mode and does
not change the production dependency constraint. Notebook execution remains a
separate production gate rather than being duplicated by this structural
evaluation.

`make check-book-content` validates that the ordered page list and published
module labels in `myst.yml` remain in parity with `_toc.yml`. A successful
structural Book 2 build does not by itself promote the candidate: notebook
execution, visual output, redirects, and deployment still require their own
evidence.

## Promotion gates

Jupyter Book 2 can replace the current build only after all of these checks
pass:

- every `myst.yml` TOC entry resolves and all citations compile;
- `make check-book-content` confirms ordered TOC and module-status parity;
- notebook outputs and Plotly figures render without remote CDN dependencies;
- equations, dropdown solutions, cross-references, logo, and custom styles are
  visually reviewed;
- old public URLs have an explicit redirect map;
- GitHub Pages deploys the new output from pull requests without publishing
  preview branches;
- the executable-notebook report remains warning-free and within output
  budgets;
- the README, Make targets, and CI workflow are updated in the same promotion
  change.

Until those gates are met, `_config.outputs.yml` and `_toc.yml` are the
production source of truth; `myst.yml` is the migration candidate.
