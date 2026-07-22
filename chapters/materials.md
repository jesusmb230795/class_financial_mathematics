# Course Materials

This page records where reusable course inputs belong. It does not list private
readings or files that are not present in the repository.

## Published and source-only material

- Published narrative is the subset of `intro.md`, `chapters/`, and Markdown
  lessons under `notebooks/course/` listed in both `_toc.yml` and `myst.yml`.
- Published executable lessons are the notebook entries listed under
  `published` in `notebooks/manifest.json` and in both TOCs.
- A file under `chapters/` or `notebooks/course/` that is absent from both TOCs
  is reviewed source material, not a published page.
- `notebooks/labs/` is reserved for reviewed executable work awaiting
  promotion; it is empty in the current canonical release.
- Superseded executable sources live in `notebooks/legacy/`; consolidated
  narrative overviews live in `chapters/legacy/`. Both are provenance only.
- Notebook status and execution policy live in `notebooks/manifest.json`.

Do not add private assessment files or licensed readings to the public
repository. A citation in `references.bib` does not grant permission to copy a
source document.

## New text

Stage new source text under `content/incoming/` with `make new-content`.
`content/plan.json` records the canonical module, target path, current status,
legacy-source mapping, known gaps, and next useful text. The intake and
promotion contract is documented in `content/README.md`.

Incoming text may be English or Spanish, but it must record its source language
and be normalized to the book's current English publication surface before
promotion.

## Code and data

- Move reusable functions to `src/`.
- Keep provider-specific API logic in the relevant provider module under
  `src/`, not in copied notebook cells.
- Use `src/cache.py` for local caching.
- Keep small, versioned publication inputs under `data/snapshots/`.
- Keep raw downloads and local caches outside Git.
- Document provider, identifier, unit, frequency, calendar, revision behavior,
  license note, and snapshot/live mode for empirical examples.

## Visual assets

Generated assets are managed through
`img/generated/visual-assets.json` and its module-scoped files under
`img/generated/visual-assets/`.

- do not add new stock-style images as final visual material;
- use existing images only as temporary legacy support where already referenced;
- create generated replacements in the exact `target_path` listed by the manifest;
- generate the deterministic module concept maps with `make visual-assets-concept-maps`;
- update each asset status with `python3 scripts/visual_assets.py mark <asset-id> generated`;
- integrate a new path in the owning Markdown or canonical Jupytext source, then
  use `python3 scripts/visual_assets.py apply --asset <asset-id>` to verify the
  declared placement;
- validate progress with `python3 scripts/visual_assets.py validate`.

Alt text, source or generation status, placement, and visual intent must stay in
the manifest. Run `make visual-assets-validate` before publication.
