# Course Materials

The repository keeps source material and visual resources to support the classes.

## Available PDFs

- `intro_fm_2023_2.pdf`: previous course introduction.
- `material/harvard_portfolio_theory.pdf`: supporting reading on portfolio theory.

## Available images

Images in `img/` are reused in the notes on markets, statistics, and time series. Before publishing the book openly, review licenses and attribution for each visual resource.

## Recommended convention

- Keep original material in `material/`.
- Keep reusable images in `img/`.
- Keep class notebooks in `notebooks/class/`.
- Keep deep research drafts locally in `deep-research/` with the prefix `DR_`, for example `deep-research/DR_02.md`. These drafts are not committed to Git and are not published in the book.
- Move reusable functions to `src/`.
- Keep provider-specific API logic in `src/banxico.py`, `src/fred.py`, and `src/market_data.py`.
- Use `src/cache.py` for local data caching rather than writing ad hoc cache files from notebooks.
- Avoid heavy raw data in Git; prefer download scripts or small samples.
- Follow `chapters/course-standards.md` when turning deep research into module pages and notebooks.
- Keep applied exercises and module checkpoints synchronized in `chapters/exercises.md`.
- Keep grading policy synchronized in `chapters/assessment.md`.
- Add glossary terms to `chapters/glossary.md` when introducing new technical vocabulary.

## Deep research draft convention

Deep research files are treated as source inputs, not final book pages. Store them in `deep-research/` with the `DR_` prefix naming pattern:

```text
deep-research/DR_00.md
deep-research/DR_01.md
deep-research/DR_02.md
deep-research/DR_03.md
deep-research/DR_04.md
deep-research/DR_05.md
deep-research/DR_06.md
deep-research/DR_07.md
```

Files in `deep-research/` and legacy files matching `chapters/DR_*.md` or another chapter draft name containing `DR` are excluded from Git and from the Jupyter Book build until their content is reviewed, condensed, and integrated into the module pages or notebooks.

## Image generation plan

Final visual assets will be generated in a later design pass. Until then:

- do not add new stock-style images as final visual material;
- use existing images only as temporary legacy support where already referenced;
- document any missing visual as a future design task rather than blocking the content build;
- keep prompt drafts and visual style decisions in a dedicated design artifact before generating images.

The later design pass should define a consistent visual system for the book before creating individual image prompts.
