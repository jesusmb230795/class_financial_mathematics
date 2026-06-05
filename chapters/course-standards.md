# Course Structure Standards

This page defines the structure used to turn deep research drafts into polished Jupyter Book material.

## Module page standard

Each module overview in `chapters/` should follow this structure:

| Section | Purpose |
| --- | --- |
| Title | Clear module name aligned with the course roadmap |
| Opening paragraph | One short statement of the module role in the course |
| Expected outcome | Skills students should have by the end of the module |
| Included notes | Navigable lessons or notebooks already integrated into the book |
| Recommended next improvements | Specific remaining improvements, not generic wishes |
| Class sequence | Instructor-facing order for teaching the module |
| In-class practice | Short applied activity done during class |
| Module checkpoint | Deliverable that proves the student can use the material |

## Notebook standard

Each notebook or executable MyST note should include:

| Section | Purpose |
| --- | --- |
| Title | One H1 only |
| Module | Module name or quoted module line |
| Lesson summary | What the lesson does and why it matters |
| Learning objectives | Four to seven observable outcomes |
| Setup | Imports and deterministic inputs |
| Concept blocks | Definitions, formulas, and interpretation |
| Code workflow | Small, reusable steps rather than one large script |
| Diagnostics | Checks, plots, or model validation where applicable |
| In-class questions | Conceptual prompts for discussion |
| Student deliverable | A concrete artifact students submit |

## Data layer standard

Provider-specific access should stay out of notebooks:

| Module | Responsibility |
| --- | --- |
| `src/cache.py` | Local JSON and tabular cache utilities |
| `src/banxico.py` | Banxico SIE request, parsing, and cache integration |
| `src/fred.py` | FRED request, parsing, and cache integration |
| `src/market_data.py` | Yahoo Finance facade, return transformation, alignment, and classroom panels |

Notebooks should call high-level functions from `src/`, document source assumptions, and avoid duplicating API tokens, request URLs, parsing logic, or cache paths.

## Dashboard standard

Interactive notebooks should:

- run with synthetic or instructor-provided data by default;
- state where live data enters the workflow;
- use `ipywidgets` controls for key assumptions;
- include a chart and an interpretation requirement;
- avoid hidden credentials and network requirements during validation;
- end with a student deliverable linked to a grading checkpoint.

## Assessment and glossary standard

- Keep grading structure and final project requirements in `chapters/assessment.md`.
- Keep module exercises and checkpoints in `chapters/exercises.md`.
- Add new technical vocabulary to `chapters/glossary.md` when introducing a durable concept.
- Link assessment and glossary pages from `_toc.yml` under Course Path.

## Deep research integration workflow

Deep research drafts are stored in `deep-research/` with names like:

```text
deep-research/DR_00.md
deep-research/DR_01.md
deep-research/DR_02.md
```

The drafts are excluded from the Jupyter Book build. They should not be linked directly in `_toc.yml`.
They are also ignored by Git through `.gitignore`, so they remain local source material unless explicitly renamed and promoted.

When a new `DR_*.md` file is added:

1. Read the draft as source material.
2. Extract durable concepts, formulas, datasets, algorithms, exercises, and implementation risks.
3. Synthesize the material into the corresponding module overview.
4. Create or extend practical notebooks in `notebooks/class/`.
5. Move repeated logic into `src/`.
6. Add exercises and checkpoints to `chapters/exercises.md`.
7. Add grading implications to `chapters/assessment.md` when the module changes evaluated work.
8. Add durable technical terms to `chapters/glossary.md`.
9. Build the book with `make book`.
10. Confirm the raw DR title does not appear in `_build/html`.

## Content rules

- Book content is written in English.
- Deep research files remain source inputs, not final student-facing pages.
- Deep research files are not committed or published.
- External API examples must not require secrets during the default book build.
- Notebook execution remains disabled in the default build.
- Any live-data notebook should include a classroom-safe synthetic or static fallback.
- Repeated functions belong in `src/`, not copied across notebooks.
- Student deliverables should require interpretation, not only code execution.
- Final generated images should be handled in a later visual-design pass with a shared style guide and prompts.

## Legacy notebook cleanup standard

Legacy `.ipynb` notebooks should be cleaned before publication:

- remove execution counts;
- remove large output cells;
- keep source code and markdown;
- keep one lesson overview at the top;
- avoid hardcoded "today" output as evidence;
- move reusable helpers into `src/` when a pattern repeats.

The recommended cleanup command is:

```bash
uv run jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/class/*.ipynb
```

After cleanup:

```bash
git diff --check
make book
```
