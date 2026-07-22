# Contributing to the book

Start with `content/README.md`. It is the canonical editorial workflow for
receiving, reviewing, integrating, and promoting new text.

## Small fixes

For a typo, broken local link, citation correction, or focused code fix in an
existing page:

1. edit the owning source;
2. regenerate the `.ipynb` file when the source is a paired `.py` notebook;
3. run the focused validation;
4. run `make publish-check` before proposing publication.

## New or substantially revised content

1. Run `make content-status`.
2. Create an intake file with `make new-content`.
3. Record language, module, target, content type, citation status, and
   integration notes.
4. Review the draft against the templates in `content/templates/`.
5. Update `content/plan.json`, glossary, bibliography, notebook manifest, and
   both TOCs only where their ownership boundary changes.
6. Update `content/debt.json` when the change resolves, re-scores, or discovers
   a tracked structural or editorial gap.
7. Run the focused checks documented in `README.md` and `Makefile`, then run
   `make publish-check` before proposing publication.

Do not publish unreviewed source text directly from `content/incoming/`. Do not
commit credentials, private assessments, licensed readings, unapproved
third-party prose, generated build output, or large raw datasets.

The current book is published in English. Spanish source text is welcome in
the intake stage and must record `source_language: "es"` until translation and
terminology review are complete.
