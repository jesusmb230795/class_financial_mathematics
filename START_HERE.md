# Financial Mathematics Lab

Use this page as the entry point for the local course workspace.

## Student workflow

1. Follow `_toc.yml` or open `notebooks/course/` for the complete published
   path across Modules 0 through 10.
2. Treat `notebooks/labs/` as an author staging area and
   `notebooks/legacy/` as provenance, not as additional student routes.
3. Run a notebook from top to bottom with **Run → Run All Cells**.
4. Keep the kernel as **Python 3 (ipykernel)**.
5. Save your work in a personal branch or copy; generated outputs are removed
   from the canonical notebooks.

The first code cell of every executable notebook contains the common setup.
Lessons use worked checks, interpretation prompts, or decision notes when those
activities strengthen a specific objective. The published repository does not
append a universal checkpoint or public answer key to every notebook.

## Author workflow

Start with `content/README.md` and `content/plan.json` when receiving or
integrating text. Stage new prose with `make new-content`; an incoming draft is
not published until its module placement, language, citations, notation, and
promotion checks are complete.

For executable material, `notebooks/manifest.json` owns notebook status and the
percent-format `.py` file is the editable source of each Jupytext pair.
Regenerate the `.ipynb` surface after changing the `.py` source, then run:

```bash
make sync-jupytext
make check-notebook-sources
make check-book-content
```

Before publishing, run `make publish-check`. To restore the tracked JupyterLab
profile after experimenting with local settings, run `make lab-reset`.
