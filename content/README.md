# Book content workflow

This directory is the editorial intake layer for the book. It accepts new text
without making that text part of the published navigation before it has been
reviewed, normalized, cited, and validated.

## Sources of truth

The repository deliberately separates three concerns:

- `content/plan.json` records module readiness, known gaps, target paths, and
  the next useful text for each module.
- `content/debt.json` records unresolved structural, editorial, pedagogical,
  and migration debt with owners and validation criteria.
- `_toc.yml` is the production Jupyter Book 1 navigation. `myst.yml` is the
  Jupyter Book 2 candidate and must contain the same published pages in the same
  order while the migration remains parallel.
- `notebooks/manifest.json` owns notebook status, execution policy, and the
  published/labs/legacy boundary.

Do not repeat those inventories in a new document. Update the owning register
or manifest and run the repository checks instead.

## Language policy

The current published surface is English, matching the existing book and
Sphinx configuration. Incoming source text may be English or Spanish. Record
its language in the intake header; Spanish source text can remain Spanish
during review, but it must be translated and terminology-checked before it is
promoted to the English publication surface.

Every intake starts with `rights_status: "needs-review"`. Confirm authorship or
license before integration; the intake command cannot infer reuse rights from a
file or prompt.

## Terminology and notation contract

Publication text and code must use these conventions unless a lesson explicitly
introduces and contrasts another convention:

- **Units:** store rates and returns as decimal values in code. Display them as
  percentages only with an explicit `%` label. State currency, frequency,
  horizon, calendar, and annualization factor next to reported results.
- **Returns:** name simple returns and log returns explicitly. The quantity
  `mean(log_return) * periods_per_year` is an annualized log return; convert it
  with `expm1` before calling it an annualized simple return. Annualized
  volatility uses `sqrt(periods_per_year)` and must declare the period count.
- **Losses and tail risk:** define loss as `L = -R`. VaR and Expected Shortfall
  are reported as non-negative loss magnitudes. If `alpha` is the lower-tail
  probability, `VaR_alpha = max(0, -Q_alpha(R))` and
  `ES_alpha = max(0, -E[R | R <= Q_alpha(R)])`. Before applying the reporting
  floor, `-Q_alpha(R) = Q_(1-alpha)(L)`. A negative return quantile is a
  *return threshold*, not a positive-loss VaR. Pages using a confidence level
  must label it as `confidence = 1 - alpha`.
- **Volatility models:** for `GARCH(p, q)` in this book and in the `arch`
  package, `p` counts lagged squared shocks and `q` counts lagged conditional
  variances. State any alternative textbook convention before using it.
  Student-t innovations used for conditional risk must be standardized to unit
  variance when the fitted model assumes standardized residuals.
- **Rates and fixed income:** state simple, periodic, or continuous
  compounding; payment frequency; settlement convention; and day-count basis.
  Do not compare or convert rates without passing the convention explicitly.
- **FX:** write a quote as `QUOTE per BASE`. The canonical classroom example is
  `MXN per USD`; an increase is a USD appreciation and MXN depreciation. Define
  domestic and foreign rates before applying parity conditions.
- **Samples and vintages:** report inclusive start and end dates from the data
  actually analyzed, not only the intended request. Name the snapshot vintage
  and retrieval date, and distinguish observed series from constructed indices.
- **Curve language:** reserve *yield curve*, *level*, *slope*, and *curvature*
  for homogeneous rate observations ordered by tenor. A mixed set of policy,
  money-market, or benchmark rates is a rate panel.
- **Module names:** use the canonical titles in `content/plan.json`. Legacy
  numbers may appear only in an explicitly labeled migration note.

Terminology review is a semantic gate. A notebook can execute successfully and
still fail promotion when its labels, units, signs, sample, or interpretation
do not match this contract.

## Figure and exploratory-analysis contract

This contract applies to every programmatically rendered chart, dashboard,
diagnostic, and static figure in the published book. Generated editorial
illustrations retain their asset-specific requirements in
`img/generated/visual-assets.json`, but they use the same visual language.
Reusable styling and figure construction belong in `src/`; notebook cells
should select data, call a builder, and explain the result rather than repeat
formatting code or literal color values.

### Canonical palette and semantic roles

Use the repository palette consistently across Matplotlib, Seaborn, Plotly,
and generated assets. A color has the same role everywhere:

| Token | Hex | Semantic role |
| --- | --- | --- |
| Background | `#F7F3EA` | Primary figure canvas and neutral empty space. |
| Ink | `#102A43` | Titles, labels, axes, reference lines, and primary text. |
| Teal | `#0F766E` | Primary observed series or the reader's current selection. |
| Muted blue | `#4F6F9F` | Comparison series, historical reference, or secondary context. |
| Brand amber | `#D97706` | Non-textual fills and secondary accents for an estimate, benchmark, threshold, or item requiring attention. |
| Amber dark | `#B45309` | Essential amber-coded lines, marker edges, outlines, and state indicators. |
| Coral | `#C2410C` | Downside, loss, breach, drawdown, or risk emphasis. |
| Soft grid | `#D8DEE9` | Subtle gridlines, separators, and low-emphasis structure. |

Essential graphical objects and state indicators must reach at least a 3:1
contrast ratio against the adjacent background. Brand amber is approximately
2.88:1 against the canonical `#F7F3EA` background, so it must not be the only
encoding for an essential thin line or marker. Amber dark is approximately
4.53:1 against that background and is the required amber variant for essential
strokes and marks. Keep normal-size text in Ink and meet at least 4.5:1 text
contrast; color remains a secondary encoding even when its contrast passes.

Do not assign a new meaning to a color within a figure. For additional
categorical series, cycle through the canonical accents in a stable order and
pair them with line styles or markers. Diverging scales must have a meaningful
center, normally zero, a labeled color bar, and visible direction at both ends.
Sequential scales must follow the magnitude of the encoded quantity. Avoid
decorative gradients and rainbow palettes.

### Hierarchy, typography, and layout

- Use a reproducible sans-serif face, with `DejaVu Sans` as the default for
  programmatic figures. Keep body and axis labels at least 10 pt at final book
  width; compact tick, legend, and source-note text at least 9 pt; panel titles
  at 11-12 pt; and the figure title at 14-16 pt.
- Render inline notebook figures at 96 dpi to keep executed notebooks within
  publication-output budgets; save standalone PNG assets at 160 dpi. Resolution
  changes must not alter analytical scales, labels, or panel content.
- Use sentence case. Give each figure one concise title that identifies the
  analytical question or comparison; give small multiples short parallel panel
  titles. Do not repeat the title in every axis or legend.
- Preserve a clear hierarchy: title, plot, direct labels or legend, then source
  and methodological note. Keep annotations short and place long interpretation
  in the surrounding lesson text.
- Prefer direct labels when they remain legible. Otherwise order the legend to
  match the visual reading order and use the same names as the data inventory.
- Remove non-informative borders and decoration. Use light horizontal grids
  when they support comparison, retain reference lines when analytically
  meaningful, and leave enough margin for labels at normal and mobile widths.
- Bars normally begin at zero. A non-zero baseline or transformed axis must be
  visibly labeled and justified. Avoid dual axes unless the analytical question
  cannot be answered more truthfully with aligned panels or normalized series.

### Titles, axes, units, source, and sample

Every published figure must be interpretable without inspecting its code:

- Axis titles name the quantity and unit. Rates and returns displayed as
  percentages include `%`; currency axes state the currency or FX quote;
  volatility states its horizon and annualization; index levels state the base;
  and time axes use readable calendar dates.
- Titles and legends distinguish levels, simple returns, log returns,
  cumulative returns, constructed indices, estimates, and forecasts. Hover text
  uses the same units and terminology as the visible axes.
- The caption or adjacent note states the analytical question, actual inclusive
  sample dates, provider and series or instrument, snapshot vintage or retrieval
  date, frequency and calendar, offline/live mode, and any material construction
  or limitation. Do not report only the intended request window when the
  observed sample is shorter.
- Comparisons use compatible definitions, units, frequencies, and samples. If a
  transformation, rebasing, filtering rule, benchmark, or rolling window is
  essential to interpretation, state it next to the figure.
- Reference lines, confidence bands, thresholds, and event annotations are
  labeled in the plot or caption; their meaning must not be inferred from color.

### Accessibility and truthful encoding

- Never encode meaning by color alone. Combine color with labels, line styles,
  markers, hatching, shapes, or position, especially for gains versus losses,
  actual versus estimate, and normal versus breach states.
- Maintain readable contrast against the background and verify that adjacent
  series remain distinguishable in grayscale and under common color-vision
  deficiencies. Red/green-only comparisons are not permitted.
- Keep text, markers, and interactive targets legible at the book's rendered
  width. Avoid crowded annotations, overlapping dates, unexplained acronyms,
  and legends that obscure data.
- Static embedded images require concise alt text. Complex programmatic figures
  require a nearby prose summary of the main pattern, exception, and limitation
  so the conclusion is not available only visually.
- Heatmaps include a labeled scale and, when space permits, readable values or
  a companion table. Scatterplots make overplotting visible through opacity,
  aggregation, or density treatment rather than hiding it.
- Visual emphasis must follow analytical importance. Do not use area, volume,
  perspective, smoothing, truncated scales, or animation in ways that exaggerate
  a difference or imply unsupported precision or causality.

### Matplotlib and Plotly parity

Interactive Plotly figures and publication-safe Matplotlib fallbacks are two
representations of the same analytical contract. They must preserve trace order,
semantic colors, line and marker distinctions, titles, axis labels and formats,
reference values, sample, and the central conclusion. Hover, zoom, selectors,
and widgets may add detail, but the static fallback must still answer the stated
question without a browser runtime, network request, or remote CDN.

Build pure figure functions in `src/` so layout and trace contracts can be
tested independently from display and widgets. Use deterministic dimensions and
data ordering. The notebook decides whether to display the interactive builder
or its fallback; it must not maintain two independent calculations.

### Exploratory-analysis sequence

EDA is an evidence sequence, not a gallery of plots. Use the following order,
omitting only steps that are not relevant to the lesson's question:

1. State the financial question, decision context, data source, variables,
   conventions, and expected comparison.
2. Audit the observed sample: boundaries, frequency, missingness, duplicates,
   units, revisions, calendar alignment, and data-quality exceptions.
3. Inspect levels and univariate distributions with appropriate summaries,
   robust statistics, and visible sample counts.
4. Apply and label financial transformations such as simple or log returns,
   rebasing, spreads, real values, or annualization.
5. Examine time variation with rolling diagnostics, drawdowns, regimes, or
   event annotations while declaring every window and benchmark.
6. Examine dependence with correlations, scatterplots, heatmaps, and rolling or
   conditional relationships; never describe correlation as causation.
7. Investigate tails, outliers, instability, and sensitivity. Distinguish a
   data-quality anomaly from a plausible financial event before filtering it.
8. Close with a concise narrative: question, data, method, finding,
   interpretation, limitation, and next analytical step.

### Figure review gate

A figure is not publication-ready merely because its cell executes. Reviewers
must confirm that:

- the chart answers one explicit analytical question and its takeaway agrees
  with the calculation and surrounding prose;
- the shared palette and style helpers are used without ad hoc semantic colors;
- labels, signs, units, horizons, annualization, sample dates, source, vintage,
  transformation, benchmark, and data mode are accurate and visible;
- the chart type, baseline, scale, aggregation, and annotations represent the
  data truthfully and do not imply unsupported causality or precision;
- color-independent encodings, contrast, text size, alt text or prose summary,
  mobile layout, and grayscale reading have been checked;
- Plotly and Matplotlib outputs preserve the same analytical meaning and the
  offline build does not depend on remote assets;
- figure builders have focused tests for trace or artist structure, semantic
  roles, labels, units, and reference values; and
- Jupytext synchronization, source validation, fresh execution, tests, and the
  rendered-book review pass before `make publish-check` is accepted.

## Receive new text

Inspect the current editorial state:

```bash
make content-status
```

Create a safe intake file:

```bash
make new-content \
  MODULE=04 \
  SLUG=financial-statement-architecture \
  TITLE="Financial Statement Architecture" \
  TYPE=lesson \
  LANGUAGE=es
```

The command writes a file under `content/incoming/`. Paste or append the source
text under `## Draft text`; preserve the metadata block and required headings.
Then run:

```bash
make check-book-content
```

New text is not published merely because it exists in `content/incoming/`.
That directory is a review boundary, not a second book.

## Integrate reviewed text

1. Identify the module, target page, prerequisites, and duplicate coverage with
   `content/plan.json`.
2. Normalize notation, terminology, language, claims, citations, examples, and
   learning objectives against the contract above.
3. Move narrative prose into `chapters/` or a narrative `.md` lesson under
   `notebooks/course/`. For executable lessons, author the percent-format `.py`
   source and run `make sync-jupytext` to regenerate the `.ipynb` surface.
   Use an executable pair only when the page must run Python to produce a
   table, figure, diagnostic, or interactive result.
4. Move reusable Python into `src/`; keep notebook cells focused on the lesson.
5. Add glossary terms and bibliography entries at the same time as the text
   that needs them.
6. Update `content/plan.json`. Add a page to both `_toc.yml` and `myst.yml` only
   when it is ready for publication, and add notebook sources to
   `notebooks/manifest.json` with the correct status.
7. Run the focused checks, then the publication gate.

Use `content/templates/module-overview.md` and
`content/templates/lesson.md` as integration checklists. They are scaffolds,
not pages to publish verbatim.

## Promotion contract

A page is ready for the published TOCs only when:

- its role in the module sequence is explicit and it does not duplicate an
  existing page without a documented reason;
- learning objectives, notation, definitions, examples, limitations, and
  handoff to the next page are coherent;
- factual or methodological claims have traceable citations;
- data examples declare provider, series or instrument, unit, frequency,
  calendar, revision policy, license note, and snapshot/live mode as relevant;
- executable notebooks use the canonical kernel and pass source validation;
- both TOCs stay in parity and the module status is updated;
- `make check-book-content`, `make check-notebook-sources`, and
  `make publish-check` pass.

Do not place licensed readings, private assessment material, credentials,
unreviewed third-party text, or large raw datasets in the public intake area.

## Status vocabulary

- `published`: visible in both TOCs and covered by the publication gate.
- `source-only`: useful material exists outside the published navigation and
  still needs consolidation, renumbering, or review.
- `roadmap`: the module has a defined target but needs new text.
- `substantial`: the current module is coherent for its present scope.
- `partial`: a published branch is usable, but promised coverage is missing.
- `not-started`: no reusable module text currently exists.

The status terms describe repository readiness, not the pedagogical quality of
future text.
