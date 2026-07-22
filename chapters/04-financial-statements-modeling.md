# Module 4: Financial Statement Analysis and Financial Modeling

Financial statements turn operating activity, financing decisions, and
accounting judgments into a structured record. Analysts use that record to
understand how a business earns money, how it funds itself, where cash is tied
up, and whether reported performance is likely to persist. The central skill in
this module is therefore not ratio memorization. It is maintaining an auditable
chain from filings and notes to normalized history, forecast assumptions, and a
balanced, simplified three-statement model.

This module follows Markets, Instruments, and Data and the macroeconomic
foundation. It prepares the cash-flow inputs used in Module 5, Corporate
Issuers and Equity Valuation. Valuation is deliberately deferred until the
reader can explain what each accounting number represents, reconcile earnings
to cash, and identify the judgments embedded in the source documents
{cite}`damodaran2012investment`.

## Learning objectives

By the end of this module, readers should be able to:

- scope a financial statement analysis and collect the controlling filing,
  notes, management commentary, the auditor's report, and relevant assurance
  disclosures;
- connect the income statement, balance sheet, cash flow statement, and equity
  roll-forward through explicit accounting identities;
- calculate and interpret profitability, liquidity, leverage, coverage,
  DuPont, return on equity (ROE), and return on invested capital (ROIC)
  measures with consistent periods and units;
- distinguish recurring operating performance from accounting classification,
  estimation, and non-recurring items;
- adapt an analysis for banks, insurers, multinational groups, acquisitions,
  intercorporate investments, and special-purpose entities;
- trace, reconcile, and review a simplified one-year driver-based pro forma
  model whose statements balance and whose cash movement reconciles;
- communicate normalized results, sensitivities, model checks, and limitations
  in a concise review memo.

## Prerequisites

Readers should complete:

- [Market Foundations](../notebooks/course/1.1.stock_markets.md) for instrument,
  issuer, and data-source distinctions;
- [Market Data Quality Framework](../notebooks/course/1.6.market_data_quality_framework.ipynb)
  for provenance, units, and sample controls;
- [Macro Indicators and Policy](../notebooks/course/3.2.macro_indicators_policy.md)
  for inflation, rates, growth, and currency effects on operating assumptions.

This module assumes comfort with percentages, present values, and basic
algebra. Rates and ratios are stored as decimals in calculations and displayed
as percentages only when the `%` label is explicit.

## Notation and sign conventions

The module separates **stocks**, measured at a reporting date, from **flows**,
measured over the period ending on that date. Unless a lesson states otherwise:

- $t-1$ and $t$ denote the opening and closing reporting dates;
- $\Delta X_t=X_t-X_{t-1}$ is the change in a balance during the period;
- $\overline{B}_t=(B_{t-1}+B_t)/2$ is the two-date average of balance $B$;
- $T_t$ is the normalized marginal cash tax rate for the period;
- $\mathrm{OWC}_t$ is operating working capital under a stated account
  perimeter; and
- simulated statement amounts are USD millions, written as USD m.

Tables show expenses and cash outflows in parentheses. In driver equations,
capital expenditures, dividends, and debt repayments are positive magnitudes
that are subtracted. When CFO, CFI, or CFF is copied from a cash flow statement,
the reported section total retains its own sign. This distinction prevents a
negative outflow from being subtracted twice.

## Conceptual spine

### The accounting system

At every reporting date, the balance sheet must satisfy:

```{math}
\text{Assets}_t=\text{Liabilities}_t+\text{Equity}_t.
```

For a simplified company without other comprehensive income or other direct
equity adjustments, retained earnings ($\mathrm{RE}$) satisfy:

```{math}
\mathrm{RE}_t
=\mathrm{RE}_{t-1}
+\text{Net income}_t
-\text{Dividends declared}_t.
```

The cash flow statement can include a separately reported exchange-rate effect.
Using the statement's own cash-and-cash-equivalents scope:

```{math}
\Delta\text{Cash}^{\mathrm{SCF}}_t
=\mathrm{CFO}_t+\mathrm{CFI}_t+\mathrm{CFF}_t
+\mathrm{FX\ effect}_t.
```

CFO is cash flow from operating activities, CFI is cash flow from investing
activities, and CFF is cash flow from financing activities. The analyst must
bridge the statement scope to balance-sheet cash, restricted cash, and any
qualifying overdrafts. When the scopes are identical and the exchange effect is
zero, the identity simplifies to CFO + CFI + CFF. Actual filings can classify
some cash flows differently across reporting frameworks, so the analyst must
record the issuer's policy before comparing companies
{cite}`ifrsConceptualFramework2018,ias1FinancialStatements2026,ias7CashFlows2026`.

For IFRS reporters, this presentation baseline is date-sensitive: IFRS 18
replaces IAS 1 for annual periods beginning on or after 1 January 2027, with
earlier application permitted. An analysis should record both the reporting
period and the presentation standard actually applied
{cite}`ifrs18Presentation2024`.

```{figure} ../img/generated/m4-three-statement-evidence-model-map.png
:alt: Financial modeling map linking the income statement, balance sheet, cash flow, and equity while separating reported, reclassified, normalized, and forecast evidence layers.
:width: 900px
:align: center
:name: fig-m4-evidence-statement-map

The model is reviewable only when statement linkages reconcile and every value
retains its evidence layer from filing through forecast. Operating-asset,
liability, fixed-asset, and debt schedules also feed the cash bridge; closing
cash then returns to the balance sheet.
```

### Earnings, capital, and cash flow

The module uses net operating profit after tax (NOPAT) and the following core
quantities:

```{math}
\mathrm{NOPAT}_t=\text{Normalized EBIT}_t(1-T_t),
```

This compact form is valid only when the normalized EBIT tax effect is
realizable at the stated rate. Losses, credits, valuation allowances,
jurisdiction mix, and permanently non-deductible items require an item-level
tax bridge rather than a mechanical multiplication.

```{math}
\mathrm{ROIC}_t
=\frac{\mathrm{NOPAT}_t}
{\overline{\text{Invested capital}}_t},
```

```{math}
\mathrm{FCFF}_t
=\mathrm{NOPAT}_t
+\mathrm{D\&A}_t
-\text{Capital expenditures}_t
-\Delta\mathrm{OWC}_t.
```

$\mathrm{D\&A}$ means depreciation and amortization. Operating working
capital includes only the operating current assets and non-interest-bearing
operating liabilities selected for the analysis. Free cash flow to the firm
(FCFF) is a valuation input, not a line item that can be copied from every
filing. Its definition must be reconstructed consistently
{cite}`damodaran2012investment`.

### Evidence before interpretation

The analysis preserves four layers:

1. **Reported:** values exactly as presented in the controlling filing.
2. **Reclassified:** presentation changes that improve comparison without
   changing total economics.
3. **Normalized:** clearly supported adjustments intended to estimate
   sustainable performance.
4. **Forecast:** assumptions about future operations, financing, and cash flow.

The layers must never be silently mixed. Each adjustment needs a source,
direction, amount, tax treatment, period, rationale, and reversibility note.

## Lesson map

| Order | Page | Role | Evidence or output |
| --- | --- | --- | --- |
| 1 | [Analysis Framework and Statement Linkages](../notebooks/course/4.1.analysis_framework_and_statement_linkages.md) | Establish the evidence hierarchy and accounting identities | Source map and balanced statement bridge |
| 2 | [Ratios, DuPont, and Capital Efficiency](../notebooks/course/4.2.ratios_dupont_and_capital_efficiency.md) | Translate statements into comparable operating and financing diagnostics | Ratio tree and ROE/ROIC decomposition |
| 3 | [Accounting Quality and Normalization](../notebooks/course/4.3.accounting_quality_and_normalization.md) | Separate recurring economics from reporting judgments | Normalized earnings and owner-earnings bridge |
| 4 | [Specialized Entities and Consolidation](../notebooks/course/4.4.specialized_entities_and_consolidation.md) | Adapt the framework to different institutional forms | Entity-specific analysis map |
| 5 | [Three-Statement Model and Review Memo](../notebooks/course/4.5.three_statement_model_and_review.md) | Trace evidence through a simplified balanced forecast and review conclusion | Integrated one-year example, stress-case design, and model review memo |

## Data and reproducibility contract

Every financial statement table or model should identify:

- issuer, legal entity, ticker or identifier, and reporting jurisdiction;
- accounting framework and whether the statements are consolidated or
  standalone;
- reporting currency, display unit, fiscal year-end, period length, and whether
  the period is annual, interim, year-to-date, or trailing;
- filing type, filing date, source URL or local snapshot, retrieval date, and
  amendment or restatement status;
- audit or review status, opinion type, and any material uncertainty related to
  going concern, scope limitation, or emphasis-of-matter paragraph relevant to
  the analysis;
- whether a value is reported, reclassified, normalized, calculated, or
  forecast;
- sign convention and formula for every derived metric;
- snapshot vintage and a source-to-output trace for model inputs.

Financial statement values are period-specific and may be revised. A model
must not overwrite a historical snapshot when a new filing appears. Store the
new vintage separately and document the restatement bridge. Traceability and
machine-readable source structure are part of the analytical result, not
clerical work {cite}`wilkinson2016fair,wickham2014tidy`.

## Reading sequence

1. Begin with the source hierarchy and statement linkages.
2. Use ratios only after confirming comparable definitions and periods.
3. Normalize accounting items before projecting margins or cash flow.
4. Adapt the analysis for the issuer's institutional form.
5. End with the three-statement model, sensitivities, and review memo.

## Module practice

Choose one public issuer and build a one-page source map containing the latest
annual filing, latest interim filing, notes, management commentary, and the
auditor's report. Record currency, display units, fiscal calendar,
consolidation scope, and any restatement. For every source, record the document
date, filing or stable source URL, retrieval date, and whether access or reuse
restrictions limit redistribution. Then identify one item that would prevent a
mechanical comparison with a peer; do not copy filing text or tables into the
repository without a rights review.

```{dropdown} Review standard
A complete response names the exact documents and dates, distinguishes audited
annual information from interim or management information, states the
accounting and unit conventions, and explains a specific comparability issue
such as a fiscal-year mismatch, acquisition, lease policy, segment change, or
currency translation effect. It also records retrieval and access information.
A generic statement that the companies are "different" is not sufficient.
```

## Handoff

Start with [Analysis Framework and Statement Linkages](../notebooks/course/4.1.analysis_framework_and_statement_linkages.md).
It establishes the evidence and reconciliation discipline used by every later
ratio, adjustment, forecast, and valuation.
