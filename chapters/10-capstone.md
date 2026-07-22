# Module 10: Advanced Pathways and Capstone

Module 10 closes the book by requiring a decision-quality synthesis rather than
another isolated technique. Every learner examines three professional
pathways—Portfolio Management, Private Markets, and Private Wealth—then selects
one pathway for an integrative capstone. The common core combines real data,
valuation, positive-loss risk analysis, portfolio implications, reproducible
evidence, a dashboard, written memoranda, and executive communication.

The capstone is not judged by model complexity. It is judged by whether another
analyst can reproduce the result, trace every material claim to evidence,
challenge the assumptions, and understand the recommended action and its
conditions.

## Learning objectives

By the end of this module, readers should be able to:

- compare the decisions, clients, evidence, and implementation constraints in
  portfolio management, private markets, and private wealth;
- design index, active-equity, active-fixed-income, curve, credit, and
  execution decisions within an institutional mandate;
- screen, value, structure, monitor, and plan the realization of a
  private-market investment;
- translate client objectives, family dynamics, human capital, concentrated
  positions, liquidity, philanthropy, and wealth transfer into a governed
  private-wealth plan;
- answer structured investment questions with calculations, evidence,
  limitations, and a decision;
- produce a reproducible capstone containing real-data inputs, a dashboard,
  valuation memo, risk memo, investment-committee memo, and presentation;
- communicate a recommendation, alternatives, failure conditions, and
  monitoring triggers to an executive audience.

## Prerequisites

The capstone assumes that readers have completed or can demonstrate equivalent
competence in:

- data provenance, quality controls, versioned snapshots, and exploratory
  analysis from Module 1;
- statistics, time-series diagnostics, volatility, and scenario analysis from
  Modules 2 and 3;
- financial statements, valuation, and cash-flow modeling from Modules 4 and
  5;
- rate, credit, derivative, hedge, and positive-loss risk conventions from
  Modules 6 and 7;
- alternative-asset structures, fees, valuation, and liquidity from
  [Module 8](08-alternative-investments.md);
- investment policy, portfolio construction, implementation, and performance
  evaluation from Module 9.

The [curriculum roadmap](course-roadmap.md) remains the source for module
publication status. A learner may draft a capstone before every module is
published, but the final submission must identify and close any prerequisite
gap rather than assume it away.

## Conceptual spine

### One decision chain

Every pathway uses the same chain:

```{math}
\text{Decision}
\longleftarrow
\text{Recommendation}
\longleftarrow
\text{Analysis}
\longleftarrow
\text{Evidence}.
```

Monitoring then tests whether the evidence and recommendation remain valid:

```{math}
\text{Evidence update}
\rightarrow
\text{Trigger}
\rightarrow
\text{Review or action}.
```

The capstone preserves six layers:

1. **Observed evidence:** source records, market data, filings, contracts, and
   client-approved facts.
2. **Transformations:** cleaning, alignment, unit conversion, derived fields,
   and quality flags.
3. **Assumptions:** forecasts, scenarios, model parameters, and policy choices.
4. **Outputs:** valuation, risk, allocation, performance, and sensitivity.
5. **Decision:** recommendation, alternatives, sizing, conditions, and owner.
6. **Monitoring:** triggers, frequency, tolerance, escalation, and exit.

Observed facts and assumptions must never share an unlabeled column.

```{figure} ../img/generated/m10-capstone-evidence-decision-monitoring-map.png
:alt: Capstone evidence chain from observed evidence through transformations, assumptions, outputs, recommendation, and decision, with monitoring triggers and three professional pathways.
:width: 900px
:align: center

Every pathway uses the same traceable evidence chain and closes the loop through
monitoring, triggers, and review.
```

### Common measurement contract

- Rates and returns are decimals in calculations and have explicit frequency,
  horizon, compounding, currency, and annualization when displayed.
- Loss is \(L=-R\). VaR and Expected Shortfall are non-negative loss
  magnitudes, with `confidence = 1 - alpha`.
- Valuation ranges identify the claim, valuation date, cash-flow unit,
  discount-rate convention, and terminal assumption.
- Portfolio outputs identify benchmark, constraints, transaction costs,
  currency, and whether returns are gross or net.
- Private-vehicle outputs identify capital calls, distributions, residual NAV,
  fees, carry, and valuation status.
- Client facts are separated from analyst assumptions and are handled under the
  applicable confidentiality and authorization rules.

## Pathways

| Pathway | Primary decision | Required synthesis |
| --- | --- | --- |
| Portfolio Management | How should an institutional portfolio deviate from or implement its benchmark? | index design, active positions, rates and credit, execution, risk, governance |
| Private Markets | Should capital be committed to a deal or vehicle, on what terms, and how is value realized? | screening, valuation, structure, agreements, value creation, calls, distributions, carry, exit |
| Private Wealth | How should a family convert resources and constraints into a durable plan? | discovery, human and financial capital, liquidity, concentration, planning, philanthropy, transfer |

All three pathways require a benchmark or counterfactual, a valuation or
resource model, a risk memo, an implementation plan, and decision governance.

## Lesson map

| Order | Page | Role | Evidence or output |
| --- | --- | --- | --- |
| 1 | [Portfolio Management Pathway](../notebooks/course/10.1.portfolio_management_pathway.md) | Apply index, active, fixed-income, credit, and execution decisions | Institutional active-risk and implementation case |
| 2 | [Private Markets Pathway](../notebooks/course/10.2.private_markets_pathway.md) | Integrate deal, vehicle, agreement, and realization economics | Deal screen, valuation, structure, and exit case |
| 3 | [Private Wealth Pathway](../notebooks/course/10.3.private_wealth_pathway.md) | Convert client and family context into a governed wealth plan | Wealth balance sheet and liquidity/concentration plan |
| 4 | [Structured Response and Committee Communication](../notebooks/course/10.4.structured_response_and_committee_communication.md) | Turn analysis into reviewable written and oral decisions | Response lab, investment memo, and committee briefing |
| 5 | [Integrative Capstone](../notebooks/course/10.5.integrative_capstone.md) | Complete the common evidence and delivery contract | Reproducible project, dashboard, two technical memos, decision memo, and presentation |

## Capstone input contract

The final project uses real, auditable input data for the current-state
analysis. Forecasts and stress scenarios may be simulated, but they are labeled
as assumptions. Every dataset has:

| Field | Requirement |
| --- | --- |
| Identity | provider, series/instrument/document identifier, legal entity or client-approved source |
| Measurement | unit, currency, frequency, calendar, timezone, inclusive sample dates |
| Vintage | as-of date, retrieval date, revision or restatement status, snapshot path |
| Access | public/private status, license or rights note, credential requirement |
| Transformation | raw-to-clean mapping, formula, missing-data rule, quality flag |
| Ownership | person responsible for refresh, review, and exception resolution |

Versioned, machine-readable inputs and tidy transformation logs support
reproducibility {cite}`wilkinson2016fair,wickham2014tidy`. Secrets, private
client records, and licensed raw data remain outside the public repository.
Synthetic replacements must preserve the teaching structure without being
presented as observed evidence.

## Required capstone outputs

Every pathway delivers:

1. a one-page decision brief and scope;
2. a source inventory, data dictionary, and reproducible transformation
   pipeline;
3. an exploratory and quality-control appendix;
4. a valuation or resource-model memo with sensitivities;
5. a risk memo with positive-loss measures and named stress scenarios;
6. a decision dashboard with source dates, units, limits, and exceptions;
7. an implementation and monitoring plan;
8. an investment-committee memorandum;
9. an executive presentation;
10. a limitations, conflicts, and unresolved-evidence register.

The pathway adds specialized evidence; it does not replace the common outputs.

## Data, confidentiality, and reproducibility

The repository bundle must execute from documented local commands using
pinned dependencies and a versioned snapshot mode. Live acquisition is an
optional refresh path, not the only way to reproduce the submitted figures.
Outputs must be regenerable without editing notebook cells by hand.

Private or personal data is minimized and separated from public artifacts.
Client names, account identifiers, contracts, data-room documents, credentials,
and restricted provider data are never committed. A redacted evidence index
can prove that a document was reviewed without redistributing it.

## Reading sequence

1. Read all three pathway briefs and identify the decision, stakeholder, and
   evidence differences.
2. Select one pathway and write a testable capstone question.
3. Complete the structured-response lab before building the full dashboard.
4. Freeze the input snapshot and evidence inventory.
5. Build valuation, risk, portfolio, and implementation analyses.
6. Draft the technical memos before compressing the result into executive
   communication.
7. Run an independent reproduction and challenge review.

## Module practice

Write a 150-word capstone pre-mortem. State the intended pathway and decision,
then identify one data failure, one model failure, one implementation failure,
and one governance failure that could make a polished recommendation wrong.
For each failure, name the evidence or control that would detect it.

```{dropdown} Review standard
A complete pre-mortem identifies failures specific to the proposed decision,
not generic statements such as "markets are uncertain." Each control has an
owner, timing, and observable trigger. At least one failure must challenge the
central thesis rather than only reduce the expected return.
```

## Handoff

Begin with [Portfolio Management Pathway](../notebooks/course/10.1.portfolio_management_pathway.md)
and compare its institutional decision structure with the two following
pathways before choosing a capstone.
