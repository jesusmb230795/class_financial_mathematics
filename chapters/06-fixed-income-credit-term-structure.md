# Fixed Income, Credit, and Term Structure

This module connects time-value-of-money foundations with instrument valuation,
credit loss, spread risk, yield curves, and interest-rate models. It begins with
cash-flow and convention discipline, adds issuer and securitization risk, then
replaces a flat yield with a term structure. The purpose is to support valuation,
scenario analysis, hedging, and model-risk communication rather than to present
one yield as a complete description of a bond {cite}`macaulay1938interest,fabozzi2019foundations`.

Readers should first complete Quantitative Methods and Financial Time Series
and the Module 3 macro-scenario case. Those prerequisites establish compounding,
annualization, versioned data, and the rate/inflation scenarios reused here.

```{figure} ../img/generated/m6-fixed-income-cash-flow-duration-map.png
:alt: Fixed-income map connecting contractual cash flows and discounting to price sensitivity, credit and spread risk, and asset-liability immunization.
:width: 900px
:align: center

Contractual cash flows, curve conventions, rate sensitivity, credit loss, and
liability matching form one controlled valuation chain.
```

## Expected outcome

By the end of this module, readers should be able to:

- discount dated cash flows under an explicit compounding and day-count
  convention;
- price zero-coupon, coupon, inflation-linked, and simplified Mexican
  government instruments;
- calculate yield to maturity, duration, convexity, and DV01;
- explain the sign and scale of rate sensitivities and test a simple
  Redington-style immunization;
- separate benchmark-curve, spread, default, recovery, liquidity, and embedded
  option risk;
- bootstrap discount factors and spot, forward, and par rates, fit a
  parsimonious term structure, and construct controlled rate scenarios; and
- explain calibration, approximation, data, and model limitations.

## Conceptual spine

The module uses one sequence:

```text
cash flows and conventions
→ price and yield
→ duration, convexity, and DV01
→ credit loss, spread, and optionality
→ discount factors and spot/forward rates
→ curve fitting and rate-panel scenarios
→ short-rate calibration and model risk
```

Rate inputs remain decimals in code. Every example must state currency,
compounding, payment frequency, settlement, day count, horizon, and whether the
input is an observed quote, a constructed classroom index, or a simulation.
Nominal spread, Z-spread, and option-adjusted spread are distinct measures.

```{figure} ../img/generated/m6-yield-curve-factor-scenarios.png
:alt: Term-structure map separating homogeneous yield-curve construction and scenarios from principal components of a heterogeneous rate panel and short-rate calibration.
:width: 900px
:align: center

Tenor-ordered curve factors and principal components of a heterogeneous rate
panel are different analytical objects and retain separate labels.
```

## Published lesson map

| Lesson | Main role |
| --- | --- |
| `6.1.bond_pricing_duration_convexity` | Coupon-bond pricing, yield, duration, and convexity under explicit conventions {cite}`macaulay1938interest,fisherWeil1971immunization` |
| `6.2.interactive_bond_sensitivity` | Exact repricing versus duration-convexity approximation |
| `6.3.mexican_government_bond_valuation` | CETES, Bonos M, UDIBONOS, and simplified settlement conventions {cite}`banxicoGovSecurities,banxicoGovSecuritiesTechnical` |
| `6.4.ytm_dv01_and_immunization_lab` | Root-solved YTM, DV01, and Redington-style immunization {cite}`redington1952immunization` |
| `6.5.credit_spreads_securitized_products` | Expected loss, spreads, seniority, waterfalls, and securitized-product risks {cite}`baselFrameworkCreditSecuritisation2026,mcneil2015quantitative` |
| `6.6.yield_curve_bootstrapping` | Spot, forward, par, and complete-cash-flow bootstrap mechanics |
| `6.7.short_rate_models` | Vasicek and CIR foundations {cite}`vasicek1977termStructure,coxIngersollRoss1985termStructure` |
| `6.8.nelson_siegel_curve_fitting` | Parsimonious synthetic-curve fitting and diagnostics {cite}`nelsonSiegel1987yieldCurves,svensson1994forwardRates` |
| `6.9.rate_panel_pca_and_scenarios` | PCA and scenarios for a heterogeneous rate panel without false yield-curve labels {cite}`littermanScheinkman1991bondFactors` |
| `6.10.short_rate_calibration_lab` | Calibration, simulation, parameter uncertainty, and model risk |

The superseded fixed-income and term-structure overviews are retained under
`chapters/legacy/` for provenance only. Each executable `.py` above is the
canonical Jupytext authoring source; its `.ipynb` pair is generated
deterministically.

## Data and reproducibility contract

The Mexican government-security and rate-panel examples rely on committed
Banxico-backed snapshots and must retain series identifiers, units, actual
sample boundaries, vintage, and construction notes {cite}`banxicoSIE2025,banxicoTIIETransition2025`.
Synthetic curves, bonds, and paths must be labeled as simulations and retain
their parameters and random seed. A committed snapshot enables a reproducible
build but does not by itself establish redistribution rights. The published
notebooks therefore identify the provider and series, distinguish provider-dated
observations from weekly Friday alignment, disclose the 2025 TIIE methodology
break, and make no claim that redistribution rights have been verified.

Curve language is strict:

- a **yield curve** is a homogeneous set of rates ordered by tenor;
- a **rate panel** may contain policy and money-market rates with different
  economic roles;
- level, slope, and curvature labels apply only when tenor ordering supports
  that interpretation.

## Reading sequence

1. Reuse the Module 2 compounding and present-value conventions.
2. Price coupon and Mexican government instruments, first statically and then
   with the sensitivity dashboard.
3. Measure duration, convexity, YTM, DV01, and immunization.
4. Add expected loss, spreads, seniority, optionality, and securitization
   waterfalls.
5. Replace a flat yield with discount factors, spot rates, and forward rates.
6. Bootstrap a complete cash-flow structure.
7. Study Vasicek and CIR foundations before returning to cross-sectional curve
   fitting and rate-panel PCA.
8. Fit a parsimonious curve and construct rate-panel scenarios without calling
   heterogeneous instruments a yield curve.
9. Calibrate and simulate a short-rate model on an explicitly aligned time grid.
10. Close with convention, liquidity, credit, calibration, data-rights, and
    model-risk limitations.

## Module limitations

This substantial core does not provide effective or key-rate duration,
homogeneous-tenor curve PCA, interest-rate trees, convertibles, CDS valuation,
production-grade credit-curve or OAS calibration, loan-level securitization,
legal waterfall modeling, default dependence, transaction costs, or
dealer-quality settlement logic. Redistribution rights for the committed
observed-data snapshot also require an owner-authorized review. Those omissions
must remain visible when its methods are applied.

## Handoff

Module 7 uses these discount factors and sensitivities to price forwards, swaps,
and options, design hedges, and measure residual market risk.
