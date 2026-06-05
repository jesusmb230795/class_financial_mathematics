# Term Structure and Interest Rate Models

This module extends fixed-income valuation from a single yield to the full term structure of interest rates.

## Expected outcome

By the end of this module, students should be able to:

- distinguish spot rates, forward rates, par rates, and discount factors;
- bootstrap a zero-coupon curve from observed instruments;
- interpolate and visualize a yield curve;
- fit a Nelson-Siegel curve as a parsimonious term-structure model;
- use PCA to create level, slope, and curvature stress scenarios;
- calibrate and simulate Vasicek and Cox-Ingersoll-Ross short-rate models.

## Included notes

- yield curve bootstrapping;
- short-rate models;
- Nelson-Siegel curve fitting;
- yield curve PCA and scenarios;
- short-rate calibration lab.

## Recommended next improvements

- add Banxico and Mexican yield curve data sources;
- compare local interpolation methods, including monotone-convex ideas;
- implement Nelson-Siegel-Svensson estimation as an extension to Nelson-Siegel;
- add discount-factor distribution summaries from simulated short-rate paths;
- connect PCA scenarios to fixed-income portfolio valuation and liabilities.

## Class sequence

1. Convert market rates into discount factors.
2. Bootstrap a spot curve from short deposits and coupon bonds.
3. Compare spot, forward, and par curves.
4. Fit a smooth curve and interpret level, slope, and curvature.
5. Use PCA to decompose curve changes into level, slope, and curvature-like shocks.
6. Introduce mean-reverting short-rate dynamics.
7. Connect Vasicek calibration with AR(1) estimation.
8. Compare exact Vasicek simulation against boundary-safe CIR discretization.
9. Discuss calibration, simulation, and model limitations.

## In-class practice

Students should build a small synthetic yield curve, bootstrap discount factors, derive spot and forward rates, fit a smooth Nelson-Siegel curve, and create PCA-style stress scenarios.

## Module checkpoint

The checkpoint is a yield curve notebook with a bootstrapped curve, a fitted smooth curve, PCA scenario interpretation, short-rate calibration output, and a short explanation of model limitations.
