# Legacy source: Term Structure and Interest Rate Models

This module extends fixed-income valuation from a single yield to the full term structure of interest rates.

## Expected outcome

By the end of this module, students should be able to:

- distinguish spot rates, forward rates, par rates, and discount factors;
- bootstrap a zero-coupon curve from observed instruments;
- interpolate and visualize a yield curve;
- fit a Nelson-Siegel curve as a parsimonious term-structure model;
- recognize Nelson-Siegel-Svensson and monotone-convex interpolation as natural extensions;
- use PCA to create level, slope, and curvature stress scenarios;
- calibrate and simulate Vasicek and Cox-Ingersoll-Ross short-rate models;
- separate parameter uncertainty, calibration error, recalibration risk, and model misspecification.

## Included notes

- yield curve bootstrapping;
- short-rate models;
- Nelson-Siegel curve fitting;
- yield curve PCA and scenarios;
- short-rate calibration lab.

## Reading sequence

1. Convert market rates into discount factors.
2. Bootstrap a spot curve from short deposits and coupon bonds.
3. Compare spot, forward, and par curves.
4. Fit a smooth curve and interpret level, slope, and curvature.
5. Treat Nelson-Siegel-Svensson and monotone-convex interpolation as extensions when a simple curve is not enough.
6. Use PCA to decompose curve changes into level, slope, and curvature-like shocks.
7. Introduce mean-reverting short-rate dynamics.
8. Connect Vasicek calibration with AR(1) estimation.
9. Compare exact Vasicek simulation against boundary-safe CIR discretization.
10. Discuss calibration, simulation, and model limitations.
