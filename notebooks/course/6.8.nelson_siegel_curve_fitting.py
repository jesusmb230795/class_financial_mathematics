# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Nelson-Siegel Curve Fitting
#
# Module: Fixed Income, Credit, and Term Structure
#
# ## Lesson summary
#
# A bootstrapped curve can match input instruments closely, but it may also
# inherit microstructure noise. Nelson-Siegel gives a smooth four-parameter
# representation of a zero-coupon curve and separates level, slope, and
# curvature loadings {cite}`nelsonSiegel1987yieldCurves`. This lesson fits only
# deterministic synthetic rates; it makes no claim about a current Mexican
# market curve.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - explain the Nelson-Siegel basis functions;
# - fit a smooth curve to homogeneous zero-coupon inputs;
# - interpret level, slope, curvature, and decay parameters;
# - compare input zero rates with fitted zero rates and residuals;
# - derive discount factors from the fitted curve.
#
# ## Prerequisites
#
# Complete the yield-curve bootstrap before fitting this parametric curve.
# Readers should understand spot rates, discount factors, maturity grids, and
# residual diagnostics. In empirical work, the inputs must be comparable
# zero-coupon rates under one currency and convention; a mixed rate panel is not
# a substitute for a zero curve.
#
# ## Python setup

# %% tags=["setup", "hide-input"]
import numpy as np
import pandas as pd

from src.module6_visuals import build_nelson_siegel_diagnostic_figure
from src.term_structure import (
    discount_factor_from_spot_rate,
    fit_nelson_siegel,
    nelson_siegel_yield,
    spot_rate_from_discount_factor,
)

# %% [markdown]
# ## Synthetic nominal zero curve
#
# The data below are deterministic classroom inputs. They form a downward-sloping,
# or inverted, nominal zero curve: the 3-month annual rate is 10.1% and the
# 30-year annual rate is 8.3%. They are not observed market quotes,
# have no currency assignment, and have no provider date, retrieval date, or
# market vintage.

# %%
maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
synthetic_zero_rates = np.array(
    [0.101, 0.099, 0.096, 0.092, 0.089, 0.086, 0.085, 0.084, 0.083, 0.083]
)

curve = pd.DataFrame(
    {
        "maturity_years": maturities,
        "synthetic_zero_rate_annual": synthetic_zero_rates,
    }
)
curve

# %% [markdown]
# ## Model form
#
# The Nelson-Siegel zero-coupon yield is:
#
# $$
# y(t) =
# \beta_0
# + \beta_1 \left(\frac{1-e^{-t/\tau}}{t/\tau}\right)
# + \beta_2 \left(\frac{1-e^{-t/\tau}}{t/\tau} - e^{-t/\tau}\right).
# $$
#
# The interpretation is:
#
# - $\beta_0$: long-term level;
# - $\beta_1$: short-end loading; the long-minus-short slope is approximately
#   $-\beta_1$;
# - $\beta_2$: medium-term curvature component;
# - $\tau$: decay parameter that controls where the curvature loading is strongest.
#
# With equal weights, this lesson estimates the parameter vector
# $\vartheta=(\beta_0,\beta_1,\beta_2,\tau)$ by least squares:
#
# $$
# \widehat\vartheta
# = \arg\min_{\vartheta:\,\tau>0}
# \sum_{i=1}^{N}
# \left[y_i-y_{\mathrm{NS}}(t_i;\vartheta)\right]^2.
# $$
#
# The objective treats every maturity quote equally. Alternative weights based
# on duration, bid-ask width, or instrument reliability answer different fitting
# questions and must be declared.
#
# ## Fit the curve

# %%
params = fit_nelson_siegel(maturities, synthetic_zero_rates)
pd.Series(params)

# %%
dense_maturities = np.linspace(0.25, 30, 200)
fitted_yields = nelson_siegel_yield(
    dense_maturities,
    params["beta0"],
    params["beta1"],
    params["beta2"],
    params["tau"],
)

curve["fitted_zero_rate_annual"] = nelson_siegel_yield(
    maturities,
    params["beta0"],
    params["beta1"],
    params["beta2"],
    params["tau"],
)
curve["residual_bp"] = (
    curve["synthetic_zero_rate_annual"] - curve["fitted_zero_rate_annual"]
) * 10_000
curve

# %%
fit_diagnostics = pd.Series(
    {
        "observations": len(curve),
        "sse_decimal_rate_squared": params["sse"],
        "rmse_basis_points": float(np.sqrt(np.mean(curve["residual_bp"] ** 2))),
        "maximum_absolute_residual_basis_points": float(curve["residual_bp"].abs().max()),
    }
)
fit_diagnostics

# %% [markdown]
# ## Visual diagnostic

# %% mystnb={"image": {"alt": "Two panels compare ten deterministic synthetic annual zero rates with a Nelson-Siegel fitted curve from 0.25 to 30 years and show signed fitting residuals in basis points by maturity."}}
nelson_siegel_source_note = (
    "Source: author-created deterministic nominal zero rates; annual compounding; "
    "0.25- to 30-year maturities; equal-weight least squares. No currency, "
    "provider, observation date, retrieval date, or market vintage."
)
build_nelson_siegel_diagnostic_figure(
    curve,
    dense_maturities,
    fitted_yields,
    source_note=nelson_siegel_source_note,
)

# %% [markdown]
# The main pattern is a smooth decline from the short end toward the long-run
# level. The residual panel is the important exception: even a visually smooth
# four-parameter fit does not pass through every synthetic input. The diagnostic
# is limited to in-sample, equal-weight fit; it provides no evidence of
# no-arbitrage dynamics, parameter stability, or forecast accuracy.

# %% [markdown]
# ## Nelson-Siegel-Svensson extension
#
# Svensson adds a second curvature loading with its own decay parameter
# {cite}`svensson1994forwardRates`:
#
# $$
# y_{\mathrm{NSS}}(t)
# = \beta_0
# + \beta_1\left(\frac{1-e^{-t/\tau_1}}{t/\tau_1}\right)
# + \beta_2\left(\frac{1-e^{-t/\tau_1}}{t/\tau_1}-e^{-t/\tau_1}\right)
# + \beta_3\left(\frac{1-e^{-t/\tau_2}}{t/\tau_2}-e^{-t/\tau_2}\right),
# $$
#
# with $\tau_1>0$ and $\tau_2>0$. The additional hump can capture a second
# bend in a richer curve, but it also weakens parameter identification and can
# overfit sparse maturities. The four-parameter Nelson-Siegel specification is
# retained here because ten smooth synthetic points do not justify that added
# flexibility.

# %% [markdown]
# ## Discount factors from the fitted curve
#
# Once the smooth curve is fitted, any maturity can be converted into a discount
# factor. This table explicitly uses **annual compounding**, matching the bootstrap
# lesson. The recovered spot must roundtrip to the fitted spot.

# %%
selected_maturities = np.array([1, 3, 5, 10, 20, 30], dtype=float)
selected_yields = nelson_siegel_yield(
    selected_maturities,
    params["beta0"],
    params["beta1"],
    params["beta2"],
    params["tau"],
)

discount_table = pd.DataFrame(
    {
        "maturity_years": selected_maturities,
        "fitted_zero_rate_annual": selected_yields,
        "discount_factor_annual": [
            discount_factor_from_spot_rate(
                rate,
                maturity,
                compounding="annual",
            )
            for rate, maturity in zip(selected_yields, selected_maturities)
        ],
    }
)
discount_table["recovered_yield_annual"] = [
    spot_rate_from_discount_factor(
        discount,
        maturity,
        compounding="annual",
    )
    for discount, maturity in zip(
        discount_table["discount_factor_annual"],
        discount_table["maturity_years"],
    )
]
discount_table["roundtrip_error"] = (
    discount_table["recovered_yield_annual"]
    - discount_table["fitted_zero_rate_annual"]
)
discount_table

# %% [markdown]
# ## Model limitations
#
# - Nelson-Siegel is smooth and parsimonious, so it can miss local pricing kinks or illiquidity effects.
# - Parameter estimates can be unstable when maturities are sparse or concentrated.
# - A visually good fit does not guarantee arbitrage-free dynamics or stable out-of-sample forecasts.
# - Calling synthetic inputs "observed" would overstate provenance; empirical
#   use requires a documented homogeneous zero-rate source, currency, date, and convention.

# %% [markdown]
# ## Handoff
#
# The following PCA lesson studies historical co-movement and stress scenarios in
# a documented Banxico rate panel. Keep the distinction explicit: Nelson-Siegel
# fits a comparable zero curve across maturities, whereas the PCA inputs are
# heterogeneous observed rates and cannot inherit curve-factor labels by order.
