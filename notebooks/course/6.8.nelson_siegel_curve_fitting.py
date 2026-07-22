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
# A bootstrapped curve can match observed instruments closely, but it may also inherit microstructure noise. Nelson-Siegel gives a smooth four-parameter representation of the zero-coupon curve and separates level, slope, and curvature in a way that is useful for macro-financial interpretation {cite}`fabozzi2019foundations`.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - explain the Nelson-Siegel basis functions;
# - fit a smooth curve to observed zero-coupon yields;
# - interpret level, slope, curvature, and decay parameters;
# - compare market yields with fitted yields and residuals;
# - derive discount factors from the fitted curve.
#
# ## Prerequisites
#
# Complete the yield-curve bootstrap before fitting this parametric curve.
# Readers should understand spot rates, discount factors, maturity grids, and
# residual diagnostics. The observed inputs must be comparable zero-coupon rates
# under one currency and convention; a mixed rate panel is not a substitute for
# a zero curve.
#
# ## Python setup

# %% tags=["setup", "hide-input"]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.term_structure import (
    discount_factor_from_spot_rate,
    fit_nelson_siegel,
    nelson_siegel_yield,
    spot_rate_from_discount_factor,
)

# %% [markdown]
# ## Synthetic Mexican-style zero curve
#
# The data below are deterministic classroom inputs. They form a downward-sloping,
# or inverted, nominal zero curve: the 3-month yield is 10.1% and the 30-year
# yield is 8.3%. They are not live market quotes.

# %%
maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
observed_yields = np.array([0.101, 0.099, 0.096, 0.092, 0.089, 0.086, 0.085, 0.084, 0.083, 0.083])

curve = pd.DataFrame({"maturity_years": maturities, "observed_yield": observed_yields})
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
# ## Fit the curve

# %%
params = fit_nelson_siegel(maturities, observed_yields)
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

curve["fitted_yield"] = nelson_siegel_yield(
    maturities,
    params["beta0"],
    params["beta1"],
    params["beta2"],
    params["tau"],
)
curve["residual_bp"] = (curve["observed_yield"] - curve["fitted_yield"]) * 10_000
curve

# %% [markdown]
# ## Visual diagnostic

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(dense_maturities, fitted_yields, label="Nelson-Siegel fit")
axes[0].scatter(maturities, observed_yields, color="black", label="Observed")
axes[0].set_title("Fitted Zero-Coupon Curve")
axes[0].set_xlabel("Maturity in years")
axes[0].set_ylabel("Yield")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].bar(curve["maturity_years"].astype(str), curve["residual_bp"])
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("Fit Residuals")
axes[1].set_xlabel("Maturity")
axes[1].set_ylabel("Basis points")
axes[1].grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

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
        "fitted_yield_annual": selected_yields,
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
    discount_table["recovered_yield_annual"] - discount_table["fitted_yield_annual"]
)
discount_table

# %% [markdown]
# ## Model limitations
#
# - Nelson-Siegel is smooth and parsimonious, so it can miss local pricing kinks or illiquidity effects.
# - Parameter estimates can be unstable when maturities are sparse or concentrated.
# - A visually good fit does not guarantee arbitrage-free dynamics or stable out-of-sample forecasts.

# %% [markdown] tags=["exercise"]
# ## Checkpoint exercise
#
# Increase the 10-year observed yield by 25 basis points and refit the curve.
#
# 1. Report the baseline and shocked $\beta_0$, $\beta_1$, $\beta_2$, and $\tau$.
# 2. Compare the fitted 2-, 10-, and 30-year yields in basis points and identify
#    whether the original curve is upward-sloping or inverted.
# 3. Rebuild the annual-compounding discount table and verify that every
#    spot-rate roundtrip error is below $10^{-12}$.
# 4. Explain why a local quote shock can change several fitted maturities.
#

# %% [markdown] tags=["solution"]
# ```{dropdown} Suggested answer rubric
# A complete answer labels the original curve as inverted, reports parameter and
# yield changes with units, verifies the annual-compounding roundtrip, and notes
# that a parsimonious curve spreads local information across maturities.
# ```

# %% [markdown]
# ## Handoff
#
# The following PCA lesson studies historical co-movement and stress scenarios in
# a documented Banxico rate panel. Keep the distinction explicit: Nelson-Siegel
# fits a comparable zero curve across maturities, whereas the PCA inputs are
# heterogeneous observed rates and cannot inherit curve-factor labels by order.
