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
# # Mexican Rate-Panel PCA and Scenarios
#
# Module: Fixed Income, Credit, and Term Structure
#
# ## Lesson summary
#
# Principal Component Analysis compresses co-movements in a panel of rates into a
# small number of orthogonal statistical factors. This lesson uses three official
# Banxico series with different economic meanings, so it does **not** call them
# zero-curve tenors or automatically label their components level, slope, and
# curvature {cite}`banxicoSIE2025,mishkin2019financial`.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - build a real Banxico rate-history panel;
# - compute yield changes and run PCA;
# - interpret the first three principal components as statistical rate-panel shocks;
# - construct common-rate and instrument-specific scenarios;
# - explain the limits of PCA-based stress testing.
#
# ## Prerequisites
#
# Complete the quantitative time-series foundation and the curve-construction
# lessons first. Readers should be able to align dated series, compute changes,
# standardize variables, and interpret eigenvectors without assigning economic
# labels mechanically. This notebook uses a heterogeneous Banxico rate panel,
# not interchangeable zero-curve tenors.
#
# ## PCA representation
#
# Let $\Delta r_t$ be the vector of changes across the documented rate series.
# PCA approximates standardized panel changes with a small number of orthogonal factors:
#
# $$
# \Delta r_t \approx b_1 f_{1,t}+b_2 f_{2,t}+b_3 f_{3,t}.
# $$
#
# The loading vectors $b_1$, $b_2$, and $b_3$ describe co-movement among the
# policy rate, CETES 28-day yield, and TIIE 28-day rate. Because these are not
# homogeneous zero rates ordered by maturity, their positions cannot support
# yield-curve slope or curvature labels.
#
# ## Python setup

# %% tags=["setup", "hide-input"]
import pandas as pd
import matplotlib.pyplot as plt

from src.term_structure import official_mexican_rate_history, rate_panel_pca

# %% [markdown]
# ## Official Banxico rate history
#
# This panel reads the committed Banxico daily snapshot and uses real observed Mexican rates: the policy rate, CETES 28-day rate, and TIIE 28-day rate. It is a short-rate/rate-history panel for PCA mechanics, not a full licensed zero-coupon curve.

# %%
rate_history = official_mexican_rate_history(start="2018-01-01", end="2026-06-05")
rate_history.tail()

# %%
rate_history.iloc[-252:].plot(figsize=(10, 4), title="Official Banxico Rate Panel")
plt.xlabel("Date")
plt.ylabel("Rate")
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## PCA on standardized rate changes

# %%
components, explained = rate_panel_pca(rate_history, n_components=3)

explained[["component", "explained_variance_ratio", "cumulative_variance"]]

# %%
components

# %% [markdown]
# ## Loading interpretation
#
# The sign of a PCA component is arbitrary. Interpret each loading only by the
# named rate series:
#
# - same-sign loadings indicate a common rate-panel move;
# - opposite policy and market-rate signs indicate relative repricing;
# - a dominant TIIE loading indicates interbank-rate variation.
#
# These descriptions must be checked against the displayed loadings. They are not
# structural economic identities.

# %%
fig, ax = plt.subplots(figsize=(10, 4))
for component in components.index:
    ax.plot(components.columns, components.loc[component], marker="o", label=component)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("PCA Loadings")
ax.set_xlabel("Rate series")
ax.set_ylabel("Standardized loading")
ax.grid(True, alpha=0.3)
ax.legend()
plt.show()

# %% [markdown]
# ## Scenario construction
#
# The scenarios below apply stylized basis-point shocks to the latest observed
# Banxico rate panel. They are named by economic exposure, not by curve geometry.
# This is neither a regulatory stress test nor a full zero-curve scenario.

# %%
latest_rates = rate_history.iloc[-1]
common_up = pd.Series(0.0100, index=latest_rates.index)
policy_led_tightening = pd.Series(
    {
        "policy_rate": 0.0100,
        "cetes_28d": 0.0060,
        "tiie_28d": 0.0040,
    }
)
interbank_spread_widening = pd.Series(
    {
        "policy_rate": 0.0000,
        "cetes_28d": 0.0000,
        "tiie_28d": 0.0075,
    }
)

scenarios = pd.DataFrame(
    {
        "base": latest_rates,
        "common_up_100bp": latest_rates + common_up,
        "policy_led_tightening": latest_rates + policy_led_tightening,
        "interbank_spread_widening": latest_rates + interbank_spread_widening,
    },
)
scenarios

# %%
scenarios.plot(figsize=(10, 4), marker="o", title="Rate-Panel Stress Scenarios")
plt.xlabel("Rate series")
plt.ylabel("Yield")
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## Model limitations
#
# - PCA factors are sample-dependent and can change when the yield-curve regime changes.
# - Component labels are statistical interpretations, not fixed economic laws.
# - A true yield-curve PCA requires comparable zero rates ordered by maturity.
# - Scenario shocks should be checked against portfolio exposures and historical plausibility.

# %% [markdown]
# ## Handoff
#
# The calibration lab turns one documented short-rate proxy into estimated
# Vasicek parameters and compares simulated rate distributions. Treat the PCA
# scenarios here as separate multivariate stress inputs: they do not calibrate a
# one-factor short-rate model or prove that it fits the current curve.
