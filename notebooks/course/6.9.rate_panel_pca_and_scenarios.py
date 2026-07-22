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
# curvature
# {cite}`banxicoSIE2025,banxicoTIIETransition2025,littermanScheinkman1991bondFactors`.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - build a real Banxico rate-history panel;
# - compute weekly rate changes and run standardized PCA;
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
# Let $\Delta r_t$ be the vector of weekly changes across the documented rate
# series, $\mu_{\Delta r}$ its in-sample mean, and $S$ the diagonal matrix of
# in-sample standard deviations computed with the population denominator,
# matching `StandardScaler`. The actual transformation implemented here is
#
# $$
# z_t
# = S^{-1}\!\left(\Delta r_t-\mu_{\Delta r}\right)
# \approx Bf_t,
# $$
#
# where the columns of $B$ are orthonormal loading vectors and $f_t$ contains
# principal-component scores. Equivalently,
#
# $$
# z_t \approx b_1 f_{1,t}+b_2 f_{2,t}+b_3 f_{3,t}.
# $$
#
# The loading vectors $b_1$, $b_2$, and $b_3$ describe co-movement among the
# policy rate, CETES 28-day rate, and TIIE 28-day rate. Because these are not
# homogeneous zero rates ordered by maturity, their positions cannot support
# yield-curve slope or curvature labels.
#
# ## Python setup

# %% tags=["setup", "hide-input"]
import pandas as pd

from src.module6_visuals import (
    build_rate_history_figure,
    build_rate_panel_diagnostics_figure,
)
from src.term_structure import official_mexican_rate_history, rate_panel_pca

# %% [markdown]
# ## Official Banxico rate history
#
# This panel reads the committed Banxico snapshot and takes the last provider
# observation for each series **within** each W-FRI week: policy rate SF61745,
# CETES 28-day rate SF60633, and TIIE 28-day rate SF60648. It retains only weeks
# that are complete across the three series and never carries a value across an
# empty week. Rates are annualized decimals in code. As defined in Lesson 6.3,
# the CETES field is interpreted as an ACT/360 return-yield quote; the policy and
# TIIE fields retain their distinct provider quotation meanings. Converting all
# three fields to decimal scale does not make their instruments or conventions
# interchangeable. This is a heterogeneous rate panel for PCA mechanics, not a
# homogeneous zero-coupon curve. Redistribution rights have not been independently verified
# {cite}`banxicoSIE2025,banxicoGovSecuritiesTechnical`.
#
# Banco de México changed the TIIE 28-day methodology from submitted bank quotes
# to a transaction-based method effective 2025-01-01. The snapshot preserves
# that break rather than backcasting a homogeneous history
# {cite}`banxicoTIIETransition2025`.

# %%
rate_history = official_mexican_rate_history(
    start="2018-01-01",
    end="2026-06-05",
    frequency="weekly",
)
rate_metadata = rate_history.attrs.copy()
rate_history.tail()

# %%
rate_audit = pd.Series(
    {
        "actual_start": rate_history.index.min().date().isoformat(),
        "actual_end": rate_history.index.max().date().isoformat(),
        "weekly_observations": len(rate_history),
        "missing_values": int(rate_history.isna().sum().sum()),
        "duplicate_dates": int(rate_history.index.duplicated().sum()),
        "frequency": rate_metadata["frequency"],
        "calendar": rate_metadata["calendar"],
        "alignment": rate_metadata["alignment"],
        "observation_policy": rate_metadata["observation_policy"],
        "series_ids": rate_metadata["series_ids"],
        "snapshot_vintage": rate_metadata["source_vintage"],
        "retrieved_at": rate_metadata["retrieved_at"],
        "methodology_breaks": rate_metadata["methodology_breaks"],
    }
)
rate_audit

# %% mystnb={"image": {"alt": "Three line-and-marker series show weekly Friday-aligned Banco de México policy, CETES 28-day, and TIIE 28-day annual rates from 5 January 2018 through 5 June 2026; a vertical marker identifies the 1 January 2025 TIIE methodology break."}}
series_note = ", ".join(
    f"{name} {series_id}" for name, series_id in rate_metadata["series_ids"].items()
)
banxico_source_note = (
    f"Source: {rate_metadata['source']}; {series_note}; "
    f"{rate_metadata['alignment']}; actual sample {rate_metadata['sample_start']} "
    f"to {rate_metadata['sample_end']} ({len(rate_history)} weeks); snapshot "
    f"vintage {rate_metadata['source_vintage']}; retrieved "
    f"{rate_metadata['retrieved_at']}; data mode {rate_metadata['data_mode']}. "
    "The TIIE methodology break effective 2025-01-01 is preserved."
)
build_rate_history_figure(rate_history, source_note=banxico_source_note)

# %% [markdown]
# The main pattern is a shared tightening-and-easing cycle across the three
# annual rates. The main exception is that policy, Treasury-bill, and interbank
# rates retain distinct economic meanings and do not move one-for-one; the 2025
# TIIE methodology break is also a comparability exception. The weekly alignment
# avoids treating unchanged calendar-filled values as daily information, but it
# cannot remove publication timing differences or the structural break.

# %% [markdown]
# ## PCA on standardized rate changes

# %%
components, explained = rate_panel_pca(rate_history, n_components=3)

explained[["component", "explained_variance_ratio", "cumulative_variance"]]

# %%
components

# %%
explained_by_component = explained.set_index("component")
dominant_series = components.abs().idxmax(axis=1)
pca_interpretation = pd.DataFrame(
    {
        "explained_variance_ratio": explained_by_component[
            "explained_variance_ratio"
        ],
        "cumulative_variance": explained_by_component["cumulative_variance"],
        "dominant_rate_series": dominant_series,
        "dominant_signed_loading": [
            components.loc[component, series]
            for component, series in dominant_series.items()
        ],
    }
)
pca_interpretation

# %% [markdown]
# ## Methodology-break sensitivity
#
# Re-estimating explained variance before and after the TIIE methodology change
# is a diagnostic, not a causal test. The post-break window is much shorter and
# also reflects a different monetary regime, so any difference combines sample,
# regime, and measurement effects.

# %%
_, pre_break_explained = rate_panel_pca(
    rate_history.loc[:"2024-12-27"],
    n_components=3,
)
_, post_break_explained = rate_panel_pca(
    rate_history.loc["2025-01-03":],
    n_components=3,
)

break_sensitivity = pd.DataFrame(
    {
        "full_sample": explained.set_index("component")["explained_variance_ratio"],
        "pre_2025_method": pre_break_explained.set_index("component")[
            "explained_variance_ratio"
        ],
        "post_2025_method": post_break_explained.set_index("component")[
            "explained_variance_ratio"
        ],
    }
)
break_sensitivity

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
scenario_changes_bp = scenarios.sub(scenarios["base"], axis=0) * 10_000
scenario_changes_bp

# %% mystnb={"image": {"alt": "Two panels show standardized PCA loadings for three named Banco de México rate series and grouped scenario changes in basis points; the scenario marks are separated by rate series and are not connected as a yield curve."}}
diagnostic_source_note = (
    f"PCA source: {rate_metadata['source']}; {series_note}; actual sample "
    f"{rate_metadata['sample_start']} to {rate_metadata['sample_end']} "
    f"({len(rate_history)} observations); standardized weekly decimal-rate "
    f"changes; snapshot vintage {rate_metadata['source_vintage']}; retrieved "
    f"{rate_metadata['retrieved_at']}. Scenarios: author-created deterministic "
    "shocks to the final panel observation; not forecasts or regulatory stresses."
)
build_rate_panel_diagnostics_figure(
    components,
    scenarios,
    source_note=diagnostic_source_note,
)

# %% [markdown]
# The loading panel shows which named rates co-move after each weekly change is
# standardized; the arbitrary component sign is the important exception, so
# only relative signs and magnitudes are meaningful. The reproducible
# interpretation and scenario-change tables identify the dominant loading and
# each basis-point shock without relying on visual estimation. Grouped marks
# deliberately avoid connecting heterogeneous rates as though they were ordered
# curve tenors. These deterministic shocks carry no probability and do not
# measure portfolio loss without exposure and valuation models.

# %% [markdown]
# ## Model limitations
#
# - PCA factors are sample-dependent and can change when the rate regime or
#   measurement method changes.
# - Component labels are statistical interpretations, not fixed economic laws.
# - A true yield-curve PCA requires comparable zero rates ordered by maturity.
# - Scenario shocks should be checked against portfolio exposures and historical plausibility.
# - Standardization gives each input unit variance, so loadings describe
#   correlation structure rather than the basis-point covariance matrix.

# %% [markdown]
# ## Handoff
#
# The calibration lab turns one documented short-rate proxy into estimated
# Vasicek parameters and compares simulated rate distributions. Treat the PCA
# scenarios here as separate multivariate stress inputs: they do not calibrate a
# one-factor short-rate model or prove that it fits the current curve.
