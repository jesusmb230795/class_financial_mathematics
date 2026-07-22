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
# # American and Exotic Option Methods
#
# Module: Derivatives and Risk Management
#
# ## Lesson summary
#
# Many options cannot be handled by a single closed-form Black-Scholes price.
# American exercise, path dependence, averaging, and barriers require numerical
# methods. This lesson compares CRR convergence, Leisen-Reimer American put
# pricing, Asian option control variates, and a continuity correction for
# discretely monitored barriers {cite}`hull2022options`.
#
# ## Learning objectives
#
# By the end of this lesson, students should be able to:
#
# - compare CRR binomial convergence against a Black-Scholes benchmark;
# - explain early exercise in American put valuation;
# - use a control variate to reduce Monte Carlo error for an Asian call;
# - apply the BGK continuity correction to a discrete barrier;
# - state the model-risk tradeoffs of lattice and simulation methods.
#
# ## Prerequisites
#
# Complete the European pricing, numerical-pricing, and interactive-dashboard
# lessons first. Students should understand backward induction, Monte Carlo
# standard error, local Greeks, and why convergence to a benchmark under shared
# assumptions is an implementation check rather than market validation.
#
# ## Python setup

# %% tags=["setup", "hide-input"]
import pandas as pd
import matplotlib.pyplot as plt

from src.derivatives import (
    arithmetic_asian_call_control_variate,
    bgk_adjusted_barrier,
    black_scholes_price,
    crr_binomial_option_price,
    leisen_reimer_american_put,
)

# %% [markdown]
# ## CRR convergence
#
# The CRR tree converges to Black-Scholes for European options, but the convergence can oscillate because terminal nodes do not always align well with the payoff kink.

# %%
spot = 100.0
strike = 100.0
rate = 0.05
volatility = 0.25
maturity = 1.0
dividend_yield = 0.0

benchmark = black_scholes_price(spot, strike, rate, volatility, maturity, "call", dividend_yield)
steps_grid = [5, 10, 25, 50, 100, 250, 500]

convergence = pd.DataFrame({"steps": steps_grid})
convergence["crr_call"] = [
    crr_binomial_option_price(
        spot,
        strike,
        rate,
        volatility,
        maturity,
        steps=steps,
        option_type="call",
        dividend_yield=dividend_yield,
    )
    for steps in steps_grid
]
convergence["black_scholes_call"] = benchmark
convergence["error"] = convergence["crr_call"] - benchmark
convergence

# %%
ax = convergence.plot(x="steps", y="error", marker="o", figsize=(8, 4), legend=False)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("CRR Convergence Error")
ax.set_xlabel("Steps")
ax.set_ylabel("Price error")
ax.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## American put early exercise
#
# American options require an optimal stopping check at every node:
#
# $$
# V = \max(\text{continuation value}, \text{intrinsic value}).
# $$

# %%
american_comparison = pd.DataFrame(
    {
        "method": ["European Black-Scholes", "CRR American", "Leisen-Reimer American"],
        "put_price": [
            black_scholes_price(spot, strike, rate, volatility, maturity, "put"),
            crr_binomial_option_price(
                spot,
                strike,
                rate,
                volatility,
                maturity,
                steps=501,
                option_type="put",
                american=True,
            ),
            leisen_reimer_american_put(
                spot,
                strike,
                rate,
                volatility,
                maturity,
                steps=501,
            ),
        ],
    }
)
american_comparison["early_exercise_premium"] = (
    american_comparison["put_price"] - american_comparison.loc[0, "put_price"]
)
american_comparison

# %% [markdown]
# ## Asian option control variate
#
# Arithmetic Asian options usually need simulation. A geometric Asian option is a useful control variate because it is highly correlated with the arithmetic payoff and has a known analytical price.

# %%
asian = arithmetic_asian_call_control_variate(
    spot=100,
    strike=100,
    rate=0.05,
    volatility=0.25,
    maturity=1.0,
    observations=52,
    paths=20_000,
    seed=2027,
)

pd.Series(asian.__dict__)

# %%
asian.naive_standard_error / asian.control_variate_standard_error

# %% [markdown]
# The ratio above is the standard-error reduction factor. With a fixed control
# coefficient and known control expectation, the adjustment preserves the target
# expectation; estimating the coefficient from the same finite sample adds a
# small estimation nuance. The principal benefit is much lower sampling error.
#
# ## Barrier continuity correction
#
# For discretely monitored barriers, the BGK adjustment shifts the barrier away from the spot to approximate the difference between continuous and discrete monitoring:
#
# $$
# H_{adj} = H \exp(\pm \beta \sigma \sqrt{\Delta t}),
# \qquad
# \beta \approx 0.5826.
# $$

# %%
barrier = 80.0
daily_dt = 1 / 252

pd.Series(
    {
        "physical_down_barrier": barrier,
        "bgk_adjusted_down_barrier": bgk_adjusted_barrier(
            barrier,
            volatility=0.25,
            monitoring_interval=daily_dt,
            barrier_type="down",
        ),
        "physical_up_barrier": 120.0,
        "bgk_adjusted_up_barrier": bgk_adjusted_barrier(
            120.0,
            volatility=0.25,
            monitoring_interval=daily_dt,
            barrier_type="up",
        ),
    }
)

# %% [markdown]
# ## Model limitations
#
# - American and exotic option prices are sensitive to numerical method, monitoring convention, and exercise policy.
# - Control variates reduce simulation noise only when the control remains highly correlated with the target payoff.
# - Barrier corrections are approximations and can fail when barriers are close to spot or volatility is unstable.
#
# ## Handoff
#
# Early-exercise premium, path dependence, and barrier monitoring create risks
# that a European delta alone cannot describe. A hedge report should pair the
# numerical price with exercise-boundary, monitoring-frequency, Greek, and
# discrete scenario P&L; model-method differences are explicit valuation reserves
# or governance items rather than hidden inside VaR. The next lesson adds
# stochastic variance and correlation-driven skew to the model-risk inventory.

# %% [markdown] tags=["exercise"]
# ## Checkpoint exercise
#
# Repeat the American-put comparison for rates of 0%, 5%, and 10%.
#
# 1. Report European value, CRR American value, Leisen-Reimer value, early-exercise
#    premium, and the CRR/LR method difference at each rate.
# 2. Double Asian Monte Carlo paths and report naive and control-variate standard
#    errors plus their reduction factor.
# 3. Compare daily and weekly BGK-adjusted barriers for both up and down barriers.
# 4. Identify which differences are sampling error, discretization/model error,
#    and economically hedgeable exposure.
#

# %% [markdown] tags=["solution"]
# ```{dropdown} Suggested answer rubric
# A complete answer reports option values and premiums in currency units,
# separates Monte Carlo error from model-method differences, uses the correct
# barrier direction, and specifies what enters a hedge versus model governance.
# ```
