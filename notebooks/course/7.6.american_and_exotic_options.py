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
# methods. This lesson compares Cox-Ross-Rubinstein (CRR) convergence,
# Leisen-Reimer American put pricing, Asian option control variates, and the
# Broadie-Glasserman-Kou (BGK) continuity correction for discretely monitored
# barriers
# {cite}`hull2022options,leisenReimer1996,kemnaVorst1990,broadieGlassermanKou1997`.
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
from IPython.display import display

from src.derivatives import (
    arithmetic_asian_call_control_variate,
    bgk_adjusted_barrier,
    black_scholes_price,
    crr_binomial_option_price,
    leisen_reimer_american_put,
)
from src.module7_visuals import build_convergence_figure

# %% [markdown]
# ## CRR convergence
#
# The CRR tree converges to Black-Scholes-Merton for European options under
# matched inputs. Convergence can oscillate because terminal nodes do not always
# align with the payoff kink. This deterministic synthetic comparison is an
# implementation diagnostic, not evidence about market prices.

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
convergence.attrs = {
    "source": "deterministic synthetic calculation from src.derivatives",
    "data_mode": "synthetic",
    "units": "generic option-price units",
}
convergence

# %% mystnb={"image": {"alt": "Two aligned panels show a synthetic European call's Cox-Ross-Rubinstein price converging toward its Black-Scholes-Merton benchmark and the signed price error across seven tree sizes from 5 to 500 steps."}}
convergence_figure = build_convergence_figure(convergence)
display(convergence_figure)
plt.close(convergence_figure)

# %% [markdown]
# ## American put early exercise
#
# American options require an optimal stopping check at every node:
#
# $$
# V_n = \max\!\left(\text{continuation value}_n,
#                         \text{intrinsic value}_n\right).
# $$
#
# Under otherwise identical assumptions, define the American put
# early-exercise premium as
#
# $$
# \operatorname{EEP}=P_{\mathrm{American}}-P_{\mathrm{European}}\ge 0.
# $$
#
# The Leisen-Reimer construction uses an odd-step probability transformation to
# improve lattice convergence around the payoff kink
# {cite}`leisenReimer1996`. A positive numerical premium is meaningful only
# after checking that both prices share spot, strike, carry, maturity, and
# volatility conventions.

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
assert (american_comparison["early_exercise_premium"] >= -1e-10).all()
american_comparison

# %% [markdown]
# ## Asian option control variate
#
# Arithmetic Asian options usually need simulation. A geometric Asian option is
# a useful control variate because it is highly correlated with the arithmetic
# payoff and has a known analytical price {cite}`kemnaVorst1990`. If $X_A$
# and $X_G$ are discounted arithmetic and geometric payoffs, respectively,
# the adjusted observation is
#
# $$
# X_{\mathrm{cv}}
# =X_A-\widehat{\beta}\left(X_G-\mathbb{E}[X_G]\right),
# \qquad
# \widehat{\beta}=\frac{\widehat{\operatorname{Cov}}(X_A,X_G)}
#                            {\widehat{\operatorname{Var}}(X_G)}.
# $$
#
# The estimator is the sample mean of $X_{\mathrm{cv}}$, with Monte Carlo
# standard error $s_{\mathrm{cv}}/\sqrt{M}$ for $M$ simulated paths.

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
# For discretely monitored barriers, the BGK adjustment shifts the barrier away
# from the spot to approximate the difference between continuous and discrete
# monitoring {cite}`broadieGlassermanKou1997`:
#
# $$
# H_{\mathrm{adj}}^{\mathrm{down}}
# =H\exp(-\beta\sigma\sqrt{\Delta t}),\qquad
# H_{\mathrm{adj}}^{\mathrm{up}}
# =H\exp(+\beta\sigma\sqrt{\Delta t}),\qquad
# \beta \approx 0.5826.
# $$
#
# Under identical payoff, rebate, monitoring, and settlement conventions,
# barrier claims also provide the decomposition check
#
# $$
# V_{\mathrm{down\text{-}in}}+V_{\mathrm{down\text{-}out}}
# =V_{\mathrm{vanilla}},
# $$
#
# with an analogous up-barrier identity. The cell below adjusts barriers only;
# it does **not** price the in/out claims and therefore does not, by itself,
# verify that parity.

# %%
barrier = 80.0
daily_dt = 1 / 252

pd.Series(
    {
        "physical_down_barrier_price_units": barrier,
        "bgk_adjusted_down_barrier_price_units": bgk_adjusted_barrier(
            barrier,
            volatility=0.25,
            monitoring_interval=daily_dt,
            barrier_type="down",
        ),
        "physical_up_barrier_price_units": 120.0,
        "bgk_adjusted_up_barrier_price_units": bgk_adjusted_barrier(
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
# - Every numerical value in this lesson is synthetic and uses continuously
#   compounded annual decimal rates, annualized decimal volatility, years, and
#   generic price units.
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
