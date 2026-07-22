"""Deterministic teaching calculations and figures for Module 8.

All amounts are simulated MXN millions. Rates are effective annual decimals
unless a caller explicitly selects a simple preferred-return convention. The
calculation functions are separate from rendering so the financial identities
can be tested before the figures are published.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.visual_style import (
    AMBER_DARK,
    BACKGROUND,
    CORAL,
    INK,
    MUTED_BLUE,
    TEAL,
    add_figure_note,
    matplotlib_style,
    style_axes,
)


@dataclass(frozen=True)
class LpWaterfall:
    """Allocation of one simplified whole-fund terminal distribution."""

    contributed_capital: float
    total_proceeds: float
    return_of_capital: float
    preferred_return: float
    lp_residual_profit: float
    gp_carry: float
    lp_total: float


@dataclass(frozen=True)
class LiquidityBudget:
    """One-horizon stressed liquidity bridge without double counting."""

    opening_liquid_assets: float
    proposed_allocations: float
    post_allocation_liquid_assets: float
    unfunded_commitments: float
    available_after_commitments: float
    planned_cash_need: float
    policy_buffer: float
    denominator_need: float
    minimum_lcr: float
    required_liquid_assets: float
    liquidity_headroom: float
    lcr: float


def _finite_non_negative(value: float, name: str) -> float:
    """Return one finite non-negative scalar."""

    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real scalar") from error
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def calculate_lp_waterfall(
    *,
    contributed_capital: float,
    total_proceeds: float,
    preferred_rate: float,
    years: float,
    carry_rate: float,
    compound_preference: bool,
) -> LpWaterfall:
    """Allocate proceeds through capital, preference, and residual tiers."""

    capital = _finite_non_negative(contributed_capital, "contributed_capital")
    proceeds = _finite_non_negative(total_proceeds, "total_proceeds")
    rate = _finite_non_negative(preferred_rate, "preferred_rate")
    horizon = _finite_non_negative(years, "years")
    carry = _finite_non_negative(carry_rate, "carry_rate")
    if capital == 0:
        raise ValueError("contributed_capital must be positive")
    if carry > 1:
        raise ValueError("carry_rate must not exceed 1")

    accrued_preference = (
        capital * ((1 + rate) ** horizon - 1)
        if compound_preference
        else capital * rate * horizon
    )
    return_of_capital = min(proceeds, capital)
    remaining = max(0.0, proceeds - return_of_capital)
    preferred_return = min(remaining, accrued_preference)
    residual_profit = max(0.0, remaining - preferred_return)
    gp_carry = residual_profit * carry
    lp_residual = residual_profit - gp_carry
    lp_total = return_of_capital + preferred_return + lp_residual

    return LpWaterfall(
        contributed_capital=capital,
        total_proceeds=proceeds,
        return_of_capital=return_of_capital,
        preferred_return=preferred_return,
        lp_residual_profit=lp_residual,
        gp_carry=gp_carry,
        lp_total=lp_total,
    )


def calculate_liquidity_budget(
    *,
    opening_liquid_assets: float,
    proposed_allocations: Sequence[float],
    unfunded_commitments: float,
    planned_cash_need: float,
    policy_buffer: float,
    minimum_lcr: float,
) -> LiquidityBudget:
    """Reconcile stressed liquid assets to needs, buffer, and an LCR floor."""

    opening = _finite_non_negative(opening_liquid_assets, "opening_liquid_assets")
    allocations = np.asarray(proposed_allocations, dtype=float)
    if allocations.ndim != 1 or not np.isfinite(allocations).all():
        raise ValueError("proposed_allocations must be a finite one-dimensional sequence")
    if (allocations < 0).any():
        raise ValueError("proposed_allocations must be non-negative")
    total_allocations = float(allocations.sum())
    unfunded = _finite_non_negative(unfunded_commitments, "unfunded_commitments")
    need = _finite_non_negative(planned_cash_need, "planned_cash_need")
    buffer = _finite_non_negative(policy_buffer, "policy_buffer")
    minimum = _finite_non_negative(minimum_lcr, "minimum_lcr")
    denominator = need + buffer
    if denominator == 0:
        raise ValueError("planned_cash_need plus policy_buffer must be positive")

    post_allocation = opening - total_allocations
    available = post_allocation - unfunded
    if post_allocation < 0 or available < 0:
        raise ValueError("allocations and commitments must not exceed opening liquid assets")
    required = minimum * denominator

    return LiquidityBudget(
        opening_liquid_assets=opening,
        proposed_allocations=total_allocations,
        post_allocation_liquid_assets=post_allocation,
        unfunded_commitments=unfunded,
        available_after_commitments=available,
        planned_cash_need=need,
        policy_buffer=buffer,
        denominator_need=denominator,
        minimum_lcr=minimum,
        required_liquid_assets=required,
        liquidity_headroom=available - required,
        lcr=available / denominator,
    )


LP_WATERFALL_CASE = calculate_lp_waterfall(
    contributed_capital=12.0,
    total_proceeds=20.0,
    preferred_rate=0.08,
    years=3.0,
    carry_rate=0.20,
    compound_preference=False,
)

LIQUIDITY_CASE = calculate_liquidity_budget(
    opening_liquid_assets=850.0,
    proposed_allocations=(70.0, 40.0, 20.0),
    unfunded_commitments=50.0,
    planned_cash_need=80.0,
    policy_buffer=60.0,
    minimum_lcr=3.0,
)


def build_lp_waterfall_figure() -> Figure:
    """Show how the Module 8 example allocates gross proceeds to LP and GP."""

    case = LP_WATERFALL_CASE
    labels = ("Return of capital", "Preferred return", "LP residual", "GP carry")
    values = (
        case.return_of_capital,
        case.preferred_return,
        case.lp_residual_profit,
        case.gp_carry,
    )
    colors = (MUTED_BLUE, TEAL, AMBER_DARK, CORAL)
    hatches = ("//", "..", "xx", "\\\\")

    with matplotlib_style():
        figure, axis = plt.subplots(figsize=(10.0, 5.625))
        figure.subplots_adjust(left=0.08, right=0.97, top=0.79, bottom=0.20)
        left = 0.0
        for label, value, color, hatch in zip(labels, values, colors, hatches, strict=True):
            axis.barh(
                [0],
                [value],
                left=left,
                height=0.42,
                color=color,
                edgecolor=INK,
                linewidth=0.8,
                hatch=hatch,
                label=label,
            )
            midpoint = left + value / 2
            if label == "GP carry":
                axis.annotate(
                    f"GP carry · MXN {value:.3f}m",
                    xy=(midpoint, 0.20),
                    xytext=(midpoint, 0.36),
                    arrowprops={"arrowstyle": "-", "color": CORAL, "linewidth": 1.2},
                    color=CORAL,
                    fontsize=9,
                    fontweight="semibold",
                    ha="center",
                    va="bottom",
                )
            else:
                axis.text(
                    midpoint,
                    0,
                    f"{label}\nMXN {value:.3f}m",
                    color=(
                        BACKGROUND
                        if color in {MUTED_BLUE, TEAL, AMBER_DARK, CORAL}
                        else INK
                    ),
                    fontsize=9,
                    fontweight="semibold",
                    ha="center",
                    va="center",
                )
            left += value

        axis.set_xlim(0, 21.0)
        axis.set_ylim(-0.48, 0.48)
        axis.set_yticks([])
        axis.set_xlabel("Terminal proceeds allocated (MXN millions)")
        axis.set_title("A whole-fund waterfall converts gross proceeds into LP and GP cash flows")
        style_axes(axis, grid_axis="x")
        figure.text(
            0.03,
            0.97,
            "LP waterfall: capital, preference, residual profit, and carry",
            color=INK,
            fontsize=15,
            fontweight="semibold",
            ha="left",
            va="top",
        )
        figure.text(
            0.03,
            0.885,
            "Simulated terminal case · simple 8% annual preference for 3 years · 20% carry · no GP catch-up",
            color=INK,
            fontsize=10,
            ha="left",
        )
        add_figure_note(
            figure,
            "Source: simulated teaching assumptions in Module 8.1; all layers reconcile to MXN 20.000m. "
            "The legal agreement controls actual fee and waterfall terms.",
        )
    return figure


def build_liquidity_budget_figure() -> Figure:
    """Show the Module 8 liquidity bridge and policy headroom."""

    case = LIQUIDITY_CASE
    labels = (
        "Opening stressed liquid assets",
        "After proposed allocations",
        "After unfunded commitments",
        "Required at 3.0× LCR",
    )
    values = (
        case.opening_liquid_assets,
        case.post_allocation_liquid_assets,
        case.available_after_commitments,
        case.required_liquid_assets,
    )
    colors = (MUTED_BLUE, TEAL, TEAL, AMBER_DARK)
    hatches = ("//", "..", "xx", "\\\\")

    with matplotlib_style():
        figure, axis = plt.subplots(figsize=(10.0, 5.625))
        figure.subplots_adjust(left=0.31, right=0.96, top=0.79, bottom=0.20)
        positions = np.arange(len(labels))[::-1]
        bars = axis.barh(
            positions,
            values,
            color=colors,
            edgecolor=INK,
            linewidth=0.8,
            hatch=hatches,
            height=0.56,
        )
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                value + 12,
                bar.get_y() + bar.get_height() / 2,
                f"MXN {value:.0f}m",
                color=INK,
                fontsize=10,
                fontweight="semibold",
                ha="left",
                va="center",
            )
        axis.set_yticks(positions, labels)
        axis.set_xlim(0, 930)
        axis.set_xlabel("Stressed liquidity within the one-year policy horizon (MXN millions)")
        axis.set_title("Reserve commitments once, then compare liquidity with the policy floor")
        style_axes(axis, grid_axis="x")
        axis.set_ylim(-0.72, 3.55)
        axis.annotate(
            "",
            xy=(case.required_liquid_assets, -0.48),
            xytext=(case.available_after_commitments, -0.48),
            arrowprops={"arrowstyle": "<->", "color": CORAL, "linewidth": 1.8},
        )
        axis.text(
            (case.required_liquid_assets + case.available_after_commitments) / 2,
            -0.40,
            f"Policy headroom = MXN {case.liquidity_headroom:.0f}m",
            color=CORAL,
            fontsize=10,
            fontweight="semibold",
            ha="center",
            va="bottom",
        )
        figure.text(
            0.03,
            0.97,
            "Liquidity budget: commitments and policy coverage",
            color=INK,
            fontsize=15,
            fontweight="semibold",
            ha="left",
            va="top",
        )
        figure.text(
            0.03,
            0.885,
            f"Simulated one-year case · cash need MXN 80m · buffer MXN 60m · resulting LCR {case.lcr:.2f}×",
            color=INK,
            fontsize=10,
            ha="left",
        )
        add_figure_note(
            figure,
            "Source: simulated teaching assumptions in Module 8.5. Haircuts are already embedded in opening "
            "liquid assets; taxes, credit-line capacity, and investor-specific constraints are excluded.",
        )
    return figure
