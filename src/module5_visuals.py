"""Deterministic teaching figures for corporate finance and equity valuation.

All numerical inputs are simulated USD teaching assumptions. Rates are
effective annual unless a figure states otherwise. The builders keep
calculations separate from export so tests can verify the financial
relationships before publication.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle

from src.visual_style import (
    AMBER_DARK,
    BACKGROUND,
    CORAL,
    FINMATH_DIVERGING_CMAP,
    INK,
    MUTED_BLUE,
    PLOT_BACKGROUND,
    TEAL,
    add_figure_note,
    matplotlib_style,
    style_axes,
)


RISK_FREE_RATE = 0.04
EQUITY_RISK_PREMIUM = 0.055
MARGINAL_TAX_RATE = 0.25
UNLEVERED_BETA = 0.80

TARGET_DEBT_WEIGHTS = np.array([0.20, 1 / 3, 0.45, 0.55])
TARGET_DEBT_COSTS = np.array([0.060, 0.065, 0.080, 0.105])

DCF_WACC = np.array([0.08, 0.09, 0.10])
DCF_PERPETUAL_GROWTH = np.array([0.02, 0.03, 0.04])
DCF_FCFF = np.array([80.0, 88.0, 95.0])
DCF_DEBT = 400.0
DCF_NCI = 20.0
DCF_EXCESS_CASH = 80.0
DCF_DILUTED_SHARES = 100.0

PROJECT_CASH_FLOWS = np.array([-110.0, 28.0, 28.0, 28.0, 28.0, 46.0])
PROJECT_REQUIRED_RETURN = 0.10
PROJECT_TERMINAL_ADDITIONS = 18.0

FORECAST_REVENUE = 26.640
FORECAST_SERVICE_VARIABLE_COST = 7.992
FORECAST_FIXED_CASH_OPERATING_EXPENSE = 10.000
FORECAST_DEPRECIATION_AMORTIZATION = 2.000
FORECAST_CASH_TAX_RATE = 0.25
FORECAST_CAPITAL_EXPENDITURES = 2.500
FORECAST_CHANGE_IN_OPERATING_WORKING_CAPITAL = 0.600


@dataclass(frozen=True)
class CapitalStructureSensitivity:
    """Calculated required returns for ordered target leverage scenarios."""

    debt_weights: np.ndarray
    debt_to_equity: np.ndarray
    debt_costs: np.ndarray
    levered_beta: np.ndarray
    equity_cost: np.ndarray
    after_tax_debt_cost: np.ndarray
    wacc: np.ndarray


@dataclass(frozen=True)
class FcffAnatomy:
    """Reconciled simulated operating forecast from revenue to FCFF."""

    revenue: float
    service_variable_cost: float
    gross_profit: float
    fixed_cash_operating_expense: float
    ebitda: float
    depreciation_amortization: float
    ebit: float
    cash_tax: float
    nopat: float
    capital_expenditures: float
    change_in_operating_working_capital: float
    fcff: float


def _finite_scalar(value: float, name: str) -> float:
    """Return one finite float or raise an actionable validation error."""

    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real scalar") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_vector(
    values: Sequence[float] | np.ndarray,
    name: str,
    *,
    minimum_size: int = 1,
) -> np.ndarray:
    """Return a finite one-dimensional float vector with a minimum size."""

    result = np.asarray(values, dtype=float)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if result.size < minimum_size:
        raise ValueError(f"{name} must contain at least {minimum_size} values")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


def calculate_capital_structure_sensitivity(
    debt_weights: Sequence[float] | np.ndarray,
    debt_costs: Sequence[float] | np.ndarray,
    *,
    risk_free_rate: float,
    equity_risk_premium: float,
    marginal_tax_rate: float,
    unlevered_beta: float,
) -> CapitalStructureSensitivity:
    """Relever beta and calculate component costs and WACC at each debt weight."""

    weights = _finite_vector(debt_weights, "debt_weights")
    costs = _finite_vector(debt_costs, "debt_costs")
    if weights.size != costs.size:
        raise ValueError("debt_weights and debt_costs must have the same length")
    if (weights < 0).any() or (weights >= 1).any():
        raise ValueError("debt_weights must be in the interval [0, 1)")
    if weights.size > 1 and (np.diff(weights) <= 0).any():
        raise ValueError("debt_weights must be strictly increasing")
    if (costs < 0).any():
        raise ValueError("debt_costs must be non-negative")

    risk_free = _finite_scalar(risk_free_rate, "risk_free_rate")
    premium = _finite_scalar(equity_risk_premium, "equity_risk_premium")
    tax_rate = _finite_scalar(marginal_tax_rate, "marginal_tax_rate")
    beta = _finite_scalar(unlevered_beta, "unlevered_beta")
    if risk_free <= -1:
        raise ValueError("risk_free_rate must be greater than -1")
    if premium < 0:
        raise ValueError("equity_risk_premium must be non-negative")
    if not 0 <= tax_rate < 1:
        raise ValueError("marginal_tax_rate must be in the interval [0, 1)")
    if beta < 0:
        raise ValueError("unlevered_beta must be non-negative")

    debt_to_equity = weights / (1 - weights)
    levered_beta = beta * (1 + (1 - tax_rate) * debt_to_equity)
    equity_cost = risk_free + levered_beta * premium
    after_tax_debt_cost = costs * (1 - tax_rate)
    wacc = (1 - weights) * equity_cost + weights * after_tax_debt_cost
    return CapitalStructureSensitivity(
        debt_weights=weights,
        debt_to_equity=debt_to_equity,
        debt_costs=costs,
        levered_beta=levered_beta,
        equity_cost=equity_cost,
        after_tax_debt_cost=after_tax_debt_cost,
        wacc=wacc,
    )


def calculate_dcf_value_per_share(
    wacc: float,
    growth: float,
    *,
    fcff: Sequence[float] | np.ndarray,
    debt: float,
    nci: float,
    excess_cash: float,
    diluted_shares: float,
) -> float:
    """Calculate DCF value per diluted share for one WACC-growth pair."""

    discount_rate = _finite_scalar(wacc, "wacc")
    perpetual_growth = _finite_scalar(growth, "growth")
    cash_flows = _finite_vector(fcff, "fcff")
    debt_value = _finite_scalar(debt, "debt")
    nci_value = _finite_scalar(nci, "nci")
    cash_value = _finite_scalar(excess_cash, "excess_cash")
    shares = _finite_scalar(diluted_shares, "diluted_shares")
    if discount_rate <= perpetual_growth:
        raise ValueError("wacc must be greater than perpetual growth")
    if discount_rate <= -1 or perpetual_growth <= -1:
        raise ValueError("wacc and growth must be greater than -1")
    if shares <= 0:
        raise ValueError("diluted_shares must be positive")

    explicit_value = sum(
        cash_flow / (1 + discount_rate) ** period
        for period, cash_flow in enumerate(cash_flows, start=1)
    )
    terminal_value = cash_flows[-1] * (1 + perpetual_growth) / (discount_rate - perpetual_growth)
    enterprise_value = explicit_value + terminal_value / (1 + discount_rate) ** len(cash_flows)
    common_equity_value = enterprise_value - debt_value - nci_value + cash_value
    return float(common_equity_value / shares)


def calculate_dcf_sensitivity(
    wacc_values: Sequence[float] | np.ndarray,
    growth_values: Sequence[float] | np.ndarray,
    *,
    fcff: Sequence[float] | np.ndarray,
    debt: float,
    nci: float,
    excess_cash: float,
    diluted_shares: float,
) -> np.ndarray:
    """Return a WACC-by-growth matrix of DCF value per diluted share."""

    discount_rates = _finite_vector(wacc_values, "wacc_values")
    growth_rates = _finite_vector(growth_values, "growth_values")
    if discount_rates.size > 1 and (np.diff(discount_rates) <= 0).any():
        raise ValueError("wacc_values must be strictly increasing")
    if growth_rates.size > 1 and (np.diff(growth_rates) <= 0).any():
        raise ValueError("growth_values must be strictly increasing")
    return np.array(
        [
            [
                calculate_dcf_value_per_share(
                    wacc,
                    growth,
                    fcff=fcff,
                    debt=debt,
                    nci=nci,
                    excess_cash=excess_cash,
                    diluted_shares=diluted_shares,
                )
                for growth in growth_rates
            ]
            for wacc in discount_rates
        ]
    )


def calculate_project_npv(
    cash_flows: Sequence[float] | np.ndarray,
    discount_rate: float,
) -> float:
    """Calculate time-zero NPV for equally spaced annual cash flows."""

    values = _finite_vector(cash_flows, "cash_flows", minimum_size=2)
    rate = _finite_scalar(discount_rate, "discount_rate")
    if rate <= -1:
        raise ValueError("discount_rate must be greater than -1")
    discount_factors = (1 + rate) ** np.arange(values.size)
    return float(np.sum(values / discount_factors))


def calculate_project_irr(cash_flows: Sequence[float] | np.ndarray) -> float:
    """Calculate the unique conventional-project IRR by deterministic bisection."""

    values = _finite_vector(cash_flows, "cash_flows", minimum_size=2)
    nonzero_signs = np.sign(values[values != 0])
    if nonzero_signs.size < 2 or np.sum(nonzero_signs[1:] != nonzero_signs[:-1]) != 1:
        raise ValueError("cash_flows must have exactly one sign change for a unique IRR")

    lower = -0.9999
    upper = 1.0
    lower_npv = calculate_project_npv(values, lower)
    upper_npv = calculate_project_npv(values, upper)
    while lower_npv * upper_npv > 0 and upper < 1_000_000:
        upper = 2 * upper + 1
        upper_npv = calculate_project_npv(values, upper)
    if lower_npv * upper_npv > 0:
        raise ValueError("cash_flows do not have a bracketed IRR above -100%")

    for _ in range(200):
        midpoint = (lower + upper) / 2
        midpoint_npv = calculate_project_npv(values, midpoint)
        if lower_npv * midpoint_npv <= 0:
            upper = midpoint
        else:
            lower = midpoint
            lower_npv = midpoint_npv
    return float((lower + upper) / 2)


def calculate_fcff_anatomy(
    *,
    revenue: float,
    service_variable_cost: float,
    fixed_cash_operating_expense: float,
    depreciation_amortization: float,
    cash_tax_rate: float,
    capital_expenditures: float,
    change_in_operating_working_capital: float,
    recognize_immediate_tax_loss_benefit: bool = False,
) -> FcffAnatomy:
    """Reconcile a simulated operating forecast from revenue through FCFF.

    By default, a negative EBIT does not create an immediate tax refund. Set
    ``recognize_immediate_tax_loss_benefit`` only when tax capacity and timing
    make the loss benefit usable in the modeled period.
    """

    names = (
        "revenue",
        "service_variable_cost",
        "fixed_cash_operating_expense",
        "depreciation_amortization",
        "capital_expenditures",
        "change_in_operating_working_capital",
    )
    raw_values = (
        revenue,
        service_variable_cost,
        fixed_cash_operating_expense,
        depreciation_amortization,
        capital_expenditures,
        change_in_operating_working_capital,
    )
    checked = {
        name: _finite_scalar(value, name) for name, value in zip(names, raw_values, strict=True)
    }
    if checked["revenue"] <= 0:
        raise ValueError("revenue must be positive")
    nonnegative_names = (
        "service_variable_cost",
        "fixed_cash_operating_expense",
        "depreciation_amortization",
        "capital_expenditures",
    )
    if any(checked[name] < 0 for name in nonnegative_names):
        raise ValueError("forecast costs and capital expenditures must be non-negative")
    tax_rate = _finite_scalar(cash_tax_rate, "cash_tax_rate")
    if not 0 <= tax_rate <= 1:
        raise ValueError("cash_tax_rate must be in the interval [0, 1]")

    gross_profit = checked["revenue"] - checked["service_variable_cost"]
    ebitda = gross_profit - checked["fixed_cash_operating_expense"]
    ebit = ebitda - checked["depreciation_amortization"]
    taxable_ebit = ebit if recognize_immediate_tax_loss_benefit else max(ebit, 0.0)
    cash_tax = taxable_ebit * tax_rate
    nopat = ebit - cash_tax
    fcff = (
        nopat
        + checked["depreciation_amortization"]
        - checked["capital_expenditures"]
        - checked["change_in_operating_working_capital"]
    )
    return FcffAnatomy(
        revenue=checked["revenue"],
        service_variable_cost=checked["service_variable_cost"],
        gross_profit=gross_profit,
        fixed_cash_operating_expense=checked["fixed_cash_operating_expense"],
        ebitda=ebitda,
        depreciation_amortization=checked["depreciation_amortization"],
        ebit=ebit,
        cash_tax=cash_tax,
        nopat=nopat,
        capital_expenditures=checked["capital_expenditures"],
        change_in_operating_working_capital=checked["change_in_operating_working_capital"],
        fcff=fcff,
    )


TARGET_SENSITIVITY = calculate_capital_structure_sensitivity(
    TARGET_DEBT_WEIGHTS,
    TARGET_DEBT_COSTS,
    risk_free_rate=RISK_FREE_RATE,
    equity_risk_premium=EQUITY_RISK_PREMIUM,
    marginal_tax_rate=MARGINAL_TAX_RATE,
    unlevered_beta=UNLEVERED_BETA,
)
TARGET_DEBT_TO_EQUITY = TARGET_SENSITIVITY.debt_to_equity
TARGET_LEVERED_BETA = TARGET_SENSITIVITY.levered_beta
TARGET_EQUITY_COST = TARGET_SENSITIVITY.equity_cost
TARGET_AFTER_TAX_DEBT_COST = TARGET_SENSITIVITY.after_tax_debt_cost
TARGET_WACC = TARGET_SENSITIVITY.wacc

DCF_VALUE_PER_SHARE = calculate_dcf_sensitivity(
    DCF_WACC,
    DCF_PERPETUAL_GROWTH,
    fcff=DCF_FCFF,
    debt=DCF_DEBT,
    nci=DCF_NCI,
    excess_cash=DCF_EXCESS_CASH,
    diluted_shares=DCF_DILUTED_SHARES,
)

PROJECT_NPV = calculate_project_npv(PROJECT_CASH_FLOWS, PROJECT_REQUIRED_RETURN)
PROJECT_IRR = calculate_project_irr(PROJECT_CASH_FLOWS)

FORECAST_FCFF_ANATOMY = calculate_fcff_anatomy(
    revenue=FORECAST_REVENUE,
    service_variable_cost=FORECAST_SERVICE_VARIABLE_COST,
    fixed_cash_operating_expense=FORECAST_FIXED_CASH_OPERATING_EXPENSE,
    depreciation_amortization=FORECAST_DEPRECIATION_AMORTIZATION,
    cash_tax_rate=FORECAST_CASH_TAX_RATE,
    capital_expenditures=FORECAST_CAPITAL_EXPENDITURES,
    change_in_operating_working_capital=(FORECAST_CHANGE_IN_OPERATING_WORKING_CAPITAL),
)


def build_capital_structure_wacc_figure() -> Figure:
    """Show how required returns change across target leverage scenarios."""

    with matplotlib_style():
        figure, axis = plt.subplots(figsize=(10, 5.625))
        figure.patch.set_facecolor(BACKGROUND)
        axis.set_facecolor(PLOT_BACKGROUND)

        x_values = TARGET_DEBT_WEIGHTS * 100
        axis.plot(
            x_values,
            TARGET_WACC * 100,
            color=TEAL,
            marker="o",
            markersize=7,
            linewidth=2.4,
            label="WACC",
        )
        axis.plot(
            x_values,
            TARGET_EQUITY_COST * 100,
            color=MUTED_BLUE,
            marker="s",
            markersize=6,
            linestyle="--",
            label="Cost of equity",
        )
        axis.plot(
            x_values,
            TARGET_AFTER_TAX_DEBT_COST * 100,
            color=AMBER_DARK,
            marker="^",
            markersize=6,
            linestyle=":",
            label="After-tax cost of debt",
        )

        for x_value, wacc in zip(x_values, TARGET_WACC * 100, strict=True):
            axis.annotate(
                f"{wacc:.2f}%",
                (x_value, wacc),
                xytext=(0, -12),
                textcoords="offset points",
                color=INK,
                fontsize=9,
                fontweight="semibold",
                ha="center",
                va="top",
            )

        axis.axvline(
            x_values[1],
            color=CORAL,
            linewidth=1.0,
            linestyle=(0, (3, 3)),
            alpha=0.75,
            label="Illustrated base target",
        )
        style_axes(axis, grid_axis="y")
        axis.set_xlim(16, 59)
        axis.set_ylim(0, 14)
        axis.set_xticks(x_values, ["20%", "33%", "45%", "55%"])
        axis.set_xlabel("Target debt / capital (%)")
        axis.set_ylabel("Effective annual USD rate (%)")
        axis.legend(loc="upper left", ncols=2)

        figure.suptitle(
            "Required returns across target capital structures",
            x=0.08,
            y=0.965,
            ha="left",
            color=INK,
        )
        figure.text(
            0.08,
            0.905,
            "Recalculate beta and debt spread at each leverage point; do not hold financing costs fixed.",
            color=INK,
            fontsize=10.5,
            ha="left",
        )
        add_figure_note(
            figure,
            "Source: simulated teaching scenarios. Effective annual USD assumptions: "
            "risk-free rate 4.0%, equity risk premium 5.5%, marginal tax rate 25%, "
            "unlevered beta 0.80, and negligible debt beta. Values are conditional, "
            "not an estimate for an issuer.",
        )
        figure.subplots_adjust(left=0.10, right=0.97, top=0.80, bottom=0.20)
        return figure


def build_dcf_value_sensitivity_figure() -> Figure:
    """Render the worked DCF value-per-share sensitivity as an annotated matrix."""

    with matplotlib_style():
        figure, axis = plt.subplots(figsize=(10, 5.625))
        figure.patch.set_facecolor(BACKGROUND)
        axis.set_facecolor(PLOT_BACKGROUND)

        base_value = DCF_VALUE_PER_SHARE[1, 1]
        image = axis.imshow(
            DCF_VALUE_PER_SHARE,
            cmap=FINMATH_DIVERGING_CMAP,
            norm=TwoSlopeNorm(
                vmin=float(DCF_VALUE_PER_SHARE.min()),
                vcenter=float(base_value),
                vmax=float(DCF_VALUE_PER_SHARE.max()),
            ),
            aspect="auto",
        )

        for row in range(DCF_VALUE_PER_SHARE.shape[0]):
            for column in range(DCF_VALUE_PER_SHARE.shape[1]):
                value = DCF_VALUE_PER_SHARE[row, column]
                far_from_center = abs(value - base_value) > 3.0
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    color=PLOT_BACKGROUND if far_from_center else INK,
                    fontsize=12,
                    fontweight="bold" if (row, column) == (1, 1) else "semibold",
                    ha="center",
                    va="center",
                )

        axis.add_patch(
            Rectangle(
                (0.5, 0.5),
                1,
                1,
                fill=False,
                edgecolor=INK,
                linewidth=2.0,
            )
        )
        column_positions = np.arange(DCF_PERPETUAL_GROWTH.size)
        axis.set_xticks(column_positions)
        axis.set_xticklabels([f"{growth:.1%}" for growth in DCF_PERPETUAL_GROWTH])
        axis.set_yticks(
            range(len(DCF_WACC)),
            [f"{wacc:.1%}" for wacc in DCF_WACC],
        )
        axis.set_xlabel("Perpetual nominal growth, $g$")
        axis.set_ylabel("Effective annual USD WACC")
        axis.set_xlim(-0.5, DCF_PERPETUAL_GROWTH.size - 0.5)
        axis.tick_params(length=0)

        colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.04)
        colorbar.set_label("USD per diluted share")
        colorbar.outline.set_edgecolor(INK)

        figure.suptitle(
            "DCF value per share across WACC and terminal growth",
            x=0.08,
            y=0.965,
            ha="left",
            color=INK,
        )
        figure.text(
            0.08,
            0.905,
            "The outlined 9.0% / 3.0% cell is the worked base case; color shows direction from that case.",
            color=INK,
            fontsize=10.5,
            ha="left",
        )
        add_figure_note(
            figure,
            "Source: simulated teaching example with effective annual USD WACC. "
            "FCFF = USD 80m, 88m, and 95m in Years 1-3; "
            "debt = USD 400m; NCI = USD 20m; excess cash = USD 80m; diluted shares = 100m. "
            "This is conditional sensitivity, not a probability distribution.",
        )
        figure.subplots_adjust(left=0.17, right=0.90, top=0.80, bottom=0.22)
        return figure


def build_project_cash_flow_npv_profile_figure() -> Figure:
    """Pair the expansion-project cash-flow timeline with its NPV profile."""

    profile_rates = np.linspace(0.0, 0.25, 301)
    profile_values = np.array(
        [calculate_project_npv(PROJECT_CASH_FLOWS, rate) for rate in profile_rates]
    )
    years = np.arange(PROJECT_CASH_FLOWS.size)

    with matplotlib_style():
        figure, (cash_axis, npv_axis) = plt.subplots(
            1,
            2,
            figsize=(10, 5.625),
            gridspec_kw={"width_ratios": (1.08, 1)},
        )
        figure.patch.set_facecolor(BACKGROUND)

        cash_axis.bar(
            [0],
            [PROJECT_CASH_FLOWS[0]],
            width=0.64,
            color=CORAL,
            edgecolor=INK,
            linewidth=0.7,
            hatch="//",
            label="Initial investment",
        )
        cash_axis.bar(
            years[1:],
            np.repeat(28.0, 5),
            width=0.64,
            color=TEAL,
            edgecolor=INK,
            linewidth=0.7,
            hatch="..",
            label="After-tax operating cash flow",
        )
        cash_axis.bar(
            [5],
            [PROJECT_TERMINAL_ADDITIONS],
            bottom=[28.0],
            width=0.64,
            color=AMBER_DARK,
            edgecolor=INK,
            linewidth=0.7,
            hatch="xx",
            label="Working-capital recovery + salvage",
        )
        cash_axis.axhline(0, color=INK, linewidth=0.9)
        cash_axis.annotate(
            "−110",
            (0, PROJECT_CASH_FLOWS[0]),
            xytext=(0, -15),
            textcoords="offset points",
            color=INK,
            fontsize=9,
            fontweight="semibold",
            ha="center",
        )
        for year in years[1:5]:
            cash_axis.annotate(
                "+28",
                (year, 28),
                xytext=(0, 5),
                textcoords="offset points",
                color=INK,
                fontsize=9,
                fontweight="semibold",
                ha="center",
            )
        cash_axis.annotate(
            "+46 total\n(28 + 18 terminal)",
            (5, 46),
            xytext=(0, 6),
            textcoords="offset points",
            color=INK,
            fontsize=8.3,
            fontweight="semibold",
            ha="center",
        )
        cash_axis.text(
            0.72,
            -43,
            "Payback conventions\n"
            "4.00 years = end-of-year cash flows\n"
            "3.93 years = uniform-within-year approximation",
            color=INK,
            fontsize=8.2,
            ha="left",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": BACKGROUND,
                "edgecolor": MUTED_BLUE,
                "linewidth": 0.8,
            },
        )
        cash_axis.set_title("Incremental cash-flow timeline", loc="left")
        cash_axis.set_xlabel("Year (end-of-year cash-flow convention)")
        cash_axis.set_ylabel("Incremental cash flow (USD millions)")
        cash_axis.set_xticks(years)
        cash_axis.set_xlim(-0.6, 5.7)
        cash_axis.set_ylim(-135, 66)
        cash_axis.legend(loc="lower right", fontsize=7.8)
        style_axes(cash_axis, grid_axis="y")

        npv_axis.plot(
            profile_rates * 100,
            profile_values,
            color=TEAL,
            linewidth=2.4,
            marker="o",
            markevery=50,
            markersize=4.5,
            label="NPV profile",
        )
        npv_axis.axhline(0, color=INK, linewidth=0.9)
        npv_axis.axvline(
            PROJECT_REQUIRED_RETURN * 100,
            color=MUTED_BLUE,
            linewidth=1.2,
            linestyle="--",
        )
        npv_axis.axvline(
            PROJECT_IRR * 100,
            color=AMBER_DARK,
            linewidth=1.2,
            linestyle=":",
        )
        npv_axis.scatter(
            [PROJECT_REQUIRED_RETURN * 100],
            [PROJECT_NPV],
            color=MUTED_BLUE,
            edgecolor=INK,
            marker="s",
            s=48,
            zorder=4,
        )
        npv_axis.scatter(
            [PROJECT_IRR * 100],
            [0],
            color=AMBER_DARK,
            edgecolor=INK,
            marker="D",
            s=48,
            zorder=4,
        )
        npv_axis.annotate(
            f"10.00% required return\nNPV = USD {PROJECT_NPV:.2f}m",
            (PROJECT_REQUIRED_RETURN * 100, PROJECT_NPV),
            xytext=(-12, 20),
            textcoords="offset points",
            color=INK,
            fontsize=8.5,
            ha="right",
            bbox={
                "boxstyle": "round,pad=0.24",
                "facecolor": PLOT_BACKGROUND,
                "edgecolor": "none",
                "alpha": 0.94,
            },
        )
        npv_axis.annotate(
            f"IRR = {PROJECT_IRR:.2%}\nNPV = 0",
            (PROJECT_IRR * 100, 0),
            xytext=(12, -32),
            textcoords="offset points",
            color=INK,
            fontsize=8.5,
            ha="left",
        )
        npv_axis.set_title("NPV declines as the required return rises", loc="left")
        npv_axis.set_xlabel("Effective annual USD required return (%)")
        npv_axis.set_ylabel("NPV at time zero (USD millions)")
        npv_axis.set_xlim(0, 25)
        style_axes(npv_axis, grid_axis="both")

        figure.suptitle(
            "Expansion project: timing, NPV, and IRR",
            x=0.08,
            y=0.965,
            ha="left",
            color=INK,
        )
        figure.text(
            0.08,
            0.905,
            "The same simulated cash flows support both the timeline and the discount-rate decision profile.",
            color=INK,
            fontsize=10.5,
            ha="left",
        )
        add_figure_note(
            figure,
            "Source: simulated annual USD teaching example with an effective annual required "
            "return. Cash flows (USD m) = "
            "[-110, 28, 28, 28, 28, 46]; Year 5 includes USD 10m working-capital "
            "recovery and USD 8m after-tax salvage. NPV at 10% = USD 7.32m; IRR = "
            "12.39%. The 3.93-year payback assumes uniform cash generation within Year 4; "
            "with end-of-year cash flows, payback is 4.00 years.",
        )
        figure.subplots_adjust(left=0.09, right=0.98, top=0.79, bottom=0.24, wspace=0.28)
        return figure


def _draw_waterfall(
    axis: plt.Axes,
    *,
    labels: Sequence[str],
    bottoms: Sequence[float],
    heights: Sequence[float],
    annotations: Sequence[str],
    colors: Sequence[str],
    hatches: Sequence[str],
    connector_values: Sequence[float],
) -> None:
    """Draw one deterministic waterfall panel from pre-reconciled values."""

    positions = np.arange(len(labels))
    for position, bottom, height, annotation, color, hatch in zip(
        positions,
        bottoms,
        heights,
        annotations,
        colors,
        hatches,
        strict=True,
    ):
        axis.bar(
            position,
            height,
            bottom=bottom,
            width=0.66,
            color=color,
            edgecolor=INK,
            linewidth=0.7,
            hatch=hatch,
        )
        axis.annotate(
            annotation,
            (position, bottom + height),
            xytext=(0, 4),
            textcoords="offset points",
            color=INK,
            fontsize=8.2,
            fontweight="semibold",
            ha="center",
        )
    for position, value in enumerate(connector_values):
        axis.plot(
            [position + 0.33, position + 0.67],
            [value, value],
            color=MUTED_BLUE,
            linewidth=0.9,
        )
    axis.set_xticks(positions, labels)
    axis.tick_params(axis="x", labelsize=7.7, length=0)
    style_axes(axis, grid_axis="y", show_zero_line=True)


def build_fcff_forecast_waterfall_figure() -> Figure:
    """Show the simulated subscription forecast from revenue through FCFF."""

    values = FORECAST_FCFF_ANATOMY
    profit_labels = (
        "Revenue",
        "Service +\nvariable\ncost",
        "Gross\nprofit",
        "Fixed cash\nopex",
        "EBITDA",
        "D&A",
        "EBIT",
        "Cash tax",
        "NOPAT",
    )
    profit_bottoms = (
        0.0,
        values.gross_profit,
        0.0,
        values.ebitda,
        0.0,
        values.ebit,
        0.0,
        values.nopat,
        0.0,
    )
    profit_heights = (
        values.revenue,
        values.service_variable_cost,
        values.gross_profit,
        values.fixed_cash_operating_expense,
        values.ebitda,
        values.depreciation_amortization,
        values.ebit,
        values.cash_tax,
        values.nopat,
    )
    profit_annotations = (
        f"{values.revenue:.3f}",
        f"−{values.service_variable_cost:.3f}",
        f"{values.gross_profit:.3f}",
        f"−{values.fixed_cash_operating_expense:.3f}",
        f"{values.ebitda:.3f}",
        f"−{values.depreciation_amortization:.3f}",
        f"{values.ebit:.3f}",
        f"−{values.cash_tax:.3f}",
        f"{values.nopat:.3f}",
    )
    profit_colors = (
        TEAL,
        CORAL,
        TEAL,
        CORAL,
        TEAL,
        CORAL,
        TEAL,
        CORAL,
        TEAL,
    )
    profit_hatches = ("..", "//", "..", "//", "..", "//", "..", "//", "..")
    profit_connectors = (
        values.revenue,
        values.gross_profit,
        values.gross_profit,
        values.ebitda,
        values.ebitda,
        values.ebit,
        values.ebit,
        values.nopat,
    )

    cash_labels = (
        "NOPAT",
        "D&A\nadd-back",
        "Capital\nexpenditure",
        "Working\ncapital\nincrease",
        "FCFF",
    )
    cash_before_reinvestment = values.nopat + values.depreciation_amortization
    cash_after_capex = cash_before_reinvestment - values.capital_expenditures
    cash_bottoms = (
        0.0,
        values.nopat,
        cash_after_capex,
        values.fcff,
        0.0,
    )
    cash_heights = (
        values.nopat,
        values.depreciation_amortization,
        values.capital_expenditures,
        values.change_in_operating_working_capital,
        values.fcff,
    )
    cash_annotations = (
        f"{values.nopat:.3f}",
        f"+{values.depreciation_amortization:.3f}",
        f"−{values.capital_expenditures:.3f}",
        f"−{values.change_in_operating_working_capital:.3f}",
        f"{values.fcff:.3f}",
    )
    cash_colors = (TEAL, AMBER_DARK, CORAL, CORAL, TEAL)
    cash_hatches = ("..", "xx", "//", "//", "..")
    cash_connectors = (
        values.nopat,
        cash_before_reinvestment,
        cash_after_capex,
        values.fcff,
    )

    with matplotlib_style():
        figure, (profit_axis, cash_axis) = plt.subplots(
            1,
            2,
            figsize=(10, 5.625),
            gridspec_kw={"width_ratios": (1.68, 1)},
        )
        figure.patch.set_facecolor(BACKGROUND)
        _draw_waterfall(
            profit_axis,
            labels=profit_labels,
            bottoms=profit_bottoms,
            heights=profit_heights,
            annotations=profit_annotations,
            colors=profit_colors,
            hatches=profit_hatches,
            connector_values=profit_connectors,
        )
        _draw_waterfall(
            cash_axis,
            labels=cash_labels,
            bottoms=cash_bottoms,
            heights=cash_heights,
            annotations=cash_annotations,
            colors=cash_colors,
            hatches=cash_hatches,
            connector_values=cash_connectors,
        )
        profit_axis.set_title("Operating forecast to NOPAT", loc="left")
        profit_axis.set_ylabel("USD millions (annual simulation)")
        profit_axis.set_ylim(0, 30)
        cash_axis.set_title("Cash conversion: NOPAT to FCFF", loc="left")
        cash_axis.tick_params(axis="x", labelsize=7.2)
        cash_axis.set_ylim(0, 8)

        figure.suptitle(
            "Forecast anatomy: revenue to FCFF",
            x=0.08,
            y=0.965,
            ha="left",
            color=INK,
        )
        figure.text(
            0.08,
            0.905,
            "D&A reduces EBIT, then is added back before capital expenditure and working-capital reinvestment.",
            color=INK,
            fontsize=10.5,
            ha="left",
        )
        figure.legend(
            handles=(
                Patch(facecolor=TEAL, edgecolor=INK, hatch="..", label="Checkpoint total"),
                Patch(facecolor=CORAL, edgecolor=INK, hatch="//", label="Cost or cash use"),
                Patch(
                    facecolor=AMBER_DARK,
                    edgecolor=INK,
                    hatch="xx",
                    label="Non-cash D&A add-back",
                ),
            ),
            loc="upper center",
            bbox_to_anchor=(0.62, 0.855),
            ncols=3,
            fontsize=8.2,
        )
        add_figure_note(
            figure,
            "Source: simulated annual subscription-business teaching example, USD millions. "
            "Revenue 26.640 less service and variable cost 7.992 and fixed cash operating "
            "expense 10.000 gives EBITDA 8.648; after D&A 2.000 and 25% cash tax, NOPAT "
            "is 4.986. Add back D&A, then deduct capital expenditure 2.500 and the 0.600 "
            "increase in operating working capital to obtain FCFF 3.886.",
        )
        figure.subplots_adjust(left=0.08, right=0.98, top=0.74, bottom=0.29, wspace=0.25)
        return figure
