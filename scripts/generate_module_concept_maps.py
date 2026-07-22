#!/usr/bin/env python3
"""Generate deterministic module concept maps for the Jupyter Book."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_hex, to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

from src.visual_style import (
    AMBER_DARK as AMBER,
    BACKGROUND,
    CORAL,
    EXPORT_FIGURE_DPI,
    INK,
    MUTED_BLUE,
    PLOT_BACKGROUND,
    SOFT_GRID,
    TEAL,
)


OUTPUT_DIR = ROOT / "img" / "generated"
CANVAS_SIZE = (1920, 1080)
FIGURE_SIZE = (16, 9)
EXPORT_DPI = EXPORT_FIGURE_DPI
RENDER_SIZE = tuple(round(side * EXPORT_DPI) for side in FIGURE_SIZE)


@dataclass(frozen=True)
class ConceptNode:
    """One concise step in a concept map."""

    title: str
    detail: str
    accent: str


@dataclass(frozen=True)
class BoxPlacement:
    """Normalized axes coordinates for one concept box."""

    x: float
    y: float
    width: float
    height: float

    def anchor(self, side: str) -> tuple[float, float]:
        anchors = {
            "left": (self.x, self.y + self.height / 2),
            "right": (self.x + self.width, self.y + self.height / 2),
            "top": (self.x + self.width / 2, self.y + self.height),
            "bottom": (self.x + self.width / 2, self.y),
        }
        return anchors[side]


@dataclass(frozen=True)
class FlowSpec:
    """Content for a numbered, snake-layout concept map."""

    title: str
    subtitle: str
    nodes: tuple[ConceptNode, ...]
    loop_back: bool = False
    closing_note: str = "Follow the arrows from evidence to action."


def blend_with_background(color: str, strength: float = 0.10) -> str:
    """Return a quiet tint of a palette color on the book background."""

    foreground = to_rgb(color)
    background = to_rgb(BACKGROUND)
    blended = tuple(
        background_channel * (1 - strength) + foreground_channel * strength
        for foreground_channel, background_channel in zip(
            foreground,
            background,
            strict=True,
        )
    )
    return to_hex(blended)


def new_canvas(title: str, subtitle: str) -> tuple[Figure, Axes]:
    """Create the shared 16:9 canvas and title block."""

    figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=EXPORT_DPI)
    figure.patch.set_facecolor(BACKGROUND)
    axis.set_facecolor(BACKGROUND)
    axis.set_position([0, 0, 1, 1])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(
        0.07,
        0.955,
        "MODULE CONCEPT MAP",
        color=MUTED_BLUE,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top",
        transform=axis.transAxes,
    )
    axis.text(
        0.07,
        0.905,
        title,
        color=INK,
        fontsize=27,
        fontweight="bold",
        ha="left",
        va="center",
        transform=axis.transAxes,
    )
    axis.text(
        0.07,
        0.848,
        subtitle,
        color=INK,
        fontsize=14,
        ha="left",
        va="center",
        transform=axis.transAxes,
    )
    axis.plot(
        [0.07, 0.93],
        [0.805, 0.805],
        color=SOFT_GRID,
        linewidth=1.5,
        transform=axis.transAxes,
        clip_on=False,
    )
    return figure, axis


def add_box(
    axis: Axes,
    node: ConceptNode,
    placement: BoxPlacement,
    *,
    step: int | None = None,
    title_size: float | None = None,
) -> None:
    """Draw a labeled concept box with an optional sequence marker."""

    x, y, width, height = (
        placement.x,
        placement.y,
        placement.width,
        placement.height,
    )
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=2.0,
        edgecolor=node.accent,
        facecolor=blend_with_background(node.accent),
        zorder=3,
        transform=axis.transAxes,
    )
    axis.add_patch(patch)
    axis.plot(
        [x + 0.018, x + width - 0.018],
        [y + height - 0.027, y + height - 0.027],
        color=node.accent,
        linewidth=3.0,
        solid_capstyle="round",
        zorder=4,
        transform=axis.transAxes,
    )

    text_x = x + 0.032
    if step is not None:
        axis.text(
            x + 0.031,
            y + height - 0.065,
            f"{step:02d}",
            color=INK,
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "circle,pad=0.34",
                "facecolor": BACKGROUND,
                "edgecolor": node.accent,
                "linewidth": 1.4,
            },
            zorder=5,
            transform=axis.transAxes,
        )
        text_x = x + 0.061

    if title_size is None:
        title_size = 15 if width <= 0.17 else 17
    axis.text(
        text_x,
        y + height * 0.62,
        node.title,
        color=INK,
        fontsize=title_size,
        fontweight="bold",
        ha="left",
        va="center",
        linespacing=1.05,
        zorder=5,
        transform=axis.transAxes,
    )
    axis.text(
        x + 0.032,
        y + height * 0.25,
        node.detail,
        color=INK,
        fontsize=11.5 if width >= 0.17 else 10.5,
        ha="left",
        va="center",
        linespacing=1.15,
        zorder=5,
        transform=axis.transAxes,
    )


def add_arrow(
    axis: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED_BLUE,
    curve: float = 0.0,
    mutation_scale: float = 18,
    shrink: float = 8,
    zorder: float = 1,
) -> None:
    """Draw a restrained directional connector behind concept boxes."""

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=2.0,
        color=color,
        shrinkA=shrink,
        shrinkB=shrink,
        connectionstyle=f"arc3,rad={curve}",
        zorder=zorder,
        transform=axis.transAxes,
    )
    axis.add_patch(arrow)


def connect_boxes(
    axis: Axes,
    first: BoxPlacement,
    second: BoxPlacement,
    *,
    curve: float = 0.0,
    mutation_scale: float = 18,
    shrink: float = 8,
    zorder: float = 1,
) -> None:
    """Connect adjacent boxes using the shortest readable edge anchors."""

    if abs(first.y - second.y) < 0.03:
        if second.x > first.x:
            start_side, end_side = "right", "left"
        else:
            start_side, end_side = "left", "right"
    elif second.y < first.y:
        start_side, end_side = "bottom", "top"
    else:
        start_side, end_side = "top", "bottom"
    add_arrow(
        axis,
        first.anchor(start_side),
        second.anchor(end_side),
        curve=curve,
        mutation_scale=mutation_scale,
        shrink=shrink,
        zorder=zorder,
    )


def add_lane_label(axis: Axes, y: float, label: str, color: str) -> None:
    """Label one conceptual lane without adding another panel."""

    axis.text(
        0.07,
        y,
        label,
        color=INK,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="center",
        transform=axis.transAxes,
    )
    axis.plot(
        [0.29, 0.93],
        [y, y],
        color=blend_with_background(color, 0.35),
        linewidth=1.4,
        transform=axis.transAxes,
    )


def flow_positions(count: int) -> list[BoxPlacement]:
    """Return a two-row reading path for six to eight steps."""

    if count == 6:
        top = [BoxPlacement(x, 0.52, 0.26, 0.21) for x in (0.07, 0.37, 0.67)]
        bottom = [BoxPlacement(x, 0.18, 0.26, 0.21) for x in (0.67, 0.37, 0.07)]
    elif count == 7:
        top = [
            BoxPlacement(x, 0.52, 0.17, 0.21)
            for x in (0.07, 0.295, 0.52, 0.745)
        ]
        bottom = [BoxPlacement(x, 0.18, 0.245, 0.21) for x in (0.67, 0.37, 0.07)]
    elif count == 8:
        top = [
            BoxPlacement(x, 0.52, 0.17, 0.21)
            for x in (0.07, 0.295, 0.52, 0.745)
        ]
        bottom = [
            BoxPlacement(x, 0.18, 0.17, 0.21)
            for x in (0.745, 0.52, 0.295, 0.07)
        ]
    else:
        raise ValueError("Flow maps support six, seven, or eight nodes.")
    return top + bottom


def render_flow(spec: FlowSpec) -> Figure:
    """Render a numbered process map with a consistent snake reading path."""

    figure, axis = new_canvas(spec.title, spec.subtitle)
    placements = flow_positions(len(spec.nodes))
    for first, second in zip(placements, placements[1:]):
        connect_boxes(
            axis,
            first,
            second,
            mutation_scale=16,
            shrink=15,
            zorder=3.5,
        )
    if spec.loop_back:
        add_arrow(
            axis,
            placements[-1].anchor("top"),
            placements[0].anchor("bottom"),
            color=TEAL,
            curve=-0.12,
        )

    for step, (node, placement) in enumerate(
        zip(spec.nodes, placements, strict=True),
        start=1,
    ):
        add_box(axis, node, placement, step=step)

    axis.text(
        0.5,
        0.085,
        spec.closing_note,
        color=MUTED_BLUE,
        fontsize=11.5,
        ha="center",
        va="center",
        transform=axis.transAxes,
    )
    return figure


def render_m4_three_statement_model() -> Figure:
    """Render evidence states and the four linked modeling surfaces."""

    figure, axis = new_canvas(
        "Three-statement model: evidence states and reconciliation",
        "Move reported evidence through controlled adjustments, then keep every forecast surface linked.",
    )
    add_lane_label(axis, 0.755, "EVIDENCE STATES", MUTED_BLUE)
    add_lane_label(axis, 0.445, "INTEGRATED FORECAST SURFACES", TEAL)

    evidence_nodes = (
        ConceptNode("Reported", "source values · disclosures", MUTED_BLUE),
        ConceptNode("Reclassified", "consistent analytical view", AMBER),
        ConceptNode("Normalized", "recurring economics", CORAL),
        ConceptNode("Forecast", "drivers · timing · scenarios", TEAL),
    )
    surface_nodes = (
        ConceptNode("Income statement", "earnings flow", TEAL),
        ConceptNode("Balance sheet", "stocks · funding", MUTED_BLUE),
        ConceptNode("Cash flow", "cash bridge", AMBER),
        ConceptNode("Equity", "retained value · claims", CORAL),
    )
    evidence_positions = [
        BoxPlacement(x, 0.51, 0.145, 0.18)
        for x in (0.055, 0.235, 0.415, 0.595)
    ]
    surface_positions = [
        BoxPlacement(x, 0.17, 0.145, 0.18)
        for x in (0.055, 0.235, 0.415, 0.595)
    ]
    checks = BoxPlacement(0.79, 0.285, 0.15, 0.28)

    for first, second in zip(evidence_positions, evidence_positions[1:]):
        connect_boxes(axis, first, second)
    for first, second in zip(surface_positions, surface_positions[1:]):
        connect_boxes(axis, first, second)
    add_arrow(axis, evidence_positions[-1].anchor("right"), checks.anchor("left"), curve=0.10)
    add_arrow(axis, surface_positions[-1].anchor("right"), checks.anchor("left"), curve=-0.10)

    for step, (node, placement) in enumerate(
        zip(evidence_nodes, evidence_positions, strict=True),
        start=1,
    ):
        add_box(axis, node, placement, step=step, title_size=13.5)
    for step, (node, placement) in enumerate(
        zip(surface_nodes, surface_positions, strict=True),
        start=1,
    ):
        add_box(axis, node, placement, step=step, title_size=13.5)
    add_box(
        axis,
        ConceptNode(
            "Reconciliation\nchecks",
            "balance · cash bridge\nretained earnings",
            TEAL,
        ),
        checks,
        title_size=14,
    )
    axis.text(
        0.5,
        0.08,
        "Reported → Reclassified → Normalized → Forecast, with linked statements and equity.",
        color=MUTED_BLUE,
        fontsize=11.5,
        ha="center",
        va="center",
        transform=axis.transAxes,
    )
    return figure


def render_m5_corporate_valuation() -> Figure:
    """Render paired firm-value and equity-value valuation routes."""

    figure, axis = new_canvas(
        "Corporate valuation: paired cash-flow routes",
        "Keep firm cash flow and equity cash flow internally consistent, then bridge value to a monitored thesis.",
    )
    band_labels = (
        ("Governance & capital allocation", TEAL, BoxPlacement(0.20, 0.735, 0.24, 0.045)),
        ("Operating forecast", MUTED_BLUE, BoxPlacement(0.47, 0.735, 0.18, 0.045)),
        ("ROIC vs WACC", AMBER, BoxPlacement(0.68, 0.735, 0.15, 0.045)),
    )
    for label, accent, placement in band_labels:
        patch = FancyBboxPatch(
            (placement.x, placement.y),
            placement.width,
            placement.height,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            linewidth=1.3,
            edgecolor=accent,
            facecolor=blend_with_background(accent, 0.08),
            zorder=3,
            transform=axis.transAxes,
        )
        axis.add_patch(patch)
        axis.text(
            placement.x + placement.width / 2,
            placement.y + placement.height / 2,
            label,
            color=INK,
            fontsize=9.5,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=4,
            transform=axis.transAxes,
        )
    for (_, _, first), (_, _, second) in zip(band_labels, band_labels[1:]):
        connect_boxes(axis, first, second)

    for y, label, color in (
        (0.665, "FIRM VALUE ROUTE", TEAL),
        (0.365, "EQUITY VALUE ROUTE", AMBER),
    ):
        axis.text(
            0.12,
            y,
            label,
            color=INK,
            fontsize=10.5,
            fontweight="bold",
            ha="left",
            va="center",
            transform=axis.transAxes,
        )
        axis.plot(
            [0.32, 0.93],
            [y, y],
            color=blend_with_background(color, 0.35),
            linewidth=1.4,
            transform=axis.transAxes,
        )

    firm_nodes = (
        ConceptNode("FCFF", "cash flow to all capital", TEAL),
        ConceptNode("WACC", "blended required return", CORAL),
        ConceptNode("Enterprise\nvalue", "operating asset value", MUTED_BLUE),
        ConceptNode("EV → equity\nbridge", "debt · NCI · cash", AMBER),
    )
    equity_nodes = (
        ConceptNode("FCFE", "cash flow to equity", TEAL),
        ConceptNode("Cost of equity", "shareholder required\nreturn", CORAL),
        ConceptNode("Equity value", "value attributable\nto owners", MUTED_BLUE),
        ConceptNode("Thesis ·\nmonitoring", "risks · catalysts · triggers", AMBER),
    )
    firm_positions = [
        BoxPlacement(x, 0.45, 0.14, 0.17)
        for x in (0.12, 0.33, 0.54, 0.75)
    ]
    equity_positions = [
        BoxPlacement(x, 0.15, 0.14, 0.17)
        for x in (0.12, 0.33, 0.54, 0.75)
    ]

    split_x = band_labels[1][2].x + band_labels[1][2].width / 2
    axis.plot(
        [split_x, split_x, 0.07, 0.07],
        [0.735, 0.705, 0.705, equity_positions[0].y + equity_positions[0].height / 2],
        color=MUTED_BLUE,
        linewidth=1.6,
        zorder=1,
        transform=axis.transAxes,
    )
    add_arrow(
        axis,
        (0.07, firm_positions[0].y + firm_positions[0].height / 2),
        firm_positions[0].anchor("left"),
        color=TEAL,
    )
    add_arrow(
        axis,
        (0.07, equity_positions[0].y + equity_positions[0].height / 2),
        equity_positions[0].anchor("left"),
        color=AMBER,
    )

    for first, second in zip(firm_positions, firm_positions[1:]):
        connect_boxes(axis, first, second)
    for first, second in zip(equity_positions, equity_positions[1:]):
        connect_boxes(axis, first, second)
    add_arrow(
        axis,
        firm_positions[-1].anchor("bottom"),
        equity_positions[2].anchor("top"),
        color=AMBER,
        curve=-0.18,
    )

    for step, (node, placement) in enumerate(
        zip(firm_nodes, firm_positions, strict=True),
        start=1,
    ):
        add_box(axis, node, placement, step=step, title_size=12.5)
    for step, (node, placement) in enumerate(
        zip(equity_nodes, equity_positions, strict=True),
        start=1,
    ):
        add_box(axis, node, placement, step=step, title_size=12.5)

    axis.text(
        0.5,
        0.08,
        "FCFF + WACC → Enterprise value | FCFE + cost of equity → Equity value.",
        color=MUTED_BLUE,
        fontsize=11.5,
        ha="center",
        va="center",
        transform=axis.transAxes,
    )
    return figure


def render_m6_fixed_income() -> Figure:
    """Render the fixed-income map with credit and structural adjustments."""

    figure, axis = new_canvas(
        "Fixed income: cash flows, credit, and sensitivities",
        "Value promised cash flows, adjust for credit, liquidity, and optionality, then control risk.",
    )
    add_lane_label(axis, 0.755, "CONTRACTUAL VALUE AND RATE INPUTS", TEAL)
    add_lane_label(axis, 0.445, "CREDIT AND STRUCTURAL ADJUSTMENTS", CORAL)

    top_nodes = (
        ConceptNode("Cash-flow timeline", "coupon · principal · dates", TEAL),
        ConceptNode("Discount curve", "time value by maturity", MUTED_BLUE),
        ConceptNode("Present value · YTM", "price and yield summary", AMBER),
    )
    bottom_nodes = (
        ConceptNode("PD · LGD", "default likelihood and loss", CORAL),
        ConceptNode("Spread · liquidity", "credit and trading compensation", AMBER),
        ConceptNode("Optionality", "cash flows may change", MUTED_BLUE),
    )
    top_positions = [BoxPlacement(x, 0.51, 0.19, 0.19) for x in (0.06, 0.295, 0.53)]
    bottom_positions = [BoxPlacement(x, 0.17, 0.19, 0.19) for x in (0.06, 0.295, 0.53)]
    outcome = BoxPlacement(0.78, 0.285, 0.16, 0.28)

    for first, second in zip(top_positions, top_positions[1:]):
        connect_boxes(axis, first, second)
    for first, second in zip(bottom_positions, bottom_positions[1:]):
        connect_boxes(axis, first, second)
    add_arrow(axis, top_positions[-1].anchor("right"), outcome.anchor("left"), curve=0.10)
    add_arrow(axis, bottom_positions[-1].anchor("right"), outcome.anchor("left"), curve=-0.10)

    for step, (node, placement) in enumerate(
        zip(top_nodes, top_positions, strict=True),
        start=1,
    ):
        add_box(axis, node, placement, step=step)
    for step, (node, placement) in enumerate(
        zip(bottom_nodes, bottom_positions, strict=True),
        start=4,
    ):
        add_box(axis, node, placement, step=step)
    add_box(
        axis,
        ConceptNode(
            "Risk control",
            "duration · DV01\nconvexity · immunization\nliability matching",
            TEAL,
        ),
        outcome,
        step=7,
        title_size=15,
    )
    axis.text(
        0.5,
        0.08,
        "Rate, spread, and cash-flow effects meet in scenario analysis.",
        color=MUTED_BLUE,
        fontsize=11.5,
        ha="center",
        va="center",
        transform=axis.transAxes,
    )
    return figure


def render_m6_term_structure() -> Figure:
    """Render separate curve-model and heterogeneous rate-panel lanes."""

    figure, axis = new_canvas(
        "Term structure: curve models and rate-panel scenarios",
        "Keep tenor-ordered curve models separate from heterogeneous rate-panel PCA.",
    )
    add_lane_label(axis, 0.755, "HOMOGENEOUS TENOR-ORDERED CURVE", TEAL)
    add_lane_label(axis, 0.425, "HETEROGENEOUS RATE PANEL", CORAL)

    curve_nodes = (
        ConceptNode("Market quotes", "bonds · swaps", MUTED_BLUE),
        ConceptNode("Bootstrap", "discount factors", TEAL),
        ConceptNode("Zero curve", "tenor-ordered rates", TEAL),
        ConceptNode("Nelson–Siegel", "level · slope · curvature", AMBER),
        ConceptNode("Curve scenarios", "parallel · twist · bend", CORAL),
        ConceptNode("Short-rate model", "calibration · simulation", MUTED_BLUE),
    )
    curve_positions = [
        BoxPlacement(0.07 + index * 0.147, 0.51, 0.125, 0.18)
        for index in range(len(curve_nodes))
    ]
    panel_nodes = (
        ConceptNode("Rate panel", "mixed instruments", MUTED_BLUE),
        ConceptNode("Standardize", "comparable changes", AMBER),
        ConceptNode("PCA", "statistical directions", TEAL),
        ConceptNode("PC1 · PC2 · PC3", "do not rename by shape", CORAL),
    )
    panel_positions = [
        BoxPlacement(x, 0.17, 0.16, 0.17)
        for x in (0.10, 0.32, 0.54, 0.76)
    ]

    for first, second in zip(curve_positions, curve_positions[1:]):
        connect_boxes(axis, first, second)
    for first, second in zip(panel_positions, panel_positions[1:]):
        connect_boxes(axis, first, second)
    for step, (node, placement) in enumerate(
        zip(curve_nodes, curve_positions, strict=True),
        start=1,
    ):
        add_box(axis, node, placement, step=step, title_size=12.5)
    for step, (node, placement) in enumerate(
        zip(panel_nodes, panel_positions, strict=True),
        start=1,
    ):
        add_box(axis, node, placement, step=step, title_size=14)

    axis.text(
        0.5,
        0.08,
        "Name rate-panel components statistically unless tenor structure supports curve labels.",
        color=MUTED_BLUE,
        fontsize=11.5,
        ha="center",
        va="center",
        transform=axis.transAxes,
    )
    return figure


def add_frontier_inset(figure: Figure) -> None:
    """Add a deliberately small, unlabeled-scale frontier schematic."""

    inset = figure.add_axes([0.407, 0.355, 0.186, 0.18])
    inset.set_facecolor(PLOT_BACKGROUND)
    inset.spines[["top", "right"]].set_visible(False)
    inset.spines[["left", "bottom"]].set_color(SOFT_GRID)
    inset.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    inset.set_xlim(0, 1)
    inset.set_ylim(0, 1)

    feasible_x = [0.18 + 0.055 * index for index in range(12)]
    feasible_y = [
        0.20 + 0.032 * index + 0.075 * math.sin(index * 1.7)
        for index in range(12)
    ]
    inset.scatter(feasible_x, feasible_y, s=18, color=SOFT_GRID, zorder=1)
    frontier_x = [0.26, 0.34, 0.43, 0.53, 0.64, 0.76, 0.88]
    frontier_y = [0.28, 0.39, 0.49, 0.59, 0.68, 0.76, 0.83]
    inset.plot(frontier_x, frontier_y, color=TEAL, linewidth=2.4, zorder=2)
    inset.scatter([0.64], [0.68], s=42, color=AMBER, edgecolor=INK, linewidth=0.8, zorder=3)
    inset.text(
        0.04,
        0.93,
        "schematic frontier check",
        color=INK,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="top",
        transform=inset.transAxes,
    )
    inset.set_xlabel("risk", color=INK, fontsize=8, labelpad=2)
    inset.set_ylabel("expected return", color=INK, fontsize=8, labelpad=2)


def render_m9_portfolio_cycle() -> Figure:
    """Render the IPS-led portfolio-management cycle with a small frontier inset."""

    figure, axis = new_canvas(
        "Portfolio management cycle",
        "Move from the IPS and risk budget through construction, implementation, evidence, and formal review.",
    )
    nodes = (
        ConceptNode("IPS &\nrisk budget", "objectives · constraints", TEAL),
        ConceptNode("Capital-market\nexpectations", "return · risk · liquidity", MUTED_BLUE),
        ConceptNode("Allocation &\nconstruction", "policy mix · frontier check", AMBER),
        ConceptNode("Implementation &\nrebalancing", "vehicles · costs · drift", MUTED_BLUE),
        ConceptNode("Performance &\nattribution", "return · contribution", TEAL),
        ConceptNode("Monitoring &\nexceptions", "limits · breaches · changes", CORAL),
        ConceptNode("IPS review", "approve · revise · escalate", AMBER),
    )
    placements = (
        BoxPlacement(0.07, 0.57, 0.20, 0.16),
        BoxPlacement(0.40, 0.625, 0.20, 0.15),
        BoxPlacement(0.73, 0.57, 0.20, 0.16),
        BoxPlacement(0.75, 0.31, 0.18, 0.16),
        BoxPlacement(0.40, 0.12, 0.20, 0.15),
        BoxPlacement(0.07, 0.20, 0.20, 0.16),
        BoxPlacement(0.07, 0.40, 0.20, 0.13),
    )
    connectors = (
        (placements[0].anchor("right"), placements[1].anchor("left"), -0.08),
        (placements[1].anchor("right"), placements[2].anchor("left"), -0.08),
        (placements[2].anchor("bottom"), placements[3].anchor("top"), 0.0),
        (placements[3].anchor("left"), placements[4].anchor("right"), -0.10),
        (placements[4].anchor("left"), placements[5].anchor("right"), -0.08),
        (placements[5].anchor("top"), placements[6].anchor("bottom"), 0.0),
        (placements[6].anchor("top"), placements[0].anchor("bottom"), 0.0),
    )
    for start, end, curve in connectors:
        add_arrow(axis, start, end, color=TEAL, curve=curve)
    for step, (node, placement) in enumerate(
        zip(nodes, placements, strict=True),
        start=1,
    ):
        add_box(axis, node, placement, step=step, title_size=13.5)

    add_frontier_inset(figure)
    axis.text(
        0.5,
        0.055,
        "IPS review closes the loop when objectives, constraints, or evidence change.",
        color=MUTED_BLUE,
        fontsize=11.5,
        ha="center",
        va="center",
        transform=axis.transAxes,
    )
    return figure


def add_capstone_pathways(axis: Axes) -> None:
    """Add the three capstone pathways as compact, non-sequential tags."""

    axis.text(
        0.07,
        0.765,
        "CAPSTONE PATHWAYS",
        color=INK,
        fontsize=9.5,
        fontweight="bold",
        ha="left",
        va="center",
        transform=axis.transAxes,
    )
    labels = ("Portfolio Management", "Private Markets", "Private Wealth")
    accents = (TEAL, AMBER, MUTED_BLUE)
    for x, label, accent in zip((0.30, 0.49, 0.68), labels, accents, strict=True):
        patch = FancyBboxPatch(
            (x, 0.745),
            0.16,
            0.04,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            linewidth=1.2,
            edgecolor=accent,
            facecolor=blend_with_background(accent, 0.08),
            zorder=2,
            transform=axis.transAxes,
        )
        axis.add_patch(patch)
        axis.text(
            x + 0.08,
            0.765,
            label,
            color=INK,
            fontsize=9.5,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=3,
            transform=axis.transAxes,
        )


def render_m10_capstone() -> Figure:
    """Render the exact capstone evidence-to-monitoring sequence."""

    spec = FlowSpec(
        title="Capstone: evidence to decision and monitoring",
        subtitle="Make transformations and assumptions explicit before recommending, deciding, and monitoring.",
        nodes=(
            ConceptNode("Observed evidence", "sources · provenance", MUTED_BLUE),
            ConceptNode("Transformations", "clean · align · derive", TEAL),
            ConceptNode("Assumptions", "drivers · scenarios · limits", AMBER),
            ConceptNode("Outputs", "results · uncertainty", TEAL),
            ConceptNode("Recommendation", "alternatives · trade-offs", MUTED_BLUE),
            ConceptNode("Decision", "owner · rationale · trigger", CORAL),
            ConceptNode("Monitoring · update", "KPI · variance · revise", AMBER),
        ),
        loop_back=True,
        closing_note="Monitoring updates observed evidence, transformations, assumptions, or action.",
    )
    figure = render_flow(spec)
    add_capstone_pathways(figure.axes[0])
    return figure


FLOW_SPECS: dict[str, FlowSpec] = {
    "m2-price-return-volume-volatility-bridge": FlowSpec(
        title="Financial time series: from prices to conditional risk",
        subtitle="Transform observations carefully, add market context, diagnose dependence, then model volatility.",
        nodes=(
            ConceptNode("Price", "ordered market observations", TEAL),
            ConceptNode("Return", "simple · log", MUTED_BLUE),
            ConceptNode("Volume", "trading activity context", AMBER),
            ConceptNode("Liquidity", "interpret price movement", AMBER),
            ConceptNode("Volatility clusters", "quiet and turbulent periods", CORAL),
            ConceptNode("Diagnostics", "ACF · PACF · residuals", MUTED_BLUE),
            ConceptNode("ARCH · GARCH", "conditional variance model", TEAL),
        ),
        closing_note="Volume and liquidity add context; they are not automatic model inputs.",
    ),
    "m7-option-payoff-greeks-map": FlowSpec(
        title="Derivatives pricing, hedging, and governance",
        subtitle="Move from contract exposure to model sensitivities, hedge execution, residual P&L, and limits.",
        nodes=(
            ConceptNode("Contract &\nunderlying", "exposure drivers", MUTED_BLUE),
            ConceptNode("Payoff &\nreplication", "call · put · no-arbitrage", TEAL),
            ConceptNode("Pricing inputs", "spot · strike · time · rates", AMBER),
            ConceptNode("Price & Greeks", "delta · gamma · vega · theta", TEAL),
            ConceptNode("Volatility &\nexercise", "smile · American features", CORAL),
            ConceptNode("Hedge", "instruments · frequency", MUTED_BLUE),
            ConceptNode("Residual P&L\n& stress", "model · basis · scenarios", CORAL),
            ConceptNode("Limits &\ngovernance", "escalation · review", AMBER),
        ),
        closing_note="A price is incomplete without hedge behavior, residual P&L, and governance.",
    ),
    "m8-alternatives-asset-vehicle-strategy-map": FlowSpec(
        title="Alternative investments: asset, vehicle, and strategy",
        subtitle="Trace asset economics through the vehicle, strategy, investor cash flows, risk, and diligence.",
        nodes=(
            ConceptNode("Asset\neconomics", "underlying cash flows", TEAL),
            ConceptNode("Vehicle & terms", "ownership · access", MUTED_BLUE),
            ConceptNode("Strategy", "manager · return drivers", TEAL),
            ConceptNode(
                "Investor cash\nflows",
                "contributions · distributions\nNAV",
                MUTED_BLUE,
            ),
            ConceptNode("Fees · carry", "investor waterfall", AMBER),
            ConceptNode("Leverage ·\nliquidity", "financing · gates · stress", CORAL),
            ConceptNode(
                "Valuation &\nportfolio role",
                "evidence · diversification",
                TEAL,
            ),
            ConceptNode("Due diligence", "asset · vehicle · manager", AMBER),
        ),
        closing_note="Due diligence connects cash flows, terms, valuation, leverage, liquidity, and manager evidence.",
    ),
}


def render_flow_asset(asset_id: str) -> Figure:
    """Render one data-driven flow asset."""

    return render_flow(FLOW_SPECS[asset_id])


RENDERERS: dict[str, Callable[[], Figure]] = {
    "m2-price-return-volume-volatility-bridge": lambda: render_flow_asset(
        "m2-price-return-volume-volatility-bridge"
    ),
    "m4-three-statement-evidence-model-map": render_m4_three_statement_model,
    "m5-corporate-value-creation-valuation-map": render_m5_corporate_valuation,
    "m6-fixed-income-cash-flow-duration-map": render_m6_fixed_income,
    "m6-yield-curve-factor-scenarios": render_m6_term_structure,
    "m7-option-payoff-greeks-map": lambda: render_flow_asset(
        "m7-option-payoff-greeks-map"
    ),
    "m8-alternatives-asset-vehicle-strategy-map": lambda: render_flow_asset(
        "m8-alternatives-asset-vehicle-strategy-map"
    ),
    "m9-efficient-frontier-risk-budget-map": render_m9_portfolio_cycle,
    "m10-capstone-evidence-decision-monitoring-map": render_m10_capstone,
}


def save_rgb(figure: Figure, output: Path) -> None:
    """Save an exact-size, opaque RGB PNG and verify the export contract."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    try:
        figure.savefig(
            temporary,
            dpi=EXPORT_DPI,
            facecolor=BACKGROUND,
            edgecolor="none",
            bbox_inches=None,
            pad_inches=0,
        )
    finally:
        plt.close(figure)

    try:
        with Image.open(temporary) as image:
            if image.size != RENDER_SIZE:
                raise RuntimeError(
                    f"Expected render size {RENDER_SIZE[0]}x{RENDER_SIZE[1]}, "
                    f"got {image.size[0]}x{image.size[1]} for {output.name}."
                )
            image.convert("RGB").resize(
                CANVAS_SIZE,
                Image.Resampling.LANCZOS,
            ).save(
                output,
                format="PNG",
                optimize=True,
                dpi=(EXPORT_DPI, EXPORT_DPI),
            )
    finally:
        temporary.unlink(missing_ok=True)

    with Image.open(output) as image:
        if image.size != CANVAS_SIZE or image.mode != "RGB":
            raise RuntimeError(
                f"Invalid final export for {output.name}: size={image.size}, mode={image.mode}."
            )


def selected_asset_ids(asset_ids: Sequence[str] | None, generate_all: bool) -> list[str]:
    """Resolve CLI selection while preserving canonical output order."""

    if generate_all or not asset_ids:
        return list(RENDERERS)
    requested = set(asset_ids)
    return [asset_id for asset_id in RENDERERS if asset_id in requested]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--asset",
        action="append",
        choices=tuple(RENDERERS),
        help="Generate one asset; repeat the option to select more than one.",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Generate all concept maps (the default when no selection is given).",
    )
    args = parser.parse_args()

    for asset_id in selected_asset_ids(args.asset, args.all):
        output = OUTPUT_DIR / f"{asset_id}.png"
        save_rgb(RENDERERS[asset_id](), output)
        print(
            f"Wrote {output.relative_to(ROOT)} "
            f"({CANVAS_SIZE[0]}x{CANVAS_SIZE[1]}, RGB)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
