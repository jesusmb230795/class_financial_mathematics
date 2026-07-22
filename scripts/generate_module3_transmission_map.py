#!/usr/bin/env python3
"""Generate the Module 3 macro-to-investment transmission map."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visual_style import AMBER_DARK, BACKGROUND, CORAL, INK, MUTED_BLUE, TEAL


DEFAULT_OUTPUT = (
    ROOT / "img" / "generated" / "m3-macro-transmission-currency-map.png"
)


def _box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    edge_color: str,
    face_color: str = BACKGROUND,
    font_size: int = 18,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=3,
        edgecolor=edge_color,
        facecolor=face_color,
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        color=INK,
        fontsize=font_size,
        fontweight="semibold",
        ha="center",
        va="center",
        linespacing=1.25,
    )


def _arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=24,
            linewidth=2.6,
            color=color,
            connectionstyle="arc3,rad=0",
        )
    )


def build_transmission_map(output: Path) -> None:
    """Render an exact 1920x1080 PNG with the repository visual language."""

    figure, axis = plt.subplots(figsize=(12, 6.75), dpi=160)
    figure.patch.set_facecolor(BACKGROUND)
    axis.set_facecolor(BACKGROUND)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(
        0.06,
        0.91,
        "Macro evidence to investment decision",
        color=INK,
        fontsize=24,
        fontweight="bold",
        ha="left",
        va="center",
    )
    axis.text(
        0.06,
        0.855,
        "Preserve source, vintage, units, horizon, mechanism, and uncertainty.",
        color=INK,
        fontsize=13,
        ha="left",
        va="center",
    )

    _box(
        axis,
        (0.04, 0.37),
        0.19,
        0.25,
        "Observed\nreleases and\nvintages",
        edge_color=TEAL,
        font_size=14,
    )
    _box(
        axis,
        (0.29, 0.37),
        0.22,
        0.25,
        "Economic mechanism\nand scenario\nassumptions",
        edge_color=MUTED_BLUE,
        font_size=14,
    )

    channel_specs = (
        (0.57, 0.66, "Expected\ncash flows", TEAL),
        (0.57, 0.48, "Discount\nrates", MUTED_BLUE),
        (0.57, 0.30, "Risk\npremia", AMBER_DARK),
        (0.57, 0.12, "Currency\nconversion", CORAL),
    )
    for x, y, label, color in channel_specs:
        _box(
            axis,
            (x, y),
            0.15,
            0.12,
            label,
            edge_color=color,
            font_size=15,
        )

    _box(
        axis,
        (0.80, 0.37),
        0.16,
        0.25,
        "Value, position,\nand monitoring\nrule",
        edge_color=INK,
        font_size=13,
    )

    _arrow(axis, (0.23, 0.495), (0.29, 0.495), color=TEAL)
    for _, y, _, color in channel_specs:
        _arrow(axis, (0.51, 0.495), (0.57, y + 0.06), color=color)
        _arrow(axis, (0.72, y + 0.06), (0.80, 0.495), color=color)

    axis.text(
        0.055,
        0.055,
        "Decision rule: reject when timing, benchmark, or uncertainty evidence fails.",
        color=INK,
        fontsize=12,
        fontweight="semibold",
        ha="left",
        va="center",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        dpi=160,
        facecolor=BACKGROUND,
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_transmission_map(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
