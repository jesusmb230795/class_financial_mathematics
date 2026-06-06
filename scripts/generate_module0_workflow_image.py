#!/usr/bin/env python3
"""Generate the Module 0 reproducible workflow figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "img" / "generated" / "module0-reproducible-workflow.png"

PALETTE = {
    "background": "#F7F3EA",
    "ink": "#102A43",
    "teal": "#0F766E",
    "amber": "#D97706",
    "coral": "#C2410C",
    "muted_blue": "#4F6F9F",
    "soft_grid": "#D8DEE9",
}


def add_box(ax, xy, width, height, title, subtitle, color):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.03",
        linewidth=2.4,
        edgecolor=color,
        facecolor="#FFFDF8",
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height * 0.66,
        title,
        ha="center",
        va="center",
        fontsize=21,
        fontweight="bold",
        color=PALETTE["ink"],
        linespacing=0.88,
    )
    ax.text(
        x + width / 2,
        y + height * 0.24,
        subtitle,
        ha="center",
        va="center",
        fontsize=14,
        color=PALETTE["ink"],
    )


def add_arrow(ax, start, end, color=PALETTE["muted_blue"]):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=28,
        linewidth=2.8,
        color=color,
        shrinkA=10,
        shrinkB=10,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(PALETTE["background"])
    ax.set_facecolor(PALETTE["background"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.91,
        "Module 0 Reproducible Workflow",
        ha="center",
        va="center",
        fontsize=30,
        fontweight="bold",
        color=PALETTE["ink"],
    )

    ax.text(
        0.5,
        0.84,
        "data + code + environment + parameters -> reviewed output",
        ha="center",
        va="center",
        fontsize=18,
        color=PALETTE["ink"],
    )

    width = 0.18
    height = 0.17
    y = 0.52
    boxes = [
        ((0.07, y), "Locked\nenvironment", "pyenv + uv.lock", PALETTE["teal"]),
        ((0.29, y), "Reusable\ncode", "src/ helpers", PALETTE["muted_blue"]),
        ((0.51, y), "Notebook\nexecution", "JupyterLab", PALETTE["amber"]),
        ((0.73, y), "Published\nbook", "cached HTML", PALETTE["coral"]),
    ]

    for xy, title, subtitle, color in boxes:
        add_box(ax, xy, width, height, title, subtitle, color)

    add_arrow(ax, (0.25, y + height / 2), (0.29, y + height / 2))
    add_arrow(ax, (0.47, y + height / 2), (0.51, y + height / 2))
    add_arrow(ax, (0.69, y + height / 2), (0.73, y + height / 2))

    add_box(
        ax,
        (0.18, 0.22),
        0.26,
        0.14,
        "Credentials",
        ".env stays local",
        PALETTE["teal"],
    )
    add_box(
        ax,
        (0.56, 0.22),
        0.26,
        0.14,
        "Validation",
        "git diff + make book",
        PALETTE["amber"],
    )
    add_arrow(ax, (0.31, 0.36), (0.58, 0.52), PALETTE["teal"])
    add_arrow(ax, (0.69, 0.36), (0.78, 0.52), PALETTE["amber"])

    ax.text(
        0.5,
        0.11,
        "A result is publishable when the same inputs reconstruct the same output.",
        ha="center",
        va="center",
        fontsize=16,
        color=PALETTE["ink"],
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp_output = OUTPUT.with_suffix(".tmp.png")
    fig.savefig(temp_output, facecolor=PALETTE["background"], bbox_inches=None)
    plt.close(fig)
    with Image.open(temp_output) as image:
        image.convert("RGB").save(OUTPUT)
    temp_output.unlink()


if __name__ == "__main__":
    main()
