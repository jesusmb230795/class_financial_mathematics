#!/usr/bin/env python3
"""Generate deterministic 1920x1080 RGB tail-risk diagrams for Module 7."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.module7_visuals import (
    build_expected_shortfall_diagram_figure,
    build_skewness_tail_diagram_figure,
    build_var_loss_diagram_figure,
)
from src.visual_style import BACKGROUND, EXPORT_FIGURE_DPI


CANVAS_SIZE = (1920, 1080)
FIGURE_SIZE = (12.0, 6.75)

ASSETS: dict[str, tuple[Callable[[], Figure], Path]] = {
    "risk-var-tail-loss": (
        build_var_loss_diagram_figure,
        ROOT / "img/generated/risk-var-tail-loss.png",
    ),
    "risk-cvar-expected-shortfall": (
        build_expected_shortfall_diagram_figure,
        ROOT / "img/generated/risk-cvar-expected-shortfall.png",
    ),
    "risk-skewness-tail-orientation": (
        build_skewness_tail_diagram_figure,
        ROOT / "img/generated/risk-skewness-tail-orientation.png",
    ),
}


def render_asset(asset_id: str) -> Path:
    """Render one exact opaque RGB PNG and return its repository path."""

    builder, output = ASSETS[asset_id]
    figure = builder()
    temporary = output.with_suffix(".tmp.png")
    try:
        actual_inches = tuple(float(value) for value in figure.get_size_inches())
        if actual_inches != FIGURE_SIZE:
            raise ValueError(f"{asset_id}: expected figure size {FIGURE_SIZE}, got {actual_inches}")
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            temporary,
            dpi=EXPORT_FIGURE_DPI,
            facecolor=BACKGROUND,
            bbox_inches=None,
            pad_inches=0,
        )
    finally:
        plt.close(figure)

    try:
        with Image.open(temporary) as image:
            image.convert("RGB").save(output, format="PNG", optimize=True)
    finally:
        temporary.unlink(missing_ok=True)

    with Image.open(output) as image:
        if image.size != CANVAS_SIZE or image.mode != "RGB":
            raise RuntimeError(
                f"Invalid export for {asset_id}: size={image.size}, mode={image.mode}"
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", choices=sorted(ASSETS))
    args = parser.parse_args()

    selected = [args.asset] if args.asset else list(ASSETS)
    for asset_id in selected:
        print(render_asset(asset_id))


if __name__ == "__main__":
    main()
