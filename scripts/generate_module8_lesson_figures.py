#!/usr/bin/env python3
"""Generate deterministic quantitative figures for Module 8 lessons."""

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

from src.module8_visuals import build_liquidity_budget_figure, build_lp_waterfall_figure
from src.visual_style import BACKGROUND, EXPORT_FIGURE_DPI


ASSETS: dict[str, tuple[Callable[[], Figure], Path]] = {
    "m8-lp-waterfall-economics": (
        build_lp_waterfall_figure,
        ROOT / "img/generated/m8-lp-waterfall-economics.png",
    ),
    "m8-liquidity-budget-coverage": (
        build_liquidity_budget_figure,
        ROOT / "img/generated/m8-liquidity-budget-coverage.png",
    ),
}


def render_asset(asset_id: str) -> Path:
    """Render one exact 1600x900 opaque RGB PNG and return its path."""

    builder, output = ASSETS[asset_id]
    figure = builder()
    temporary = output.with_suffix(".tmp.png")
    try:
        expected_inches = (10.0, 5.625)
        actual_inches = tuple(float(value) for value in figure.get_size_inches())
        if actual_inches != expected_inches:
            raise ValueError(
                f"{asset_id}: expected figure size {expected_inches}, got {actual_inches}"
            )
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
        if image.size != (1600, 900) or image.mode != "RGB":
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
