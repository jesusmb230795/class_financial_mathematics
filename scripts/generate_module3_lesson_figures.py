#!/usr/bin/env python3
"""Generate deterministic conceptual figures for Module 3 lessons."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.module3_visuals import (
    build_fx_carry_break_even_figure,
    build_scenario_valuation_bridge_figure,
    build_supply_demand_identification_figure,
)
from src.visual_style import BACKGROUND, EXPORT_FIGURE_DPI


ASSETS: dict[str, tuple[Callable[[], Figure], Path]] = {
    "m3-supply-demand-identification": (
        build_supply_demand_identification_figure,
        ROOT / "img/generated/m3-supply-demand-identification.png",
    ),
    "m3-fx-carry-break-even": (
        build_fx_carry_break_even_figure,
        ROOT / "img/generated/m3-fx-carry-break-even.png",
    ),
    "m3-scenario-valuation-bridge": (
        build_scenario_valuation_bridge_figure,
        ROOT / "img/generated/m3-scenario-valuation-bridge.png",
    ),
}


def render_asset(asset_id: str) -> Path:
    """Render one exact 1600x900 PNG and return its path."""

    builder, output = ASSETS[asset_id]
    figure = builder()
    try:
        expected_inches = (10.0, 5.625)
        actual_inches = tuple(float(value) for value in figure.get_size_inches())
        if actual_inches != expected_inches:
            raise ValueError(
                f"{asset_id}: expected figure size {expected_inches}, got {actual_inches}"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output,
            dpi=EXPORT_FIGURE_DPI,
            facecolor=BACKGROUND,
            bbox_inches=None,
            pad_inches=0,
        )
    finally:
        plt.close(figure)
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
