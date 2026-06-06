#!/usr/bin/env python3
"""Generate the global exchange market-cap barplot used by the Jupyter Book."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "img" / "generated" / "data" / "global_exchange_market_caps.csv"
DEFAULT_OUTPUT = ROOT / "img" / "generated" / "m1-global-exchanges-ranking.png"

COLORS = {
    "background": "#F7F3EA",
    "ink": "#102A43",
    "teal": "#0F766E",
    "amber": "#D97706",
    "coral": "#C2410C",
    "grid": "#D8DEE9",
}


def load_data(path: Path, top: int) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {
        "rank",
        "exchange",
        "country",
        "market_cap_usd_trn",
        "listed_companies",
        "data_as_of",
        "retrieved_on",
        "source_name",
        "source_url",
    }
    missing = required - set(data.columns)
    if missing:
        raise SystemExit(f"Missing columns in {path}: {', '.join(sorted(missing))}")
    data = data.sort_values("rank").head(top).copy()
    data["label"] = data["exchange"] + " (" + data["country"] + ")"
    return data


def build_chart(data: pd.DataFrame, output: Path) -> None:
    plot_data = data.sort_values("market_cap_usd_trn", ascending=True)
    data_as_of = str(data["data_as_of"].iloc[0])
    retrieved_on = str(data["retrieved_on"].iloc[0])
    source_name = str(data["source_name"].iloc[0])
    source_url = str(data["source_url"].iloc[0])

    fig, ax = plt.subplots(figsize=(16, 12), dpi=100)
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    bar_colors = [COLORS["teal"]] * len(plot_data)
    bar_colors[-1] = COLORS["amber"]
    bars = ax.barh(plot_data["label"], plot_data["market_cap_usd_trn"], color=bar_colors)

    for bar, value, listed in zip(
        bars,
        plot_data["market_cap_usd_trn"],
        plot_data["listed_companies"],
        strict=True,
    ):
        ax.text(
            value + 0.45,
            bar.get_y() + bar.get_height() / 2,
            f"${value:.2f}T  |  {listed:,} listings",
            va="center",
            ha="left",
            fontsize=13,
            color=COLORS["ink"],
        )

    fig.text(
        0.28,
        0.91,
        "Largest Stock Exchanges by Market Capitalization",
        fontsize=26,
        fontweight="bold",
        color=COLORS["ink"],
        ha="left",
        va="top",
    )
    fig.text(
        0.28,
        0.875,
        f"USD trillions. Data as of {data_as_of}; retrieved {retrieved_on}.",
        fontsize=14,
        color=COLORS["ink"],
        ha="left",
        va="top",
    )
    source_note = (
        f"Source: {source_name} ({source_url}). Values reflect the source snapshot "
        "and may differ from official monthly WFE statistics."
    )
    fig.text(
        0.28,
        0.075,
        "\n".join(textwrap.wrap(source_note, width=145)),
        fontsize=10,
        color=COLORS["ink"],
        ha="left",
        va="bottom",
    )

    ax.set_xlabel("Total market capitalization, USD trillions", fontsize=14, color=COLORS["ink"])
    ax.tick_params(axis="x", colors=COLORS["ink"], labelsize=12)
    ax.tick_params(axis="y", colors=COLORS["ink"], labelsize=12)
    ax.xaxis.grid(True, color=COLORS["grid"], linewidth=1)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0, max(plot_data["market_cap_usd_trn"]) * 1.28)

    fig.subplots_adjust(left=0.28, right=0.92, top=0.80, bottom=0.17)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100, facecolor=COLORS["background"])
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    data = load_data(args.data, args.top)
    build_chart(data, args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
