from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure
from matplotlib.text import Text
from PIL import Image

from scripts.generate_module_concept_maps import CANVAS_SIZE, RENDERERS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE4_RENDERERS = {
    "m4-three-statement-evidence-model-map": (
        "Reported",
        "Reclassified",
        "Normalized",
        "Forecast",
        "Balance schedules → Cash",
        "BS diff. = 0",
        "Cash diff. = 0",
    ),
    "m4-roe-roic-driver-map": (
        "9.00%",
        "1.307×",
        "1.811×",
        "21.30%",
        "11.25%",
        "1.688×",
        "18.99%",
    ),
    "m4-normalized-ebit-bridge": (
        "USD m",
        "120",
        "+18",
        "-10",
        "-6",
        "122",
        "Stock compensation",
    ),
    "m4-entity-perimeter-map": (
        "Financial asset",
        "IFRS 9",
        "Significant influence",
        "Equity method · IAS 28",
        "Joint arrangement",
        "Rights & obligations",
        "IFRS 11",
        "Control",
        "Consolidate · IFRS 10",
        "Structured entity",
        "Guarantees · liquidity support",
        "Disclose & stress",
        "IFRS 12 · residual exposure",
        "NCI attribution",
    ),
    "m4-forecast-cash-reconciliation": (
        "USD m",
        "60.00",
        "+146.05",
        "-70.00",
        "-20.00",
        "-30.00",
        "86.05",
    ),
}


def _visible_text(figure: Figure) -> list[Text]:
    return [
        artist
        for artist in figure.findobj(match=Text)
        if artist.get_visible() and artist.get_text().strip()
    ]


@pytest.mark.parametrize(
    ("asset_id", "required_fragments"),
    MODULE4_RENDERERS.items(),
)
def test_module4_renderers_preserve_content_geometry_and_legibility(
    asset_id: str,
    required_fragments: tuple[str, ...],
) -> None:
    assert asset_id in RENDERERS
    figure = RENDERERS[asset_id]()

    try:
        assert isinstance(figure, Figure)
        assert tuple(figure.get_size_inches()) == pytest.approx((16, 9))
        texts = _visible_text(figure)
        combined = " ".join(text.get_text().replace("\n", " ") for text in texts)
        assert all(fragment in combined for fragment in required_fragments)
        assert min(text.get_fontsize() for text in texts) >= 18
    finally:
        plt.close(figure)


@pytest.mark.parametrize("asset_id", MODULE4_RENDERERS)
def test_module4_exported_pngs_are_exact_size_and_rgb(asset_id: str) -> None:
    path = PROJECT_ROOT / "img" / "generated" / f"{asset_id}.png"

    with Image.open(path) as image:
        assert image.size == CANVAS_SIZE
        assert image.mode == "RGB"
