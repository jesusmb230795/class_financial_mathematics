from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import pytest

from src.visual_style import (
    AMBER_DARK,
    BACKGROUND,
    FINMATH_PLOTLY_TEMPLATE,
    EXPORT_FIGURE_DPI,
    INLINE_FIGURE_DPI,
    INK,
    MATPLOTLIB_RC,
    PALETTE,
    SERIES_COLORS,
    add_figure_note,
    apply_plotly_style,
    matplotlib_style,
    series_color,
    series_linestyle,
    series_marker,
    style_axes,
)


def _relative_luminance(hex_color: str) -> float:
    channels = [int(hex_color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(first: str, second: str) -> float:
    luminances = sorted((_relative_luminance(first), _relative_luminance(second)))
    return (luminances[1] + 0.05) / (luminances[0] + 0.05)


def test_python_palette_matches_editorial_visual_manifest() -> None:
    manifest = json.loads(
        Path("img/generated/visual-assets.json").read_text(encoding="utf-8")
    )

    assert {
        key: PALETTE[key]
        for key in (
            "background",
            "ink",
            "teal",
            "amber",
            "coral",
            "muted_blue",
            "soft_grid",
        )
    } == manifest["style"]["palette"]
    assert _contrast_ratio(INK, BACKGROUND) >= 4.5
    assert _contrast_ratio(AMBER_DARK, BACKGROUND) >= 3.0


def test_matplotlib_context_is_scoped_and_pairs_color_with_non_color_cues() -> None:
    original_facecolor = mpl.rcParams["figure.facecolor"]

    with matplotlib_style():
        assert mpl.rcParams["figure.facecolor"] == BACKGROUND
        assert list(mpl.rcParams["axes.prop_cycle"].by_key()["color"]) == list(
            SERIES_COLORS
        )
        figure, axis = plt.subplots()
        try:
            style_axes(axis, grid_axis="both", show_zero_line=True)
            add_figure_note(figure, "Source: deterministic test data.")
            assert not axis.spines["top"].get_visible()
            assert not axis.spines["right"].get_visible()
            assert figure.texts[-1].get_fontsize() == 9
        finally:
            plt.close(figure)

    assert mpl.rcParams["figure.facecolor"] == original_facecolor
    assert MATPLOTLIB_RC["axes.prop_cycle"].by_key()["color"] == list(SERIES_COLORS)
    assert len({series_color(index) for index in range(5)}) == 5
    assert len({str(series_linestyle(index)) for index in range(5)}) == 5
    assert len({series_marker(index) for index in range(5)}) == 5
    assert MATPLOTLIB_RC["figure.dpi"] == INLINE_FIGURE_DPI == 96
    assert MATPLOTLIB_RC["savefig.dpi"] == EXPORT_FIGURE_DPI == 160


def test_plotly_template_is_scoped_and_applied_without_global_default_mutation() -> None:
    original_default = pio.templates.default
    figure = go.Figure(go.Scatter(x=[1, 2], y=[2, 3]))

    result = apply_plotly_style(figure, title="Visual contract", height=480)

    assert result is figure
    assert pio.templates.default == original_default
    assert figure.layout.height == 480
    assert figure.layout.title.text == "Visual contract"
    assert tuple(figure.layout.template.layout.colorway) == SERIES_COLORS
    assert tuple(FINMATH_PLOTLY_TEMPLATE.layout.colorway) == SERIES_COLORS


def test_style_axes_rejects_unknown_grid_axis() -> None:
    figure, axis = plt.subplots()
    try:
        with pytest.raises(ValueError, match="grid_axis"):
            style_axes(axis, grid_axis="diagonal")
    finally:
        plt.close(figure)


def test_module1_uses_shared_tokens_instead_of_ad_hoc_hex_colors() -> None:
    paths = sorted(Path("notebooks/course").glob("1.*.py"))
    paths.extend(
        [
            Path("src/module1_visuals.py"),
            Path("src/dashboards.py"),
            Path("src/dashboard_fallbacks.py"),
        ]
    )

    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert re.search(r"#[0-9A-Fa-f]{6}", source) is None, path
        assert "plotly_white" not in source, path
