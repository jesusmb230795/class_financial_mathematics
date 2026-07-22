"""Shared visual language for charts in the financial mathematics book.

The palette mirrors the editorial illustrations declared in
``img/generated/visual-assets.json``.  Matplotlib and Plotly helpers keep that
identity consistent while preserving semantic color roles and non-color cues.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from textwrap import fill
from types import MappingProxyType

import matplotlib as mpl
import plotly.graph_objects as go
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure


PALETTE = MappingProxyType(
    {
        "background": "#F7F3EA",
        "ink": "#102A43",
        "teal": "#0F766E",
        "amber": "#D97706",
        "amber_dark": "#B45309",
        "coral": "#C2410C",
        "muted_blue": "#4F6F9F",
        "soft_grid": "#D8DEE9",
        "plot_background": "#FFFFFF",
    }
)

BACKGROUND = PALETTE["background"]
INK = PALETTE["ink"]
TEAL = PALETTE["teal"]
AMBER = PALETTE["amber"]
AMBER_DARK = PALETTE["amber_dark"]
CORAL = PALETTE["coral"]
MUTED_BLUE = PALETTE["muted_blue"]
SOFT_GRID = PALETTE["soft_grid"]
PLOT_BACKGROUND = PALETTE["plot_background"]

# Notebook figures stay crisp at the book's rendered width while respecting
# execution-output budgets. Exported PNG assets retain extra raster density.
INLINE_FIGURE_DPI = 96
EXPORT_FIGURE_DPI = 160

# Categorical series always keep this order across charting backends.
SERIES_COLORS = (TEAL, MUTED_BLUE, AMBER_DARK, CORAL, INK)
SERIES_LINESTYLES = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)))
SERIES_MARKERS = ("o", "s", "^", "D", "P")

# Color roles are semantic.  Positive and negative states must still be paired
# with a sign, label, marker, line style, or position rather than color alone.
SEMANTIC_COLORS = MappingProxyType(
    {
        "primary": TEAL,
        "comparison": MUTED_BLUE,
        "highlight": AMBER_DARK,
        "negative": CORAL,
        "reference": INK,
        "neutral": MUTED_BLUE,
    }
)

MATPLOTLIB_RC = MappingProxyType(
    {
        "axes.axisbelow": True,
        "axes.edgecolor": SOFT_GRID,
        "axes.facecolor": PLOT_BACKGROUND,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.labelcolor": INK,
        "axes.labelsize": 10,
        "axes.prop_cycle": cycler(color=SERIES_COLORS),
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "figure.dpi": INLINE_FIGURE_DPI,
        "figure.facecolor": BACKGROUND,
        "figure.titleweight": "semibold",
        "figure.titlesize": 15,
        "font.family": "sans-serif",
        "font.size": 10,
        "grid.alpha": 0.8,
        "grid.color": SOFT_GRID,
        "grid.linewidth": 0.7,
        "legend.edgecolor": "none",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "savefig.bbox": "tight",
        "savefig.dpi": EXPORT_FIGURE_DPI,
        "savefig.facecolor": BACKGROUND,
        "text.color": INK,
        "xtick.color": INK,
        "xtick.labelsize": 9,
        "ytick.color": INK,
        "ytick.labelsize": 9,
    }
)

FINMATH_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "finmath_diverging",
    (CORAL, BACKGROUND, TEAL),
)


def _plotly_axis_template() -> dict[str, object]:
    return {
        "automargin": True,
        "gridcolor": SOFT_GRID,
        "linecolor": SOFT_GRID,
        "showline": True,
        "ticks": "outside",
        "tickcolor": SOFT_GRID,
        "title": {"font": {"color": INK, "size": 13}},
        "zerolinecolor": SOFT_GRID,
    }


FINMATH_PLOTLY_TEMPLATE = go.layout.Template(
    layout={
        "annotationdefaults": {"font": {"color": INK, "size": 12}},
        "colorway": list(SERIES_COLORS),
        "font": {
            "color": INK,
            "family": "DejaVu Sans, Arial, sans-serif",
            "size": 12,
        },
        "hoverlabel": {
            "bgcolor": PLOT_BACKGROUND,
            "bordercolor": SOFT_GRID,
            "font": {"color": INK},
        },
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "orientation": "h",
            "title": {"font": {"color": INK}},
        },
        "paper_bgcolor": BACKGROUND,
        "plot_bgcolor": PLOT_BACKGROUND,
        "title": {
            "font": {"color": INK, "size": 20},
            "x": 0.01,
            "xanchor": "left",
        },
        "xaxis": _plotly_axis_template(),
        "yaxis": _plotly_axis_template(),
    }
)


def series_color(index: int) -> str:
    """Return the stable categorical color for a zero-based series index."""
    return SERIES_COLORS[index % len(SERIES_COLORS)]


def series_linestyle(index: int) -> object:
    """Return a stable non-color line cue for a zero-based series index."""
    return SERIES_LINESTYLES[index % len(SERIES_LINESTYLES)]


def series_marker(index: int) -> str:
    """Return a stable marker cue for a zero-based series index."""
    return SERIES_MARKERS[index % len(SERIES_MARKERS)]


@contextmanager
def matplotlib_style() -> Iterator[None]:
    """Apply the book style inside an isolated Matplotlib context."""
    with mpl.rc_context(dict(MATPLOTLIB_RC)):
        yield


def set_matplotlib_theme() -> None:
    """Apply the book style to the current notebook kernel."""
    mpl.rcParams.update(dict(MATPLOTLIB_RC))


def _flatten_axes(axes: Axes | Iterable[object]) -> Iterator[Axes]:
    if isinstance(axes, Axes):
        yield axes
        return
    for item in axes:
        if isinstance(item, Axes):
            yield item
        elif isinstance(item, Iterable):
            yield from _flatten_axes(item)


def style_axes(
    axes: Axes | Iterable[object],
    *,
    grid_axis: str | None = "y",
    show_zero_line: bool = False,
) -> None:
    """Apply the shared axis treatment without changing analytical scales."""
    if grid_axis not in {None, "x", "y", "both"}:
        raise ValueError("grid_axis must be None, 'x', 'y', or 'both'")
    for axis in _flatten_axes(axes):
        axis.grid(False)
        if grid_axis is not None:
            axis.grid(
                True,
                axis=grid_axis,
                color=SOFT_GRID,
                linewidth=0.7,
                alpha=0.8,
            )
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(SOFT_GRID)
        axis.spines["bottom"].set_color(SOFT_GRID)
        if show_zero_line:
            axis.axhline(0, color=INK, linewidth=0.8, alpha=0.65, zorder=1)


def add_figure_note(figure: Figure, note: str) -> None:
    """Add a compact source/sample note to the lower-left figure margin."""
    figure.text(
        0.01,
        0.005,
        fill(
            note,
            width=96,
            break_long_words=False,
            break_on_hyphens=False,
        ),
        color=INK,
        fontsize=9,
        ha="left",
        va="bottom",
    )


def apply_plotly_style(
    figure: go.Figure,
    *,
    title: str | None = None,
    height: int | None = None,
) -> go.Figure:
    """Apply the shared Plotly template and return the same figure."""
    updates: dict[str, object] = {
        "template": FINMATH_PLOTLY_TEMPLATE,
        "margin": {"l": 70, "r": 30, "t": 90, "b": 70},
    }
    if title is not None:
        updates["title"] = title
    if height is not None:
        updates["height"] = height
    figure.update_layout(**updates)
    return figure
