from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dashboard_fallbacks import build_volatility_dashboard_fallback
from src.dashboards import build_volatility_dashboard


def _provider_interval_filter() -> pd.DataFrame:
    index = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    frame = pd.DataFrame(
        {
            "return": [0.001, -0.002, 0.0005],
            "garch_filtered_volatility": [0.01, 0.011, 0.0105],
            "ewma_volatility": [0.009, 0.0105, 0.0102],
        },
        index=index,
    )
    frame.attrs.update(
        {
            "data_mode": "provider-dated snapshot",
            "sources": "Banxico SIE SF43718 test snapshot",
        }
    )
    return frame


def test_volatility_dashboards_accept_provider_interval_axis_labels() -> None:
    filtered = _provider_interval_filter()
    return_label = "Return per FIX publication interval"
    volatility_label = "Volatility per FIX publication interval"

    static = build_volatility_dashboard_fallback(
        filtered,
        asset="usd_mxn",
        persistence=0.98,
        return_axis_label=return_label,
        volatility_axis_label=volatility_label,
    )
    interactive = build_volatility_dashboard(
        filtered,
        asset="usd_mxn",
        persistence=0.98,
        return_axis_label=return_label,
        volatility_axis_label=volatility_label,
    )

    try:
        assert static.axes[0].get_ylabel() == return_label
        assert static.axes[1].get_ylabel() == volatility_label
        assert interactive.layout.yaxis.title.text == return_label
        assert interactive.layout.yaxis2.title.text == volatility_label
    finally:
        plt.close(static)
