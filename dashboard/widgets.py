"""Reusable UI fragments shared across dashboard pages."""

from __future__ import annotations

import numpy as np
import streamlit as st

from dashboard.core import ALERT_COLORS, TwinArtifacts
from src.impact import ImpactReport, summarize_state


def impact_report_for_state(art: TwinArtifacts, state: np.ndarray) -> ImpactReport:
    """Build an :class:`ImpactReport` for one physical-unit forecast state.

    ``state`` is a length-``n_features`` vector ordered like
    ``art.feature_order``. Hourly precipitation is scaled to a 24-hour
    equivalent for the IMD rainfall label, matching ``scripts/scenario_demo.py``.
    """
    idx = {f: i for i, f in enumerate(art.feature_order)}
    return summarize_state(
        t2m_c=float(state[idx["t2m"]]),
        d2m_c=float(state[idx["d2m"]]),
        tp_mm_24h=max(0.0, float(state[idx["tp"]])) * 24.0,
        u10=float(state[idx["u10"]]),
        v10=float(state[idx["v10"]]),
    )


def render_alert_cards(report: ImpactReport) -> None:
    """Render colour-coded alert cards, or a calm banner when there are none."""
    if not report.alerts:
        st.markdown(
            f"<div style='background:{ALERT_COLORS['none']};color:white;"
            "padding:10px 14px;border-radius:8px;font-weight:600;'>"
            "No active alerts — conditions within normal range.</div>",
            unsafe_allow_html=True,
        )
        return

    level_of = {
        report.heat_alert: report.heat_label,
        report.rainfall_alert: report.rainfall_label,
        report.wind_alert: report.wind_label,
    }
    for text in report.alerts:
        # The alert level is embedded as "[LEVEL] ..." at the start of each card.
        level = text.split("]", 1)[0].lstrip("[").lower()
        color = ALERT_COLORS.get(level, "#555")
        _ = level_of  # (kept for clarity; label already in text)
        st.markdown(
            f"<div style='background:{color};color:white;padding:10px 14px;"
            "border-radius:8px;font-weight:600;margin-bottom:6px;'>"
            f"{text}</div>",
            unsafe_allow_html=True,
        )


def render_impact_metrics(report: ImpactReport) -> None:
    """Render the headline impact numbers as Streamlit metrics."""
    c1, c2, c3 = st.columns(3)
    c1.metric("Air temperature", f"{report.air_temp_c:.1f} °C")
    c2.metric("Feels like", f"{report.feels_like_c:.1f} °C",
              delta=f"{report.feels_like_c - report.air_temp_c:+.1f} °C")
    c3.metric("Relative humidity", f"{report.relative_humidity_pct:.0f}%")

    c4, c5 = st.columns(2)
    c4.metric("Heat category", report.heat_label)
    c5.metric("Rainfall category", report.rainfall_label)
