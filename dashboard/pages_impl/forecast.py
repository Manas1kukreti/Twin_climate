"""Forecast & current-state page.

Pick a moment in the 2024 test period, roll the trained twin forward
autoregressively (via ``src.scenario.run_scenario`` with a no-op perturbation),
and read the forecast trajectory alongside human-meaningful impact indicators.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.core import (
    TwinArtifacts,
    feature_label,
    max_window_start,
    window_at,
    window_end_timestamp,
)
from dashboard.widgets import (
    impact_report_for_state,
    render_alert_cards,
    render_impact_metrics,
)
from src.scenario import SCENARIO_DISCLAIMER, run_scenario


def _forecast(art: TwinArtifacts, start_index: int, horizon: int,
              mc_samples: int) -> object:
    """Run a baseline-only forecast (zero perturbation) and return the result."""
    window = window_at(art, start_index)
    # A zero-delta perturbation on the first perturbable feature makes the
    # "perturbed" trajectory identical to the baseline; we only read baseline*.
    zero = {art.perturbable[0]: 0.0}
    return run_scenario(
        model=art.model,
        baseline_input=window,
        perturbations=zero,
        horizon=horizon,
        feature_names=art.feature_order,
        scaler=art.scaler,
        mc_samples=mc_samples,
        interval=(5.0, 95.0),
        seed=42,
        sustained=False,
    )


def render(art: TwinArtifacts) -> None:
    """Render the forecast page."""
    st.header("Forecast & current state")
    st.caption(
        f"Autoregressive rollout of the trained {art.location} LSTM over the "
        "held-out 2024 test period. The starting state is a real 24-hour window; "
        "the twin predicts forward one hour at a time."
    )

    max_start = max_window_start(art)

    with st.sidebar:
        st.subheader("Forecast controls")
        start_index = st.slider(
            "Start hour (index into 2024 test split)",
            min_value=0,
            max_value=max_start,
            value=min(4000, max_start),
            step=1,
            help="Selects which real 24-hour window seeds the forecast.",
        )
        horizon = st.slider("Forecast horizon (hours)", 1, 48, 12,
                            help="How many hours ahead to roll the twin forward.")
        mc_samples = st.select_slider(
            "Uncertainty (MC-dropout passes)",
            options=[1, 10, 20, 40, 60],
            value=40,
            help="1 = single deterministic line; higher = predictive band.",
        )

    now_ts = window_end_timestamp(art, start_index)
    st.markdown(f"**Current state ('now'):** {now_ts:%Y-%m-%d %H:%M} (local ERA5-Land time)")

    with st.spinner("Rolling the twin forward..."):
        result = _forecast(art, start_index, horizon, mc_samples)

    # --- Impact at hour 1 (the immediate next-hour forecast) ---
    st.subheader("Next-hour impact")
    report = impact_report_for_state(art, result.baseline[0])
    render_impact_metrics(report)
    st.markdown("**Alerts**")
    render_alert_cards(report)

    # --- Trajectory plots ---
    st.subheader("Forecast trajectories")
    default_vars = [v for v in ("t2m", "tp") if v in art.feature_order]
    chosen = st.multiselect(
        "Variables to plot",
        options=art.feature_order,
        default=default_vars or art.feature_order[:1],
        format_func=lambda f: feature_label(f, art.units),
    )

    hours = np.arange(1, result.horizon + 1)
    has_band = result.baseline_lower is not None

    for feat in chosen:
        vi = art.feature_order.index(feat)
        fig = go.Figure()
        if has_band:
            lower = result.baseline_lower[:, vi]
            upper = result.baseline_upper[:, vi]
            if feat == "tp":
                lower = np.clip(lower, 0, None)
            fig.add_trace(go.Scatter(
                x=np.concatenate([hours, hours[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself", fillcolor="rgba(31,119,180,0.18)",
                line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
                name="90% interval", showlegend=True,
            ))
        fig.add_trace(go.Scatter(
            x=hours, y=result.baseline[:, vi], mode="lines+markers",
            line=dict(color="#1f77b4", width=2), name="Forecast",
        ))
        fig.update_layout(
            title=feature_label(feat, art.units),
            xaxis_title="Hours ahead",
            yaxis_title=art.units.get(feat, ""),
            height=340,
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig, width='stretch')

    if not has_band:
        st.info("Set MC-dropout passes above 1 to see predictive uncertainty bands.")

    st.caption(f"_{SCENARIO_DISCLAIMER}_")
