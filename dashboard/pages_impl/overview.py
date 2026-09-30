"""Overview page: what ClimateTwin is, model provenance, and skill metrics."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.core import VARIABLE_LABELS, TwinArtifacts


def render(art: TwinArtifacts) -> None:
    """Render the overview / model-performance page."""
    st.header("Overview")
    st.markdown(
        f"""
**ClimateTwin** is a localized AI climate digital twin for **{art.location}**.
A trained LSTM predicts the next hour's six climate variables from the previous
{art.input_window} hours. Rolled forward autoregressively, it tracks a current
state; combined with the what-if engine it supports controlled sensitivity
experiments, each carrying a Monte-Carlo-dropout uncertainty band and a
human-meaningful impact layer.
"""
    )

    m = art.metrics

    # --- Run metadata ---
    st.subheader("Trained model")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", str(m.get("model", "lstm")).upper())
    c2.metric("Parameters", f"{m.get('parameter_count', 0):,}")
    c3.metric("Input window", f"{m.get('input_window', art.input_window)} h")
    c4.metric("Test samples", f"{m.get('test_sample_count', 0):,}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Training epochs", m.get("training_epochs", "—"))
    c6.metric("Best epoch", m.get("best_epoch", "—"))
    best_val = m.get("best_val_loss")
    c7.metric("Best val loss", f"{best_val:.4f}" if isinstance(best_val, (int, float)) else "—")

    # --- Per-variable skill vs persistence ---
    st.subheader("Forecast skill vs. persistence baseline")
    st.caption(
        "Persistence = 'next hour equals this hour'. Positive improvement means "
        "the LSTM beats that naive baseline (RMSE in physical units)."
    )

    comp = m.get("persistence_comparison", {})
    per_var = m.get("per_variable_metrics", {})
    units = m.get("units", art.units)

    rows = []
    for feat in art.feature_order:
        c = comp.get(feat, {})
        pv = per_var.get(feat, {})
        rows.append(
            {
                "Variable": VARIABLE_LABELS.get(feat, feat),
                "Unit": units.get(feat, ""),
                "LSTM RMSE": pv.get("rmse"),
                "LSTM MAE": pv.get("mae"),
                "Persistence RMSE": c.get("persistence_rmse"),
                "Improvement %": c.get("pct_improvement"),
                "Beats baseline": "✅" if c.get("lstm_better") else "—",
            }
        )
    df = pd.DataFrame(rows)

    st.dataframe(
        df.style.format(
            {
                "LSTM RMSE": "{:.3f}",
                "LSTM MAE": "{:.3f}",
                "Persistence RMSE": "{:.3f}",
                "Improvement %": "{:+.1f}",
            },
            na_rep="—",
        ),
        width='stretch',
        hide_index=True,
    )

    # --- Improvement bar chart ---
    improvements = [
        (VARIABLE_LABELS.get(f, f), comp.get(f, {}).get("pct_improvement"))
        for f in art.feature_order
    ]
    improvements = [(name, val) for name, val in improvements if val is not None]
    if improvements:
        names = [n for n, _ in improvements]
        vals = [v for _, v in improvements]
        colors = ["#2ecc71" if v >= 0 else "#c0392b" for v in vals]
        fig = go.Figure(
            go.Bar(x=names, y=vals, marker_color=colors, text=[f"{v:+.1f}%" for v in vals],
                   textposition="outside")
        )
        fig.update_layout(
            title="RMSE improvement over persistence, per variable",
            yaxis_title="Improvement (%)",
            xaxis_title="",
            height=380,
            margin=dict(t=50, b=40),
        )
        fig.add_hline(y=0, line_color="#888", line_width=1)
        st.plotly_chart(fig, width='stretch')

    with st.expander("What the other pages do"):
        st.markdown(
            """
- **Forecast & current state** — pick a moment in the 2024 test period, roll the
  twin forward, and read the forecast alongside feels-like temperature and
  IMD/NOAA alert cards.
- **What-if scenario simulator** — apply a physical-unit perturbation
  (e.g. *+4 °C*), compare against a baseline with paired MC-dropout uncertainty
  bands, and download a structured scenario report.
"""
        )
