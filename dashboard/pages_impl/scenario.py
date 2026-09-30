"""What-if scenario simulator page.

Apply physical-unit perturbations to the perturbable variables, roll both a
baseline and a perturbed world forward with paired MC-dropout, and compare
them: trajectories with uncertainty bands, the isolated sensitivity (delta),
impact-level changes, and a downloadable text report.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.core import (
    TwinArtifacts,
    available_cities,
    feature_label,
    load_artifacts,
    max_window_start,
    window_at,
    window_end_timestamp,
)
from dashboard.widgets import impact_report_for_state, render_alert_cards
from src.impact import summarize_sectors
from src.scenario import SCENARIO_DISCLAIMER, ScenarioResult, run_scenario
from src.sectors import SWBGT_BIAS_CAVEAT, assess_scenario

# Sensible physical-unit ranges for the what-if sliders, per variable.
PERTURB_RANGES: dict[str, tuple[float, float, float, float]] = {
    # feature: (min, max, default, step)
    "t2m": (-10.0, 10.0, 4.0, 0.5),
    "d2m": (-10.0, 10.0, 0.0, 0.5),
    "sp": (-30.0, 30.0, 0.0, 1.0),
    # Precipitation in mm/h. The upper bound spans cloudburst intensity
    # (IMD treats ~50 mm/h as flash-flood scale), so the urban-flood
    # assessment can be exercised across all IMD warning classes.
    "tp": (0.0, 30.0, 0.0, 1.0),
}


def _build_report_text(
    art: TwinArtifacts,
    result: ScenarioResult,
    now_ts,
    perturbations: dict[str, float],
) -> str:
    """Assemble a downloadable, human-readable scenario report."""
    idx = {f: i for i, f in enumerate(art.feature_order)}
    base_report = impact_report_for_state(art, result.baseline[0])
    scen_report = impact_report_for_state(art, result.perturbed[0])

    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("ClimateTwin — What-if Scenario Report")
    lines.append("=" * 64)
    lines.append(f"Generated : {datetime.now(UTC):%Y-%m-%d %H:%M UTC}")
    lines.append(f"Location  : {art.location}")
    lines.append(f"Start ('now'): {now_ts:%Y-%m-%d %H:%M}")
    lines.append(f"Horizon   : {result.horizon} hours")
    lines.append(f"MC-dropout: {result.mc_samples} passes")
    lines.append("")
    lines.append("Perturbation (physical units):")
    for name, delta in perturbations.items():
        unit = art.units.get(name, "")
        lines.append(f"  {name}: {delta:+.2f} {unit}")
    lines.append("")

    lines.append("-" * 64)
    lines.append("Impact @ hour 1")
    lines.append("-" * 64)
    lines.append(
        f"BASELINE : air {base_report.air_temp_c:.1f} C, "
        f"feels {base_report.feels_like_c:.1f} C, "
        f"RH {base_report.relative_humidity_pct:.0f}%, "
        f"heat={base_report.heat_label}, rain={base_report.rainfall_label}"
    )
    lines.append(
        f"SCENARIO : air {scen_report.air_temp_c:.1f} C, "
        f"feels {scen_report.feels_like_c:.1f} C, "
        f"RH {scen_report.relative_humidity_pct:.0f}%, "
        f"heat={scen_report.heat_label}, rain={scen_report.rainfall_label}"
    )
    if scen_report.alerts:
        lines.append("SCENARIO alerts:")
        for card in scen_report.alerts:
            lines.append(f"  - {card}")
    lines.append("")

    lines.append("-" * 64)
    lines.append("Trajectory sensitivity (perturbed - baseline, physical units)")
    lines.append("-" * 64)
    header = "hour | " + " | ".join(f"{f:>8}" for f in art.feature_order)
    lines.append(header)
    for h in range(result.horizon):
        row = f"{h + 1:>4} | " + " | ".join(
            f"{result.delta[h, idx[f]]:>8.3f}" for f in art.feature_order
        )
        lines.append(row)
    lines.append("")

    lines.append("-" * 64)
    lines.append("DISCLAIMER")
    lines.append("-" * 64)
    lines.append(SCENARIO_DISCLAIMER)
    lines.append("")
    return "\n".join(lines)


def _plot_pair(
    art: TwinArtifacts,
    result: ScenarioResult,
    feat: str,
    perturb_label: str,
) -> go.Figure:
    """Baseline vs scenario trajectory for one variable, with bands."""
    vi = art.feature_order.index(feat)
    hours = np.arange(1, result.horizon + 1)
    has_band = result.baseline_lower is not None
    fig = go.Figure()

    def _band(lower, upper, color):
        if feat == "tp":
            lower = np.clip(lower, 0, None)
        fig.add_trace(go.Scatter(
            x=np.concatenate([hours, hours[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself", fillcolor=color,
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False,
        ))

    if has_band:
        _band(result.baseline_lower[:, vi], result.baseline_upper[:, vi],
              "rgba(31,119,180,0.15)")
        _band(result.perturbed_lower[:, vi], result.perturbed_upper[:, vi],
              "rgba(214,39,40,0.15)")

    fig.add_trace(go.Scatter(
        x=hours, y=result.baseline[:, vi], mode="lines",
        line=dict(color="#1f77b4", width=2), name="Baseline",
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=result.perturbed[:, vi], mode="lines",
        line=dict(color="#d62728", width=2), name=f"Scenario ({perturb_label})",
    ))
    fig.update_layout(
        title=feature_label(feat, art.units),
        xaxis_title="Hours ahead",
        yaxis_title=art.units.get(feat, ""),
        height=360,
        margin=dict(t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ---------------------------------------------------------------------------
# Cross-city comparison
# ---------------------------------------------------------------------------
# Distinct colours for overlaying several cities on one chart.
CITY_COLORS = [
    "#d62728", "#1f77b4", "#2ca02c", "#9467bd",
    "#ff7f0e", "#8c564b", "#e377c2", "#17becf",
]


def _run_for_city(
    slug: str,
    perturbations: dict[str, float],
    horizon: int,
    mc_samples: int,
    sustained: bool,
    start_fraction: float,
) -> tuple[TwinArtifacts, ScenarioResult, object]:
    """Run the same scenario for one city.

    ``start_fraction`` locates the start window as a fraction through the
    city's test split, so every city is compared at the same *relative* point
    in time even if their split lengths ever differ.
    """
    city_art = load_artifacts(slug)
    max_start = max_window_start(city_art)
    start_index = max(0, min(max_start, int(round(start_fraction * max_start))))
    window = window_at(city_art, start_index)
    result = run_scenario(
        model=city_art.model,
        baseline_input=window,
        perturbations=perturbations,
        horizon=horizon,
        feature_names=city_art.feature_order,
        scaler=city_art.scaler,
        mc_samples=mc_samples,
        interval=(5.0, 95.0),
        seed=42,
        sustained=sustained,
    )
    ts = window_end_timestamp(city_art, start_index)
    return city_art, result, ts


def _render_city_comparison(
    slugs: list[str],
    perturbations: dict[str, float],
    horizon: int,
    mc_samples: int,
    sustained: bool,
    start_fraction: float,
    perturb_label: str,
) -> None:
    """Run the scenario across several cities and compare the responses."""
    st.subheader("Cross-city comparison")
    st.caption(
        "The same what-if applied to every selected city, each using its own "
        "trained twin and scaler. Because each city has a different climate "
        "regime, an identical perturbation produces a different response — "
        "that contrast is the point."
    )

    runs: list[tuple[TwinArtifacts, ScenarioResult, object]] = []
    with st.spinner(f"Running the scenario across {len(slugs)} cities..."):
        for slug in slugs:
            runs.append(
                _run_for_city(slug, perturbations, horizon, mc_samples,
                              sustained, start_fraction)
            )

    if not runs:
        return

    feature_order = runs[0][0].feature_order
    units = runs[0][0].units

    # Which variable to compare. Default to a perturbed one if it is a target.
    comparable = [f for f in feature_order]
    default_var = next((f for f in perturbations if f in comparable), comparable[0])
    var = st.selectbox(
        "Variable to compare",
        comparable,
        index=comparable.index(default_var),
        format_func=lambda f: feature_label(f, units),
        key="cmp_var",
    )
    vi = feature_order.index(var)
    hours = np.arange(1, horizon + 1)

    # --- Chart 1: scenario-minus-baseline response per city ---
    fig_delta = go.Figure()
    for i, (city_art, result, _) in enumerate(runs):
        color = CITY_COLORS[i % len(CITY_COLORS)]
        fig_delta.add_trace(go.Scatter(
            x=hours, y=result.delta[:, vi], mode="lines+markers",
            line=dict(color=color, width=2), name=city_art.location,
        ))
    fig_delta.add_hline(y=0, line_color="#888", line_width=1)
    fig_delta.update_layout(
        title=f"Scenario response (scenario − baseline): {feature_label(var, units)}",
        xaxis_title="Hours ahead",
        yaxis_title=f"Δ {units.get(var, '')}",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig_delta, width='stretch')

    # --- Chart 2: absolute scenario trajectories per city ---
    fig_abs = go.Figure()
    for i, (city_art, result, _) in enumerate(runs):
        color = CITY_COLORS[i % len(CITY_COLORS)]
        fig_abs.add_trace(go.Scatter(
            x=hours, y=result.baseline[:, vi], mode="lines",
            line=dict(color=color, width=1.5, dash="dot"),
            name=f"{city_art.location} baseline", legendgroup=city_art.location,
        ))
        fig_abs.add_trace(go.Scatter(
            x=hours, y=result.perturbed[:, vi], mode="lines",
            line=dict(color=color, width=2.5),
            name=f"{city_art.location} scenario", legendgroup=city_art.location,
        ))
    fig_abs.update_layout(
        title=f"Baseline (dotted) vs scenario (solid): {feature_label(var, units)}",
        xaxis_title="Hours ahead",
        yaxis_title=units.get(var, ""),
        height=420,
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig_abs, width='stretch')

    # --- Comparison table: impact + sector shift at hour 1 ---
    st.markdown("**Impact and hazard shift at hour 1**")
    rows = []
    for city_art, result, _ts in runs:
        base = impact_report_for_state(city_art, result.baseline[0])
        scen = impact_report_for_state(city_art, result.perturbed[0])
        idx = {f: i for i, f in enumerate(city_art.feature_order)}
        sec_scen = summarize_sectors(
            t2m_c=float(result.perturbed[0][idx["t2m"]]),
            d2m_c=float(result.perturbed[0][idx["d2m"]]),
        )
        peak = float(np.max(np.abs(result.delta[:, vi])))
        # Operational decision signals per city.
        ds = assess_scenario(
            result.baseline, result.perturbed, city_art.feature_order,
            workload="heavy", applied_warming_c=perturbations.get("t2m"),
        )
        labour_txt = (
            "already unsafe" if ds.labour_baseline_already_unsafe
            else f"{ds.work_fraction_scenario * 100:.0f}% work/hr"
        )
        rows.append({
            "City": city_art.location,
            "Baseline feels-like (°C)": round(base.feels_like_c, 1),
            "Scenario feels-like (°C)": round(scen.feels_like_c, 1),
            "Δ feels-like (°C)": round(scen.feels_like_c - base.feels_like_c, 1),
            "Scenario heat": scen.heat_label,
            "HAP trigger": ds.hap_stage_scenario,
            "Outdoor labour": labour_txt,
            "Flood risk": ds.flood_label.split(" —")[0],
            "Cooling demand": f"{ds.cooling_demand_change_pct:+.0f}%",
            "Scenario health risk": sec_scen.health_label,
            f"Peak |Δ {var}|": round(peak, 2),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, width='stretch', hide_index=True)

    # --- Ranked sensitivity bar chart ---
    st.markdown(f"**Which city responds most strongly? (peak |Δ {var}|)**")
    order = df.sort_values(f"Peak |Δ {var}|", ascending=False)
    fig_rank = go.Figure(go.Bar(
        x=order["City"], y=order[f"Peak |Δ {var}|"],
        marker_color=[CITY_COLORS[i % len(CITY_COLORS)] for i in range(len(order))],
        text=[f"{v:.2f}" for v in order[f"Peak |Δ {var}|"]],
        textposition="outside",
    ))
    fig_rank.update_layout(
        yaxis_title=f"peak |Δ| ({units.get(var, '')})",
        height=320, margin=dict(t=30, b=40),
    )
    st.plotly_chart(fig_rank, width='stretch')

    # --- Downloadable multi-city report ---
    lines = ["=" * 64, "ClimateTwin — Multi-City Scenario Comparison", "=" * 64,
             f"Generated : {datetime.now(UTC):%Y-%m-%d %H:%M UTC}",
             f"Perturbation: {perturb_label}",
             f"Horizon   : {horizon} h | Mode: {'sustained' if sustained else 'seed-only'}",
             f"MC passes : {mc_samples}", ""]
    lines.append(df.to_string(index=False))
    lines += ["", "-" * 64, "DISCLAIMER", "-" * 64, SCENARIO_DISCLAIMER, ""]
    st.download_button(
        "Download comparison report (.txt)",
        data="\n".join(lines),
        file_name=f"climatetwin_multicity_scenario_{'_'.join(perturbations)}.txt",
        mime="text/plain",
    )


ALERT_BG = {"red": "#c0392b", "orange": "#e67e22", "yellow": "#f1c40f", "none": "#2ecc71"}


def _card(title: str, body: str, level: str, action: str | None = None) -> str:
    """Coloured decision card HTML."""
    color = ALERT_BG.get(level, "#555")
    text_color = "#000" if level == "yellow" else "#fff"
    act = (f"<div style='font-size:11px;opacity:0.92;margin-top:6px'>"
           f"<b>Action:</b> {action}</div>") if action else ""
    return (
        f"<div style='background:{color};color:{text_color};padding:12px 14px;"
        f"border-radius:10px;margin-bottom:8px'>"
        f"<div style='font-size:11px;text-transform:uppercase;opacity:0.85;"
        f"letter-spacing:0.5px'>{title}</div>"
        f"<div style='font-size:15px;font-weight:700;margin-top:2px'>{body}</div>"
        f"{act}</div>"
    )


def _render_decision_support(
    art: TwinArtifacts,
    result: ScenarioResult,
    perturbations: dict[str, float],
) -> None:
    """Render the sectoral decision-support panel — who acts, and on what."""
    st.subheader("🎯 Decision support — who acts on this?")
    st.caption(
        "The scenario mapped onto **published operational thresholds** used by "
        "real institutions: Ahmedabad Heat Action Plan triggers, ACGIH/ISO 7243 "
        "work-rest limits, IMD rainfall warning classes, and Zhao et al. (2017) "
        "crop-yield sensitivities. Each card names the decision it informs."
    )

    workload = st.radio(
        "Outdoor workload class (for heat-stress limits)",
        ["heavy", "moderate", "light"],
        horizontal=True,
        help="Construction and agriculture are typically 'heavy'.",
        key="ds_workload",
    )

    applied_warming = perturbations.get("t2m")
    assessment = assess_scenario(
        result.baseline, result.perturbed, art.feature_order,
        workload=workload, applied_warming_c=applied_warming,
    )

    c1, c2 = st.columns(2)

    # --- Disaster management / municipal ---
    with c1:
        st.markdown("**🏛️ Municipal / disaster management**")
        st.markdown(
            _card(
                "Heat Action Plan trigger (Ahmedabad HAP)",
                f"{assessment.hap_stage_baseline} → {assessment.hap_stage_scenario}",
                assessment.hap_alert,
                assessment.hap_action,
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            _card(
                "Urban flood risk (IMD classes)",
                assessment.flood_label,
                assessment.flood_alert,
                assessment.flood_action,
            ),
            unsafe_allow_html=True,
        )
        st.caption(
            f"Rainfall (24 h-equivalent): {assessment.rain_24h_baseline_mm:.1f} → "
            f"{assessment.rain_24h_scenario_mm:.1f} mm"
        )

    # --- Labour + energy ---
    with c2:
        st.markdown("**👷 Labour & ⚡ energy**")
        if assessment.labour_baseline_already_unsafe:
            labour_body = (
                f"Baseline ALREADY exceeds safe limits "
                f"(sWBGT {assessment.wbgt_baseline_c:.1f} °C)"
            )
            labour_action = (
                "Outdoor heavy work is already outside ACGIH limits before the "
                "perturbation — rescheduling to cooler hours is indicated "
                "regardless of the scenario."
            )
        else:
            labour_body = (
                f"{assessment.work_fraction_baseline * 100:.0f}% → "
                f"{assessment.work_fraction_scenario * 100:.0f}% work per hour"
            )
            labour_action = (
                f"{assessment.labour_label}. Capacity change: "
                f"{-assessment.labour_capacity_lost_pct:+.0f} percentage points."
            )
        st.markdown(
            _card("Work/rest limit (ACGIH / ISO 7243)", labour_body,
                  assessment.labour_alert, labour_action),
            unsafe_allow_html=True,
        )
        st.caption(
            f"sWBGT: {assessment.wbgt_baseline_c:.1f} → "
            f"{assessment.wbgt_scenario_c:.1f} °C"
        )

        energy_level = (
            "red" if assessment.cooling_demand_change_pct >= 50
            else "orange" if assessment.cooling_demand_change_pct >= 20
            else "yellow" if assessment.cooling_demand_change_pct > 0 else "none"
        )
        st.markdown(
            _card(
                "Cooling electricity demand (18 °C base)",
                f"{assessment.cooling_demand_change_pct:+.0f}% cooling degree-hours",
                energy_level,
                "Informs grid load planning and peak-demand readiness.",
            ),
            unsafe_allow_html=True,
        )

    # --- Agriculture ---
    st.markdown("**🌾 Agriculture**")
    if applied_warming is None:
        st.info(
            "Crop-yield sensitivity needs a temperature perturbation. "
            "Set Δ t2m in the sidebar to see agricultural exposure."
        )
    else:
        cols = st.columns(len(assessment.crop_yield_delta_pct))
        for col, (crop, pct) in zip(
            cols, assessment.crop_yield_delta_pct.items(), strict=False
        ):
            col.metric(f"{crop.title()} yield", f"{pct:+.1f}%")
        st.caption(
            f"Indicative sensitivity at a **sustained {assessment.applied_warming_c:+.1f} °C**, "
            "using Zhao et al. (2017, PNAS) per-degree global yield "
            "sensitivities (wheat −6.0, rice −3.2, maize −7.4 %/°C)."
        )
        if assessment.crop_yield_extrapolated:
            st.warning(
                "⚠️ This perturbation exceeds the ~4 °C range the published "
                "crop coefficients were derived over. The linear extrapolation "
                "is not supportable — treat as illustrative only."
            )

    with st.expander("Method, sources, and limitations"):
        st.markdown(
            f"""
**Sources** — every threshold is external and citable; none are invented:

- **Heat Action Plan**: Ahmedabad HAP (South Asia's first, 2013) — yellow
  41.1–43.0 °C, orange 43.1–44.9 °C, red ≥ 45 °C.
  [Knowlton et al. 2014](https://www.mdpi.com/1660-4601/11/4/3473)
- **Work/rest limits**: ACGIH Threshold Limit Values / ISO 7243, by WBGT and
  workload, for acclimatized workers.
- **Crop yield**: [Zhao et al. 2017, PNAS](https://www.pnas.org/doi/10.1073/pnas.1701762114)
- **Rainfall classes**: [India Meteorological Department](https://mausam.imd.gov.in/)
- **Cooling degree hours**: conventional 18 °C base.

**Known limitation we do not hide:** {SWBGT_BIAS_CAVEAT}

**Crop caveat:** the published coefficients describe *sustained seasonal*
warming of global mean temperature without adaptation or CO₂ fertilisation.
Applying them to a local scenario assumes the sensitivity transfers, so the
figure indicates *exposure*, not a local yield forecast.

{assessment.disclaimer}
"""
        )


def render(art: TwinArtifacts) -> None:
    """Render the what-if scenario simulator page."""
    st.header("What-if scenario simulator")
    st.caption(
        "Change one or more input variables in physical units and roll both a "
        "baseline and the perturbed world forward. Paired MC-dropout isolates "
        "the perturbation's effect from mask noise; the widening band shows when "
        "the scenario stops being trustworthy."
    )

    max_start = max_window_start(art)

    with st.sidebar:
        st.subheader("Scenario controls")
        start_index = st.slider(
            "Start hour (index into 2024 test split)",
            0, max_start, min(4000, max_start), 1,
        )
        horizon = st.slider("Horizon (hours)", 1, 48, 12)
        mc_samples = st.select_slider(
            "MC-dropout passes", options=[1, 10, 20, 40, 60], value=40,
        )
        sustained = st.checkbox(
            "Sustain perturbation", value=False,
            help="If on, the perturbation is re-injected every step (a "
                 "persistently changed world). If off, it only seeds the "
                 "initial window and the twin relaxes back toward climatology.",
        )

        # Perturbation controls come first: they are the primary interaction.
        st.markdown("### 🎛️ What-if perturbations")
        st.caption("Set a change in real physical units.")
        perturbations: dict[str, float] = {}
        for feat in art.perturbable:
            lo, hi, default, step = PERTURB_RANGES.get(feat, (-5.0, 5.0, 0.0, 0.5))
            unit = art.units.get(feat, "")
            val = st.slider(
                f"Δ {feat} ({unit})", lo, hi, default, step,
                help=feature_label(feat, art.units),
            )
            if val != 0.0:
                perturbations[feat] = float(val)
        if "tp" not in art.perturbable:
            st.caption(
                "ℹ️ No rainfall control: add `tp` to `perturbable_features` in "
                "`configs/data.yaml` to enable flood what-ifs."
            )

        st.divider()
        st.markdown("**Compare across cities**")
        all_cities = available_cities()
        city_labels = {c.name: c.slug for c in all_cities}
        compare_names = st.multiselect(
            "Cities",
            list(city_labels.keys()),
            default=[c.name for c in all_cities],
            help="Run this same what-if on several cities and compare how each "
                 "one responds. Leave empty to see only the detailed "
                 "single-city view.",
        )
        compare_slugs = [city_labels[n] for n in compare_names]

    if not perturbations:
        st.info(
            "Set at least one non-zero perturbation in the sidebar to run a "
            "scenario. Try Δ t2m = +4 °C for a heatwave what-if."
        )
        return

    perturb_label = ", ".join(
        f"{k} {v:+.1f}{art.units.get(k, '')}" for k, v in perturbations.items()
    )
    now_ts = window_end_timestamp(art, start_index)
    st.markdown(
        f"**Scenario:** {perturb_label}  |  **Start:** {now_ts:%Y-%m-%d %H:%M}  |  "
        f"**Horizon:** {horizon} h  |  **Mode:** "
        f"{'sustained' if sustained else 'seed-only'}"
    )

    # Run the selected city's scenario up front so the decision-support tab
    # (the most important output) can lead rather than being buried.
    window = window_at(art, start_index)
    with st.spinner("Running paired baseline vs scenario rollout..."):
        result = run_scenario(
            model=art.model,
            baseline_input=window,
            perturbations=perturbations,
            horizon=horizon,
            feature_names=art.feature_order,
            scaler=art.scaler,
            mc_samples=mc_samples,
            interval=(5.0, 95.0),
            seed=42,
            sustained=sustained,
        )

    # Tabs keep every output one click away instead of stacking a long page,
    # so nothing important sits below the fold during a live demo.
    tab_decide, tab_impact, tab_traj, tab_cities, tab_export = st.tabs([
        "🎯 Decision support",
        "🌡️ Impact",
        "📊 Trajectories",
        "🌍 Cross-city",
        "📄 Export",
    ])

    # --- Decision support: the "so what, and who acts?" layer ---
    with tab_decide:
        _render_decision_support(art, result, perturbations)

    # --- Impact comparison at hour 1 ---
    with tab_impact:
        st.subheader(f"Impact shift @ hour 1 — {art.location}")
        base_report = impact_report_for_state(art, result.baseline[0])
        scen_report = impact_report_for_state(art, result.perturbed[0])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Baseline**")
            st.metric("Feels like", f"{base_report.feels_like_c:.1f} °C")
            st.write(f"Heat: {base_report.heat_label}")
            st.write(f"Rain: {base_report.rainfall_label}")
            render_alert_cards(base_report)
        with c2:
            st.markdown("**Scenario**")
            st.metric(
                "Feels like", f"{scen_report.feels_like_c:.1f} °C",
                delta=f"{scen_report.feels_like_c - base_report.feels_like_c:+.1f} °C",
            )
            st.write(f"Heat: {scen_report.heat_label}")
            st.write(f"Rain: {scen_report.rainfall_label}")
            render_alert_cards(scen_report)

    # --- Cross-city comparison (runs the same what-if everywhere) ---
    with tab_cities:
        if len(compare_slugs) > 1:
            max_start = max_window_start(art)
            start_fraction = (start_index / max_start) if max_start > 0 else 0.0
            _render_city_comparison(
                compare_slugs, perturbations, horizon, mc_samples, sustained,
                start_fraction, perturb_label,
            )
        else:
            st.info(
                "Select two or more cities under **Compare across cities** in the "
                "sidebar to compare how the same what-if plays out in each."
            )

    # --- Trajectory plots ---
    with tab_traj:
        st.subheader("Baseline vs scenario trajectories")
        default_vars = list(perturbations.keys())
        if "tp" in art.feature_order and "tp" not in default_vars:
            default_vars = default_vars + ["tp"]
        chosen = st.multiselect(
            "Variables to plot",
            options=art.feature_order,
            default=default_vars,
            format_func=lambda f: feature_label(f, art.units),
        )
        for feat in chosen:
            st.plotly_chart(_plot_pair(art, result, feat, perturb_label),
                            width='stretch')

        _render_sensitivity(art, result, chosen)

    with tab_export:
        _render_export(art, result, now_ts, perturbations)


def _render_sensitivity(art: TwinArtifacts, result: ScenarioResult,
                        chosen: list[str]) -> None:
    """Isolated paired-MC sensitivity charts for the chosen variables."""
    if result.delta_lower is not None and chosen:
        st.subheader("Isolated sensitivity (scenario − baseline)")
        st.caption(
            "Paired MC-dropout: the band here reflects genuine sensitivity "
            "uncertainty, with per-sample mask noise cancelled out."
        )
        hours = np.arange(1, result.horizon + 1)
        for feat in chosen:
            vi = art.feature_order.index(feat)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=np.concatenate([hours, hours[::-1]]),
                y=np.concatenate([result.delta_upper[:, vi],
                                  result.delta_lower[::-1, vi]]),
                fill="toself", fillcolor="rgba(142,68,173,0.18)",
                line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=hours, y=result.delta[:, vi], mode="lines+markers",
                line=dict(color="#8e44ad", width=2), name="Δ",
            ))
            fig.add_hline(y=0, line_color="#888", line_width=1)
            fig.update_layout(
                title=f"Δ {feature_label(feat, art.units)}",
                xaxis_title="Hours ahead",
                yaxis_title=f"Δ {art.units.get(feat, '')}",
                height=300, margin=dict(t=50, b=40),
            )
            st.plotly_chart(fig, width='stretch')


def _render_export(art: TwinArtifacts, result: ScenarioResult, now_ts,
                   perturbations: dict[str, float]) -> None:
    """Downloadable single-city scenario report."""
    st.subheader("Export")
    report_text = _build_report_text(art, result, now_ts, perturbations)
    fname = (
        f"climatetwin_scenario_{art.location.lower()}_"
        f"{now_ts:%Y%m%d%H}_{'_'.join(perturbations)}.txt"
    )
    st.download_button(
        "Download scenario report (.txt)",
        data=report_text,
        file_name=fname,
        mime="text/plain",
    )
    with st.expander("Preview report"):
        st.code(report_text, language="text")

    st.caption(f"_{SCENARIO_DISCLAIMER}_")
