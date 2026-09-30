"""Multi-city map page: anomaly + uncertainty overlays on a Folium map.

Reads the precomputed SQLite map DB (``data/climatetwin.db``, built by
``scripts/build_map_db.py``) and renders each city as a circle marker:

- **Colour = temperature anomaly** (z-score vs the city's own training-period
  climatology). Blue = colder than normal, red = hotter than normal, through a
  diverging scale — the "how unusual is right now" signal.
- **Radius = forecast uncertainty** (mean MC-dropout band width across
  variables at the forecast horizon). Larger circle = the twin is less certain
  about this city's short-term outlook.

Clicking a city shows a popup with its current state, per-variable anomalies,
sector indicators, and the forecast uncertainty. If the DB is missing, the page
explains how to build it rather than erroring.
"""

from __future__ import annotations

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from dashboard.core import PROJECT_ROOT, TwinArtifacts
from src.impact import summarize_sectors, summarize_state

DB_PATH = PROJECT_ROOT / "data" / "climatetwin.db"

# India-centred default view.
INDIA_CENTER = (21.5, 80.0)


def _anomaly_color(z: float | None) -> str:
    """Diverging blue-white-red colour for a temperature anomaly z-score."""
    if z is None:
        return "#888888"
    # Clamp to +/- 3 sigma for the colour ramp.
    z = max(-3.0, min(3.0, z))
    if z >= 0:
        # white -> red
        t = z / 3.0
        r = 255
        g = int(255 * (1 - t))
        b = int(255 * (1 - t))
    else:
        # white -> blue
        t = -z / 3.0
        r = int(255 * (1 - t))
        g = int(255 * (1 - t))
        b = 255
    return f"#{r:02x}{g:02x}{b:02x}"


def _radius_from_uncertainty(band: float | None) -> float:
    """Map a forecast band width (physical units) to a marker radius in px."""
    if band is None:
        return 8.0
    # Band widths seen in practice run ~1-5 units; scale into 8-26 px.
    return float(min(26.0, 8.0 + band * 3.5))


def _popup_html(art: TwinArtifacts, detail: dict) -> str:
    """Build the popup HTML for one city from its DB detail rows."""
    city = detail["city"]
    cur = {r["variable"]: r for r in detail["current_state"]}

    # Derive impact + sector indicators from the stored current state.
    def val(v: str) -> float:
        return float(cur[v]["value"]) if v in cur else 0.0

    impact = summarize_state(
        t2m_c=val("t2m"), d2m_c=val("d2m"),
        tp_mm_24h=max(0.0, val("tp")) * 24.0, u10=val("u10"), v10=val("v10"),
    )
    sectors = summarize_sectors(t2m_c=val("t2m"), d2m_c=val("d2m"))

    syn = " <span style='color:#b30000'>(SYNTHETIC)</span>" if city.get("synthetic") else ""
    rows = "".join(
        f"<tr><td>{v}</td><td style='text-align:right'>{cur[v]['value']:.1f} "
        f"{cur[v]['unit']}</td><td style='text-align:right'>z={cur[v]['anomaly_z']:+.2f}</td></tr>"
        for v in art.feature_order if v in cur
    )
    return f"""
    <div style='font-family:sans-serif;min-width:250px'>
      <h4 style='margin:2px 0'>{city['name']}{syn}</h4>
      <div style='color:#666;font-size:11px'>{city.get('climate_zone','')}</div>
      <div style='font-size:11px;color:#888'>as of {cur.get('t2m',{}).get('timestamp','?')}</div>
      <hr style='margin:4px 0'>
      <b>Feels like {impact.feels_like_c:.1f} °C</b> · RH {impact.relative_humidity_pct:.0f}%<br>
      Heat: {impact.heat_label} · Rain: {impact.rainfall_label}<br>
      <div style='margin-top:4px'>
        🌾 {sectors.crop_label}<br>
        🩺 {sectors.health_label}<br>
        ⚡ {sectors.cooling_label}
      </div>
      <hr style='margin:4px 0'>
      <table style='font-size:11px;border-collapse:collapse;width:100%'>
        <tr><th style='text-align:left'>var</th><th>value</th><th>anomaly</th></tr>
        {rows}
      </table>
    </div>
    """


def render(art: TwinArtifacts) -> None:
    """Render the multi-city Folium map page."""
    st.header("Multi-city climate map")

    if not DB_PATH.exists():
        st.warning(
            "The map database has not been built yet. Run:\n\n"
            "```bash\npython scripts/build_map_db.py\n```\n\n"
            "This precomputes each city's current state, anomaly, and forecast "
            "uncertainty from the trained artifacts."
        )
        return

    from src import db as dbmod

    meta = dbmod.get_meta(DB_PATH)
    live_rows = dbmod.load_live_rows(DB_PATH)
    reanalysis_rows = dbmod.load_map_rows(DB_PATH)

    # --- Data-source selector: live observations vs reanalysis snapshot ---
    has_live = len(live_rows) > 0
    if has_live:
        mode = st.radio(
            "Data source",
            ["🔴 Live observations", "📚 ERA5-Land reanalysis snapshot"],
            horizontal=True,
            help=(
                "Live = current conditions from Open-Meteo. Reanalysis = the "
                "end of the ERA5-Land test split, which the models were "
                "evaluated on. ERA5-Land has days-to-months latency so it "
                "cannot show 'now'."
            ),
        )
        live_mode = mode.startswith("🔴")
    else:
        live_mode = False
        st.warning(
            "No live observations stored. Run `python scripts/build_map_db.py` "
            "to fetch current conditions from Open-Meteo."
        )

    rows = live_rows if live_mode else reanalysis_rows

    if live_mode:
        observed = rows[0]["observed_at"] if rows else "?"
        st.success(
            f"**Live conditions** observed at **{observed} UTC** · "
            f"source: {rows[0]['source'] if rows else 'open-meteo'}"
        )
    else:
        ts = next((r["timestamp"] for r in rows if r.get("timestamp")), "?")
        st.info(
            f"**Reanalysis snapshot** — ERA5-Land at **{ts}** (end of the 2024 "
            "test split). This is historical by design: ERA5-Land is a "
            "reanalysis product and cannot provide real-time data."
        )

    st.caption(
        "Circle **colour** = temperature anomaly vs each city's own climatology, "
        "matched by **calendar month and hour of day** (blue = colder than "
        f"normal, red = hotter). Climatology basis: {meta.get('climatology_basis','per month/hour')}. "
        + ("Circle **size** = forecast uncertainty (MC-dropout band width)."
           if not live_mode else "")
    )

    if any(r.get("synthetic") for r in rows):
        st.info("Cities marked SYNTHETIC use generated data, not real ERA5-Land.")

    # Use OpenStreetMap's standard tiles: no API key/token required. (Some
    # named providers in recent folium route through endpoints that demand an
    # access token, which surfaces as an "API key required" tile overlay.)
    fmap = folium.Map(
        location=INDIA_CENTER,
        zoom_start=5,
        tiles="OpenStreetMap",
    )

    for r in rows:
        color = _anomaly_color(r["t2m_anomaly_z"])
        z = r["t2m_anomaly_z"]

        if live_mode:
            radius = 12.0  # uniform: no forecast band on a live observation
            v = r["variables"]
            rows_html = "".join(
                f"<tr><td>{name}</td><td style='text-align:right'>"
                f"{d['value']:.1f} {d['unit']}</td>"
                f"<td style='text-align:right'>"
                f"{('z=' + format(d['anomaly_z'], '+.2f')) if d['anomaly_z'] is not None else '—'}"
                f"</td></tr>"
                for name, d in v.items()
            )
            popup_html = (
                f"<div style='font-family:sans-serif;min-width:240px'>"
                f"<h4 style='margin:2px 0'>{r['name']} "
                f"<span style='color:#c0392b;font-size:11px'>● LIVE</span></h4>"
                f"<div style='color:#666;font-size:11px'>{r['climate_zone']}</div>"
                f"<div style='font-size:11px;color:#888'>observed {r['observed_at']} UTC</div>"
                f"<hr style='margin:4px 0'>"
                f"<table style='font-size:11px;border-collapse:collapse;width:100%'>"
                f"<tr><th style='text-align:left'>var</th><th>value</th>"
                f"<th>anomaly</th></tr>{rows_html}</table>"
                f"<div style='font-size:10px;color:#888;margin-top:4px'>"
                f"Source: Open-Meteo</div></div>"
            )
            tooltip = (
                f"{r['name']} (LIVE): {r['t2m_value']:.1f} {r['t2m_unit']}"
                + (f" · anomaly z={z:+.2f}" if z is not None else "")
            )
        else:
            detail = dbmod.load_city_detail(r["slug"], DB_PATH)
            radius = _radius_from_uncertainty(r["forecast_uncertainty"])
            popup_html = _popup_html(art, detail)
            tooltip = (
                f"{r['name']}: {r['t2m_value']:.1f} {r['t2m_unit']} "
                f"(anomaly z={z:+.2f}), forecast band ±{r['forecast_uncertainty']:.1f}"
                if z is not None else r["name"]
            )

        folium.CircleMarker(
            location=(r["latitude"], r["longitude"]),
            radius=radius,
            color="#333333",
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=tooltip,
        ).add_to(fmap)

    st_folium(fmap, width=None, height=520, returned_objects=[])

    # --- Summary table under the map ---
    st.subheader("City snapshot" + (" — live" if live_mode else " — reanalysis"))

    def _anom_label(z: float | None) -> str:
        if z is None:
            return "—"
        if z >= 2:
            return f"{z:+.2f} (much hotter than normal)"
        if z >= 1:
            return f"{z:+.2f} (hotter than normal)"
        if z <= -2:
            return f"{z:+.2f} (much colder than normal)"
        if z <= -1:
            return f"{z:+.2f} (colder than normal)"
        return f"{z:+.2f} (near normal)"

    records = []
    for r in rows:
        rec = {
            "City": r["name"],
            "Climate zone": r["climate_zone"],
            "Temp (°C)": round(r["t2m_value"], 1) if r["t2m_value"] is not None else None,
            "Anomaly vs month/hour normal": _anom_label(r["t2m_anomaly_z"]),
        }
        if live_mode:
            rec["Observed (UTC)"] = r["observed_at"]
            rec["Source"] = r["source"]
        else:
            rec["Forecast uncertainty (±)"] = (
                round(r["forecast_uncertainty"], 2)
                if r["forecast_uncertainty"] is not None else None
            )
            rec["Source"] = "synthetic" if r.get("synthetic") else "ERA5-Land"
        records.append(rec)

    st.dataframe(pd.DataFrame(records), width='stretch', hide_index=True)

    if live_mode:
        st.caption(
            "Live weather data by [Open-Meteo.com](https://open-meteo.com/) "
            "(CC-BY 4.0). Anomalies are computed against ERA5-Land 2018–2022 "
            "climatological normals for the same calendar month and hour."
        )
