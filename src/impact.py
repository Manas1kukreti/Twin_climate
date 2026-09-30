"""Human-meaningful impact indicators for ClimateTwin forecasts.

This module translates raw forecasted meteorological variables (temperature,
dewpoint, precipitation, wind) into indicators a non-specialist can act on:
apparent "feels-like" temperature, relative humidity, IMD-based heat-risk and
rainfall categories, and colour-coded alert cards.

Design principles
-----------------
- Indicators use *standard, citable formulas and thresholds* (NOAA Heat Index;
  IMD heatwave and rainfall categories), not invented coefficients. This keeps
  the impact layer defensible.
- The forecast itself is a model output; these indicators are deterministic
  functions applied on top of it. They do not add predictive skill, they add
  interpretation.

Sources (thresholds paraphrased for compliance)
-----------------------------------------------
- Heat Index (apparent temperature): NOAA/NWS Rothfusz regression.
- IMD heatwave criteria (plains), actual-max-temperature basis:
  heatwave >= 45 C, severe heatwave >= 47 C; heatwave onset >= 40 C.
  https://www.ncdc.gov.in/uploads/pdf/heat_ppt1.pdf
- IMD 24-hour rainfall categories (mm/24h).
  https://mausam.imd.gov.in/

Disclaimer
----------
Impact indicators are standard formulas applied to model forecasts. They are
informational, not official warnings, and combine with the scenario module's
disclaimer that outputs are model-based sensitivity experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IMPACT_DISCLAIMER: str = (
    "Impact indicators are standard meteorological formulas (NOAA Heat Index, "
    "IMD categories) applied to model forecasts. They are informational only, "
    "not official warnings, and do not establish real-world causal effects."
)


# ---------------------------------------------------------------------------
# Relative humidity from temperature and dewpoint (Magnus formula)
# ---------------------------------------------------------------------------
def relative_humidity(t2m_c: np.ndarray | float, d2m_c: np.ndarray | float) -> np.ndarray:
    """Relative humidity (%) from air temperature and dewpoint (both deg C).

    Uses the Magnus-Tetens approximation. Result is clipped to [0, 100].

    Parameters
    ----------
    t2m_c : array or float
        2 m air temperature in degrees Celsius.
    d2m_c : array or float
        2 m dewpoint temperature in degrees Celsius.

    Returns
    -------
    np.ndarray
        Relative humidity in percent.
    """
    a, b = 17.625, 243.04
    t = np.asarray(t2m_c, dtype=float)
    d = np.asarray(d2m_c, dtype=float)
    rh = 100.0 * np.exp((a * d) / (b + d)) / np.exp((a * t) / (b + t))
    return np.clip(rh, 0.0, 100.0)


# ---------------------------------------------------------------------------
# Heat Index / apparent temperature (NOAA Rothfusz regression)
# ---------------------------------------------------------------------------
def heat_index(t2m_c: np.ndarray | float, d2m_c: np.ndarray | float) -> np.ndarray:
    """Apparent "feels-like" temperature (deg C) via NOAA Heat Index.

    The NOAA Rothfusz regression is defined in Fahrenheit with adjustments; we
    convert to/from Celsius. For temperatures below ~27 C (80 F) the Heat Index
    is approximately equal to the air temperature, matching NOAA behaviour.

    Parameters
    ----------
    t2m_c : array or float
        2 m air temperature in degrees Celsius.
    d2m_c : array or float
        2 m dewpoint temperature in degrees Celsius (used to derive humidity).

    Returns
    -------
    np.ndarray
        Apparent temperature in degrees Celsius.
    """
    t_c = np.asarray(t2m_c, dtype=float)
    rh = relative_humidity(t_c, d2m_c)

    t_f = t_c * 9.0 / 5.0 + 32.0

    # Simple formula (valid at lower heat); used when HI < 80 F.
    hi_simple_f = 0.5 * (t_f + 61.0 + (t_f - 68.0) * 1.2 + rh * 0.094)

    # Full Rothfusz regression.
    hi_full_f = (
        -42.379
        + 2.04901523 * t_f
        + 10.14333127 * rh
        - 0.22475541 * t_f * rh
        - 0.00683783 * t_f**2
        - 0.05481717 * rh**2
        + 0.00122874 * t_f**2 * rh
        + 0.00085282 * t_f * rh**2
        - 0.00000199 * t_f**2 * rh**2
    )

    # Adjustments (per NOAA) for low-humidity and high-humidity edge cases.
    low_rh = (rh < 13) & (t_f >= 80) & (t_f <= 112)
    # Clip the radicand to >= 0 so the sqrt is always defined; the result is
    # only used where `low_rh` is True (t_f within [80,112], radicand >= 0).
    radicand = np.clip((17 - np.abs(t_f - 95.0)) / 17.0, 0.0, None)
    adj_low = ((13 - rh) / 4.0) * np.sqrt(radicand)
    hi_full_f = np.where(low_rh, hi_full_f - adj_low, hi_full_f)

    high_rh = (rh > 85) & (t_f >= 80) & (t_f <= 87)
    adj_high = ((rh - 85) / 10.0) * ((87 - t_f) / 5.0)
    hi_full_f = np.where(high_rh, hi_full_f + adj_high, hi_full_f)

    # Use the simple formula where the averaged HI is below 80 F.
    hi_f = np.where(hi_simple_f < 80.0, hi_simple_f, hi_full_f)

    return (hi_f - 32.0) * 5.0 / 9.0


# ---------------------------------------------------------------------------
# IMD heatwave category (plains, actual-max-temperature basis)
# ---------------------------------------------------------------------------
def heat_category(t2m_c: float) -> tuple[str, str]:
    """Classify a temperature against IMD heatwave thresholds (plains).

    Actual-maximum-temperature basis: heatwave onset at 40 C, heatwave at
    45 C, severe heatwave at 47 C. Returns a (label, alert_level) pair where
    alert_level is one of "none", "yellow", "orange", "red".

    Note
    ----
    IMD criteria formally use daily maximum temperature and/or departure from
    local normals; here we apply the actual-temperature thresholds to a
    forecast value as an interpretable proxy.
    """
    if t2m_c >= 47.0:
        return "Severe heatwave", "red"
    if t2m_c >= 45.0:
        return "Heatwave", "orange"
    if t2m_c >= 40.0:
        return "Hot (heatwave onset)", "yellow"
    return "Normal", "none"


# ---------------------------------------------------------------------------
# IMD 24-hour rainfall category
# ---------------------------------------------------------------------------
def rainfall_category(rain_mm_24h: float) -> tuple[str, str]:
    """Classify a 24-hour rainfall total (mm) into IMD categories.

    Returns a (label, alert_level) pair. Alert levels escalate for the heavier
    categories that carry flooding risk.
    """
    r = float(rain_mm_24h)
    if r >= 204.5:
        return "Extremely heavy rain", "red"
    if r >= 115.6:
        return "Very heavy rain", "orange"
    if r >= 64.5:
        return "Heavy rain", "yellow"
    if r >= 15.6:
        return "Moderate rain", "none"
    if r >= 2.5:
        return "Light rain", "none"
    if r >= 0.1:
        return "Very light rain", "none"
    return "No rain", "none"


def wind_category(u10: float, v10: float) -> tuple[str, str]:
    """Classify 10 m wind speed (from u/v components, m/s) into simple bands."""
    speed = float(np.hypot(u10, v10))
    if speed >= 17.2:  # gale and above (Beaufort >= 8)
        return "Gale / very windy", "orange"
    if speed >= 10.8:  # strong breeze / near-gale
        return "Strong wind", "yellow"
    if speed >= 5.5:
        return "Breezy", "none"
    return "Calm / light wind", "none"


# ---------------------------------------------------------------------------
# Aggregated impact summary + alert cards
# ---------------------------------------------------------------------------
@dataclass
class ImpactReport:
    """Human-meaningful impact summary for one forecast state."""

    air_temp_c: float
    feels_like_c: float
    relative_humidity_pct: float
    heat_label: str
    heat_alert: str
    rainfall_label: str
    rainfall_alert: str
    wind_label: str
    wind_alert: str
    alerts: list[str]
    disclaimer: str = IMPACT_DISCLAIMER


def summarize_state(
    t2m_c: float,
    d2m_c: float,
    tp_mm_24h: float,
    u10: float,
    v10: float,
) -> ImpactReport:
    """Build a full impact report for a single forecast state.

    Parameters
    ----------
    t2m_c, d2m_c : float
        Air temperature and dewpoint in deg C.
    tp_mm_24h : float
        24-hour-equivalent precipitation total in mm.
    u10, v10 : float
        10 m wind components in m/s.

    Returns
    -------
    ImpactReport
        Feels-like temperature, humidity, IMD categories, and alert cards.
    """
    feels = float(heat_index(t2m_c, d2m_c))
    rh = float(relative_humidity(t2m_c, d2m_c))
    heat_label, heat_alert = heat_category(t2m_c)
    rain_label, rain_alert = rainfall_category(tp_mm_24h)
    wind_label, wind_alert = wind_category(u10, v10)

    # Compose alert cards (only for non-"none" levels), ordered by severity.
    severity = {"red": 3, "orange": 2, "yellow": 1, "none": 0}
    cards: list[tuple[int, str]] = []
    if heat_alert != "none":
        cards.append((severity[heat_alert],
                      f"[{heat_alert.upper()}] {heat_label}: air {t2m_c:.1f} C, "
                      f"feels like {feels:.1f} C"))
    if rain_alert != "none":
        cards.append((severity[rain_alert],
                      f"[{rain_alert.upper()}] {rain_label}: {tp_mm_24h:.1f} mm/24h"))
    if wind_alert != "none":
        cards.append((severity[wind_alert],
                      f"[{wind_alert.upper()}] {wind_label}"))

    cards.sort(key=lambda c: c[0], reverse=True)
    alerts = [text for _, text in cards]

    return ImpactReport(
        air_temp_c=float(t2m_c),
        feels_like_c=feels,
        relative_humidity_pct=rh,
        heat_label=heat_label,
        heat_alert=heat_alert,
        rainfall_label=rain_label,
        rainfall_alert=rain_alert,
        wind_label=wind_label,
        wind_alert=wind_alert,
        alerts=alerts,
    )


# ---------------------------------------------------------------------------
# Sector indicators (Phase: multi-city)
# ---------------------------------------------------------------------------
# These translate a forecast state into sector-relevant indicators for
# agriculture, public health, and energy. As with the core impact layer, every
# threshold is drawn from a standard, citable source and each indicator adds
# *interpretation*, not predictive skill. They are informational, not official
# advisories (see SECTOR_DISCLAIMER).
#
# Sources (thresholds paraphrased for compliance):
# - Crop heat stress: cardinal temperatures for cereals. Wheat grain-set is
#   impaired above ~34 C and severely above ~38 C; many C4/tropical cereals
#   (rice, maize) show heat stress in the mid-30s C during sensitive stages.
#   Ref: FAO crop-ecology guidance and Hatfield & Prueger (2015),
#   "Temperature extremes: Effect on plant growth and development".
# - Heat-health caution: NOAA/NWS Heat Index caution bands
#   (Caution >= 27 C, Extreme Caution >= 32 C, Danger >= 41 C,
#    Extreme Danger >= 54 C apparent temperature).
#   https://www.weather.gov/safety/heat-index
# - Cooling demand: Cooling Degree Hours relative to an 18 C base temperature,
#   the conventional base for building energy-demand estimation (ASHRAE).

SECTOR_DISCLAIMER: str = (
    "Sector indicators (agriculture, health, energy) apply standard published "
    "thresholds (FAO/Hatfield crop cardinal temperatures, NOAA Heat Index "
    "caution bands, ASHRAE 18 C cooling base) to model forecasts. They are "
    "informational interpretations, not official agronomic, medical, or "
    "utility advisories."
)


def crop_heat_stress(t2m_c: float) -> tuple[str, str]:
    """Classify air temperature against cereal crop heat-stress thresholds.

    Returns a ``(label, alert_level)`` pair. Thresholds follow cardinal
    temperatures for staple cereals: reproductive-stage stress begins in the
    mid-30s C and becomes severe approaching ~38 C.

    Note
    ----
    Real crop impact depends on growth stage, duration, humidity and cultivar;
    this is a single-value proxy for interpretation only.
    """
    t = float(t2m_c)
    if t >= 38.0:
        return "Severe crop heat stress", "red"
    if t >= 34.0:
        return "Crop heat stress", "orange"
    if t >= 30.0:
        return "Mild crop heat stress", "yellow"
    return "No crop heat stress", "none"


def heat_health_caution(feels_like_c: float) -> tuple[str, str]:
    """Map apparent ("feels-like") temperature to NOAA Heat Index caution bands.

    Parameters
    ----------
    feels_like_c : float
        Apparent temperature in deg C (e.g. from :func:`heat_index`).

    Returns
    -------
    tuple[str, str]
        ``(label, alert_level)`` using NOAA/NWS bands converted to Celsius.
    """
    hi = float(feels_like_c)
    if hi >= 54.0:
        return "Extreme danger (heat stroke likely)", "red"
    if hi >= 41.0:
        return "Danger (heat cramps/exhaustion likely)", "red"
    if hi >= 32.0:
        return "Extreme caution (heat exhaustion possible)", "orange"
    if hi >= 27.0:
        return "Caution (fatigue with prolonged exposure)", "yellow"
    return "No heat-health concern", "none"


def cooling_degree_hours(t2m_c: float | np.ndarray,
                         base_c: float = 18.0) -> np.ndarray:
    """Cooling Degree Hours above a base temperature (default 18 C, ASHRAE).

    CDH = max(T - base, 0), summed over hours by the caller. A higher value
    means more cooling energy demand. Returned per-hour so callers can sum or
    average over a horizon.
    """
    t = np.asarray(t2m_c, dtype=float)
    return np.clip(t - base_c, 0.0, None)


def cooling_demand_category(cdh_24h: float) -> tuple[str, str]:
    """Classify a 24-hour Cooling Degree Hours total into demand bands.

    Bands are pragmatic groupings over the 18 C-base CDH accumulated across a
    day: higher CDH implies sustained cooling load. Provided for at-a-glance
    interpretation of the energy indicator.
    """
    c = float(cdh_24h)
    if c >= 240.0:      # ~ sustained 10 C above base all day
        return "Very high cooling demand", "red"
    if c >= 120.0:      # ~ sustained 5 C above base
        return "High cooling demand", "orange"
    if c >= 24.0:       # ~ sustained 1 C above base
        return "Moderate cooling demand", "yellow"
    return "Low cooling demand", "none"


@dataclass
class SectorReport:
    """Sector-level interpretation of one forecast state."""

    crop_label: str
    crop_alert: str
    health_label: str
    health_alert: str
    cooling_label: str
    cooling_alert: str
    cooling_degree_hours_24h: float
    disclaimer: str = SECTOR_DISCLAIMER


def summarize_sectors(
    t2m_c: float,
    d2m_c: float,
) -> SectorReport:
    """Build a sector report (agriculture, health, energy) for one state.

    Parameters
    ----------
    t2m_c, d2m_c : float
        Air temperature and dewpoint in deg C. Dewpoint feeds the feels-like
        temperature used for the heat-health band.
    """
    feels = float(heat_index(t2m_c, d2m_c))
    crop_label, crop_alert = crop_heat_stress(t2m_c)
    health_label, health_alert = heat_health_caution(feels)
    # Single-hour CDH scaled to a 24h-equivalent for the demand band.
    cdh_24h = float(cooling_degree_hours(t2m_c)) * 24.0
    cooling_label, cooling_alert = cooling_demand_category(cdh_24h)

    return SectorReport(
        crop_label=crop_label,
        crop_alert=crop_alert,
        health_label=health_label,
        health_alert=health_alert,
        cooling_label=cooling_label,
        cooling_alert=cooling_alert,
        cooling_degree_hours_24h=cdh_24h,
    )
