"""Sectoral decision-support layer: turning a what-if scenario into decisions.

Why this module exists
----------------------
A scenario that outputs "+4 °C, feels like 46 °C" is meteorologically correct
but operationally inert. The question a panel — or a municipal officer — will
ask is *"so what, and who acts on it?"*

This module answers that by mapping a scenario's temperature/rainfall response
onto **published, operational decision thresholds** used by real institutions:

=========================  ==========================================================
Indicator                  Basis (all external, citable — no invented coefficients)
=========================  ==========================================================
sWBGT                      Australian BOM simplified approximation of Wet Bulb Globe
                           Temperature. NOTE: known to overestimate heat stress in
                           hot-humid regions (Kong & Huber 2022) — see
                           :data:`SWBGT_BIAS_CAVEAT`.
Labour work/rest capacity  ACGIH / ISO 7243 Threshold Limit Values — permissible
                           work fraction per hour by WBGT and workload.
Crop yield sensitivity     Zhao et al. (2017), PNAS: per +1 °C, global mean yields
                           fall ~6.0% wheat, ~3.2% rice, ~7.4% maize, ~3.1% soybean.
Heat Action Plan trigger   Ahmedabad Heat Action Plan (South Asia's first, 2013):
                           yellow 41.1–43.0 °C, orange 43.1–44.9 °C, red >= 45 °C.
Urban flood risk           IMD 24-hour rainfall warning classes: heavy 64.5–115.5,
                           very heavy 115.6–204.4, extremely heavy >= 204.5 mm.
Cooling energy demand      Cooling Degree Hours on the conventional 18 °C base.
=========================  ==========================================================

Design principle
----------------
Every threshold above comes from an external published source. This module
performs **arithmetic on those thresholds**, it does not invent dose-response
coefficients. Where a published sensitivity is applied outside its original
scope (notably crop yield, see :func:`crop_yield_delta`), the docstring and
:data:`SECTORAL_DISCLAIMER` say so explicitly.

Scope limits (important)
------------------------
These are *indicative decision-support signals derived from a model scenario*,
not official warnings, agronomic advice, medical guidance, or utility planning
figures. They inherit the scenario engine's disclaimer: the underlying
perturbation is a model-sensitivity experiment, not a validated physical
prediction.

References
----------
- Zhao, C. et al. (2017). "Temperature increase reduces global yields of major
  crops in four independent estimates." PNAS 114(35).
  https://www.pnas.org/doi/10.1073/pnas.1701762114
- ACGIH Threshold Limit Values (heat stress work/rest regimens); ISO 7243.
- Knowlton, K. et al. (2014). "Development and Implementation of South Asia's
  First Heat-Health Action Plan in Ahmedabad." IJERPH 11(4).
  https://www.mdpi.com/1660-4601/11/4/3473
- India Meteorological Department rainfall warning classes.
  https://mausam.imd.gov.in/
- Australian Bureau of Meteorology, thermal stress / WBGT approximation.
  http://www.bom.gov.au/info/thermal_stress
- Kong, Q. & Huber, M. (2022). "Explicit Calculations of Wet-Bulb Globe
  Temperature Compared With Approximations and Why It Matters for Labor
  Productivity." Earth's Future 10(3). — documents the simplified-WBGT
  overestimation bias in hot-humid regions that applies to this project.
  https://ui.adsabs.harvard.edu/abs/2022EaFut..1002334K/abstract
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.impact import relative_humidity

SECTORAL_DISCLAIMER: str = (
    "Sectoral indicators apply published operational thresholds (ACGIH/ISO 7243 "
    "work-rest limits, Zhao et al. 2017 crop-yield sensitivities, Ahmedabad Heat "
    "Action Plan triggers, IMD rainfall classes, 18 C cooling-degree base) to a "
    "MODEL SCENARIO. They are indicative decision-support signals for "
    "exploration, not official warnings, agronomic advice, medical guidance, or "
    "utility planning figures. Simplified WBGT overestimates heat stress in "
    "hot-humid regions, so labour restrictions are conservative."
)


# ---------------------------------------------------------------------------
# Wet Bulb Globe Temperature (heat-stress standard used by ACGIH / ISO 7243)
# ---------------------------------------------------------------------------
def vapour_pressure_hpa(t2m_c: np.ndarray | float,
                        rh_pct: np.ndarray | float) -> np.ndarray:
    """Water-vapour pressure (hPa) from temperature (deg C) and RH (%)."""
    t = np.asarray(t2m_c, dtype=float)
    rh = np.asarray(rh_pct, dtype=float)
    return (rh / 100.0) * 6.105 * np.exp((17.27 * t) / (237.7 + t))


SWBGT_BIAS_CAVEAT: str = (
    "Simplified WBGT (sWBGT) is known to OVERESTIMATE heat stress in hot-humid "
    "regions such as India (Kong & Huber, 2022, Earth's Future). Labour "
    "restrictions derived from it should therefore be read as conservative / "
    "over-cautious rather than precise. The lower-bias alternative (ESI) "
    "requires solar-radiation input, which the ERA5-Land time-series endpoint "
    "does not provide."
)


def swbgt(t2m_c: np.ndarray | float,
          d2m_c: np.ndarray | float) -> np.ndarray:
    """Estimate **simplified** WBGT (sWBGT, deg C) — BOM approximation.

    ``sWBGT ~= 0.567 * Ta + 0.393 * e + 3.94`` where ``e`` is vapour pressure
    in hPa. WBGT (not plain air temperature) is the quantity occupational
    heat-stress standards are written against, which is why it is computed here.

    Known bias — read this before using the output
    ----------------------------------------------
    This simplified form **systematically overestimates heat stress in
    hot-humid climates**, and consequently overestimates labour-capacity loss
    (Kong & Huber, 2022, *Earth's Future*, "Explicit Calculations of Wet-Bulb
    Globe Temperature Compared With Approximations and Why It Matters for Labor
    Productivity"). India is squarely in that regime.

    Treat derived work/rest restrictions as **conservative upper bounds on
    heat stress**, not calibrated measurements. The lower-bias Environmental
    Stress Index (ESI) needs solar radiation, which is unavailable from the
    ERA5-Land time-series endpoint used by this project.

    It also excludes direct solar loading, so true outdoor-in-sun WBGT differs
    again.
    """
    t = np.asarray(t2m_c, dtype=float)
    rh = relative_humidity(t, d2m_c)
    e = vapour_pressure_hpa(t, rh)
    return 0.567 * t + 0.393 * e + 3.94


# ---------------------------------------------------------------------------
# Labour capacity (ACGIH / ISO 7243 work-rest regimens)
# ---------------------------------------------------------------------------
# ACGIH TLV work/rest regimens, expressed as the permissible *work fraction* of
# each hour. Published in deg F; converted to deg C here. Values are for
# acclimatized workers. Keyed by workload intensity.
#   Light  : ~<=200 kcal/h (e.g. light assembly)
#   Moderate: ~200-350 kcal/h (e.g. walking with load)
#   Heavy  : ~>350 kcal/h (e.g. manual construction, harvesting)
_ACGIH_WBGT_LIMITS_C: dict[str, list[tuple[float, float]]] = {
    # (WBGT upper bound deg C, permissible work fraction)
    "light": [(30.0, 1.00), (30.6, 0.75), (31.7, 0.50), (32.2, 0.25)],
    "moderate": [(26.7, 1.00), (27.8, 0.75), (29.4, 0.50), (31.1, 0.25)],
    "heavy": [(25.0, 1.00), (25.6, 0.75), (27.8, 0.50), (30.0, 0.25)],
}


def labour_capacity(wbgt_c: float, workload: str = "moderate") -> tuple[float, str, str]:
    """Permissible work fraction per hour at a given WBGT (ACGIH TLVs).

    Parameters
    ----------
    wbgt_c : float
        Wet Bulb Globe Temperature in deg C.
    workload : {"light", "moderate", "heavy"}
        Work intensity class. Outdoor construction/agriculture is typically
        "heavy"; "moderate" is a reasonable default for mixed outdoor work.

    Returns
    -------
    tuple[float, str, str]
        ``(work_fraction, label, alert_level)`` where ``work_fraction`` is the
        permissible fraction of each hour that can be worked (1.0 = continuous
        work permissible) and ``alert_level`` is none/yellow/orange/red.

    Note
    ----
    ACGIH values assume acclimatized workers and shaded conditions consistent
    with the WBGT estimate supplied. Below the lowest threshold continuous work
    is permissible; above the highest, work should effectively stop.
    """
    key = workload.lower()
    if key not in _ACGIH_WBGT_LIMITS_C:
        raise ValueError(
            f"Unknown workload '{workload}'. Expected one of "
            f"{sorted(_ACGIH_WBGT_LIMITS_C)}."
        )
    bands = _ACGIH_WBGT_LIMITS_C[key]
    w = float(wbgt_c)

    for upper, fraction in bands:
        if w <= upper:
            if fraction >= 1.0:
                return 1.0, "Continuous work permissible", "none"
            if fraction >= 0.75:
                return 0.75, "75% work / 25% rest per hour", "yellow"
            if fraction >= 0.50:
                return 0.50, "50% work / 50% rest per hour", "orange"
            return 0.25, "25% work / 75% rest per hour", "red"

    return 0.0, "Work should cease (above ACGIH limits)", "red"


# ---------------------------------------------------------------------------
# Crop yield sensitivity (Zhao et al. 2017, PNAS)
# ---------------------------------------------------------------------------
# Percentage yield change per +1 deg C, from a four-method meta-analysis.
CROP_YIELD_PCT_PER_DEG_C: dict[str, float] = {
    "wheat": -6.0,
    "rice": -3.2,
    "maize": -7.4,
    "soybean": -3.1,
}


# The published sensitivities were derived over roughly 1-4 deg C of warming.
# Beyond this the linear extrapolation is not supportable, so we flag it.
CROP_COEFF_VALID_MAX_DEG_C: float = 4.0


def crop_yield_delta(
    delta_t_c: float,
    crops: tuple[str, ...] = ("wheat", "rice", "maize"),
) -> tuple[dict[str, float], bool]:
    """Indicative yield change (%) for a **sustained** temperature change.

    Applies the Zhao et al. (2017) per-degree sensitivities.

    Parameters
    ----------
    delta_t_c : float
        *Sustained* temperature change in deg C (positive = warming). This
        should be the scenario's **intended perturbation** (the policy question,
        e.g. "what if it were 2 deg C warmer?"), **not** the model's transient
        hour-by-hour response — see the note below.
    crops : tuple[str, ...]
        Which crops to report.

    Returns
    -------
    tuple[dict[str, float], bool]
        ``(crop -> percentage yield change, extrapolated)`` where
        ``extrapolated`` is True when ``|delta_t_c|`` exceeds the range the
        published coefficients were derived over.

    Important scope caveats
    -----------------------
    1. The coefficients describe yield response to **global mean** warming
       sustained across a growing season, without CO2 fertilisation,
       adaptation, or genetic improvement. Applying them to a *local* delta
       assumes the sensitivity transfers.
    2. They are calibrated over roughly 1-4 deg C. Larger deltas are flagged as
       extrapolation.
    3. Losses are floored at -100% because a yield cannot fall below zero; the
       linear relationship is not meaningful near that floor.
    4. This is an order-of-magnitude illustration of agricultural exposure,
       **not** a local yield forecast.
    """
    delta = float(delta_t_c)
    extrapolated = abs(delta) > CROP_COEFF_VALID_MAX_DEG_C

    out: dict[str, float] = {}
    for crop in crops:
        key = crop.lower()
        if key not in CROP_YIELD_PCT_PER_DEG_C:
            raise ValueError(
                f"Unknown crop '{crop}'. Known: {sorted(CROP_YIELD_PCT_PER_DEG_C)}."
            )
        # Floor at -100%: a crop cannot lose more than its entire yield.
        out[key] = max(-100.0, CROP_YIELD_PCT_PER_DEG_C[key] * delta)
    return out, extrapolated


# ---------------------------------------------------------------------------
# Heat Action Plan trigger (Ahmedabad HAP — South Asia's first, 2013)
# ---------------------------------------------------------------------------
def heat_action_plan_level(t_max_c: float) -> tuple[str, str, str]:
    """Map a maximum temperature to Ahmedabad Heat Action Plan alert stages.

    Returns ``(stage, alert_level, recommended_action)``. Thresholds follow the
    published Ahmedabad HAP: hot-day advisory (yellow) 41.1-43.0 deg C, orange
    43.1-44.9 deg C, red at or above 45 deg C.

    This is the operational bridge: it converts a scenario into the same
    trigger language an Indian municipal heat-action plan already uses.
    """
    t = float(t_max_c)
    if t >= 45.0:
        return (
            "Red alert — extreme heat",
            "red",
            "Full HAP activation: emergency cooling centres, public warnings, "
            "hospital surge readiness, halt outdoor labour in peak hours.",
        )
    if t >= 43.1:
        return (
            "Orange alert",
            "orange",
            "Heightened response: extend cooling-centre and water-kiosk hours, "
            "target vulnerable groups, brief health facilities.",
        )
    if t >= 41.1:
        return (
            "Yellow alert — hot day advisory",
            "yellow",
            "Issue public advisory, pre-position ORS/water, alert outdoor "
            "worker supervisors.",
        )
    return ("No HAP trigger", "none", "Routine monitoring.")


# ---------------------------------------------------------------------------
# Urban flood risk (IMD 24-hour rainfall warning classes)
# ---------------------------------------------------------------------------
def urban_flood_risk(rain_mm_24h: float,
                     peak_intensity_mm_h: float | None = None) -> tuple[str, str, str]:
    """Assess urban flood risk from accumulated rainfall and peak intensity.

    Accumulation is scored against IMD 24-hour warning classes. Peak hourly
    intensity is used as a *flash-flood* modifier, since short intense bursts
    overwhelm urban drainage even when the daily total is moderate.

    Returns ``(label, alert_level, recommended_action)``.
    """
    r = float(rain_mm_24h)
    if r >= 204.5:
        label, level = "Extremely heavy rain — severe urban flood risk", "red"
        action = ("Activate flood response: pumping stations, evacuate low-lying "
                  "areas, suspend transport in vulnerable corridors.")
    elif r >= 115.6:
        label, level = "Very heavy rain — high urban flood risk", "orange"
        action = ("Pre-deploy pumps and rescue teams, clear drains, issue "
                  "waterlogging advisories.")
    elif r >= 64.5:
        label, level = "Heavy rain — moderate urban flood risk", "yellow"
        action = "Monitor drainage hotspots, ready pumping capacity."
    else:
        label, level = "No significant flood risk", "none"
        action = "Routine monitoring."

    # Flash-flood escalation: >50 mm in a single hour is a recognised
    # cloudburst-scale intensity that can flood regardless of the daily total.
    if peak_intensity_mm_h is not None and peak_intensity_mm_h >= 50.0:
        if level in ("none", "yellow"):
            level = "orange"
        label += f" (flash-flood intensity: {peak_intensity_mm_h:.0f} mm/h)"
        action = ("Flash-flood risk from short-duration intensity: " + action)

    return label, level, action


# ---------------------------------------------------------------------------
# Cooling energy demand
# ---------------------------------------------------------------------------
def cooling_energy_demand(t2m_series_c: np.ndarray,
                          base_c: float = 18.0) -> float:
    """Total Cooling Degree Hours over a temperature series (18 deg C base).

    CDH = sum over hours of max(T - base, 0). The 18 deg C base is the
    conventional reference for building cooling-energy estimation. Higher CDH
    implies higher cooling electricity demand.
    """
    t = np.asarray(t2m_series_c, dtype=float)
    return float(np.sum(np.clip(t - base_c, 0.0, None)))


# ---------------------------------------------------------------------------
# Aggregate scenario assessment
# ---------------------------------------------------------------------------
@dataclass
class SectoralAssessment:
    """Decision-support view of a baseline-vs-scenario pair."""

    # Heat stress / labour
    wbgt_baseline_c: float
    wbgt_scenario_c: float
    work_fraction_baseline: float
    work_fraction_scenario: float
    labour_label: str
    labour_alert: str
    labour_capacity_lost_pct: float
    # True when the BASELINE already exceeds ACGIH limits, so the scenario
    # cannot show further measurable loss ("already unsafe", not "no impact").
    labour_baseline_already_unsafe: bool

    # Heat Action Plan
    hap_stage_baseline: str
    hap_stage_scenario: str
    hap_alert: str
    hap_action: str

    # Agriculture
    peak_delta_t_c: float
    applied_warming_c: float
    crop_yield_delta_pct: dict[str, float]
    crop_yield_extrapolated: bool

    # Water / flood
    flood_label: str
    flood_alert: str
    flood_action: str
    rain_24h_baseline_mm: float
    rain_24h_scenario_mm: float

    # Energy
    cdh_baseline: float
    cdh_scenario: float
    cooling_demand_change_pct: float

    disclaimer: str = field(default=SECTORAL_DISCLAIMER)
    # Travels with every assessment so the sWBGT bias is never surfaced silently.
    swbgt_caveat: str = field(default=SWBGT_BIAS_CAVEAT)


def assess_scenario(
    baseline: np.ndarray,
    perturbed: np.ndarray,
    feature_names: list[str],
    workload: str = "heavy",
    applied_warming_c: float | None = None,
) -> SectoralAssessment:
    """Build a full sectoral decision-support assessment for a scenario.

    Parameters
    ----------
    baseline, perturbed : np.ndarray
        Trajectories in physical units, shape ``(horizon, n_features)``, as
        returned by :func:`src.scenario.run_scenario`.
    feature_names : list[str]
        Ordered feature names (must include ``t2m``, ``d2m``, ``tp``).
    workload : {"light", "moderate", "heavy"}
        Work intensity for the labour assessment. Outdoor construction and
        agriculture are typically "heavy".
    applied_warming_c : float, optional
        The scenario's **intended** temperature perturbation in deg C (e.g.
        ``4.0`` for a "+4 deg C" what-if). Crop-yield sensitivity is evaluated
        against this rather than the model's transient response, because the
        published coefficients describe *sustained seasonal* warming and the
        autoregressive rollout can transiently overshoot. Falls back to the
        mean modelled delta when not supplied.

    Returns
    -------
    SectoralAssessment
        Heat-stress/labour, Heat-Action-Plan, agricultural, flood, and energy
        indicators for the baseline and the scenario.
    """
    required = {"t2m", "d2m", "tp"}
    missing = required - set(feature_names)
    if missing:
        raise ValueError(f"feature_names missing required variables: {sorted(missing)}")

    idx = {f: i for i, f in enumerate(feature_names)}
    bt = baseline[:, idx["t2m"]]
    pt = perturbed[:, idx["t2m"]]
    bd = baseline[:, idx["d2m"]]
    pd_ = perturbed[:, idx["d2m"]]
    b_rain = np.clip(baseline[:, idx["tp"]], 0.0, None)
    p_rain = np.clip(perturbed[:, idx["tp"]], 0.0, None)

    # --- Heat stress at the hottest hour of each trajectory ---
    b_hot = int(np.argmax(bt))
    p_hot = int(np.argmax(pt))
    wbgt_b = float(swbgt(bt[b_hot], bd[b_hot]))
    wbgt_p = float(swbgt(pt[p_hot], pd_[p_hot]))

    frac_b, _, _ = labour_capacity(wbgt_b, workload)
    frac_p, lab_label, lab_alert = labour_capacity(wbgt_p, workload)
    capacity_lost = (frac_b - frac_p) * 100.0
    # If the baseline is already at zero permissible work, the scenario cannot
    # show additional measurable loss. Flag this so the UI can say "already
    # beyond safe limits" rather than the misleading "0% capacity lost".
    baseline_already_unsafe = frac_b <= 0.0

    # --- Heat Action Plan on peak temperature ---
    hap_b, _, _ = heat_action_plan_level(float(np.max(bt)))
    hap_p, hap_alert, hap_action = heat_action_plan_level(float(np.max(pt)))

    # --- Agriculture ---
    # Crop coefficients describe SUSTAINED seasonal warming, so evaluate them
    # against the intended perturbation rather than the model's transient
    # (and possibly overshooting) response.
    mean_delta = float(np.mean(pt - bt))
    peak_delta = float(np.max(np.abs(pt - bt)))
    warming_for_crops = (
        float(applied_warming_c) if applied_warming_c is not None else mean_delta
    )
    yields, yields_extrapolated = crop_yield_delta(warming_for_crops)

    # --- Flood: scale the horizon's accumulation to a 24-hour equivalent ---
    horizon = max(1, baseline.shape[0])
    scale = 24.0 / horizon
    rain_b_24 = float(np.sum(b_rain) * scale)
    rain_p_24 = float(np.sum(p_rain) * scale)
    flood_label, flood_alert, flood_action = urban_flood_risk(
        rain_p_24, peak_intensity_mm_h=float(np.max(p_rain)) if p_rain.size else None
    )

    # --- Energy ---
    cdh_b = cooling_energy_demand(bt)
    cdh_p = cooling_energy_demand(pt)
    demand_change = ((cdh_p - cdh_b) / cdh_b * 100.0) if cdh_b > 0 else (
        100.0 if cdh_p > 0 else 0.0
    )

    return SectoralAssessment(
        wbgt_baseline_c=wbgt_b,
        wbgt_scenario_c=wbgt_p,
        work_fraction_baseline=frac_b,
        work_fraction_scenario=frac_p,
        labour_label=lab_label,
        labour_alert=lab_alert,
        labour_capacity_lost_pct=capacity_lost,
        labour_baseline_already_unsafe=baseline_already_unsafe,
        hap_stage_baseline=hap_b,
        hap_stage_scenario=hap_p,
        hap_alert=hap_alert,
        hap_action=hap_action,
        peak_delta_t_c=peak_delta,
        applied_warming_c=warming_for_crops,
        crop_yield_delta_pct=yields,
        crop_yield_extrapolated=yields_extrapolated,
        flood_label=flood_label,
        flood_alert=flood_alert,
        flood_action=flood_action,
        rain_24h_baseline_mm=rain_b_24,
        rain_24h_scenario_mm=rain_p_24,
        cdh_baseline=cdh_b,
        cdh_scenario=cdh_p,
        cooling_demand_change_pct=demand_change,
    )
