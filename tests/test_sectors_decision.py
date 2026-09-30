"""Tests for the sectoral decision-support layer (`src/sectors.py`).

These guard the *quantitative* claims the dashboard surfaces to a panel, so the
arithmetic against published thresholds must stay correct — including the
physical floor on crop-yield loss and the "baseline already unsafe" semantics
that a naive implementation gets wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.sectors import (
    CROP_YIELD_PCT_PER_DEG_C,
    assess_scenario,
    cooling_energy_demand,
    crop_yield_delta,
    heat_action_plan_level,
    labour_capacity,
    swbgt,
    urban_flood_risk,
    vapour_pressure_hpa,
)

FEATURES = ["t2m", "d2m", "sp", "tp", "u10", "v10"]


def _traj(t2m: float, d2m: float, tp: float, hours: int = 12) -> np.ndarray:
    """Build a constant trajectory with the given values."""
    arr = np.zeros((hours, len(FEATURES)), dtype=float)
    arr[:, 0] = t2m
    arr[:, 1] = d2m
    arr[:, 2] = 1000.0
    arr[:, 3] = tp
    return arr


class TestSWBGT:
    def test_increases_with_temperature(self) -> None:
        assert float(swbgt(40, 20)) > float(swbgt(30, 20))

    def test_increases_with_humidity(self) -> None:
        # Higher dewpoint at the same air temperature = more humid = higher WBGT.
        assert float(swbgt(35, 30)) > float(swbgt(35, 10))

    def test_vapour_pressure_positive(self) -> None:
        assert float(vapour_pressure_hpa(30, 50)) > 0


class TestLabourCapacity:
    def test_cool_allows_continuous_work(self) -> None:
        frac, _, alert = labour_capacity(20.0, "moderate")
        assert frac == 1.0
        assert alert == "none"

    def test_capacity_decreases_monotonically(self) -> None:
        fracs = [labour_capacity(w, "moderate")[0] for w in (20, 27, 28.5, 30, 40)]
        assert fracs == sorted(fracs, reverse=True)

    def test_extreme_heat_stops_work(self) -> None:
        frac, label, alert = labour_capacity(45.0, "heavy")
        assert frac == 0.0
        assert alert == "red"
        assert "cease" in label.lower()

    def test_heavy_work_stricter_than_light(self) -> None:
        # At the same WBGT, heavy work must permit no more work than light work.
        w = 29.0
        assert labour_capacity(w, "heavy")[0] <= labour_capacity(w, "light")[0]

    def test_unknown_workload_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown workload"):
            labour_capacity(30.0, "extreme")


class TestCropYield:
    def test_matches_published_coefficients(self) -> None:
        out, extrapolated = crop_yield_delta(1.0)
        assert out["wheat"] == pytest.approx(CROP_YIELD_PCT_PER_DEG_C["wheat"])
        assert not extrapolated

    def test_scales_linearly_in_valid_range(self) -> None:
        out, _ = crop_yield_delta(2.0)
        assert out["rice"] == pytest.approx(2 * CROP_YIELD_PCT_PER_DEG_C["rice"])

    def test_flags_extrapolation_beyond_validated_range(self) -> None:
        _, extrapolated = crop_yield_delta(10.0)
        assert extrapolated

    def test_loss_floored_at_minus_100_percent(self) -> None:
        # A crop cannot lose more than its entire yield; this was a real bug.
        out, _ = crop_yield_delta(50.0)
        assert all(v >= -100.0 for v in out.values())

    def test_cooling_reduces_loss(self) -> None:
        out, _ = crop_yield_delta(-1.0)
        assert out["wheat"] > 0  # negative warming -> positive yield change

    def test_unknown_crop_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown crop"):
            crop_yield_delta(1.0, crops=("banana",))


class TestHeatActionPlan:
    def test_no_trigger_when_mild(self) -> None:
        stage, alert, _ = heat_action_plan_level(35.0)
        assert alert == "none"
        assert "No HAP" in stage

    def test_ahmedabad_yellow_band(self) -> None:
        _, alert, _ = heat_action_plan_level(42.0)
        assert alert == "yellow"

    def test_ahmedabad_orange_band(self) -> None:
        _, alert, _ = heat_action_plan_level(44.0)
        assert alert == "orange"

    def test_ahmedabad_red_band(self) -> None:
        stage, alert, action = heat_action_plan_level(46.0)
        assert alert == "red"
        assert "Red alert" in stage
        assert action  # an actionable recommendation is always provided


class TestUrbanFloodRisk:
    def test_no_risk_when_dry(self) -> None:
        _, alert, _ = urban_flood_risk(5.0)
        assert alert == "none"

    def test_imd_heavy_band(self) -> None:
        _, alert, _ = urban_flood_risk(80.0)
        assert alert == "yellow"

    def test_imd_very_heavy_band(self) -> None:
        _, alert, _ = urban_flood_risk(150.0)
        assert alert == "orange"

    def test_imd_extremely_heavy_band(self) -> None:
        _, alert, _ = urban_flood_risk(250.0)
        assert alert == "red"

    def test_flash_flood_intensity_escalates(self) -> None:
        # Modest daily total but cloudburst intensity should still escalate.
        _, alert, _ = urban_flood_risk(20.0, peak_intensity_mm_h=60.0)
        assert alert == "orange"


class TestCoolingEnergy:
    def test_zero_below_base(self) -> None:
        assert cooling_energy_demand(np.array([10.0, 15.0, 18.0])) == 0.0

    def test_accumulates_above_base(self) -> None:
        assert cooling_energy_demand(np.array([20.0, 20.0])) == 4.0


class TestAssessScenario:
    def test_produces_full_assessment(self) -> None:
        base = _traj(30.0, 20.0, 0.0)
        pert = _traj(34.0, 20.0, 0.0)
        a = assess_scenario(base, pert, FEATURES, workload="heavy",
                            applied_warming_c=4.0)
        assert a.wbgt_scenario_c > a.wbgt_baseline_c
        assert a.applied_warming_c == 4.0
        assert a.crop_yield_delta_pct["wheat"] == pytest.approx(-24.0)
        assert a.disclaimer and a.swbgt_caveat

    def test_flags_baseline_already_unsafe(self) -> None:
        # Very hot+humid baseline already exceeds ACGIH heavy-work limits.
        base = _traj(45.0, 30.0, 0.0)
        pert = _traj(48.0, 30.0, 0.0)
        a = assess_scenario(base, pert, FEATURES, workload="heavy")
        assert a.labour_baseline_already_unsafe
        assert a.work_fraction_baseline == 0.0

    def test_cool_baseline_not_flagged_unsafe(self) -> None:
        base = _traj(18.0, 10.0, 0.0)
        pert = _traj(20.0, 10.0, 0.0)
        a = assess_scenario(base, pert, FEATURES, workload="heavy")
        assert not a.labour_baseline_already_unsafe

    def test_rain_scenario_triggers_flood(self) -> None:
        # 10 mm/h over 12 h -> 240 mm/24h-equivalent -> extremely heavy.
        base = _traj(28.0, 22.0, 0.0)
        pert = _traj(28.0, 22.0, 10.0)
        a = assess_scenario(base, pert, FEATURES)
        assert a.flood_alert == "red"
        assert a.rain_24h_scenario_mm > a.rain_24h_baseline_mm

    def test_warming_raises_cooling_demand(self) -> None:
        base = _traj(25.0, 18.0, 0.0)
        pert = _traj(31.0, 18.0, 0.0)
        a = assess_scenario(base, pert, FEATURES, applied_warming_c=6.0)
        assert a.cooling_demand_change_pct > 0

    def test_missing_variable_raises(self) -> None:
        base = _traj(30.0, 20.0, 0.0)
        with pytest.raises(ValueError, match="missing required variables"):
            assess_scenario(base, base, ["t2m", "d2m"])  # no tp
