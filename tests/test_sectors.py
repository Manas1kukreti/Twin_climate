"""Tests for the sector impact indicators and the multi-city registry.

Covers the citable sector layer added for the multi-city map (crop heat
stress, heat-health caution, cooling demand) and the City path-resolution
helper that keeps the primary city (Delhi) on its original artifact paths.
"""

from __future__ import annotations

import numpy as np

from src.impact import (
    SectorReport,
    cooling_degree_hours,
    cooling_demand_category,
    crop_heat_stress,
    heat_health_caution,
    summarize_sectors,
)


class TestCropHeatStress:
    def test_no_stress_when_mild(self) -> None:
        label, alert = crop_heat_stress(25.0)
        assert alert == "none"
        assert "No crop" in label

    def test_mild_stress_band(self) -> None:
        _, alert = crop_heat_stress(31.0)
        assert alert == "yellow"

    def test_stress_band(self) -> None:
        _, alert = crop_heat_stress(35.0)
        assert alert == "orange"

    def test_severe_band(self) -> None:
        _, alert = crop_heat_stress(39.0)
        assert alert == "red"

    def test_monotonic_severity(self) -> None:
        order = {"none": 0, "yellow": 1, "orange": 2, "red": 3}
        temps = [20, 30, 34, 38, 45]
        sev = [order[crop_heat_stress(t)[1]] for t in temps]
        assert sev == sorted(sev)


class TestHeatHealthCaution:
    def test_no_concern_low(self) -> None:
        _, alert = heat_health_caution(24.0)
        assert alert == "none"

    def test_caution(self) -> None:
        _, alert = heat_health_caution(28.0)
        assert alert == "yellow"

    def test_extreme_caution(self) -> None:
        _, alert = heat_health_caution(35.0)
        assert alert == "orange"

    def test_danger(self) -> None:
        _, alert = heat_health_caution(45.0)
        assert alert == "red"

    def test_extreme_danger(self) -> None:
        label, alert = heat_health_caution(55.0)
        assert alert == "red"
        assert "Extreme danger" in label


class TestCoolingDemand:
    def test_cdh_zero_below_base(self) -> None:
        # At or below the 18 C base, cooling degree hours are zero.
        assert float(cooling_degree_hours(15.0)) == 0.0
        assert float(cooling_degree_hours(18.0)) == 0.0

    def test_cdh_positive_above_base(self) -> None:
        assert float(cooling_degree_hours(28.0)) == 10.0

    def test_cdh_array(self) -> None:
        out = cooling_degree_hours(np.array([10.0, 18.0, 20.0, 30.0]))
        assert list(out) == [0.0, 0.0, 2.0, 12.0]

    def test_demand_categories(self) -> None:
        assert cooling_demand_category(0.0)[1] == "none"
        assert cooling_demand_category(50.0)[1] == "yellow"
        assert cooling_demand_category(150.0)[1] == "orange"
        assert cooling_demand_category(300.0)[1] == "red"


class TestSummarizeSectors:
    def test_returns_report(self) -> None:
        rep = summarize_sectors(t2m_c=42.0, d2m_c=20.0)
        assert isinstance(rep, SectorReport)
        assert rep.disclaimer  # non-empty disclaimer attached
        assert rep.cooling_degree_hours_24h > 0

    def test_hot_dry_triggers_crop_and_cooling(self) -> None:
        rep = summarize_sectors(t2m_c=40.0, d2m_c=15.0)
        assert rep.crop_alert in {"orange", "red"}
        assert rep.cooling_alert in {"orange", "red"}

    def test_mild_is_calm(self) -> None:
        # At/below the 18 C cooling base there is no cooling demand and no
        # crop heat stress.
        rep = summarize_sectors(t2m_c=17.0, d2m_c=12.0)
        assert rep.crop_alert == "none"
        assert rep.cooling_alert == "none"

    def test_warm_day_has_moderate_cooling(self) -> None:
        # 22 C sustained is 4 cooling-degree-hours/h -> 96 CDH/24h -> moderate.
        rep = summarize_sectors(t2m_c=22.0, d2m_c=14.0)
        assert rep.cooling_alert == "yellow"


class TestCityRegistry:
    def test_primary_city_keeps_original_paths(self) -> None:
        from src.cities import primary_city

        delhi = primary_city()
        assert delhi.is_primary
        assert delhi.checkpoint_path.name == "lstm_delhi_seed42_001.pt"
        assert delhi.split_csv("test", scaled=True).name == "test_scaled.csv"
        # Primary city's processed dir is the un-namespaced data/processed.
        assert str(delhi.processed_dir).endswith("data/processed")
        assert delhi.metrics_path.name == "lstm_delhi_metrics.json"

    def test_non_primary_city_is_namespaced(self) -> None:
        from src.cities import get_city

        mum = get_city("mumbai")
        assert not mum.is_primary
        assert mum.checkpoint_path.name == "lstm_mumbai_seed42_001.pt"
        assert mum.processed_dir.name == "mumbai"
        assert mum.run_id == "lstm_scratch_mumbai_seed42_001"

    def test_all_expected_cities_present(self) -> None:
        from src.cities import city_map

        cities = city_map()
        for slug in ("delhi", "mumbai", "bengaluru", "kolkata", "chennai"):
            assert slug in cities
