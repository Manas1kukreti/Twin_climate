"""Temporary robustness audit — deleted after running."""
import numpy as np

from dashboard.pages_impl.map_view import _anomaly_color, _radius_from_uncertainty
from src.sectors import assess_scenario, crop_yield_delta, labour_capacity, urban_flood_risk

print("=== None/NaN robustness in map helpers ===")
for v in [None, 0.0, -3.5, 3.5, 99.0, -99.0]:
    print("  _anomaly_color({}) = {}".format(v, _anomaly_color(v)))
for v in [None, 0.0, 100.0]:
    print("  _radius_from_uncertainty({}) = {}".format(v, _radius_from_uncertainty(v)))

FEAT = ["t2m", "d2m", "sp", "tp", "u10", "v10"]


def traj(t, d, tp, h=12):
    a = np.zeros((h, 6))
    a[:, 0] = t
    a[:, 1] = d
    a[:, 2] = 1000
    a[:, 3] = tp
    return a


print()
print("=== sectors robustness ===")
a = assess_scenario(traj(10, 5, 0), traj(10, 5, 0), FEAT, applied_warming_c=0.0)
print("  identical traj: cooling%={:.1f} crop={:.1f} flood={}".format(
    a.cooling_demand_change_pct, a.crop_yield_delta_pct["wheat"], a.flood_alert))

a = assess_scenario(traj(30, 20, 0, 1), traj(34, 20, 0, 1), FEAT, applied_warming_c=4.0)
print("  horizon=1: hap={}".format(a.hap_stage_scenario))

a = assess_scenario(traj(-5, -10, 0), traj(-1, -10, 0), FEAT, applied_warming_c=4.0)
print("  sub-zero: work_frac={} cooling%={:.1f}".format(
    a.work_fraction_scenario, a.cooling_demand_change_pct))

print("  flood(0mm)={}  flood(80mm,None)={}".format(
    urban_flood_risk(0.0)[1], urban_flood_risk(80.0, None)[1]))
print("  labour(WBGT=0)={}".format(labour_capacity(0.0, "heavy")[0]))
print("  crop(0C)={}".format(crop_yield_delta(0.0)[0]["wheat"]))

# Extreme precipitation
a = assess_scenario(traj(28, 24, 0), traj(28, 24, 200), FEAT)
print("  extreme rain 200mm/h: flood={} rain24h={:.0f}".format(
    a.flood_alert, a.rain_24h_scenario_mm))
print()
print("ALL ROBUSTNESS CHECKS COMPLETED")
