"""Probe CDS capacity: validate extended variable names and time requests.

Two unknowns drive the whole scale-up plan:

1. **Variable-name validity** — the extended set (soil moisture, runoff,
   evaporation, radiation) must use exact ERA5-Land CDS names.
2. **Variables per request** — if one request accepts all ~15 variables, the
   full download is ~40 requests instead of ~240, turning a 12-hour job into
   roughly an hour. This dominates feasibility.

This script answers both empirically with small, cheap requests and reports
wall-clock timings so we can extrapolate the full download honestly.

Usage
-----
    python scripts/probe_cds_capacity.py
"""

from __future__ import annotations

import sys
import tempfile
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_ID = "reanalysis-era5-land-timeseries"

# Candidate extended variable set for an India-focused climate twin.
# Core 6 (already in use) + hazard/energy-relevant additions.
CORE = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "total_precipitation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]
EXTENDED = [
    "volumetric_soil_water_layer_1",   # drought (root-zone moisture)
    "volumetric_soil_water_layer_2",
    "soil_temperature_level_1",
    "surface_net_solar_radiation",     # energy / solar resource
    "surface_net_thermal_radiation",
    "total_evaporation",               # water balance
    "potential_evaporation",           # drought (atmospheric demand)
    "runoff",                          # flood
    "surface_runoff",                  # flash flood
]

TEST_LAT, TEST_LON = 28.61, 77.21  # Delhi (known-good land point)


def _request(variables: list[str], date_range: str) -> tuple[bool, str, float, int]:
    """Issue one request; return (ok, detail, seconds, bytes)."""
    import cdsapi

    client = cdsapi.Client(quiet=True, progress=False)
    req = {
        "variable": variables,
        "location": {"latitude": TEST_LAT, "longitude": TEST_LON},
        "date": [date_range],
        "data_format": "netcdf",
    }
    start = time.time()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "probe.zip"
            client.retrieve(DATASET_ID, req).download(str(out))
            size = out.stat().st_size
            # Count how many distinct .nc members / variables came back.
            with zipfile.ZipFile(out) as zf:
                members = [m for m in zf.namelist() if m.endswith(".nc")]
            elapsed = time.time() - start
            return True, f"{len(members)} nc member(s)", elapsed, size
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:200], time.time() - start, 0


def main() -> None:
    print("=" * 72)
    print("CDS CAPACITY PROBE")
    print("=" * 72)

    # --- Test 1: validate the extended variable names (1 day, cheap) ---
    print("\n[1] Validating extended variable names (1-day request each)...")
    valid, invalid = [], []
    for v in EXTENDED:
        ok, detail, secs, _ = _request([v], "2020-06-01/2020-06-01")
        status = "OK " if ok else "BAD"
        print(f"    {status} {v:34} ({secs:5.1f}s) {'' if ok else detail[:80]}")
        (valid if ok else invalid).append(v)

    # --- Test 2: how many variables per request? ---
    print("\n[2] Testing multi-variable request capacity (1-day)...")
    combined = CORE + valid
    ok, detail, secs, size = _request(combined, "2020-06-01/2020-06-01")
    print(f"    {len(combined)} variables in ONE request: "
          f"{'OK' if ok else 'FAILED'} ({secs:.1f}s, {size} bytes) {detail[:100]}")

    # --- Test 3: long-range timing (single group, to extrapolate) ---
    print("\n[3] Timing a long-range request (1995-2026, core variables)...")
    ok_long, detail_long, secs_long, size_long = _request(
        CORE, "1995-01-01/2026-09-01"
    )
    print(f"    {'OK' if ok_long else 'FAILED'} in {secs_long:.1f}s, "
          f"{size_long/1e6:.1f} MB — {detail_long[:100]}")

    # --- Summary / extrapolation ---
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Valid extended variables : {len(valid)}/{len(EXTENDED)}")
    if invalid:
        print(f"Invalid (drop these)     : {invalid}")
    print(f"Multi-variable request   : {'SUPPORTED' if ok else 'NOT supported'}")
    if ok_long and secs_long > 0:
        n_cities = 40
        per_city = secs_long if ok else secs_long * 3
        total_h = (n_cities * per_city) / 3600.0
        print(f"Long-range single request : {secs_long:.0f}s, {size_long/1e6:.1f} MB")
        print(f"Extrapolated {n_cities} cities  : ~{total_h:.1f} h sequential, "
              f"~{total_h/3:.1f} h at 3x parallel")


if __name__ == "__main__":
    main()
