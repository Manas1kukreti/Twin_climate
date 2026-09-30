"""Generate physically plausible SYNTHETIC ERA5-like data for extra cities.

WHY THIS EXISTS
---------------
Training the additional cities (Mumbai, Bengaluru, Kolkata, Chennai) requires
ERA5-Land hourly data, which is only obtainable from the Copernicus CDS API
with the user's own credentials and an accepted licence. To let the full
multi-city pipeline (training, DB, map, sector indicators) run and be verified
*without* those credentials, this script fabricates hourly climate series with
realistic seasonal + diurnal structure driven by each city's latitude and
climate zone.

⚠️ THE DATA IS SYNTHETIC. It is NOT a reanalysis product and must never be
presented as measured or real. Every artifact this script writes is flagged
``synthetic: true`` in a sidecar ``synthetic_provenance.json`` so downstream
code (and the map UI) can label it clearly. When the user later runs the real
``scripts/download_era5land.py`` + ``scripts/preprocess.py`` per city, the same
processed-file contract is produced from real data and this script is not used.

WHAT IT WRITES (per non-primary city ``<slug>``)
-----------------------------------------------
    data/processed/<slug>/train.csv, val.csv, test.csv
    data/processed/<slug>/train_scaled.csv, val_scaled.csv, test_scaled.csv
    data/processed/<slug>/scaler.joblib
    data/processed/<slug>/synthetic_provenance.json

These mirror exactly what ``scripts/preprocess.py`` produces for Delhi, so
``scripts/train_lstm.py --city <slug>`` and the dashboard consume them without
changes.

Usage
-----
    python scripts/generate_synthetic_city_data.py            # all non-primary cities
    python scripts/generate_synthetic_city_data.py --city mumbai
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cities import City, load_cities  # noqa: E402
from src.config import load_config  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402
from src.preprocessing import (  # noqa: E402
    chronological_split,
    fit_scaler,
    save_scaler,
    transform_data,
)

logger = logging.getLogger(__name__)

FEATURE_ORDER = ["t2m", "d2m", "sp", "tp", "u10", "v10"]

# Per-climate-zone tuning knobs for the synthetic generator. Values are chosen
# to give each city a recognisable, physically sensible regime — not to match
# any specific reanalysis record.
#   t_mean      annual-mean 2m temperature (°C)
#   t_seasonal  peak-to-mean seasonal amplitude (°C)
#   t_diurnal   day-night swing amplitude (°C)
#   rh_base     baseline relative humidity used to derive dewpoint (fraction)
#   monsoon     (peak_month, width_months, intensity_mm_per_h) rainy-season shape
#   sp_mean     mean surface pressure (hPa), lowered for higher elevation
#   wind        typical 10 m wind speed scale (m/s)
CLIMATE_PROFILES: dict[str, dict] = {
    "mumbai": dict(t_mean=27.5, t_seasonal=4.0, t_diurnal=6.0, rh_base=0.78,
                   monsoon=(7, 2.2, 1.4), sp_mean=1008.0, wind=3.5),
    "bengaluru": dict(t_mean=24.0, t_seasonal=3.5, t_diurnal=9.0, rh_base=0.62,
                      monsoon=(8, 3.0, 0.5), sp_mean=910.0, wind=2.8),
    "kolkata": dict(t_mean=27.0, t_seasonal=7.0, t_diurnal=8.0, rh_base=0.75,
                    monsoon=(7, 2.5, 1.2), sp_mean=1010.0, wind=2.5),
    "chennai": dict(t_mean=28.5, t_seasonal=3.5, t_diurnal=6.5, rh_base=0.72,
                    monsoon=(11, 1.8, 1.1), sp_mean=1009.0, wind=3.2),
    # Fallback for any city without a specific profile.
    "_default": dict(t_mean=26.0, t_seasonal=6.0, t_diurnal=8.0, rh_base=0.65,
                     monsoon=(7, 2.5, 0.9), sp_mean=1005.0, wind=3.0),
}


def _dewpoint_from_rh(t_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Invert the Magnus formula to get dewpoint (°C) from temp + RH."""
    a, b = 17.625, 243.04
    rh = np.clip(rh, 0.01, 1.0)
    gamma = np.log(rh) + (a * t_c) / (b + t_c)
    return (b * gamma) / (a - gamma)


def synthesize_city(city: City, start: str, end: str, seed: int) -> pd.DataFrame:
    """Build an hourly synthetic climate DataFrame for one city.

    Returns a DataFrame with a ``timestamp`` column and the six feature
    columns in physical units (°C, °C, hPa, mm, m/s, m/s) — matching the
    output of the real preprocessing's unit conversion.
    """
    rng = np.random.default_rng(seed + abs(hash(city.slug)) % 10_000)
    prof = CLIMATE_PROFILES.get(city.slug, CLIMATE_PROFILES["_default"])

    timestamps = pd.date_range(start=start, end=f"{end} 23:00", freq="h")
    n = len(timestamps)
    doy = timestamps.dayofyear.to_numpy().astype(float)
    hod = timestamps.hour.to_numpy().astype(float)

    # Seasonal cycle: warmest near mid-year for northern-hemisphere India.
    seasonal = prof["t_seasonal"] * np.cos(2 * np.pi * (doy - 172) / 365.25)
    # Diurnal cycle: peak ~15:00 local.
    diurnal = prof["t_diurnal"] * np.cos(2 * np.pi * (hod - 15) / 24.0)
    # Latitude nudge: cooler further from equator (relative to profile mean).
    lat_adj = -(city.latitude - 20.0) * 0.15
    noise_t = rng.normal(0, 1.1, n)
    t2m = prof["t_mean"] + lat_adj + seasonal + diurnal + noise_t

    # Relative humidity: higher in the monsoon window; derive dewpoint from it.
    peak_month, width, intensity = prof["monsoon"]
    month = timestamps.month.to_numpy().astype(float)
    # Gaussian bump around the monsoon peak month (wrap-aware distance).
    dmonth = np.abs(((month - peak_month + 6) % 12) - 6)
    monsoon_factor = np.exp(-0.5 * (dmonth / width) ** 2)
    rh = np.clip(prof["rh_base"] + 0.18 * monsoon_factor + rng.normal(0, 0.04, n),
                 0.15, 0.99)
    d2m = _dewpoint_from_rh(t2m, rh)
    d2m = np.minimum(d2m, t2m - 0.1)  # dewpoint cannot exceed air temp

    # Precipitation: mostly zero, with monsoon-weighted wet hours (mm/hour).
    wet_prob = 0.02 + 0.22 * monsoon_factor
    is_wet = rng.random(n) < wet_prob
    rain_amt = rng.gamma(shape=1.3, scale=intensity * (0.5 + monsoon_factor), size=n)
    tp = np.where(is_wet, rain_amt, 0.0)

    # Surface pressure: mean per profile, mild seasonal + noise.
    sp = prof["sp_mean"] - 2.0 * np.cos(2 * np.pi * (doy - 172) / 365.25) + rng.normal(0, 0.6, n)

    # Wind components: scale-appropriate, slightly gustier in the monsoon.
    wind_scale = prof["wind"] * (1.0 + 0.4 * monsoon_factor)
    u10 = rng.normal(0, wind_scale, n)
    v10 = rng.normal(0, wind_scale, n)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "t2m": t2m.astype(np.float32),
        "d2m": d2m.astype(np.float32),
        "sp": sp.astype(np.float32),
        "tp": np.clip(tp, 0.0, None).astype(np.float32),
        "u10": u10.astype(np.float32),
        "v10": v10.astype(np.float32),
    })
    return df


def process_and_save(city: City, df: pd.DataFrame, split_cfg: dict,
                     scaler_type: str) -> None:
    """Split, scale, and persist a city's synthetic data using the real helpers."""
    train_end = split_cfg["train"]["end"]
    val_end = split_cfg["validation"]["end"]

    train_df, val_df, test_df = chronological_split(df, train_end=train_end, val_end=val_end)

    out_dir = city.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(city.split_csv("train", scaled=False), index=False)
    val_df.to_csv(city.split_csv("val", scaled=False), index=False)
    test_df.to_csv(city.split_csv("test", scaled=False), index=False)

    scaler = fit_scaler(train_df, scaler_type, FEATURE_ORDER)
    transform_data(train_df, scaler, FEATURE_ORDER).to_csv(
        city.split_csv("train", scaled=True), index=False)
    transform_data(val_df, scaler, FEATURE_ORDER).to_csv(
        city.split_csv("val", scaled=True), index=False)
    transform_data(test_df, scaler, FEATURE_ORDER).to_csv(
        city.split_csv("test", scaled=True), index=False)

    scaler_meta = {
        "scaler_type": scaler_type,
        "feature_order": FEATURE_ORDER,
        "fitting_period": f"{split_cfg['train']['start']} to {train_end}",
        "normalization_policy": "train_only_target_location",
        "n_train_samples": len(train_df),
        "synthetic": True,
    }
    save_scaler(scaler, FEATURE_ORDER, scaler_meta, city.scaler_path)

    provenance = {
        "synthetic": True,
        "warning": (
            "SYNTHETIC DATA — generated by scripts/generate_synthetic_city_data.py. "
            "Not a reanalysis product; do not present as measured/real ERA5-Land data."
        ),
        "city": city.name,
        "slug": city.slug,
        "latitude": city.latitude,
        "longitude": city.longitude,
        "climate_zone": city.climate_zone,
        "feature_order": FEATURE_ORDER,
        "rows": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "generated_at": datetime.now(UTC).isoformat(),
    }
    with open(out_dir / "synthetic_provenance.json", "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2)

    logger.info(
        "[%s] synthetic data written: train=%d val=%d test=%d -> %s",
        city.slug, len(train_df), len(val_df), len(test_df), out_dir,
    )


def main(city_slug: str | None = None, seed: int = 42) -> None:
    setup_logging("INFO")
    cities = load_cities()
    cities_cfg = load_config("configs/cities.yaml")
    data_cfg = load_config("configs/data.yaml")

    split_cfg = cities_cfg["split"]
    time_range = cities_cfg["time_range"]
    scaler_type = data_cfg.get("scaler_type", "StandardScaler")

    # Only non-primary cities are synthesized; the primary city (Delhi) uses
    # its real, already-produced processed data.
    targets = [c for c in cities if not c.is_primary]
    if city_slug is not None:
        targets = [c for c in cities if c.slug == city_slug.lower()]
        if not targets:
            raise SystemExit(f"Unknown city '{city_slug}'.")
        if targets[0].is_primary:
            raise SystemExit(
                f"'{city_slug}' is the primary city (real data); not synthesizing."
            )

    logger.info("Synthesizing %d city/cities: %s",
                len(targets), [c.slug for c in targets])
    logger.warning(
        "Generating SYNTHETIC data — flagged synthetic:true in each city's "
        "synthetic_provenance.json. Not real ERA5-Land."
    )

    for city in targets:
        df = synthesize_city(city, time_range["start"], time_range["end"], seed)
        process_and_save(city, df, split_cfg, scaler_type)

    logger.info("Done. Next: python scripts/train_lstm.py --city <slug>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic city data")
    parser.add_argument("--city", default=None, help="Single city slug (default: all non-primary)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.city, args.seed)
