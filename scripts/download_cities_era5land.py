"""Download REAL ERA5-Land data for the multi-city registry.

This is a thin, city-aware driver around the existing, well-tested
``scripts/download_era5land.py`` logic. For each non-primary city in
``configs/cities.yaml`` it issues the same CDS time-series requests Delhi used
(same dataset, same six variables, same 2018-2024 period), but writes each
city's raw ZIPs and extracted NetCDF into a per-city directory so nothing
collides:

    data/raw/era5land/<slug>/era5land_<group>_<start>_<end>.zip
    data/processed/source_extracted/<slug>/era5land_<group>_<start>_<end>.nc

Requirements (same as the Delhi download)
-----------------------------------------
- ``cdsapi`` installed:            make install-era5
- Valid ``~/.cdsapirc`` with your CDS API key.
- ERA5-Land time-series licence accepted on the CDS website for your account.

This script only issues authenticated requests through your own credentials;
it does not read, print, or store your API key.

Usage
-----
    python scripts/download_cities_era5land.py                # all non-primary cities
    python scripts/download_cities_era5land.py --city mumbai  # one city
    python scripts/download_cities_era5land.py --pilot-days 3 # short test slice

After downloading, run per city:
    python scripts/preprocess.py --city <slug>
    python scripts/train_lstm.py --city <slug>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the proven download/extract/hash helpers and constants.
from scripts.download_era5land import (  # noqa: E402
    DATASET_ID,
    VARIABLE_GROUPS,
    download_group,
    extract_zip,
    sha256_file,
)

from src.cities import City, load_cities  # noqa: E402
from src.config import load_config  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def _city_dirs(city: City, is_pilot: bool) -> tuple[Path, Path]:
    """Return (raw_dir, extracted_dir) for a city, namespaced by slug."""
    if is_pilot:
        base_raw = PROJECT_ROOT / "data" / "sample" / "pilot" / "raw"
        base_ext = PROJECT_ROOT / "data" / "sample" / "pilot" / "extracted"
    else:
        base_raw = PROJECT_ROOT / "data" / "raw" / "era5land"
        base_ext = PROJECT_ROOT / "data" / "processed" / "source_extracted"
    return base_raw / city.slug, base_ext / city.slug


def download_city(client, city: City, start_date: str, end_date: str,
                  is_pilot: bool) -> dict:
    """Download and extract all variable groups for one city.

    Returns a per-city provenance record. Raises on any group failure so the
    caller can report exactly which city/group broke.
    """
    raw_dir, extract_dir = _city_dirs(city, is_pilot)
    raw_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== %s (lat=%.2f, lon=%.2f) ===", city.name, city.latitude, city.longitude)
    logger.info("Raw -> %s", raw_dir)
    logger.info("Extracted -> %s", extract_dir)

    file_records: list[dict] = []
    for group_name, variables in VARIABLE_GROUPS.items():
        zip_path = download_group(
            client=client,
            group_name=group_name,
            variables=variables,
            latitude=city.latitude,
            longitude=city.longitude,
            start_date=start_date,
            end_date=end_date,
            output_dir=raw_dir,
        )
        if zip_path is None or not zip_path.exists():
            raise RuntimeError(f"Download failed for {city.slug}/{group_name}")

        nc_filename = f"era5land_{group_name}_{start_date}_{end_date}.nc"
        nc_path = extract_zip(zip_path, extract_dir, nc_filename)

        file_records.append({
            "group": group_name,
            "variables_cds_request": variables,
            "zip_path": str(zip_path.relative_to(PROJECT_ROOT)),
            "zip_sha256": sha256_file(zip_path),
            "nc_path": str(nc_path.relative_to(PROJECT_ROOT)),
            "nc_sha256": sha256_file(nc_path),
            "retrieved_at": datetime.now(UTC).isoformat(),
        })

    return {
        "synthetic": False,
        "provider": "Copernicus Climate Change Service (C3S) / ECMWF",
        "dataset_id": DATASET_ID,
        "dataset_doi": "10.24381/ee82e357",
        "license": "CC-BY 4.0",
        "city": city.name,
        "slug": city.slug,
        "requested_location": {"latitude": city.latitude, "longitude": city.longitude},
        "requested_period": {"start": start_date, "end": end_date},
        "is_pilot": is_pilot,
        "raw_files": file_records,
        "retrieval_date": datetime.now(UTC).strftime("%Y-%m-%d"),
    }


def main(city_slug: str | None = None, pilot_days: int | None = None) -> None:
    setup_logging("INFO")

    cities = load_cities()
    cities_cfg = load_config("configs/cities.yaml")
    start_date = cities_cfg["time_range"]["start"]
    end_date = cities_cfg["time_range"]["end"]

    is_pilot = pilot_days is not None
    if is_pilot:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = (start_dt + timedelta(days=pilot_days)).strftime("%Y-%m-%d")
        logger.info("PILOT MODE: %d days (%s to %s)", pilot_days, start_date, end_date)

    # Default: every non-primary city (Delhi already has real data).
    targets = [c for c in cities if not c.is_primary]
    if city_slug is not None:
        targets = [c for c in cities if c.slug == city_slug.lower()]
        if not targets:
            raise SystemExit(f"Unknown city '{city_slug}'.")

    logger.info("Downloading REAL ERA5-Land for: %s", [c.slug for c in targets])
    logger.info("Dataset: %s | period %s to %s", DATASET_ID, start_date, end_date)

    import cdsapi  # imported here so --help works without the era5 extra
    client = cdsapi.Client()

    metadata_dir = PROJECT_ROOT / "data" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for city in targets:
        record = download_city(client, city, start_date, end_date, is_pilot)
        prov_path = metadata_dir / f"provenance_{city.slug}.json"
        with open(prov_path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
        logger.info("[%s] provenance -> %s", city.slug, prov_path)
        all_records.append(record)

    logger.info("All %d city downloads complete.", len(all_records))
    logger.info("Next: python scripts/preprocess.py --city <slug>  (then train_lstm.py --city <slug>)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ERA5-Land data for multiple cities")
    parser.add_argument("--city", default=None, help="Single city slug (default: all non-primary)")
    parser.add_argument("--pilot-days", type=int, default=None,
                        help="Download only N days from the start date (quick test)")
    args = parser.parse_args()
    main(args.city, args.pilot_days)
