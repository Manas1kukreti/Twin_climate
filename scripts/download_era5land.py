"""Download ERA5-Land hourly time-series data for a single location.

Reads retrieval parameters from ``configs/data.yaml`` and downloads
raw data to ``data/raw/era5land/`` as immutable CDS ZIP response archives.
Extracts NetCDF files to ``data/processed/source_extracted/``.

Usage
-----
    python scripts/download_era5land.py
    python scripts/download_era5land.py --config configs/data.yaml
    python scripts/download_era5land.py --pilot          # 48-hour test only
    python scripts/download_era5land.py --pilot-days 3   # custom pilot length

Requires
--------
- ``cdsapi`` package installed.
- Valid ``~/.cdsapirc`` with CDS API key.
- License accepted for the ERA5-Land time-series dataset on CDS.

Notes
-----
The time-series endpoint automatically selects the nearest 0.1° grid point
to the requested coordinates.  The actual returned coordinates must be
recorded from the downloaded file during raw-data inspection (Task 1.3).

Variable names follow the ERA5-Land time-series Product User Guide (PUG):
https://confluence.ecmwf.int/pages/viewpage.action?pageId=536218894

Immutability policy
-------------------
- CDS response archives (.zip) are saved to ``data/raw/era5land/`` and
  NEVER modified or deleted after successful download.
- Extracted NetCDF files go to ``data/processed/source_extracted/``.
- Neither location overwrites existing files silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so src/ imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID = "reanalysis-era5-land-timeseries"

# CDS variable names per PUG Tables 1-3, grouped by PUG data-group.
VARIABLE_GROUPS: dict[str, list[str]] = {
    "2m_temperature": [
        "2m_temperature",
        "2m_dewpoint_temperature",
    ],
    "pressure_precipitation": [
        "surface_pressure",
        "total_precipitation",
    ],
    "wind": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ],
}

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "era5land"
EXTRACTED_DIR = PROJECT_ROOT / "data" / "processed" / "source_extracted"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sha256_file(filepath: Path) -> str:
    """Compute SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_request(
    variables: list[str],
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    data_format: str = "netcdf",
) -> dict:
    """Build a CDS API request dict for the time-series endpoint."""
    return {
        "variable": variables,
        "location": {"latitude": latitude, "longitude": longitude},
        "date": [f"{start_date}/{end_date}"],
        "data_format": data_format,
    }


def extract_zip(zip_path: Path, extract_dir: Path, expected_nc_name: str) -> Path:
    """Extract a single NetCDF from a CDS ZIP response.

    Parameters
    ----------
    zip_path : Path
        Path to the immutable ZIP archive.
    extract_dir : Path
        Destination directory for the extracted NetCDF.
    expected_nc_name : str
        The filename to use for the extracted NetCDF.

    Returns
    -------
    Path
        Path to the extracted NetCDF file.

    Raises
    ------
    ValueError
        If the ZIP does not contain exactly one .nc file.
    """
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a valid ZIP file: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        nc_members = [m for m in zf.namelist() if m.endswith(".nc")]
        if len(nc_members) != 1:
            raise ValueError(
                f"Expected exactly 1 .nc file in ZIP, found {len(nc_members)}: {nc_members}"
            )
        inner_name = nc_members[0]
        data = zf.read(inner_name)

    extract_dir.mkdir(parents=True, exist_ok=True)
    nc_path = extract_dir / expected_nc_name
    if nc_path.exists():
        logger.warning("Extracted file already exists, skipping: %s", nc_path)
        return nc_path

    with open(nc_path, "wb") as f:
        f.write(data)

    logger.info(
        "Extracted: %s -> %s (%d bytes)",
        zip_path.name,
        nc_path.name,
        len(data),
    )
    return nc_path


def download_group(
    client,
    group_name: str,
    variables: list[str],
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    output_dir: Path,
) -> Path | None:
    """Download one variable group as a ZIP and return the archive path.

    The CDS time-series endpoint returns ZIP archives containing NetCDF.
    This function saves the raw ZIP with a .zip extension — never modifying
    it after download.
    """
    zip_filename = f"era5land_{group_name}_{start_date}_{end_date}.zip"
    zip_path = output_dir / zip_filename

    if zip_path.exists():
        logger.info("Archive already exists, skipping download: %s", zip_path)
        return zip_path

    request = build_request(
        variables=variables,
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
    )

    logger.info(
        "Requesting %s: variables=%s, location=(%s, %s), date=%s/%s",
        DATASET_ID,
        variables,
        latitude,
        longitude,
        start_date,
        end_date,
    )

    try:
        result = client.retrieve(DATASET_ID, request)
        result.download(str(zip_path))
        size = zip_path.stat().st_size
        logger.info("Downloaded archive: %s (%d bytes)", zip_path.name, size)

        # Validate it is actually a ZIP
        if not zipfile.is_zipfile(zip_path):
            logger.error("Downloaded file is not a valid ZIP: %s", zip_path)
            return None

        return zip_path
    except Exception:
        logger.exception("Failed to download group '%s'", group_name)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(config_path: str = "configs/data.yaml", pilot_days: int | None = None) -> None:
    """Run the ERA5-Land download pipeline.

    Parameters
    ----------
    config_path : str
        Path to the data YAML config.
    pilot_days : int or None
        If set, override the date range to download only this many days
        starting from ``time_range.start``. Pilot data is saved to
        ``data/sample/pilot/`` instead of the authoritative locations.
    """
    setup_logging("INFO")

    # Load configuration
    config = load_config(config_path)
    latitude = config.get("latitude", config.get("requested_latitude"))
    longitude = config.get("longitude", config.get("requested_longitude"))
    time_range = config.get("time_range", {})
    start_date = str(time_range.get("start"))
    end_date = str(time_range.get("end"))

    # Validate required fields
    if latitude is None or longitude is None:
        raise ValueError(
            f"latitude and longitude must be set in data config. "
            f"Current values: latitude={latitude}, longitude={longitude}"
        )
    if not start_date or start_date == "None" or not end_date or end_date == "None":
        raise ValueError(
            f"time_range.start and time_range.end must be set in data config. "
            f"Current values: start={start_date}, end={end_date}"
        )

    # Pilot mode: override date range and output directories
    is_pilot = pilot_days is not None
    if is_pilot:
        from datetime import timedelta

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        pilot_end_dt = start_dt + timedelta(days=pilot_days)
        end_date = pilot_end_dt.strftime("%Y-%m-%d")
        logger.info("PILOT MODE: %d days (%s to %s)", pilot_days, start_date, end_date)

    # Set output directories
    if is_pilot:
        raw_dir = PROJECT_ROOT / "data" / "sample" / "pilot" / "raw"
        extract_dir = PROJECT_ROOT / "data" / "sample" / "pilot" / "extracted"
    else:
        raw_dir = RAW_DIR
        extract_dir = EXTRACTED_DIR

    raw_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Dataset: %s", DATASET_ID)
    logger.info("Location: lat=%s, lon=%s", latitude, longitude)
    logger.info("Period: %s to %s", start_date, end_date)
    logger.info("Raw archives -> %s", raw_dir)
    logger.info("Extracted NetCDF -> %s", extract_dir)

    # Initialize CDS client
    import cdsapi

    client = cdsapi.Client()

    # Download and extract each variable group
    file_records: list[dict] = []

    for group_name, variables in VARIABLE_GROUPS.items():
        # Download ZIP
        zip_path = download_group(
            client=client,
            group_name=group_name,
            variables=variables,
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            output_dir=raw_dir,
        )

        if zip_path is None or not zip_path.exists():
            logger.error("Download failed for group '%s'. Aborting.", group_name)
            sys.exit(1)

        # Hash the immutable ZIP
        zip_hash = sha256_file(zip_path)
        zip_size = zip_path.stat().st_size

        # Extract NetCDF
        nc_filename = f"era5land_{group_name}_{start_date}_{end_date}.nc"
        nc_path = extract_zip(zip_path, extract_dir, nc_filename)
        nc_hash = sha256_file(nc_path)
        nc_size = nc_path.stat().st_size

        file_records.append(
            {
                "group": group_name,
                "variables_cds_request": variables,
                "zip_filename": zip_path.name,
                "zip_path": str(zip_path.relative_to(PROJECT_ROOT)),
                "zip_size_bytes": zip_size,
                "zip_sha256": zip_hash,
                "nc_filename": nc_path.name,
                "nc_path": str(nc_path.relative_to(PROJECT_ROOT)),
                "nc_size_bytes": nc_size,
                "nc_sha256": nc_hash,
                "extracted_from": zip_path.name,
                "retrieved_at": datetime.now(UTC).isoformat(),
            }
        )

    # Write provenance
    cdsapi_version = getattr(cdsapi, "__version__", "unknown")

    import numpy
    import pandas
    import torch
    import xarray

    provenance = {
        "provider": "Copernicus Climate Change Service (C3S) / ECMWF",
        "dataset_id": DATASET_ID,
        "dataset_doi": "10.24381/ee82e357",
        "source_type": "reanalysis_grid_cell",
        "product_user_guide": (
            "https://confluence.ecmwf.int/pages/viewpage.action?pageId=536218894"
        ),
        "license": "CC-BY 4.0",
        "requested_location": {"latitude": latitude, "longitude": longitude},
        "actual_returned_location": "PENDING — inspect extracted NetCDF",
        "requested_period": {"start": start_date, "end": end_date},
        "actual_first_timestamp": "PENDING — inspect extracted NetCDF",
        "actual_last_timestamp": "PENDING — inspect extracted NetCDF",
        "validated_sampling_interval": "PENDING — inspect extracted NetCDF",
        "variable_request_names": [
            v for group_vars in VARIABLE_GROUPS.values() for v in group_vars
        ],
        "returned_netcdf_short_names": "PENDING — inspect extracted NetCDF",
        "units": "PENDING — inspect extracted NetCDF",
        "cds_request_parameters": {
            "dataset": DATASET_ID,
            "data_format": "netcdf",
            "location": {"latitude": latitude, "longitude": longitude},
            "date_range": f"{start_date}/{end_date}",
            "note": "credentials excluded",
        },
        "retrieval_date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "is_pilot": is_pilot,
        "raw_files": file_records,
        "software_versions": {
            "python": sys.version.split()[0],
            "cdsapi": cdsapi_version,
            "xarray": xarray.__version__,
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "torch": torch.__version__,
        },
    }

    suffix = "_pilot" if is_pilot else ""
    provenance_path = METADATA_DIR / f"provenance{suffix}.json"
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    logger.info("Provenance written to %s", provenance_path)

    # Summary
    logger.info(
        "All %d groups downloaded and extracted successfully.%s",
        len(file_records),
        " (PILOT)" if is_pilot else "",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ERA5-Land time-series data")
    parser.add_argument(
        "--config",
        default="configs/data.yaml",
        help="Path to data configuration YAML (default: configs/data.yaml)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run in pilot mode (48 hours from start date)",
    )
    parser.add_argument(
        "--pilot-days",
        type=int,
        default=None,
        help="Number of days for pilot (overrides --pilot default of 2)",
    )
    args = parser.parse_args()

    pilot_days = args.pilot_days
    if args.pilot and pilot_days is None:
        pilot_days = 2  # default 48 hours

    main(args.config, pilot_days=pilot_days)
