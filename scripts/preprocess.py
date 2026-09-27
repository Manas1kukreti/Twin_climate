"""Production preprocessing pipeline for ClimateTwin Delhi dataset.

Reads authoritative extracted NetCDF files, cleans, converts units,
splits chronologically, fits scaler on training data, transforms all
splits, and saves processed artifacts.

Usage
-----
    python scripts/preprocess.py
    python scripts/preprocess.py --config configs/data.yaml

Source data (read-only):
    data/processed/source_extracted/*.nc

Output artifacts:
    data/processed/train.parquet
    data/processed/val.parquet
    data/processed/test.parquet
    data/processed/train_scaled.parquet
    data/processed/val_scaled.parquet
    data/processed/test_scaled.parquet
    data/processed/scaler.joblib
    data/metadata/preprocessing_report.json
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402
from src.preprocessing import (  # noqa: E402
    chronological_split,
    fit_scaler,
    handle_missing_values,
    remove_duplicates,
    save_scaler,
    transform_data,
    validate_timestamps,
)

logger = logging.getLogger(__name__)

EXTRACTED_DIR = PROJECT_ROOT / "data" / "processed" / "source_extracted"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"


def load_and_merge_netcdf(
    extracted_dir: Path,
    feature_order: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load all extracted NetCDF files and merge into a single DataFrame.

    Parameters
    ----------
    extracted_dir : Path
        Directory containing extracted .nc files.
    feature_order : list[str]
        Ordered list of NetCDF short variable names.
    start_date, end_date : str
        Date range substring for filename matching.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with 'timestamp' column and feature columns.
    """
    nc_files = sorted(extracted_dir.glob(f"era5land_*_{start_date}_{end_date}.nc"))
    if not nc_files:
        raise FileNotFoundError(
            f"No extracted NetCDF files found in {extracted_dir} "
            f"matching *_{start_date}_{end_date}.nc"
        )

    all_vars: dict[str, pd.Series] = {}
    timestamps = None

    for nc_path in nc_files:
        ds = xr.open_dataset(nc_path)
        t_key = "valid_time" if "valid_time" in ds.coords else "time"
        ts = pd.DatetimeIndex(ds[t_key].values)

        if timestamps is None:
            timestamps = ts
        else:
            assert ts.equals(timestamps), (
                f"Timestamp mismatch between files: {nc_path.name}"
            )

        for var in ds.data_vars:
            if var in feature_order:
                all_vars[var] = pd.Series(ds[var].values, name=var)

        ds.close()

    # Verify all features found
    missing = [f for f in feature_order if f not in all_vars]
    if missing:
        raise ValueError(f"Features not found in NetCDF files: {missing}")

    df = pd.DataFrame({"timestamp": timestamps})
    for feat in feature_order:
        df[feat] = all_vars[feat].values

    logger.info("Loaded %d rows, %d features from %d files.", len(df), len(feature_order), len(nc_files))
    return df


def convert_units(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Convert native ERA5-Land units to conventional units.

    Conversions:
        t2m: K → °C  (subtract 273.15)
        d2m: K → °C  (subtract 273.15)
        sp:  Pa → hPa (divide by 100)
        tp:  m → mm   (multiply by 1000)
        u10: already m/s
        v10: already m/s

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Converted DataFrame and a conversion log dict.
    """
    df_out = df.copy()
    log: dict = {}

    if "t2m" in df_out.columns:
        df_out["t2m"] = df_out["t2m"] - 273.15
        log["t2m"] = "K → °C (subtract 273.15)"

    if "d2m" in df_out.columns:
        df_out["d2m"] = df_out["d2m"] - 273.15
        log["d2m"] = "K → °C (subtract 273.15)"

    if "sp" in df_out.columns:
        df_out["sp"] = df_out["sp"] / 100.0
        log["sp"] = "Pa → hPa (divide by 100)"

    if "tp" in df_out.columns:
        df_out["tp"] = df_out["tp"] * 1000.0
        log["tp"] = "m → mm (multiply by 1000)"

    # u10, v10 are already in m/s
    log["u10"] = "m s**-1 → m/s (no conversion needed)"
    log["v10"] = "m s**-1 → m/s (no conversion needed)"

    logger.info("Unit conversions applied: %s", list(log.keys()))
    return df_out, log


def fix_precipitation_negatives(
    df: pd.DataFrame,
    col: str = "tp",
) -> tuple[pd.DataFrame, int]:
    """Clamp tiny negative precipitation values to zero.

    ERA5-Land de-accumulation can produce tiny negative values (order 1e-8 m
    = 1e-5 mm). These are numerical artifacts, not physical precipitation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with precipitation column (already in mm after conversion).
    col : str
        Precipitation column name.

    Returns
    -------
    tuple[pd.DataFrame, int]
        DataFrame with negatives clamped to 0, and count of values corrected.
    """
    df_out = df.copy()
    neg_mask = df_out[col] < 0
    n_corrected = int(neg_mask.sum())

    if n_corrected > 0:
        max_neg = float(df_out.loc[neg_mask, col].min())
        df_out.loc[neg_mask, col] = 0.0
        logger.info(
            "Clamped %d tiny negative precipitation values to 0 (min was %.2e mm).",
            n_corrected,
            max_neg,
        )

    return df_out, n_corrected


def main(config_path: str = "configs/data.yaml") -> None:
    """Run the full preprocessing pipeline."""
    setup_logging("INFO")
    config = load_config(config_path)

    feature_order = config["feature_order"]
    start = config["time_range"]["start"]
    end = config["time_range"]["end"]

    split_cfg = config["split"]
    train_end = split_cfg["train"]["end"]
    val_end = split_cfg["validation"]["end"]
    scaler_type = config["scaler_type"]
    input_window = config["input_window"]

    logger.info("=== ClimateTwin Preprocessing Pipeline ===")
    logger.info("Features: %s", feature_order)
    logger.info("Period: %s to %s", start, end)
    logger.info("Split: train→%s, val→%s", train_end, val_end)
    logger.info("Scaler: %s (train-only fit)", scaler_type)
    logger.info("Input window: %d", input_window)

    # --- Step 1: Load raw data ---
    df = load_and_merge_netcdf(EXTRACTED_DIR, feature_order, start, end)

    # --- Step 2: Validate timestamps ---
    ts_report = validate_timestamps(df, expected_interval="1h")

    # --- Step 3: Remove duplicates ---
    df = remove_duplicates(df)

    # --- Step 4: Handle missing values ---
    df = handle_missing_values(df, strategy="drop", feature_cols=feature_order)

    # --- Step 5: Convert units ---
    df, unit_log = convert_units(df)

    # --- Step 6: Fix precipitation negatives ---
    df, n_precip_corrected = fix_precipitation_negatives(df)

    # --- Step 7: Chronological split ---
    train_df, val_df, test_df = chronological_split(
        df, train_end=train_end, val_end=val_end
    )

    # --- Step 8: Save unscaled splits ---
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    logger.info(
        "Saved unscaled splits: train=%d, val=%d, test=%d",
        len(train_df), len(val_df), len(test_df),
    )

    # --- Step 9: Fit scaler on training data ONLY ---
    scaler = fit_scaler(train_df, scaler_type, feature_order)

    # --- Step 10: Transform all splits ---
    train_scaled = transform_data(train_df, scaler, feature_order)
    val_scaled = transform_data(val_df, scaler, feature_order)
    test_scaled = transform_data(test_df, scaler, feature_order)

    # --- Step 11: Save scaled splits ---
    train_scaled.to_csv(PROCESSED_DIR / "train_scaled.csv", index=False)
    val_scaled.to_csv(PROCESSED_DIR / "val_scaled.csv", index=False)
    test_scaled.to_csv(PROCESSED_DIR / "test_scaled.csv", index=False)
    logger.info("Saved scaled splits.")

    # --- Step 12: Save scaler ---
    scaler_meta = {
        "scaler_type": scaler_type,
        "feature_order": feature_order,
        "fitting_period": f"{split_cfg['train']['start']} to {train_end}",
        "normalization_policy": "train_only_target_location",
        "converted_units": config.get("converted_units", {}),
        "n_train_samples": len(train_df),
    }
    save_scaler(scaler, feature_order, scaler_meta, PROCESSED_DIR / "scaler.joblib")

    # --- Step 13: Compute sequence counts (for reporting, not saving sequences yet) ---
    n_seq_train = max(0, len(train_scaled) - input_window)
    n_seq_val = max(0, len(val_scaled) - input_window)
    n_seq_test = max(0, len(test_scaled) - input_window)

    # --- Step 14: Generate preprocessing report ---
    report = {
        "report_type": "preprocessing",
        "report_date": datetime.now(UTC).isoformat(),
        "source": config["source"],
        "target_location": config["target_location"],
        "feature_order": feature_order,
        "n_features": len(feature_order),
        "n_targets": len(feature_order),
        "input_window": input_window,
        "unit_conversions": unit_log,
        "precipitation_negatives_corrected": n_precip_corrected,
        "timestamp_validation": ts_report,
        "split_summary": {
            "train": {
                "rows": len(train_df),
                "start": str(train_df["timestamp"].min()),
                "end": str(train_df["timestamp"].max()),
                "sequences": n_seq_train,
            },
            "validation": {
                "rows": len(val_df),
                "start": str(val_df["timestamp"].min()),
                "end": str(val_df["timestamp"].max()),
                "sequences": n_seq_val,
            },
            "test": {
                "rows": len(test_df),
                "start": str(test_df["timestamp"].min()),
                "end": str(test_df["timestamp"].max()),
                "sequences": n_seq_test,
            },
        },
        "total_rows": len(df),
        "total_sequences": n_seq_train + n_seq_val + n_seq_test,
        "scaler_type": scaler_type,
        "scaler_path": "data/processed/scaler.joblib",
        "scaler_means": {
            feat: float(scaler.mean_[i]) for i, feat in enumerate(feature_order)
        },
        "scaler_scales": {
            feat: float(scaler.scale_[i]) for i, feat in enumerate(feature_order)
        },
        "artifacts": [
            "data/processed/train.csv",
            "data/processed/val.csv",
            "data/processed/test.csv",
            "data/processed/train_scaled.csv",
            "data/processed/val_scaled.csv",
            "data/processed/test_scaled.csv",
            "data/processed/scaler.joblib",
        ],
        "missing_value_strategy": "drop",
        "derived_features": [],
    }

    report_path = METADATA_DIR / "preprocessing_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Preprocessing report: %s", report_path)

    # --- Summary ---
    logger.info("=== Preprocessing Complete ===")
    logger.info("Train: %d rows, %d sequences", len(train_df), n_seq_train)
    logger.info("Val:   %d rows, %d sequences", len(val_df), n_seq_val)
    logger.info("Test:  %d rows, %d sequences", len(test_df), n_seq_test)
    logger.info("Scaler: %s fitted on %d training samples", scaler_type, len(train_df))
    logger.info("Precipitation negatives corrected: %d", n_precip_corrected)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess ClimateTwin data")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    main(args.config)
