"""Data cleaning, splitting, and normalization for ClimateTwin.

This module provides reusable preprocessing functions that are called by
``scripts/preprocess.py`` and tested by ``tests/test_preprocessing.py``.

All functions operate on pandas DataFrames with a ``timestamp`` column
(or DatetimeIndex) and numeric climate-variable columns.

Scientific constraints
----------------------
- Chronological train → validation → test ordering enforced.
- Scalers fitted on training data only; applied to val/test.
- No future-data leakage in any operation.
- Source files are never modified.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task 2.1 — Cleaning utilities
# ---------------------------------------------------------------------------


def remove_duplicates(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    keep: str = "first",
) -> pd.DataFrame:
    """Remove duplicate timestamps, keeping the specified occurrence.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a timestamp column.
    timestamp_col : str
        Name of the timestamp column.
    keep : str
        Which duplicate to keep: ``"first"`` or ``"last"``.

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates removed.
    """
    n_before = len(df)
    df_out = df.drop_duplicates(subset=[timestamp_col], keep=keep).copy()
    n_removed = n_before - len(df_out)
    if n_removed > 0:
        logger.warning("Removed %d duplicate timestamps (keep=%s).", n_removed, keep)
    else:
        logger.info("No duplicate timestamps found.")
    return df_out


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "drop",
    limit: int | None = None,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Handle missing values in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    strategy : str
        One of ``"drop"`` (remove rows with any NaN), ``"ffill"`` (forward
        fill), ``"linear"`` (linear interpolation bounded by *limit*).
    limit : int or None
        Maximum consecutive NaN gap to interpolate/fill.  Gaps larger than
        this remain NaN when using ``"ffill"`` or ``"linear"``.
    feature_cols : list[str] or None
        Columns to check/fill.  If ``None``, uses all numeric columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled.

    Raises
    ------
    ValueError
        If *strategy* is not recognised.

    Notes
    -----
    This function never uses future-period information for imputation.
    Forward-fill by definition only propagates past values.
    Linear interpolation is bounded by *limit* to avoid spanning large gaps.
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    n_missing_before = int(df[feature_cols].isna().sum().sum())

    if strategy == "drop":
        df_out = df.dropna(subset=feature_cols).copy()
    elif strategy == "ffill":
        df_out = df.copy()
        df_out[feature_cols] = df_out[feature_cols].ffill(limit=limit)
    elif strategy == "linear":
        df_out = df.copy()
        df_out[feature_cols] = df_out[feature_cols].interpolate(
            method="linear", limit=limit, limit_direction="forward"
        )
    else:
        raise ValueError(
            f"Unknown missing-value strategy '{strategy}'. "
            "Expected one of: 'drop', 'ffill', 'linear'."
        )

    n_missing_after = int(df_out[feature_cols].isna().sum().sum())
    logger.info(
        "Missing values: before=%d, after=%d (strategy=%s, limit=%s).",
        n_missing_before,
        n_missing_after,
        strategy,
        limit,
    )
    return df_out


def validate_units(
    df: pd.DataFrame,
    expected_units: dict[str, str],
) -> dict[str, str]:
    """Validate that expected columns exist and log their expected units.

    This function does NOT convert units — it records the expected units
    and checks column presence.  Actual conversion is a separate step.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    expected_units : dict[str, str]
        Mapping of column name → expected unit string.

    Returns
    -------
    dict[str, str]
        The validated units mapping (same as input if all columns exist).

    Raises
    ------
    KeyError
        If any expected column is missing from the DataFrame.
    """
    missing = [col for col in expected_units if col not in df.columns]
    if missing:
        raise KeyError(f"Expected columns not found in DataFrame: {missing}")

    for col, unit in expected_units.items():
        logger.info("Column '%s': expected unit = %s", col, unit)

    return expected_units


def validate_timestamps(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    expected_interval: str | None = None,
) -> dict[str, Any]:
    """Validate timestamp column for ordering, gaps, and regularity.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a timestamp column.
    timestamp_col : str
        Name of the timestamp column.
    expected_interval : str or None
        Expected pandas frequency string (e.g. ``"1h"``).  If ``None``,
        the interval is inferred from the most common difference.

    Returns
    -------
    dict[str, Any]
        Validation results including detected interval, gap count,
        duplicate count, and whether the series is sorted.

    Raises
    ------
    ValueError
        If the timestamp column is not present or not datetime-like.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame.")

    ts = pd.DatetimeIndex(df[timestamp_col])

    is_sorted = bool(ts.is_monotonic_increasing)
    n_duplicates = int(ts.duplicated().sum())

    diffs = ts.diff()[1:]
    if len(diffs) == 0:
        detected_interval = None
        n_irregular = 0
    else:
        most_common = diffs.value_counts().index[0]
        detected_interval = str(most_common)
        if expected_interval is not None:
            expected_td = pd.Timedelta(expected_interval)
            n_irregular = int((diffs != expected_td).sum())
        else:
            n_irregular = int((diffs != most_common).sum())

    # Missing timestamps (gaps)
    if len(ts) > 1 and detected_interval is not None:
        freq = expected_interval or detected_interval
        try:
            expected_range = pd.date_range(ts.min(), ts.max(), freq=freq)
            n_missing = len(expected_range.difference(ts))
        except ValueError:
            n_missing = -1  # cannot compute
    else:
        n_missing = 0

    result = {
        "is_sorted": is_sorted,
        "n_duplicates": n_duplicates,
        "n_records": len(ts),
        "first_timestamp": str(ts.min()) if len(ts) > 0 else None,
        "last_timestamp": str(ts.max()) if len(ts) > 0 else None,
        "detected_interval": detected_interval,
        "expected_interval": expected_interval,
        "n_irregular_intervals": n_irregular,
        "n_missing_timestamps": n_missing,
    }

    if not is_sorted:
        logger.warning("Timestamps are NOT sorted chronologically.")
    if n_duplicates > 0:
        logger.warning("Found %d duplicate timestamps.", n_duplicates)
    if n_irregular > 0:
        logger.warning("Found %d irregular intervals.", n_irregular)
    if n_missing > 0:
        logger.warning("Found %d missing timestamps (gaps).", n_missing)

    logger.info(
        "Timestamp validation: records=%d, sorted=%s, duplicates=%d, "
        "irregular=%d, missing=%d, interval=%s",
        len(ts),
        is_sorted,
        n_duplicates,
        n_irregular,
        n_missing,
        detected_interval,
    )
    return result


# ---------------------------------------------------------------------------
# Task 2.2 — Chronological split
# ---------------------------------------------------------------------------


def chronological_split(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-series DataFrame into chronological train/val/test.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame sorted by *timestamp_col*.
    train_end : str
        Last date (inclusive) for training, e.g. ``"2022-12-31"``.
    val_end : str
        Last date (inclusive) for validation, e.g. ``"2023-12-31"``.
    timestamp_col : str
        Name of the timestamp column.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train_df, val_df, test_df)``

    Raises
    ------
    ValueError
        If ordering constraints are violated or any split is empty.
    """
    ts = pd.DatetimeIndex(df[timestamp_col])
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    if train_end_ts >= val_end_ts:
        raise ValueError(
            f"train_end ({train_end}) must be before val_end ({val_end})."
        )

    train_df = df[ts <= train_end_ts].copy()
    val_df = df[(ts > train_end_ts) & (ts <= val_end_ts)].copy()
    test_df = df[ts > val_end_ts].copy()

    # --- Strict ordering assertions ---
    train_ts = pd.DatetimeIndex(train_df[timestamp_col])
    val_ts = pd.DatetimeIndex(val_df[timestamp_col])
    test_ts = pd.DatetimeIndex(test_df[timestamp_col])

    if len(train_df) == 0:
        raise ValueError("Training split is empty.")
    if len(val_df) == 0:
        raise ValueError("Validation split is empty.")
    if len(test_df) == 0:
        raise ValueError("Test split is empty.")

    assert train_ts.max() < val_ts.min(), (
        f"Train max ({train_ts.max()}) must be < val min ({val_ts.min()})"
    )
    assert val_ts.max() < test_ts.min(), (
        f"Val max ({val_ts.max()}) must be < test min ({test_ts.min()})"
    )

    # No index overlap
    assert len(set(train_df.index) & set(val_df.index)) == 0, "Train/val index overlap"
    assert len(set(val_df.index) & set(test_df.index)) == 0, "Val/test index overlap"
    assert len(set(train_df.index) & set(test_df.index)) == 0, "Train/test index overlap"

    logger.info(
        "Chronological split: train=%d (%s to %s), val=%d (%s to %s), test=%d (%s to %s)",
        len(train_df),
        train_ts.min(),
        train_ts.max(),
        len(val_df),
        val_ts.min(),
        val_ts.max(),
        len(test_df),
        test_ts.min(),
        test_ts.max(),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Task 2.3 — Normalization infrastructure
# ---------------------------------------------------------------------------

_SCALER_CLASSES = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
}


def fit_scaler(
    train_df: pd.DataFrame,
    scaler_type: str,
    feature_names: list[str],
) -> StandardScaler | MinMaxScaler:
    """Fit a scaler on training data only.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split DataFrame.
    scaler_type : str
        One of ``"StandardScaler"`` or ``"MinMaxScaler"``.
    feature_names : list[str]
        Ordered list of feature column names to scale.

    Returns
    -------
    StandardScaler or MinMaxScaler
        Fitted scaler instance.

    Raises
    ------
    ValueError
        If *scaler_type* is not recognised.
    KeyError
        If any *feature_names* column is missing.
    """
    if scaler_type not in _SCALER_CLASSES:
        raise ValueError(
            f"Unknown scaler_type '{scaler_type}'. Expected one of {list(_SCALER_CLASSES)}."
        )

    missing = [c for c in feature_names if c not in train_df.columns]
    if missing:
        raise KeyError(f"Feature columns not found in training data: {missing}")

    scaler = _SCALER_CLASSES[scaler_type]()
    scaler.fit(train_df[feature_names].values)

    logger.info(
        "Fitted %s on %d training samples, %d features.",
        scaler_type,
        len(train_df),
        len(feature_names),
    )
    return scaler


def transform_data(
    df: pd.DataFrame,
    scaler: StandardScaler | MinMaxScaler,
    feature_names: list[str],
) -> pd.DataFrame:
    """Apply a fitted scaler to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to transform.
    scaler : StandardScaler or MinMaxScaler
        Already-fitted scaler (from :func:`fit_scaler`).
    feature_names : list[str]
        Ordered list of feature columns (must match fit order).

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame (copy — original is not modified).
    """
    df_out = df.copy()
    df_out[feature_names] = scaler.transform(df[feature_names].values)
    return df_out


def save_scaler(
    scaler: StandardScaler | MinMaxScaler,
    feature_names: list[str],
    metadata: dict[str, Any],
    path: str | Path,
) -> Path:
    """Persist a fitted scaler with its feature names and metadata.

    Parameters
    ----------
    scaler : StandardScaler or MinMaxScaler
        Fitted scaler.
    feature_names : list[str]
        Ordered feature names used during fitting.
    metadata : dict
        Additional metadata (e.g. scaler_type, fitting period, units).
    path : str or Path
        Destination file path (``.joblib``).

    Returns
    -------
    Path
        The resolved path that was written.
    """
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "scaler": scaler,
        "feature_names": feature_names,
        "metadata": metadata,
    }
    joblib.dump(artifact, filepath)
    logger.info("Saved scaler to %s", filepath)
    return filepath


def load_scaler(
    path: str | Path,
) -> tuple[StandardScaler | MinMaxScaler, list[str], dict[str, Any]]:
    """Load a persisted scaler artifact.

    Parameters
    ----------
    path : str or Path
        Path to the saved ``.joblib`` file.

    Returns
    -------
    tuple
        ``(scaler, feature_names, metadata)``

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Scaler file not found: {filepath}")

    artifact = joblib.load(filepath)
    scaler = artifact["scaler"]
    feature_names = artifact["feature_names"]
    metadata = artifact.get("metadata", {})

    logger.info("Loaded scaler from %s (features=%s)", filepath, feature_names)
    return scaler, feature_names, metadata
