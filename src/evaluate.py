"""Evaluation utilities for ClimateTwin experiments.

Tier 1 — available from Phase 3 onward:
    compute_mae, compute_rmse, inverse_transform_predictions,
    compute_per_variable_metrics, align_predictions_with_ground_truth,
    save_metrics, load_metrics.

All per-variable metrics are computed in original physical units after
inverse transformation.  No mixed-unit aggregate physical metric is
computed as a primary result.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Arrays of the same shape.

    Returns
    -------
    float
        MAE = mean(|y_true - y_pred|)
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Arrays of the same shape.

    Returns
    -------
    float
        RMSE = sqrt(mean((y_true - y_pred)^2))
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------------
# Inverse transform
# ---------------------------------------------------------------------------


def inverse_transform_predictions(
    y_scaled: np.ndarray,
    scaler,
    feature_names: list[str],
) -> np.ndarray:
    """Inverse-transform scaled predictions back to original physical units.

    Parameters
    ----------
    y_scaled : np.ndarray
        Scaled array, shape ``(n_samples, n_features)`` or ``(n_features,)``.
    scaler : fitted sklearn scaler
        The scaler used to transform the data.
    feature_names : list[str]
        Ordered feature names matching the scaler's fit order.

    Returns
    -------
    np.ndarray
        Array in original physical units, same shape as input.
    """
    was_1d = y_scaled.ndim == 1
    if was_1d:
        y_scaled = y_scaled.reshape(1, -1)

    y_original = scaler.inverse_transform(y_scaled)

    if was_1d:
        y_original = y_original.ravel()

    return y_original


# ---------------------------------------------------------------------------
# Per-variable metrics
# ---------------------------------------------------------------------------


def compute_per_variable_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    variable_names: list[str],
) -> dict[str, dict[str, float]]:
    """Compute MAE and RMSE per variable.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth, shape ``(n_samples, n_variables)``.
    y_pred : np.ndarray
        Predictions, shape ``(n_samples, n_variables)``.
    variable_names : list[str]
        Ordered variable names.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{"t2m": {"mae": ..., "rmse": ...}, ...}``
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )
    if y_true.shape[1] != len(variable_names):
        raise ValueError(
            f"y_true has {y_true.shape[1]} columns but "
            f"{len(variable_names)} variable_names provided."
        )

    metrics: dict[str, dict[str, float]] = {}
    for i, var in enumerate(variable_names):
        metrics[var] = {
            "mae": compute_mae(y_true[:, i], y_pred[:, i]),
            "rmse": compute_rmse(y_true[:, i], y_pred[:, i]),
        }

    return metrics


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def align_predictions_with_ground_truth(
    predictions: np.ndarray,
    targets: np.ndarray,
    timestamps: np.ndarray,
) -> dict[str, Any]:
    """Align predictions with ground truth and timestamps.

    Parameters
    ----------
    predictions : np.ndarray
        Shape ``(n_samples, n_variables)``.
    targets : np.ndarray
        Shape ``(n_samples, n_variables)``.
    timestamps : np.ndarray
        Shape ``(n_samples,)`` — timestamps corresponding to targets.

    Returns
    -------
    dict
        ``{"predictions": ..., "targets": ..., "timestamps": ...}``
    """
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions={predictions.shape}, targets={targets.shape}"
        )
    if len(timestamps) != predictions.shape[0]:
        raise ValueError(
            f"Timestamp count ({len(timestamps)}) != sample count ({predictions.shape[0]})"
        )

    return {
        "predictions": predictions,
        "targets": targets,
        "timestamps": timestamps,
    }


# ---------------------------------------------------------------------------
# Metric I/O
# ---------------------------------------------------------------------------


def save_metrics(metrics: dict[str, Any], path: str | Path) -> Path:
    """Save metrics dict to JSON.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary.
    path : str or Path
        Output file path.

    Returns
    -------
    Path
        Resolved path written.
    """
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Saved metrics to %s", filepath)
    return filepath


def load_metrics(path: str | Path) -> dict[str, Any]:
    """Load metrics from JSON.

    Parameters
    ----------
    path : str or Path
        Path to metrics JSON file.

    Returns
    -------
    dict
        Loaded metrics.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Metrics file not found: {filepath}")
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded metrics from %s", filepath)
    return data


# ---------------------------------------------------------------------------
# Baseline comparison loader — reads prior real experiment artifacts
# ---------------------------------------------------------------------------


def _get_feature_order(metrics_doc: dict[str, Any]) -> list[str]:
    """Return the feature/target order field, supporting either key name."""
    if "feature_order" in metrics_doc:
        return list(metrics_doc["feature_order"])
    if "target_order" in metrics_doc:
        return list(metrics_doc["target_order"])
    raise KeyError("Metrics document has neither 'feature_order' nor 'target_order'")


def _get_sample_count(metrics_doc: dict[str, Any]) -> int:
    """Return the evaluated-sample count field, supporting either key name."""
    if "test_sample_count" in metrics_doc:
        return int(metrics_doc["test_sample_count"])
    if "n_evaluated_samples" in metrics_doc:
        return int(metrics_doc["n_evaluated_samples"])
    raise KeyError(
        "Metrics document has neither 'test_sample_count' nor 'n_evaluated_samples'"
    )


def _get_input_window(metrics_doc: dict[str, Any]) -> int | None:
    """Return the declared input_window, checking top level and hyperparameters."""
    if "input_window" in metrics_doc:
        return int(metrics_doc["input_window"])
    if "hyperparameters" in metrics_doc and "input_window" in metrics_doc["hyperparameters"]:
        return int(metrics_doc["hyperparameters"]["input_window"])
    return None


def _get_first_last_timestamp_from_metrics(
    metrics_doc: dict[str, Any],
) -> tuple[str, str] | None:
    """Return (first, last) timestamp strings if present in the metrics doc.

    Supports the ``timestamp_range: {"first": ..., "last": ...}`` schema
    used by the persistence metrics file. Returns ``None`` if not present.
    """
    ts_range = metrics_doc.get("timestamp_range")
    if ts_range and "first" in ts_range and "last" in ts_range:
        return str(ts_range["first"]), str(ts_range["last"])
    return None


def _get_first_last_timestamp_from_predictions(
    predictions_path: str | Path,
) -> tuple[str, str]:
    """Read the first and last timestamp directly from a predictions CSV.

    This is used as a fallback when a metrics JSON does not itself record
    timestamp range metadata, so real timestamps are read from the actual
    persisted prediction artifact rather than fabricated.

    Parameters
    ----------
    predictions_path : str or Path
        Path to a predictions CSV with a ``timestamp`` column.

    Returns
    -------
    tuple[str, str]
        ``(first_timestamp, last_timestamp)`` as strings.

    Raises
    ------
    FileNotFoundError
        If the predictions file does not exist.
    ValueError
        If the predictions file has no ``timestamp`` column or is empty.
    """
    import pandas as pd

    filepath = Path(predictions_path)
    if not filepath.exists():
        raise FileNotFoundError(f"Predictions file not found: {filepath}")

    df = pd.read_csv(filepath)
    if "timestamp" not in df.columns:
        raise ValueError(f"Predictions file has no 'timestamp' column: {filepath}")
    if len(df) == 0:
        raise ValueError(f"Predictions file is empty: {filepath}")

    return str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1])


def load_and_validate_baseline_metrics(
    paths: dict[str, str | Path],
    expected_feature_order: list[str],
    expected_units: dict[str, str],
    expected_sample_count: int = 8760,
    expected_input_window: int = 24,
    expected_first_timestamp: str = "2024-01-02T00:00:00",
    expected_last_timestamp: str = "2024-12-31T23:00:00",
    predictions_paths: dict[str, str | Path] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Load prior baseline metrics artifacts and validate consistency.

    Parameters
    ----------
    paths : dict[str, str | Path]
        Mapping of ``model_name`` → path to that model's metrics JSON
        (e.g. ``{"persistence": "results/metrics/persistence_delhi_metrics.json"}``).
    expected_feature_order : list[str]
        The feature/target order that every baseline must match exactly.
    expected_units : dict[str, str]
        The per-variable units that every baseline must match exactly.
    expected_sample_count : int
        The number of evaluated test samples every baseline must match
        (default 8760, matching input_window=24 on the 8784-row test split).
    expected_input_window : int
        The input window every baseline must have used (default 24).
    expected_first_timestamp : str
        The exact first evaluated target timestamp every baseline must match.
    expected_last_timestamp : str
        The exact last evaluated target timestamp every baseline must match.
    predictions_paths : dict[str, str | Path] or None
        Mapping of ``model_name`` → path to that model's predictions CSV.
        Used as a fallback to read real first/last timestamps when a
        metrics JSON does not itself record timestamp-range metadata.
        Required for any baseline whose metrics file lacks
        ``timestamp_range``.

    Returns
    -------
    dict[str, dict]
        Mapping of ``model_name`` → ``per_variable_metrics`` dict
        (``{"t2m": {"mae": ..., "rmse": ...}, ...}``).

    Raises
    ------
    FileNotFoundError
        If any baseline metrics or (fallback) predictions file does not exist.
    ValueError
        If any baseline fails a consistency check (feature order, units,
        sample count, input window, or first/last timestamp alignment).
    """
    predictions_paths = predictions_paths or {}
    results: dict[str, dict[str, dict[str, float]]] = {}

    for model_name, path in paths.items():
        doc = load_metrics(path)

        # --- Feature/target order check ---
        actual_order = _get_feature_order(doc)
        if actual_order != expected_feature_order:
            raise ValueError(
                f"[{model_name}] feature/target order mismatch: "
                f"expected {expected_feature_order}, got {actual_order}"
            )

        # --- Units check ---
        actual_units = doc.get("units")
        if actual_units is None:
            raise ValueError(f"[{model_name}] metrics document has no 'units' field")
        if actual_units != expected_units:
            raise ValueError(
                f"[{model_name}] units mismatch: "
                f"expected {expected_units}, got {actual_units}"
            )

        # --- Sample count check ---
        actual_count = _get_sample_count(doc)
        if actual_count != expected_sample_count:
            raise ValueError(
                f"[{model_name}] evaluated-sample count mismatch: "
                f"expected {expected_sample_count}, got {actual_count}"
            )

        # --- Input window check ---
        actual_window = _get_input_window(doc)
        if actual_window is None:
            raise ValueError(
                f"[{model_name}] could not find 'input_window' "
                "(checked top level and hyperparameters)"
            )
        if actual_window != expected_input_window:
            raise ValueError(
                f"[{model_name}] input_window mismatch: "
                f"expected {expected_input_window}, got {actual_window}"
            )

        # --- First/last target timestamp check ---
        # Prefer timestamp metadata already recorded in the metrics JSON.
        # If absent, fall back to reading the real persisted predictions
        # artifact rather than fabricating a value.
        ts_pair = _get_first_last_timestamp_from_metrics(doc)
        if ts_pair is None:
            if model_name not in predictions_paths:
                raise ValueError(
                    f"[{model_name}] metrics document has no 'timestamp_range' and "
                    "no predictions_paths entry was provided to verify timestamps "
                    "from the real predictions artifact."
                )
            ts_pair = _get_first_last_timestamp_from_predictions(
                predictions_paths[model_name]
            )

        actual_first, actual_last = ts_pair

        # Compare as parsed datetimes, not raw strings, so that equivalent
        # timestamps differing only in formatting (e.g. trailing
        # ".000000" microseconds) are correctly treated as matching.
        import pandas as pd

        actual_first_ts = pd.Timestamp(actual_first)
        actual_last_ts = pd.Timestamp(actual_last)
        expected_first_ts = pd.Timestamp(expected_first_timestamp)
        expected_last_ts = pd.Timestamp(expected_last_timestamp)

        if actual_first_ts != expected_first_ts:
            raise ValueError(
                f"[{model_name}] first target timestamp mismatch: "
                f"expected {expected_first_timestamp}, got {actual_first}"
            )
        if actual_last_ts != expected_last_ts:
            raise ValueError(
                f"[{model_name}] last target timestamp mismatch: "
                f"expected {expected_last_timestamp}, got {actual_last}"
            )

        results[model_name] = doc["per_variable_metrics"]
        logger.info(
            "Validated baseline '%s': %d samples, window=%d, feature_order=%s, "
            "units=%s, timestamps=[%s, %s]",
            model_name, actual_count, actual_window, actual_order, actual_units,
            actual_first, actual_last,
        )

    return results
