"""Persistence baseline model for ClimateTwin.

Prediction rule:
    prediction(t+1) = observed_state(t)

For all six target variables: t2m, d2m, sp, tp, u10, v10.

This model has no learned parameters. It serves as the minimum-skill
reference against which all learned models are compared.

Evaluation alignment
--------------------
When evaluating alongside learned models that use ``input_window=24``,
persistence must be evaluated on exactly the same target timestamps.
The first evaluated target corresponds to the 25th row of the test split
(index 24), with the prediction being the 24th row (index 23).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def persistence_predict(
    data: np.ndarray | pd.DataFrame,
    feature_names: list[str],
    target_names: list[str] | None = None,
    input_window: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Generate persistence predictions aligned with sequence-model evaluation.

    For each valid target index ``i`` (where ``i >= input_window``):
        prediction[i] = data[i - 1]   (the immediately preceding observation)
        target[i]     = data[i]        (the actual observation)

    This produces predictions for exactly ``len(data) - input_window``
    samples, matching ``ClimateSequenceDataset`` output count.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        The test split data.  If DataFrame, must contain ``feature_names``
        columns and optionally a ``"timestamp"`` column.
    feature_names : list[str]
        Ordered feature column names (e.g. ``["t2m", "d2m", ...]``).
    target_names : list[str] or None
        Target variable names.  If ``None``, same as ``feature_names``
        (n_targets == n_features).
    input_window : int
        The input window length used by learned models.  Persistence
        skips the first ``input_window`` rows to align evaluation with
        sequence-based models.

    Returns
    -------
    predictions : np.ndarray
        Shape ``(n_eval, n_targets)`` — persistence predictions.
    targets : np.ndarray
        Shape ``(n_eval, n_targets)`` — ground-truth values.
    timestamps : np.ndarray or None
        Shape ``(n_eval,)`` — target timestamps, or ``None`` if not
        available.
    """
    if target_names is None:
        target_names = list(feature_names)

    # Extract arrays
    if isinstance(data, pd.DataFrame):
        timestamps_available = "timestamp" in data.columns
        ts_array = data["timestamp"].values if timestamps_available else None
        values = data[feature_names].values
    else:
        timestamps_available = False
        ts_array = None
        values = data

    n_total = len(values)
    if input_window >= n_total:
        raise ValueError(
            f"input_window ({input_window}) >= data length ({n_total}). "
            "Cannot produce any evaluation samples."
        )

    # Determine target column indices
    target_indices = [feature_names.index(t) for t in target_names]

    n_eval = n_total - input_window

    # Predictions: row[i-1] for target at row[i], where i starts at input_window
    predictions = values[input_window - 1 : n_total - 1, :][:, target_indices]
    targets = values[input_window:, :][:, target_indices]
    timestamps = ts_array[input_window:] if ts_array is not None else None

    assert predictions.shape == (n_eval, len(target_names)), (
        f"Expected predictions shape ({n_eval}, {len(target_names)}), "
        f"got {predictions.shape}"
    )
    assert targets.shape == predictions.shape

    logger.info(
        "Persistence predictions: %d samples, %d targets, "
        "first_target_idx=%d, last_target_idx=%d",
        n_eval,
        len(target_names),
        input_window,
        n_total - 1,
    )

    return predictions, targets, timestamps
