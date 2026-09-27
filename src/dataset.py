"""PyTorch Dataset classes for temporal sequence construction.

Converts cleaned, scaled climate DataFrames into ``(input, target)`` tensor
pairs suitable for LSTM and Transformer training.

Scientific constraints
----------------------
- Sequence windows must not cross train/validation/test boundaries.
- Target indices are configurable (§9.13 unresolved — infrastructure only).
- Multi-location datasets must not create sequences spanning different stations.
- Feature ordering is stable and explicitly tracked.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ClimateSequenceDataset(Dataset):
    """A PyTorch Dataset producing ``(input_window, target)`` pairs.

    For one-step prediction:

    .. code-block:: text

        input:  data[i : i + input_window]      shape: (input_window, n_features)
        target: data[i + input_window]           shape: (n_targets,)

    Parameters
    ----------
    data : np.ndarray
        2-D array of shape ``(n_timesteps, n_features)`` — already scaled.
    input_window : int
        Number of timesteps in each input sequence.
    feature_names : list[str]
        Ordered feature column names.
    target_indices : list[int] or None
        Column indices of the target variables.  If ``None``, all columns
        are targets (``n_targets == n_features``).
    timestamps : np.ndarray or pd.DatetimeIndex or None
        Timestamp array aligned with *data* rows.  Used for boundary
        checking but not returned in ``__getitem__``.
    max_target_timestamp : pd.Timestamp or str or None
        If provided, no target row may have a timestamp beyond this value.
        Used to prevent sequences from crossing split boundaries.

    Attributes
    ----------
    n_features : int
    n_targets : int
    input_window : int
    feature_names : list[str]
    target_indices : list[int]
    """

    def __init__(
        self,
        data: np.ndarray,
        input_window: int,
        feature_names: list[str],
        target_indices: list[int] | None = None,
        timestamps: np.ndarray | pd.DatetimeIndex | None = None,
        max_target_timestamp: pd.Timestamp | str | None = None,
    ) -> None:
        if data.ndim != 2:
            raise ValueError(f"data must be 2-D, got shape {data.shape}")
        if data.shape[1] != len(feature_names):
            raise ValueError(
                f"data has {data.shape[1]} columns but {len(feature_names)} "
                f"feature_names were provided."
            )
        if input_window < 1:
            raise ValueError(f"input_window must be >= 1, got {input_window}")
        if input_window >= len(data):
            raise ValueError(
                f"input_window ({input_window}) must be < data length ({len(data)})"
            )

        self.data = data.astype(np.float32)
        self.input_window = input_window
        self.feature_names = list(feature_names)
        self.n_features = len(feature_names)

        if target_indices is None:
            self.target_indices = list(range(self.n_features))
        else:
            self.target_indices = list(target_indices)
        self.n_targets = len(self.target_indices)

        # Compute valid indices
        # For index i: input = data[i : i+window], target = data[i+window]
        max_start = len(data) - input_window - 1  # last valid start index
        self._valid_indices = list(range(max_start + 1))

        # Boundary enforcement
        if timestamps is not None and max_target_timestamp is not None:
            ts = pd.DatetimeIndex(timestamps)
            max_ts = pd.Timestamp(max_target_timestamp)
            # Target row is at index i + input_window
            self._valid_indices = [
                i
                for i in self._valid_indices
                if ts[i + input_window] <= max_ts
            ]

        if len(self._valid_indices) == 0:
            raise ValueError(
                "No valid sequences after boundary enforcement. "
                "Check input_window, data length, and max_target_timestamp."
            )

        logger.info(
            "ClimateSequenceDataset: %d sequences, window=%d, "
            "n_features=%d, n_targets=%d",
            len(self._valid_indices),
            input_window,
            self.n_features,
            self.n_targets,
        )

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(input_tensor, target_tensor)``.

        Returns
        -------
        input_tensor : Tensor
            Shape ``(input_window, n_features)``.
        target_tensor : Tensor
            Shape ``(n_targets,)``.
        """
        i = self._valid_indices[idx]
        x = self.data[i : i + self.input_window]  # (window, n_features)
        y = self.data[i + self.input_window, self.target_indices]  # (n_targets,)
        return torch.from_numpy(x), torch.from_numpy(y)


class MultiLocationDataset(Dataset):
    """A PyTorch Dataset for multi-location pretraining.

    Concatenates per-location sequences while ensuring no sequence spans
    multiple locations.

    Parameters
    ----------
    location_data : dict[str, dict[str, Any]]
        Mapping of ``location_name`` → dict with keys:

        - ``"data"``: np.ndarray of shape ``(n_timesteps, n_features)``
        - ``"timestamps"``: np.ndarray or pd.DatetimeIndex

    input_window : int
        Number of timesteps per input sequence.
    feature_names : list[str]
        Ordered feature names (must be consistent across locations).
    target_indices : list[int] or None
        Target column indices.  ``None`` → all features are targets.
    max_target_timestamps : dict[str, str | pd.Timestamp] | None
        Optional per-location boundary timestamps.

    Notes
    -----
    Each location's data must be pre-sorted chronologically.
    Sequences are constructed independently per location and then
    concatenated.  No sequence's input or target crosses location
    boundaries.
    """

    def __init__(
        self,
        location_data: dict[str, dict[str, Any]],
        input_window: int,
        feature_names: list[str],
        target_indices: list[int] | None = None,
        max_target_timestamps: dict[str, str | pd.Timestamp] | None = None,
    ) -> None:
        self.input_window = input_window
        self.feature_names = list(feature_names)
        self.n_features = len(feature_names)

        if target_indices is None:
            self.target_indices = list(range(self.n_features))
        else:
            self.target_indices = list(target_indices)
        self.n_targets = len(self.target_indices)

        # Build per-location datasets
        self._datasets: list[ClimateSequenceDataset] = []
        self._location_names: list[str] = []
        self._cumulative_lengths: list[int] = []
        cumulative = 0

        for loc_name in sorted(location_data.keys()):
            loc = location_data[loc_name]
            data = loc["data"]
            timestamps = loc.get("timestamps")

            # Validate chronological ordering within location
            if timestamps is not None:
                ts = pd.DatetimeIndex(timestamps)
                if not ts.is_monotonic_increasing:
                    raise ValueError(
                        f"Timestamps for location '{loc_name}' are not "
                        "sorted chronologically."
                    )

            max_ts = None
            if max_target_timestamps and loc_name in max_target_timestamps:
                max_ts = max_target_timestamps[loc_name]

            ds = ClimateSequenceDataset(
                data=data,
                input_window=input_window,
                feature_names=feature_names,
                target_indices=target_indices,
                timestamps=timestamps,
                max_target_timestamp=max_ts,
            )
            self._datasets.append(ds)
            self._location_names.append(loc_name)
            cumulative += len(ds)
            self._cumulative_lengths.append(cumulative)

        if cumulative == 0:
            raise ValueError("No valid sequences across any location.")

        logger.info(
            "MultiLocationDataset: %d total sequences from %d locations, "
            "window=%d, n_features=%d, n_targets=%d",
            cumulative,
            len(self._datasets),
            input_window,
            self.n_features,
            self.n_targets,
        )

    def __len__(self) -> int:
        if not self._cumulative_lengths:
            return 0
        return self._cumulative_lengths[-1]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(input_tensor, target_tensor)`` from the correct location."""
        # Binary search for the location
        for i, cum_len in enumerate(self._cumulative_lengths):
            if idx < cum_len:
                local_idx = idx if i == 0 else idx - self._cumulative_lengths[i - 1]
                return self._datasets[i][local_idx]
        raise IndexError(f"Index {idx} out of range [0, {len(self)})")

    @property
    def location_names(self) -> list[str]:
        """Return the sorted list of location names."""
        return list(self._location_names)
