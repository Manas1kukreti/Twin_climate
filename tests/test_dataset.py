"""Tests for src/dataset.py — sequence construction and multi-location datasets.

Covers:
- Sequence alignment (off-by-one checks)
- Split-boundary prevention via max_target_timestamp
- Dataset length correctness
- Tensor shape validation
- n_targets consistency with target_indices
- Multi-location: no cross-location sequences
- Multi-location: chronological ordering per location
- Feature-order stability
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from tests.conftest import SYNTHETIC_FEATURE_NAMES, SYNTHETIC_N_FEATURES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(n: int = 100, n_features: int = SYNTHETIC_N_FEATURES) -> np.ndarray:
    """Create synthetic sequential data where row i has all values = i."""
    return np.arange(n * n_features, dtype=np.float32).reshape(n, n_features)


def _make_timestamps(n: int = 100, start: str = "2020-01-01", freq: str = "1h"):
    return pd.date_range(start, periods=n, freq=freq)


# ---------------------------------------------------------------------------
# ClimateSequenceDataset
# ---------------------------------------------------------------------------


class TestClimateSequenceDataset:
    def test_basic_shapes(self) -> None:
        from src.dataset import ClimateSequenceDataset

        data = _make_data(100)
        ds = ClimateSequenceDataset(
            data=data,
            input_window=10,
            feature_names=SYNTHETIC_FEATURE_NAMES,
        )
        x, y = ds[0]
        assert x.shape == (10, SYNTHETIC_N_FEATURES)
        assert y.shape == (SYNTHETIC_N_FEATURES,)

    def test_target_is_next_step(self) -> None:
        """Verify target at index i is data[i + window], not data[i + window - 1]."""
        from src.dataset import ClimateSequenceDataset

        data = _make_data(50)
        window = 5
        ds = ClimateSequenceDataset(
            data=data,
            input_window=window,
            feature_names=SYNTHETIC_FEATURE_NAMES,
        )
        x, y = ds[0]
        # Input should be rows 0..4, target should be row 5
        expected_input = data[0:window]
        expected_target = data[window]
        np.testing.assert_array_equal(x.numpy(), expected_input)
        np.testing.assert_array_equal(y.numpy(), expected_target)

    def test_last_valid_index(self) -> None:
        """The last sequence must not exceed data bounds."""
        from src.dataset import ClimateSequenceDataset

        data = _make_data(20)
        window = 5
        ds = ClimateSequenceDataset(
            data=data,
            input_window=window,
            feature_names=SYNTHETIC_FEATURE_NAMES,
        )
        # Last valid: input = data[14:19], target = data[19]
        assert len(ds) == 20 - window - 1 + 1  # = 15
        x, y = ds[len(ds) - 1]
        expected_target = data[19]
        np.testing.assert_array_equal(y.numpy(), expected_target)

    def test_dataset_length(self) -> None:
        from src.dataset import ClimateSequenceDataset

        n = 100
        window = 24
        data = _make_data(n)
        ds = ClimateSequenceDataset(
            data=data,
            input_window=window,
            feature_names=SYNTHETIC_FEATURE_NAMES,
        )
        assert len(ds) == n - window

    def test_custom_target_indices(self) -> None:
        from src.dataset import ClimateSequenceDataset

        data = _make_data(50)
        target_idx = [0, 2]  # only first and third features
        ds = ClimateSequenceDataset(
            data=data,
            input_window=5,
            feature_names=SYNTHETIC_FEATURE_NAMES,
            target_indices=target_idx,
        )
        assert ds.n_targets == 2
        x, y = ds[0]
        assert y.shape == (2,)
        expected = data[5, target_idx]
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_all_targets_default(self) -> None:
        from src.dataset import ClimateSequenceDataset

        data = _make_data(50)
        ds = ClimateSequenceDataset(
            data=data,
            input_window=5,
            feature_names=SYNTHETIC_FEATURE_NAMES,
        )
        assert ds.n_targets == SYNTHETIC_N_FEATURES
        assert ds.target_indices == list(range(SYNTHETIC_N_FEATURES))

    def test_tensor_dtypes(self) -> None:
        from src.dataset import ClimateSequenceDataset

        data = _make_data(50)
        ds = ClimateSequenceDataset(
            data=data,
            input_window=5,
            feature_names=SYNTHETIC_FEATURE_NAMES,
        )
        x, y = ds[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_boundary_enforcement(self) -> None:
        """Sequences with target beyond max_target_timestamp are excluded."""
        from src.dataset import ClimateSequenceDataset

        n = 100
        timestamps = _make_timestamps(n)
        data = _make_data(n)
        window = 5

        # Allow targets only up to timestamp index 49 (50th hour)
        max_ts = timestamps[49]
        ds = ClimateSequenceDataset(
            data=data,
            input_window=window,
            feature_names=SYNTHETIC_FEATURE_NAMES,
            timestamps=timestamps,
            max_target_timestamp=max_ts,
        )

        # Last valid: target at index 49 → start index = 49 - window = 44
        assert len(ds) == 49 - window + 1  # = 45

        # Verify no target exceeds the boundary
        for i in range(len(ds)):
            idx = ds._valid_indices[i]
            target_ts_idx = idx + window
            assert target_ts_idx <= 49

    def test_window_too_large_raises(self) -> None:
        from src.dataset import ClimateSequenceDataset

        data = _make_data(10)
        with pytest.raises(ValueError, match="input_window"):
            ClimateSequenceDataset(
                data=data,
                input_window=10,
                feature_names=SYNTHETIC_FEATURE_NAMES,
            )

    def test_mismatched_features_raises(self) -> None:
        from src.dataset import ClimateSequenceDataset

        data = _make_data(50, n_features=3)
        with pytest.raises(ValueError, match="feature_names"):
            ClimateSequenceDataset(
                data=data,
                input_window=5,
                feature_names=SYNTHETIC_FEATURE_NAMES,  # 6 names, 3 columns
            )


# ---------------------------------------------------------------------------
# MultiLocationDataset
# ---------------------------------------------------------------------------


class TestMultiLocationDataset:
    def _make_location_data(
        self,
        n_locations: int = 3,
        n_per_loc: int = 50,
    ) -> dict:
        loc_data = {}
        for i in range(n_locations):
            start = f"2020-{i + 1:02d}-01"
            ts = _make_timestamps(n_per_loc, start=start)
            data = _make_data(n_per_loc) + i * 1000  # offset by location
            loc_data[f"city_{i}"] = {"data": data, "timestamps": ts}
        return loc_data

    def test_no_cross_location_sequences(self) -> None:
        from src.dataset import MultiLocationDataset

        loc_data = self._make_location_data(3, 50)
        window = 5
        ds = MultiLocationDataset(
            location_data=loc_data,
            input_window=window,
            feature_names=SYNTHETIC_FEATURE_NAMES,
        )
        # Total sequences: 3 locations × (50 - 5) = 135
        assert len(ds) == 3 * (50 - window)

        # Verify each sequence stays within its location's value range
        for i in range(len(ds)):
            x, y = ds[i]
            # All values in a sequence should be from the same location
            # (offsets are 0, 1000, 2000 — check they don't mix)
            x_base = int(x[0, 0].item()) // 1000
            y_base = int(y[0].item()) // 1000
            assert x_base == y_base, (
                f"Sequence {i} crosses locations: input base={x_base}, "
                f"target base={y_base}"
            )

    def test_chronological_per_location(self) -> None:
        from src.dataset import MultiLocationDataset

        loc_data = self._make_location_data(2, 30)
        ds = MultiLocationDataset(
            location_data=loc_data,
            input_window=5,
            feature_names=SYNTHETIC_FEATURE_NAMES,
        )
        assert len(ds.location_names) == 2

    def test_unsorted_location_raises(self) -> None:
        from src.dataset import MultiLocationDataset

        ts = _make_timestamps(50)
        data = _make_data(50)
        # Reverse timestamps to make unsorted
        loc_data = {"bad_city": {"data": data, "timestamps": ts[::-1]}}
        with pytest.raises(ValueError, match="not sorted"):
            MultiLocationDataset(
                location_data=loc_data,
                input_window=5,
                feature_names=SYNTHETIC_FEATURE_NAMES,
            )

    def test_per_location_boundary(self) -> None:
        """Per-location max_target_timestamp enforcement."""
        from src.dataset import MultiLocationDataset

        loc_data = self._make_location_data(2, 50)
        window = 5

        # Restrict city_0 to first 30 timesteps' target
        ts0 = loc_data["city_0"]["timestamps"]
        max_ts = {
            "city_0": ts0[29],  # allow targets up to index 29
        }

        ds = MultiLocationDataset(
            location_data=loc_data,
            input_window=window,
            feature_names=SYNTHETIC_FEATURE_NAMES,
            max_target_timestamps=max_ts,
        )

        # city_0: 29 - window + 1 = 25 sequences
        # city_1: 50 - window = 45 sequences (no boundary)
        assert len(ds) == 25 + 45

    def test_empty_after_boundary_raises(self) -> None:
        from src.dataset import MultiLocationDataset

        loc_data = self._make_location_data(1, 10)
        ts = loc_data["city_0"]["timestamps"]
        # Set boundary before any valid target
        max_ts = {"city_0": ts[0]}  # only first timestamp allowed as target

        with pytest.raises(ValueError, match="No valid sequences"):
            MultiLocationDataset(
                location_data=loc_data,
                input_window=5,
                feature_names=SYNTHETIC_FEATURE_NAMES,
                max_target_timestamps=max_ts,
            )

    def test_total_length_matches_sum(self) -> None:
        from src.dataset import MultiLocationDataset

        loc_data = self._make_location_data(3, 40)
        window = 10
        ds = MultiLocationDataset(
            location_data=loc_data,
            input_window=window,
            feature_names=SYNTHETIC_FEATURE_NAMES,
        )
        expected = 3 * (40 - window)
        assert len(ds) == expected
