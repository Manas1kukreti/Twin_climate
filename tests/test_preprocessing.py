"""Tests for src/preprocessing.py — cleaning, splitting, normalization.

Covers:
- Duplicate handling
- Missing-value strategies
- Timestamp validation and detection
- Chronological split ordering and overlap
- Scaler train-only fitting
- Feature-order stability
- Inverse-transform roundtrip
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import SYNTHETIC_FEATURE_NAMES

# ---------------------------------------------------------------------------
# Duplicate handling
# ---------------------------------------------------------------------------


class TestRemoveDuplicates:
    def test_no_duplicates(self, synthetic_timeseries_df: pd.DataFrame) -> None:
        from src.preprocessing import remove_duplicates

        result = remove_duplicates(synthetic_timeseries_df)
        assert len(result) == len(synthetic_timeseries_df)

    def test_removes_duplicates(self, synthetic_timeseries_df: pd.DataFrame) -> None:
        from src.preprocessing import remove_duplicates

        # Inject a duplicate
        dup_row = synthetic_timeseries_df.iloc[[5]].copy()
        df = pd.concat([synthetic_timeseries_df, dup_row], ignore_index=True)
        assert len(df) == 201

        result = remove_duplicates(df)
        assert len(result) == 200

    def test_keep_last(self, synthetic_timeseries_df: pd.DataFrame) -> None:
        from src.preprocessing import remove_duplicates

        dup_row = synthetic_timeseries_df.iloc[[5]].copy()
        df = pd.concat([synthetic_timeseries_df, dup_row], ignore_index=True)
        result = remove_duplicates(df, keep="last")
        assert len(result) == 200


# ---------------------------------------------------------------------------
# Missing-value handling
# ---------------------------------------------------------------------------


class TestHandleMissingValues:
    def test_drop_strategy(self, synthetic_timeseries_df: pd.DataFrame) -> None:
        from src.preprocessing import handle_missing_values

        df = synthetic_timeseries_df.copy()
        df.iloc[10, 1] = np.nan  # inject NaN in first feature col
        result = handle_missing_values(df, strategy="drop")
        assert len(result) == 199

    def test_ffill_strategy(self, synthetic_timeseries_df: pd.DataFrame) -> None:
        from src.preprocessing import handle_missing_values

        df = synthetic_timeseries_df.copy()
        df.iloc[10, 1] = np.nan
        result = handle_missing_values(df, strategy="ffill")
        assert len(result) == 200
        assert not pd.isna(result.iloc[10, 1])  # should not be NaN after ffill

    def test_linear_strategy(self, synthetic_timeseries_df: pd.DataFrame) -> None:
        from src.preprocessing import handle_missing_values

        df = synthetic_timeseries_df.copy()
        df.iloc[10, 1] = np.nan
        result = handle_missing_values(df, strategy="linear")
        assert len(result) == 200

    def test_invalid_strategy_raises(self, synthetic_timeseries_df: pd.DataFrame) -> None:
        from src.preprocessing import handle_missing_values

        with pytest.raises(ValueError, match="Unknown missing-value strategy"):
            handle_missing_values(synthetic_timeseries_df, strategy="magic")

    def test_no_missing(self, synthetic_timeseries_df: pd.DataFrame) -> None:
        from src.preprocessing import handle_missing_values

        result = handle_missing_values(synthetic_timeseries_df, strategy="drop")
        assert len(result) == 200


# ---------------------------------------------------------------------------
# Timestamp validation
# ---------------------------------------------------------------------------


class TestValidateTimestamps:
    def test_valid_hourly(self, synthetic_timeseries_df: pd.DataFrame) -> None:
        from src.preprocessing import validate_timestamps

        result = validate_timestamps(
            synthetic_timeseries_df,
            expected_interval="6h",  # synthetic data uses 6h
        )
        assert result["is_sorted"] is True
        assert result["n_duplicates"] == 0
        assert result["n_irregular_intervals"] == 0
        assert result["n_missing_timestamps"] == 0

    def test_detects_unsorted(self, synthetic_timeseries_df: pd.DataFrame) -> None:
        from src.preprocessing import validate_timestamps

        df = synthetic_timeseries_df.iloc[::-1].reset_index(drop=True)
        result = validate_timestamps(df)
        assert result["is_sorted"] is False

    def test_detects_duplicates(self, synthetic_timeseries_df: pd.DataFrame) -> None:
        from src.preprocessing import validate_timestamps

        dup = synthetic_timeseries_df.iloc[[5]].copy()
        df = pd.concat([synthetic_timeseries_df, dup], ignore_index=True).sort_values(
            "timestamp"
        )
        result = validate_timestamps(df)
        assert result["n_duplicates"] == 1

    def test_detects_gaps(self) -> None:
        from src.preprocessing import validate_timestamps

        timestamps = pd.date_range("2020-01-01", periods=10, freq="1h")
        # Remove one to create a gap
        timestamps = timestamps.delete(5)
        df = pd.DataFrame({"timestamp": timestamps, "val": range(9)})
        result = validate_timestamps(df, expected_interval="1h")
        assert result["n_irregular_intervals"] >= 1

    def test_missing_column_raises(self) -> None:
        from src.preprocessing import validate_timestamps

        df = pd.DataFrame({"val": [1, 2, 3]})
        with pytest.raises(ValueError, match="not found"):
            validate_timestamps(df)


# ---------------------------------------------------------------------------
# Unit validation
# ---------------------------------------------------------------------------


class TestValidateUnits:
    def test_valid_columns(self, synthetic_timeseries_df: pd.DataFrame) -> None:
        from src.preprocessing import validate_units

        units = {col: "test_unit" for col in SYNTHETIC_FEATURE_NAMES}
        result = validate_units(synthetic_timeseries_df, units)
        assert result == units

    def test_missing_column_raises(self, synthetic_timeseries_df: pd.DataFrame) -> None:
        from src.preprocessing import validate_units

        with pytest.raises(KeyError, match="not found"):
            validate_units(synthetic_timeseries_df, {"nonexistent_col": "K"})


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------


class TestChronologicalSplit:
    def test_basic_split(self, synthetic_timeseries_df: pd.DataFrame) -> None:
        from src.preprocessing import chronological_split

        train, val, test = chronological_split(
            synthetic_timeseries_df,
            train_end="2020-01-25",
            val_end="2020-02-08",
        )
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert len(train) + len(val) + len(test) == len(synthetic_timeseries_df)

    def test_strict_chronological_ordering(
        self, synthetic_timeseries_df: pd.DataFrame
    ) -> None:
        from src.preprocessing import chronological_split

        train, val, test = chronological_split(
            synthetic_timeseries_df,
            train_end="2020-01-25",
            val_end="2020-02-08",
        )
        assert train["timestamp"].max() < val["timestamp"].min()
        assert val["timestamp"].max() < test["timestamp"].min()

    def test_no_overlap(self, synthetic_timeseries_df: pd.DataFrame) -> None:
        from src.preprocessing import chronological_split

        train, val, test = chronological_split(
            synthetic_timeseries_df,
            train_end="2020-01-25",
            val_end="2020-02-08",
        )
        train_ts = set(train["timestamp"])
        val_ts = set(val["timestamp"])
        test_ts = set(test["timestamp"])
        assert len(train_ts & val_ts) == 0
        assert len(val_ts & test_ts) == 0
        assert len(train_ts & test_ts) == 0

    def test_invalid_order_raises(self, synthetic_timeseries_df: pd.DataFrame) -> None:
        from src.preprocessing import chronological_split

        with pytest.raises(ValueError, match="before"):
            chronological_split(
                synthetic_timeseries_df,
                train_end="2020-02-08",
                val_end="2020-01-25",
            )

    def test_empty_split_raises(self, synthetic_timeseries_df: pd.DataFrame) -> None:
        from src.preprocessing import chronological_split

        with pytest.raises(ValueError, match="empty"):
            chronological_split(
                synthetic_timeseries_df,
                train_end="2019-01-01",  # before data starts
                val_end="2019-06-01",
            )


# ---------------------------------------------------------------------------
# Scaler — train-only fitting
# ---------------------------------------------------------------------------


class TestScalerTrainOnlyFit:
    def test_fit_on_train_only(self, synthetic_train_val_test) -> None:
        from src.preprocessing import fit_scaler, transform_data

        train_df, val_df, test_df = synthetic_train_val_test
        features = SYNTHETIC_FEATURE_NAMES

        scaler = fit_scaler(train_df, "StandardScaler", features)

        # Verify scaler learned from training data
        train_mean = train_df[features].mean().values
        np.testing.assert_allclose(scaler.mean_, train_mean, atol=1e-6)

        # Transform all splits using the SAME scaler
        train_scaled = transform_data(train_df, scaler, features)
        val_scaled = transform_data(val_df, scaler, features)
        _ = transform_data(test_df, scaler, features)  # verify test transform works

        # Training data should be approximately zero-mean
        assert abs(train_scaled[features].mean().mean()) < 0.1

        # Val/test should NOT necessarily be zero-mean (different distribution)
        # Just verify they were transformed (values changed)
        assert not np.allclose(
            val_df[features].values, val_scaled[features].values
        )

    def test_minmax_scaler(self, synthetic_train_val_test) -> None:
        from src.preprocessing import fit_scaler, transform_data

        train_df, _, _ = synthetic_train_val_test
        features = SYNTHETIC_FEATURE_NAMES

        scaler = fit_scaler(train_df, "MinMaxScaler", features)
        train_scaled = transform_data(train_df, scaler, features)

        # Training data should be in [0, 1] range
        assert train_scaled[features].min().min() >= -1e-10
        assert train_scaled[features].max().max() <= 1.0 + 1e-10

    def test_unknown_scaler_raises(self, synthetic_train_val_test) -> None:
        from src.preprocessing import fit_scaler

        train_df, _, _ = synthetic_train_val_test
        with pytest.raises(ValueError, match="Unknown scaler_type"):
            fit_scaler(train_df, "BogusScaler", SYNTHETIC_FEATURE_NAMES)

    def test_missing_feature_raises(self, synthetic_train_val_test) -> None:
        from src.preprocessing import fit_scaler

        train_df, _, _ = synthetic_train_val_test
        with pytest.raises(KeyError, match="not found"):
            fit_scaler(train_df, "StandardScaler", ["nonexistent_column"])


# ---------------------------------------------------------------------------
# Scaler save/load roundtrip and inverse transform
# ---------------------------------------------------------------------------


class TestScalerPersistence:
    def test_save_load_roundtrip(self, synthetic_train_val_test, tmp_path) -> None:
        from src.preprocessing import (
            fit_scaler,
            load_scaler,
            save_scaler,
            transform_data,
        )

        train_df, _, _ = synthetic_train_val_test
        features = SYNTHETIC_FEATURE_NAMES

        scaler = fit_scaler(train_df, "StandardScaler", features)
        path = tmp_path / "test_scaler.joblib"
        save_scaler(scaler, features, {"scaler_type": "StandardScaler"}, path)

        loaded_scaler, loaded_features, loaded_meta = load_scaler(path)
        assert loaded_features == features
        assert loaded_meta["scaler_type"] == "StandardScaler"

        # Verify identical transform
        orig = transform_data(train_df, scaler, features)
        loaded = transform_data(train_df, loaded_scaler, features)
        np.testing.assert_array_equal(
            orig[features].values, loaded[features].values
        )

    def test_inverse_transform_roundtrip(self, synthetic_train_val_test) -> None:
        from src.preprocessing import fit_scaler, transform_data

        train_df, _, _ = synthetic_train_val_test
        features = SYNTHETIC_FEATURE_NAMES

        scaler = fit_scaler(train_df, "StandardScaler", features)
        scaled = transform_data(train_df, scaler, features)

        # Inverse transform
        recovered = scaled.copy()
        recovered[features] = scaler.inverse_transform(scaled[features].values)

        np.testing.assert_allclose(
            train_df[features].values,
            recovered[features].values,
            atol=1e-6,
        )

    def test_load_nonexistent_raises(self, tmp_path) -> None:
        from src.preprocessing import load_scaler

        with pytest.raises(FileNotFoundError):
            load_scaler(tmp_path / "nonexistent.joblib")

    def test_feature_order_stability(self, synthetic_train_val_test) -> None:
        """Verify that feature order is preserved through save/load."""
        from src.preprocessing import fit_scaler, load_scaler, save_scaler

        train_df, _, _ = synthetic_train_val_test
        features = SYNTHETIC_FEATURE_NAMES

        scaler = fit_scaler(train_df, "StandardScaler", features)
        path = "results/checkpoints/_test_scaler_order.joblib"
        save_scaler(scaler, features, {}, path)
        _, loaded_features, _ = load_scaler(path)

        assert loaded_features == features

        # Cleanup
        import os

        os.remove(path)
