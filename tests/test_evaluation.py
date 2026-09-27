"""Tests for src/evaluate.py — metric correctness and I/O.

Covers:
- MAE correctness on known values
- RMSE correctness on known values
- Inverse transform roundtrip
- Per-variable metric structure and content
- Metric JSON save/load roundtrip
- No mixed-unit aggregate as primary metric
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


class TestComputeMAE:
    def test_zero_error(self) -> None:
        from src.evaluate import compute_mae

        y = np.array([1.0, 2.0, 3.0])
        assert compute_mae(y, y) == 0.0

    def test_known_values(self) -> None:
        from src.evaluate import compute_mae

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert compute_mae(y_true, y_pred) == pytest.approx(1.0)

    def test_negative_errors(self) -> None:
        from src.evaluate import compute_mae

        y_true = np.array([3.0, 2.0, 1.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        # |3-1| + |2-1| + |1-1| = 2 + 1 + 0 = 3, mean = 1.0
        assert compute_mae(y_true, y_pred) == pytest.approx(1.0)


class TestComputeRMSE:
    def test_zero_error(self) -> None:
        from src.evaluate import compute_rmse

        y = np.array([1.0, 2.0, 3.0])
        assert compute_rmse(y, y) == 0.0

    def test_known_values(self) -> None:
        from src.evaluate import compute_rmse

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        # (1+1+1)/3 = 1, sqrt(1) = 1
        assert compute_rmse(y_true, y_pred) == pytest.approx(1.0)

    def test_unequal_errors(self) -> None:
        from src.evaluate import compute_rmse

        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        # (9 + 16) / 2 = 12.5, sqrt(12.5) ≈ 3.5355
        assert compute_rmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5))


class TestPerVariableMetrics:
    def test_structure(self) -> None:
        from src.evaluate import compute_per_variable_metrics

        y_true = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        y_pred = np.array([[1.5, 11.0], [2.5, 21.0], [3.5, 31.0]])
        result = compute_per_variable_metrics(y_true, y_pred, ["a", "b"])
        assert "a" in result
        assert "b" in result
        assert "mae" in result["a"]
        assert "rmse" in result["a"]

    def test_values(self) -> None:
        from src.evaluate import compute_per_variable_metrics

        y_true = np.zeros((10, 2))
        y_pred = np.ones((10, 2))
        result = compute_per_variable_metrics(y_true, y_pred, ["x", "y"])
        assert result["x"]["mae"] == pytest.approx(1.0)
        assert result["x"]["rmse"] == pytest.approx(1.0)

    def test_shape_mismatch_raises(self) -> None:
        from src.evaluate import compute_per_variable_metrics

        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_per_variable_metrics(
                np.zeros((5, 2)), np.zeros((3, 2)), ["a", "b"]
            )

    def test_variable_count_mismatch_raises(self) -> None:
        from src.evaluate import compute_per_variable_metrics

        with pytest.raises(ValueError, match="variable_names"):
            compute_per_variable_metrics(
                np.zeros((5, 2)), np.zeros((5, 2)), ["a"]
            )


class TestMetricIO:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        from src.evaluate import load_metrics, save_metrics

        metrics = {"t2m": {"mae": 1.5, "rmse": 2.0}, "d2m": {"mae": 0.8, "rmse": 1.1}}
        path = tmp_path / "test_metrics.json"
        save_metrics(metrics, path)
        loaded = load_metrics(path)
        assert loaded == metrics

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        from src.evaluate import load_metrics

        with pytest.raises(FileNotFoundError):
            load_metrics(tmp_path / "nonexistent.json")


class TestInverseTransform:
    def test_roundtrip(self) -> None:
        from sklearn.preprocessing import StandardScaler

        from src.evaluate import inverse_transform_predictions

        rng = np.random.RandomState(42)
        data = rng.randn(100, 3) * 10 + 5
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
        recovered = inverse_transform_predictions(scaled, scaler, ["a", "b", "c"])
        np.testing.assert_allclose(recovered, data, atol=1e-10)

    def test_1d_input(self) -> None:
        from sklearn.preprocessing import StandardScaler

        from src.evaluate import inverse_transform_predictions

        scaler = StandardScaler()
        scaler.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
        scaled_1d = scaler.transform(np.array([[2.0, 3.0]]))[0]
        recovered = inverse_transform_predictions(scaled_1d, scaler, ["a", "b"])
        np.testing.assert_allclose(recovered, [2.0, 3.0], atol=1e-10)


class TestLoadAndValidateBaselineMetrics:
    """Tests for the baseline-comparison loader used before Transformer evaluation."""

    FEATURE_ORDER = ["t2m", "d2m", "sp", "tp", "u10", "v10"]
    UNITS = {"t2m": "°C", "d2m": "°C", "sp": "hPa", "tp": "mm", "u10": "m/s", "v10": "m/s"}

    def test_real_baseline_artifacts_pass_validation(self) -> None:
        """Persistence and LSTM metrics on disk must satisfy the comparison gate."""
        from src.evaluate import load_and_validate_baseline_metrics

        results = load_and_validate_baseline_metrics(
            paths={
                "persistence": "results/metrics/persistence_delhi_metrics.json",
                "lstm": "results/metrics/lstm_delhi_metrics.json",
            },
            expected_feature_order=self.FEATURE_ORDER,
            expected_units=self.UNITS,
            expected_sample_count=8760,
            expected_input_window=24,
            expected_first_timestamp="2024-01-02T00:00:00",
            expected_last_timestamp="2024-12-31T23:00:00",
            predictions_paths={
                "lstm": "results/predictions/lstm_delhi_predictions.csv",
            },
        )

        assert "persistence" in results
        assert "lstm" in results
        assert "t2m" in results["persistence"]
        assert "rmse" in results["persistence"]["t2m"]

    def test_feature_order_mismatch_raises(self, tmp_path: Path) -> None:
        from src.evaluate import load_and_validate_baseline_metrics, save_metrics

        bad_doc = {
            "feature_order": ["t2m", "sp"],
            "units": {"t2m": "°C", "sp": "hPa"},
            "test_sample_count": 8760,
            "input_window": 24,
            "timestamp_range": {"first": "2024-01-02T00:00:00", "last": "2024-12-31T23:00:00"},
            "per_variable_metrics": {"t2m": {"mae": 1.0, "rmse": 1.0}},
        }
        path = tmp_path / "bad_metrics.json"
        save_metrics(bad_doc, path)

        with pytest.raises(ValueError, match="feature/target order mismatch"):
            load_and_validate_baseline_metrics(
                paths={"bad_model": path},
                expected_feature_order=self.FEATURE_ORDER,
                expected_units={"t2m": "°C"},
            )

    def test_sample_count_mismatch_raises(self, tmp_path: Path) -> None:
        from src.evaluate import load_and_validate_baseline_metrics, save_metrics

        bad_doc = {
            "feature_order": ["t2m"],
            "units": {"t2m": "°C"},
            "test_sample_count": 100,
            "input_window": 24,
            "timestamp_range": {"first": "2024-01-02T00:00:00", "last": "2024-12-31T23:00:00"},
            "per_variable_metrics": {"t2m": {"mae": 1.0, "rmse": 1.0}},
        }
        path = tmp_path / "bad_metrics.json"
        save_metrics(bad_doc, path)

        with pytest.raises(ValueError, match="evaluated-sample count mismatch"):
            load_and_validate_baseline_metrics(
                paths={"bad_model": path},
                expected_feature_order=["t2m"],
                expected_units={"t2m": "°C"},
                expected_sample_count=8760,
            )

    def test_first_timestamp_mismatch_raises(self, tmp_path: Path) -> None:
        from src.evaluate import load_and_validate_baseline_metrics, save_metrics

        bad_doc = {
            "feature_order": ["t2m"],
            "units": {"t2m": "°C"},
            "test_sample_count": 8760,
            "input_window": 24,
            "timestamp_range": {"first": "2024-01-01T00:00:00", "last": "2024-12-31T23:00:00"},
            "per_variable_metrics": {"t2m": {"mae": 1.0, "rmse": 1.0}},
        }
        path = tmp_path / "bad_metrics.json"
        save_metrics(bad_doc, path)

        with pytest.raises(ValueError, match="first target timestamp mismatch"):
            load_and_validate_baseline_metrics(
                paths={"bad_model": path},
                expected_feature_order=["t2m"],
                expected_units={"t2m": "°C"},
                expected_sample_count=8760,
                expected_first_timestamp="2024-01-02T00:00:00",
                expected_last_timestamp="2024-12-31T23:00:00",
            )

    def test_last_timestamp_mismatch_raises(self, tmp_path: Path) -> None:
        from src.evaluate import load_and_validate_baseline_metrics, save_metrics

        bad_doc = {
            "feature_order": ["t2m"],
            "units": {"t2m": "°C"},
            "test_sample_count": 8760,
            "input_window": 24,
            "timestamp_range": {"first": "2024-01-02T00:00:00", "last": "2024-12-30T23:00:00"},
            "per_variable_metrics": {"t2m": {"mae": 1.0, "rmse": 1.0}},
        }
        path = tmp_path / "bad_metrics.json"
        save_metrics(bad_doc, path)

        with pytest.raises(ValueError, match="last target timestamp mismatch"):
            load_and_validate_baseline_metrics(
                paths={"bad_model": path},
                expected_feature_order=["t2m"],
                expected_units={"t2m": "°C"},
                expected_sample_count=8760,
                expected_first_timestamp="2024-01-02T00:00:00",
                expected_last_timestamp="2024-12-31T23:00:00",
            )

    def test_missing_input_window_raises(self, tmp_path: Path) -> None:
        from src.evaluate import load_and_validate_baseline_metrics, save_metrics

        bad_doc = {
            "feature_order": ["t2m"],
            "units": {"t2m": "°C"},
            "test_sample_count": 8760,
            # no input_window anywhere
            "timestamp_range": {"first": "2024-01-02T00:00:00", "last": "2024-12-31T23:00:00"},
            "per_variable_metrics": {"t2m": {"mae": 1.0, "rmse": 1.0}},
        }
        path = tmp_path / "bad_metrics.json"
        save_metrics(bad_doc, path)

        with pytest.raises(ValueError, match="could not find 'input_window'"):
            load_and_validate_baseline_metrics(
                paths={"bad_model": path},
                expected_feature_order=["t2m"],
                expected_units={"t2m": "°C"},
                expected_sample_count=8760,
            )

    def test_missing_timestamp_without_predictions_path_raises(self, tmp_path: Path) -> None:
        """If timestamp_range is absent and no predictions_paths fallback is
        given, the function must raise rather than fabricate timestamps."""
        from src.evaluate import load_and_validate_baseline_metrics, save_metrics

        bad_doc = {
            "feature_order": ["t2m"],
            "units": {"t2m": "°C"},
            "test_sample_count": 8760,
            "input_window": 24,
            # no timestamp_range
            "per_variable_metrics": {"t2m": {"mae": 1.0, "rmse": 1.0}},
        }
        path = tmp_path / "bad_metrics.json"
        save_metrics(bad_doc, path)

        with pytest.raises(ValueError, match="no predictions_paths entry"):
            load_and_validate_baseline_metrics(
                paths={"bad_model": path},
                expected_feature_order=["t2m"],
                expected_units={"t2m": "°C"},
                expected_sample_count=8760,
            )

    def test_fallback_to_predictions_csv_reads_real_timestamps(self, tmp_path: Path) -> None:
        """When timestamp_range is absent, the real predictions CSV is read."""
        import pandas as pd

        from src.evaluate import load_and_validate_baseline_metrics, save_metrics

        # Metrics doc with no timestamp_range
        doc = {
            "feature_order": ["t2m"],
            "units": {"t2m": "°C"},
            "test_sample_count": 3,
            "input_window": 24,
            "per_variable_metrics": {"t2m": {"mae": 1.0, "rmse": 1.0}},
        }
        metrics_path = tmp_path / "metrics.json"
        save_metrics(doc, metrics_path)

        # Real predictions CSV with actual timestamps
        pred_df = pd.DataFrame({
            "timestamp": ["2024-01-02T00:00:00", "2024-01-02T01:00:00", "2024-12-31T23:00:00"],
            "actual_t2m": [1.0, 2.0, 3.0],
            "pred_t2m": [1.1, 2.1, 3.1],
        })
        pred_path = tmp_path / "predictions.csv"
        pred_df.to_csv(pred_path, index=False)

        results = load_and_validate_baseline_metrics(
            paths={"model_x": metrics_path},
            expected_feature_order=["t2m"],
            expected_units={"t2m": "°C"},
            expected_sample_count=3,
            expected_first_timestamp="2024-01-02T00:00:00",
            expected_last_timestamp="2024-12-31T23:00:00",
            predictions_paths={"model_x": pred_path},
        )
        assert "model_x" in results
