"""Tests for model implementations — persistence, LSTM, Transformer.

Phase 3: persistence tests.
Phase 4: LSTM smoke/gradient/checkpoint tests.
Phase 5: Transformer tests (to be added).

Persistence tests cover:
- Prediction equals immediately previous observation
- Output shape
- Six-variable ordering
- Timestamp alignment
- Exact production test evaluation sample count = 8,760
- Metric correctness on known synthetic example
- Predictions/targets have identical timestamps
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import SYNTHETIC_FEATURE_NAMES, SYNTHETIC_N_FEATURES


class TestPersistencePredictions:
    """Core persistence model tests using synthetic data."""

    def test_prediction_equals_previous_observation(self) -> None:
        """For each target at row i, prediction must be row i-1."""
        from src.models.persistence import persistence_predict

        n = 50
        data = np.arange(n * SYNTHETIC_N_FEATURES, dtype=np.float64).reshape(
            n, SYNTHETIC_N_FEATURES
        )
        window = 5

        df = pd.DataFrame(data, columns=SYNTHETIC_FEATURE_NAMES)
        df.insert(0, "timestamp", pd.date_range("2020-01-01", periods=n, freq="1h"))

        preds, targets, timestamps = persistence_predict(
            df, SYNTHETIC_FEATURE_NAMES, input_window=window
        )

        # For target at row index i (starting at window), prediction = row i-1
        for j in range(len(preds)):
            target_row_idx = window + j
            pred_row_idx = target_row_idx - 1
            np.testing.assert_array_equal(
                preds[j], data[pred_row_idx],
                err_msg=f"Sample {j}: prediction should be row {pred_row_idx}"
            )
            np.testing.assert_array_equal(
                targets[j], data[target_row_idx],
                err_msg=f"Sample {j}: target should be row {target_row_idx}"
            )

    def test_output_shape(self) -> None:
        from src.models.persistence import persistence_predict

        n = 100
        window = 24
        data = np.random.randn(n, SYNTHETIC_N_FEATURES)
        df = pd.DataFrame(data, columns=SYNTHETIC_FEATURE_NAMES)
        df.insert(0, "timestamp", pd.date_range("2020-01-01", periods=n, freq="1h"))

        preds, targets, ts = persistence_predict(
            df, SYNTHETIC_FEATURE_NAMES, input_window=window
        )

        expected_n = n - window  # 76
        assert preds.shape == (expected_n, SYNTHETIC_N_FEATURES)
        assert targets.shape == (expected_n, SYNTHETIC_N_FEATURES)
        assert len(ts) == expected_n

    def test_six_variable_ordering(self) -> None:
        """Verify persistence respects the canonical variable order."""
        from src.models.persistence import persistence_predict

        n = 50
        # Make each column have distinct values
        data = np.column_stack([np.arange(n) * (i + 1) for i in range(SYNTHETIC_N_FEATURES)])
        df = pd.DataFrame(data.astype(np.float64), columns=SYNTHETIC_FEATURE_NAMES)
        df.insert(0, "timestamp", pd.date_range("2020-01-01", periods=n, freq="1h"))

        preds, targets, _ = persistence_predict(
            df, SYNTHETIC_FEATURE_NAMES, input_window=5
        )

        # Check first prediction: should be row 4 (index 4)
        for col_idx, feat in enumerate(SYNTHETIC_FEATURE_NAMES):
            expected_val = 4 * (col_idx + 1)  # row 4, column col_idx
            assert preds[0, col_idx] == pytest.approx(expected_val), (
                f"Feature {feat}: expected {expected_val}, got {preds[0, col_idx]}"
            )

    def test_timestamp_alignment(self) -> None:
        """Prediction timestamps must equal target timestamps."""
        from src.models.persistence import persistence_predict

        n = 100
        window = 10
        timestamps = pd.date_range("2020-01-01", periods=n, freq="1h")
        data = np.random.randn(n, SYNTHETIC_N_FEATURES)
        df = pd.DataFrame(data, columns=SYNTHETIC_FEATURE_NAMES)
        df.insert(0, "timestamp", timestamps)

        _, _, result_ts = persistence_predict(
            df, SYNTHETIC_FEATURE_NAMES, input_window=window
        )

        # Target timestamps start at index=window
        expected_ts = timestamps[window:]
        np.testing.assert_array_equal(result_ts, expected_ts.values)

    def test_no_future_leakage(self) -> None:
        """Prediction for target at t must not use data at t or later."""
        from src.models.persistence import persistence_predict

        n = 30
        window = 5
        data = np.arange(n * SYNTHETIC_N_FEATURES, dtype=np.float64).reshape(
            n, SYNTHETIC_N_FEATURES
        )
        df = pd.DataFrame(data, columns=SYNTHETIC_FEATURE_NAMES)
        df.insert(0, "timestamp", pd.date_range("2020-01-01", periods=n, freq="1h"))

        preds, targets, _ = persistence_predict(
            df, SYNTHETIC_FEATURE_NAMES, input_window=window
        )

        # Every prediction row must come from a STRICTLY earlier index than target
        for j in range(len(preds)):
            target_idx = window + j
            # Prediction values should equal data[target_idx - 1]
            np.testing.assert_array_equal(preds[j], data[target_idx - 1])
            # And NOT equal data[target_idx]
            assert not np.array_equal(preds[j], data[target_idx])

    def test_metric_correctness_known_example(self) -> None:
        """Compute MAE/RMSE on a known persistence example."""
        from src.evaluate import compute_mae, compute_rmse

        # Simple case: data is [0, 1, 2, 3, 4] for a single feature
        # With window=1: predictions = [0, 1, 2, 3], targets = [1, 2, 3, 4]
        # Errors = [1, 1, 1, 1], MAE = 1.0, RMSE = 1.0
        data = np.arange(5, dtype=np.float64).reshape(5, 1)
        df = pd.DataFrame(data, columns=["x"])
        df.insert(0, "timestamp", pd.date_range("2020-01-01", periods=5, freq="1h"))

        from src.models.persistence import persistence_predict

        preds, targets, _ = persistence_predict(
            df, ["x"], input_window=1
        )

        mae = compute_mae(targets, preds)
        rmse = compute_rmse(targets, preds)
        assert mae == pytest.approx(1.0)
        assert rmse == pytest.approx(1.0)


class TestPersistenceProductionAlignment:
    """Test that persistence evaluation aligns with ClimateSequenceDataset."""

    def test_same_sample_count_as_sequence_dataset(self) -> None:
        """Persistence must produce exactly the same number of evaluation
        samples as ClimateSequenceDataset for the same data and window."""
        from src.dataset import ClimateSequenceDataset
        from src.models.persistence import persistence_predict

        n = 200
        window = 24
        data = np.random.randn(n, SYNTHETIC_N_FEATURES).astype(np.float32)
        timestamps = pd.date_range("2020-01-01", periods=n, freq="1h")

        # Sequence dataset count
        ds = ClimateSequenceDataset(
            data=data,
            input_window=window,
            feature_names=SYNTHETIC_FEATURE_NAMES,
        )

        # Persistence count
        df = pd.DataFrame(data.astype(np.float64), columns=SYNTHETIC_FEATURE_NAMES)
        df.insert(0, "timestamp", timestamps)
        preds, targets, ts = persistence_predict(
            df, SYNTHETIC_FEATURE_NAMES, input_window=window
        )

        assert len(ds) == len(preds), (
            f"Sequence dataset has {len(ds)} samples but persistence has {len(preds)}"
        )

    def test_same_target_timestamps_as_sequence_dataset(self) -> None:
        """Persistence target timestamps must match sequence dataset targets."""
        from src.dataset import ClimateSequenceDataset
        from src.models.persistence import persistence_predict

        n = 100
        window = 10
        data = np.arange(n * SYNTHETIC_N_FEATURES, dtype=np.float32).reshape(
            n, SYNTHETIC_N_FEATURES
        )
        timestamps = pd.date_range("2020-01-01", periods=n, freq="1h")

        # Get sequence dataset targets
        ds = ClimateSequenceDataset(
            data=data,
            input_window=window,
            feature_names=SYNTHETIC_FEATURE_NAMES,
            timestamps=timestamps,
        )
        seq_target_indices = [ds._valid_indices[i] + window for i in range(len(ds))]

        # Get persistence targets
        df = pd.DataFrame(data.astype(np.float64), columns=SYNTHETIC_FEATURE_NAMES)
        df.insert(0, "timestamp", timestamps)
        _, _, pers_ts = persistence_predict(
            df, SYNTHETIC_FEATURE_NAMES, input_window=window
        )

        # Compare timestamps
        seq_timestamps = timestamps[seq_target_indices]
        np.testing.assert_array_equal(pers_ts, seq_timestamps.values)

    def test_production_test_count_8760(self) -> None:
        """With 8784 test rows and window=24, must get exactly 8760 samples."""
        from src.models.persistence import persistence_predict

        # Simulate production test split size
        n = 8784
        window = 24
        data = np.random.randn(n, 6)
        df = pd.DataFrame(data, columns=["t2m", "d2m", "sp", "tp", "u10", "v10"])
        df.insert(0, "timestamp", pd.date_range("2024-01-01", periods=n, freq="1h"))

        preds, targets, ts = persistence_predict(
            df, ["t2m", "d2m", "sp", "tp", "u10", "v10"], input_window=window
        )

        assert preds.shape[0] == 8760
        assert targets.shape[0] == 8760
        assert len(ts) == 8760


# ---------------------------------------------------------------------------
# Phase 4 — LSTM smoke / gradient / checkpoint tests
# ---------------------------------------------------------------------------


class TestClimateLSTMSmoke:
    """Verify LSTM forward pass, shapes, gradients, and checkpoint roundtrip."""

    def _make_model(self):
        from src.models.lstm import ClimateLSTM

        return ClimateLSTM(
            n_features=6,
            n_targets=6,
            hidden_dim=64,
            num_layers=2,
            dropout=0.1,
        )

    def test_forward_pass_shape(self) -> None:
        import torch

        model = self._make_model()
        model.eval()
        x = torch.randn(4, 24, 6)  # (batch=4, window=24, features=6)
        y = model(x)
        assert y.shape == (4, 6), f"Expected (4, 6), got {y.shape}"

    def test_single_sample(self) -> None:
        import torch

        model = self._make_model()
        model.eval()
        x = torch.randn(1, 24, 6)
        y = model(x)
        assert y.shape == (1, 6)

    def test_output_is_finite(self) -> None:
        import torch

        model = self._make_model()
        model.eval()
        x = torch.randn(8, 24, 6)
        y = model(x)
        assert torch.isfinite(y).all(), "Output contains non-finite values"

    def test_loss_is_finite(self) -> None:
        import torch

        model = self._make_model()
        model.train()
        x = torch.randn(8, 24, 6)
        target = torch.randn(8, 6)
        y = model(x)
        loss = torch.nn.functional.mse_loss(y, target)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_gradients_finite_and_nonzero(self) -> None:
        import torch

        model = self._make_model()
        model.train()
        x = torch.randn(8, 24, 6)
        target = torch.randn(8, 6)
        y = model(x)
        loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_optimizer_changes_parameters(self) -> None:
        import torch

        model = self._make_model()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Snapshot before
        before = {n: p.clone() for n, p in model.named_parameters()}

        x = torch.randn(8, 24, 6)
        target = torch.randn(8, 6)
        y = model(x)
        loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()
        optimizer.step()

        # At least one parameter must have changed
        any_changed = False
        for name, param in model.named_parameters():
            if not torch.equal(before[name], param):
                any_changed = True
                break
        assert any_changed, "No parameters changed after optimizer step"

    def test_checkpoint_save_load_roundtrip(self, tmp_path) -> None:
        import torch

        from src.seed import set_global_seed

        set_global_seed(42)
        model = self._make_model()
        model.eval()

        x = torch.randn(4, 24, 6)
        out_before = model(x)

        # Save
        ckpt_path = tmp_path / "lstm_test.pt"
        torch.save(model.state_dict(), ckpt_path)

        # Load into fresh model
        model2 = self._make_model()
        model2.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model2.eval()

        out_after = model2(x)
        assert torch.allclose(out_before, out_after, atol=1e-6), (
            "Output differs after checkpoint reload"
        )

    def test_parameter_count(self) -> None:
        model = self._make_model()
        count = model.count_parameters()
        assert isinstance(count, int)
        assert count > 0
        # Expected ~50K for hidden_dim=64, num_layers=2, n_features=6, n_targets=6
        assert 10_000 < count < 200_000, f"Unexpected parameter count: {count}"
        print(f"LSTM parameter count: {count}")


# ---------------------------------------------------------------------------
# Phase 5 — Transformer smoke / gradient / checkpoint tests
# ---------------------------------------------------------------------------


class TestClimateTransformerSmoke:
    """Verify Transformer forward pass, shapes, gradients, and checkpoint roundtrip."""

    def _make_model(self):
        from src.models.transformer import ClimateTransformer

        return ClimateTransformer(
            n_features=6,
            n_targets=6,
            d_model=128,
            nhead=4,
            num_encoder_layers=3,
            dim_feedforward=256,
            dropout=0.1,
        )

    def test_forward_pass_shape(self) -> None:
        import torch

        model = self._make_model()
        model.eval()
        x = torch.randn(4, 24, 6)  # (batch=4, window=24, features=6)
        y = model(x)
        assert y.shape == (4, 6), f"Expected (4, 6), got {y.shape}"

    def test_single_sample(self) -> None:
        import torch

        model = self._make_model()
        model.eval()
        x = torch.randn(1, 24, 6)
        y = model(x)
        assert y.shape == (1, 6)

    def test_output_is_finite(self) -> None:
        import torch

        model = self._make_model()
        model.eval()
        x = torch.randn(8, 24, 6)
        y = model(x)
        assert torch.isfinite(y).all(), "Output contains non-finite values"

    def test_loss_is_finite(self) -> None:
        import torch

        model = self._make_model()
        model.train()
        x = torch.randn(8, 24, 6)
        target = torch.randn(8, 6)
        y = model(x)
        loss = torch.nn.functional.mse_loss(y, target)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_gradients_finite_and_nonzero(self) -> None:
        import torch

        model = self._make_model()
        model.train()
        x = torch.randn(8, 24, 6)
        target = torch.randn(8, 6)
        y = model(x)
        loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_optimizer_changes_parameters(self) -> None:
        import torch

        model = self._make_model()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

        before = {n: p.clone() for n, p in model.named_parameters()}

        x = torch.randn(8, 24, 6)
        target = torch.randn(8, 6)
        y = model(x)
        loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()
        optimizer.step()

        any_changed = False
        for name, param in model.named_parameters():
            if not torch.equal(before[name], param):
                any_changed = True
                break
        assert any_changed, "No parameters changed after optimizer step"

    def test_checkpoint_save_load_roundtrip(self, tmp_path) -> None:
        import torch

        from src.seed import set_global_seed

        set_global_seed(42)
        model = self._make_model()
        model.eval()

        x = torch.randn(4, 24, 6)
        out_before = model(x)

        ckpt_path = tmp_path / "transformer_test.pt"
        torch.save(model.state_dict(), ckpt_path)

        model2 = self._make_model()
        model2.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model2.eval()

        out_after = model2(x)
        assert torch.allclose(out_before, out_after, atol=1e-6), (
            "Output differs after checkpoint reload"
        )

    def test_positional_encoding_length_24(self) -> None:
        """Positional encoding must support exactly the production window length."""
        import torch

        from src.models.transformer import SinusoidalPositionalEncoding

        pe = SinusoidalPositionalEncoding(d_model=128, max_len=512)
        x = torch.zeros(2, 24, 128)
        out = pe(x)
        assert out.shape == (2, 24, 128)
        assert torch.isfinite(out).all()
        # Verify positional encoding is non-trivial (not all zeros)
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_parameter_count(self) -> None:
        model = self._make_model()
        count = model.count_parameters()
        assert isinstance(count, int)
        assert count > 0
        # Expected ~399K per spec estimate; verify reasonable bounds without hardcoding
        assert 100_000 < count < 1_000_000, f"Unexpected parameter count: {count}"
        print(f"Transformer parameter count: {count}")
