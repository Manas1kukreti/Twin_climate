"""Phase 0 — Environment and infrastructure smoke tests.

These tests verify that:
- Core library imports succeed.
- The seed utility runs without error and produces deterministic results.
- The manifest utility produces a valid dict and round-trips through JSON.
- The config loader reads YAML and the validator catches problems.
- The logging setup runs without error.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# ── Import smoke tests ────────────────────────────────────────────────


class TestCoreImports:
    """Verify that every core dependency can be imported."""

    def test_import_torch(self) -> None:
        import torch  # noqa: F401

    def test_import_numpy(self) -> None:
        import numpy  # noqa: F401

    def test_import_pandas(self) -> None:
        import pandas  # noqa: F401

    def test_import_sklearn(self) -> None:
        import sklearn  # noqa: F401

    def test_import_yaml(self) -> None:
        import yaml  # noqa: F401

    def test_import_joblib(self) -> None:
        import joblib  # noqa: F401


# ── Seed utility tests ────────────────────────────────────────────────


class TestSeedUtility:
    """Verify that set_global_seed works and produces deterministic output."""

    def test_seed_runs_without_error(self) -> None:
        from src.seed import set_global_seed

        set_global_seed(42)

    def test_seed_determinism_numpy(self) -> None:
        from src.seed import set_global_seed

        set_global_seed(0)
        a = np.random.rand(5)

        set_global_seed(0)
        b = np.random.rand(5)

        np.testing.assert_array_equal(a, b)

    def test_seed_determinism_torch(self) -> None:
        import torch

        from src.seed import set_global_seed

        set_global_seed(0)
        a = torch.randn(5)

        set_global_seed(0)
        b = torch.randn(5)

        assert torch.equal(a, b)

    def test_seed_rejects_negative(self) -> None:
        from src.seed import set_global_seed

        with pytest.raises(ValueError, match="non-negative"):
            set_global_seed(-1)


# ── Manifest utility tests ────────────────────────────────────────────


class TestManifestUtility:
    """Verify manifest creation, saving, and loading."""

    def test_create_manifest_returns_dict(self) -> None:
        from src.manifest import create_manifest

        m = create_manifest(
            run_id="test_run_001",
            stage="baseline",
            model="persistence",
            seed=42,
        )

        assert isinstance(m, dict)
        assert m["run_id"] == "test_run_001"
        assert m["stage"] == "baseline"
        assert m["model"] == "persistence"
        assert m["seed"] == 42

    def test_manifest_has_required_schema_fields(self) -> None:
        from src.manifest import create_manifest

        m = create_manifest(
            run_id="schema_test",
            stage="scratch",
            model="transformer",
            seed=7,
            device="cpu",
            dataset={"source": "synthetic"},
            model_config={"n_features": 6, "n_targets": 6},
        )

        # Check all §2.5 top-level keys are present
        expected_keys = {
            "run_id",
            "stage",
            "model",
            "seed",
            "device",
            "dataset",
            "preprocessing",
            "model_config",
            "training_config",
            "hyperparameter_selection",
            "checkpoint_path",
            "metrics_path",
            "predictions_path",
            "figures",
            "software",
            "git_commit",
            "created_at",
        }
        assert expected_keys.issubset(set(m.keys())), (
            f"Missing keys: {expected_keys - set(m.keys())}"
        )

    def test_manifest_software_versions(self) -> None:
        from src.manifest import create_manifest

        m = create_manifest(run_id="ver_test", stage="evaluate", model="lstm", seed=1)
        sw = m["software"]
        assert "python" in sw
        assert "pytorch" in sw
        assert "numpy" in sw
        assert "pandas" in sw

    def test_manifest_save_load_roundtrip(self, tmp_path: Path) -> None:
        from src.manifest import create_manifest, load_manifest, save_manifest

        m = create_manifest(
            run_id="roundtrip_test",
            stage="baseline",
            model="persistence",
            seed=42,
        )

        filepath = tmp_path / "test_manifest.json"
        save_manifest(m, filepath)

        assert filepath.exists()

        loaded = load_manifest(filepath)
        assert loaded["run_id"] == "roundtrip_test"
        assert loaded["seed"] == 42

    def test_load_manifest_missing_file(self, tmp_path: Path) -> None:
        from src.manifest import load_manifest

        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "nonexistent.json")

    def test_load_manifest_invalid_json(self, tmp_path: Path) -> None:
        from src.manifest import load_manifest

        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json at all", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            load_manifest(bad_file)

    def test_load_manifest_missing_required_keys(self, tmp_path: Path) -> None:
        from src.manifest import load_manifest

        incomplete = tmp_path / "incomplete.json"
        incomplete.write_text('{"run_id": "x"}', encoding="utf-8")

        with pytest.raises(ValueError, match="missing required keys"):
            load_manifest(incomplete)


# ── Config loading tests ──────────────────────────────────────────────


class TestConfigLoading:
    """Verify config loading and validation."""

    def test_load_data_config_from_file(self) -> None:
        """Load the actual data.yaml template and verify it parses."""
        from src.config import load_config

        cfg = load_config("configs/data.yaml")
        assert isinstance(cfg, dict)
        assert "source" in cfg
        assert "variables" in cfg

    def test_validate_data_config_resolved_after_task_1(self) -> None:
        """After Task 1.1/1.2, data.yaml has resolved core fields and passes validation."""
        from src.config import load_config, validate_data_config

        cfg = load_config("configs/data.yaml")
        errors = validate_data_config(cfg)

        # Core fields (source, source_type, target_location, sampling_interval)
        # were resolved during Task 1.1. Variables list is populated.
        assert errors == [], f"Unexpected validation errors: {errors}"

    def test_validate_model_config_still_unresolved(self) -> None:
        """Pretrain config should still have UNRESOLVED fields before Phase 8."""
        from src.config import load_config, validate_model_config

        cfg = load_config("configs/pretrain.yaml")
        errors = validate_model_config(cfg, "pretrain")
        assert len(errors) > 0

    def test_validate_data_config_valid(self, minimal_data_config: dict) -> None:
        """A fully resolved synthetic data config should pass validation."""
        from src.config import validate_data_config

        errors = validate_data_config(minimal_data_config)
        assert errors == []

    def test_validate_model_config_valid(self, minimal_lstm_config: dict) -> None:
        from src.config import validate_model_config

        errors = validate_model_config(minimal_lstm_config, "lstm")
        assert errors == []

    def test_validate_model_config_missing_fields(self) -> None:
        from src.config import validate_model_config

        errors = validate_model_config({}, "lstm")
        assert len(errors) > 0

    def test_validate_n_features_targets_consistency(self) -> None:
        from src.config import validate_model_config

        bad_cfg = {
            "n_features": 3,
            "n_targets": 10,  # exceeds n_features
            "hidden_dim": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "learning_rate": 1e-3,
            "batch_size": 16,
            "epochs": 2,
            "seed": 42,
        }
        errors = validate_model_config(bad_cfg, "lstm")
        assert any("n_targets" in e and "n_features" in e for e in errors)

    def test_load_nonexistent_config(self) -> None:
        from src.config import load_config

        with pytest.raises(FileNotFoundError):
            load_config("configs/does_not_exist.yaml")


# ── Logging setup tests ──────────────────────────────────────────────


class TestLoggingSetup:
    """Verify logging configuration runs without error."""

    def test_setup_logging_default(self) -> None:
        from src.logging_config import setup_logging

        setup_logging()

    def test_setup_logging_debug(self) -> None:
        from src.logging_config import setup_logging

        setup_logging("DEBUG")

    def test_setup_logging_to_file(self, tmp_path: Path) -> None:
        from src.logging_config import setup_logging

        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file)
        assert log_file.exists()


# ── Synthetic fixture sanity tests ────────────────────────────────────


class TestSyntheticFixtures:
    """Verify that conftest fixtures produce usable test data."""

    def test_synthetic_df_shape(self, synthetic_timeseries_df) -> None:
        assert len(synthetic_timeseries_df) == 200
        assert "timestamp" in synthetic_timeseries_df.columns

    def test_synthetic_df_has_all_features(
        self, synthetic_timeseries_df, synthetic_feature_names
    ) -> None:
        for name in synthetic_feature_names:
            assert name in synthetic_timeseries_df.columns

    def test_synthetic_split_chronological(self, synthetic_train_val_test) -> None:
        train_df, val_df, test_df = synthetic_train_val_test

        assert train_df["timestamp"].max() < val_df["timestamp"].min()
        assert val_df["timestamp"].max() < test_df["timestamp"].min()

    def test_synthetic_split_no_overlap(self, synthetic_train_val_test) -> None:
        train_df, val_df, test_df = synthetic_train_val_test

        train_ts = set(train_df["timestamp"])
        val_ts = set(val_df["timestamp"])
        test_ts = set(test_df["timestamp"])

        assert len(train_ts & val_ts) == 0
        assert len(val_ts & test_ts) == 0
        assert len(train_ts & test_ts) == 0

    def test_synthetic_split_sizes(self, synthetic_train_val_test) -> None:
        train_df, val_df, test_df = synthetic_train_val_test
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == 200
