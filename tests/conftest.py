"""Shared pytest fixtures for ClimateTwin tests.

Provides small, deterministic synthetic data generators and temporary
directory helpers used across multiple test modules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic time-series data
# ---------------------------------------------------------------------------

# Representative climate variable names (not tied to a real dataset yet)
SYNTHETIC_FEATURE_NAMES: list[str] = [
    "temperature_2m",
    "relative_humidity",
    "surface_pressure",
    "wind_u_10m",
    "wind_v_10m",
    "precipitation",
]

SYNTHETIC_N_FEATURES: int = len(SYNTHETIC_FEATURE_NAMES)
SYNTHETIC_N_TARGETS: int = SYNTHETIC_N_FEATURES  # predict all features


@pytest.fixture()
def synthetic_feature_names() -> list[str]:
    """Return the canonical list of synthetic feature names."""
    return list(SYNTHETIC_FEATURE_NAMES)


@pytest.fixture()
def synthetic_timeseries_df() -> pd.DataFrame:
    """Create a small, deterministic synthetic climate time-series DataFrame.

    Returns a DataFrame with 200 rows, 6 climate variables, and a
    ``timestamp`` column at 6-hour intervals starting 2020-01-01.
    Values are sinusoidal with small random noise (seeded).
    """
    rng = np.random.RandomState(42)
    n_rows = 200
    timestamps = pd.date_range("2020-01-01", periods=n_rows, freq="6h")

    data: dict[str, np.ndarray] = {}
    for i, name in enumerate(SYNTHETIC_FEATURE_NAMES):
        base = np.sin(np.linspace(0, 4 * np.pi, n_rows) + i)
        noise = rng.normal(0, 0.1, n_rows)
        data[name] = base + noise

    df = pd.DataFrame(data)
    df.insert(0, "timestamp", timestamps)
    return df


@pytest.fixture()
def synthetic_train_val_test(
    synthetic_timeseries_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the synthetic time series into train / val / test (60/20/20).

    Chronological split — no shuffling.
    """
    df = synthetic_timeseries_df
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Minimal valid config dicts
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_data_config() -> dict:
    """Return a minimal valid data config dict (all fields resolved)."""
    return {
        "source": "synthetic",
        "source_type": "synthetic_test",
        "version": "0.0.0",
        "target_location": "test_city",
        "latitude": 28.6,
        "longitude": 77.2,
        "variables": list(SYNTHETIC_FEATURE_NAMES),
        "target_variables": list(SYNTHETIC_FEATURE_NAMES),
        "perturbable_features": ["temperature_2m", "relative_humidity"],
        "sampling_interval": "6h",
        "time_range": {"start": "2020-01-01", "end": "2020-02-18"},
        "split": {
            "train_end": "2020-01-25",
            "val_end": "2020-02-08",
            "test_end": "2020-02-18",
        },
        "input_window": 24,
        "scaler_type": "StandardScaler",
    }


@pytest.fixture()
def minimal_lstm_config() -> dict:
    """Return a minimal valid LSTM config dict."""
    return {
        "n_features": SYNTHETIC_N_FEATURES,
        "n_targets": SYNTHETIC_N_TARGETS,
        "input_window": 24,
        "hidden_dim": 32,
        "num_layers": 1,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "batch_size": 16,
        "epochs": 2,
        "early_stopping": {"patience": 5, "metric": "val_loss"},
        "seed": 42,
        "device": "cpu",
        "checkpoint_dir": "results/checkpoints",
        "metrics_dir": "results/metrics",
        "predictions_dir": "results/predictions",
    }


@pytest.fixture()
def minimal_transformer_config() -> dict:
    """Return a minimal valid Transformer config dict."""
    return {
        "n_features": SYNTHETIC_N_FEATURES,
        "n_targets": SYNTHETIC_N_TARGETS,
        "d_model": 64,
        "nhead": 4,
        "num_encoder_layers": 2,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "batch_size": 16,
        "epochs": 2,
        "early_stopping": {"patience": 5, "metric": "val_loss"},
        "seed": 42,
        "device": "cpu",
        "checkpoint_dir": "results/checkpoints",
        "metrics_dir": "results/metrics",
        "predictions_dir": "results/predictions",
    }


# ---------------------------------------------------------------------------
# Temporary directory
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_results(tmp_path: Path) -> Path:
    """Create and return a temporary results directory tree."""
    for subdir in ("checkpoints", "metrics", "manifests", "predictions", "figures"):
        (tmp_path / subdir).mkdir()
    return tmp_path
