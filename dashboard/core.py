"""Shared loaders, caching, and helpers for the ClimateTwin dashboard.

Everything here is read-only with respect to the project's artifacts. Heavy
objects (model, scaler, data, metrics) are loaded once and cached by Streamlit
so page switches and widget interactions stay responsive.

Design notes
------------
- Paths are resolved relative to the project root (the parent of ``dashboard/``)
  so the app works regardless of the current working directory.
- The feature order is taken from ``configs/data.yaml`` and cross-checked
  against the fitted scaler, exactly as ``scripts/scenario_demo.py`` does.
- No retraining ever happens here; the trained checkpoint is loaded with
  ``weights_only=True``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cities import City, load_cities, primary_city  # noqa: E402
from src.config import load_config  # noqa: E402
from src.models.lstm import ClimateLSTM  # noqa: E402
from src.preprocessing import load_scaler  # noqa: E402

# --- Config locations (shared across cities) ---
DATA_CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"
LSTM_CONFIG_PATH = PROJECT_ROOT / "configs" / "lstm.yaml"

# Human-readable names for the six climate variables.
VARIABLE_LABELS: dict[str, str] = {
    "t2m": "Air temperature (2 m)",
    "d2m": "Dewpoint (2 m)",
    "sp": "Surface pressure",
    "tp": "Precipitation",
    "u10": "Wind (u, east-west)",
    "v10": "Wind (v, north-south)",
}

# Colour used for alert cards, keyed by IMD-style alert level.
ALERT_COLORS: dict[str, str] = {
    "red": "#c0392b",
    "orange": "#e67e22",
    "yellow": "#f1c40f",
    "none": "#2ecc71",
}


@dataclass
class TwinArtifacts:
    """Everything the dashboard needs for one city, loaded once and cached."""

    model: ClimateLSTM
    scaler: Any
    feature_order: list[str]
    units: dict[str, str]
    input_window: int
    perturbable: list[str]
    test_physical: pd.DataFrame
    test_scaled: np.ndarray
    timestamps: pd.Series
    metrics: dict[str, Any]
    location: str
    slug: str


def available_cities() -> list[City]:
    """Cities that have complete trained artifacts, ready to load in the UI."""
    return [c for c in load_cities() if c.artifacts_exist()]


def artifacts_available() -> tuple[bool, list[str]]:
    """Check that at least the primary city's artifacts exist.

    Returns ``(ok, missing)`` where ``missing`` lists absent files for the
    primary city so the boot screen can show a helpful message.
    """
    delhi = primary_city()
    required = {
        "configs/data.yaml": DATA_CONFIG_PATH,
        "configs/lstm.yaml": LSTM_CONFIG_PATH,
        str(delhi.scaler_path.relative_to(PROJECT_ROOT)): delhi.scaler_path,
        str(delhi.checkpoint_path.relative_to(PROJECT_ROOT)): delhi.checkpoint_path,
        str(delhi.split_csv("test", False).relative_to(PROJECT_ROOT)): delhi.split_csv("test", False),
        str(delhi.split_csv("test", True).relative_to(PROJECT_ROOT)): delhi.split_csv("test", True),
        str(delhi.metrics_path.relative_to(PROJECT_ROOT)): delhi.metrics_path,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    return (len(missing) == 0, missing)


@st.cache_resource(show_spinner="Loading ClimateTwin model and data...")
def load_artifacts(slug: str | None = None) -> TwinArtifacts:
    """Load and cache one city's trained artifacts.

    Cached per-slug with ``st.cache_resource`` so each city's model/scaler/data
    are read from disk only once per server process. ``slug=None`` loads the
    primary city (Delhi). Validates the scaler/feature-order agreement.
    """
    import json

    from src.cities import get_city
    from src.scenario import get_perturbable_features

    city = primary_city() if slug is None else get_city(slug)

    data_cfg = load_config(DATA_CONFIG_PATH)
    model_cfg = load_config(LSTM_CONFIG_PATH)

    feature_order: list[str] = data_cfg["feature_order"]
    units: dict[str, str] = data_cfg["converted_units"]
    input_window: int = int(model_cfg["input_window"])

    perturbable = get_perturbable_features(
        feature_order, data_cfg.get("perturbable_features", [])
    )

    scaler, scaler_features, _ = load_scaler(city.scaler_path)
    if scaler_features != feature_order:
        raise ValueError(
            "Scaler feature order does not match config feature_order: "
            f"{scaler_features} != {feature_order}"
        )

    model = ClimateLSTM(
        n_features=int(model_cfg["n_features"]),
        n_targets=int(model_cfg["n_targets"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
    )
    model.load_state_dict(torch.load(city.checkpoint_path, weights_only=True))
    model.eval()

    test_physical = pd.read_csv(city.split_csv("test", scaled=False))
    test_scaled_df = pd.read_csv(city.split_csv("test", scaled=True))
    timestamps = pd.to_datetime(test_physical["timestamp"])
    test_scaled = test_scaled_df[feature_order].to_numpy(dtype=np.float32)

    with open(city.metrics_path, encoding="utf-8") as fh:
        metrics = json.load(fh)

    return TwinArtifacts(
        model=model,
        scaler=scaler,
        feature_order=feature_order,
        units=units,
        input_window=input_window,
        perturbable=perturbable,
        test_physical=test_physical,
        test_scaled=test_scaled,
        timestamps=timestamps,
        metrics=metrics,
        location=city.name,
        slug=city.slug,
    )


def window_at(art: TwinArtifacts, start_index: int) -> np.ndarray:
    """Return the scaled input window starting at ``start_index``.

    Shape is ``(input_window, n_features)``. Raises if the window would run
    past the end of the test split.
    """
    end = start_index + art.input_window
    if start_index < 0 or end > len(art.test_scaled):
        raise IndexError(
            f"Window [{start_index}:{end}] out of range for test split of "
            f"length {len(art.test_scaled)}."
        )
    return art.test_scaled[start_index:end]


def window_end_timestamp(art: TwinArtifacts, start_index: int) -> pd.Timestamp:
    """Timestamp of the last hour in the input window (the 'now' state)."""
    return art.timestamps.iloc[start_index + art.input_window - 1]


def max_window_start(art: TwinArtifacts) -> int:
    """Largest valid window start index in the test split."""
    return len(art.test_scaled) - art.input_window - 1


def feature_label(name: str, units: dict[str, str]) -> str:
    """Pretty label for a feature including its unit, e.g. 'Air temperature (°C)'."""
    base = VARIABLE_LABELS.get(name, name)
    unit = units.get(name, "")
    return f"{base} ({unit})" if unit else base
