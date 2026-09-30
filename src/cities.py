"""Multi-city registry and per-city artifact-path resolution.

The original ClimateTwin pipeline was single-city (Delhi) with artifact paths
hard-coded as literals (``data/processed/train.csv``,
``results/checkpoints/lstm_delhi_seed42_001.pt``, ...). This module centralizes
the mapping from a *city* to its data and artifact locations so the same
preprocessing / training / dashboard code can serve many cities.

Backward-compatibility contract
-------------------------------
The **primary** city (Delhi, ``is_primary: true`` in ``configs/cities.yaml``)
keeps its original, un-namespaced paths so everything already built
(the trained Delhi checkpoint, the dashboard's default view, existing tests)
continues to work unchanged. Every *other* city is namespaced under a
per-city subdirectory / filename so artifacts never collide.

Paths for the primary city:
    data/processed/{train,val,test}[_scaled].csv, scaler.joblib
    results/checkpoints/lstm_delhi_seed42_001.pt
    results/metrics/lstm_delhi_metrics.json
    results/predictions/lstm_delhi_predictions.csv

Paths for a non-primary city ``<slug>``:
    data/processed/<slug>/{train,val,test}[_scaled].csv, scaler.joblib
    results/checkpoints/lstm_<slug>_seed42_001.pt
    results/metrics/lstm_<slug>_metrics.json
    results/predictions/lstm_<slug>_predictions.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CITIES_CONFIG_PATH = PROJECT_ROOT / "configs" / "cities.yaml"

PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
CHECKPOINTS_ROOT = PROJECT_ROOT / "results" / "checkpoints"
METRICS_ROOT = PROJECT_ROOT / "results" / "metrics"
PREDICTIONS_ROOT = PROJECT_ROOT / "results" / "predictions"
MANIFESTS_ROOT = PROJECT_ROOT / "results" / "manifests"


@dataclass(frozen=True)
class City:
    """A single city and everything needed to locate its artifacts."""

    name: str
    slug: str
    latitude: float
    longitude: float
    climate_zone: str
    is_primary: bool

    # --- Processed data locations ---
    @property
    def processed_dir(self) -> Path:
        """Directory holding this city's processed splits + scaler."""
        return PROCESSED_ROOT if self.is_primary else PROCESSED_ROOT / self.slug

    def split_csv(self, split: str, scaled: bool) -> Path:
        """Path to a processed split CSV (split in {'train','val','test'})."""
        suffix = "_scaled" if scaled else ""
        return self.processed_dir / f"{split}{suffix}.csv"

    @property
    def scaler_path(self) -> Path:
        return self.processed_dir / "scaler.joblib"

    # --- Trained-artifact locations ---
    @property
    def run_id(self) -> str:
        return f"lstm_scratch_{self.slug}_seed42_001"

    @property
    def checkpoint_path(self) -> Path:
        return CHECKPOINTS_ROOT / f"lstm_{self.slug}_seed42_001.pt"

    @property
    def metrics_path(self) -> Path:
        return METRICS_ROOT / f"lstm_{self.slug}_metrics.json"

    @property
    def training_log_path(self) -> Path:
        return METRICS_ROOT / f"lstm_{self.slug}_training_log.json"

    @property
    def predictions_path(self) -> Path:
        return PREDICTIONS_ROOT / f"lstm_{self.slug}_predictions.csv"

    @property
    def manifest_path(self) -> Path:
        return MANIFESTS_ROOT / f"{self.run_id}.json"

    def artifacts_exist(self) -> bool:
        """True if this city has a trained checkpoint, scaler and test split."""
        return (
            self.checkpoint_path.exists()
            and self.scaler_path.exists()
            and self.split_csv("test", scaled=True).exists()
        )


def load_cities(config_path: str | Path = CITIES_CONFIG_PATH) -> list[City]:
    """Load the city registry from ``configs/cities.yaml``."""
    cfg = load_config(config_path)
    cities: list[City] = []
    for entry in cfg["cities"]:
        cities.append(
            City(
                name=entry["name"],
                slug=entry["slug"],
                latitude=float(entry["latitude"]),
                longitude=float(entry["longitude"]),
                climate_zone=entry.get("climate_zone", ""),
                is_primary=bool(entry.get("is_primary", False)),
            )
        )
    if not cities:
        raise ValueError(f"No cities defined in {config_path}.")
    return cities


def city_map(config_path: str | Path = CITIES_CONFIG_PATH) -> dict[str, City]:
    """Return a slug -> City mapping."""
    return {c.slug: c for c in load_cities(config_path)}


def get_city(slug: str, config_path: str | Path = CITIES_CONFIG_PATH) -> City:
    """Look up a single city by slug (case-insensitive)."""
    cities = city_map(config_path)
    key = slug.lower()
    if key not in cities:
        raise KeyError(
            f"Unknown city slug '{slug}'. Known: {sorted(cities)}."
        )
    return cities[key]


def primary_city(config_path: str | Path = CITIES_CONFIG_PATH) -> City:
    """Return the primary city (Delhi), or the first city if none is flagged."""
    cities = load_cities(config_path)
    for c in cities:
        if c.is_primary:
            return c
    return cities[0]
