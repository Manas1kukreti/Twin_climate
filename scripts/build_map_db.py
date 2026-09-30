"""Build the SQLite map database from trained per-city artifacts.

For every city with trained artifacts (checkpoint + scaler + test split), this:

1. Loads the LSTM checkpoint and fitted scaler.
2. Computes the training-period climatology (per-variable mean/std) as the
   anomaly reference.
3. Takes the most recent real test window as the city's "current state" and
   expresses each variable as a z-score anomaly vs its climatology.
4. Runs a short MC-dropout forecast (via ``src.scenario.run_scenario`` with a
   zero perturbation) to get a predictive band, and records the band width per
   variable at the final horizon as the "uncertainty".
5. Writes everything to ``data/climatetwin.db``.

Reads artifacts only; trains nothing.

Usage
-----
    python scripts/build_map_db.py
"""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import db as dbmod  # noqa: E402
from src.cities import City, load_cities  # noqa: E402
from src.config import load_config  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402
from src.models.lstm import ClimateLSTM  # noqa: E402
from src.preprocessing import load_scaler  # noqa: E402
from src.scenario import run_scenario  # noqa: E402

logger = logging.getLogger(__name__)

HORIZON = 12
MC_SAMPLES = 40
SEED = 42


def _load_model(city: City, model_cfg: dict) -> ClimateLSTM:
    model = ClimateLSTM(
        n_features=int(model_cfg["n_features"]),
        n_targets=int(model_cfg["n_targets"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
    )
    model.load_state_dict(torch.load(city.checkpoint_path, weights_only=True))
    model.eval()
    return model


def build_city(conn, city: City, feature_order: list[str], units: dict[str, str],
               input_window: int, model_cfg: dict) -> None:
    """Compute and store all DB rows for one city."""
    model = _load_model(city, model_cfg)
    scaler, scaler_features, _ = load_scaler(city.scaler_path)
    assert scaler_features == feature_order, f"{city.slug}: scaler feature mismatch"

    # --- Climatology from the training split, per (month, hour-of-day) ---
    # Matching month AND hour removes the seasonal and diurnal cycles before
    # computing an anomaly. A flat annual mean would make any winter-midnight
    # observation look extremely cold purely because it is compared against
    # summer afternoons.
    train_phys = pd.read_csv(city.split_csv("train", scaled=False),
                             parse_dates=["timestamp"])
    train_phys["_month"] = train_phys["timestamp"].dt.month
    train_phys["_hour"] = train_phys["timestamp"].dt.hour

    clim: dict[tuple[str, int, int], dict[str, float]] = {}
    grouped = train_phys.groupby(["_month", "_hour"])
    for (month, hour), grp in grouped:
        for var in feature_order:
            col = grp[var].to_numpy(dtype=float)
            std = float(np.std(col))
            clim[(var, int(month), int(hour))] = {
                "mean": float(np.mean(col)),
                # Guard against a degenerate zero std (would divide by zero).
                "std": std if std > 1e-6 else 1.0,
                "n": int(col.size),
            }

    def _z(var: str, value: float, month: int, hour: int) -> float:
        """Anomaly z-score against the matching (month, hour) normal."""
        ref = clim.get((var, month, hour))
        if ref is None:
            return 0.0
        return (value - ref["mean"]) / ref["std"]

    # --- Current state = last real test window's final row (physical units) ---
    test_phys = pd.read_csv(city.split_csv("test", scaled=False), parse_dates=["timestamp"])
    test_scaled = pd.read_csv(city.split_csv("test", scaled=True))
    now_row = test_phys.iloc[-1]
    now_dt = pd.Timestamp(now_row["timestamp"])
    now_ts = str(now_dt)
    current: dict[str, dict] = {}
    for var in feature_order:
        val = float(now_row[var])
        current[var] = {
            "value": val,
            "anomaly_z": _z(var, val, now_dt.month, now_dt.hour),
            "timestamp": now_ts,
        }

    # --- Short MC-dropout forecast for uncertainty (band width at max horizon) ---
    window = test_scaled[feature_order].to_numpy(dtype=np.float32)[-input_window:]
    result = run_scenario(
        model=model,
        baseline_input=window,
        perturbations={feature_order[0]: 0.0},  # zero perturbation -> pure forecast
        horizon=HORIZON,
        feature_names=feature_order,
        scaler=scaler,
        mc_samples=MC_SAMPLES,
        interval=(5.0, 95.0),
        seed=SEED,
        sustained=False,
    )
    forecast_rows: list[dict] = []
    for i, var in enumerate(feature_order):
        band = 0.0
        if result.baseline_lower is not None:
            band = float(result.baseline_upper[-1, i] - result.baseline_lower[-1, i])
        forecast_rows.append({
            "variable": var,
            "horizon_hours": HORIZON,
            "mean_value": float(result.baseline[-1, i]),
            "band_width": band,
        })

    synthetic = dbmod._synthetic_flag(city.slug)
    dbmod.upsert_city(conn, city, synthetic)
    dbmod.store_climatology(conn, city.slug, clim, units)
    dbmod.store_current_state(conn, city.slug, current, units)
    dbmod.store_forecast(conn, city.slug, forecast_rows, units)
    conn.commit()

    logger.info("[%s] reanalysis state: t2m=%.1f%s (z=%+.2f vs %s-month/%sh normal)",
                city.slug, current["t2m"]["value"], units.get("t2m", ""),
                current["t2m"]["anomaly_z"], now_dt.month, now_dt.hour)

    return clim


def ingest_live(conn, cities: list[City], clims: dict[str, dict],
                feature_order: list[str], units: dict[str, str]) -> int:
    """Fetch live observations and store them with month/hour-matched anomalies."""
    from src.live import fetch_all

    observations, errors = fetch_all(cities)
    for obs in observations:
        clim = clims.get(obs.slug, {})
        # Use the observation's own month/hour for a like-for-like comparison.
        try:
            obs_dt = pd.Timestamp(obs.observed_at)
            month, hour = int(obs_dt.month), int(obs_dt.hour)
        except (ValueError, TypeError):
            month, hour = None, None

        rows: dict[str, dict] = {}
        for var, value in obs.values.items():
            z = None
            if month is not None:
                ref = clim.get((var, month, hour))
                if ref is not None:
                    z = (value - ref["mean"]) / ref["std"]
            rows[var] = {
                "value": value,
                "anomaly_z": z,
                "observed_at": obs.observed_at,
                "fetched_at": obs.fetched_at,
            }
        dbmod.store_live_observations(conn, obs.slug, rows, units, source=obs.source)
        t2m_z = rows["t2m"]["anomaly_z"]
        logger.info("[%s] LIVE t2m=%.1f%s (z=%s) at %s",
                    obs.slug, rows["t2m"]["value"], units.get("t2m", ""),
                    f"{t2m_z:+.2f}" if t2m_z is not None else "n/a",
                    obs.observed_at)
    conn.commit()

    for err in errors:
        logger.warning("live fetch error: %s", err)
    return len(observations)


def main(with_live: bool = True) -> None:
    setup_logging("INFO")
    data_cfg = load_config("configs/data.yaml")
    model_cfg = load_config("configs/lstm.yaml")
    feature_order = data_cfg["feature_order"]
    units = data_cfg["converted_units"]
    input_window = int(model_cfg["input_window"])

    conn = dbmod.connect()
    dbmod.init_schema(conn)

    built, skipped = [], []
    clims: dict[str, dict] = {}
    ready: list[City] = []
    for city in load_cities():
        if not city.artifacts_exist():
            logger.warning("[%s] no trained artifacts yet; skipping.", city.slug)
            skipped.append(city.slug)
            continue
        clims[city.slug] = build_city(
            conn, city, feature_order, units, input_window, model_cfg
        )
        built.append(city.slug)
        ready.append(city)

    n_live = 0
    if with_live and ready:
        logger.info("=== Fetching live observations (Open-Meteo) ===")
        n_live = ingest_live(conn, ready, clims, feature_order, units)

    dbmod.set_meta(conn, "built_at", datetime.now(UTC).isoformat())
    dbmod.set_meta(conn, "cities_built", ",".join(built))
    dbmod.set_meta(conn, "horizon_hours", str(HORIZON))
    dbmod.set_meta(conn, "mc_samples", str(MC_SAMPLES))
    dbmod.set_meta(conn, "live_cities", str(n_live))
    dbmod.set_meta(conn, "climatology_basis", "per (month, hour-of-day)")
    conn.commit()
    conn.close()

    logger.info("Map DB built at %s", dbmod.DEFAULT_DB_PATH)
    logger.info("Built: %s | Skipped: %s | Live: %d", built, skipped, n_live)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the ClimateTwin map DB")
    parser.add_argument("--no-live", action="store_true",
                        help="Skip the live Open-Meteo fetch (offline mode)")
    args = parser.parse_args()
    main(with_live=not args.no_live)
