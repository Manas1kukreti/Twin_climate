"""SQLite storage for the multi-city ClimateTwin map.

The Folium map needs, per city, a compact snapshot it can render quickly:
its location, the latest "current" state, how anomalous that state is versus
the city's own climatology (a z-score, coloured on the map), and how uncertain
the twin's short forecast is (band width, shown via marker opacity/size).

Rather than recompute all of that from checkpoints and CSVs on every page load,
we materialize it once into a small SQLite database. This mirrors the
"precompute into a DB, render from the DB" pattern and keeps the dashboard
responsive.

Schema
------
cities        : one row per city (name, slug, lat/lon, climate zone, data source)
climatology   : per-city, per-variable training-period mean/std (anomaly basis)
current_state : per-city latest observed state + per-variable anomaly z-score
forecast      : per-city short-horizon forecast summary + uncertainty band width

All numeric climate values are in physical units (°C, hPa, mm, m/s). The DB is
built by :func:`build_database`, invoked from ``scripts/build_map_db.py``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "climatetwin.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS cities (
    slug          TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    latitude      REAL NOT NULL,
    longitude     REAL NOT NULL,
    climate_zone  TEXT,
    is_primary    INTEGER NOT NULL DEFAULT 0,
    data_source   TEXT NOT NULL DEFAULT 'ERA5-Land',
    synthetic     INTEGER NOT NULL DEFAULT 0
);

-- Climatology is stored per (month, hour-of-day) so anomalies compare like
-- with like. A flat annual mean would make any winter-midnight observation
-- look extremely cold simply because it is being compared against summer
-- afternoons; month+hour matching removes both the seasonal and the diurnal
-- cycle before computing the anomaly.
CREATE TABLE IF NOT EXISTS climatology (
    slug      TEXT NOT NULL,
    variable  TEXT NOT NULL,
    month     INTEGER NOT NULL,   -- 1-12
    hour      INTEGER NOT NULL,   -- 0-23
    mean      REAL NOT NULL,
    std       REAL NOT NULL,
    n_samples INTEGER NOT NULL,
    unit      TEXT,
    PRIMARY KEY (slug, variable, month, hour),
    FOREIGN KEY (slug) REFERENCES cities(slug)
);

CREATE TABLE IF NOT EXISTS current_state (
    slug          TEXT NOT NULL,
    variable      TEXT NOT NULL,
    value         REAL NOT NULL,
    anomaly_z     REAL NOT NULL,
    unit          TEXT,
    timestamp     TEXT,
    PRIMARY KEY (slug, variable),
    FOREIGN KEY (slug) REFERENCES cities(slug)
);

CREATE TABLE IF NOT EXISTS forecast (
    slug            TEXT NOT NULL,
    variable        TEXT NOT NULL,
    horizon_hours   INTEGER NOT NULL,
    mean_value      REAL NOT NULL,
    band_width      REAL NOT NULL,
    unit            TEXT,
    PRIMARY KEY (slug, variable, horizon_hours),
    FOREIGN KEY (slug) REFERENCES cities(slug)
);

-- Live observations from a real-time weather API (Open-Meteo). ERA5-Land is a
-- reanalysis product with days-to-months latency and therefore CANNOT supply
-- "now"; this table is the separate real-time path, while ERA5-Land remains
-- the training and climatology backbone.
CREATE TABLE IF NOT EXISTS live_observations (
    slug          TEXT NOT NULL,
    variable      TEXT NOT NULL,
    value         REAL NOT NULL,
    anomaly_z     REAL,
    unit          TEXT,
    observed_at   TEXT NOT NULL,   -- ISO timestamp of the observation
    fetched_at    TEXT NOT NULL,   -- ISO timestamp of our retrieval
    source        TEXT NOT NULL DEFAULT 'open-meteo',
    PRIMARY KEY (slug, variable),
    FOREIGN KEY (slug) REFERENCES cities(slug)
);

CREATE TABLE IF NOT EXISTS meta (
    key    TEXT PRIMARY KEY,
    value  TEXT
);
"""


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with row access by column name."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they do not exist."""
    conn.executescript(SCHEMA)
    conn.commit()


def _synthetic_flag(slug: str) -> bool:
    """Read the per-city provenance to see if the data is synthetic."""
    prov = PROJECT_ROOT / "data" / "metadata" / f"provenance_{slug}.json"
    if not prov.exists():
        return False
    try:
        with open(prov, encoding="utf-8") as fh:
            return bool(json.load(fh).get("synthetic", False))
    except (json.JSONDecodeError, OSError):
        return False


def upsert_city(conn: sqlite3.Connection, city: Any, synthetic: bool) -> None:
    """Insert or replace a city row."""
    conn.execute(
        "INSERT OR REPLACE INTO cities "
        "(slug, name, latitude, longitude, climate_zone, is_primary, data_source, synthetic) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (city.slug, city.name, city.latitude, city.longitude, city.climate_zone,
         int(city.is_primary), "ERA5-Land", int(synthetic)),
    )


def store_climatology(
    conn: sqlite3.Connection,
    slug: str,
    stats: dict[tuple[str, int, int], dict[str, float]],
    units: dict[str, str],
) -> None:
    """Store per-(variable, month, hour) climatological normals.

    ``stats`` is keyed by ``(variable, month, hour)`` and each value carries
    ``mean``, ``std`` and ``n``.
    """
    rows = [
        (slug, var, int(month), int(hour), float(s["mean"]), float(s["std"]),
         int(s["n"]), units.get(var, ""))
        for (var, month, hour), s in stats.items()
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO climatology "
        "(slug, variable, month, hour, mean, std, n_samples, unit) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def store_current_state(conn: sqlite3.Connection, slug: str,
                        state: dict[str, dict[str, Any]], units: dict[str, str]) -> None:
    for var, s in state.items():
        conn.execute(
            "INSERT OR REPLACE INTO current_state "
            "(slug, variable, value, anomaly_z, unit, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (slug, var, float(s["value"]), float(s["anomaly_z"]),
             units.get(var, ""), s.get("timestamp")),
        )


def store_forecast(conn: sqlite3.Connection, slug: str,
                   rows: list[dict[str, Any]], units: dict[str, str]) -> None:
    for r in rows:
        conn.execute(
            "INSERT OR REPLACE INTO forecast "
            "(slug, variable, horizon_hours, mean_value, band_width, unit) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (slug, r["variable"], int(r["horizon_hours"]), float(r["mean_value"]),
             float(r["band_width"]), units.get(r["variable"], "")),
        )


def store_live_observations(
    conn: sqlite3.Connection,
    slug: str,
    observations: dict[str, dict],
    units: dict[str, str],
    source: str = "open-meteo",
) -> None:
    """Store live (real-time) observations for a city.

    ``observations`` maps variable -> ``{value, anomaly_z, observed_at,
    fetched_at}``.
    """
    rows = [
        (slug, var, float(o["value"]),
         (float(o["anomaly_z"]) if o.get("anomaly_z") is not None else None),
         units.get(var, ""), o["observed_at"], o["fetched_at"], source)
        for var, o in observations.items()
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO live_observations "
        "(slug, variable, value, anomaly_z, unit, observed_at, fetched_at, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def get_normal(conn: sqlite3.Connection, slug: str, variable: str,
               month: int, hour: int) -> tuple[float, float] | None:
    """Return ``(mean, std)`` climatological normal for a (month, hour) cell."""
    row = conn.execute(
        "SELECT mean, std FROM climatology "
        "WHERE slug=? AND variable=? AND month=? AND hour=?",
        (slug, variable, int(month), int(hour)),
    ).fetchone()
    if row is None:
        return None
    return float(row["mean"]), float(row["std"])


def load_live_rows(db_path: str | Path = DEFAULT_DB_PATH) -> list[dict]:
    """Return live observations per city, or an empty list if none stored."""
    conn = connect(db_path)
    try:
        cities = conn.execute("SELECT * FROM cities").fetchall()
        out: list[dict] = []
        for c in cities:
            slug = c["slug"]
            t2m = conn.execute(
                "SELECT value, anomaly_z, unit, observed_at, fetched_at, source "
                "FROM live_observations WHERE slug=? AND variable='t2m'", (slug,)
            ).fetchone()
            if t2m is None:
                continue
            allvars = conn.execute(
                "SELECT variable, value, anomaly_z, unit FROM live_observations "
                "WHERE slug=?", (slug,)
            ).fetchall()
            out.append({
                "slug": slug,
                "name": c["name"],
                "latitude": c["latitude"],
                "longitude": c["longitude"],
                "climate_zone": c["climate_zone"],
                "t2m_value": t2m["value"],
                "t2m_anomaly_z": t2m["anomaly_z"],
                "t2m_unit": t2m["unit"],
                "observed_at": t2m["observed_at"],
                "fetched_at": t2m["fetched_at"],
                "source": t2m["source"],
                "variables": {r["variable"]: dict(r) for r in allvars},
            })
        return out
    finally:
        conn.close()


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))


# ---------------------------------------------------------------------------
# Read helpers used by the dashboard map
# ---------------------------------------------------------------------------
def load_map_rows(db_path: str | Path = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Return one dict per city with location + the headline t2m anomaly and
    a mean forecast uncertainty, ready for the map layer.
    """
    conn = connect(db_path)
    try:
        cities = conn.execute("SELECT * FROM cities").fetchall()
        rows: list[dict[str, Any]] = []
        for c in cities:
            slug = c["slug"]
            t2m = conn.execute(
                "SELECT value, anomaly_z, unit, timestamp FROM current_state "
                "WHERE slug=? AND variable='t2m'", (slug,)
            ).fetchone()
            # Mean forecast band width across variables at the max horizon =
            # a single "how uncertain is this city's outlook" scalar.
            unc = conn.execute(
                "SELECT AVG(band_width) AS mean_band FROM forecast "
                "WHERE slug=? AND horizon_hours=(SELECT MAX(horizon_hours) FROM forecast WHERE slug=?)",
                (slug, slug),
            ).fetchone()
            rows.append({
                "slug": slug,
                "name": c["name"],
                "latitude": c["latitude"],
                "longitude": c["longitude"],
                "climate_zone": c["climate_zone"],
                "is_primary": bool(c["is_primary"]),
                "synthetic": bool(c["synthetic"]),
                "t2m_value": t2m["value"] if t2m else None,
                "t2m_anomaly_z": t2m["anomaly_z"] if t2m else None,
                "t2m_unit": t2m["unit"] if t2m else "",
                "timestamp": t2m["timestamp"] if t2m else None,
                "forecast_uncertainty": unc["mean_band"] if unc and unc["mean_band"] is not None else None,
            })
        return rows
    finally:
        conn.close()


def load_city_detail(slug: str, db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Return full current-state + forecast detail for one city (for popups)."""
    conn = connect(db_path)
    try:
        city = conn.execute("SELECT * FROM cities WHERE slug=?", (slug,)).fetchone()
        current = conn.execute(
            "SELECT * FROM current_state WHERE slug=?", (slug,)
        ).fetchall()
        forecast = conn.execute(
            "SELECT * FROM forecast WHERE slug=? ORDER BY variable, horizon_hours", (slug,)
        ).fetchall()
        return {
            "city": dict(city) if city else {},
            "current_state": [dict(r) for r in current],
            "forecast": [dict(r) for r in forecast],
        }
    finally:
        conn.close()


def get_meta(db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, str]:
    conn = connect(db_path)
    try:
        return {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM meta").fetchall()}
    finally:
        conn.close()
