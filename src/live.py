"""Real-time observation ingestion — the "live" half of the digital twin.

Why a second data source is necessary
-------------------------------------
ERA5-Land is a **reanalysis** product: it is assembled after the fact and
published with a latency of roughly five days to two months. It therefore
*cannot* provide "now", no matter how it is queried. The dashboard showing
December 2024 was not a bug — that is the end of the ERA5-Land test split.

A digital twin that "ingests real-time observations" needs a genuinely live
feed. This module adds one, while ERA5-Land remains the training and
climatology backbone:

    ERA5-Land (reanalysis) --> model training + climatological normals
    Open-Meteo (live)      --> current state, anomaly vs those normals

Data source
-----------
`Open-Meteo <https://open-meteo.com/>`_ — free for non-commercial use, no API
key or signup required, CC-BY 4.0 attribution. It serves current conditions and
short-range forecasts from national weather-service models.

.. note::
   This module makes outbound HTTPS requests to ``api.open-meteo.com``. Only a
   latitude/longitude and variable list are sent; no project data leaves the
   machine.

Variable mapping
----------------
Open-Meteo's fields are mapped onto the project's six ERA5-Land short names so
live observations are directly comparable with the model's feature space:

===========  ==========================  ===============================
Ours         Open-Meteo field            Conversion
===========  ==========================  ===============================
``t2m``      ``temperature_2m``          none (deg C)
``d2m``      ``dew_point_2m``            none (deg C)
``sp``       ``surface_pressure``        none (hPa)
``tp``       ``precipitation``           none (mm, last hour)
``u10``      ``wind_speed_10m`` + dir    vector decomposition -> m/s
``v10``      ``wind_speed_10m`` + dir    vector decomposition -> m/s
===========  ==========================  ===============================
"""

from __future__ import annotations

import json
import logging
import math
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
ATTRIBUTION = "Live weather data by Open-Meteo.com (CC-BY 4.0)"

# Open-Meteo "current" fields we request.
_CURRENT_FIELDS = [
    "temperature_2m",
    "dew_point_2m",
    "surface_pressure",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
]

FEATURE_ORDER = ["t2m", "d2m", "sp", "tp", "u10", "v10"]


class LiveFetchError(RuntimeError):
    """Raised when live observations cannot be retrieved."""


@dataclass
class LiveObservation:
    """A single city's live observation mapped to project variables."""

    slug: str
    name: str
    latitude: float
    longitude: float
    observed_at: str          # ISO timestamp reported by the provider
    fetched_at: str           # ISO timestamp of our retrieval
    values: dict[str, float]  # variable -> physical-unit value
    source: str = "open-meteo"


def _wind_components(speed_ms: float, direction_deg: float) -> tuple[float, float]:
    """Decompose wind speed/direction into u (east) and v (north) components.

    Meteorological convention: direction is the bearing the wind blows *from*,
    so the vector points the opposite way — hence the negative signs.
    """
    rad = math.radians(direction_deg)
    u = -speed_ms * math.sin(rad)
    v = -speed_ms * math.cos(rad)
    return u, v


def fetch_live(
    slug: str,
    name: str,
    latitude: float,
    longitude: float,
    timeout: float = 20.0,
) -> LiveObservation:
    """Fetch current conditions for one location from Open-Meteo.

    Raises
    ------
    LiveFetchError
        On network failure, malformed response, or missing fields.
    """
    params = {
        "latitude": f"{latitude:.4f}",
        "longitude": f"{longitude:.4f}",
        "current": ",".join(_CURRENT_FIELDS),
        "wind_speed_unit": "ms",      # metres/second, matching ERA5-Land
        "timezone": "UTC",
    }
    url = f"{OPEN_METEO_URL}?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LiveFetchError(f"Open-Meteo request failed for {slug}: {exc}") from exc

    current = payload.get("current")
    if not isinstance(current, dict):
        raise LiveFetchError(f"Open-Meteo response missing 'current' for {slug}.")

    missing = [f for f in _CURRENT_FIELDS if f not in current]
    if missing:
        raise LiveFetchError(f"Open-Meteo response missing fields {missing} for {slug}.")

    speed = float(current["wind_speed_10m"])
    direction = float(current["wind_direction_10m"])
    u10, v10 = _wind_components(speed, direction)

    values = {
        "t2m": float(current["temperature_2m"]),
        "d2m": float(current["dew_point_2m"]),
        "sp": float(current["surface_pressure"]),
        "tp": float(current["precipitation"]),
        "u10": u10,
        "v10": v10,
    }

    return LiveObservation(
        slug=slug,
        name=name,
        latitude=latitude,
        longitude=longitude,
        observed_at=str(current.get("time", "")),
        fetched_at=datetime.now(UTC).isoformat(),
        values=values,
    )


def fetch_all(cities: list, timeout: float = 20.0) -> tuple[list[LiveObservation], list[str]]:
    """Fetch live observations for many cities.

    Returns ``(observations, errors)`` so a single city's failure does not abort
    the whole refresh.
    """
    obs: list[LiveObservation] = []
    errors: list[str] = []
    for city in cities:
        try:
            obs.append(
                fetch_live(city.slug, city.name, city.latitude, city.longitude, timeout)
            )
            logger.info("[%s] live observation fetched", city.slug)
        except LiveFetchError as exc:
            logger.warning("[%s] live fetch failed: %s", city.slug, exc)
            errors.append(f"{city.slug}: {exc}")
    return obs, errors
