"""Logging configuration for ClimateTwin scripts and modules.

Provides a single ``setup_logging`` function that configures the Python
logging system with timestamps, levels, and module names.

Usage
-----
>>> from src.logging_config import setup_logging
>>> setup_logging()           # INFO to console
>>> setup_logging("DEBUG")    # DEBUG to console
>>> setup_logging(log_file="results/run.log")  # also write to file
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
) -> None:
    """Configure root logger for ClimateTwin.

    Parameters
    ----------
    level : str
        Logging level name (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, etc.).
    log_file : str or Path, optional
        If provided, also write log output to this file (append mode).
        Parent directories are created automatically.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=handlers,
        force=True,
    )

    logging.getLogger(__name__).debug("Logging configured: level=%s, file=%s", level, log_file)
