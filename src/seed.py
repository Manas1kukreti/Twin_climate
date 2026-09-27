"""Central seed utility for reproducible experiments.

Seeds Python stdlib ``random``, NumPy, and PyTorch (CPU + CUDA when
available).  Sets deterministic flags where practical.

Usage
-----
>>> from src.seed import set_global_seed
>>> set_global_seed(42)
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Seed all relevant RNGs for reproducibility.

    Seeds:
        - ``random`` (Python stdlib)
        - ``numpy.random``
        - ``torch`` (CPU and CUDA if available)

    Also sets:
        - ``torch.backends.cudnn.deterministic = True``
        - ``torch.backends.cudnn.benchmark = False``
        - ``PYTHONHASHSEED`` environment variable

    Parameters
    ----------
    seed : int
        Non-negative integer seed value.

    Raises
    ------
    ValueError
        If *seed* is negative.

    Note
    ----
    Perfect deterministic reproducibility across all hardware and
    platform combinations is not guaranteed.  The goal is controlled,
    traceable, repeatable experimentation.
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)  # noqa: NPY002  — legacy API, intentional for global seeding

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Global seed set to %d", seed)
