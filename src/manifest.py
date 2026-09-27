"""Run-manifest creation, saving, and loading for ClimateTwin experiments.

Every serious training or evaluation run should produce a machine-readable
JSON manifest that links configuration, dataset metadata, model state,
checkpoint, metrics, and software versions.

Schema follows §2.5 of the ClimateTwin Master Specification.
"""

from __future__ import annotations

import json
import logging
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Software version helpers
# ---------------------------------------------------------------------------


def _get_software_versions() -> dict[str, str]:
    """Collect versions for Python and core dependencies.

    Returns best-effort strings; missing packages are recorded as
    ``"not installed"``.
    """
    versions: dict[str, str] = {
        "python": platform.python_version(),
    }

    for pkg, import_name in [
        ("pytorch", "torch"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
    ]:
        try:
            mod = __import__(import_name)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"

    return versions


def _get_git_commit() -> str | None:
    """Return the current short Git commit hash, or ``None`` if unavailable."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


# ---------------------------------------------------------------------------
# Manifest creation
# ---------------------------------------------------------------------------

# Required top-level keys in a valid manifest
_REQUIRED_KEYS = {"run_id", "stage", "model", "seed"}


def create_manifest(
    *,
    run_id: str,
    stage: str,
    model: str,
    seed: int,
    device: str = "cpu",
    dataset: dict[str, Any] | None = None,
    preprocessing: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
    hyperparameter_selection: dict[str, Any] | None = None,
    checkpoint_path: str | None = None,
    metrics_path: str | None = None,
    predictions_path: str | None = None,
    figures: list[str] | None = None,
) -> dict[str, Any]:
    """Build a run manifest dict conforming to the §2.5 schema.

    Parameters
    ----------
    run_id : str
        Stable identifier, e.g. ``"transformer_scratch_delhi_seed42_001"``.
    stage : str
        One of ``"baseline"``, ``"scratch"``, ``"pretrain"``, ``"finetune"``,
        ``"evaluate"``.
    model : str
        Model type, e.g. ``"persistence"``, ``"lstm"``, ``"transformer"``.
    seed : int
        Random seed used for the run.
    device : str
        Device string, e.g. ``"cpu"`` or ``"cuda"``.
    dataset : dict, optional
        Dataset metadata (source, location, variables, splits, etc.).
    preprocessing : dict, optional
        Scaler type, feature order, input window, etc.
    model_config : dict, optional
        Model-specific hyperparameters including ``n_features`` and
        ``n_targets``.
    training_config : dict, optional
        Training details: best epoch, wall-clock seconds, convergence info.
    hyperparameter_selection : dict, optional
        Record of candidates evaluated and selection rationale.
    checkpoint_path : str, optional
        Path to saved model checkpoint.
    metrics_path : str, optional
        Path to saved metrics JSON.
    predictions_path : str, optional
        Path to saved predictions.
    figures : list[str], optional
        Paths to generated figures.

    Returns
    -------
    dict[str, Any]
        Complete manifest dictionary.
    """
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "stage": stage,
        "model": model,
        "seed": seed,
        "device": device,
        "dataset": dataset or {},
        "preprocessing": preprocessing or {},
        "model_config": model_config or {},
        "training_config": training_config or {},
        "hyperparameter_selection": hyperparameter_selection or {},
        "checkpoint_path": checkpoint_path,
        "metrics_path": metrics_path,
        "predictions_path": predictions_path,
        "figures": figures or [],
        "software": _get_software_versions(),
        "git_commit": _get_git_commit(),
        "created_at": datetime.now(UTC).isoformat(),
    }

    logger.info("Created manifest for run_id=%s (stage=%s, model=%s)", run_id, stage, model)
    return manifest


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    """Write a manifest dict to a JSON file.

    Creates parent directories if needed.

    Parameters
    ----------
    manifest : dict
        Manifest dictionary (as returned by :func:`create_manifest`).
    path : str or Path
        Destination file path (should end in ``.json``).

    Returns
    -------
    Path
        The resolved path that was written.
    """
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    logger.info("Saved manifest to %s", filepath)
    return filepath


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a manifest JSON file.

    Parameters
    ----------
    path : str or Path
        Path to the manifest file.

    Returns
    -------
    dict[str, Any]
        Parsed manifest.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file is not valid JSON or is missing required keys.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Manifest file not found: {filepath}")

    with open(filepath, encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a JSON object, got {type(data).__name__}")

    missing = _REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Manifest is missing required keys: {sorted(missing)}")

    logger.info("Loaded manifest from %s (run_id=%s)", filepath, data.get("run_id"))
    return data
