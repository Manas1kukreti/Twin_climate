"""Configuration loading and validation for ClimateTwin experiments.

Reads YAML configuration files from ``configs/`` and provides typed
dictionaries with validation for required fields.

Usage
-----
>>> from src.config import load_config, validate_data_config
>>> cfg = load_config("configs/data.yaml")
>>> validate_data_config(cfg)  # raises on missing required fields
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic loader
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return its contents as a dict.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    yaml.YAMLError
        If the file contains invalid YAML.
    ValueError
        If the file is empty or does not parse to a dict.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if data is None:
        raise ValueError(f"Config file is empty: {filepath}")
    if not isinstance(data, dict):
        raise ValueError(f"Config file must parse to a dict, got {type(data).__name__}: {filepath}")

    logger.info("Loaded config from %s", filepath)
    return data


# ---------------------------------------------------------------------------
# Field-level helpers
# ---------------------------------------------------------------------------


def _check_required_fields(
    config: dict[str, Any],
    required: list[str],
    config_name: str,
) -> list[str]:
    """Return a list of error messages for missing or UNRESOLVED required fields."""
    errors: list[str] = []
    for field in required:
        val = config.get(field)
        if val is None:
            errors.append(f"[{config_name}] Required field '{field}' is null or missing.")
        elif isinstance(val, str) and val.upper().startswith("UNRESOLVED"):
            errors.append(f"[{config_name}] Field '{field}' is still UNRESOLVED: {val}")
    return errors


def _check_n_features_targets(
    config: dict[str, Any],
    config_name: str,
) -> list[str]:
    """Validate n_features / n_targets consistency when both are present."""
    errors: list[str] = []
    n_feat = config.get("n_features")
    n_targ = config.get("n_targets")

    if n_feat is not None and not isinstance(n_feat, int):
        errors.append(f"[{config_name}] 'n_features' must be an int, got {type(n_feat).__name__}.")
    if n_targ is not None and not isinstance(n_targ, int):
        errors.append(f"[{config_name}] 'n_targets' must be an int, got {type(n_targ).__name__}.")

    if isinstance(n_feat, int) and n_feat <= 0:
        errors.append(f"[{config_name}] 'n_features' must be positive, got {n_feat}.")
    if isinstance(n_targ, int) and n_targ <= 0:
        errors.append(f"[{config_name}] 'n_targets' must be positive, got {n_targ}.")
    if isinstance(n_feat, int) and isinstance(n_targ, int) and n_targ > n_feat:
        errors.append(
            f"[{config_name}] 'n_targets' ({n_targ}) cannot exceed 'n_features' ({n_feat})."
        )

    return errors


# ---------------------------------------------------------------------------
# Per-config validators
# ---------------------------------------------------------------------------


_DATA_REQUIRED = [
    "source",
    "source_type",
    "target_location",
    "sampling_interval",
]

_LSTM_REQUIRED = [
    "hidden_dim",
    "num_layers",
    "dropout",
    "learning_rate",
    "batch_size",
    "epochs",
    "seed",
]

_TRANSFORMER_REQUIRED = [
    "d_model",
    "nhead",
    "num_encoder_layers",
    "dropout",
    "learning_rate",
    "batch_size",
    "epochs",
    "seed",
]

_PRETRAIN_REQUIRED = [
    "pretrain_locations",
    "architecture",
    "learning_rate",
    "batch_size",
    "epochs",
    "seed",
]

_FINETUNE_REQUIRED = [
    "pretrain_checkpoint",
    "target_location",
    "learning_rate",
    "batch_size",
    "epochs",
    "seed",
]


def validate_data_config(config: dict[str, Any]) -> list[str]:
    """Validate data configuration and return a list of error messages.

    Returns an empty list when validation passes.
    """
    errors = _check_required_fields(config, _DATA_REQUIRED, "data")

    # variables list should be non-empty when resolved
    variables = config.get("variables")
    if isinstance(variables, list) and len(variables) == 0:
        errors.append("[data] 'variables' list is empty — must be populated after dataset selection.")

    return errors


def validate_model_config(config: dict[str, Any], model_type: str) -> list[str]:
    """Validate a model config (lstm / transformer / pretrain / finetune).

    Parameters
    ----------
    config : dict
        Parsed YAML config.
    model_type : str
        One of ``"lstm"``, ``"transformer"``, ``"pretrain"``, ``"finetune"``.

    Returns
    -------
    list[str]
        Error messages.  Empty when valid.
    """
    required_map = {
        "lstm": _LSTM_REQUIRED,
        "transformer": _TRANSFORMER_REQUIRED,
        "pretrain": _PRETRAIN_REQUIRED,
        "finetune": _FINETUNE_REQUIRED,
    }

    if model_type not in required_map:
        return [f"Unknown model_type '{model_type}'. Expected one of {list(required_map)}."]

    errors = _check_required_fields(config, required_map[model_type], model_type)
    errors.extend(_check_n_features_targets(config, model_type))
    return errors


def validate_config(config: dict[str, Any], config_name: str) -> list[str]:
    """Convenience dispatcher that picks the right validator.

    Parameters
    ----------
    config : dict
        Parsed YAML config.
    config_name : str
        One of ``"data"``, ``"lstm"``, ``"transformer"``, ``"pretrain"``,
        ``"finetune"``.

    Returns
    -------
    list[str]
        Error messages.  Empty when valid.
    """
    if config_name == "data":
        return validate_data_config(config)
    return validate_model_config(config, config_name)
