"""What-if scenario sensitivity engine for ClimateTwin.

Implements the scenario module described in the master specification (§2.9,
§1.7). Given a trained one-step forecaster and a real input window, this
module applies controlled perturbations to selected input variables and
compares the resulting autoregressive forecast trajectory against an
unperturbed baseline trajectory.

Novelty
-------
When ``mc_samples > 1`` and the model contains dropout layers, the rollout is
repeated under Monte-Carlo dropout to produce a *distribution* of trajectories
rather than a single line. This yields calibrated-style predictive intervals
that widen with the forecast horizon, and it lets the scenario tool report not
just "how the forecast shifts" but "how confident the twin is in that shift".

Honesty / scope disclaimer (mandatory per §1.7)
-----------------------------------------------
Scenario outputs are model-based sensitivity experiments. They are NOT
physically validated climate intervention simulations and do NOT establish
real-world causal effects. The :data:`SCENARIO_DISCLAIMER` text must be
attached to any surfaced output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mandatory disclaimer (spec §1.7 / §2.9)
# ---------------------------------------------------------------------------
SCENARIO_DISCLAIMER: str = (
    "Scenario outputs are model-based sensitivity experiments. They are not "
    "physically validated climate intervention simulations and do not "
    "establish real-world causal effects."
)


# ---------------------------------------------------------------------------
# Perturbable feature resolution (spec §2.9)
# ---------------------------------------------------------------------------
def get_perturbable_features(
    feature_names: list[str],
    configured_perturbable: list[str],
) -> list[str]:
    """Return the perturbable-feature controls, computed dynamically.

    The result is the intersection of the model's actual feature list and the
    configured perturbable subset, preserving ``feature_names`` order. No
    variable names are hard-coded; controls are derived from the model and the
    config so this generalizes to any dataset.

    Parameters
    ----------
    feature_names : list[str]
        Ordered feature names the model was trained on.
    configured_perturbable : list[str]
        Candidate perturbable feature names from
        ``configs/data.yaml::perturbable_features``.

    Returns
    -------
    list[str]
        Perturbable feature names, ordered to match ``feature_names``.

    Raises
    ------
    ValueError
        If the intersection is empty (no valid controls could be built).
    """
    configured = set(configured_perturbable)
    result = [f for f in feature_names if f in configured]

    if not result:
        raise ValueError(
            "No perturbable features available: the intersection of model "
            f"features {feature_names} and configured perturbable features "
            f"{configured_perturbable} is empty."
        )

    unknown = configured - set(feature_names)
    if unknown:
        logger.warning(
            "Ignoring configured perturbable features not in model features: %s",
            sorted(unknown),
        )

    return result


# ---------------------------------------------------------------------------
# Perturbation application (spec §2.9)
# ---------------------------------------------------------------------------
def apply_perturbation(
    baseline_input: np.ndarray,
    perturbations: dict[str, float],
    feature_names: list[str],
    scaler,
) -> np.ndarray:
    """Apply unit-aware perturbations to a *scaled* input window.

    Perturbations are expressed in physical units (e.g. ``{"t2m": +4.0}`` =
    +4 degrees C). The window is inverse-transformed to physical units, the
    perturbation is added to the selected feature columns, and the result is
    re-scaled into the model's standardized space. Only the specified features
    are altered; all other columns are numerically unchanged.

    Parameters
    ----------
    baseline_input : np.ndarray
        Scaled input window, shape ``(window, n_features)``.
    perturbations : dict[str, float]
        Mapping of feature name to physical-unit delta. Features not present
        are treated as zero-delta.
    feature_names : list[str]
        Ordered feature names matching the scaler / model.
    scaler : fitted sklearn scaler
        Scaler used to move between physical and standardized space.

    Returns
    -------
    np.ndarray
        Perturbed, scaled input window, shape ``(window, n_features)``.

    Raises
    ------
    ValueError
        If a perturbation references an unknown feature, or shapes mismatch.
    """
    if baseline_input.ndim != 2 or baseline_input.shape[1] != len(feature_names):
        raise ValueError(
            f"baseline_input must have shape (window, {len(feature_names)}); "
            f"got {baseline_input.shape}."
        )

    unknown = set(perturbations) - set(feature_names)
    if unknown:
        raise ValueError(
            f"Perturbation references unknown feature(s): {sorted(unknown)}. "
            f"Known features: {feature_names}."
        )

    # Move to physical units, apply deltas, move back to scaled space.
    physical = scaler.inverse_transform(baseline_input)
    index = {name: i for i, name in enumerate(feature_names)}
    for name, delta in perturbations.items():
        physical[:, index[name]] += float(delta)

    return scaler.transform(physical)


# ---------------------------------------------------------------------------
# Autoregressive rollout
# ---------------------------------------------------------------------------
def _rollout(
    model: nn.Module,
    window_scaled: np.ndarray,
    horizon: int,
    device: torch.device,
    sustain_delta_scaled: np.ndarray | None = None,
) -> np.ndarray:
    """Roll a one-step model forward ``horizon`` steps autoregressively.

    Each prediction is appended to the window and the oldest timestep dropped,
    so the model consumes its own output as it advances.

    Parameters
    ----------
    model : nn.Module
        One-step forecaster mapping ``(1, window, n_features)`` -> ``(1, n_targets)``.
    window_scaled : np.ndarray
        Starting window in scaled space, shape ``(window, n_features)``.
    horizon : int
        Number of steps to predict.
    device : torch.device
        Device for inference.
    sustain_delta_scaled : np.ndarray, optional
        Per-feature perturbation expressed in *scaled* units (i.e. delta /
        scaler.scale_). When provided, this offset is re-added to each new
        prediction before it re-enters the window, so the perturbation is
        *sustained* across the whole rollout ("the world stays warmer")
        rather than only seeding the initial window. Shape ``(n_features,)``.

    Returns
    -------
    np.ndarray
        Predicted trajectory in scaled space, shape ``(horizon, n_features)``.
    """
    w = window_scaled.astype(np.float32).copy()
    preds: list[np.ndarray] = []

    for _ in range(horizon):
        x = torch.from_numpy(w).unsqueeze(0).to(device)  # (1, window, n_features)
        y = model(x)  # (1, n_features)
        y_np = y.squeeze(0).detach().cpu().numpy()
        if sustain_delta_scaled is not None:
            # Keep the perturbed variable(s) held at the offset as the world
            # advances, so the scenario represents sustained conditions.
            y_np = y_np + sustain_delta_scaled.astype(np.float32)
        preds.append(y_np)
        # Slide window: drop oldest, append (possibly held) prediction.
        w = np.concatenate([w[1:], y_np[None, :]], axis=0)

    return np.stack(preds, axis=0)  # (horizon, n_features)


def _enable_mc_dropout(model: nn.Module) -> bool:
    """Put dropout layers in train mode while keeping the rest in eval mode.

    Returns True if any dropout module was found (MC-dropout is meaningful),
    else False.
    """
    found = False
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
            found = True
    # nn.LSTM applies inter-layer dropout internally; it is active in train()
    # mode. Detect it so single-layer/zero-dropout models degrade gracefully.
    for module in model.modules():
        if isinstance(module, nn.LSTM) and getattr(module, "dropout", 0):
            module.train()
            found = True
    return found


@dataclass
class ScenarioResult:
    """Structured result of a what-if scenario run.

    All trajectory arrays are in physical units, shape ``(horizon, n_features)``
    unless noted. Interval arrays are ``None`` when ``mc_samples <= 1``.
    """

    feature_names: list[str]
    horizon: int
    perturbations: dict[str, float]
    baseline: np.ndarray
    perturbed: np.ndarray
    delta: np.ndarray  # perturbed - baseline
    baseline_lower: np.ndarray | None = None
    baseline_upper: np.ndarray | None = None
    perturbed_lower: np.ndarray | None = None
    perturbed_upper: np.ndarray | None = None
    delta_lower: np.ndarray | None = None  # paired sensitivity band (lower)
    delta_upper: np.ndarray | None = None  # paired sensitivity band (upper)
    mc_samples: int = 1
    disclaimer: str = field(default=SCENARIO_DISCLAIMER)


# ---------------------------------------------------------------------------
# Scenario driver (spec §2.9)
# ---------------------------------------------------------------------------
def run_scenario(
    model: nn.Module,
    baseline_input: np.ndarray,
    perturbations: dict[str, float],
    horizon: int,
    feature_names: list[str],
    scaler,
    mc_samples: int = 1,
    interval: tuple[float, float] = (5.0, 95.0),
    device: torch.device | None = None,
    seed: int | None = None,
    sustained: bool = False,
) -> ScenarioResult:
    """Run a baseline-vs-perturbed what-if forecast experiment.

    Parameters
    ----------
    model : nn.Module
        Trained one-step forecaster.
    baseline_input : np.ndarray
        Real starting window in *scaled* space, shape ``(window, n_features)``.
    perturbations : dict[str, float]
        Physical-unit deltas per feature (e.g. ``{"t2m": 4.0}``).
    horizon : int
        Forecast horizon in steps (hours).
    feature_names : list[str]
        Ordered feature names.
    scaler : fitted sklearn scaler
        For unit-aware perturbation and inverse transforms.
    mc_samples : int, default 1
        Number of Monte-Carlo dropout passes. ``1`` disables MC-dropout and
        returns point trajectories only.
    interval : tuple[float, float], default (5.0, 95.0)
        Lower/upper percentiles for the predictive band when ``mc_samples > 1``.
    device : torch.device, optional
        Inference device. Defaults to CPU.
    seed : int, optional
        Seed for reproducible MC-dropout sampling.
    sustained : bool, default False
        If True, the perturbation is *held* throughout the autoregressive
        rollout (the scenario represents sustained conditions, e.g. a
        persistent +4 degrees C warmer world). If False, the perturbation only
        seeds the initial input window and the model is free to relax back
        toward its learned climatology.

    Returns
    -------
    ScenarioResult
        Baseline and perturbed trajectories (physical units), their difference,
        and optional MC-dropout predictive bands, plus the mandatory disclaimer.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}.")
    if mc_samples < 1:
        raise ValueError(f"mc_samples must be >= 1; got {mc_samples}.")

    device = device or torch.device("cpu")
    model = model.to(device)

    perturbed_input = apply_perturbation(
        baseline_input, perturbations, feature_names, scaler
    )

    # Build a scaled-space sustain offset (delta / scaler.scale_) so the
    # perturbation can be re-applied at each rollout step when sustained.
    sustain_delta = None
    if sustained:
        index = {name: i for i, name in enumerate(feature_names)}
        sustain_delta = np.zeros(len(feature_names), dtype=np.float32)
        for name, delta in perturbations.items():
            sustain_delta[index[name]] = float(delta) / float(scaler.scale_[index[name]])

    if mc_samples == 1:
        # Deterministic point forecast.
        model.eval()
        with torch.no_grad():
            base_scaled = _rollout(model, baseline_input, horizon, device)
            pert_scaled = _rollout(
                model, perturbed_input, horizon, device, sustain_delta
            )

        baseline = scaler.inverse_transform(base_scaled)
        perturbed = scaler.inverse_transform(pert_scaled)
        return ScenarioResult(
            feature_names=list(feature_names),
            horizon=horizon,
            perturbations=dict(perturbations),
            baseline=baseline,
            perturbed=perturbed,
            delta=perturbed - baseline,
            mc_samples=1,
        )

    # --- MC-dropout: repeat the rollout with dropout active ---
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    model.eval()  # base mode
    has_dropout = _enable_mc_dropout(model)  # re-enable dropout layers only
    if not has_dropout:
        logger.warning(
            "mc_samples=%d requested but model has no active dropout; "
            "predictive bands will be degenerate (all samples identical).",
            mc_samples,
        )

    # Paired MC-dropout: for each sample, the baseline and perturbed rollouts
    # are drawn under the SAME dropout realization (same per-sample seed set
    # immediately before each rollout). This makes the per-sample difference
    # isolate the perturbation effect rather than dropout-mask noise, so the
    # delta distribution is a clean sensitivity estimate.
    base_rng = int(torch.randint(0, 2**31 - 1, (1,)).item()) if seed is None else seed

    base_samples: list[np.ndarray] = []
    pert_samples: list[np.ndarray] = []
    with torch.no_grad():
        for s in range(mc_samples):
            sample_seed = base_rng + s

            torch.manual_seed(sample_seed)
            base_samples.append(scaler.inverse_transform(
                _rollout(model, baseline_input, horizon, device)
            ))

            torch.manual_seed(sample_seed)  # identical mask for the paired run
            pert_samples.append(scaler.inverse_transform(
                _rollout(model, perturbed_input, horizon, device, sustain_delta)
            ))
    model.eval()  # restore fully-deterministic eval mode

    base_stack = np.stack(base_samples, axis=0)  # (S, horizon, n_features)
    pert_stack = np.stack(pert_samples, axis=0)
    delta_stack = pert_stack - base_stack        # paired per-sample sensitivity

    lo, hi = interval
    baseline = base_stack.mean(axis=0)
    perturbed = pert_stack.mean(axis=0)

    return ScenarioResult(
        feature_names=list(feature_names),
        horizon=horizon,
        perturbations=dict(perturbations),
        baseline=baseline,
        perturbed=perturbed,
        delta=delta_stack.mean(axis=0),  # mean paired sensitivity
        baseline_lower=np.percentile(base_stack, lo, axis=0),
        baseline_upper=np.percentile(base_stack, hi, axis=0),
        perturbed_lower=np.percentile(pert_stack, lo, axis=0),
        perturbed_upper=np.percentile(pert_stack, hi, axis=0),
        delta_lower=np.percentile(delta_stack, lo, axis=0),
        delta_upper=np.percentile(delta_stack, hi, axis=0),
        mc_samples=mc_samples,
    )
