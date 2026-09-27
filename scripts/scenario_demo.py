"""Proof-of-concept scenario demos with human-meaningful impact indicators.

Loads the trained ClimateLSTM checkpoint and fitted scaler, takes a real
24-hour window from the 2024 Delhi test split as the starting state, and runs
what-if scenarios via ``src.scenario.run_scenario`` with Monte-Carlo dropout
uncertainty. On top of the raw forecast it derives IMD / NOAA impact
indicators (``src.impact``) so each scenario shows not just how the weather
variables shift, but what that means for people (feels-like temperature,
heat-risk and rainfall alert levels).

Two scenarios are produced:
    1. Heatwave       — t2m +4 degrees C
    2. Heavy rainfall — tp perturbation (a wetter what-if)

Reads only existing artifacts; retrains nothing.

Usage
-----
    python scripts/scenario_demo.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.impact import heat_index, summarize_state  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402
from src.models.lstm import ClimateLSTM  # noqa: E402
from src.preprocessing import load_scaler  # noqa: E402
from src.scenario import (  # noqa: E402
    SCENARIO_DISCLAIMER,
    ScenarioResult,
    get_perturbable_features,
    run_scenario,
)
from src.seed import set_global_seed  # noqa: E402

logger = logging.getLogger(__name__)

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

HORIZON = 12          # hours ahead — sensitivity stays interpretable
MC_SAMPLES = 40       # MC-dropout passes for uncertainty band
WINDOW_START = 4000   # index into the test split for the start window
SEED = 42

ALERT_COLORS = {"red": "#c0392b", "orange": "#e67e22", "yellow": "#f1c40f", "none": "#2ecc71"}


def _plot_temperature_scenario(
    result: ScenarioResult,
    feature_order: list[str],
    units: dict[str, str],
    d2m_baseline: np.ndarray,
    d2m_scenario: np.ndarray,
    title: str,
    perturb_label: str,
    perturb_value: float,
    out_path: Path,
) -> None:
    """Temperature scenario figure: (top) t2m trajectories, (bottom) feels-like."""
    vi = feature_order.index("t2m")
    hours = np.arange(1, result.horizon + 1)

    # Feels-like ("apparent temperature") for baseline and scenario.
    feels_base = heat_index(result.baseline[:, vi], d2m_baseline)
    feels_scen = heat_index(result.perturbed[:, vi], d2m_scenario)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )

    # Top: raw t2m baseline vs scenario with predictive bands.
    ax1.plot(hours, result.baseline[:, vi], color="#1f77b4", linewidth=2,
             label="Baseline air temp")
    ax1.fill_between(hours, result.baseline_lower[:, vi], result.baseline_upper[:, vi],
                     color="#1f77b4", alpha=0.18, label="Baseline 90% interval")
    ax1.plot(hours, result.perturbed[:, vi], color="#d62728", linewidth=2,
             label=f"Scenario air temp ({perturb_label})")
    ax1.fill_between(hours, result.perturbed_lower[:, vi], result.perturbed_upper[:, vi],
                     color="#d62728", alpha=0.18, label="Scenario 90% interval")
    ax1.set_title(title, fontsize=12)
    ax1.set_ylabel(f"air temperature ({units['t2m']})")
    ax1.legend(loc="best", fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)

    # Bottom: human impact — feels-like temperature with IMD alert shading.
    ax2.plot(hours, feels_base, color="#1f77b4", linewidth=2, label="Baseline feels-like")
    ax2.plot(hours, feels_scen, color="#d62728", linewidth=2, label="Scenario feels-like")
    # IMD heatwave reference lines.
    ax2.axhline(40, color="#f1c40f", linewidth=1, linestyle="--", label="40 C onset")
    ax2.axhline(45, color="#e67e22", linewidth=1, linestyle="--", label="45 C heatwave")
    ax2.axhline(47, color="#c0392b", linewidth=1, linestyle="--", label="47 C severe")
    ax2.set_xlabel("Forecast horizon (hours ahead)")
    ax2.set_ylabel("feels-like (C)")
    ax2.legend(loc="best", fontsize=7, ncol=3)
    ax2.grid(alpha=0.3)

    fig.text(0.5, 0.005,
             "Model-based sensitivity experiment with standard IMD/NOAA impact "
             "indicators — not a physically validated climate simulation.",
             ha="center", fontsize=8, style="italic", color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_rainfall_scenario(
    result: ScenarioResult,
    feature_order: list[str],
    units: dict[str, str],
    title: str,
    perturb_label: str,
    out_path: Path,
) -> None:
    """Rainfall scenario figure: baseline vs wetter what-if with bands."""
    vi = feature_order.index("tp")
    hours = np.arange(1, result.horizon + 1)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(hours, result.baseline[:, vi], color="#1f77b4", linewidth=2,
            label="Baseline hourly rain")
    ax.fill_between(hours, np.clip(result.baseline_lower[:, vi], 0, None),
                    result.baseline_upper[:, vi],
                    color="#1f77b4", alpha=0.18, label="Baseline 90% interval")
    ax.plot(hours, result.perturbed[:, vi], color="#8e44ad", linewidth=2,
            label=f"Scenario hourly rain ({perturb_label})")
    ax.fill_between(hours, np.clip(result.perturbed_lower[:, vi], 0, None),
                    result.perturbed_upper[:, vi],
                    color="#8e44ad", alpha=0.18, label="Scenario 90% interval")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Forecast horizon (hours ahead)")
    ax.set_ylabel(f"precipitation ({units['tp']}/h)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    fig.text(0.5, 0.005,
             "Rainfall is inherently harder to forecast than temperature; shown "
             "as a sensitivity experiment with IMD categories, not a validated prediction.",
             ha="center", fontsize=8, style="italic", color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _log_impact(result: ScenarioResult, feature_order: list[str], tag: str) -> None:
    """Log a human-meaningful impact summary at the first forecast hour."""
    idx = {f: i for i, f in enumerate(feature_order)}
    # Use hour-1 forecast state; scale hourly tp to a 24h-equivalent for IMD label.
    b = result.baseline[0]
    p = result.perturbed[0]
    base_report = summarize_state(
        t2m_c=b[idx["t2m"]], d2m_c=b[idx["d2m"]],
        tp_mm_24h=max(0.0, b[idx["tp"]]) * 24.0,
        u10=b[idx["u10"]], v10=b[idx["v10"]],
    )
    scen_report = summarize_state(
        t2m_c=p[idx["t2m"]], d2m_c=p[idx["d2m"]],
        tp_mm_24h=max(0.0, p[idx["tp"]]) * 24.0,
        u10=p[idx["u10"]], v10=p[idx["v10"]],
    )
    logger.info("--- %s impact @ hour 1 ---", tag)
    logger.info("BASELINE: air %.1f C, feels %.1f C, RH %.0f%%, heat=%s, rain=%s",
                base_report.air_temp_c, base_report.feels_like_c,
                base_report.relative_humidity_pct, base_report.heat_label,
                base_report.rainfall_label)
    logger.info("SCENARIO: air %.1f C, feels %.1f C, RH %.0f%%, heat=%s, rain=%s",
                scen_report.air_temp_c, scen_report.feels_like_c,
                scen_report.relative_humidity_pct, scen_report.heat_label,
                scen_report.rainfall_label)
    for card in scen_report.alerts:
        logger.info("  ALERT: %s", card)


def main() -> None:
    setup_logging("INFO")
    set_global_seed(SEED)

    data_cfg = load_config("configs/data.yaml")
    model_cfg = load_config("configs/lstm.yaml")
    feature_order = data_cfg["feature_order"]
    units = data_cfg["converted_units"]
    input_window = model_cfg["input_window"]

    perturbable = get_perturbable_features(
        feature_order, data_cfg.get("perturbable_features", [])
    )
    logger.info("Perturbable features (dynamic): %s", perturbable)

    scaler, scaler_features, _ = load_scaler("data/processed/scaler.joblib")
    assert scaler_features == feature_order, "Scaler feature order mismatch!"

    model = ClimateLSTM(
        n_features=model_cfg["n_features"], n_targets=model_cfg["n_targets"],
        hidden_dim=model_cfg["hidden_dim"], num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
    )
    model.load_state_dict(
        torch.load("results/checkpoints/lstm_delhi_seed42_001.pt", weights_only=True)
    )

    test_scaled = pd.read_csv("data/processed/test_scaled.csv")
    data = test_scaled[feature_order].values.astype(np.float32)
    window = data[WINDOW_START : WINDOW_START + input_window]
    assert window.shape == (input_window, len(feature_order))

    di = feature_order.index("d2m")

    # ---- Scenario 1: heatwave (+4 C) ----
    logger.info("=== Scenario 1: heatwave (t2m +4 C) ===")
    heat = run_scenario(
        model=model, baseline_input=window, perturbations={"t2m": 4.0},
        horizon=HORIZON, feature_names=feature_order, scaler=scaler,
        mc_samples=MC_SAMPLES, interval=(5.0, 95.0), seed=SEED, sustained=False,
    )
    _plot_temperature_scenario(
        heat, feature_order, units,
        d2m_baseline=heat.baseline[:, di], d2m_scenario=heat.perturbed[:, di],
        title=("ClimateTwin Scenario Sensitivity — Delhi\n"
               "Heatwave what-if (+4 C): air temperature and feels-like impact"),
        perturb_label="+4 C heatwave", perturb_value=4.0,
        out_path=FIGURES_DIR / "scenario_heatwave_t2m_delhi.png",
    )
    _log_impact(heat, feature_order, "HEATWAVE")

    # ---- Scenario 2: heavy rainfall (wetter what-if) ----
    # Perturb tp upward by a physically meaningful hourly amount (+5 mm/h).
    logger.info("=== Scenario 2: heavy rainfall (tp +5 mm/h) ===")
    rain = run_scenario(
        model=model, baseline_input=window, perturbations={"tp": 5.0},
        horizon=HORIZON, feature_names=feature_order, scaler=scaler,
        mc_samples=MC_SAMPLES, interval=(5.0, 95.0), seed=SEED, sustained=False,
    )
    _plot_rainfall_scenario(
        rain, feature_order, units,
        title=("ClimateTwin Scenario Sensitivity — Delhi\n"
               "Heavy-rain what-if (+5 mm/h): precipitation response"),
        perturb_label="+5 mm/h",
        out_path=FIGURES_DIR / "scenario_rainfall_tp_delhi.png",
    )
    _log_impact(rain, feature_order, "RAINFALL")

    logger.info("Figures written to %s", FIGURES_DIR.relative_to(PROJECT_ROOT))
    logger.info("Disclaimer: %s", SCENARIO_DISCLAIMER)


if __name__ == "__main__":
    main()
