"""Generate diagnostic figures from saved ClimateTwin prediction/metric artifacts.

Reads real, already-executed prediction and metrics files under
``results/predictions/`` and ``results/metrics/`` and produces reproducible
PNG figures under ``results/figures/``. Does not recompute or retrain
anything, and does not fabricate any values.

Figures produced per variable (t2m, d2m, sp, tp, u10, v10):
    1. Actual vs predicted time series (short window) — Persistence/LSTM/Transformer
    2. Predicted vs actual scatter plot — Persistence/LSTM/Transformer
    3. Residual (error) distribution histogram — Persistence/LSTM/Transformer

Additional summary figures:
    4. Per-variable MAE/RMSE bar chart across the three models
    5. Training vs validation loss curves — LSTM and Transformer

Usage
-----
    python scripts/generate_figures.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, safe for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

MODELS: dict[str, str] = {
    "Persistence": "persistence_delhi_predictions.csv",
    "LSTM": "lstm_delhi_predictions.csv",
    "Transformer": "transformer_scratch_delhi_predictions.csv",
}

MODEL_COLORS: dict[str, str] = {
    "Persistence": "#888888",
    "LSTM": "#1f77b4",
    "Transformer": "#d62728",
}

METRICS_FILES: dict[str, str] = {
    "Persistence": "persistence_delhi_metrics.json",
    "LSTM": "lstm_delhi_metrics.json",
    "Transformer": "transformer_scratch_delhi_metrics.json",
}

TRAINING_LOG_FILES: dict[str, str] = {
    "LSTM": "lstm_delhi_training_log.json",
    "Transformer": "transformer_scratch_delhi_training_log.json",
}


def load_predictions() -> dict[str, pd.DataFrame]:
    """Load all three models' predictions, parsing timestamps."""
    data: dict[str, pd.DataFrame] = {}
    for model_name, filename in MODELS.items():
        path = PREDICTIONS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Predictions file not found: {path}")
        df = pd.read_csv(path, parse_dates=["timestamp"])
        data[model_name] = df
    return data


def load_metrics_docs() -> dict[str, dict]:
    """Load all three models' metrics JSON documents."""
    docs: dict[str, dict] = {}
    for model_name, filename in METRICS_FILES.items():
        path = METRICS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Metrics file not found: {path}")
        with open(path, encoding="utf-8") as f:
            docs[model_name] = json.load(f)
    return docs


def load_training_logs() -> dict[str, dict]:
    """Load LSTM and Transformer training logs (Persistence has none)."""
    logs: dict[str, dict] = {}
    for model_name, filename in TRAINING_LOG_FILES.items():
        path = METRICS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Training log not found: {path}")
        with open(path, encoding="utf-8") as f:
            logs[model_name] = json.load(f)
    return logs


def plot_actual_vs_predicted_timeseries(
    predictions: dict[str, pd.DataFrame],
    variable: str,
    unit: str,
    window_hours: int = 240,
) -> Path:
    """Plot actual vs predicted for a short window (default 10 days = 240h)."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ref_df = next(iter(predictions.values()))
    n = min(window_hours, len(ref_df))
    ts = ref_df["timestamp"].iloc[:n]

    ax.plot(ts, ref_df[f"actual_{variable}"].iloc[:n], color="black", linewidth=1.8,
            label="Actual", zorder=5)

    for model_name, df in predictions.items():
        ax.plot(
            ts, df[f"pred_{variable}"].iloc[:n],
            color=MODEL_COLORS[model_name], linewidth=1.2, alpha=0.85,
            linestyle="--", label=f"{model_name} predicted",
        )

    ax.set_title(f"Actual vs Predicted — {variable} (first {n}h of 2024 test period)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(f"{variable} ({unit})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = FIGURES_DIR / f"{variable}_actual_vs_predicted_timeseries.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_predicted_vs_actual_scatter(
    predictions: dict[str, pd.DataFrame],
    variable: str,
    unit: str,
) -> Path:
    """Scatter plot of predicted vs actual for all three models, one subplot each."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    for ax, (model_name, df) in zip(axes, predictions.items(), strict=True):
        actual = df[f"actual_{variable}"].values
        pred = df[f"pred_{variable}"].values

        ax.scatter(actual, pred, s=4, alpha=0.25, color=MODEL_COLORS[model_name])

        lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
        ax.plot(lims, lims, color="black", linestyle="--", linewidth=1, label="y = x")

        ax.set_title(model_name)
        ax.set_xlabel(f"Actual {variable} ({unit})")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    axes[0].set_ylabel(f"Predicted {variable} ({unit})")
    fig.suptitle(f"Predicted vs Actual — {variable} (2024 test period, n=8760)")
    fig.tight_layout()

    out_path = FIGURES_DIR / f"{variable}_predicted_vs_actual_scatter.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_residual_histogram(
    predictions: dict[str, pd.DataFrame],
    variable: str,
    unit: str,
) -> Path:
    """Overlaid residual (predicted - actual) histograms for all three models."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for model_name, df in predictions.items():
        residual = df[f"pred_{variable}"].values - df[f"actual_{variable}"].values
        ax.hist(
            residual, bins=60, alpha=0.5, density=True,
            color=MODEL_COLORS[model_name], label=model_name,
        )

    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title(f"Residual Distribution (Predicted − Actual) — {variable}")
    ax.set_xlabel(f"Residual ({unit})")
    ax.set_ylabel("Density")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = FIGURES_DIR / f"{variable}_residual_histogram.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_metric_comparison_bar(
    metrics_docs: dict[str, dict],
    feature_order: list[str],
    units: dict[str, str],
) -> list[Path]:
    """Grouped bar chart of MAE and RMSE per variable across all three models."""
    out_paths: list[Path] = []

    for metric_name in ("mae", "rmse"):
        fig, ax = plt.subplots(figsize=(11, 6))

        x = np.arange(len(feature_order))
        width = 0.25

        for i, (model_name, doc) in enumerate(metrics_docs.items()):
            values = [doc["per_variable_metrics"][var][metric_name] for var in feature_order]
            ax.bar(
                x + (i - 1) * width, values, width,
                label=model_name, color=MODEL_COLORS[model_name],
            )

        ax.set_xticks(x)
        labels = [f"{var}\n({units[var]})" for var in feature_order]
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric_name.upper())
        ax.set_title(f"Per-Variable {metric_name.upper()} Comparison — 2024 Test Period (n=8760)")
        ax.legend(loc="best")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()

        out_path = FIGURES_DIR / f"model_comparison_{metric_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


def plot_training_curves(training_logs: dict[str, dict]) -> list[Path]:
    """Training vs validation loss curves for LSTM and Transformer."""
    out_paths: list[Path] = []

    for model_name, log in training_logs.items():
        fig, ax = plt.subplots(figsize=(9, 5))

        epochs = np.arange(1, len(log["train_losses"]) + 1)
        ax.plot(epochs, log["train_losses"], color=MODEL_COLORS[model_name],
                linewidth=1.5, label="Train loss")
        ax.plot(epochs, log["val_losses"], color=MODEL_COLORS[model_name],
                linewidth=1.5, linestyle="--", label="Validation loss")

        best_epoch = log["best_epoch"]
        ax.axvline(best_epoch, color="black", linewidth=1, linestyle=":",
                   label=f"Best epoch ({best_epoch})")

        ax.set_title(f"{model_name} — Training vs Validation Loss (MSE, standardized space)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        out_path = FIGURES_DIR / f"{model_name.lower()}_training_curves.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


def main(config_path: str = "configs/data.yaml") -> None:
    setup_logging("INFO")

    data_cfg = load_config(config_path)
    feature_order: list[str] = data_cfg["feature_order"]
    units: dict[str, str] = data_cfg["converted_units"]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading predictions, metrics, and training logs...")
    predictions = load_predictions()
    metrics_docs = load_metrics_docs()
    training_logs = load_training_logs()

    # Sanity: all three prediction sets must have identical timestamps
    ref_ts = next(iter(predictions.values()))["timestamp"]
    for model_name, df in predictions.items():
        if not df["timestamp"].equals(ref_ts):
            raise ValueError(
                f"Timestamp mismatch: {model_name} predictions do not align "
                "with the other models. Cannot generate comparison figures."
            )
    logger.info("Verified all models share identical %d test timestamps.", len(ref_ts))

    generated: list[Path] = []

    for variable in feature_order:
        unit = units[variable]
        logger.info("Generating figures for variable: %s", variable)

        generated.append(
            plot_actual_vs_predicted_timeseries(predictions, variable, unit)
        )
        generated.append(
            plot_predicted_vs_actual_scatter(predictions, variable, unit)
        )
        generated.append(
            plot_residual_histogram(predictions, variable, unit)
        )

    logger.info("Generating model comparison bar charts...")
    generated.extend(plot_metric_comparison_bar(metrics_docs, feature_order, units))

    logger.info("Generating training curve figures...")
    generated.extend(plot_training_curves(training_logs))

    logger.info("=== Figure generation complete ===")
    logger.info("Generated %d figures in %s", len(generated), FIGURES_DIR)
    for p in generated:
        logger.info("  %s", p.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
