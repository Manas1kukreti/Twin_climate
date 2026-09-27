"""Train the ClimateLSTM model on the Delhi dataset.

Reads configs, loads processed scaled data, trains with early stopping
on validation loss, saves best checkpoint, training log, and run manifest.

Usage
-----
    python scripts/train_lstm.py
    python scripts/train_lstm.py --config-data configs/data.yaml --config-model configs/lstm.yaml
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging  # noqa: E402

from src.config import load_config  # noqa: E402
from src.dataset import ClimateSequenceDataset  # noqa: E402
from src.evaluate import (  # noqa: E402
    compute_per_variable_metrics,
    inverse_transform_predictions,
    save_metrics,
)
from src.logging_config import setup_logging  # noqa: E402
from src.manifest import create_manifest, save_manifest  # noqa: E402
from src.models.lstm import ClimateLSTM  # noqa: E402
from src.preprocessing import load_scaler  # noqa: E402
from src.seed import set_global_seed  # noqa: E402

logger = logging.getLogger(__name__)


def main(
    config_data_path: str = "configs/data.yaml",
    config_model_path: str = "configs/lstm.yaml",
) -> None:
    """Train ClimateLSTM and evaluate on test set."""
    setup_logging("INFO")

    data_cfg = load_config(config_data_path)
    model_cfg = load_config(config_model_path)

    # --- Configuration ---
    feature_order = data_cfg["feature_order"]
    input_window = model_cfg["input_window"]
    n_features = model_cfg["n_features"]
    n_targets = model_cfg["n_targets"]
    hidden_dim = model_cfg["hidden_dim"]
    num_layers = model_cfg["num_layers"]
    dropout = model_cfg["dropout"]
    batch_size = model_cfg["batch_size"]
    lr = model_cfg["learning_rate"]
    max_epochs = model_cfg["epochs"]
    seed = model_cfg["seed"]
    patience = model_cfg["early_stopping"]["patience"]

    device_str = model_cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    logger.info("=== ClimateLSTM Training ===")
    logger.info("Device: %s", device)
    logger.info("Config: hidden=%d, layers=%d, dropout=%.2f, lr=%.4f, batch=%d",
                hidden_dim, num_layers, dropout, lr, batch_size)

    # --- Seed ---
    set_global_seed(seed)

    # --- Load data ---
    train_df = pd.read_csv("data/processed/train_scaled.csv")
    val_df = pd.read_csv("data/processed/val_scaled.csv")

    train_data = train_df[feature_order].values.astype(np.float32)
    val_data = val_df[feature_order].values.astype(np.float32)

    train_ds = ClimateSequenceDataset(
        data=train_data, input_window=input_window, feature_names=feature_order
    )
    val_ds = ClimateSequenceDataset(
        data=val_data, input_window=input_window, feature_names=feature_order
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    logger.info("Train sequences: %d, Val sequences: %d", len(train_ds), len(val_ds))

    # --- Model ---
    model = ClimateLSTM(n_features, n_targets, hidden_dim, num_layers, dropout).to(device)
    param_count = model.count_parameters()
    logger.info("Model parameters: %d", param_count)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # --- Training loop ---
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    ckpt_dir = Path(model_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "lstm_delhi_seed42_001.pt"

    wall_start = time.time()

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        epoch_train_loss = 0.0
        n_train_batches = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = epoch_train_loss / n_train_batches

        # Validate
        model.eval()
        epoch_val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                epoch_val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = epoch_val_loss / n_val_batches

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == 1 or epochs_no_improve == 0:
            logger.info(
                "Epoch %03d: train_loss=%.6f, val_loss=%.6f%s",
                epoch, avg_train_loss, avg_val_loss,
                " *" if epochs_no_improve == 0 else "",
            )

        if epochs_no_improve >= patience:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
            break

    wall_seconds = time.time() - wall_start
    total_epochs = len(train_losses)

    logger.info("Training complete: %d epochs, best_epoch=%d, best_val_loss=%.6f, wall=%.1fs",
                total_epochs, best_epoch, best_val_loss, wall_seconds)

    # --- Save training log ---
    training_log = {
        "model": "lstm",
        "seed": seed,
        "device": str(device),
        "parameter_count": param_count,
        "hyperparameters": {
            "n_features": n_features,
            "n_targets": n_targets,
            "input_window": input_window,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": lr,
            "batch_size": batch_size,
            "optimizer": "Adam",
            "loss": "MSE",
            "max_epochs": max_epochs,
            "early_stopping_patience": patience,
        },
        "total_epochs": total_epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "wall_clock_seconds": round(wall_seconds, 2),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_sequences": len(train_ds),
        "val_sequences": len(val_ds),
    }

    log_path = Path(model_cfg["metrics_dir"]) / "lstm_delhi_training_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    logger.info("Training log: %s", log_path)

    # --- Test evaluation ---
    logger.info("=== Test Evaluation ===")

    # Load best checkpoint
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()
    logger.info("Loaded best checkpoint from epoch %d", best_epoch)

    # Load test data
    test_scaled_df = pd.read_csv("data/processed/test_scaled.csv")
    test_unscaled_df = pd.read_csv("data/processed/test.csv", parse_dates=["timestamp"])
    test_data = test_scaled_df[feature_order].values.astype(np.float32)

    test_ds = ClimateSequenceDataset(
        data=test_data, input_window=input_window, feature_names=feature_order
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # Collect predictions
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            pred = model(x_batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())

    preds_scaled = np.concatenate(all_preds, axis=0)
    targets_scaled = np.concatenate(all_targets, axis=0)

    logger.info("Test predictions: %d samples", preds_scaled.shape[0])
    assert preds_scaled.shape[0] == 8760, f"Expected 8760, got {preds_scaled.shape[0]}"

    # Inverse transform to physical units
    scaler, scaler_features, _ = load_scaler("data/processed/scaler.joblib")
    assert scaler_features == feature_order, "Scaler feature order mismatch!"

    preds_physical = inverse_transform_predictions(preds_scaled, scaler, feature_order)
    targets_physical = inverse_transform_predictions(targets_scaled, scaler, feature_order)

    # Timestamps: target timestamps start at row input_window of test split
    test_timestamps = test_unscaled_df["timestamp"].values[input_window:]
    assert len(test_timestamps) == 8760

    # Per-variable metrics
    converted_units = data_cfg["converted_units"]
    metrics = compute_per_variable_metrics(targets_physical, preds_physical, feature_order)

    logger.info("Per-variable test metrics (physical units):")
    for var in feature_order:
        unit = converted_units[var]
        m = metrics[var]
        logger.info("  %s: MAE=%.4f %s, RMSE=%.4f %s", var, m["mae"], unit, m["rmse"], unit)

    # Persistence comparison
    persistence_rmse = {
        "t2m": 1.2729, "d2m": 0.6883, "sp": 0.5044,
        "tp": 0.3841, "u10": 0.4778, "v10": 0.4717,
    }
    comparison: dict = {}
    for var in feature_order:
        p_rmse = persistence_rmse[var]
        l_rmse = metrics[var]["rmse"]
        abs_diff = p_rmse - l_rmse
        pct_change = 100.0 * abs_diff / p_rmse if p_rmse > 0 else 0.0
        comparison[var] = {
            "persistence_rmse": p_rmse,
            "lstm_rmse": round(l_rmse, 4),
            "abs_diff": round(abs_diff, 4),
            "pct_improvement": round(pct_change, 2),
            "lstm_better": l_rmse < p_rmse,
        }
        logger.info("  %s: pers=%.4f, lstm=%.4f, diff=%.4f, pct=%.2f%%",
                     var, p_rmse, l_rmse, abs_diff, pct_change)

    # --- Save predictions ---
    pred_df = pd.DataFrame({"timestamp": test_timestamps})
    for i, var in enumerate(feature_order):
        pred_df[f"actual_{var}"] = targets_physical[:, i]
        pred_df[f"pred_{var}"] = preds_physical[:, i]

    pred_path = Path(model_cfg["predictions_dir"]) / "lstm_delhi_predictions.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_path, index=False)
    logger.info("Predictions: %s", pred_path)

    # --- Save metrics ---
    metrics_doc = {
        "run_id": "lstm_scratch_delhi_seed42_001",
        "model": "lstm",
        "parameter_count": param_count,
        "seed": seed,
        "input_window": input_window,
        "feature_order": feature_order,
        "target_order": feature_order,
        "hyperparameters": training_log["hyperparameters"],
        "training_epochs": total_epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "wall_clock_seconds": round(wall_seconds, 2),
        "test_sample_count": int(preds_scaled.shape[0]),
        "per_variable_metrics": metrics,
        "units": converted_units,
        "persistence_comparison": comparison,
        "evaluation_date": datetime.now(UTC).isoformat(),
    }

    metrics_path = Path(model_cfg["metrics_dir"]) / "lstm_delhi_metrics.json"
    save_metrics(metrics_doc, metrics_path)

    # --- Save manifest ---
    manifest = create_manifest(
        run_id="lstm_scratch_delhi_seed42_001",
        stage="scratch",
        model="lstm",
        seed=seed,
        device=str(device),
        dataset={
            "source": data_cfg["source"],
            "source_type": data_cfg["source_type"],
            "target_location": data_cfg["target_location"],
            "variables": feature_order,
            "target_variables": feature_order,
            "units": converted_units,
            "sampling_interval": data_cfg["sampling_interval"],
            "train_period": (
                f"{data_cfg['split']['train']['start']} to "
                f"{data_cfg['split']['train']['end']}"
            ),
            "validation_period": (
                f"{data_cfg['split']['validation']['start']} to "
                f"{data_cfg['split']['validation']['end']}"
            ),
            "test_period": (
                f"{data_cfg['split']['test']['start']} to "
                f"{data_cfg['split']['test']['end']}"
            ),
        },
        preprocessing={
            "scaler_type": data_cfg["scaler_type"],
            "scaler_path": "data/processed/scaler.joblib",
            "feature_order": feature_order,
            "input_window": input_window,
            "normalization_policy": data_cfg["normalization_policy"],
        },
        model_config={
            "n_features": n_features,
            "n_targets": n_targets,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "parameter_count": param_count,
        },
        training_config={
            "best_epoch": best_epoch,
            "wall_clock_seconds": round(wall_seconds, 2),
            "total_epochs": total_epochs,
            "best_val_loss": best_val_loss,
            "optimizer": "Adam",
            "learning_rate": lr,
            "batch_size": batch_size,
            "loss": "MSE",
            "early_stopping_patience": patience,
        },
        hyperparameter_selection={
            "method": "single primary configuration per §9.11",
            "candidates_evaluated": [
                {
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "learning_rate": lr,
                }
            ],
            "selection_rationale": "Primary configuration as approved, no search conducted.",
        },
        checkpoint_path=str(ckpt_path),
        metrics_path=str(metrics_path),
        predictions_path=str(pred_path),
    )

    manifest_path = Path(model_cfg["manifests_dir"]) / "lstm_scratch_delhi_seed42_001.json"
    save_manifest(manifest, manifest_path)

    logger.info("=== Phase 4 artifacts saved ===")
    logger.info("Checkpoint: %s", ckpt_path)
    logger.info("Predictions: %s", pred_path)
    logger.info("Metrics: %s", metrics_path)
    logger.info("Training log: %s", log_path)
    logger.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ClimateLSTM")
    parser.add_argument("--config-data", default="configs/data.yaml")
    parser.add_argument("--config-model", default="configs/lstm.yaml")
    args = parser.parse_args()
    main(args.config_data, args.config_model)
