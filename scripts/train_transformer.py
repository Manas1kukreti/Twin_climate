"""Train the ClimateTransformer model (scratch) on the Delhi dataset.

Reads configs, loads processed scaled data, trains with early stopping
on validation loss, saves best checkpoint, training log, and run manifest.
Requires CUDA; does not silently fall back to CPU.

Usage
-----
    python scripts/train_transformer.py
"""

from __future__ import annotations

import json
import logging
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

from src.config import load_config  # noqa: E402
from src.dataset import ClimateSequenceDataset  # noqa: E402
from src.evaluate import (  # noqa: E402
    compute_per_variable_metrics,
    inverse_transform_predictions,
    load_and_validate_baseline_metrics,
    save_metrics,
)
from src.logging_config import setup_logging  # noqa: E402
from src.manifest import create_manifest, save_manifest  # noqa: E402
from src.models.transformer import ClimateTransformer  # noqa: E402
from src.preprocessing import load_scaler  # noqa: E402
from src.seed import set_global_seed  # noqa: E402

logger = logging.getLogger(__name__)


def main(
    config_data_path: str = "configs/data.yaml",
    config_model_path: str = "configs/transformer.yaml",
) -> None:
    """Train ClimateTransformer and evaluate on test set."""
    setup_logging("INFO")

    data_cfg = load_config(config_data_path)
    model_cfg = load_config(config_model_path)

    feature_order = data_cfg["feature_order"]
    input_window = model_cfg["input_window"]
    n_features = model_cfg["n_features"]
    n_targets = model_cfg["n_targets"]
    d_model = model_cfg["d_model"]
    nhead = model_cfg["nhead"]
    num_encoder_layers = model_cfg["num_encoder_layers"]
    dim_feedforward = model_cfg["dim_feedforward"]
    dropout = model_cfg["dropout"]
    batch_size = model_cfg["batch_size"]
    lr = model_cfg["learning_rate"]
    max_epochs = model_cfg["epochs"]
    seed = model_cfg["seed"]
    patience = model_cfg["early_stopping"]["patience"]

    # --- CUDA requirement: no silent fallback ---
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for Transformer training per approved instructions, "
            "but torch.cuda.is_available() returned False. Aborting rather than "
            "silently falling back to CPU."
        )
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)

    logger.info("=== ClimateTransformer Training (Scratch) ===")
    logger.info("PyTorch version: %s", torch.__version__)
    logger.info("CUDA runtime: %s", torch.version.cuda)
    logger.info("Device: %s", device)
    logger.info("GPU: %s", gpu_name)
    logger.info(
        "Initial GPU memory allocated: %.2f MB",
        torch.cuda.memory_allocated(0) / 1024**2,
    )
    logger.info(
        "Config: d_model=%d, nhead=%d, layers=%d, ffn=%d, dropout=%.2f, lr=%.5f, batch=%d",
        d_model, nhead, num_encoder_layers, dim_feedforward, dropout, lr, batch_size,
    )

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

    assert len(train_ds) == 43800, f"Expected 43800 train sequences, got {len(train_ds)}"
    assert len(val_ds) == 8736, f"Expected 8736 val sequences, got {len(val_ds)}"

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    logger.info("Train sequences: %d, Val sequences: %d", len(train_ds), len(val_ds))

    # --- Model ---
    model = ClimateTransformer(
        n_features, n_targets, d_model, nhead, num_encoder_layers, dim_feedforward, dropout
    ).to(device)
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
    ckpt_path = ckpt_dir / "transformer_scratch_delhi_seed42_001.pt"

    wall_start = time.time()

    try:
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

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                epochs_no_improve += 1

            if epoch % 5 == 0 or epoch == 1 or epochs_no_improve == 0:
                logger.info(
                    "Epoch %03d: train_loss=%.6f, val_loss=%.6f, lr=%.5f%s",
                    epoch, avg_train_loss, avg_val_loss, lr,
                    " *" if epochs_no_improve == 0 else "",
                )

            if epochs_no_improve >= patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                break
    except torch.cuda.OutOfMemoryError:
        logger.exception(
            "CUDA out-of-memory. Per instructions, reduce batch_size "
            "(256 -> 128 -> 64) before changing architecture."
        )
        raise

    wall_seconds = time.time() - wall_start
    total_epochs = len(train_losses)

    logger.info(
        "Training complete: %d epochs, best_epoch=%d, best_val_loss=%.6f, wall=%.1fs",
        total_epochs, best_epoch, best_val_loss, wall_seconds,
    )
    logger.info(
        "Peak GPU memory allocated: %.2f MB", torch.cuda.max_memory_allocated(0) / 1024**2
    )

    # --- Save training log ---
    training_log = {
        "model": "transformer_scratch",
        "seed": seed,
        "device": str(device),
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "parameter_count": param_count,
        "hyperparameters": {
            "n_features": n_features,
            "n_targets": n_targets,
            "input_window": input_window,
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "learning_rate": lr,
            "batch_size": batch_size,
            "optimizer": "Adam",
            "loss": "MSE",
            "max_epochs": max_epochs,
            "early_stopping_patience": patience,
            "positional_encoding": "sinusoidal",
        },
        "total_epochs": total_epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "wall_clock_seconds": round(wall_seconds, 2),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated(0) / 1024**2, 2),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_sequences": len(train_ds),
        "val_sequences": len(val_ds),
    }

    log_path = Path(model_cfg["metrics_dir"]) / "transformer_scratch_delhi_training_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    logger.info("Training log: %s", log_path)

    # --- Checkpoint validation: reload and verify finite forward pass ---
    logger.info("=== Checkpoint Validation ===")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        sanity_x = torch.randn(2, input_window, n_features).to(device)
        sanity_out = model(sanity_x)
    assert sanity_out.shape == (2, n_targets)
    assert torch.isfinite(sanity_out).all()
    logger.info("Checkpoint reload verified: shape=%s, finite=True", tuple(sanity_out.shape))

    # --- Test evaluation ---
    logger.info("=== Test Evaluation ===")
    logger.info("Loaded best checkpoint from epoch %d", best_epoch)

    test_scaled_df = pd.read_csv("data/processed/test_scaled.csv")
    test_unscaled_df = pd.read_csv("data/processed/test.csv", parse_dates=["timestamp"])
    test_data = test_scaled_df[feature_order].values.astype(np.float32)

    test_ds = ClimateSequenceDataset(
        data=test_data, input_window=input_window, feature_names=feature_order
    )
    assert len(test_ds) == 8760, f"Expected 8760 test sequences, got {len(test_ds)}"
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

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

    scaler, scaler_features, _ = load_scaler("data/processed/scaler.joblib")
    assert scaler_features == feature_order, "Scaler feature order mismatch!"

    preds_physical = inverse_transform_predictions(preds_scaled, scaler, feature_order)
    targets_physical = inverse_transform_predictions(targets_scaled, scaler, feature_order)

    test_timestamps = test_unscaled_df["timestamp"].values[input_window:]
    assert len(test_timestamps) == 8760

    converted_units = data_cfg["converted_units"]
    metrics = compute_per_variable_metrics(targets_physical, preds_physical, feature_order)

    logger.info("Per-variable test metrics (physical units):")
    for var in feature_order:
        unit = converted_units[var]
        m = metrics[var]
        logger.info("  %s: MAE=%.4f %s, RMSE=%.4f %s", var, m["mae"], unit, m["rmse"], unit)

    # --- Comparisons ---
    # Load and validate prior real experiment metrics rather than
    # hardcoding comparison values. This enforces identical feature order,
    # units, sample count, input window, and target-timestamp alignment
    # before any comparison is made.
    baseline_metrics = load_and_validate_baseline_metrics(
        paths={
            "persistence": "results/metrics/persistence_delhi_metrics.json",
            "lstm": "results/metrics/lstm_delhi_metrics.json",
        },
        expected_feature_order=feature_order,
        expected_units=converted_units,
        expected_sample_count=8760,
        expected_input_window=input_window,
        expected_first_timestamp="2024-01-02T00:00:00",
        expected_last_timestamp="2024-12-31T23:00:00",
        predictions_paths={
            "lstm": "results/predictions/lstm_delhi_predictions.csv",
        },
    )
    persistence_rmse = {var: baseline_metrics["persistence"][var]["rmse"] for var in feature_order}
    lstm_rmse = {var: baseline_metrics["lstm"][var]["rmse"] for var in feature_order}

    comparison_persistence: dict = {}
    comparison_lstm: dict = {}
    for var in feature_order:
        t_rmse = metrics[var]["rmse"]

        p_rmse = persistence_rmse[var]
        p_diff = p_rmse - t_rmse
        p_pct = 100.0 * p_diff / p_rmse if p_rmse > 0 else 0.0
        comparison_persistence[var] = {
            "baseline_rmse": p_rmse,
            "transformer_rmse": round(t_rmse, 4),
            "abs_diff": round(p_diff, 4),
            "pct_improvement": round(p_pct, 2),
            "transformer_better": t_rmse < p_rmse,
        }

        l_rmse = lstm_rmse[var]
        l_diff = l_rmse - t_rmse
        l_pct = 100.0 * l_diff / l_rmse if l_rmse > 0 else 0.0
        comparison_lstm[var] = {
            "baseline_rmse": l_rmse,
            "transformer_rmse": round(t_rmse, 4),
            "abs_diff": round(l_diff, 4),
            "pct_improvement": round(l_pct, 2),
            "transformer_better": t_rmse < l_rmse,
        }

        logger.info(
            "  %s: pers=%.4f lstm=%.4f transformer=%.4f | vs_pers=%.2f%% vs_lstm=%.2f%%",
            var, p_rmse, l_rmse, t_rmse, p_pct, l_pct,
        )

    # --- Save predictions ---
    pred_df = pd.DataFrame({"timestamp": test_timestamps})
    for i, var in enumerate(feature_order):
        pred_df[f"actual_{var}"] = targets_physical[:, i]
        pred_df[f"pred_{var}"] = preds_physical[:, i]

    pred_path = Path(model_cfg["predictions_dir"]) / "transformer_scratch_delhi_predictions.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_path, index=False)
    logger.info("Predictions: %s", pred_path)

    # --- Save metrics ---
    metrics_doc = {
        "run_id": "transformer_scratch_delhi_seed42_001",
        "model": "transformer_scratch",
        "parameter_count": param_count,
        "architecture": {
            "type": "lightweight_transformer_encoder",
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "positional_encoding": "sinusoidal",
        },
        "hyperparameters": training_log["hyperparameters"],
        "seed": seed,
        "device": str(device),
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "input_window": input_window,
        "feature_order": feature_order,
        "target_order": feature_order,
        "training_epochs": total_epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "wall_clock_seconds": round(wall_seconds, 2),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated(0) / 1024**2, 2),
        "test_sample_count": int(preds_scaled.shape[0]),
        "per_variable_metrics": metrics,
        "units": converted_units,
        "comparison_vs_persistence": comparison_persistence,
        "comparison_vs_lstm": comparison_lstm,
        "evaluation_date": datetime.now(UTC).isoformat(),
    }

    metrics_path = Path(model_cfg["metrics_dir"]) / "transformer_scratch_delhi_metrics.json"
    save_metrics(metrics_doc, metrics_path)

    # --- Save manifest ---
    manifest = create_manifest(
        run_id="transformer_scratch_delhi_seed42_001",
        stage="scratch",
        model="transformer",
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
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "parameter_count": param_count,
            "positional_encoding": "sinusoidal",
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
            "gpu": gpu_name,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        hyperparameter_selection={
            "method": "single primary configuration per §9.6",
            "candidates_evaluated": [
                {
                    "d_model": d_model,
                    "nhead": nhead,
                    "num_encoder_layers": num_encoder_layers,
                    "dim_feedforward": dim_feedforward,
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

    manifest_path = Path(model_cfg["manifests_dir"]) / "transformer_scratch_delhi_seed42_001.json"
    save_manifest(manifest, manifest_path)

    logger.info("=== Phase 5 artifacts saved ===")
    logger.info("Checkpoint: %s", ckpt_path)
    logger.info("Predictions: %s", pred_path)
    logger.info("Metrics: %s", metrics_path)
    logger.info("Training log: %s", log_path)
    logger.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
