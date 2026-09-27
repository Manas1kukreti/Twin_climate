---
inclusion: fileMatch
fileMatchPattern:
  - "src/train.py"
  - "src/evaluate.py"
  - "src/forecasting.py"
  - "src/scenario.py"
  - "src/models/**/*.py"
  - "scripts/train_*.py"
  - "scripts/pretrain.py"
  - "scripts/finetune.py"
  - "scripts/evaluate.py"
  - "scripts/generate_figures.py"
  - "configs/lstm.yaml"
  - "configs/transformer.yaml"
  - "configs/pretrain.yaml"
  - "configs/finetune.yaml"
  - "results/manifests/**/*.json"
  - "tests/test_models.py"
  - "tests/test_forecasting.py"
  - "tests/test_evaluation.py"
---

# ClimateTwin Experiment Methodology

Apply these rules to model implementation, training, pretraining, fine-tuning, evaluation, forecasting, comparison, and figure generation.

## Experiment Philosophy

ClimateTwin is a research prototype. Code completion is not the same as experiment completion.

An experiment is complete only when the intended code has executed successfully and produced traceable artifacts.

## Required Models

Implement and preserve a clear comparison among:

1. Persistence baseline
2. LSTM
3. Lightweight Transformer trained from scratch
4. Pretrained + fine-tuned Transformer

A conventional ML baseline can be added, but it must not replace the persistence baseline or delay the required models.

## Persistence Baseline

Minimum behavior:

```text
prediction at t+1 = observed state at t
```

For multivariate forecasting, define explicitly which state variables are persisted and how the baseline output aligns with model targets.

Use the same evaluation split and physical-unit metric pipeline as learned models where applicable.

## LSTM Experiment

The LSTM consumes an input sequence of previous multivariate climate states and predicts the next climate state.

Make configurable:

- input window length;
- number of input/output variables;
- hidden dimension;
- number of recurrent layers;
- dropout;
- learning rate;
- batch size;
- epochs;
- early stopping.

Test tensor shapes before full training.

## Transformer Experiment

Use a lightweight encoder-style architecture conceptually containing:

```text
feature projection
→ positional encoding
→ Transformer encoder
→ regression head
→ next climate state
```

Keep architecture small enough for laptop/Colab experiments.

Starting points such as `d_model=64/128`, `nhead=4/8`, `2-4` encoder layers, and dropout around `0.1` are tuning candidates rather than Aurora-derived settings.

Record parameter count.

## Scratch vs Transfer-Learning Experiment

This is the main Aurora-inspired experiment.

### Scratch path

```text
randomly initialized Transformer
→ target-location train data
→ validation/model selection
→ target-location test evaluation
```

### Transfer path

```text
broad/multi-location data
→ Transformer pretraining
→ save pretrained checkpoint
→ load same architecture for target location
→ fine-tune with smaller learning rate/fewer epochs as appropriate
→ target-location test evaluation
```

Keep the architecture and target evaluation procedure comparable.

Record exactly which parameters are loaded, frozen, unfrozen, or reinitialized during fine-tuning.

Do not assume transfer learning wins.

## Experiment Manifest

Every serious training/evaluation run should produce or update a machine-readable manifest containing at least:

```json
{
  "run_id": "...",
  "stage": "scratch|pretrain|finetune|evaluate",
  "model": "...",
  "seed": 42,
  "device": "...",
  "dataset": {
    "source": "...",
    "target_location": "...",
    "variables": [],
    "sampling_interval": "...",
    "train_period": "...",
    "validation_period": "...",
    "test_period": "..."
  },
  "model_config": {},
  "training_config": {},
  "checkpoint_path": "...",
  "metrics_path": "..."
}
```

Populate values from executed code/configuration rather than manually inventing them.

## Random Seeds

Use explicit seeds.

Record the seed in the run manifest and training logs.

When comparing two methods, avoid introducing unnecessary seed differences that confound the comparison. If time permits, repeat key comparisons across multiple seeds and report the aggregation transparently.

Multiple-seed experiments are an enhancement, not a reason to block the minimum required prototype.

## Training Logs

Record training and validation loss by epoch.

Save enough information to recreate:

- training curve;
- validation curve;
- best epoch;
- selected checkpoint.

If early stopping is used, record patience/criterion and whether the final evaluation uses the best checkpoint rather than merely the last epoch.

## Evaluation

Required metrics:

### MAE

```text
mean(abs(y - y_hat))
```

### RMSE

```text
sqrt(mean((y - y_hat)^2))
```

Training loss may be computed in normalized space when appropriate, but final reported metrics should be converted to original physical units where possible.

For multivariate outputs, report aggregate metrics only if their interpretation is meaningful. Prefer per-variable original-unit metrics in addition to any aggregate summary.

R² is optional.

ACC is optional and only valid with a correct anomaly climatology.

## One-Step Evaluation

Evaluate each model consistently on the same target test set when comparison is intended.

Persist predictions and aligned timestamps/ground truth so figures can be generated later without rerunning inference unnecessarily.

## Autoregressive Forecasting

Implement recursive rollout from observed history:

```text
history
→ predict t+1
→ update input window using prediction
→ predict t+2
→ repeat
```

The update logic must match the feature design. If some future exogenous features are not predicted by the model, explicitly define how they are supplied; never silently use unavailable future observations in a way that leaks information.

Evaluate horizons:

- 1;
- 3;
- 6;
- 12;
- 24 steps.

Record horizon-wise metrics in a structured file.

## Forecast-Horizon Figure

Generate Forecast Horizon vs RMSE from saved evaluation output.

Do not manually sketch an expected increasing-error curve and present it as a result. The curve may increase, flatten, fluctuate, or behave differently; show what the model actually produced.

## Required Figures

Generate from actual outputs:

1. actual vs predicted;
2. training loss vs validation loss;
3. forecast horizon vs RMSE;
4. model comparison;
5. scratch vs pretrained/fine-tuned comparison;
6. optional extreme-event case study when valid.

Prefer a reproducible figure-generation script reading machine-readable metrics/predictions.

## Model Comparison Table

At minimum compare:

```text
Persistence
LSTM
Transformer
Pretrained + Fine-Tuned Transformer
```

Include MAE and RMSE. Add R² only if meaningful.

Keep notes factual, for example "temporal recurrent model" or "Aurora-inspired transfer-learning experiment" rather than unsupported superiority claims.

## Extreme-Event Case Study

Only perform when a documented event exists in the test period.

Record the event source/reference separately from model predictions.

Assess timing, magnitude, and trend using actual observations versus forecasts.

## Scenario Sensitivity Experiment

Implement controlled input perturbations such as temperature, humidity, or wind adjustments only when those variables exist in the model input.

Compare:

```text
baseline forecast
vs
perturbed-input forecast
```

Do not interpret differences as causal effects.

## Tests Before Expensive Training

Before a long training run, verify at minimum:

- model forward-pass shapes;
- loss computation;
- one optimizer step;
- checkpoint save/load;
- scaler loading/inverse transform;
- tiny-dataset overfit/smoke behavior when useful;
- evaluation metric correctness;
- autoregressive rollout length and alignment.

Do not run expensive training from automatic file-save hooks.

## Completion Criteria for a Research Stage

A stage is complete only when applicable artifacts exist and are internally consistent:

```text
code
+ configuration
+ tests/smoke checks
+ executed run
+ checkpoint (for trained models)
+ metrics
+ manifest/provenance
+ figures when required
```

If execution has not happened, describe the stage as implemented but not yet experimentally completed.
