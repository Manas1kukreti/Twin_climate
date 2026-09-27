# ClimateTwin — Master Design-First Implementation Specification

**Project:** ClimateTwin: A Lightweight AI-Driven Local Climate Digital Twin
**Version:** 2.0 — Revised Pre-Implementation Review Draft
**Status:** Awaiting review before any implementation begins

**Authoritative sources consumed:**

| Document | Path |
|---|---|
| Research specification | `docs/research/ClimateTwin_Research_Implementation_Prompt.md` |
| Product steering | `.kiro/steering/product.md` |
| Research guardrails | `.kiro/steering/research-guardrails.md` |
| Technology stack | `.kiro/steering/tech.md` |
| Project structure | `.kiro/steering/structure.md` |
| Data methodology | `.kiro/steering/data-methodology.md` |
| Experiment methodology | `.kiro/steering/experiment-methodology.md` |

**Precedence rule:** Where the research specification and a steering file
conflict on a scientific requirement, the research specification governs
unless the user explicitly approves a documented change (per `product.md`).

**Revision history:**

| Version | Change |
|---|---|
| 1.0 | Initial specification |
| 2.0 | 18-point revision — see changelog at end of document |

---

# PART 1 — REQUIREMENTS

## 1.1 Scientific Objective

ClimateTwin is a college-scale research prototype that adapts selected
methodological principles from two 2025 papers:

- **Aurora** (Bodnar et al., Nature 2025) — pretraining → fine-tuning →
  downstream adaptation; autoregressive forecasting.
- **CREDIT** (Schreck et al., npj Climate and Atmospheric Science 2025) —
  modular end-to-end AI weather prediction framework; standardized
  evaluation; multi-step forecast comparison.

ClimateTwin does **not** reproduce either system at their original scale,
architecture, data volume, or computational budget.

## 1.2 Research Questions

### RQ1 — Transfer Learning

Does broader climate-data pretraining improve localized forecasting compared
with training the same lightweight Transformer architecture from scratch on
the target location?

**Required measurements:** per-variable original-unit MAE and RMSE,
convergence behavior (training/validation loss curves), best epoch,
wall-clock training duration, best/final validation loss.

**Null hypothesis is acceptable:** A neutral or negative transfer-learning
result is scientifically valid and must be reported.

### RQ2 — Forecast Horizon

How does forecast error change as a one-step predictor is recursively rolled
forward through increasing autoregressive horizons?

**Required horizons:** 1, 3, 6, 12, 24 steps.

**Required models for horizon evaluation:** All four required model variants
(Persistence, LSTM, scratch Transformer, pretrained+fine-tuned Transformer)
must be evaluated at all horizons where technically applicable. If any model
is excluded from horizon evaluation, the exclusion must be documented with a
specific technical reason.

**Step-to-time conversion:** Steps must not be labeled as hours or any other
physical duration until the actual dataset sampling interval has been
validated.

### RQ3 — Model Architecture

How do different forecasting architectures compare on the same localized
multivariate forecasting task?

**Required comparison set:**

1. Persistence baseline
2. Multivariate LSTM
3. Lightweight Transformer trained from scratch
4. Pretrained + fine-tuned Transformer (same architecture as #3)

An optional conventional ML baseline (e.g. Random Forest) may be added but
must not delay the four required models.

## 1.3 Model Requirements

### 1.3.1 Target Schema

**Decision point (§9.13):** The initial prototype should prefer predicting
all selected dynamic climate-state variables, making `n_targets == n_features`.
This simplifies autoregressive rollout because every predicted variable can
be fed back directly. If certain variables are exogenous (not predicted by
the model), `n_targets` must be explicit in all model constructors and the
autoregressive rollout must document how non-predicted features are supplied.

All model constructors must accept `n_features` (input dimensionality) and
`n_targets` (output dimensionality) as separate arguments, even when they
are equal. This keeps the API consistent regardless of which target schema
is selected.

### 1.3.2 Persistence Baseline

- `prediction(t+1) = observed_state(t)` for all target variables.
- Must use the same evaluation split and physical-unit metric pipeline as
  learned models.

### 1.3.3 Multivariate LSTM

- Input: N previous timesteps × `n_features` climate variables.
- Output: next climate state (`n_targets` variables).
- Constructor: `ClimateLSTM(n_features, n_targets, hidden_dim, num_layers, dropout)`.
- Configurable: input window, hidden dimension, number of layers, dropout,
  learning rate, batch size, epochs, early stopping.
- Validation data used for model selection. Test data never used for
  hyperparameter selection.

### 1.3.4 Lightweight Transformer

Architecture:

```text
climate feature sequence
→ feature projection / embedding
→ positional encoding
→ Transformer encoder
→ regression head
→ predicted next climate state
```

Constructor: `ClimateTransformer(n_features, n_targets, d_model, nhead,
num_encoder_layers, dropout)`.

Starting hyperparameter region (tuning candidates, not Aurora-derived):

- `d_model`: 64 or 128
- `nhead`: 4 or 8
- encoder layers: 2–4
- dropout: ~0.1

Record parameter count.

### 1.3.5 Pretrained + Fine-Tuned Transformer

- Same architecture as 1.3.4.
- Pretrained on broader multi-location data.
- Fine-tuned on target-location training data with smaller learning rate
  and early stopping.
- Document: which parameters loaded, frozen, unfrozen, reinitialized.

### 1.3.6 Hyperparameter Selection Protocol

All hyperparameter selection must use validation data only. Test data must
remain completely untouched until final evaluation. The protocol:

1. Define the candidate hyperparameter space in configuration before training.
2. Train using training data; evaluate candidates on validation data.
3. Select the best configuration based on validation performance.
4. Record every evaluated configuration and its validation result.
5. Record the rationale for the final selection.
6. Evaluate the selected model on the test set exactly once for final reporting.

**Decision points:** §9.6 (Transformer hyperparameters), §9.11 (LSTM
hyperparameters). Both must be resolved via this protocol, documented in
config and manifest.

## 1.4 Data Requirements

### 1.4.1 Source

A legitimate public climate/weather dataset. Candidate categories:

- ERA5 / ERA5-Land reanalysis (via CDS API / `cdsapi`).
- Weather-station data.
- Legitimately accessible IMD data.

**Decision point:** The exact dataset is an unresolved design decision
(see §9.1).

### 1.4.2 Geography and Data Type

Prefer a single station or location for the initial prototype. Candidate
targets: Delhi, Mumbai, Bengaluru, one Indian state, or a small
India-centered region.

**Station vs reanalysis distinction (mandatory):** During dataset selection,
explicitly document whether the selected target represents:

- A weather-station observation (point measurement from an instrument), or
- A reanalysis grid cell (model-assimilated estimate for a spatial grid point).

Do not describe an ERA5/ERA5-Land grid point as a "weather station." Use
accurate terminology: "reanalysis grid cell at (lat, lon)" or "station
observation at (station_id, lat, lon)."

**Decision point:** Target location is unresolved (§9.2).

### 1.4.3 Variables

Use only variables actually available in the selected dataset. Candidate
variables:

- 2 m temperature
- Relative humidity (or correctly derived humidity)
- Surface pressure
- Precipitation
- 10 m wind speed
- U wind component
- V wind component
- Optional: solar radiation

Do not invent missing variables.

**Decision point:** Final selected variables depend on dataset (§9.4).

### 1.4.4 Pipeline Contract

```text
legitimate source
→ raw immutable data
→ metadata validation
→ cleaning (duplicates, missing values, invalid values, units)
→ chronological split
→ train-only preprocessing fit
→ transform train/validation/test
→ sequence construction
→ model-ready datasets
```

This order must not be rearranged in a way that creates leakage.

### 1.4.5 Chronological Split

- Never randomly shuffle.
- Strict ordering: `max(train_ts) < min(val_ts)` and `max(val_ts) < min(test_ts)`.
- Exact boundaries depend on the dataset time range.
- Validation used for model selection/early stopping. Test never used for
  hyperparameter selection.

**Decision point:** Exact date boundaries are unresolved (§9.5).

### 1.4.6 Normalization

- Fit scalers on training data only.
- Apply fitted transform to validation and test.
- Never `fit` or `fit_transform` on validation/test.
- Persist scaler with: ordered feature names, units, fitting period, scaler
  type, target-column mapping for inverse transformation.

**Decision point (§9.14):** The normalization policy across pretraining,
scratch training, and fine-tuning is an unresolved design decision. It must
define:

- Which scaler is used for broad pretraining (fit on pretraining train data).
- Which scaler is used for scratch target-location training (fit on
  target-location train data).
- Which scaler is used for fine-tuning (pretraining scaler, target-location
  scaler, or a documented hybrid).
- Which scaler is used for inverse transformation during evaluation of each
  model variant.
- How the comparison between scratch and transfer models remains scientifically
  fair given potentially different scaling.

This must be decided and documented before training begins. Do not choose
silently in code.

### 1.4.7 Sequence Construction

- Input window of N timesteps → target at t+1.
- No training sequences may include targets from validation/test periods.
- Boundary-aware construction with explicit assertions.
- For multi-location pretraining: group/sort by location and time; never
  create sequences that jump across stations.

**Decision point:** Input window length is unresolved (§9.7).

## 1.5 Evaluation Requirements

### 1.5.1 Mandatory Metrics

- **MAE** = mean(|y − ŷ|) — per variable, in original physical units
- **RMSE** = √(mean((y − ŷ)²)) — per variable, in original physical units

**Per-variable original-unit MAE and RMSE are the primary scientific
reporting.** Do not compute or present a mixed-unit aggregate physical RMSE
that combines temperature (°C/K), pressure (hPa), wind (m/s), and
precipitation (mm) into a single number. Such an aggregate is physically
meaningless.

If an aggregate metric is desired for convenience (e.g. for loss curves or
quick comparison), it must use a mathematically meaningful documented
normalization (e.g. normalized-space metrics, or a documented weighted
scheme). The aggregate must always be presented alongside per-variable
original-unit metrics, never as a replacement.

### 1.5.2 Optional Metrics

- **R²** — may be included for individual targets but must not replace
  MAE/RMSE.

### 1.5.3 ACC — Conditional Only

ACC may only be implemented when anomalies are correctly defined relative to
an appropriate climatological reference. Ordinary Pearson correlation must
not be renamed as ACC. If no correct anomaly climatology is available, ACC
must be omitted and the omission documented.

**Decision point:** Whether ACC is scientifically feasible is unresolved
(§9.9).

### 1.5.4 Convergence Measurement Definition

"Convergence speed" must be defined before experiments, not retroactively.

**Required convergence measurements:**

- **Best epoch:** The epoch at which the best validation loss was achieved.
- **Wall-clock training duration:** Total elapsed time for the training run.

**Optional convergence measurement:**

- **Convergence epoch:** If this metric is retained in comparisons, its
  exact definition must be recorded before any training run. Example
  definition: "the first epoch after which validation loss does not improve
  by more than ε for P consecutive epochs." The values of ε and P must be
  fixed in configuration before training, not chosen after seeing results.

## 1.6 Autoregressive Forecasting Requirements

- Train or use a one-step predictor.
- Recursive rollout: feed predictions back into the input window according
  to the exact feature/sequence design.
- Evaluate at horizons: 1, 3, 6, 12, 24 steps.
- **All four required model variants** (Persistence, LSTM, scratch
  Transformer, pretrained+fine-tuned Transformer) must be evaluated at all
  horizons where technically applicable. If any model is excluded, the
  exclusion must be documented with a specific technical reason.
- Store horizon-wise predictions and metrics (per-variable original-unit).
- Explicitly define how features not recursively predicted by the model are
  supplied during rollout.
- Never silently use unavailable future ground-truth observations.

## 1.7 Scenario Sensitivity Requirements

- Controlled perturbation of selected model input variables.
- **Perturbable features must be generated dynamically** from the actual
  selected model features and a configured list of perturbable feature names.
  Do not hard-code specific variable names (e.g. "humidity", "wind") that
  may not exist in the selected dataset. The dashboard UI controls must be
  built from the intersection of model features and configured perturbable
  features.
- Compare baseline trajectory vs perturbed-input trajectory.
- Mandatory disclaimer:

> Scenario outputs are model-based sensitivity experiments. They are not
> physically validated climate intervention simulations and do not establish
> causal effects.

- Must not use causal language when interpreting results.

## 1.8 Dashboard Requirements

Streamlit application with five logical pages:

1. **Climate Overview** — latest observation, historical time series,
   selected location, key variables.
2. **AI Forecast** — model selection, actual vs predicted, metrics.
3. **Multi-Step Forecast** — selected horizon, autoregressive trajectory,
   actual values on historical test data.
4. **Model Comparison** — per-variable MAE/RMSE, training curves, horizon
   degradation.
5. **Scenario Sensitivity** — dynamically generated input perturbation
   controls, baseline vs perturbed trajectories, disclaimer.

Constraints:

- Consume saved experiment artifacts (checkpoints, predictions, metrics,
  figures).
- Must not silently retrain models.
- Cache/precompute expensive operations.
- Depend on `src/` and `results/`, not the other way around.
- **Research mode gate:** The final research dashboard requires real
  artifacts for all four required models. A development/debug mode may
  tolerate unavailable models only if they are clearly marked as
  "unavailable — not yet trained" in the UI. Never substitute placeholder
  metrics or fabricated results for missing models.

## 1.9 Visualization Requirements

All figures generated from saved real artifacts only. Never fabricated.

Required:

1. Actual vs predicted (short-horizon and longer autoregressive examples).
2. Training loss vs validation loss curves.
3. Forecast horizon vs RMSE (per-variable).
4. Model comparison bar chart (Persistence / LSTM / Transformer /
   Pretrained+Fine-Tuned Transformer) — per-variable.
5. Scratch vs pretrained+fine-tuned validation curve comparison.
6. Optional per-variable figures.
7. Optional extreme-event case study (only if feasible).

## 1.10 Reproducibility Requirements

- Central seed utility seeding Python, NumPy, PyTorch.
- Configuration-driven experiments (YAML configs in `configs/`).
- Machine-readable run manifests recording: run ID, seed, device, dataset
  metadata, model config, training config, checkpoint path, metrics path,
  software versions, Git commit.
- No hard-coded local absolute paths in research code.

## 1.11 Prohibited Claims and Actions

- Never claim to reproduce Aurora or CREDIT.
- Never fabricate metrics, graphs, benchmark comparisons, or event results.
- Never label Pearson correlation as ACC.
- Never use causal language for scenario sensitivity.
- Never present placeholders as final research results.
- Never claim an experiment completed when only code was written.
- Never silently skip failed prerequisites.
- Never present a mixed-unit aggregate physical metric as a primary result.
- Never describe a reanalysis grid cell as a weather station.


---

# PART 2 — TECHNICAL DESIGN

## 2.1 Technology Stack

| Concern | Technology |
|---|---|
| Language | Python 3.11 (preferred) |
| Deep learning | PyTorch |
| Numerical ops | NumPy |
| Tabular/time-series | pandas |
| Gridded data (if needed) | xarray |
| Scalers/baselines/metrics | scikit-learn |
| Preprocessing artifact persistence | joblib |
| Configuration | PyYAML + YAML files in `configs/` |
| Research figures | matplotlib |
| Interactive dashboard figures | Plotly (where it materially improves Streamlit UX) |
| Dashboard | Streamlit |
| Testing | pytest |
| Linting | Ruff |
| ERA5 acquisition (if selected) | `cdsapi` |

No heavyweight infrastructure (Airflow, Kubernetes, Ray, Spark, MLflow, DVC,
Hydra, databases) unless a concrete requirement justifies it and the user
approves.

## 2.2 Repository Layout

```text
ClimateTwin/
├── .kiro/
│   ├── steering/
│   ├── hooks/
│   ├── specs/
│   └── agents/
│
├── .kiroignore
│
├── configs/
│   ├── data.yaml
│   ├── lstm.yaml
│   ├── transformer.yaml
│   ├── pretrain.yaml
│   └── finetune.yaml
│
├── data/
│   ├── raw/          # immutable source data — never overwritten
│   ├── processed/    # derived clean/split/sequence-ready data
│   ├── sample/       # small representative data for fast tests
│   └── metadata/     # provenance, units, variable defs, retrieval params
│
├── docs/
│   └── research/
│       └── ClimateTwin_Research_Implementation_Prompt.md
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_experiments.ipynb
│
├── scripts/
│   ├── download_era5.py      # or download_<source>.py
│   ├── preprocess.py
│   ├── train_lstm.py
│   ├── train_transformer.py
│   ├── pretrain.py
│   ├── finetune.py
│   ├── evaluate.py
│   └── generate_figures.py
│
├── src/
│   ├── __init__.py
│   ├── config.py             # YAML loading, experiment config dataclasses
│   ├── seed.py               # central seed utility
│   ├── preprocessing.py      # cleaning, splitting, scaling
│   ├── dataset.py            # sequence construction, PyTorch Datasets
│   ├── train.py              # training loop, early stopping, logging
│   ├── evaluate.py           # MAE, RMSE, inverse transform, metric I/O
│   ├── forecasting.py        # autoregressive rollout
│   ├── scenario.py           # perturbation engine
│   ├── manifest.py           # run manifest creation/validation
│   └── models/
│       ├── __init__.py
│       ├── persistence.py
│       ├── lstm.py
│       └── transformer.py
│
├── tests/
│   ├── conftest.py           # shared fixtures, synthetic data generators
│   ├── test_smoke.py         # environment smoke tests (Phase 0)
│   ├── test_preprocessing.py
│   ├── test_dataset.py
│   ├── test_models.py
│   ├── test_forecasting.py
│   ├── test_evaluation.py
│   └── test_dashboard.py
│
├── dashboard/
│   └── app.py
│
├── results/
│   ├── figures/
│   ├── metrics/
│   ├── manifests/
│   ├── predictions/
│   └── checkpoints/
│
├── pyproject.toml
├── README.md
└── PROJECT_REVIEW.md
```

**Changes from v1.0:** Added `results/predictions/` to the canonical
layout. Added `.kiroignore` at root. Added `tests/test_smoke.py`.

## 2.3 Dependency Direction

```text
configs/ + data/
    ↓
src/ (reusable logic)
    ↓
scripts/ + notebooks/ + dashboard/
    ↓
results/
```

Dashboard depends on `src/` and saved `results/`. No reverse dependency.
No circular dependencies between models, training, forecasting, and
dashboard code.

## 2.4 Configuration System Design

Each config file is a YAML document containing at minimum the fields
relevant to its concern:

**`configs/data.yaml`:**

```yaml
source: "<UNRESOLVED: see §9.1>"
source_type: "<UNRESOLVED: station | reanalysis_grid_cell>"
version: "<UNRESOLVED>"
target_location: "<UNRESOLVED: see §9.2>"
latitude: null
longitude: null
variables: []           # UNRESOLVED: see §9.4
target_variables: []    # UNRESOLVED: see §9.13 — defaults to all variables
perturbable_features: [] # UNRESOLVED: subset of variables for scenario controls
sampling_interval: "<UNRESOLVED: see §9.3>"
time_range:
  start: "<UNRESOLVED: see §9.5>"
  end: "<UNRESOLVED>"
split:
  train_end: "<UNRESOLVED>"
  val_end: "<UNRESOLVED>"
  test_end: "<UNRESOLVED>"
input_window: null      # UNRESOLVED: see §9.7
scaler_type: "StandardScaler"
```

**`configs/lstm.yaml`:**

```yaml
n_features: null        # set from data config
n_targets: null         # set from data config / §9.13
input_window: null      # inherits or overrides data.yaml
hidden_dim: null        # UNRESOLVED: see §9.11
num_layers: null
dropout: null
learning_rate: null
batch_size: null
epochs: null
early_stopping:
  patience: null
  metric: "val_loss"
seed: 42
device: "auto"
checkpoint_dir: "results/checkpoints"
metrics_dir: "results/metrics"
predictions_dir: "results/predictions"
```

**`configs/transformer.yaml`:**

```yaml
n_features: null        # set from data config
n_targets: null         # set from data config / §9.13
d_model: null           # UNRESOLVED: see §9.6 — starting region 64/128
nhead: null             # starting region 4/8
num_encoder_layers: null  # starting region 2–4
dropout: 0.1
learning_rate: null
batch_size: null
epochs: null
early_stopping:
  patience: null
  metric: "val_loss"
seed: 42
device: "auto"
checkpoint_dir: "results/checkpoints"
metrics_dir: "results/metrics"
predictions_dir: "results/predictions"
```

**`configs/pretrain.yaml`:**

```yaml
pretrain_locations: []   # UNRESOLVED: see §9.8
exclude_target_location: true  # default: exclude target from pretraining
pretrain_data_source: "<UNRESOLVED>"
architecture: "transformer"  # must match transformer.yaml architecture
normalization_policy: "<UNRESOLVED: see §9.14>"
learning_rate: null
batch_size: null
epochs: null
early_stopping:
  patience: null
  metric: "val_loss"
pretrain_split:
  method: "chronological_per_location"
  val_fraction: null
seed: 42
device: "auto"
checkpoint_dir: "results/checkpoints"
```

**`configs/finetune.yaml`:**

```yaml
pretrain_checkpoint: null    # path to validated pretrain checkpoint
target_location: "<UNRESOLVED>"
normalization_policy: "<UNRESOLVED: see §9.14>"
scaler_path: null            # which scaler to use for fine-tuning
learning_rate: null          # typically smaller than pretrain LR
batch_size: null
epochs: null
early_stopping:
  patience: null
  metric: "val_loss"
frozen_layers: []            # document frozen/unfrozen/reinitialized
unfrozen_layers: []
reinitialized_layers: []
seed: 42
device: "auto"
checkpoint_dir: "results/checkpoints"
```

## 2.5 Run Manifest Schema

Each serious run produces a JSON manifest:

```json
{
  "run_id": "<model>_<stage>_<location>_seed<N>_<seq>",
  "stage": "baseline | scratch | pretrain | finetune | evaluate",
  "model": "persistence | lstm | transformer",
  "seed": 42,
  "device": "cpu | cuda",
  "dataset": {
    "source": "...",
    "source_type": "station | reanalysis_grid_cell",
    "source_version": "...",
    "target_location": "...",
    "latitude": null,
    "longitude": null,
    "variables": [],
    "target_variables": [],
    "units": {},
    "sampling_interval": "...",
    "train_period": "...",
    "validation_period": "...",
    "test_period": "..."
  },
  "preprocessing": {
    "scaler_type": "...",
    "scaler_path": "...",
    "normalization_policy": "...",
    "feature_order": [],
    "target_order": [],
    "input_window": null
  },
  "model_config": {
    "n_features": null,
    "n_targets": null
  },
  "training_config": {
    "best_epoch": null,
    "wall_clock_seconds": null,
    "convergence_epoch": null,
    "convergence_definition": "..."
  },
  "hyperparameter_selection": {
    "method": "validation-based",
    "candidates_evaluated": [],
    "selection_rationale": "..."
  },
  "checkpoint_path": "...",
  "metrics_path": "...",
  "predictions_path": "...",
  "figures": [],
  "software": {
    "python": "...",
    "pytorch": "...",
    "numpy": "...",
    "pandas": "..."
  },
  "git_commit": "...",
  "created_at": "ISO-8601 timestamp"
}
```

## 2.6 Seed Utility Design

`src/seed.py` provides a single function:

```text
set_global_seed(seed: int) → None
    Seeds: random, numpy, torch (CPU + CUDA if available)
    Sets torch deterministic flags where practical
    Logs the seed value
```

Called at the start of every training/evaluation script.

## 2.7 Evaluation Module Design

`src/evaluate.py` provides two tiers of functionality:

### Tier 1 — Evaluation Core (available from Phase 3 onward)

```text
compute_mae(y_true, y_pred) → float
compute_rmse(y_true, y_pred) → float
inverse_transform_predictions(y_scaled, scaler, feature_names) → y_original
compute_per_variable_metrics(
    y_true, y_pred, variable_names
) → dict[str, dict[str, float]]
    Returns {"temperature_2m": {"mae": ..., "rmse": ...}, ...}
align_predictions_with_ground_truth(
    predictions, targets, timestamps
) → aligned_dict
save_metrics(metrics: dict, path: str) → None
load_metrics(path: str) → dict
```

All per-variable metrics are computed in original physical units after
inverse transformation. No mixed-unit aggregate physical metric is computed
as a primary result.

### Tier 2 — Extended Evaluation (Phase 6)

```text
compute_r2(y_true, y_pred) → float           # optional
evaluate_checkpoint(
    checkpoint_path, model_type, config, scaler, test_data
) → dict
    Generic evaluator that loads any model type, runs inference,
    computes per-variable metrics, saves aligned predictions.
```

This two-tier design avoids duplication: Tier 1 functions are implemented
once and reused by both the baseline phase and all later evaluations.

## 2.8 Autoregressive Rollout Design

`src/forecasting.py` provides:

```text
autoregressive_forecast(
    model,
    initial_window: Tensor,       # shape: (input_window, n_features)
    horizon: int,
    feature_names: list[str],
    target_indices: list[int],    # indices of recursively predicted features
    exogenous_future: Tensor | None  # pre-supplied exogenous features if any
) → Tensor                        # shape: (horizon, n_targets)
```

**Critical constraint:** For features not recursively predicted by the
model, the rollout must explicitly define how they are supplied. If future
exogenous values are not available, the implementation must raise an error
or use a documented fallback — never silently inject ground-truth future
observations.

When `n_targets == n_features` (per §9.13 preferred default), all features
are recursively predicted and `exogenous_future` is `None`.

## 2.9 Scenario Module Design

`src/scenario.py` provides:

```text
get_perturbable_features(
    feature_names: list[str],
    configured_perturbable: list[str]
) → list[str]
    Returns the intersection of model features and configured perturbable
    features. Raises an error if the intersection is empty.

apply_perturbation(
    baseline_input: Tensor,
    perturbations: dict[str, float],  # e.g. {"temperature_2m": +2.0}
    feature_names: list[str],
    scaler                            # for unit-aware perturbation
) → Tensor

run_scenario(
    model,
    baseline_input: Tensor,
    perturbations: dict[str, float],
    horizon: int,
    feature_names: list[str],
    ...
) → tuple[Tensor, Tensor]           # (baseline_trajectory, perturbed_trajectory)
```

Perturbable features are determined dynamically from model features and
`configs/data.yaml::perturbable_features`. No hard-coded variable names.

The scenario module must attach the disclaimer text to any output.

## 2.10 Dashboard Architecture

```text
dashboard/app.py
    ├── page: Climate Overview    → reads data/processed/, data/metadata/
    ├── page: AI Forecast         → reads results/checkpoints/, results/metrics/,
    │                                results/predictions/
    ├── page: Multi-Step Forecast → reads results/metrics/ (horizon-wise),
    │                                results/predictions/
    ├── page: Model Comparison    → reads results/metrics/, results/figures/
    └── page: Scenario Sensitivity → loads model from results/checkpoints/,
                                     uses src/scenario.py,
                                     builds controls from get_perturbable_features(),
                                     displays disclaimer
```

All pages import reusable logic from `src/`. No training logic in dashboard.
Expensive inference (scenario) cached via `@st.cache_data` or
`@st.cache_resource` as appropriate.

**Model availability policy:** In development mode, if a model's artifacts
are not yet available, the dashboard displays "Model X: not yet trained —
artifacts unavailable" and disables that model's selection. It never
substitutes placeholder metrics or fabricated results. In research/final
mode, all four required models must have real artifacts.

## 2.11 Data Provenance Schema

`data/metadata/provenance.yaml` includes raw-file fingerprints:

```yaml
source: "..."
source_type: "station | reanalysis_grid_cell"
dataset_name: "..."
retrieval_date: "..."
source_version: "..."
geographic_area: "..."
latitude: null
longitude: null
variables_requested: []
original_units: {}
time_range: {start: "...", end: "..."}
sampling_interval: "..."
retrieval_parameters: {}
license: "..."
raw_files:
  - filename: "..."
    sha256: "..."
    size_bytes: null
    retrieved_at: "ISO-8601"
```


---

# PART 3 — TASK / DEPENDENCY PLAN

## Dependency Graph Overview

```text
Phase 0  (foundation)
   ↓
Phase 1  (data acquisition)
   ↓
Phase 2  (preprocessing)
   ↓
Phase 3  (evaluation core + persistence baseline)
   ↓
Phase 4  (LSTM)
   ↓
Phase 5  (Transformer scratch)
   ↓
Phase 6  (extended evaluation — generic evaluator, R², ACC gate)
   ↓
Phase 7  (autoregressive rollout — all models at all horizons)
   ↓
Phase 8  (pretraining — requires Phase 5 architecture + Phase 2 pipeline)
   ↓
Phase 9  (fine-tuning — requires Phase 8 checkpoint)
   ↓
Phase 10 (scratch vs transfer comparison — requires Phase 5 + Phase 9)
   ↓
Phase 11 (figures — requires Phases 3,4,5,7,10 metrics)
   ↓
Phase 12 (extreme event — optional, requires Phase 7 predictions in test period)
   ↓
Phase 13 (scenario — requires at least one trained model)
   ↓
Phase 14 (dashboard — requires all four model artifacts for research mode)
   ↓
Phase 15 (final audit)
```

**Key change from v1.0:** Evaluation Core (Tier 1: MAE, RMSE, inverse
transform, metric I/O) is now built as part of Phase 3 so the persistence
baseline can use it immediately. Phase 6 is retained for the extended
evaluator, per-variable reporting refinements, optional R², and the ACC
decision gate. No evaluation logic is duplicated.

---

## Phase 0 — Reproducible Project Foundation

**Depends on:** Nothing.

### Task 0.1 — Initialize repository structure

Create all directories per §2.2, including `results/predictions/`.
Create `pyproject.toml` with project metadata, dependencies (pinned or
bounded versions), and tool configuration (Ruff, pytest).

### Task 0.2 — Configuration system

Implement `src/config.py`:

- Load YAML config files.
- Provide typed dataclasses or dictionaries for data config, LSTM config,
  Transformer config, pretrain config, finetune config.
- Validate required fields; raise actionable errors for missing values.
- Validate `n_features`/`n_targets` consistency with variable lists.

Create initial config templates in `configs/` with `UNRESOLVED` placeholders
for dataset-dependent fields. Include `target_variables`,
`perturbable_features`, `source_type`, and `normalization_policy` fields.

### Task 0.3 — Seed utility

Implement `src/seed.py` with `set_global_seed(seed)` seeding Python
`random`, NumPy, and PyTorch.

### Task 0.4 — Manifest utility

Implement `src/manifest.py`:

- `create_manifest(...)` → dict conforming to §2.5 schema (including
  `source_type`, `n_features`, `n_targets`, `target_variables`,
  `best_epoch`, `wall_clock_seconds`, `hyperparameter_selection`).
- `save_manifest(manifest, path)` → writes JSON.
- `load_manifest(path)` → reads and validates JSON.

### Task 0.5 — Logging setup

Configure Python logging for scripts with timestamps, levels, and module
names. Log to console and optionally to file.

### Task 0.6 — `.gitignore`

Exclude: `data/raw/`, `data/processed/`, `results/checkpoints/`,
`results/predictions/`, `__pycache__/`, `.env`, `.cdsapirc`, `*.egg-info/`,
virtual environments, large generated files.

### Task 0.7 — `.kiroignore`

Create `.kiroignore` at workspace root to control which files Kiro indexes.

**Exclude (large/binary/sensitive):**

```text
data/raw/**
data/processed/**/*.nc
data/processed/**/*.grib
data/processed/**/*.parquet
results/checkpoints/**
results/predictions/**/*.parquet
results/predictions/**/*.csv
*.pt
*.pth
.env
.cdsapirc
__pycache__/
.venv/
env/
```

**Keep available to Kiro (small/metadata):**

```text
data/metadata/**
data/sample/**
results/manifests/**
results/metrics/**
configs/**
```

### Task 0.8 — Smoke-test infrastructure

Create `tests/conftest.py` with:

- Synthetic time-series data fixtures (small, deterministic).
- Temporary directory fixtures.
- Helper to generate a minimal valid config dict.

Create `tests/test_smoke.py` with at least one actual environment test:

- Verify core imports succeed (torch, numpy, pandas, sklearn, yaml).
- Verify `set_global_seed` runs without error.
- Verify `create_manifest` produces a valid dict.

Write initial `pytest` configuration in `pyproject.toml`.

Verify: `pytest` collects and passes at least one test from
`tests/test_smoke.py`.

### Phase 0 Acceptance Criteria

- [ ] All directories exist per §2.2, including `results/predictions/`.
- [ ] `pyproject.toml` lists all core dependencies with pinned or bounded versions.
- [ ] `src/config.py` loads YAML and validates required fields including
      `n_features`/`n_targets`.
- [ ] `src/seed.py` seeds Python/NumPy/PyTorch deterministically.
- [ ] `src/manifest.py` creates, saves, and loads valid manifests conforming
      to §2.5 schema.
- [ ] `tests/conftest.py` provides synthetic data fixtures.
- [ ] `tests/test_smoke.py` contains at least one passing test.
- [ ] `pytest` collects ≥1 test and all collected tests pass.
- [ ] `.gitignore` excludes specified patterns.
- [ ] `.kiroignore` excludes large binaries and keeps metadata accessible.
- [ ] Ruff passes on all `src/` files.

---

## Phase 1 — Dataset Acquisition and Validation

**Depends on:** Phase 0.

### Task 1.1 — Dataset selection research

**MCP usage:** Use **Tavily** to search for current ERA5/ERA5-Land API
access, weather-station data availability for Indian cities, and any
access restrictions. Use **Fetch** to inspect official CDS documentation
or dataset landing pages.

Produce a written decision record in `data/metadata/dataset_selection.md`
documenting:

- Candidate datasets considered.
- Selected dataset and rationale.
- Source URL / provider.
- License / access terms.
- **Data type: station observation or reanalysis grid cell** — with accurate
  terminology. Do not describe an ERA5/ERA5-Land grid point as a weather
  station.
- Available variables.
- Geographic coverage.
- Temporal coverage.
- Sampling interval.
- Known limitations.

**Decision point resolution:** This task resolves §9.1 (dataset), §9.2
(target location and type), §9.3 (sampling interval), §9.4 (selected
variables).

### Task 1.2 — Data acquisition script

Implement `scripts/download_era5.py` (or `download_<source>.py`):

- Reads `configs/data.yaml` for retrieval parameters.
- Downloads raw data to `data/raw/`.
- Does not modify raw data.
- Logs retrieval parameters.
- Computes SHA-256 checksums and file sizes for downloaded files.

If using `cdsapi`, use **Context7** to verify current `cdsapi` API usage.

### Task 1.3 — Raw data inspection and EDA

Load raw data, produce initial inspection:

- Row/timestamp count.
- Variable inventory.
- Date range.
- Sampling interval detection.
- Missing-value summary by variable.
- Duplicate timestamp detection.
- Basic statistical summary.
- Geographic metadata.
- Confirmation of data type (station vs reanalysis grid cell).

Record findings in `data/metadata/raw_data_report.json` or equivalent
machine-readable format, and optionally in `notebooks/01_eda.ipynb`.

### Task 1.4 — Data provenance metadata

Write `data/metadata/provenance.yaml` per §2.11 schema, including:

- `source_type: station | reanalysis_grid_cell`
- Raw file fingerprints (SHA-256, file size, retrieval timestamp).

### Task 1.5 — Update configs with resolved values

Update `configs/data.yaml` replacing `UNRESOLVED` placeholders with actual
dataset-derived values for: source, source_type, target location, variables,
target_variables, perturbable_features, sampling interval, time range.

### Phase 1 Acceptance Criteria

- [ ] Dataset selection decision is documented in `data/metadata/`.
- [ ] Data type (station vs reanalysis) is explicitly recorded.
- [ ] Raw data exists in `data/raw/` and is unmodified.
- [ ] Provenance metadata with SHA-256 checksums is recorded.
- [ ] Variables, geography, timestamps, sampling interval are validated.
- [ ] Missing data and duplicates are quantified.
- [ ] `configs/data.yaml` contains resolved values for source, source_type,
      location, variables, target_variables, and sampling interval.
- [ ] No UNRESOLVED placeholders remain for dataset-dependent fields that
      can now be determined.

---

## Phase 2 — Preprocessing and Chronological Split

**Depends on:** Phase 1.

### Task 2.1 — Cleaning implementation

Implement in `src/preprocessing.py`:

- `remove_duplicates(df) → df` — log count removed.
- `handle_missing_values(df, strategy) → df` — document strategy, log count
  affected, never impute training data from future periods.
- `validate_units(df, expected_units) → df` — check physical plausibility,
  record conversions.
- `validate_timestamps(df, expected_interval) → df` — parse, sort, detect
  gaps, detect irregular sampling.

### Task 2.2 — Chronological split implementation

Implement in `src/preprocessing.py`:

- `chronological_split(df, train_end, val_end) → (train_df, val_df, test_df)`

Assertions enforced inside the function:

```text
assert max(train_df.timestamp) < min(val_df.timestamp)
assert max(val_df.timestamp) < min(test_df.timestamp)
assert len(set(train_df.index) & set(val_df.index)) == 0
assert len(set(val_df.index) & set(test_df.index)) == 0
assert len(set(train_df.index) & set(test_df.index)) == 0
```

**Decision point resolution:** This task triggers resolution of §9.5
(split date boundaries) based on the actual dataset time range.

### Task 2.3 — Normalization implementation

Implement in `src/preprocessing.py`:

- `fit_scaler(train_df, scaler_type, feature_names) → scaler`
- `transform_data(df, scaler) → df_scaled`
- `save_scaler(scaler, feature_names, metadata, path)`
- `load_scaler(path) → (scaler, feature_names, metadata)`

Invariant: `fit_scaler` is called once on `train_df`. `transform_data` is
called on train, val, and test with the same already-fitted scaler.

**Note:** The normalization policy decision (§9.14) determines which scaler
is used for each experimental stage. This implementation provides the
machinery; the policy is configured externally.

### Task 2.4 — Sequence construction

Implement in `src/dataset.py`:

- `ClimateSequenceDataset(data, input_window, target_indices, feature_names)`

  A PyTorch `Dataset` producing `(input_tensor, target_tensor)` pairs.

- Boundary-aware: constructor accepts optional `max_timestamp` to prevent
  sequences from crossing split boundaries.
- Assert: no target timestamp exceeds the split's allowed range.
- `target_indices` defaults to all feature indices when `n_targets == n_features`.

### Task 2.5 — Multi-location sequence construction (for pretraining)

Extend `src/dataset.py`:

- `MultiLocationDataset(location_dataframes, input_window, ...)`
- Group/sort by location and time before sequence construction.
- Never create sequences whose adjacent rows span different stations.
- Apply chronological train/validation split per location, preserving
  temporal ordering within each location.

### Task 2.6 — Preprocessing script

Implement `scripts/preprocess.py`:

- Reads `configs/data.yaml`.
- Loads raw data.
- Runs cleaning pipeline.
- Splits chronologically.
- Fits scaler on training data.
- Saves processed splits to `data/processed/`.
- Saves scaler artifact.
- Produces `data/metadata/preprocessing_report.json`.

### Task 2.7 — Leakage and alignment tests

Implement in `tests/test_preprocessing.py` and `tests/test_dataset.py`:

- Chronological ordering assertion.
- No train/val/test timestamp overlap.
- Scaler fitted only on training data.
- Sequence input/target alignment (off-by-one checks).
- No sequence target crosses split boundary.
- Feature ordering stability across splits.
- Duplicate handling correctness.
- Sampling-interval detection.
- Inverse transform roundtrip consistency.

Implement in `tests/test_dataset.py`:

- Multi-location sequence: no cross-station jumps.
- Multi-location: chronological ordering preserved within each location.
- Dataset length correctness.
- Tensor shape validation.
- `n_targets` consistency with target_indices.

### Phase 2 Acceptance Criteria

- [ ] Cleaning functions handle duplicates, missing values, invalid values,
      units.
- [ ] Chronological split enforces strict temporal ordering with assertions.
- [ ] Scaler is fit on training data only and persisted with metadata.
- [ ] Sequence datasets produce correct `(input, target)` pairs.
- [ ] Multi-location dataset prevents cross-station sequences and preserves
      chronological ordering per location.
- [ ] Processed data saved to `data/processed/`.
- [ ] Preprocessing report generated.
- [ ] All leakage/alignment tests pass.
- [ ] Split date boundaries are documented in `configs/data.yaml`.
- [ ] Ruff passes.

---

## Phase 3 — Evaluation Core and Persistence Baseline

**Depends on:** Phase 2.

### Task 3.1 — Evaluation Core implementation (Tier 1)

Implement in `src/evaluate.py`:

- `compute_mae(y_true, y_pred) → float`
- `compute_rmse(y_true, y_pred) → float`
- `inverse_transform_predictions(y_scaled, scaler, feature_names) → y_original`
- `compute_per_variable_metrics(y_true, y_pred, variable_names) → dict`
  — returns per-variable MAE and RMSE in original physical units.
- `align_predictions_with_ground_truth(predictions, targets, timestamps) → dict`
- `save_metrics(metrics, path) → None`
- `load_metrics(path) → dict`

No mixed-unit aggregate physical metric is computed as primary output.

### Task 3.2 — Evaluation Core tests

Implement in `tests/test_evaluation.py`:

- MAE correctness on known values.
- RMSE correctness on known values.
- Inverse transform roundtrip.
- Per-variable metric structure and content.
- Metric JSON save/load roundtrip.
- Verify no mixed-unit aggregate is produced as primary metric.

### Task 3.3 — Persistence model

Implement `src/models/persistence.py`:

- `PersistenceModel` that implements `prediction(t+1) = observed_state(t)`
  for all target variables.
- Conforms to the same prediction interface as learned models where
  practical.

### Task 3.4 — Persistence evaluation

- Generate persistence predictions on the test split.
- Compute per-variable MAE, RMSE in original physical units using Tier 1
  evaluation functions.
- Save predictions to `results/predictions/`.
- Save metrics to `results/metrics/persistence_<location>_metrics.json`.

### Task 3.5 — Persistence run manifest

Create and save manifest per §2.5 schema.

### Task 3.6 — Persistence smoke test

In `tests/test_models.py`:

- Verify persistence predictions equal the last observation for each
  variable.
- Verify output shape matches `n_targets`.

### Phase 3 Acceptance Criteria

- [ ] Evaluation Core (Tier 1) implemented: MAE, RMSE, inverse transform,
      per-variable metrics, alignment, metric I/O.
- [ ] Evaluation Core tests pass.
- [ ] Persistence predictions generated on test set.
- [ ] Per-variable MAE and RMSE computed in original units.
- [ ] Predictions saved to `results/predictions/`.
- [ ] Metrics saved as machine-readable JSON.
- [ ] Run manifest saved.
- [ ] Persistence smoke test passes.

---

## Phase 4 — LSTM

**Depends on:** Phase 3 (evaluation core + baseline comparison context).

### Task 4.1 — LSTM model implementation

Implement `src/models/lstm.py`:

- `ClimateLSTM(n_features, n_targets, hidden_dim, num_layers, dropout)` —
  PyTorch `nn.Module`.
- Input shape: `(batch, input_window, n_features)`.
- Output shape: `(batch, n_targets)`.
- Docstring with shape annotations.

### Task 4.2 — LSTM shape/smoke tests

In `tests/test_models.py`:

- Forward pass with synthetic data.
- Output shape check (verify `n_targets` dimensionality).
- Loss computation produces finite value.
- After one optimizer step: verify at least one trainable parameter changed
  and all gradients are finite and non-null.
- Checkpoint save/load roundtrip: put model in eval mode, verify identical
  output from loaded checkpoint.

### Task 4.3 — LSTM training script

Implement `scripts/train_lstm.py`:

- Reads `configs/lstm.yaml` and `configs/data.yaml`.
- Seeds environment.
- Loads processed data and scaler.
- Creates sequence datasets (train, val).
- Training loop with validation, logging train/val loss per epoch.
- Records best epoch and wall-clock training duration.
- Early stopping on validation loss.
- Saves best checkpoint.
- Saves training log (JSON with per-epoch losses, best_epoch,
  wall_clock_seconds).
- Saves run manifest including hyperparameter_selection record.
- Uses validation data only for model selection; test data untouched.

### Task 4.4 — LSTM test evaluation

- Load best checkpoint.
- Generate test-set predictions.
- Compute per-variable MAE, RMSE in original units.
- Save predictions to `results/predictions/`.
- Save metrics to `results/metrics/`.
- Update or create evaluation manifest.

### Phase 4 Acceptance Criteria

- [ ] LSTM forward pass produces correct output shape (`n_targets`).
- [ ] Smoke tests pass (shape, finite loss, parameter change, finite
      gradients, eval-mode checkpoint roundtrip).
- [ ] Training runs to completion with early stopping.
- [ ] Training/validation loss log saved with best_epoch and wall_clock_seconds.
- [ ] Best checkpoint saved.
- [ ] Per-variable test MAE and RMSE computed in original units.
- [ ] Predictions saved to `results/predictions/`.
- [ ] Metrics saved.
- [ ] Run manifest saved with hyperparameter selection record.
- [ ] Ruff passes.

---

## Phase 5 — Lightweight Transformer (Scratch)

**Depends on:** Phase 3 (evaluation core), Phase 4 (for progressive
comparison context).

### Task 5.1 — Transformer model implementation

Implement `src/models/transformer.py`:

- `ClimateTransformer(n_features, n_targets, d_model, nhead,
  num_encoder_layers, dropout)` — PyTorch `nn.Module`.
- Architecture: feature projection → positional encoding → Transformer
  encoder → regression head.
- Input shape: `(batch, input_window, n_features)`.
- Output shape: `(batch, n_targets)`.
- Record parameter count.
- Docstring with shape annotations.

**Decision point:** Exact hyperparameters are unresolved (§9.6) but the
architecture must be implemented to accept them as constructor arguments.

### Task 5.2 — Transformer shape/smoke tests

In `tests/test_models.py`:

- Forward pass with synthetic data.
- Output shape check (verify `n_targets` dimensionality).
- Loss computation produces finite value.
- After one optimizer step: verify at least one trainable parameter changed
  and all gradients are finite and non-null.
- Checkpoint save/load roundtrip: put model in eval mode, verify identical
  output from loaded checkpoint.
- Parameter count logging.

### Task 5.3 — Transformer scratch training script

Implement `scripts/train_transformer.py`:

- Reads `configs/transformer.yaml` and `configs/data.yaml`.
- Seeds environment.
- Loads processed data and scaler.
- Creates sequence datasets.
- Training with validation, logging, early stopping.
- Records best epoch and wall-clock training duration.
- Saves best checkpoint.
- Saves training log.
- Saves run manifest with `stage: "scratch"` and hyperparameter selection.

### Task 5.4 — Transformer scratch test evaluation

- Load best checkpoint.
- Generate test-set predictions.
- Compute per-variable MAE, RMSE in original units.
- Save predictions to `results/predictions/`.
- Save metrics.

### Phase 5 Acceptance Criteria

- [ ] Transformer forward pass produces correct output shape (`n_targets`).
- [ ] Smoke tests pass (shape, finite loss, parameter change, finite
      gradients, eval-mode checkpoint roundtrip).
- [ ] Scratch training runs to completion with early stopping.
- [ ] Training/validation loss log saved with best_epoch and wall_clock_seconds.
- [ ] Best checkpoint saved.
- [ ] Per-variable test MAE and RMSE computed in original units.
- [ ] Predictions saved to `results/predictions/`.
- [ ] Metrics saved.
- [ ] Run manifest saved with `stage: "scratch"` and hyperparameter selection.
- [ ] Parameter count recorded.

---

## Phase 6 — Extended Evaluation Utilities

**Depends on:** Phase 3 (Evaluation Core already exists). Practically
independent of model phases but logically extends Tier 1.

### Task 6.1 — Extended evaluation module (Tier 2)

Extend `src/evaluate.py` with:

- `compute_r2(y_true, y_pred) → float` (optional, per-variable only).
- `evaluate_checkpoint(checkpoint_path, model_type, config, scaler,
  test_data) → dict` — generic evaluator loading any model type.

No duplication of Tier 1 functions. Tier 2 calls Tier 1 internally.

### Task 6.2 — Evaluation script

Implement `scripts/evaluate.py`:

- Accepts a checkpoint path, model type, and config.
- Loads model, scaler, and test data.
- Runs inference.
- Computes and saves per-variable metrics.
- Generates aligned prediction/ground-truth output for figure generation.
- Saves to `results/predictions/` and `results/metrics/`.

### Task 6.3 — Extended evaluation tests

In `tests/test_evaluation.py` (extending Tier 1 tests):

- R² correctness on known values (if R² implemented).
- Generic evaluator produces correct metrics for a mock model.

### Task 6.4 — ACC decision gate

**Decision point:** Evaluate whether a correct anomaly climatology is
available for the selected dataset and location. Document the decision.

- If yes: implement ACC with documented climatological reference.
- If no: document that ACC is outside scope. Do not implement.

This gate resolves §9.9.

### Phase 6 Acceptance Criteria

- [ ] Tier 2 extends `src/evaluate.py` without duplicating Tier 1 logic.
- [ ] `scripts/evaluate.py` produces per-variable metrics from any saved
      checkpoint.
- [ ] Extended evaluation tests pass.
- [ ] ACC decision documented.

---

## Phase 7 — Autoregressive Rollout

**Depends on:** At least one trained model checkpoint (Phase 4 or 5),
Phase 3 (evaluation core), Phase 6 (generic evaluator for convenience).

### Task 7.1 — Rollout implementation

Implement `src/forecasting.py` per §2.8:

- `autoregressive_forecast(model, initial_window, horizon,
  feature_names, target_indices, exogenous_future)`.
- Explicit handling of non-predicted features (raise error or use
  documented fallback).
- Never inject future ground-truth silently.
- When `n_targets == n_features`, all features are recursively updated.

### Task 7.2 — Horizon-wise evaluation for all models

For **all four required model variants** (Persistence, LSTM, scratch
Transformer, pretrained+fine-tuned Transformer — the last only after
Phase 9 is complete):

- Run rollout at horizons: 1, 3, 6, 12, 24 steps.
- Compute per-horizon, per-variable MAE and RMSE in original units.
- Save horizon-wise metrics to
  `results/metrics/<model>_horizon_metrics.json`.
- Save horizon-wise predictions to `results/predictions/`.

For the persistence baseline, autoregressive rollout means repeating the
persisted state for all horizons (by definition, persistence at horizon h
still predicts the last observed state).

If any model is technically excluded from horizon evaluation, document the
reason explicitly.

**Note:** Pretrained+fine-tuned Transformer horizon evaluation happens after
Phase 9, but this task specification covers the requirement for all models.

### Task 7.3 — Rollout tests

In `tests/test_forecasting.py`:

- Rollout length matches requested horizon.
- Output shape correctness (including `n_targets` dimensionality).
- Prediction at horizon=1 matches one-step inference.
- No future ground-truth leakage (mock exogenous features and verify they
  are not silently filled from targets).
- Feature update logic correctness.
- When `n_targets == n_features`, verify all features are updated.

### Phase 7 Acceptance Criteria

- [ ] Autoregressive rollout produces correct-length forecasts.
- [ ] Non-predicted feature handling is explicit and documented.
- [ ] Horizon-wise per-variable metrics saved (1, 3, 6, 12, 24 steps) for
      all available models.
- [ ] All rollout tests pass.
- [ ] Steps are not labeled as hours unless sampling interval is validated.
- [ ] Any model excluded from horizon evaluation is documented with reason.

---

## Phase 8 — Broad-Data Pretraining

**Depends on:** Phase 5 (Transformer architecture), Phase 2 (preprocessing
pipeline).

### Task 8.1 — Pretraining data acquisition

**Decision point resolution:** This task resolves §9.8 (pretraining
geography/locations).

**MCP usage:** Use **Tavily** to research availability of multi-location
data from the selected source. Use **Fetch** to inspect specific data
catalog pages.

- Select multiple locations or a broader region.
- **Preferably exclude the target location entirely from pretraining data.**
  At minimum, prohibit target-location validation/test-period observations
  from entering pretraining.
- Document the selection rationale.
- Download raw data for pretraining locations.
- Record provenance for each location (including SHA-256 checksums).

### Task 8.2 — Pretraining data preprocessing

- Apply the same cleaning pipeline per location.
- **Apply chronological train/validation split within each location:**
  for each pretraining location, split its data chronologically into
  pretraining-train and pretraining-validation periods. Never randomly
  shuffle within a location.
- Use the multi-location sequence constructor (Task 2.5).
- Maintain location boundaries — no cross-station sequence jumps.
- **Fit a pretraining scaler on pretraining training data** (per
  normalization policy §9.14).

### Task 8.3 — Transfer contamination tests

Implement in `tests/test_dataset.py` or `tests/test_preprocessing.py`:

- Verify target location is excluded from pretraining data (if policy is
  full exclusion).
- Verify target-location validation/test-period timestamps do not appear in
  pretraining data (minimum requirement regardless of exclusion policy).
- Verify pretraining data has chronological ordering within each location.

### Task 8.4 — Pretraining script

Implement `scripts/pretrain.py`:

- Reads `configs/pretrain.yaml`.
- Uses the same `ClimateTransformer` architecture from
  `src/models/transformer.py` (same `n_features`, `n_targets`, `d_model`,
  `nhead`, `num_encoder_layers`).
- Trains on multi-location data.
- Logs training/validation loss per epoch.
- Records best epoch and wall-clock training duration.
- Saves best pretrained checkpoint.
- Saves training log.
- Saves run manifest with `stage: "pretrain"`.

### Task 8.5 — Pretrain checkpoint validation

- Load the saved checkpoint.
- Verify architecture matches `configs/transformer.yaml`.
- Run a forward pass.
- Confirm the checkpoint is usable for fine-tuning.

### Phase 8 Acceptance Criteria

- [ ] Pretraining locations/region documented.
- [ ] Target location excluded from pretraining (preferred) or at minimum
      target val/test periods excluded.
- [ ] Transfer contamination tests pass.
- [ ] Pretraining data has chronological train/val split per location.
- [ ] Multi-location data preprocessed without cross-station sequence leakage.
- [ ] Pretraining runs to completion.
- [ ] Training/validation loss log saved with best_epoch and wall_clock_seconds.
- [ ] Pretrained checkpoint saved and validated (loadable, correct arch).
- [ ] Run manifest saved with `stage: "pretrain"`.

---

## Phase 9 — Target-Location Fine-Tuning

**Depends on:** Phase 8 (valid pretrained checkpoint).

**Phase gate:** Do not begin fine-tuning until Phase 8 checkpoint exists
and has been validated.

### Task 9.1 — Fine-tuning script

Implement `scripts/finetune.py`:

- Reads `configs/finetune.yaml`.
- Loads pretrained checkpoint.
- Documents: parameters loaded, frozen, unfrozen, reinitialized.
- Uses target-location training data (same split as scratch training).
- **Uses the scaler specified by the normalization policy (§9.14).**
- Smaller learning rate (configurable).
- Early stopping.
- Records best epoch and wall-clock training duration.
- Saves fine-tuned checkpoint.
- Saves training log.
- Saves run manifest with `stage: "finetune"`.

### Task 9.2 — Fine-tuned model test evaluation

- Load fine-tuned checkpoint.
- Generate test-set predictions on same target-location test set used for
  scratch evaluation.
- **Use the scaler appropriate for inverse transformation per §9.14.**
- Compute per-variable MAE, RMSE in original units.
- Save predictions to `results/predictions/` and metrics.
- Run autoregressive evaluation at horizons 1, 3, 6, 12, 24.
- Save horizon-wise metrics.

### Phase 9 Acceptance Criteria

- [ ] Fine-tuning loads a validated pretrained checkpoint.
- [ ] Parameter handling (loaded/frozen/unfrozen/reinitialized) documented.
- [ ] Normalization policy followed per §9.14.
- [ ] Fine-tuning runs to completion with early stopping.
- [ ] Fine-tuned checkpoint saved.
- [ ] Per-variable test MAE and RMSE computed in original units.
- [ ] Horizon-wise metrics saved.
- [ ] Run manifest saved with `stage: "finetune"` and best_epoch,
      wall_clock_seconds.


---

## Phase 10 — Fair Scratch vs Transfer Comparison

**Depends on:** Phase 5 (scratch metrics), Phase 9 (fine-tuned metrics).

**Phase gate:** Both scratch and fine-tuned checkpoints must exist with
valid metrics before comparison.

### Task 10.1 — Comparison validation

Verify before comparison:

- Same Transformer architecture (same `n_features`, `n_targets`, `d_model`,
  `nhead`, `num_encoder_layers`).
- Same target-location test set.
- Same evaluation implementation.
- Normalization policy documented and evaluation fairness confirmed per §9.14.
- Seeds documented.

### Task 10.2 — Comparison report

Produce `results/metrics/scratch_vs_transfer_comparison.json`:

```json
{
  "scratch": {
    "run_id": "...",
    "test_mae_per_variable": {},
    "test_rmse_per_variable": {},
    "best_val_loss": null,
    "best_epoch": null,
    "wall_clock_seconds": null,
    "convergence_epoch": null,
    "convergence_definition": "..."
  },
  "transfer": {
    "pretrain_run_id": "...",
    "finetune_run_id": "...",
    "test_mae_per_variable": {},
    "test_rmse_per_variable": {},
    "best_val_loss": null,
    "best_epoch": null,
    "wall_clock_seconds": null,
    "convergence_epoch": null,
    "convergence_definition": "..."
  },
  "normalization_policy": "...",
  "comparison_notes": "..."
}
```

### Task 10.3 — Comparison interpretation

Document the observed result honestly:

- If transfer wins: report by how much on which per-variable metrics.
- If scratch wins or ties: report as a valid scientific finding.
- Do not fabricate an improvement.

### Phase 10 Acceptance Criteria

- [ ] Architecture identity verified.
- [ ] Same test set and evaluation used.
- [ ] Normalization fairness confirmed.
- [ ] Per-variable comparison metrics JSON saved.
- [ ] Convergence measurements (best_epoch, wall_clock_seconds) recorded.
- [ ] Result documented without fabrication.

---

## Phase 11 — Model Comparison and Scientific Figures

**Depends on:** Phases 3, 4, 5, 7, 10 metrics all exist.

**Phase gate:** Do not generate comparison figures from placeholder data.

### Task 11.1 — Figure generation script

Implement `scripts/generate_figures.py`:

- Reads saved metrics and predictions from `results/`.
- Generates:

  1. **Actual vs predicted** — for each key variable, short-horizon and
     longer autoregressive examples.
  2. **Training vs validation loss** — for LSTM, scratch Transformer,
     fine-tuned Transformer.
  3. **Forecast horizon vs RMSE** — per-variable, from horizon-wise metrics,
     for all four model variants.
  4. **Model comparison bar chart** — per-variable MAE/RMSE for Persistence /
     LSTM / Transformer / Pretrained+Fine-Tuned Transformer.
  5. **Scratch vs pretrained+fine-tuned** — validation curves overlaid.
  6. Optional additional per-variable figures.

- Saves all figures to `results/figures/`.

### Task 11.2 — Figure tests

In `tests/test_evaluation.py` or a dedicated test:

- Verify figure generation script runs without error given saved metrics.
- Verify output files exist in `results/figures/`.

### Phase 11 Acceptance Criteria

- [ ] All required figures generated from real metrics/predictions.
- [ ] Per-variable figures use original physical units.
- [ ] No placeholder or fabricated figures.
- [ ] Figures saved to `results/figures/`.
- [ ] Figure generation is reproducible via script.

---

## Phase 12 — Optional Extreme-Event Case Study

**Depends on:** Phase 7 (autoregressive predictions in test period).

**Decision point:** Whether a documented extreme event exists inside the
test period. Resolves §9.10.

### Task 12.1 — Event identification

Search for documented weather events (heatwave, heavy rainfall, strong
wind) within the test period for the target location.

**MCP usage:** Use **Tavily** to search for documented events.

If no suitable event is found, skip this phase and document the omission.

### Task 12.2 — Event evaluation (if feasible)

- Extract model predictions for the event period.
- Compare actual observations vs autoregressive forecast.
- Assess timing, magnitude, trend.
- Generate event-study figure.
- Do not claim successful capture unless metrics support it.

### Phase 12 Acceptance Criteria (if attempted)

- [ ] Event is documented with external source reference.
- [ ] Event falls within test period.
- [ ] Comparison uses actual model predictions.
- [ ] Assessment is honest about model performance.

---

## Phase 13 — Scenario Sensitivity

**Depends on:** At least one trained model checkpoint (Phase 4 or 5).

### Task 13.1 — Scenario module implementation

Implement `src/scenario.py` per §2.9:

- `get_perturbable_features(feature_names, configured_perturbable)` —
  dynamically computes available perturbation controls.
- `apply_perturbation(...)` for controlled input modification.
- `run_scenario(...)` returning baseline and perturbed trajectories.

No hard-coded variable names. Controls derived from feature list and config.

### Task 13.2 — Scenario tests

In `tests/test_forecasting.py` or dedicated tests:

- Verify perturbation is applied only to specified features.
- Verify baseline trajectory is unaffected.
- Verify perturbed trajectory differs.
- Verify `get_perturbable_features` returns correct intersection.
- Verify `get_perturbable_features` raises error on empty intersection.
- Verify disclaimer text is available/attached.

### Phase 13 Acceptance Criteria

- [ ] Scenario module produces baseline and perturbed trajectories.
- [ ] Perturbable features are dynamically determined from config and model.
- [ ] Only specified features are perturbed.
- [ ] Disclaimer text is present.
- [ ] No causal language in any output.
- [ ] No hard-coded variable names.

---

## Phase 14 — Streamlit Climate Digital Twin Dashboard

**Depends on:** Phases 3, 4, 5, 7, 10, 11, 13 artifacts.

**Phase gate (research mode):** The final research dashboard requires real
artifacts for all four required models (Persistence, LSTM, scratch
Transformer, pretrained+fine-tuned Transformer). A development mode may
tolerate unavailable models only if they are clearly marked "unavailable —
not yet trained" in the UI. Never substitute placeholders for missing
models.

### Task 14.1 — Dashboard skeleton

Create `dashboard/app.py` with five-page navigation structure.

**MCP usage:** Use **Context7** to verify current Streamlit API (page
navigation, caching, components).

### Task 14.2 — Model availability detection

Implement artifact discovery logic:

- For each required model, check whether the expected manifest, checkpoint,
  metrics, and predictions files exist.
- In development mode: mark unavailable models as "not yet trained."
- In research mode: require all four models or refuse to launch.

### Task 14.3 — Climate Overview page

- Display dataset metadata (location, data type, variables, time range).
- Historical time-series plots from `data/processed/`.
- Latest available observation summary.

### Task 14.4 — AI Forecast page

- Model selection from available models (dynamically detected).
- Load saved predictions and metrics from `results/`.
- Display actual vs predicted.
- Display per-variable metrics.

### Task 14.5 — Multi-Step Forecast page

- Horizon selector.
- Display autoregressive forecast trajectory.
- Display actual values alongside (for test-period evaluation).
- Display per-variable horizon-wise RMSE chart.

### Task 14.6 — Model Comparison page

- Display per-variable model comparison charts.
- Display training curves.
- Display horizon degradation comparison.
- All from saved `results/figures/` and `results/metrics/`.

### Task 14.7 — Scenario Sensitivity page

- Build input perturbation controls dynamically from
  `get_perturbable_features()`.
- Run scenario using loaded model.
- Display baseline vs perturbed trajectory.
- Display mandatory disclaimer prominently.

### Task 14.8 — Dashboard artifact loading tests

In `tests/test_dashboard.py`:

- Verify required artifact files can be located and loaded.
- Verify no training code is executed on import.
- Verify model availability detection logic.
- Verify unavailable models show "not yet trained" rather than placeholders.

### Task 14.9 — Browser validation

**MCP usage:** Use **Playwright** to launch the dashboard and verify:

- All five pages render.
- No uncaught errors in console.
- Scenario disclaimer is visible.
- Scenario controls are dynamically generated.
- Figures load from saved artifacts.
- Unavailable models are properly indicated (if in dev mode).

### Phase 14 Acceptance Criteria

- [ ] Five pages implemented and navigable.
- [ ] All displayed data comes from saved artifacts.
- [ ] No silent retraining during interaction.
- [ ] Scenario controls generated dynamically from features/config.
- [ ] Scenario disclaimer is visible.
- [ ] Model availability detection works correctly.
- [ ] Research mode requires all four model artifacts.
- [ ] Playwright browser check passes (or manual equivalent).

---

## Phase 15 — Final Scientific Audit

**Depends on:** All mandatory phases (0–11, 13–14) plus any optional phases
actually attempted (12).

### Task 15.1 — Traceability audit

For every claimed experimental result, verify the chain:

```text
configuration → run ID → preprocessing state → checkpoint →
predictions → metrics → figures/dashboard
```

### Task 15.2 — Leakage audit

Re-run all leakage tests. Verify:

- Chronological split ordering holds.
- Scaler was fit on training data only.
- No sequence targets cross split boundaries.
- No future ground-truth used in autoregressive rollout.
- Target location excluded from pretraining (or val/test periods excluded).

### Task 15.3 — Claims audit

Search all documentation, README, PROJECT_REVIEW, and dashboard text for:

- Prohibited claims ("reproduced Aurora", "reproduced CREDIT").
- Causal language in scenario descriptions.
- Placeholder or fabricated metrics.
- Unlabeled placeholders.
- Mixed-unit aggregate metrics presented as primary results.
- ERA5 grid cells described as weather stations.

### Task 15.4 — Completeness audit

Verify each mandatory experiment stage has:

- Code + configuration + tests/smoke checks + successful execution +
  checkpoint (for trained models) + metrics + manifest + required figures.

Verify optional phases (extreme event) are documented as either completed
or explicitly skipped.

### Task 15.5 — Metric reporting audit

Verify:

- All primary metrics are per-variable, in original physical units.
- No mixed-unit aggregate is presented as a primary research result.
- Convergence measurements (best_epoch, wall_clock_seconds) are recorded.
- Hyperparameter selection rationale is recorded in manifests.

### Task 15.6 — README and PROJECT_REVIEW

Write/update:

- `README.md` — project description, setup, usage, structure.
- `PROJECT_REVIEW.md` — honest summary of what was implemented, executed,
  observed, and what remains as limitations or future work.

Clearly distinguish: implemented and executed vs. implemented but not
executed vs. planned/future.

### Phase 15 Acceptance Criteria

- [ ] Full traceability chain verified for all reported results.
- [ ] All leakage tests pass (including transfer contamination).
- [ ] No prohibited claims found.
- [ ] No placeholder metrics in final outputs.
- [ ] All primary metrics are per-variable original-unit.
- [ ] Convergence and hyperparameter records present.
- [ ] README and PROJECT_REVIEW reflect actual status.

---

# PART 4 — ACCEPTANCE CRITERIA, TEST REQUIREMENTS, ARTIFACT REQUIREMENTS, AND PHASE GATES

## 4.1 Master Test Matrix

| Test Area | File | Critical Assertions |
|---|---|---|
| Environment smoke | `test_smoke.py` | Core imports succeed; seed utility runs; manifest creates valid dict |
| Chronological ordering | `test_preprocessing.py` | `max(train_ts) < min(val_ts) < min(test_ts)` |
| No split overlap | `test_preprocessing.py` | Zero intersection between split indices |
| Train-only scaler fit | `test_preprocessing.py` | Scaler `.fit()` called only on train data |
| No cross-boundary targets | `test_dataset.py` | No sequence target timestamp in wrong split |
| Sequence alignment | `test_dataset.py` | Input window + target indexing correctness |
| Feature order stability | `test_dataset.py` | Feature names identical across splits/loads |
| Station-safe sequences | `test_dataset.py` | No cross-station jumps in multi-location data |
| Multi-location chronology | `test_dataset.py` | Chronological ordering within each location |
| n_targets consistency | `test_dataset.py` | Target tensor dim matches configured n_targets |
| Transfer contamination | `test_dataset.py` / `test_preprocessing.py` | Target location (or val/test period) absent from pretraining data |
| Duplicate handling | `test_preprocessing.py` | Duplicates detected and removed per policy |
| Sampling detection | `test_preprocessing.py` | Expected interval matches actual |
| Inverse transform | `test_evaluation.py` | Roundtrip: scale → inverse_scale ≈ original |
| Per-variable metrics | `test_evaluation.py` | Per-variable MAE/RMSE structure correct; no mixed-unit aggregate as primary |
| Model output shapes | `test_models.py` | LSTM and Transformer output shapes match n_targets |
| Checkpoint roundtrip | `test_models.py` | Save → load → eval-mode forward pass produces identical output |
| Optimizer validity | `test_models.py` | Finite loss, finite non-null gradients, ≥1 parameter changed after step |
| Rollout length | `test_forecasting.py` | Output length = requested horizon |
| Rollout no leakage | `test_forecasting.py` | No future ground-truth in predictions |
| Horizon=1 consistency | `test_forecasting.py` | Rollout at h=1 matches one-step inference |
| Feature update logic | `test_forecasting.py` | Recursive update modifies correct indices |
| All-feature update | `test_forecasting.py` | When n_targets==n_features, all features updated |
| MAE correctness | `test_evaluation.py` | Known input/output pair produces expected MAE |
| RMSE correctness | `test_evaluation.py` | Known input/output pair produces expected RMSE |
| Metric JSON roundtrip | `test_evaluation.py` | Save → load preserves all values |
| Scenario perturbation | `test_forecasting.py` | Only specified features modified |
| Scenario dynamic features | `test_forecasting.py` | get_perturbable_features returns correct intersection; errors on empty |
| Dashboard artifact load | `test_dashboard.py` | Required files locatable and parseable |
| Dashboard no training | `test_dashboard.py` | Import does not trigger model training |
| Dashboard availability | `test_dashboard.py` | Unavailable models shown as "not yet trained" |

**Training constraint:** No test in the matrix triggers long model training.
Tests use synthetic data or tiny datasets.

## 4.2 Experiment Artifact Requirements

Every completed experimental stage must produce:

| Artifact | Location | Format |
|---|---|---|
| Run manifest | `results/manifests/` | JSON per §2.5 |
| Checkpoint | `results/checkpoints/` | `.pt` / `.pth` |
| Predictions | `results/predictions/` | Parquet or CSV with timestamps |
| Metrics | `results/metrics/` | JSON (per-variable original-unit) |
| Training log | `results/metrics/` | JSON (per-epoch losses, best_epoch, wall_clock_seconds) |
| Scaler | `data/processed/` or `results/` | joblib |
| Figures | `results/figures/` | PNG (or SVG/PDF) |
| Config snapshot | `results/manifests/` | embedded in manifest |
| Provenance | `data/metadata/` | YAML with SHA-256 checksums |

## 4.3 Scientific Phase Gates

A downstream phase must **not** proceed unless the required upstream
artifacts exist and are internally consistent. Specifically:

| Gate | Upstream Requirement | Downstream Blocked |
|---|---|---|
| G0 | pytest collects ≥1 test and all pass | Any further development |
| G1 | Raw data + provenance metadata (with checksums) exist | Preprocessing |
| G2 | Processed splits + scaler + leakage tests pass | Any model training |
| G3 | Evaluation Core (Tier 1) tests pass + persistence metrics exist | LSTM |
| G4 | At least one trained model checkpoint exists | Autoregressive rollout |
| G5 | Scratch Transformer checkpoint + metrics exist | Pretraining can proceed |
| G6 | Valid pretrained checkpoint exists, loads, and passes contamination tests | Fine-tuning |
| G7 | Both scratch + fine-tuned per-variable metrics exist | Scratch vs transfer comparison |
| G8 | All four model per-variable metrics exist | Model comparison figures |
| G9 | All four model artifacts (metrics, predictions) exist for research mode | Final research dashboard |
| G9-dev | At least persistence metrics exist for development mode | Development dashboard |
| G10 | Dashboard functional + all mandatory phases complete | Final audit |

"Exists" means the file is present, parseable, and traceable to an actual
executed run — not merely that code capable of generating it has been
written.

---

# PART 5 — RISKS, MITIGATIONS, AND UNRESOLVED DECISIONS

## 5.1 Risks and Mitigations

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | CDS API access delays, rate limits, or authentication issues for ERA5 data | Blocks Phase 1 | Have fallback dataset strategy (weather-station CSV). Research access requirements early via Tavily/Fetch. |
| R2 | Selected dataset lacks sufficient variables or temporal coverage | Weakens research questions | Validate variable inventory and date range in Phase 1 before committing. |
| R3 | Pretraining on limited locations shows no transfer benefit | RQ1 answer may be negative | A negative result is scientifically valid per guardrails. Document it. |
| R4 | Autoregressive error accumulates rapidly, making 24-step horizon unusable | Weakens RQ2 analysis | Report the degradation honestly. Error accumulation is itself a valid finding. |
| R5 | Transformer overfits on small single-location dataset | Poor test metrics | Use dropout, early stopping, and keep architecture small. Compare against LSTM and baseline. |
| R6 | Colab GPU session limits or local compute constraints | Slow training | Keep models lightweight. Use CPU smoke tests first. Batch appropriately. |
| R7 | Multi-location pretraining data acquisition is expensive or unavailable | Blocks Phase 8 | Identify pretraining data availability during Phase 1, not Phase 8. |
| R8 | No documented extreme event in test period | Phase 12 skipped | Phase 12 is optional. Document the omission. |
| R9 | Scaler or feature-order mismatch between training and inference | Incorrect predictions | Persist feature names with scaler. Test roundtrip. Assert order at load time. |
| R10 | Dashboard displays stale or mismatched artifacts | Misleading presentation | Dashboard loads artifacts by manifest-linked paths. Add version checks. |
| R11 | Normalization policy mismatch between scratch and transfer makes comparison unfair | Invalid RQ1 conclusion | Resolve §9.14 before training. Document policy in every manifest. |
| R12 | Target-location data leaks into pretraining | Inflated transfer-learning results | Exclude target location from pretraining (preferred). Test for contamination. |

## 5.2 Unresolved Design Decisions

Each decision must be resolved with documented rationale before the
dependent phase proceeds.

| ID | Decision | Dependent Phase | Resolution Criteria |
|---|---|---|---|
| §9.1 | Exact public dataset (ERA5-Land, weather station, other) | Phase 1 | Dataset identified, access confirmed, license documented |
| §9.2 | Target location and data type (station observation vs reanalysis grid cell) | Phase 1 | Location selected, data type explicitly documented, data availability confirmed |
| §9.3 | Sampling interval (hourly, 6-hourly, daily, other) | Phase 1 | Interval validated from actual data |
| §9.4 | Final selected variables | Phase 1 | Variables confirmed present in dataset |
| §9.5 | Exact chronological split date boundaries | Phase 2 | Boundaries set based on dataset time range, documented in config |
| §9.6 | Transformer hyperparameters (d_model, nhead, layers) | Phase 5 | Selected via validation-based protocol (§1.3.6), documented in config |
| §9.7 | Input sequence window length | Phase 2 | Selected relative to sampling frequency, documented in config |
| §9.8 | Pretraining geography/number of locations | Phase 8 | Locations selected (excluding target preferred), data availability confirmed, rationale documented |
| §9.9 | Whether ACC is scientifically feasible | Phase 6 | Climatological reference availability assessed, decision documented |
| §9.10 | Whether an extreme-event case study is feasible | Phase 12 | Event searched in test period, feasibility documented |
| §9.11 | Exact LSTM hyperparameters (hidden_dim, layers, etc.) | Phase 4 | Selected via validation-based protocol (§1.3.6), documented in config |
| §9.12 | Fine-tuning parameter strategy (frozen/unfrozen/reinitialized layers) | Phase 9 | Strategy documented in config and manifest |
| §9.13 | Target schema: n_targets == n_features (all-variable prediction) vs explicit exogenous features | Phase 2 | Decision documented; model constructors accept both n_features and n_targets; rollout handles accordingly |
| §9.14 | Normalization policy across pretraining, scratch, fine-tuning, and evaluation | Phase 2/8 | Policy documented before training: which scaler for each stage, how inverse transform is applied, how comparison fairness is maintained |

---

# PART 6 — MCP TOOL USAGE PLAN

| MCP Tool | Phase(s) | Purpose |
|---|---|---|
| **Tavily** | 1, 8, 12 | Search for dataset availability, multi-location data sources, documented extreme events |
| **Fetch** | 1, 8 | Inspect official CDS pages, dataset documentation, API docs |
| **Context7** | 0, 1, 2, 4, 5, 14 | Verify current PyTorch, pandas, scikit-learn, xarray, cdsapi, Streamlit APIs |
| **Playwright** | 14 | Browser-level validation of the Streamlit dashboard |

Web search is **not** a substitute for recording dataset provenance from
official metadata.

---

# PART 7 — SUMMARY OF SCIENTIFIC CONSTRAINTS

These constraints are non-negotiable throughout implementation:

1. **No fabricated results.** Every metric, figure, and comparison must
   trace to actual executed code.

2. **Chronological integrity.** Train < validation < test temporally.
   Never shuffle time series. No cross-boundary sequence targets.

3. **Normalization integrity.** Scaler fit on training data only.
   Normalization policy documented and consistent across stages.

4. **Fair comparison.** Scratch vs transfer uses same architecture, same
   test set, same evaluation code. Normalization fairness addressed.

5. **Per-variable reporting.** Primary metrics are per-variable MAE/RMSE in
   original physical units. No mixed-unit aggregate as primary result.

6. **Honest reporting.** Negative results are valid. Limitations are
   stated. Placeholders are labeled. Claims match evidence.

7. **No reproduction claims.** ClimateTwin adapts methodological ideas
   from Aurora and CREDIT. It does not reproduce either system.

8. **Non-causal scenarios.** Scenario sensitivity is a model-based
   perturbation experiment, not a causal simulator.

9. **Phase gates enforced.** No downstream work on artifacts that don't
   exist or weren't generated by actual execution.

10. **Accurate data terminology.** Reanalysis grid cells are not weather
    stations.

11. **Convergence defined before experiments.** Best epoch and wall-clock
    duration always recorded. Any additional convergence metric defined
    before training, not after.

12. **Hyperparameters selected on validation only.** Test data untouched
    until final evaluation. Selection rationale recorded.

---

# CHANGELOG — v1.0 → v2.0

| # | Change | Sections Affected |
|---|---|---|
| 1 | Added `results/predictions/` to the canonical repository structure. | §2.2, Task 0.1, Task 0.6, artifact table §4.2 |
| 2 | Moved Evaluation Core (MAE, RMSE, inverse transform, metric I/O, per-variable metrics, alignment) into Phase 3 as Tier 1. Phase 6 retained for generic checkpoint evaluator, R², ACC gate as Tier 2. No duplication. | §2.7, Phase 3 (renamed to "Evaluation Core and Persistence Baseline"), Phase 6 (renamed to "Extended Evaluation Utilities"), dependency graph |
| 3 | Replaced "pytest passes with zero tests" with requirement for `tests/test_smoke.py` containing at least one environment smoke test. Phase 0 requires `pytest` to collect ≥1 test. | Task 0.8, Phase 0 acceptance criteria, §2.2 (added test_smoke.py), gate G0 |
| 4 | Added `.kiroignore` creation to Phase 0. Excludes raw/processed climate binaries, checkpoints, large predictions, environments, secrets. Keeps metadata, manifests, metrics, sample data. | Task 0.7, §2.2, Phase 0 acceptance criteria |
| 5 | Resolved n_features/n_targets API inconsistency. All model constructors accept both `n_features` and `n_targets`. Added §9.13 target-schema decision. Preferred default: n_targets == n_features. | §1.3.1 (new), §1.3.3, §1.3.4, §2.8, configs, manifest schema, smoke tests, rollout tests, §9.13 |
| 6 | Added transfer-learning contamination guard. Prefer excluding target location from pretraining entirely. At minimum exclude target val/test periods. Added automated contamination tests. | Task 8.1, Task 8.3 (new), Phase 8 acceptance criteria, test matrix, gate G6, §5.1 R12 |
| 7 | Added §9.14 normalization policy as unresolved design decision. Requires documenting which scaler is used per stage and how comparison fairness is maintained. | §1.4.6, configs/pretrain.yaml, configs/finetune.yaml, manifest schema, Phase 9, Phase 10, §9.14, §5.1 R11 |
| 8 | Defined chronological train/validation handling for multi-location pretraining data. Per-location chronological split, location boundary preservation. | Task 2.5, Task 8.2, Phase 8 acceptance criteria, test matrix (multi-location chronology) |
| 9 | Strengthened dashboard phase gate. Research mode requires all four model artifacts. Development mode may tolerate "unavailable" but never placeholder metrics. | §1.8, §2.10, Phase 14 gate, Task 14.2 (new), Task 14.8, gate G9/G9-dev, test matrix (dashboard availability) |
| 10 | Scenario controls generated dynamically from model features and configured perturbable_features. No hard-coded variable names. | §1.7, §2.9 (get_perturbable_features), configs/data.yaml, Task 13.1, Task 13.2, Task 14.7, Phase 13 acceptance criteria, test matrix (scenario dynamic features) |
| 11 | Per-variable original-unit MAE/RMSE is primary reporting. Mixed-unit aggregate prohibited as primary result. Any aggregate requires documented normalization. | §1.5.1, §2.7, Phase 3, Phase 10, Phase 11, Phase 15, §7 constraint 5, §1.11 |
| 12 | Defined convergence measurements: best_epoch and wall_clock_seconds mandatory. convergence_epoch optional with pre-defined rule. | §1.2 RQ1, §1.5.4 (new), manifest schema, Tasks 4.3/5.3/8.4/9.1, Phase 10 comparison report, §7 constraint 11 |
| 13 | Required horizon-wise evaluation for all four model variants. Exclusions require documented reason. | §1.2 RQ2, §1.6, Task 7.2, Phase 7 acceptance criteria |
| 14 | Changed optimizer smoke test to: finite loss, finite/non-null gradients, ≥1 parameter changed. Checkpoint roundtrip uses eval mode. | Tasks 4.2, 5.2, test matrix (optimizer validity, checkpoint roundtrip) |
| 15 | Extended provenance metadata with SHA-256 checksums and file sizes. | §2.11 (new), Task 1.2, Task 1.4, Task 8.1, artifact table |
| 16 | Changed Phase 15 dependency from "all prior phases" to "all mandatory phases plus any optional phases actually attempted." | Phase 15 depends-on |
| 17 | Phase 1 must distinguish station observation vs reanalysis grid cell. Added `source_type` field. Prohibited describing ERA5 grid points as stations. | §1.4.2, §2.4 (data.yaml), §2.5 (manifest), Task 1.1, Task 1.3, Task 1.4, Phase 1 acceptance criteria, §1.11, §7 constraint 10, Phase 15 claims audit |
| 18 | Added explicit hyperparameter-selection protocol: validation-only selection, record all candidates and rationale, test data untouched. | §1.3.6 (new), manifest schema (hyperparameter_selection), Tasks 4.3/5.3, §9.6/§9.11, §7 constraint 12 |

---

*End of specification v2.0. Awaiting review before implementation begins.*
