---
inclusion: always
---

# ClimateTwin Scientific and Research Guardrails

These rules are mandatory for all ClimateTwin implementation, analysis, documentation, figures, and dashboard work.

## Scientific Positioning

ClimateTwin is a lightweight localized prototype inspired by selected methodological principles from Aurora and CREDIT.

Never state or imply that ClimateTwin reproduces Aurora or CREDIT.

Acceptable framing:

> ClimateTwin adapts selected methodological ideas from Aurora and CREDIT at a localized, computationally feasible scale.

Unacceptable framing includes claims such as:

> We reproduced Aurora.

> We reproduced CREDIT.

## No Fabricated Results

Never invent or manually fill in experimental values that have not been produced by actual executed code.

This applies to:

- MAE;
- RMSE;
- R²;
- ACC;
- training loss;
- validation loss;
- best epoch;
- convergence speed;
- benchmark comparisons;
- forecast-horizon values;
- actual-vs-predicted plots;
- model-comparison plots;
- event-study results;
- scenario-study results.

Placeholders must be clearly labeled as placeholders and must never appear in final research claims or review results.

## Traceability Requirement

Every reported experimental result must be traceable to:

```text
data source/version
→ preprocessing configuration
→ chronological split
→ scaler/preprocessing artifact
→ model configuration
→ random seed
→ training run/checkpoint
→ evaluation code
→ machine-readable metric
→ figure/dashboard display
```

If this chain cannot be established, do not present the value as a research result.

## Time-Series Integrity

Never randomly shuffle the time series before creating train/validation/test periods.

Use chronological partitions:

```text
earlier period     → train
later period       → validation
latest unseen data → test
```

Training data must temporally precede validation data, and validation data must precede test data.

Never allow a sequence window to cross a split boundary in a way that leaks future target information into training.

## Normalization Integrity

Fit scalers and other learned preprocessing transformations on training data only.

Apply the already-fitted transformation to validation and test data.

Never fit or refit normalization using validation/test statistics.

Save preprocessing artifacts needed for inverse transformation and later inference.

## Feature Integrity

Use only variables actually present in the selected legitimate dataset or variables that are transparently and correctly derived from available variables.

Do not invent missing climate variables.

Record units and conversions explicitly.

Validate feature order and target order and keep them stable between training, evaluation, checkpoint loading, and dashboard inference.

## Evaluation Integrity

Required final metrics:

- MAE;
- RMSE.

Where practical, report final metrics in the original physical units after inverse transformation.

R² is optional and must not replace forecasting metrics.

ACC may only be implemented when anomalies are correctly defined relative to an appropriate climatological reference. Do not rename ordinary Pearson correlation as ACC.

If a correct climatology is unavailable, omit ACC and document that it is outside the lightweight prototype scope.

## Transfer-Learning Fairness

The core Aurora-inspired comparison is:

```text
Scratch Transformer
random initialization
→ target-location training

versus

Transfer Transformer
broad-data pretraining
→ target-location fine-tuning
```

The scratch and transfer comparison should use the same underlying Transformer architecture unless a deliberate ablation explicitly studies architecture differences.

Keep target-location splits and evaluation procedure comparable.

Do not claim pretraining helps unless the actual experiment demonstrates it.

A neutral or negative result is scientifically valid.

## Autoregressive Forecast Integrity

For recursive forecasting, predictions must be fed back into the next model input according to the exact feature/sequence design.

Evaluate the intended horizons:

- 1 step;
- 3 steps;
- 6 steps;
- 12 steps;
- 24 steps.

Do not label a step as an hour, 6 hours, or any other physical duration until the actual dataset sampling interval has been validated.

Store horizon-wise metrics. Do not report only a final horizon if the research question concerns degradation across lead time.

## Extreme-Event Integrity

An extreme-event case study is optional.

Only select an event that is documented and actually falls within the evaluation/test period.

Compare actual observations with model forecasts and assess timing, magnitude, and trend.

Do not claim successful event capture unless the metrics and visualization support that statement.

## Scenario Module Integrity

Scenario Sensitivity is a model-input perturbation experiment.

It is not a physically validated intervention simulator and does not establish causal real-world effects.

The dashboard/report must include a clear disclaimer equivalent to:

> Scenario outputs are model-based sensitivity experiments. They are not physically validated climate intervention simulations and do not establish causal effects.

Do not use causal language when interpreting scenario trajectories.

## Phase Gate

Do not silently skip failed prerequisites.

Before depending on an artifact, verify that it exists and was generated successfully.

Examples:

- do not build final model-comparison figures before real metrics exist;
- do not fine-tune before a valid pretraining checkpoint exists;
- do not evaluate a checkpoint that cannot be loaded with its matching preprocessing state;
- do not build the final dashboard around placeholder results;
- do not claim an experiment completed when only its code was written.

## Review Honesty

Clearly distinguish:

- implemented and executed;
- implemented but not yet executed;
- planned/future work;
- optional extensions.

The project review should communicate limitations including localized scale, smaller models/data than global foundation systems, limited variables, possible autoregressive error accumulation, and the non-causal nature of scenario perturbations.
