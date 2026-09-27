# ClimateTwin --- Research-Inspired Implementation Prompt

## Purpose

Build a **college-scale AI-driven Climate Digital Twin** inspired by two
2025 research papers:

1.  **Aurora: A Foundation Model of the Earth System** --- Nature (2025)
2.  **Community Research Earth Digital Intelligence Twin (CREDIT): a
    scalable framework for AI-driven Earth System Modeling** --- npj
    Climate and Atmospheric Science (2025)

The goal is **not to reproduce either paper at their original scale**.
The goal is to reproduce selected **methodological ideas and
experimental patterns at a small, computationally feasible scale** for a
project review.

------------------------------------------------------------------------

# 1. Source-Paper Basis

## 1.1 Aurora --- ideas to adapt

Aurora is a **1.3-billion-parameter foundation model** for the Earth
system. Its architecture consists of:

-   an **encoder** that converts heterogeneous inputs into a universal
    latent 3D representation,
-   a **processor** that evolves that representation through time,
-   a **decoder** that converts the latent representation back into
    physical predictions.

The processor uses a 3D Swin Transformer, while the encoder and decoder
are Perceiver-based modules.

Aurora uses:

-   heterogeneous Earth-system datasets,
-   **pretraining** on large and diverse data,
-   **fine-tuning** for downstream tasks,
-   recursive/autoregressive forecasting, where predictions are fed back
    as future inputs.

The key idea for this project is:

> **Pretrain a general climate forecasting model on broader data, then
> fine-tune it on a localized forecasting task and compare this against
> training from scratch.**

Aurora also demonstrates that pretraining can be important when
target-task data are limited.

### What we will NOT reproduce

Do **not** claim to reproduce:

-   Aurora's 1.3B-parameter scale,
-   its exact Perceiver + 3D Swin architecture,
-   its massive heterogeneous pretraining corpus,
-   its large GPU infrastructure,
-   its operational global benchmark results.

------------------------------------------------------------------------

## 1.2 CREDIT --- ideas to adapt

CREDIT is an **end-to-end research framework** for AI-driven numerical
weather prediction.

Its important ideas are:

-   modular data preprocessing,
-   standardized training workflows,
-   interchangeable model architectures,
-   standardized evaluation,
-   single-step and multi-step/autoregressive training and inference,
-   comparison of multiple architectures,
-   analysis over increasing forecast lead times.

The paper demonstrates the framework using architectures including:

-   WXFormer,
-   FuXi,

and evaluates forecasts over long horizons using metrics including:

-   **RMSE**,
-   **ACC (Anomaly Correlation Coefficient)**.

The project should borrow the **framework philosophy**, not attempt to
reproduce WXFormer or FuXi exactly.

### What we will reproduce from CREDIT conceptually

``` text
DATA
  ↓
PREPROCESSING
  ↓
MODEL TRAINING
  ↓
INFERENCE
  ↓
MULTI-STEP FORECASTING
  ↓
STANDARDIZED EVALUATION
  ↓
MODEL COMPARISON
  ↓
VISUALIZATION / DIGITAL TWIN
```

------------------------------------------------------------------------

# 2. Project Title

## ClimateTwin: A Lightweight AI-Driven Local Climate Digital Twin

### Suggested project statement

> ClimateTwin is a lightweight AI-driven prototype inspired by modern
> Earth-system foundation models and AI weather prediction frameworks.
> It investigates whether a general model pretrained on broader climate
> data can be efficiently adapted to a localized climate forecasting
> task, while supporting multi-step forecasting, standardized model
> comparison, and interactive visualization.

------------------------------------------------------------------------

# 3. Core Research Questions

The implementation should answer the following questions:

### RQ1 --- Transfer Learning

> Does pretraining on broader climate data improve localized forecasting
> compared with training the same model from scratch?

Compare:

``` text
MODEL A
Train from scratch on target location

VS

MODEL B
Pretrain on broad data
        ↓
Fine-tune on target location
```

Measure:

-   MAE
-   RMSE
-   convergence speed
-   final validation error

------------------------------------------------------------------------

### RQ2 --- Forecast Horizon

> How does forecast error change as predictions are recursively extended
> into the future?

Evaluate multiple horizons such as:

-   1 step
-   3 steps
-   6 steps
-   12 steps
-   24 steps

Expected pattern to investigate:

``` text
RMSE
 ^
 |                    *
 |                *
 |            *
 |        *
 |    *
 +--------------------------> Forecast horizon
```

Do **not** fabricate this result. The graph must be generated from
actual experiments.

------------------------------------------------------------------------

### RQ3 --- Model Architecture

> How do different machine-learning architectures compare for local
> multivariate climate forecasting?

Compare:

1.  Simple baseline
2.  LSTM
3.  Lightweight Transformer
4.  Pretrained + fine-tuned Transformer

------------------------------------------------------------------------

# 4. Scope

The project must run on:

-   a normal laptop for small experiments, and/or
-   Google Colab for GPU training.

Keep the initial system intentionally lightweight.

## Recommended initial geographical scope

Use one of:

-   Delhi,
-   Mumbai,
-   Bengaluru,
-   one Indian state,
-   or a small India-centred grid.

### Recommended progression

``` text
PHASE 1
One location / station
        ↓
PHASE 2
Multiple locations
        ↓
PHASE 3
Spatial grid if time and compute allow
```

A station-based prototype is easier and more realistic for a project
review.

------------------------------------------------------------------------

# 5. Data

Use a legitimate public climate/weather source.

Possible source categories:

-   ERA5 / ERA5-Land reanalysis,
-   weather-station data,
-   IMD data if legitimately accessible and permitted.

The first prototype should contain a multivariate time series.

## Suggested variables

Use only variables actually available in the chosen dataset.

Possible variables:

-   2 m temperature
-   relative humidity or dew-point-derived humidity
-   surface pressure
-   precipitation
-   10 m wind speed
-   U wind component
-   V wind component

Optional:

-   solar radiation

------------------------------------------------------------------------

# 6. Data Schema

A station-level dataset can be structured conceptually as:

``` text
timestamp
latitude
longitude
temperature
humidity
pressure
wind_u
wind_v
precipitation
```

If the dataset does not contain latitude and longitude for every row,
store location metadata separately.

Do not invent missing variables.

------------------------------------------------------------------------

# 7. Data Pipeline

Implement:

## Step 1 --- Data Loading

Create a reusable loader.

``` python
load_raw_data()
```

------------------------------------------------------------------------

## Step 2 --- Cleaning

Handle:

-   duplicate timestamps,
-   missing values,
-   invalid values,
-   irregular sampling,
-   unit consistency.

Log every cleaning operation.

------------------------------------------------------------------------

## Step 3 --- Chronological Split

Never randomly shuffle a time series before splitting.

Use:

``` text
TRAIN
Earlier period

VALIDATION
Later period

TEST
Latest unseen period
```

Example:

``` text
2018–2022 → Training
2023      → Validation
2024      → Testing
```

The exact dates depend on the chosen dataset.

------------------------------------------------------------------------

## Step 4 --- Normalization

Fit scalers **only on training data**.

Then apply the same fitted scaler to:

-   validation,
-   test.

This prevents data leakage.

Possible methods:

-   StandardScaler / z-score normalization,
-   MinMaxScaler.

Save the fitted scaler.

------------------------------------------------------------------------

## Step 5 --- Sequence Construction

Convert the data into sequences.

Example:

``` text
INPUT WINDOW

t-23
t-22
...
t-1
t

↓ MODEL

TARGET

t+1
```

Experiment with input windows such as:

-   12 timesteps,
-   24 timesteps,
-   48 timesteps.

The correct window depends on sampling frequency.

------------------------------------------------------------------------

# 8. Model 1 --- Baseline

Implement a simple baseline.

Recommended options:

-   Persistence baseline,
-   Linear Regression,
-   Random Forest,
-   XGBoost if dependencies and compute permit.

## Minimum requirement

Include a **persistence baseline**:

``` text
Prediction at t+1 = state at t
```

This is especially important because sophisticated models should be
compared against a meaningful simple forecast.

Optionally add Random Forest as a second conventional ML baseline.

------------------------------------------------------------------------

# 9. Model 2 --- LSTM

Implement a multivariate LSTM.

Concept:

``` text
Historical Climate Sequence
          ↓
         LSTM
          ↓
      Hidden State
          ↓
    Regression Layer
          ↓
Next Climate State
```

Input:

``` text
N previous timesteps × climate variables
```

Output:

``` text
next timestep climate variables
```

Experiment with:

-   sequence length,
-   hidden dimension,
-   number of layers,
-   dropout,
-   learning rate.

Use validation data for model selection.

------------------------------------------------------------------------

# 10. Model 3 --- Lightweight Transformer

Implement a lightweight Transformer encoder.

Architecture:

``` text
Climate Feature Sequence
          ↓
Feature Projection / Embedding
          ↓
Positional Encoding
          ↓
Transformer Encoder
          ↓
Regression Head
          ↓
Predicted Next Climate State
```

Keep the model small enough for practical training.

Example design parameters:

``` text
Embedding dimension: 64 or 128
Attention heads: 4 or 8
Encoder layers: 2–4
Dropout: 0.1
```

These are starting points, not paper-derived Aurora settings.

------------------------------------------------------------------------

# 11. Aurora-Inspired Pretraining and Fine-Tuning Experiment

This is the most important Aurora-inspired experiment.

## Stage A --- Broad Pretraining

Train the lightweight Transformer on:

``` text
multiple locations
or
a broader geographical region
or
multiple related climate datasets
```

Task:

``` text
previous climate states
        ↓
predict next climate state
```

The goal is to learn a broader representation.

------------------------------------------------------------------------

## Stage B --- Local Fine-Tuning

Take the pretrained weights and fine-tune on the target location.

Example:

``` text
PRETRAIN
Multiple Indian cities

        ↓

FINE-TUNE
Delhi
```

Use:

-   smaller learning rate,
-   fewer epochs if appropriate,
-   early stopping.

------------------------------------------------------------------------

## Stage C --- Scratch Model

Train an identical Transformer from random initialization using only the
target-location training data.

------------------------------------------------------------------------

## Stage D --- Comparison

Compare:

``` text
SCRATCH
Target location only

VS

PRETRAINED + FINE-TUNED
Broader climate data
        ↓
Target location
```

Report:

  --------------------------------------------------------------------------
  Model                Test MAE      Test RMSE       Training           Best
                                                       Epochs     Validation
                                                                        Loss
  -------------- -------------- -------------- -------------- --------------
  Scratch         Actual result  Actual result  Actual result  Actual result
  Transformer                                                 

  Pretrained +    Actual result  Actual result  Actual result  Actual result
  Fine-Tuned                                                  
  --------------------------------------------------------------------------

All values must come from real experiments.

------------------------------------------------------------------------

# 12. Autoregressive Forecasting

Implement recursive forecasting.

## One-step forecast

``` text
X[t] → X[t+1]
```

## Multi-step rollout

``` text
Observed history
       ↓
Predict t+1
       ↓
Use prediction
       ↓
Predict t+2
       ↓
Use prediction
       ↓
Predict t+3
       ↓
...
```

Conceptually:

``` text
t
↓
t+1 prediction
↓
t+2 prediction
↓
t+3 prediction
↓
...
↓
t+N prediction
```

The exact input-update method must match the model's feature design.

------------------------------------------------------------------------

# 13. Forecast-Horizon Experiment

Evaluate at:

-   1 step
-   3 steps
-   6 steps
-   12 steps
-   24 steps

If using 6-hour data, explicitly state that these correspond to:

``` text
1 step  = 6 hours
3 steps = 18 hours
6 steps = 36 hours
...
```

Do not label steps as hours unless the sampling interval actually
supports that conversion.

Create:

## Graph

``` text
Forecast Horizon vs RMSE
```

Optionally produce one graph per variable:

-   temperature,
-   pressure,
-   wind,
-   precipitation.

------------------------------------------------------------------------

# 14. Evaluation

## Required

### MAE

``` text
MAE = mean(|y - y_hat|)
```

### RMSE

``` text
RMSE = sqrt(mean((y - y_hat)^2))
```

Use both:

-   normalized-space loss for training if appropriate,
-   original-unit metrics for final reporting.

------------------------------------------------------------------------

## Optional

### R²

Use for an individual regression target if useful.

Do not treat R² as a replacement for forecasting-specific verification.

------------------------------------------------------------------------

### ACC --- Anomaly Correlation Coefficient

CREDIT uses RMSE and ACC in atmospheric verification.

Only implement ACC if anomalies can be correctly defined relative to a
reference climatology.

Do **not** simply rename ordinary Pearson correlation as ACC.

If a correct anomaly climatology is not available, omit ACC and
explicitly state that it was outside the lightweight prototype scope.

------------------------------------------------------------------------

# 15. Model Comparison

Create a table:

  --------------------------------------------------------------------------------
  Model                      MAE            RMSE   R² (optional) Notes
  -------------- --------------- --------------- --------------- -----------------
  Persistence             Actual          Actual       Actual/NA Simple baseline

  Conventional            Actual          Actual       Actual/NA Optional
  ML                                                             

  LSTM                    Actual          Actual       Actual/NA Temporal model

  Transformer             Actual          Actual       Actual/NA Attention-based
                                                                 model

  Pretrained +            Actual          Actual       Actual/NA Aurora-inspired
  Fine-Tuned                                                     transfer learning
  Transformer                                                    
  --------------------------------------------------------------------------------

------------------------------------------------------------------------

# 16. Required Visualizations

## 16.1 Actual vs Predicted

For each key variable:

``` text
Actual      ─────────────
Prediction  - - - - - - -
```

Use:

-   one short-horizon example,
-   one longer autoregressive example.

------------------------------------------------------------------------

## 16.2 Forecast Horizon vs Error

``` text
RMSE
 ^
 |              *
 |           *
 |        *
 |     *
 |  *
 +----------------------> Forecast horizon
```

------------------------------------------------------------------------

## 16.3 Model Comparison

Bar chart:

``` text
Persistence
LSTM
Transformer
Pretrained + Fine-Tuned Transformer
```

Plot:

-   RMSE,
-   optionally MAE.

------------------------------------------------------------------------

## 16.4 Training Curves

``` text
Loss
 ^
 |\
 | \
 |  \__
 |
 +----------------------> Epoch
```

Show:

-   training loss,
-   validation loss.

------------------------------------------------------------------------

## 16.5 Pretraining vs Fine-Tuning

Show:

``` text
Scratch training validation curve

VS

Fine-tuning validation curve
```

This is one of the strongest visuals for the review.

------------------------------------------------------------------------

# 17. Extreme-Event Case Study

If the dataset and project time allow, select a documented event inside
the test period.

Possible examples:

-   heatwave,
-   heavy rainfall,
-   strong wind event.

Show:

``` text
Actual observations
        VS
Autoregressive forecast
```

Evaluate:

-   timing,
-   magnitude,
-   trend.

Do not claim that the model accurately captured an event unless the
metrics and visualization demonstrate it.

------------------------------------------------------------------------

# 18. What-If Scenario Module

A scenario feature can be included, but it must be presented honestly.

Example controls:

``` text
Temperature adjustment
Humidity adjustment
Wind adjustment
```

Process:

``` text
Current / historical model state
        +
User perturbation
        ↓
Modified input state
        ↓
Forecast model
        ↓
Alternative predicted trajectory
```

Show:

``` text
Baseline model forecast
        VS
Perturbed-input forecast
```

## Mandatory disclaimer

> This module performs model-based sensitivity experiments by perturbing
> input features. It is not a physically validated climate intervention
> simulator and does not establish real-world causal effects.

This distinction is essential.

------------------------------------------------------------------------

# 19. Digital Twin Dashboard

Use Streamlit.

## Page 1 --- Climate Overview

Show:

-   latest available observation,
-   historical time series,
-   selected location,
-   key climate variables.

------------------------------------------------------------------------

## Page 2 --- AI Forecast

User selects:

``` text
Persistence
LSTM
Transformer
Pretrained + Fine-Tuned Transformer
```

Show:

-   actual,
-   predicted,
-   metrics.

------------------------------------------------------------------------

## Page 3 --- Multi-Step Forecast

Show:

-   selected forecast horizon,
-   autoregressive forecast trajectory,
-   actual values when evaluating historical test data.

------------------------------------------------------------------------

## Page 4 --- Model Comparison

Show:

-   MAE,
-   RMSE,
-   training curves,
-   horizon degradation.

------------------------------------------------------------------------

## Page 5 --- Scenario Analysis

Allow controlled input perturbations.

Show:

``` text
Baseline trajectory
VS
Perturbed-input trajectory
```

Include the disclaimer.

------------------------------------------------------------------------

# 20. Optional Spatial Extension

Only implement this if data are available on a geographic grid.

Then visualize:

``` text
Latitude × Longitude
        ↓
Actual field

VS

Predicted field
```

For example:

-   temperature field,
-   pressure field,
-   wind field.

Do not delay the core project for this extension.

------------------------------------------------------------------------

# 21. Project Architecture

``` text
                    CLIMATE DATA
                         │
                         ▼
                 DATA PREPROCESSING
                         │
                         ▼
                  FEATURE / SEQUENCES
                         │
                         ▼
              ┌─────────────────────┐
              │   MODEL EXPERIMENTS │
              ├─────────────────────┤
              │ Persistence         │
              │ LSTM                │
              │ Transformer         │
              │ Pretrained Model    │
              └─────────────────────┘
                         │
                         ▼
                 ONE-STEP FORECAST
                         │
                         ▼
              AUTOREGRESSIVE ROLLOUT
                         │
                         ▼
                   EVALUATION
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
           RMSE         MAE      Horizon Error
                         │
                         ▼
                CLIMATE TWIN UI
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
          Overview    Forecast    Scenarios
```

------------------------------------------------------------------------

# 22. Suggested Repository Structure

``` text
ClimateTwin/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_experiments.ipynb
│
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── forecasting.py
│   ├── scenario.py
│   │
│   └── models/
│       ├── persistence.py
│       ├── lstm.py
│       └── transformer.py
│
├── dashboard/
│   └── app.py
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── checkpoints/
│
├── requirements.txt
├── README.md
└── PROJECT_REVIEW.md
```

------------------------------------------------------------------------

# 23. Development Order

Build in this order.

## Phase 1 --- Data

-   obtain dataset,
-   validate timestamps,
-   clean data,
-   perform EDA,
-   create chronological splits.

## Phase 2 --- Baseline

-   persistence forecast,
-   MAE and RMSE.

## Phase 3 --- LSTM

-   sequence dataset,
-   training,
-   validation,
-   test evaluation.

## Phase 4 --- Transformer

-   implement lightweight Transformer,
-   train from scratch,
-   evaluate.

## Phase 5 --- Pretraining + Fine-Tuning

-   prepare broad-data pretraining set,
-   pretrain Transformer,
-   fine-tune target location,
-   compare with scratch model.

## Phase 6 --- Autoregressive Forecasting

-   recursive rollout,
-   horizon-wise evaluation.

## Phase 7 --- Dashboard

-   integrate saved model,
-   show actual outputs,
-   visualize experiments.

------------------------------------------------------------------------

# 24. Experiment Checklist

## Experiment A --- Baseline

Question:

> Does deep learning improve on a simple persistence forecast?

------------------------------------------------------------------------

## Experiment B --- LSTM vs Transformer

Question:

> Which architecture performs better for the selected local multivariate
> forecasting task?

------------------------------------------------------------------------

## Experiment C --- Aurora-Inspired Transfer Learning

Question:

> Does broader pretraining improve localized adaptation?

Compare:

``` text
Scratch Transformer
VS
Pretrained + Fine-Tuned Transformer
```

------------------------------------------------------------------------

## Experiment D --- CREDIT-Inspired Multi-Step Evaluation

Question:

> How does model error evolve over autoregressive lead time?

Plot:

``` text
Lead Time vs RMSE
```

------------------------------------------------------------------------

# 25. Important Claims to Avoid

Do NOT say:

> "We reproduced Aurora."

Do NOT say:

> "We reproduced CREDIT."

Do NOT say:

> "Our scenario slider predicts the causal effect of climate
> interventions."

Do NOT fabricate:

-   metrics,
-   graphs,
-   benchmark comparisons,
-   event results.

------------------------------------------------------------------------

# 26. Correct Project Positioning

Use this statement:

> **Our work develops a lightweight localized Climate Digital Twin
> inspired by the methodological principles of Aurora and the CREDIT
> framework. We adapt two core ideas: Aurora's
> pretraining-to-fine-tuning approach for downstream adaptation, and
> CREDIT's modular end-to-end workflow for training, autoregressive
> inference, evaluation, and model comparison.**

------------------------------------------------------------------------

# 27. Project Review Presentation Flow

## Slide 1 --- Problem

> Climate and weather systems are highly dynamic, and localized
> forecasting requires models that can learn multivariate temporal
> relationships and adapt to specific regions.

------------------------------------------------------------------------

## Slide 2 --- Research Inspiration

### Aurora

``` text
Broad heterogeneous data
        ↓
Pretraining
        ↓
General learned representation
        ↓
Fine-tuning
        ↓
Downstream forecasting task
```

### CREDIT

``` text
Data
↓
Preprocessing
↓
Model Training
↓
Inference
↓
Evaluation
↓
Model Comparison
```

------------------------------------------------------------------------

## Slide 3 --- Our Research Gap / Scope

> Large Earth-system AI models require enormous datasets and
> computational resources. Our project investigates whether their core
> methodological principles can be demonstrated at a localized scale
> using lightweight models and accessible computing.

------------------------------------------------------------------------

## Slide 4 --- ClimateTwin Architecture

Show the complete pipeline.

------------------------------------------------------------------------

## Slide 5 --- Dataset

Explain:

-   source,
-   geography,
-   time period,
-   variables,
-   sampling interval,
-   train/validation/test split.

------------------------------------------------------------------------

## Slide 6 --- Models

Show:

``` text
Persistence
    ↓
LSTM
    ↓
Transformer
    ↓
Pretrained + Fine-Tuned Transformer
```

------------------------------------------------------------------------

## Slide 7 --- Aurora-Inspired Experiment

Show:

``` text
BROAD DATA
    ↓
PRETRAINING
    ↓
LOCAL DATA
    ↓
FINE-TUNING
```

Compare against scratch training.

------------------------------------------------------------------------

## Slide 8 --- CREDIT-Inspired Experiment

Show:

``` text
One-Step Prediction
        ↓
Recursive Prediction
        ↓
Multi-Step Forecast
        ↓
Lead-Time Evaluation
```

------------------------------------------------------------------------

## Slide 9 --- Results

Show only actual experimental results.

Include:

-   MAE,
-   RMSE,
-   actual vs predicted,
-   forecast horizon vs error.

------------------------------------------------------------------------

## Slide 10 --- Demo

Demonstrate:

1.  Climate overview
2.  Model selection
3.  Forecast
4.  Model comparison
5.  Scenario sensitivity experiment

------------------------------------------------------------------------

## Slide 11 --- Limitations

State honestly:

-   localized prototype,
-   much smaller than global foundation models,
-   limited variables and data,
-   autoregressive errors may accumulate,
-   scenario module is not a physically validated causal simulator.

------------------------------------------------------------------------

## Slide 12 --- Future Work

Possible extensions:

-   spatial gridded modelling,
-   uncertainty quantification,
-   probabilistic forecasting,
-   physics-informed losses,
-   larger multi-region pretraining,
-   data assimilation,
-   domain adaptation.

------------------------------------------------------------------------

# 28. 60-Second Review Explanation

> **Our project, ClimateTwin, is a lightweight AI-driven local climate
> digital twin inspired by two recent Earth-system AI research
> directions. Aurora demonstrates the value of learning a general
> representation through large-scale pretraining and then adapting it to
> downstream tasks through fine-tuning. CREDIT demonstrates the value of
> a modular end-to-end workflow for climate data processing, model
> training, autoregressive inference, and standardized evaluation.**
>
> **We adapt these ideas at a college-project scale. We train
> forecasting models on multivariate climate data, compare baseline,
> LSTM, and Transformer architectures, and perform an Aurora-inspired
> experiment comparing training from scratch against pretraining
> followed by local fine-tuning. We then perform CREDIT-inspired
> multi-step autoregressive forecasting and evaluate how prediction
> error changes with forecast horizon. The resulting models and actual
> experimental outputs are presented through an interactive ClimateTwin
> dashboard.**
>
> **We do not claim to reproduce Aurora or CREDIT. We reproduce selected
> methodological principles at a localized and computationally feasible
> scale.**

------------------------------------------------------------------------

# 29. Master Prompt for an AI Coding Agent

Copy the following prompt into a coding agent:

------------------------------------------------------------------------

## MASTER IMPLEMENTATION PROMPT

I am building a college project called **ClimateTwin: A Lightweight
AI-Driven Local Climate Digital Twin**.

The project is inspired by two 2025 research papers:

1.  **Aurora: A Foundation Model of the Earth System**
2.  **Community Research Earth Digital Intelligence Twin (CREDIT): a
    scalable framework for AI-driven Earth System Modeling**

### IMPORTANT SCIENTIFIC POSITIONING

Do not attempt to reproduce the original Aurora or CREDIT systems.

Do not claim that this project reproduces either paper.

The project should instead implement a lightweight prototype inspired
by:

-   Aurora's **pretraining → fine-tuning → downstream adaptation**
    methodology,
-   Aurora's autoregressive forecasting concept,
-   CREDIT's modular **data → preprocessing → training → inference →
    evaluation** workflow,
-   CREDIT's model comparison and multi-step forecast evaluation.

The implementation must run on a normal laptop for small experiments or
Google Colab for training.

------------------------------------------------------------------------

### OBJECTIVE

Build an end-to-end localized climate forecasting and digital-twin
prototype that:

1.  Loads legitimate public climate/weather data.
2.  Cleans and preprocesses the data.
3.  Uses chronological train/validation/test splits.
4.  Builds temporal sequences.
5.  Implements a persistence baseline.
6.  Implements an LSTM.
7.  Implements a lightweight Transformer.
8.  Implements broad-data pretraining followed by target-location
    fine-tuning.
9.  Implements training from scratch for a fair comparison.
10. Implements autoregressive multi-step forecasting.
11. Evaluates MAE and RMSE.
12. Optionally implements R².
13. Implements ACC only if anomaly climatology is correctly defined.
14. Produces all metrics and figures from actual executions.
15. Builds a Streamlit dashboard using saved model outputs.

------------------------------------------------------------------------

### DATA REQUIREMENTS

Use a legitimate public dataset.

Start with a localized multivariate time series.

Use available variables such as:

-   temperature,
-   humidity,
-   pressure,
-   wind components or wind speed,
-   precipitation.

Do not invent missing variables.

The data pipeline must:

-   remove duplicates,
-   handle missing values,
-   validate timestamps,
-   ensure consistent units,
-   split chronologically,
-   fit normalization only on training data,
-   save preprocessing artifacts,
-   avoid data leakage.

------------------------------------------------------------------------

### MODELS

#### Model A --- Persistence baseline

Use:

``` text
prediction at t+1 = observed state at t
```

#### Model B --- LSTM

Input:

``` text
N previous timesteps × all selected variables
```

Output:

``` text
next climate state
```

#### Model C --- Lightweight Transformer

Use:

``` text
Feature Projection
↓
Positional Encoding
↓
Transformer Encoder
↓
Regression Head
↓
Next Climate State
```

Keep the model computationally feasible.

------------------------------------------------------------------------

### PRETRAINING + FINE-TUNING

Implement:

``` text
BROAD DATA
(multiple locations or broader region)
        ↓
PRETRAIN TRANSFORMER
        ↓
SAVE CHECKPOINT
        ↓
TARGET LOCATION
        ↓
FINE-TUNE
```

Then train an identical Transformer from scratch on the target location.

Compare:

``` text
SCRATCH
VS
PRETRAINED + FINE-TUNED
```

Measure:

-   validation convergence,
-   MAE,
-   RMSE,
-   final validation loss.

Do not fabricate an improvement. Report the actual result even if
pretraining does not help.

------------------------------------------------------------------------

### AUTOREGRESSIVE FORECASTING

Train or use a one-step predictor.

Implement recursive inference:

``` text
state at t
↓
predict t+1
↓
feed prediction into input
↓
predict t+2
↓
repeat
```

Evaluate:

-   1 step,
-   3 steps,
-   6 steps,
-   12 steps,
-   24 steps.

The conversion from steps to hours must depend on the actual dataset
sampling interval.

Create:

``` text
Forecast Horizon vs RMSE
```

------------------------------------------------------------------------

### EVALUATION

Required:

``` text
MAE = mean absolute error
RMSE = root mean squared error
```

Calculate final metrics in original physical units where possible.

ACC must only be used when correctly implemented as anomaly correlation
relative to a climatological reference.

------------------------------------------------------------------------

### REQUIRED FIGURES

1.  Actual vs predicted.
2.  Training loss vs validation loss.
3.  Forecast horizon vs RMSE.
4.  Model comparison.
5.  Scratch vs pretrained + fine-tuned comparison.
6.  Optional extreme-event case study if the test period contains a
    suitable documented event.

All figures must be generated from real model outputs.

------------------------------------------------------------------------

### STREAMLIT DASHBOARD

Create pages:

1.  **Climate Overview**
2.  **AI Forecast**
3.  **Multi-Step Forecast**
4.  **Model Comparison**
5.  **Scenario Sensitivity**

For Scenario Sensitivity, allow controlled perturbations to selected
input variables.

Mandatory dashboard disclaimer:

> Scenario outputs are model-based sensitivity experiments. They are not
> physically validated climate intervention simulations and do not
> establish causal effects.

------------------------------------------------------------------------

### REPOSITORY STRUCTURE

``` text
ClimateTwin/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── forecasting.py
│   ├── scenario.py
│   └── models/
│       ├── persistence.py
│       ├── lstm.py
│       └── transformer.py
├── dashboard/
│   └── app.py
├── results/
│   ├── figures/
│   ├── metrics/
│   └── checkpoints/
├── requirements.txt
├── README.md
└── PROJECT_REVIEW.md
```

------------------------------------------------------------------------

### DEVELOPMENT ORDER

Implement in exactly this order:

1.  Dataset acquisition and validation.
2.  Preprocessing and chronological split.
3.  Persistence baseline.
4.  LSTM.
5.  Transformer.
6.  Evaluation utilities.
7.  Autoregressive rollout.
8.  Broad-data pretraining.
9.  Target-location fine-tuning.
10. Scratch vs fine-tuned comparison.
11. Generate figures.
12. Streamlit dashboard.

At every stage:

-   write modular code,
-   add reproducible configuration,
-   save random seeds,
-   save metrics,
-   save checkpoints,
-   do not fabricate results,
-   clearly distinguish research inspiration from reproduction.

Before proceeding to the next stage, verify that the current stage runs
successfully.

------------------------------------------------------------------------

# 30. Final Positioning Statement

> **ClimateTwin is not a reproduction of Aurora or CREDIT. It is a
> lightweight localized prototype that adapts selected methodological
> principles from recent Earth-system AI research: general-to-local
> transfer learning, modular AI weather-model workflows, autoregressive
> forecasting, and standardized multi-horizon evaluation.**

------------------------------------------------------------------------

# 31. Source Papers

## Aurora

**Bodnar et al. (2025). A Foundation Model of the Earth System.
Nature.**

Key source concepts used here:

-   heterogeneous inputs,
-   encoder → temporal processor → decoder architecture,
-   large-scale pretraining,
-   fine-tuning,
-   recursive forecasting,
-   downstream task adaptation,
-   comparison of pretrained models with training from scratch.

## CREDIT

**Schreck et al. (2025). Community Research Earth Digital Intelligence
Twin: a scalable framework for AI-driven Earth System Modeling. npj
Climate and Atmospheric Science.**

Key source concepts used here:

-   end-to-end AI weather modelling framework,
-   preprocessing,
-   model library / interchangeable architectures,
-   scalable training,
-   single-step and multi-step workflows,
-   standardized evaluation,
-   RMSE and ACC,
-   long-horizon forecast comparison.

------------------------------------------------------------------------

# End of Document
