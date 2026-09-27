---
inclusion: always
---

# ClimateTwin Product Overview

## Project Identity

ClimateTwin is a lightweight AI-driven local climate digital twin for a college-scale research project.

It is inspired by methodological ideas from:

- Aurora: A Foundation Model of the Earth System (Nature, 2025)
- Community Research Earth Digital Intelligence Twin (CREDIT): a scalable framework for AI-driven Earth System Modeling (npj Climate and Atmospheric Science, 2025)

ClimateTwin does not reproduce either system. It adapts selected research ideas at a computationally feasible scale.

Primary research specification:

#[[file:docs/research/ClimateTwin_Research_Implementation_Prompt.md]]

If this steering file and the research specification appear to conflict on a scientific requirement, follow the research specification unless the user explicitly approves a documented change.

## Product Goal

Build an end-to-end localized multivariate climate/weather forecasting prototype that:

1. acquires legitimate public climate/weather data;
2. validates, cleans, and preprocesses the data;
3. uses chronological train/validation/test splits;
4. creates temporal input sequences;
5. implements a persistence baseline;
6. implements a multivariate LSTM;
7. implements a lightweight Transformer;
8. compares target-only scratch training against broad-data pretraining followed by local fine-tuning;
9. performs autoregressive multi-step forecasting;
10. evaluates models using actual executed results;
11. visualizes forecast behavior and model comparisons;
12. exposes saved experiment outputs through a Streamlit digital-twin dashboard.

## Research Questions

### RQ1 — Transfer Learning

Does broader climate-data pretraining improve localized forecasting compared with training the same Transformer architecture from scratch on the target location?

Compare:

- Scratch Transformer: random initialization → target-location training.
- Transfer Transformer: broad-data pretraining → target-location fine-tuning.

Measure at minimum:

- MAE;
- RMSE;
- convergence behavior;
- best/final validation loss as appropriate.

Do not assume pretraining improves performance. Report the observed result.

### RQ2 — Forecast Horizon

How does error change as a one-step model is recursively rolled forward?

Evaluate these forecast horizons where supported by the dataset and model:

- 1 step;
- 3 steps;
- 6 steps;
- 12 steps;
- 24 steps.

Do not convert steps to hours unless the actual sampling interval supports that conversion.

### RQ3 — Model Architecture

How do different forecasting architectures compare on the selected localized multivariate forecasting task?

Required comparison set:

- Persistence baseline;
- LSTM;
- Lightweight Transformer trained from scratch;
- Pretrained + fine-tuned Transformer.

A conventional ML baseline such as Random Forest may be added if useful, but it is not allowed to delay the required experiments.

## Initial Scope

Prefer a station-level or single-location prototype first. Suitable initial targets include Delhi, Mumbai, Bengaluru, one Indian state, or a small India-centered region, depending on legitimate data availability.

Progression should be:

single location → multiple locations → optional spatial grid.

Do not delay the core project to implement a spatial-grid extension.

The project must remain feasible on a normal laptop for small experiments and/or Google Colab for GPU training.

## Required Dashboard Experience

The Streamlit dashboard should expose five logical pages or equivalent navigation sections:

1. Climate Overview
2. AI Forecast
3. Multi-Step Forecast
4. Model Comparison
5. Scenario Sensitivity

The dashboard must display saved real experiment outputs rather than fabricated or placeholder scientific results.

The Scenario Sensitivity feature perturbs model inputs and compares baseline versus perturbed trajectories. It must clearly state that the output is a model-based sensitivity experiment, not a physically validated climate intervention simulator and not evidence of causal real-world effects.

## Development Order

Implement the project incrementally in this order:

1. dataset acquisition and validation;
2. preprocessing and chronological split;
3. persistence baseline;
4. LSTM;
5. lightweight Transformer;
6. evaluation utilities;
7. autoregressive rollout;
8. broad-data pretraining;
9. target-location fine-tuning;
10. scratch vs fine-tuned comparison;
11. figure generation;
12. Streamlit dashboard.

Do not skip a prerequisite stage merely to produce a demo faster. Verify each stage before depending on its artifacts.

## Definition of Success

A successful ClimateTwin implementation is reproducible, scientifically honest, computationally feasible, modular, and review-ready. It answers the research questions using actual executed experiments, preserves a traceable path from data and configuration to checkpoints and metrics, and clearly distinguishes research inspiration from reproduction.
