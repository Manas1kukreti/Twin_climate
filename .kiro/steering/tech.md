---
inclusion: always
---

# ClimateTwin Technology Stack

## Primary Language

Use Python as the primary implementation language.

Prefer Python 3.11 unless an actual dependency or runtime constraint requires another supported version. Keep the code compatible with normal local development and Google Colab wherever practical.

## Core Libraries

Prefer the following initial stack:

- PyTorch for LSTM and Transformer models;
- NumPy for numerical operations;
- pandas for station-level tabular/time-series data;
- xarray for NetCDF/gridded climate data when needed;
- scikit-learn for scalers, simple baselines, and standard metrics where appropriate;
- joblib or an equivalent simple mechanism for saving preprocessing artifacts;
- PyYAML for reproducible experiment configuration if YAML configs are used;
- matplotlib for research figures;
- Plotly when interactive dashboard visualization materially improves the Streamlit experience;
- Streamlit for the digital-twin dashboard;
- pytest for tests;
- Ruff for fast Python linting and static quality checks.

For ERA5/ERA5-Land acquisition, prefer the official Copernicus Climate Data Store API and `cdsapi` when that source is selected.

Do not introduce heavyweight infrastructure such as Airflow, Kubernetes, Ray, Spark, MLflow, DVC, Hydra, or a database unless a concrete project requirement justifies it and the user approves the additional complexity.

## MCP Responsibilities

The workspace currently has research/development MCP tools available. Use them deliberately:

- Tavily: discover current public research, documentation, datasets, and relevant official sources.
- Fetch: retrieve and inspect a known webpage or document URL.
- Context7: verify current library APIs and usage patterns before relying on uncertain framework details.
- Playwright: perform real-browser checks of the Streamlit dashboard when browser-level validation is useful.

Do not use web-search tools as a substitute for reading the selected dataset's official metadata and documentation.

## Data Formats

Prefer transparent, inspectable formats:

- raw source formats such as NetCDF/GRIB when provided by the source;
- Parquet or CSV for processed station-level datasets when appropriate;
- JSON/YAML for run manifests and configuration;
- `.pt`/`.pth` for PyTorch checkpoints;
- PNG/SVG/PDF as appropriate for generated research figures.

Do not commit large raw climate datasets or large model checkpoints to Git unless explicitly intended.

## Configuration

Separate experiment configuration from model code.

Configuration should be able to record at least:

- data source/version;
- target geography/location;
- selected variables;
- sampling interval;
- split boundaries;
- input window length;
- model architecture;
- optimizer;
- learning rate;
- batch size;
- epochs;
- early-stopping settings;
- random seed;
- device selection;
- checkpoint/output paths.

Avoid hard-coded local absolute paths inside research code.

## Reproducibility

Create and use a central seed utility. Seed Python, NumPy, and PyTorch where relevant.

Each serious experiment should record:

- run ID;
- seed;
- software/runtime versions where practical;
- device;
- dataset and split metadata;
- model configuration;
- training configuration;
- checkpoint path;
- metrics path;
- Git commit when Git is available.

Do not claim perfect deterministic reproducibility across all hardware/platform combinations. The goal is controlled, traceable, repeatable experimentation.

## Model Constraints

Keep the initial models intentionally lightweight.

Suggested Transformer starting region:

- embedding dimension: 64 or 128;
- attention heads: 4 or 8;
- encoder layers: 2 to 4;
- dropout: around 0.1.

These are starting points, not Aurora architecture claims.

The LSTM should support multivariate sequences and configurable sequence length, hidden dimension, number of layers, dropout, and learning rate.

## Training Behavior

Training must be explicit. Do not trigger long model training automatically from file-save hooks or dashboard startup.

Use quick CPU smoke tests for shapes, forward passes, serialization, and small synthetic-data checks before expensive training.

Use validation data for model selection and early stopping when applicable. Never use test data for hyperparameter selection.

## Dashboard Behavior

The Streamlit application should consume saved artifacts, checkpoints, forecasts, and metrics. It should not silently retrain models during ordinary dashboard interaction.

Keep expensive operations cached or precomputed where appropriate.

## Code Quality

Prefer:

- type hints for public functions;
- small composable functions;
- clear docstrings for research-critical logic;
- explicit shape comments/docstrings for model tensors;
- descriptive names for physical variables and units;
- exceptions with actionable messages;
- tests for leakage-sensitive and alignment-sensitive code.

Avoid opaque notebook-only logic. Important reusable logic belongs in `src/` and notebooks should call it rather than duplicate it.
