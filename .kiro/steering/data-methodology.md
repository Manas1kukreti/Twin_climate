---
inclusion: fileMatch
fileMatchPattern:
  - "src/preprocessing.py"
  - "src/dataset.py"
  - "src/config.py"
  - "scripts/download_*.py"
  - "scripts/preprocess.py"
  - "configs/data*.yaml"
  - "notebooks/**/*.ipynb"
  - "tests/test_preprocessing.py"
  - "tests/test_dataset.py"
---

# ClimateTwin Data Methodology

Apply these rules whenever acquiring, cleaning, validating, splitting, normalizing, sequencing, or loading climate data.

## Pipeline Contract

The intended data path is:

```text
legitimate source
→ raw immutable data
→ metadata validation
→ cleaning
→ chronological split
→ train-only preprocessing fit
→ transform train/validation/test
→ sequence construction
→ model-ready datasets
```

Do not rearrange this order in a way that creates leakage.

## Source and Provenance

For each downloaded/ingested dataset, record at minimum:

- source/provider;
- dataset/product name;
- retrieval date;
- source version if available;
- geographic area/location;
- latitude/longitude metadata where relevant;
- requested variables;
- original units;
- time range;
- sampling interval;
- retrieval/query parameters;
- license/access notes when relevant.

Prefer legitimate public sources such as ERA5/ERA5-Land, weather-station data, or legitimately accessible IMD data.

Do not invent unavailable variables.

## Initial Feature Set

Use only available variables. Candidate variables include:

- 2 m temperature;
- relative humidity or a correctly derived humidity variable;
- surface pressure;
- precipitation;
- 10 m wind speed;
- U wind component;
- V wind component;
- optional solar radiation.

If humidity is derived from temperature/dew point or another valid combination, document the formula, units, and assumptions.

## Timestamp Validation

Before splitting or sequence construction:

- parse timestamps explicitly;
- sort chronologically;
- detect duplicate timestamps;
- detect missing timestamps/gaps;
- infer or validate the expected sampling interval;
- detect timezone assumptions if relevant;
- ensure there is no accidental reverse ordering.

Never silently treat irregularly sampled data as regular data.

## Duplicate Handling

Detect duplicates explicitly.

If duplicate timestamps exist, decide whether they are:

- exact duplicate records;
- conflicting measurements;
- multiple stations/locations accidentally collapsed together.

Log the rule used to remove or aggregate them.

## Missing Values

Quantify missingness by variable and time period before choosing a strategy.

The chosen strategy must be appropriate for time series and documented.

Do not interpolate across large gaps without explicit justification.

Never use future test-period information to impute training data.

Record how many values/rows were affected.

## Invalid Values and Units

Validate physical plausibility and dataset-specific valid ranges where documented.

Record every unit conversion.

Do not apply guessed conversions based only on variable names. Check official metadata when uncertain.

## Chronological Split

Use chronological train/validation/test periods.

The split function should make ordering assertions such as:

```text
max(train_timestamp) < min(validation_timestamp)
max(validation_timestamp) < min(test_timestamp)
```

Exact boundary rules should be consistent and tested.

Never randomly shuffle before the split.

## Normalization

Fit normalization using training data only.

For a scaler `S`:

```text
S.fit(train)
train_scaled = S.transform(train)
val_scaled   = S.transform(val)
test_scaled  = S.transform(test)
```

Never call `fit` or `fit_transform` on validation/test data.

Persist the scaler together with:

- ordered feature names;
- units where relevant;
- fitting period;
- scaler type/version;
- any target-column mapping needed for inverse transformation.

## Sequence Construction

Convert data to temporal examples using explicit input and target indexing.

For one-step prediction:

```text
input:  x[t-window+1 : t]
target: y[t+1]
```

Ensure the exact indexing matches Python slicing semantics and test for off-by-one errors.

Candidate input windows include 12, 24, and 48 timesteps, but select them relative to the actual data frequency and experiment design.

Do not allow training sequences to use targets from validation/test periods.

Prefer constructing sequences after the split or using boundary-aware sequence generation with explicit assertions.

## Multiple Locations for Pretraining

Broad pretraining may use multiple locations or a broader region.

Represent location membership explicitly enough to prevent accidental mixing of discontinuous station sequences.

Do not create a temporal sequence whose adjacent rows jump from one city/station to another unless the model design explicitly supports that representation.

For station-based pretraining, group/sort by location and time before sequence construction.

## Data Leakage Tests

Create tests for at least:

- chronological ordering;
- no train/validation/test overlap;
- scaler fit on train only;
- sequence/target alignment;
- no sequences crossing invalid split boundaries;
- stable feature ordering;
- duplicate handling;
- missing-value strategy;
- sampling-interval detection;
- inverse-transform consistency where applicable.

Treat leakage test failures as blockers for model experiments.

## Data Report

Before model training, produce a concise dataset summary containing:

- source;
- target geography;
- date range;
- number of rows/timestamps;
- variables and units;
- sampling interval;
- missingness summary;
- cleaning operations;
- train/validation/test boundaries;
- sequence window;
- resulting example counts per split.

This summary should be reproducible from code, not manually estimated.
