# ClimateTwin — task runner
#
# Common project commands wrapped as Make targets. Everything runs through the
# project virtual environment at .venv, so you do not need to activate it first.
#
# Quick start:
#   make venv install   # one-time setup
#   make test           # run the test suite
#   make dashboard      # launch the Streamlit dashboard
#
# Run `make help` (or just `make`) to list all targets.

# --- Configuration -----------------------------------------------------------
VENV        ?= .venv
PY          := $(VENV)/bin/python
PIP         := $(VENV)/bin/pip
STREAMLIT   := $(VENV)/bin/streamlit
RUFF        := $(VENV)/bin/ruff
PYTEST      := $(VENV)/bin/pytest

DATA_CFG    := configs/data.yaml
LSTM_CFG    := configs/lstm.yaml
TRANS_CFG   := configs/transformer.yaml

DASH_PORT   ?= 8501

# Directories linted/formatted by ruff (mirrors pyproject `tool.ruff.src`).
CODE_DIRS   := src scripts tests dashboard

.DEFAULT_GOAL := help

# --- Help --------------------------------------------------------------------
.PHONY: help
help: ## Show this help
	@echo "ClimateTwin — available targets:"
	@echo ""
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Override the venv or dashboard port, e.g.:  make dashboard DASH_PORT=8600"

# --- Environment setup -------------------------------------------------------
.PHONY: venv
venv: ## Create the virtual environment at $(VENV) if missing
	@test -d $(VENV) || python3 -m venv $(VENV)
	@echo "venv ready at $(VENV)"

.PHONY: install
install: venv ## Install the package with dev extras (editable)
	$(PIP) install -e ".[dev]"

.PHONY: install-era5
install-era5: venv ## Install ERA5 download extras (cdsapi, netcdf4)
	$(PIP) install -e ".[era5]"

# --- Code quality ------------------------------------------------------------
.PHONY: lint
lint: ## Lint with ruff
	$(RUFF) check $(CODE_DIRS)

.PHONY: format
format: ## Auto-fix lint issues and format with ruff
	$(RUFF) check --fix $(CODE_DIRS)
	$(RUFF) format $(CODE_DIRS)

.PHONY: test
test: ## Run the full test suite
	$(PYTEST)

.PHONY: check
check: lint test ## Lint then run tests (CI-style gate)

# --- Data pipeline -----------------------------------------------------------
.PHONY: download
download: ## Download ERA5-Land data for Delhi (needs ~/.cdsapirc and era5 extras)
	$(PY) scripts/download_era5land.py --config $(DATA_CFG)

.PHONY: download-cities
download-cities: ## Download real ERA5-Land for all extra cities (needs ~/.cdsapirc)
	$(PY) scripts/download_cities_era5land.py

.PHONY: download-pilot
download-pilot: ## Download a 48-hour pilot slice of ERA5-Land data
	$(PY) scripts/download_era5land.py --config $(DATA_CFG) --pilot

.PHONY: preprocess
preprocess: ## Clean, split, and scale the raw data into data/processed
	$(PY) scripts/preprocess.py --config $(DATA_CFG)

# --- Training ----------------------------------------------------------------
.PHONY: train-lstm
train-lstm: ## Train the ClimateLSTM forecaster
	$(PY) scripts/train_lstm.py --config-data $(DATA_CFG) --config-model $(LSTM_CFG)

.PHONY: train-transformer
train-transformer: ## Train the Transformer forecaster
	$(PY) scripts/train_transformer.py

.PHONY: train
train: train-lstm ## Alias for the primary model (LSTM)

# --- Multi-city ---------------------------------------------------------------
CITIES ?= mumbai bengaluru kolkata chennai

.PHONY: cities
cities: ## Preprocess + train every extra city (after download-cities)
	@for c in $(CITIES); do \
		echo "=== preprocess $$c ==="; $(PY) scripts/preprocess.py --city $$c; \
		echo "=== train $$c ==="; $(PY) scripts/train_lstm.py --city $$c; \
	done

.PHONY: map-db
map-db: ## Build the map DB + fetch live weather (Open-Meteo)
	$(PY) scripts/build_map_db.py

.PHONY: map-db-offline
map-db-offline: ## Build the map DB without the live weather fetch
	$(PY) scripts/build_map_db.py --no-live

.PHONY: refresh-live
refresh-live: map-db ## Refresh live observations (alias for map-db)

# --- Analysis & demos --------------------------------------------------------
.PHONY: figures
figures: ## Regenerate evaluation figures
	$(PY) scripts/generate_figures.py

.PHONY: scenario-demo
scenario-demo: ## Run the what-if scenario demo (writes results/figures/*.png)
	$(PY) scripts/scenario_demo.py

# --- Dashboard ---------------------------------------------------------------
.PHONY: dashboard
dashboard: ## Launch the Streamlit dashboard (override port with DASH_PORT=)
	$(STREAMLIT) run dashboard/app.py --server.port $(DASH_PORT)

# --- Convenience pipelines ---------------------------------------------------
.PHONY: pipeline
pipeline: preprocess train-lstm figures ## Preprocess -> train LSTM -> figures

# --- Housekeeping ------------------------------------------------------------
.PHONY: clean
clean: ## Remove caches and build artifacts (keeps data/ and results/)
	rm -rf .pytest_cache .ruff_cache .pytest_tmp climatetwin.egg-info
	find . -type d -name __pycache__ -not -path "./$(VENV)/*" -exec rm -rf {} +
	@echo "cleaned caches"
