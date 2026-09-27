# ClimateTwin — Progress Checkpoint

_Last updated: 2026-09-27_

This document captures everything set up and built in the current work session,
so it can be picked up later or used as source material for the review PPT.

---

## 1. Summary

ClimateTwin is a localized AI climate digital twin for Indian cities (Delhi
built end-to-end so far). It has three pillars:

1. **Forecasting** — a trained model predicts the next hour's 6 climate
   variables from the previous 24 hours.
2. **Real-time state updation** — the forecaster is rolled forward
   autoregressively to track/advance the current state (multi-step rollout).
3. **Scenario sensitivity** — controlled "what-if" perturbations of input
   variables, compared against a baseline forecast.

**Novelty added this session:**
- **Uncertainty-aware forecasting** via Monte-Carlo (MC) dropout — every
  forecast and scenario carries a predictive confidence band that widens with
  horizon.
- **Human-meaningful impact layer** — raw forecast variables are translated
  into feels-like temperature, humidity, and IMD/NOAA-based heat & rainfall
  alert categories, so outputs are useful to non-specialists.

---

## 2. Environment setup (done)

- Python virtual environment at `.venv/` (Python 3.13.2).
- Installed the package editable with dev + era5 extras:
  `pip install -e ".[dev]"` and `pip install -e ".[era5]"`.
- **Bug fixed:** `pyproject.toml` `build-backend` was an invalid module
  (`setuptools.backends._legacy:_Backend`) → corrected to
  `setuptools.build_meta`. Install failed entirely before this fix.
- Test suite: **125 / 125 passing**; `ruff` clean on new modules.

---

## 3. Data pipeline (done, real data)

- CDS API key configured at `~/.cdsapirc` (chmod 600); ERA5-Land licence
  accepted on the Copernicus site.
- **Bug fixed:** `scripts/download_era5land.py` read `config.get("latitude")`
  but `configs/data.yaml` defines `requested_latitude`/`requested_longitude`
  → added fallback.
- Downloaded full ERA5-Land hourly data for Delhi, 2018–2024
  (single grid point, ~3 MB NetCDF across 3 variable groups).
- Preprocessed into chronological splits:
  - train = 43,824 rows (2018–2022)
  - val = 8,760 rows (2023)
  - test = 8,784 rows (2024) → **8,760 forecast sequences**
  - `StandardScaler` fit on training split only.
- Variables (6): `t2m`, `d2m`, `sp`, `tp`, `u10`, `v10`
  (temperature, dewpoint, surface pressure, precipitation, u/v wind).

---

## 4. Models (LSTM done)

- `ClimateLSTM` trained: 2 layers, hidden=64, dropout=0.1, ~52k params.
- 31 epochs, early-stopped at best epoch 21, val loss 0.0888.
- Test metrics (physical units), LSTM vs persistence baseline:
  - `t2m`: RMSE 0.66 °C (persistence 1.27) — ~48% better
  - `u10`, `v10`: ~14–15% better than persistence
  - `d2m`, `sp`: roughly on par with persistence
- Artifacts: checkpoint, metrics JSON, training log, run manifest, and
  `results/predictions/lstm_delhi_predictions.csv`.

_Transformer and persistence baselines exist in the codebase; only LSTM has
been trained end-to-end this session._

---

## 5. Scenario sensitivity engine (built this session)

**Module:** `src/scenario.py` (implements master-spec §2.9).

- `get_perturbable_features(feature_names, configured)` — builds the control
  set dynamically from the intersection of model features and
  `configs/data.yaml::perturbable_features` (resolved to `t2m`, `d2m`, `sp`).
  No hard-coded variable names.
- `apply_perturbation(window, perturbations, feature_names, scaler)` —
  applies deltas in **physical units** (e.g. `{"t2m": +4.0}` = +4 °C), then
  re-scales into model space. Only the specified variables change.
- `run_scenario(...)` — rolls the model forward `horizon` steps
  autoregressively for both the baseline and perturbed windows, returns a
  `ScenarioResult` with both trajectories (physical units), their difference,
  and — under MC-dropout — predictive bands.

**Key design choices (and why):**
- **Paired MC-dropout:** for each sample, baseline and perturbed rollouts use
  the *same* dropout mask, so their difference isolates the perturbation, not
  mask noise.
- **Horizon = 12 h, non-sustained perturbation:** we iterated through 48h and
  a "sustained" mode. 48h free-rollout was dominated by autoregressive drift
  (phase-shift artifact); sustained re-injection overshot (+4 → +10–18 °C).
  A 12h seed-only perturbation reads correctly: the twin carries ~+3.8 °C at
  hour 1, decaying toward 0 over ~10h. The **widening uncertainty band shows
  when the scenario stops being trustworthy** — this is the point of the
  uncertainty novelty.
- Mandatory disclaimer (`SCENARIO_DISCLAIMER`) attached to every result.

---

## 6. Impact / usefulness layer (built this session)

**Module:** `src/impact.py` — turns forecast variables into human indicators
using standard, citable formulas (not invented coefficients):

- `relative_humidity(t2m, d2m)` — Magnus formula.
- `heat_index(t2m, d2m)` — NOAA "feels-like" apparent temperature.
- `heat_category(t2m)` — IMD heatwave thresholds (plains): onset ≥40 °C,
  heatwave ≥45 °C, severe ≥47 °C.
- `rainfall_category(mm_24h)` — IMD 24-hour rainfall classes
  (light / moderate / heavy / very heavy / extremely heavy).
- `wind_category(u10, v10)` — simple Beaufort-style bands.
- `summarize_state(...)` → `ImpactReport` with feels-like, humidity, and
  colour-coded (red/orange/yellow) alert cards.

**Units caveat:** ERA5-Land `tp` is hourly mm; IMD rainfall categories are
24-hour totals. The demo scales hourly → 24h-equivalent for the label and is
explicit about this.

---

## 7. Demo & figures (built this session)

**Script:** `scripts/scenario_demo.py` — runs two scenarios on the trained
Delhi LSTM (reads artifacts only, no retraining):

1. **Heatwave** (`t2m +4 °C`) →
   `results/figures/scenario_heatwave_t2m_delhi.png`
   - Top: air-temp baseline vs scenario with 90% bands.
   - Bottom: feels-like temperature for both, with IMD 40/45/47 °C reference
     lines. Hour-1 impact: baseline "Normal" → scenario "Hot (heatwave onset)",
     YELLOW alert.
2. **Heavy rain** (`tp +5 mm/h`) →
   `results/figures/scenario_rainfall_tp_delhi.png`
   - Baseline ~dry vs a rain burst decaying over ~6–7h; wide band reflects
     rainfall's inherent unpredictability. Hour-1 impact: "Very light" →
     "Moderate rain" (and the twin also cooled temperature, a sensible learned
     coupling).

---

## 8. Files changed / added this session

**Added:**
- `src/scenario.py` — scenario engine + MC-dropout.
- `src/impact.py` — impact indicators (IMD/NOAA).
- `scripts/scenario_demo.py` — two-scenario demo with impact panels.
- `results/figures/scenario_heatwave_t2m_delhi.png`
- `results/figures/scenario_rainfall_tp_delhi.png`
- `docs/CHECKPOINT.md` (this file)

**Modified:**
- `pyproject.toml` — build-backend fix.
- `scripts/download_era5land.py` — lat/lon config fallback.
- `configs/data.yaml` — resolved `perturbable_features`.
- Regenerated metrics/manifest/provenance under `data/metadata/` and
  `results/` from the real pipeline run.

---

## 9. What the scenario feature is (plain-language, for the PPT)

- **Input:** a city, a start time, a what-if change to one or more variables
  (e.g. "+4 °C"), and a horizon (hours ahead).
- **Output:** baseline vs scenario forecast trajectories with uncertainty
  bands, plus human impact (feels-like temperature, IMD alert level), and the
  quantified difference between them.
- **Usefulness:** (1) probes what the model relies on; (2) measures how long a
  disturbance echoes forward; (3) stress-tests behaviour under unusual
  conditions; (4) the uncertainty band tells you *when to stop trusting it*;
  (5) the impact layer makes it meaningful to non-experts.
- **Honesty:** it is a model-sensitivity/interpretability tool, **not** a
  physically validated climate-intervention simulator. No causal claims.

---

## 10. Not done yet / next steps

- **Review PPT** — pending the user's required format/template.
- **Multi-city** — pipeline generalizes to any lat/lon; only Delhi trained so
  far. Planned: Mumbai, Bangalore, Kolkata, Chennai.
- **Transformer** — trained for Delhi in the codebase's history but not
  re-run this session.
- **Streamlit dashboard** — architected in the spec (§2.10), not built.
- **Formal uncertainty calibration metrics** (PICP, CRPS) — described as
  planned; MC-dropout mechanism is in place to support them.

---

## 11. Reproduce the scenario demo

```bash
source .venv/bin/activate
python scripts/scenario_demo.py
# figures -> results/figures/scenario_heatwave_t2m_delhi.png
#            results/figures/scenario_rainfall_tp_delhi.png
```
