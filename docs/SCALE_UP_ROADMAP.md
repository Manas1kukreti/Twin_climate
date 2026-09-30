# ClimateTwin — Scale-Up & Differentiation Roadmap

_Created: 2026-09-29_

This document captures the proposed plan for growing ClimateTwin from its
current state into a larger, more professional, and more distinctive project,
**including empirically verified constraints** discovered while probing the
Copernicus CDS API. Nothing here has been implemented yet; it is a costed plan.

---

## 1. Current state (baseline)

| Dimension | Value |
|---|---|
| Cities | 5 (Delhi, Mumbai, Bengaluru, Kolkata, Chennai) |
| History | 2018–2024 (7 years, hourly) |
| Variables | 6 (`t2m`, `d2m`, `sp`, `tp`, `u10`, `v10`) |
| Timesteps per city | 61,368 |
| **Total datapoints** | **~1.84 million** |
| Models | Per-city LSTM + persistence baseline |
| Data source | Real ERA5-Land reanalysis (time-series endpoint) |

Per-city temperature skill vs. persistence: Delhi +48%, Mumbai +64%,
Bengaluru +52%, Kolkata +43%, Chennai +48%.

---

## 2. Verified API constraints (measured, not assumed)

These were established by running `scripts/probe_cds_capacity.py` against the
live CDS API. **They materially change what is feasible.**

### 2.1 The time-series endpoint serves a limited variable set

Requesting these against `reanalysis-era5-land-timeseries` returned
**400 Bad Request**:

- `volumetric_soil_water_layer_1`, `volumetric_soil_water_layer_2`
- `surface_net_solar_radiation`, `surface_net_thermal_radiation`
- `total_evaporation`, `potential_evaporation`
- `runoff`, `surface_runoff`

Only `soil_temperature_level_1` was additionally accepted.

**Implication:** the "6 → 15 variables" expansion is **not possible** via the
time-series (single-point) endpoint. Soil moisture and runoff — the variables
that would enable genuine drought and flood modelling — require the **gridded**
`reanalysis-era5-land` dataset, which returns area-subset NetCDF and is far
heavier to download and store.

**Options:**
- **(a)** Keep the 6 core variables + `soil_temperature_level_1` (7 total) via
  the fast time-series endpoint.
- **(b)** Add a separate, coarser gridded pull (e.g. monthly or daily
  aggregates over an India bounding box) purely for the drought/flood hazard
  layer, keeping hourly point data for forecasting. **Recommended** — it gets
  the hazard capability without making the hourly pipeline unmanageable.

### 2.2 Multi-variable batching works

A single request with **7 variables succeeded** (returned 4 NetCDF members).
So variables can be batched per request rather than the current 3 groups of 2,
cutting the request count substantially.

### 2.3 Practical throughput notes

- Small (1-day) requests: ~13–30 s each, dominated by queue time.
- One transient `502 Bad Gateway` was observed and auto-retried by the client,
  so any bulk download driver must be **resumable and retry-tolerant**.
- Full 7-year, 2-variable single-point request (measured earlier): ~60–90 s,
  ~1 MB.

---

## 3. Proposed scale-up

### 3.1 Data volume (revised for real constraints)

| Axis | Now | Proposed | Notes |
|---|---|---|---|
| Cities | 5 | **40** | All states/UTs; coastal ones need land-point validation |
| History | 7 yr | **1995–2026 (~32 yr)** | ERA5-Land supports back to 1950 |
| Variables | 6 | **7** (time-series max) | +`soil_temperature_level_1` |
| Timesteps/city | 61,368 | **~280,000** | |
| **Total datapoints** | 1.84 M | **~78 million** | ~42× current |

Plus an optional gridded hazard layer (daily, India box) for drought/flood.

> ⚠️ **Coastal-city caveat, already hit once:** Chennai's city-centre
> coordinate (80.27°E) fell on the Bay of Bengal and returned **all-NaN**,
> because ERA5-Land is land-only. It was fixed by nudging to 80.20°E. Any
> expansion to 40 cities **must** validate each coordinate lands on land
> before bulk downloading.

### 3.2 Download cost estimate

With variable batching (1 request per city per date-chunk) and 3× parallelism,
40 cities × 32 years is plausibly **2–5 hours** of mostly queue-waiting —
not the 12 hours originally feared. Must be resumable.

---

## 4. Differentiators (what makes it distinctive)

Volume alone is not impressive; methodology is. Ranked by impact:

### 4.1 Graph Neural Network for spatial coupling ⭐ headline
Instead of 40 independent per-city models, **one** model treating cities as
nodes in a graph with distance/correlation-weighted edges, learning how weather
propagates across India. This adapts the GraphCast/Aurora spatial approach to a
city-network scale and is the single most distinctive possible addition.
_New file: `src/models/graph.py`._

### 4.2 Pretrain → finetune transfer learning
`configs/pretrain.yaml` and `configs/finetune.yaml` already exist as
**unresolved stubs** (`pretrain_locations: []`,
`target_location: "UNRESOLVED"`). Implementing pretraining across all cities
then finetuning per city fulfils the project's own Aurora-inspired design and
yields a clean three-way comparison: scratch vs. pretrained vs. finetuned.

### 4.3 Extreme-event stratified evaluation
Report skill **specifically during heatwaves and heavy-rain events**, not only
aggregate RMSE. Average error hides failure on exactly the events that matter,
and this reframes evaluation around decision-relevant performance.

### 4.4 Uncertainty calibration (PICP, CRPS)
`docs/CHECKPOINT.md` lists these as planned-but-not-done. Adding Prediction
Interval Coverage Probability and Continuous Ranked Probability Score turns
"we have uncertainty bands" into "our uncertainty bands are *verified
calibrated*" — a meaningful rigour upgrade given MC-dropout is already in place.

### 4.5 Physics-informed constraints
Enforce physical consistency in training/validation: dewpoint ≤ temperature,
precipitation ≥ 0, soil moisture within bounds. Cheap to add, defensible, and
distinguishes the work from a purely statistical fit.

### 4.6 Hazard engine (drought / flood / heatwave)
- **Heatwave** — IMD thresholds (already implemented in `src/impact.py`).
- **Drought** — SPI-style precipitation-deficit index vs. the long climatology;
  substantially stronger with soil moisture from the gridded layer.
- **Flood** — short-window rainfall accumulation vs. IMD heavy-rain classes;
  stronger with runoff from the gridded layer.

---

## 5. The "digital twin" definition gap

Target definition: _"integrates historical records, ingests real-time
observations, runs continuous ML predictions, and allows what-if simulations."_

| Pillar | Status | Gap |
|---|---|---|
| Integrates historical records | ✅ Done | — |
| Ingests **real-time** observations | ❌ Not possible with ERA5-Land | ERA5-Land is a reanalysis with ~5-day to 2-month latency. True real-time needs a **different source** (e.g. Open-Meteo, no API key; or IMD feeds). |
| Runs continuous ML predictions | ⚠️ Partial | Forecasting exists; needs a scheduled refresh loop |
| What-if simulations | ⚠️ Partial | Heatwave/rain exist; drought + flood hazards missing |

**Key honest point:** the dashboard currently shows December 2024 not because
of a bug but because that is the end of the ERA5-Land test split, and ERA5-Land
*cannot* be real-time. Delivering the real-time pillar requires adding a live
observation feed as a **second data path**, with ERA5-Land retained as the
training and climatology backbone.

---

## 6. Suggested sequencing

1. Validate land coordinates for all 40 candidate cities (cheap, prevents the
   Chennai all-NaN failure at scale).
2. Launch the resumable, parallel, batched 32-year download in the background.
3. While downloading: implement the GNN, calibration metrics, extreme-event
   evaluation, and physics constraints (all independent of the new data).
4. Retrain at scale; produce the GNN vs. LSTM vs. persistence comparison.
5. Add the live-observation path for the real-time pillar.
6. Add the gridded hazard layer for drought/flood.
7. Upgrade the dashboard to surface scale, hazards, calibration, and the model
   comparison.

---

## 7. Risks

- **Retraining churn.** Changing the variable count changes `n_features`, which
  invalidates existing checkpoints. Keep the current 6-variable models as a
  documented baseline for comparison rather than deleting them.
- **Download fragility.** A 502 was already observed; bulk download must
  checkpoint progress and resume.
- **Coastal grid points.** Must be validated per city (see Chennai).
- **Scope vs. timeline.** The GNN and transfer-learning work are the most
  time-consuming items; the calibration metrics and extreme-event evaluation
  are much cheaper and still add visible rigour. If time is short, prefer
  4.3/4.4/4.5 over 4.1/4.2.
