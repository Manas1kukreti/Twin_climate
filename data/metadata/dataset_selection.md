# ClimateTwin — Dataset Selection Research (Task 1.1)

**Date:** 2026-08-30 (revised 2026-08-30)
**Phase:** 1 — Dataset Acquisition and Validation
**Status:** Recommendation ready — awaiting user approval for Task 1.2
**Revision:** 2.0 — corrected per user review

---

## 1. Candidate Dataset Comparison

| Criterion | ERA5-Land (CDS Time-Series) | ERA5 Single Levels (CDS Time-Series) | NOAA ISD (Integrated Surface Database) | IMD (India Meteorological Department) |
|---|---|---|---|---|
| **Full name** | ERA5-Land hourly time-series data from 1950 to present | ERA5 hourly time-series data on single levels from 1940 to present | NOAA Integrated Surface Database (ISD) / Global Hourly | India Meteorological Department — Data Service Portal |
| **Provider** | Copernicus Climate Change Service (C3S) / ECMWF | Copernicus Climate Change Service (C3S) / ECMWF | NOAA National Centers for Environmental Information (NCEI) | IMD, Ministry of Earth Sciences, Government of India |
| **Source URL** | https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-timeseries | https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries | https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database | https://dsp.imdpune.gov.in/ |
| **Product User Guide** | https://confluence.ecmwf.int/pages/viewpage.action?pageId=536218894 | — | — | — |
| **Data type** | **Reanalysis grid cell** (0.1° × 0.1°, ~11 km) | **Reanalysis grid cell** (0.25° × 0.25°, ~25 km) | **Weather station observation** (point measurement) | **Weather station observation** (point measurement) |
| **License** | CC-BY 4.0 (from 2 July 2025); free, including commercial use, with attribution | CC-BY 4.0 (from 2 July 2025); free, including commercial use, with attribution | Open / public domain (NOAA open data); attribution requested | Restricted: data for stated purpose only; cannot be passed to third parties without prior written IMD approval; commercial-purpose restrictions apply; reproduction for commercial use not permitted |
| **Authentication** | Free CDS account + accept license; API key via `.cdsapirc` | Free CDS account + accept license; API key via `.cdsapirc` | None required (public HTTPS / AWS S3) | Account registration at Data Service Portal; formal data request process; data charges may apply for non-MoES parties |
| **Geographic coverage** | Global (land areas) | Global (atmosphere + land + ocean) | Global (~35,000 stations; best coverage in N. America, Europe, parts of Asia) | India only |
| **Temporal coverage** | 1950–present | 1940–present | 1901–present (varies by station) | Varies; many series start mid-20th century |
| **Native sampling interval** | **Hourly** | **Hourly** | Hourly / synoptic (varies by station; often 1h or 3h) | Typically 3-hourly or daily for station data available through the portal |
| **Spatial resolution** | 0.1° × 0.1° (~11 km) | 0.25° × 0.25° (~25 km) | Point station | Point station |
| **India availability** | ✅ Full land coverage (every land grid cell) | ✅ Full coverage (every grid cell) | ⚠️ Partial — limited active Indian stations with good hourly completeness | ✅ Indian stations available through the portal |
| **Delhi availability** | ✅ Grid cells cover Delhi | ✅ Grid cells cover Delhi | ⚠️ Delhi/Safdarjung and Palam stations exist but data gaps common | ✅ Delhi stations available |
| **Mumbai availability** | ✅ Grid cells cover Mumbai | ✅ Grid cells cover Mumbai | ⚠️ Mumbai/Santacruz station exists; completeness varies | ✅ Mumbai stations available |
| **Bengaluru availability** | ✅ Grid cells cover Bengaluru | ✅ Grid cells cover Bengaluru | ⚠️ Bengaluru station exists; completeness varies | ✅ Bengaluru stations available |
| **File format** | NetCDF and CSV (time-series endpoint) | NetCDF and CSV (time-series endpoint) | ASCII fixed-width / CSV | Various formats via portal |
| **Automated acquisition** | ✅ `cdsapi` Python package; point extraction via time-series endpoint | ✅ `cdsapi` Python package; point extraction via time-series endpoint | ✅ Direct HTTPS / AWS S3 download; no API key needed | ⚠️ Portal-based request system; no public bulk-download API for historical hourly data |
| **Reproducibility** | ✅ High — versioned dataset, DOI, deterministic grid | ✅ High — versioned dataset, DOI, deterministic grid | ⚠️ Medium — station data may be revised; stations go inactive | ⚠️ Lower — redistribution restricted; portal-based access harder to script reproducibly |

### Variable Availability Comparison

| Variable | ERA5-Land TS | ERA5 SL TS | NOAA ISD | Notes |
|---|---|---|---|---|
| 2 m temperature | ✅ (K) | ✅ (K) | ✅ (°C × 10) | All three provide this |
| 2 m dew point temperature | ✅ (K) | ✅ (K) | ✅ (°C × 10) | All three provide this |
| Relative humidity (2 m) | ⚠️ Derivable from T and Td | ⚠️ Derivable from T and Td | ✅ Direct (some stations) | Legitimate derivation from T & Td using Magnus formula |
| Surface pressure | ✅ (Pa) | ✅ (Pa) | ⚠️ Station pressure + sea-level pressure; variable completeness | ERA5/ERA5-Land provide this consistently |
| Total precipitation | ✅ (m; de-accumulated to hourly by time-series product) | ✅ (m; de-accumulated to hourly by time-series product) | ⚠️ Varies by station; often only periodic totals | See §8 on precipitation handling |
| 10 m u-component of wind | ✅ (m/s) | ✅ (m/s) | ❌ Not directly (wind speed + direction reported) | Can derive u/v from speed+direction for ISD |
| 10 m v-component of wind | ✅ (m/s) | ✅ (m/s) | ❌ Not directly | Same as above |
| Wind speed (10 m) | ⚠️ Derivable from u, v | ⚠️ Derivable from u, v | ✅ Direct | Derivable: √(u² + v²) |
| Solar radiation (downward shortwave) | ✅ (J/m²; de-accumulated to hourly by time-series product) | ✅ (J/m²) | ❌ Not available | Not included in initial variable set |

---

## 2. Candidate Target-Location Comparison

| Criterion | Delhi | Mumbai | Bengaluru |
|---|---|---|---|
| **Approximate requested coordinates** | 28.61°N, 77.21°E | 19.08°N, 72.88°E | 12.97°N, 77.59°E |
| **Actual returned grid coordinates** | To be recorded after acquisition | To be recorded after acquisition | To be recorded after acquisition |
| **Climate type** | Semi-arid (BSh); extreme seasonal variation; hot summers, cool winters, monsoon | Tropical wet-dry (Aw); moderate temperature range; heavy monsoon | Tropical savanna (Aw); mild year-round; moderate monsoon |
| **Seasonal temperature range** | ~5°C (Jan) to ~45°C (May–Jun) — large amplitude | ~20°C (Jan) to ~35°C (May) — moderate amplitude | ~15°C (Jan) to ~36°C (Apr) — moderate amplitude |
| **Monsoon intensity** | Moderate–heavy (Jul–Sep) | Very heavy (Jun–Sep) | Moderate (Jun–Oct, bimodal) |
| **Research interest** | High: large seasonal amplitude, extreme heat events, pollution-weather coupling, major population center | High: coastal dynamics, heavy monsoon, flooding events | Moderate: mild climate, less extreme variation |
| **Suitability for forecasting challenge** | ✅ High — large seasonal signals + monsoon transitions make forecasting interesting and testable | ✅ High — monsoon extremes test model limits | ⚠️ Lower — milder climate may produce less differentiation between models |
| **Documented extreme events in recent years** | Heatwaves (2022, 2023, 2024), cold waves, heavy rainfall events | Flooding (2023, 2024), cyclonic events | Less frequent documented extremes |
| **Multi-location pretraining feasibility** | ✅ Central location among Indian cities; Mumbai and Bengaluru available as pretraining locations | ✅ Coastal location; Delhi and Bengaluru as pretraining | ✅ Southern location; Delhi and Mumbai as pretraining |
| **ERA5-Land grid cell terrain** | Relatively flat (Indo-Gangetic plain) — grid cell likely representative of local conditions | Coastal — grid cell may include mixed land-sea; needs care | Plateau (~900 m elevation) — grid cell well-defined |

---

## 3. Recommended Dataset

**ERA5-Land hourly time-series data from 1950 to present**

Dataset identifier: `reanalysis-era5-land-timeseries`
DOI: 10.24381/ee82e357
Product User Guide: https://confluence.ecmwf.int/pages/viewpage.action?pageId=536218894

### Rationale

1. **Highly complete and consistent hourly data** for any land grid cell globally. Reanalysis methodology produces a spatially and temporally continuous record. However, actual downloaded timestamp continuity and value completeness must still be validated during raw-data inspection (Task 1.3).

2. **Higher spatial resolution** (0.1° ≈ 11 km) than ERA5 (0.25° ≈ 25 km). For a localized prototype, the finer grid better represents local surface conditions.

3. **All required ClimateTwin variables are available**: 2 m temperature, 2 m dewpoint temperature (from which relative humidity is legitimately derivable), surface pressure, total precipitation (de-accumulated to hourly by the time-series product), 10 m u-wind, 10 m v-wind.

4. **Consistent time-series point extraction** via the CDS time-series endpoint, returning CSV or NetCDF directly for a single grid point. This avoids downloading large gridded files — suitable for a laptop-scale prototype.

5. **Free, open license** (CC-BY 4.0 from 2 July 2025) with no redistribution restrictions beyond attribution. Fully compatible with a college research project.

6. **Reproducible access** via `cdsapi` with deterministic grid coordinates, versioned dataset (DOI: 10.24381/ee82e357), and logged retrieval parameters.

7. **Multi-location pretraining support**: The same dataset covers any Indian city. Extracting time series for Mumbai, Bengaluru, Chennai, Kolkata, Hyderabad, etc. uses identical methodology and variable definitions. Location-to-location variable consistency is guaranteed by the reanalysis framework.

8. **Autoregressive multivariate forecasting suitability**: Hourly temporal resolution with consistent coverage is well-suited for sequence-to-sequence time-series modeling. The multivariate feature set covers the primary atmospheric state variables relevant to near-surface weather.

### Why not ERA5 (single-levels)?

ERA5 single-levels is also a strong candidate. The ERA5-Land time-series endpoint provides the surface variables ClimateTwin needs at higher resolution. ERA5 offers additional variables (e.g., boundary-layer height, cloud variables) and extends back to 1940, but these extras are not required for the ClimateTwin prototype. ERA5's coarser 25 km resolution is adequate but offers no advantage for point-level forecasting.

**If ERA5-Land time-series access proves unreliable** during Task 1.2, ERA5 single-levels time-series is the direct fallback — same API, same license, slightly different dataset identifier and resolution.

### Why not NOAA ISD?

- Indian station coverage is **sparse and inconsistent**. Hourly completeness for Delhi, Mumbai, and Bengaluru stations varies year-to-year.
- **Missing variables**: No direct u/v wind components, no solar radiation, variable station-pressure reporting.
- **Data gaps** require complex imputation, introducing preprocessing risk.
- Lower reproducibility — stations go inactive, data revisions are common.

NOAA ISD could serve as an independent validation source in future phases but is not recommended as the primary training dataset.

### Why not IMD?

The IMD Data Service Portal (https://dsp.imdpune.gov.in/) is operational and actively processing data requests. However, IMD data is not recommended as the primary dataset for the following reasons:

- **Acquisition process**: Data procurement requires account registration, a formal request form, and a portal-based workflow. There is no public bulk-download API for scripted, reproducible retrieval of historical hourly data. Data charges may apply for parties outside the Ministry of Earth Sciences.
- **Redistribution restrictions**: IMD data supply terms state that data is for the stated purpose only, must not be passed to third parties without prior written IMD approval, and reproduction for commercial purpose is not permitted. These constraints make the data less suitable for a shareable, reproducible college research project.
- **Reproducibility**: Portal-based acquisition is harder to script reproducibly compared to a deterministic API call. Other researchers would need to repeat the formal request process to obtain the same data.
- **Variable coverage and format**: Hourly multivariate station data availability through the portal has not been verified to cover all six required variables at consistent hourly resolution for Delhi.

IMD gridded data (available separately) covers temperature and precipitation but not the full multivariate set ClimateTwin requires.

---

## 4. Source Type

**Reanalysis grid cell** — not a weather station observation.

ERA5-Land grid cells at (lat, lon) represent model-assimilated estimates for a spatial grid point (~11 km resolution), not point measurements from a physical instrument. This distinction must be maintained in all documentation, configs, dashboard displays, and research output.

The exact returned grid coordinates will be recorded after data acquisition. The time-series endpoint automatically selects the nearest 0.1° grid point to the requested coordinates.

---

## 5. Recommended Target Location

**Delhi** (nearest ERA5-Land grid cell to approximately 28.6°N, 77.2°E)

The exact grid-cell coordinates returned by the CDS API will be recorded during Task 1.2/1.3 and used as the authoritative location metadata.

### Rationale

1. **Large seasonal amplitude** (~5°C to ~45°C) provides a strong, learnable signal for temperature forecasting — models must capture both winter cooling and summer heating.

2. **Monsoon transition** (June–September) creates interesting multi-variable dynamics: temperature drops, humidity surges, wind-pattern shifts, heavy precipitation. This tests multivariate model capability.

3. **Documented extreme events** in recent test-period years (heatwaves in 2022–2024, cold waves, heavy rainfall episodes) — potentially usable for an optional extreme-event case study.

4. **Flat terrain** (Indo-Gangetic plain) means the ERA5-Land grid cell is more likely to be representative of local conditions without significant sub-grid terrain effects compared to mountainous or coastal locations.

5. **Multi-location pretraining feasibility**: Mumbai and Bengaluru (climatically distinct — coastal tropical and plateau tropical respectively) provide natural pretraining locations with different climate regimes, testing whether diverse climate exposure helps a Transformer adapt to Delhi's semi-arid conditions.

6. **Practically motivated**: Delhi is India's capital and a widely studied city for weather/climate impacts, making the work contextually relevant.

---

## 6. Recommended Native Sampling Interval

**Hourly (1 hour)**

This is the native temporal resolution of ERA5-Land. Using hourly data:
- Maximizes sequence resolution for the LSTM and Transformer.
- Allows experimenting with different input-window lengths (e.g., 24, 48, 72 timesteps of context).
- Supports fine-grained evaluation of autoregressive rollout degradation.
- Steps in the forecast-horizon experiment correspond to 1-hour increments once the sampling interval is validated against the actual downloaded data.

---

## 7. Recommended Initial Historical Period

**2018-01-01 to 2024-12-31** (7 years)

### Rationale

- 7 years of hourly data ≈ 61,368 hourly records (accounting for leap years).
- Sufficient for chronological train/val/test split with multiple complete annual cycles in training.
- Recent enough that ERA5-Land data quality is high (denser modern observation network feeding the reanalysis).
- Includes documented extreme events in 2022–2024 for potential case-study use.
- **Approximate data volume**: For 6 variables at hourly resolution for a single grid point over 7 years, the CSV is expected to be approximately 15–25 MB — trivially small for a laptop.

**Split date boundaries are NOT resolved here.** Indicative orientation (to be determined in Phase 2 based on actual data inspection):
- Train: ~2018–2022
- Validation: ~2023
- Test: ~2024

---

## 8. Recommended Initial Variables

The following six variables are available from the ERA5-Land time-series product. Variable names are from the official ERA5-Land time-series Product User Guide (PUG, Tables 1–3):

| # | CDS variable name | PUG name | Unit (native) | Unit (converted for use) | PUG group | PUG notes |
|---|---|---|---|---|---|---|
| 1 | `2m_temperature` | 2 metre temperature | K | °C | 2m temperature | Instantaneous |
| 2 | `2m_dewpoint_temperature` | 2 metre dewpoint temperature | K | °C | 2m temperature | Instantaneous |
| 3 | `surface_pressure` | Surface pressure | Pa | hPa | Pressure and precipitation | Instantaneous |
| 4 | `total_precipitation` | Total precipitation | m | mm | Pressure and precipitation | **De-accumulated** to hourly by the time-series product |
| 5 | `10m_u_component_of_wind` | 10 metre U wind component | m s⁻¹ | m/s | Wind | Instantaneous |
| 6 | `10m_v_component_of_wind` | 10 metre V wind component | m s⁻¹ | m/s | Wind | Instantaneous |

### Precipitation handling

According to the ERA5-Land time-series Product User Guide (PUG), accumulation variables such as `total_precipitation` are **de-accumulated to hourly resolution** when the ARCO time-series product is generated. The de-accumulation procedure is documented in the PUG with a reference Jupyter notebook.

Therefore, the returned precipitation values represent hourly totals, not running accumulations. During Task 1.2 and Phase 2, the returned time-series values and timestamps must be validated, and precipitation should be converted from metres to millimetres as appropriate (1 m = 1000 mm).

### Solar radiation

`surface_solar_radiation_downwards` is available in the time-series product (PUG Table 4, Radiation and heat group, de-accumulated). It is **not included in the initial download** because:
- The six core variables already cover the primary atmospheric state (temperature, humidity proxy, pressure, precipitation, wind).
- Adding solar radiation increases the variable count without a documented scientific necessity for the initial prototype.
- It can be added in a later phase if analysis reveals it would improve forecast skill.

### Target/feature mapping — UNRESOLVED (§9.13)

All six selected variables are dynamic climate-state variables. The mapping of which variables are **targets** (predicted by the model) versus **input-only features** is an **unresolved design decision** under §9.13.

The specification's preferred default is `n_targets == n_features` (predict all selected variables), which simplifies autoregressive rollout because every predicted variable can be fed back directly. However, this decision has not been formally made.

**Autoregressive consequence**: If any variable is designated as input-only (not predicted by the model), its future values cannot be silently taken from ground truth during recursive forecasting. The autoregressive rollout must explicitly define how non-predicted features are supplied, or raise an error if they are unavailable. This constraint must be addressed when §9.13 is resolved.

---

## 9. Variables Requiring Legitimate Derivation (Optional — Phase 2 Decision)

### Relative humidity (from 2 m temperature and 2 m dew point)

ERA5-Land provides 2 m temperature (T) and 2 m dew point temperature (Td) directly, but not relative humidity as a native variable.

**Derivation method (Magnus formula):**

```
RH = 100 × exp((b × Td) / (c + Td)) / exp((b × T) / (c + T))
```

Where (Alduchov and Eskridge, 1996):
- b = 17.625
- c = 243.04 °C
- T and Td in °C

This is a standard, well-documented meteorological derivation.

**Decision required (Phase 2):** Whether to include relative humidity as a derived feature (replacing or supplementing raw dew point) will be decided during preprocessing. The raw variables (T and Td) will be downloaded. Derived and raw representations should not be duplicated without documenting the rationale.

### Wind speed (from u and v components)

If scalar wind speed is desired as a feature:

```
wind_speed_10m = √(u² + v²)
```

**Decision required (Phase 2):** Whether to use (u, v) components directly, derive scalar wind speed, or use both. Duplication should not occur without documented rationale.

---

## 10. Suitability for Multi-Location Pretraining

**Confirmed feasible.**

ERA5-Land provides identical variable definitions, units, and temporal resolution for every land grid cell globally. For the Aurora-inspired pretraining experiment:

- **Candidate pretraining cities**: Mumbai, Bengaluru, Chennai, Kolkata, Hyderabad, Jaipur (and others).
- All use the same dataset, same variables, same units, same hourly resolution.
- Each city has a distinct climate regime, providing diversity for pretraining.
- Per the specification, Delhi data should preferably be **excluded** from pretraining, or at minimum Delhi's validation/test periods must be excluded.

**Pretraining geography is NOT resolved here.** Exact city selection will be determined during Phase 8.

---

## 11. License and Access Constraints

- **License**: CC-BY 4.0 (effective 2 July 2025). Free for all uses including commercial; attribution required.
- **Required attribution**: "Contains modified Copernicus Climate Change Service information [year]."
- **Access**: Requires free ECMWF/CDS account registration + acceptance of license terms.
- **API**: `cdsapi` Python package with API key stored in `~/.cdsapirc` (must not be committed to Git).
- **Rate limits**: CDS processes requests in a queue. For single-point time-series extraction, requests are typically fast (seconds to minutes). Bulk gridded downloads may queue longer.
- **Service note**: The PUG states that the time-series entry "may be temporarily disabled or completely deprecated at any point and it does not come with the same level of operational support as the parent entry." If the time-series endpoint becomes unavailable, the standard gridded ERA5-Land endpoint or ERA5 single-levels time-series is the fallback.
- **No redistribution restrictions** beyond attribution — data can be included in a research repository (though large files should be excluded from Git per project policy).

---

## 12. Scientific Limitations

1. **Reanalysis ≠ observation**: ERA5-Land grid-cell values are model-estimated, not directly measured. They carry systematic biases, especially for precipitation (known to be smoothed/biased in reanalysis) and near-surface variables in complex terrain.

2. **Spatial representation**: A ~11 km grid cell represents an area average, not a point. Urban heat-island effects in Delhi may not be fully captured at this resolution.

3. **Precipitation quality**: ERA5-Land uses ERA5 atmospheric forcing for precipitation. Convective precipitation (important during Delhi monsoon) is parametrized, not explicitly resolved. Precipitation totals may differ systematically from gauge observations.

4. **No direct observational anchor on land surface**: ERA5-Land assimilates observations indirectly through ERA5 atmospheric forcing. It does not directly assimilate land-surface observations.

5. **Temporal homogeneity**: The observing system feeding ERA5 has changed over decades. More satellites and observations in recent years mean ERA5 quality generally improves forward in time. The recommended 2018–2024 period mitigates this by using only the modern, data-rich era.

6. **Not a substitute for operational station data**: Results from ERA5-Land-trained models should not be presented as equivalent to models trained on direct observational data. The distinction must be documented throughout.

7. **Completeness not guaranteed a priori**: While reanalysis methodology produces highly continuous records, the actual downloaded data must be inspected for any missing timestamps, NaN values, or anomalies during Task 1.3. Claims about data completeness must be based on inspection, not assumption.

---

## 13. Data Volume and Compute Considerations

| Metric | Estimate |
|---|---|
| Single location, 6 variables, 7 years hourly | ~15–25 MB (CSV) |
| 6–7 pretraining locations, same variables, same period | ~100–175 MB total |
| Storage for processed/split data | ~50–100 MB |
| Model checkpoints (4 models × ~1–5 MB each) | ~20 MB |
| Total disk footprint | < 500 MB |

This is feasible on any laptop. Training lightweight LSTM/Transformer models on ~40,000–50,000 hourly sequences is computationally manageable on CPU (minutes) or Colab GPU (seconds).

---

## 14. Rationale Summary

| Decision | Choice | Primary reason |
|---|---|---|
| Dataset | ERA5-Land time-series | Highly complete hourly, 11 km, all variables available, free CC-BY license, reproducible API access, de-accumulated precipitation |
| Source type | Reanalysis grid cell | Not a weather station — documented accurately |
| Target location | Delhi | Large seasonal amplitude, monsoon dynamics, documented extremes, flat terrain, pretraining-friendly geography |
| Sampling interval | Hourly | Native resolution; supports fine-grained sequence modeling |
| Historical period | 2018–2024 | 7 years, modern data quality, includes recent extremes |
| Core variables | `2m_temperature`, `2m_dewpoint_temperature`, `surface_pressure`, `total_precipitation`, `10m_u_component_of_wind`, `10m_v_component_of_wind` | Covers temperature, humidity proxy, pressure, precipitation, wind — the essential atmospheric state |
| Target schema | UNRESOLVED (§9.13) | All six are dynamic climate variables; target/feature mapping not yet decided |

---

## 15. Remaining Uncertainties and Unresolved Decisions

1. **CDS time-series endpoint reliability**: The PUG notes the time-series entry may be temporarily disabled. If access fails during Task 1.2, the fallback is the standard gridded ERA5-Land dataset or the ERA5 single-levels time-series endpoint.

2. **Exact CDS API request syntax**: The variable names listed above are from the PUG parameter tables. Exact request syntax should be verified against the CDS download form during Task 1.2.

3. **Exact returned grid coordinates**: The nearest 0.1° grid point to (28.61°N, 77.21°E) will be identified from the downloaded data and recorded as authoritative location metadata.

4. **§9.13 — Target schema**: Whether all six variables are targets (`n_targets == n_features`) or some are input-only. This affects autoregressive rollout design. UNRESOLVED.

5. **Relative humidity derivation**: Whether to include RH as a derived feature, keep raw Td only, or use both. UNRESOLVED — deferred to Phase 2 preprocessing.

6. **Wind representation**: Whether to use (u, v) components, derived scalar wind speed, or both. UNRESOLVED — deferred to Phase 2 preprocessing.

7. **Solar radiation inclusion**: Not included in initial download. May be reconsidered if analysis warrants it.

8. **Split date boundaries**: NOT resolved here. To be determined during Phase 2 based on actual data inspection.

9. **Input window length**: NOT resolved here. Depends on experimental design decisions.

10. **Pretraining city selection**: NOT resolved here. Feasibility confirmed; exact cities to be selected in Phase 8.

11. **Normalization policy**: NOT resolved here (§9.14).

12. **Precipitation value validation**: Although the PUG documents that `total_precipitation` is de-accumulated to hourly by the time-series product, the returned values and timestamps must be validated during raw-data inspection to confirm this.

---

## Changelog (v1.0 → v2.0)

| # | Change | Reason |
|---|---|---|
| 1 | Corrected precipitation handling: stated that time-series product provides de-accumulated hourly precipitation per PUG; removed instruction to manually de-accumulate | PUG confirms accumulation variables are de-accumulated during ARCO time-series generation |
| 2 | Updated CDS variable identifiers to official PUG names: `2m_temperature`, `2m_dewpoint_temperature`, `surface_pressure`, `total_precipitation`, `10m_u_component_of_wind`, `10m_v_component_of_wind` | Verified against ERA5-Land time-series PUG parameter tables |
| 3 | Replaced "gap-free" with "highly complete and consistent"; added requirement to validate actual downloaded data | Reanalysis is highly continuous but completeness claims must be based on inspection |
| 4 | Corrected IMD section: removed claim that portal is locked; noted portal is operational with active delivery statistics; rejection based on acquisition process, redistribution restrictions, and reproducibility constraints | IMD DSP confirmed operational at https://dsp.imdpune.gov.in/ with August 2026 delivery statistics |
| 5 | Marked target/feature mapping as UNRESOLVED (§9.13); documented autoregressive consequence for input-only variables | All six variables are dynamic climate variables; cannot pre-assign target roles without formal decision |
| 6 | Removed solar radiation from initial variable set; documented as available but not justified for initial download | No documented scientific necessity for initial prototype |
| 7 | Marked RH and wind speed as optional Phase 2 derivation decisions; added note against duplicating raw+derived without rationale | Avoids premature feature-engineering decisions |
| 8 | Added "actual returned grid coordinates to be recorded after acquisition" throughout | Exact grid point depends on nearest-neighbor selection by CDS |
| 9 | Added PUG URL, PUG group/notes column to variable table, and service-stability note from PUG | Improves traceability to official documentation |
| 10 | Preserved all unresolved decisions (splits, input window, normalization, pretraining geography) as explicitly unresolved | Per user instruction |
