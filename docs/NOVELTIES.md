# ClimateTwin — Novelties

_Last updated: 2026-09-27_

This document summarizes what is genuinely novel in ClimateTwin relative to a
standard "train an LSTM to forecast weather" project, and gives a focused,
plain-language explanation of the scenario feature: how it works, what it is
useful for, and who benefits.

---

## 1. Novelties at a glance

ClimateTwin is a localized climate digital twin for Indian cities (Delhi built
end-to-end). Beyond baseline forecasting, three things make it novel:

| # | Novelty | One-line description |
|---|---------|----------------------|
| 1 | **Uncertainty-aware forecasting (MC-dropout)** | Every forecast and scenario carries a predictive confidence band that widens with the horizon — the twin says *how sure it is*, not just *what it thinks*. |
| 2 | **What-if scenario sensitivity engine** | Controlled, unit-aware perturbations of input variables, rolled forward and compared against a baseline, with **paired MC-dropout** so the difference isolates the perturbation. |
| 3 | **Human-meaningful impact layer** | Raw forecast variables translated into feels-like temperature, humidity, and IMD/NOAA heat & rainfall alert categories — useful to non-specialists. |

Supporting design choices that reinforce the novelty:

- **Config-driven, no hard-coded variables** — perturbable controls are the
  intersection of the model's real feature list and
  `configs/data.yaml::perturbable_features`, so the whole engine generalizes to
  any dataset/city (`get_perturbable_features` in `src/scenario.py`).
- **Physical-unit perturbations** — deltas are applied in real units
  (e.g. `+4 °C`), inverse-transformed / re-scaled correctly around the model's
  standardized space (`apply_perturbation`).
- **Honesty by construction** — a mandatory disclaimer
  (`SCENARIO_DISCLAIMER`, `IMPACT_DISCLAIMER`) is attached to every surfaced
  output; scenarios are framed as model-sensitivity experiments, not causal
  climate simulations.

---

## 2. Novelty 1 — Uncertainty-aware forecasting (MC-dropout)

**What it is.** Instead of returning a single forecast line, the twin can run
the rollout many times with dropout left active at inference (Monte-Carlo
dropout). This produces a *distribution* of trajectories, from which we report
predictive intervals (default 5th–95th percentile).

**Why it matters.** The band **widens with the forecast horizon**, which is
exactly the behavior you want: it visually communicates *when the forecast
stops being trustworthy*. This turns "the model is confidently wrong" into "the
model tells you it's now unsure."

**Where it lives.** `run_scenario(..., mc_samples > 1)` in `src/scenario.py`;
`_enable_mc_dropout` re-enables dropout layers (including `nn.LSTM` inter-layer
dropout) while keeping the rest of the network in eval mode.

---

## 3. Novelty 2 — What-if scenario sensitivity engine

_(Detailed how/use/who below in Section 5.)_

The short version: the user changes one or more input variables by a chosen
amount ("what if it were +4 °C warmer?"), and the twin rolls both the original
and the changed world forward, then shows the difference — with an uncertainty
band on that difference.

The key technical novelty is **paired MC-dropout**: for each Monte-Carlo
sample, the baseline and perturbed rollouts use the **same dropout mask**
(same per-sample seed). This means the per-sample *difference* cancels out the
random mask noise and isolates the effect of the perturbation itself, giving a
clean sensitivity estimate rather than noise-on-noise.

---

## 4. Novelty 3 — Human-meaningful impact layer

**What it is.** A deterministic layer (`src/impact.py`) that converts forecast
variables into indicators a non-specialist can act on:

- **Relative humidity** — Magnus–Tetens formula.
- **Feels-like temperature (Heat Index)** — NOAA/NWS Rothfusz regression.
- **Heat category** — IMD heatwave thresholds (plains): onset ≥ 40 °C,
  heatwave ≥ 45 °C, severe ≥ 47 °C.
- **Rainfall category** — IMD 24-hour classes (light → extremely heavy).
- **Wind category** — Beaufort-style bands.
- **Colour-coded alert cards** (red / orange / yellow), sorted by severity.

**Why it matters.** These use *standard, citable formulas and thresholds*, not
invented coefficients, so the impact layer is defensible. It adds
*interpretation*, not predictive skill — and it makes the twin's output
meaningful to people who don't read meteorological variables.

---

## 5. The scenario feature explained

### 5.1 How does it work?

1. **Start from a real window.** Take a real 24-hour input window for the city
   (in the model's standardized/scaled space).
2. **User defines a what-if.** Pick one or more variables to change and by how
   much, in physical units — e.g. `{"t2m": +4.0}` means "+4 °C". Only
   variables in the perturbable set can be changed (resolved dynamically from
   model features ∩ config, currently `t2m`, `d2m`, `sp`).
3. **Apply the perturbation correctly.** The window is inverse-transformed to
   physical units, the delta is added to the chosen columns only, then it's
   re-scaled back into model space (`apply_perturbation`). Everything else is
   numerically unchanged.
4. **Roll both worlds forward.** The one-step model is advanced
   autoregressively for `horizon` steps for **both** the baseline (unchanged)
   and the perturbed window (`_rollout`). Each prediction feeds back in as the
   next input.
   - **Seed-only (default):** the perturbation only seeds the initial window,
     and the model is free to relax back toward its learned climatology.
   - **Sustained (optional):** the perturbation is re-injected at every step,
     representing a persistently changed world (e.g. "stays +4 °C warmer").
5. **Quantify uncertainty (optional but key).** With `mc_samples > 1`, steps 4
   are repeated under **paired MC-dropout** to produce predictive bands on the
   baseline, the perturbed trajectory, and — most usefully — on their
   **difference**.
6. **Return a structured result.** A `ScenarioResult` carries baseline and
   perturbed trajectories (physical units), their delta, the confidence bands,
   and the mandatory disclaimer.

**Observed behavior (Delhi LSTM, heatwave `t2m +4 °C`, 12 h seed-only):** the
twin carries ~+3.8 °C at hour 1 and decays toward 0 over ~10 h, with the
uncertainty band widening as the horizon grows — i.e. the effect fades and the
confidence in it drops, which is the intended, honest reading. (A 48-h free
rollout was dominated by autoregressive drift; sustained re-injection overshot
to +10–18 °C. The 12-h seed-only setting reads correctly — this iteration is
documented in `docs/CHECKPOINT.md`.)

### 5.2 What is it useful for?

- **Interpretability / probing** — reveals what variables the model actually
  relies on and how strongly (a form of sensitivity analysis).
- **Persistence / memory** — measures *how long* a disturbance echoes forward
  before the model relaxes back to climatology.
- **Stress-testing** — checks the model's behavior under unusual or extreme
  conditions it may rarely have seen in training.
- **Trust boundary** — the widening uncertainty band tells you *when to stop
  trusting* the scenario.
- **Communication** — combined with the impact layer, a scenario becomes a
  human-readable story ("Normal → Hot (heatwave onset), YELLOW alert"),
  not just a set of numbers.

### 5.3 Who benefits?

- **Researchers / model developers** — as an interpretability and
  model-debugging tool: which inputs matter, how far predictions propagate,
  where the model is fragile.
- **Students / educators** — a hands-on, visual way to explore
  cause-and-effect *within a learned model* and to teach forecast uncertainty.
- **Urban planners & resilience teams** — as an exploratory "what-if" sandbox
  for stress scenarios (heatwaves, heavy-rain bursts), with clear alert-level
  framing. (Exploratory only — see the honesty note.)
- **Non-specialist / public-facing users** — the impact layer turns raw
  variables into feels-like temperature and colour-coded alerts they can act
  on.

### 5.4 Honesty / scope (important)

The scenario feature is a **model-sensitivity / interpretability tool**, **not**
a physically validated climate-intervention simulator. It makes **no causal
claims** about the real world. This is enforced in code via
`SCENARIO_DISCLAIMER`, which is attached to every result, and mirrored by
`IMPACT_DISCLAIMER` on the impact side.

---

## 6. Where each novelty lives (code map)

| Novelty | Module | Key symbols |
|---------|--------|-------------|
| Uncertainty-aware forecasting | `src/scenario.py` | `run_scenario(mc_samples>1)`, `_enable_mc_dropout` |
| Scenario engine + paired MC-dropout | `src/scenario.py` | `get_perturbable_features`, `apply_perturbation`, `_rollout`, `ScenarioResult` |
| Human-meaningful impact layer | `src/impact.py` | `relative_humidity`, `heat_index`, `heat_category`, `rainfall_category`, `wind_category`, `summarize_state` |
| Demo tying it together | `scripts/scenario_demo.py` | heatwave & heavy-rain scenarios with impact panels |
