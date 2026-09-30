# ClimateTwin — Explaining the What-If Feature to a Review Panel

_Created: 2026-09-29_

Prep notes for presenting the what-if scenario engine. Every number in here is
measured from the actual trained models, not illustrative.

---

## 0. The framing that matters most

**The what-if engine is what makes this a digital *twin* rather than just a
forecaster.** A forecast gives you one future. A twin lets you interrogate the
system and ask counterfactual questions.

Lead with that. It is also the honest justification for calling the project a
digital twin at all.

---

## 1. The one-liner

> "Our forecaster predicts one future. The what-if engine lets you ask the twin
> a counterfactual question — *what if it were 4 °C hotter right now?* — and see
> how the system responds, with an honest measure of how confident it is."

---

## 2. The 60-second version

Start from a real 24-hour window of observed ERA5-Land weather. Change one
variable by a chosen amount in real physical units — say +4 °C. Then roll the
trained model forward hour by hour **twice**: once from the untouched world,
once from the perturbed world. The difference between those two trajectories is
the twin's answer. Wrap it in an uncertainty band, and translate the output into
things people actually understand — feels-like temperature, IMD heat alerts,
crop stress, health risk.

---

## 3. How it works — four steps

1. **Start from reality.** A genuine 24-hour ERA5-Land window, not a synthetic
   starting point.
2. **Perturb in physical units.** The window is inverse-transformed out of the
   model's normalised space, +4 °C is added to temperature *only*, then it is
   re-scaled back. So "+4 °C" genuinely means four degrees, not four standard
   deviations. Every other variable stays numerically untouched.
   _(`apply_perturbation` in `src/scenario.py`)_
3. **Roll both worlds forward.** The one-step model feeds its own predictions
   back in, advancing 12 hours for both the baseline and the perturbed world.
   _(`_rollout`)_
4. **Interpret.** Apply the NOAA heat index and IMD thresholds so the output
   reads "YELLOW alert, heatwave onset", not just a number.
   _(`src/impact.py`)_

---

## 4. The technical novelty to emphasise — paired MC-dropout

This is the detail that demonstrates real sophistication. Do not skip it.

To measure uncertainty, the rollout runs ~40 times with dropout left active at
inference, producing a *distribution* of futures rather than a single line.

The subtle part: for each of those 40 samples, **the baseline and perturbed runs
use the identical dropout mask** (same per-sample seed). So when you subtract
them, the random mask noise cancels and what remains isolates the
*perturbation's* effect rather than the model's own randomness.

> Without pairing, the "sensitivity" measurement would be noise-on-noise.
> Pairing makes the difference a clean signal.

---

## 5. What the results actually show

The cross-city comparison is the strongest demo moment. Identical +4 °C
perturbation, 12-hour horizon, seed-only mode:

| City | Peak temp response | Feels-like shift |
|---|---|---|
| Delhi | 1.9 °C | 35.7 → 39.8 °C |
| Kolkata | 2.7 °C | 39.9 → **46.1 °C** |
| Chennai | 3.7 °C | 36.6 → 41.8 °C |
| Mumbai | 4.7 °C | 33.9 → 40.6 °C |
| Bengaluru | 6.5 °C | 23.4 → 28.4 °C |

Two points to draw out:

- **Cities respond differently to an identical push.** Delhi damps the
  perturbation hard; Bengaluru amplifies it. Each city's twin learned its own
  dynamics from its own climate regime.
- **Humidity amplifies human impact.** Kolkata's air temperature moves only
  2.7 °C, but its *feels-like* jumps 6.2 °C — crossing into NOAA's "Danger"
  band (≥ 41 °C). This is the entire argument for having an impact layer: the
  raw number understates the human consequence.

---

## 6. The honesty framing — do not skip

State plainly:

> This is a **model-sensitivity and interpretability tool, not a physically
> validated climate-intervention simulator.** It makes no causal claims about
> the real world.

And note that this is **enforced in code**: `SCENARIO_DISCLAIMER` is attached to
every result object, so a scenario cannot be surfaced without its caveat.

Panels reward this kind of scoping. Overclaiming is what gets torn apart.

---

## 7. Likely questions and how to answer them

**"Why does the effect fade over time?"**
The perturbation only seeds the initial window, so the model relaxes back toward
its learned climatology. That is the expected, honest behaviour. There is also a
"sustained" mode that re-injects the perturbation each step, representing a
persistently changed world.

**"Why a 12-hour horizon and not 48?"**
We tested 48 and it was dominated by autoregressive drift — the model's own
compounding error rather than the perturbation. Sustained re-injection overshot
to +10–18 °C, which is unphysical. Twelve hours is where the signal stays
interpretable.
_This answer is a strength: it shows the method was probed and its limits found._

**"How do I know the model isn't fabricating this?"**
You don't take a single line on faith — that is what the widening uncertainty
band is for. It tells you *when to stop trusting the scenario*. A useful sanity
check: when rainfall is perturbed upward, the twin also cools the temperature —
a physically sensible coupling it learned rather than was told.

**"Is this a climate projection?"**
No. Climate projection means decades and emissions scenarios. This is
hours-ahead sensitivity analysis on a learned model — a different, more modest
claim.

**"Why does Bengaluru respond more than Delhi? Is that real?"**
Be careful here. It is a property of Bengaluru's trained twin, plausibly
consistent with its milder, less variable climate regime. Present it as **a
hypothesis worth investigating**, not a validated physical finding.
Distinguishing a genuine climate-regime signal from a model artefact would
require dedicated validation.

**"What is novel versus just running the model twice?"**
Three things:
1. **Unit-aware perturbation** — controls are physically meaningful.
2. **Paired MC-dropout** — the sensitivity estimate is not noise-dominated.
3. **Config-driven control set** — no variable names are hard-coded
   (`get_perturbable_features`), so the engine generalises to any city or
   dataset.

**"Who would actually use this?"**
See section 10 — this is answered with named decision-makers and the specific
decision each one makes, not a vague list of user types.

---

## 8. Suggested live demo order

> ⚠️ **Use the exact slider settings in section 8.1.** With the default start
> hour, every alert card reads "No trigger / no risk" and the demo looks inert.
> The settings below land on real 2024 extreme-weather hours so the cards fire.

1. Single city, +4 °C → point at the **widening uncertainty band**.
2. Show the impact panel flip from "Normal" to "Hot (heatwave onset), YELLOW".
3. **Decision-support panel** — the Heat Action Plan card going Orange → **Red**.
4. Switch on all five cities → the **ranked sensitivity chart** and the
   operational comparison table.
5. Land on the **feels-like / labour** contrast as the "this is why impact
   translation matters" moment.
6. Close on the **disclaimer** — frame the limits as deliberate scope, not
   oversight.

### 8.1 Verified demo settings (measured from the real test split)

**A. Heatwave → red Heat Action Plan trigger**

| Control | Value |
|---|---|
| City | Delhi |
| Start hour | **3586** (window ends 2024-05-30, the hottest real hour: 45.5 °C) |
| Δ t2m | **+4.0 °C** |
| Horizon | 12 h |

Produces: HAP **Orange → Red alert — extreme heat**, with the action text
"Full HAP activation: emergency cooling centres, public warnings, hospital
surge readiness, halt outdoor labour in peak hours." sWBGT 32.2 → 33.6 °C,
crops −24.0% wheat / −12.8% rice / −29.6% maize, cooling demand +17%.

> The strongest line here: **the baseline is already Orange.** That is real
> observed 2024 Delhi weather, not a hypothetical. The scenario shows what one
> more push does to a city already at its limit.

**B. Flood → IMD red urban-flood risk**

| Control | Value |
|---|---|
| City | Delhi |
| Start hour | **6120** (window ends 2024-09-13, wettest 12 h: ~101 mm) |
| Δ tp | **+5 mm/h** (or higher) |
| Sustain perturbation | **ON** |
| Horizon | 12 h |

Produces: **Extremely heavy rain — severe urban flood risk (red)**, ~281 mm
24 h-equivalent, action "Activate flood response: pumping stations, evacuate
low-lying areas, suspend transport in vulnerable corridors."

**C. Cross-city labour contrast**

Keep +4 °C, enable all five cities. The table shows Delhi, Mumbai, Kolkata and
Chennai as **"already unsafe"** for outdoor heavy work while Bengaluru still has
margin at 25% work/hour — a clean illustration that the same warming lands very
differently depending on the baseline.

---

## 9. Where the code lives (if asked)

| Concept | Location |
|---|---|
| Perturbation + rollout + MC-dropout | `src/scenario.py` |
| Perturbable control resolution | `get_perturbable_features` |
| Impact indicators (IMD / NOAA) | `src/impact.py` |
| Sector indicators (crop / health / energy) | `summarize_sectors` |
| Dashboard scenario page + cross-city comparison | `dashboard/pages_impl/scenario.py` |
| Mandatory disclaimers | `SCENARIO_DISCLAIMER`, `IMPACT_DISCLAIMER`, `SECTOR_DISCLAIMER` |

---

## 10. "Who benefits?" — the decision-support answer

This is the question that most needs a sharp answer, and "researchers and
students" is not it. The scenario engine now maps its output onto **published
operational thresholds that real Indian institutions already use**, so each
output names a decision-maker and a decision.

### The beneficiary table

| Who | Decision the scenario informs | Threshold basis |
|---|---|---|
| **Municipal commissioner / city disaster cell** | Do we activate the Heat Action Plan, and at which alert stage? | Ahmedabad HAP: yellow 41.1–43 °C, orange 43.1–44.9 °C, red ≥ 45 °C |
| **Labour dept. / construction & site supervisor** | Do we mandate work-rest cycles or shift hours earlier? | ACGIH TLV / ISO 7243 work-rest regimens by WBGT |
| **Agricultural extension officer / farmer** | How exposed is this season's yield; how urgent is irrigation? | Zhao et al. 2017 (PNAS): wheat −6.0, rice −3.2, maize −7.4 %/°C |
| **Electricity load dispatcher (SLDC)** | How much cooling-load surge should we plan for? | Cooling Degree Hours, 18 °C base |
| **Municipal drainage / flood control** | Do we pre-deploy pumps and clear drains? | IMD 24 h classes: heavy 64.5, very heavy 115.6, extremely heavy 204.5 mm |
| **Public health officer** | Cooling centres, ORS stocks, hospital surge readiness? | NOAA Heat Index caution bands |

### The line to say out loud

> "The twin doesn't just say it will be hotter. It says: Ahmedabad-standard
> **orange alert**, outdoor heavy work limited to **50% of each hour** under
> ACGIH limits, **−24% indicative wheat yield** at sustained +4 °C, and
> **+81% cooling demand**. Those are four different departments' decisions,
> from one scenario."

### The finding that lands hardest

For Delhi and Kolkata in peak summer, the **baseline already exceeds ACGIH
safe limits for outdoor heavy labour** — before any perturbation is applied.
The scenario doesn't create the risk; it reveals that the safe-work margin is
already gone. That reframes the tool from "predicting a future problem" to
"quantifying a present one."

### Why this is defensible, not hand-waving

Every number traces to an external published source. We perform arithmetic on
those thresholds; we do not invent dose-response coefficients. Say this
explicitly — it is the difference between decision support and decoration.

### Two limitations to volunteer before you are asked

Volunteering these is what makes the rest credible:

1. **Simplified WBGT overestimates heat stress in hot-humid climates**
   (Kong & Huber 2022, *Earth's Future*). India is in that regime, so our
   labour restrictions are **conservative**, not calibrated. The lower-bias
   alternative (ESI) needs solar radiation, which the ERA5-Land time-series
   endpoint does not provide — a documented data constraint, not an oversight.
   _The caveat is attached in code to every assessment object._
2. **Crop coefficients describe sustained seasonal global-mean warming**, not a
   12-hour local perturbation. We evaluate them against the *intended*
   perturbation (the policy question) rather than the model's transient
   response, floor losses at −100%, and flag any perturbation beyond the ~4 °C
   range the coefficients were derived over.

### Anticipated challenge

**"Isn't the crop number misleading if it's a global coefficient?"**
Yes if presented as a local forecast — which is why we label it *indicative
exposure*, evaluate it at the intended sustained perturbation, cap it
physically, and flag extrapolation in the UI. It answers "how exposed is
agriculture to this magnitude of warming?", not "what will this district
harvest?"

---

## 11. Where the decision-support code lives

| Concept | Location |
|---|---|
| sWBGT, labour capacity, crop yield, HAP, flood, cooling | `src/sectors.py` |
| Aggregate assessment | `assess_scenario` |
| Known-bias caveat (always attached) | `SWBGT_BIAS_CAVEAT` |
| Dashboard decision panel | `_render_decision_support` in `dashboard/pages_impl/scenario.py` |
| Tests guarding the quantitative claims | `tests/test_sectors_decision.py` (31 tests) |
