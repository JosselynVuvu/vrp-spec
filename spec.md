## SPEC.md: Project Specification (Freeze)

**Title:** An Offline Digital Twin for Event-Triggered Time-Dependent Green Vehicle Routing Under Latency Constraints
**Version:** 1.7 (Final)
**Date:** December 22, 2025
**Status:** **Frozen** — agreed scope, constraints, and objectives. After Day 1, only bug fixes, refactoring, and documentation improvements are allowed.

---

# 1. Problem Definition

This project studies an **event-triggered time-dependent green vehicle routing problem** with an **offline scenario-replay digital twin**.

We define a complete directed graph (G=(V,A)), where:

* (V = {0} \cup V_c)
* (|V_c|=20)
* node (0) is the **depot**, and (V_c) are **customers**.

The task is to compute a **single vehicle tour** that:

1. starts at the depot,
2. visits each customer exactly once,
3. returns to the depot,

while minimizing a composite objective of **CO₂** and **route duration** under time-dependent travel conditions and disruptions.

Environment dynamics include:

* **Time dependency:** travel time on arc ((i,j)) depends on departure time bin.
* **Events:** congestion (data-driven via BonnTour), rain (stochastic), and one accident blockage period (binary closure).

---

# 2. Operating Window and Time Discretization (BonnTour-aligned)

* **Operating window (dataset-aligned):** **15:00–22:00** (7 hours).
* **Time origin:** (t=0) minutes corresponds to **15:00**.
* **Time bins:** **7 non-overlapping bins** of **60 minutes** each:
  [
  b \in {0,\dots,6}.
  ]
* **Bin-to-clock mapping (for logging only):**

  * bin 0 = 15:00–16:00
  * bin 1 = 16:00–17:00
  * ...
  * bin 6 = 21:00–22:00

## 2.1 Travel-time semantics (frozen)

* **Departure-bin model:** A leg’s travel time depends **only** on the bin of its **departure time**. No mid-arc bin switching is modeled.
* **FIFO compliance:** travel time functions are assumed **FIFO** compliant.

## 2.2 Travel-time symbols (frozen notation)

For any arc ((i,j)) and bin (b):

* (TT^{data}_{ij}(b)): baseline BonnTour travel time (data-driven congestion)
* (\widehat{TT}_{ij}(b)): predicted travel time used by the planner (twin output)
* (TT^{true}_{ij}(b)): simulator “ground truth” travel time used for execution metrics (**data + rain**)
* (TT^{obs}_{ij}): realized travel time observed on an executed leg (equal to the realized (TT^{true}) for that leg’s departure bin)

**Blockage note (frozen):** blockage is modeled as **forced waiting** when attempting blocked arcs during the blockage bin (Section 5.3). It is **not** represented by setting (TT^{true}) to a big-M value.

---

# 3. System Constraints

## 3.1 Compute / Runtime Measurement

* **All runtime performance metrics** (especially replanning latency) are measured on **local CPU only**:

  * AMD Ryzen Pro 5, 16 GB RAM.
* **Free Google Colab GPU** may be used **only** for offline training of the forecasting model (if used). No SLO numbers are reported from Colab.

## 3.2 Routing Constraints

* Single vehicle; **tour must start and end at depot**.
* **Each customer visited exactly once**.
* **No customer time windows** (only a soft end-of-shift constraint is modeled).
* **Service time:** fixed **2 minutes per customer**; service time advances time and is not ignored.
* Waiting is allowed.
* **Engine-off assumption:** while waiting or serving, the engine is **OFF**, so **CO₂ = 0** during stops/service; emissions occur only during travel.

## 3.3 Capacity

* Vehicle capacity is **not modeled** (out-of-scope).

---

# 4. Dataset and Episode Generation (BonnTour / VRPTDT)

* **Primary dataset:** **BonnTour / VRPTDT** benchmark dataset (time-dependent travel times).
* Each episode is created by **randomly sampling 20 customers** (plus depot) using a fixed RNG seed. Sampled node IDs and seed are logged.

## 4.1 Time-dependent travel times from dataset

* BonnTour provides time-dependent travel times over **15:00–22:00**, modeled here as **7 hourly bins** (b=0..6).
* If per-bin travel-time matrices exist, they are used directly.

## 4.2 Distances (required for emissions)

* (dist_{ij}) is the **fastest-path distance** (km) provided by the dataset (or the dataset’s documented distance field).
* If an instance lacks distance information for the sampled node set, that episode is excluded from emissions experiments (but may still be used for time-only ablations).

---

# 5. Dynamic Events (Triggers)

Each episode includes:

## 5.1 Congestion (data-driven)

* Congestion is represented via dataset time-dependent travel times (TT^{data}_{ij}(b)).

## 5.2 Rain (stochastic, bin-aligned; observable; frozen)

* Rain events are **bin-aligned**.
* Rain duration: **1–3 bins**.
* Rain intensity is sampled per episode from:
  [
  \rho_{TT}\in{0.05,0.10,0.20}.
  ]
* Simulator “truth” travel times include rain:
  [
  TT^{true}*{ij}(b) = TT^{data}*{ij}(b)\cdot (1+\rho_{TT}(b)).
  ]
* **Observability (frozen):** rain status and (\rho_{TT}) are available to the planner and applied consistently to both truth (TT^{true}) and predicted (\widehat{TT}) (Section 8.2).

### Rain schedule (frozen)

Rain start bin, duration, and intensity are **random but seeded** per episode and logged:

* Start bin sampled uniformly over feasible bins such that the event fits in the 7-bin horizon.
* Duration sampled uniformly from ({1,2,3}) bins.
* Intensity sampled uniformly from ({0.05,0.10,0.20}).

## 5.3 Total blockage (accident; exactly one per episode; k=3 blocked arcs; seeded; frozen)

* **Blockage bin (frozen):** a single fixed bin index (b_b) for all episodes (comparability).
  **Frozen project setting:** (b_b = 1) (matches the current processed runs/logs).
* The blockage represents an **accident-induced temporary closure** during bin (b_b).
* **Blocked arcs:** during (b_b), a fixed set of **(k=3)** directed OD arcs is blocked:
  [
  \mathcal{B}={(u_1!\to!v_1),(u_2!\to!v_2),(u_3!\to!v_3)}.
  ]

### Arc selection (Option B; seeded; guaranteed to matter; frozen)

1. compute an **initial planned route** at episode start (seeded),
2. select the first **(k=3)** distinct directed arcs along that route as (\mathcal{B}),
3. keep (\mathcal{B}) fixed for the episode.

**Implementation note (frozen):** (\mathcal{B}) is generated **on the fly** but deterministically from the episode seed and logged (no leakage across splits).

### Realized execution dynamics (accident-wait model; frozen)

If the vehicle attempts to traverse a blocked arc ((u\to v)\in\mathcal{B}) and the leg’s **departure time** lies in bin (b_b), the vehicle cannot traverse the arc and must **wait until the end of the current bin** (accident clears at the bin boundary).

Let (t) be elapsed minutes since (t=0) (15:00). Waiting time is:
[
w = 60 - (t \bmod 60)\ \text{minutes}.
]

* **Engine-off:** during waiting, **CO₂ = 0**.
* After waiting, execution **continues with the current plan** (no mandatory immediate replan).
* **Cost lookup / clipping (frozen):** if elapsed time exceeds the horizon and the bin index would exceed 6, lookups use the **last available bin** (b=6) (clip-to-horizon).

### Planning big-M (frozen)

Blocked arcs are discouraged in the solver by setting their **planning arc cost** to a fixed large integer (BIG_M_{Cost}) during bin (b_b) (Section 10.2).
**Important:** big-M is applied to **planning costs**, not to (TT^{true}).

---

# 6. Emissions Model (MEET-style, speed-dependent; tailpipe proxy)

**Vehicle assumption (frozen):** diesel delivery van (LCV class). Coefficients (\alpha,\beta,\gamma,\delta) are treated as fixed scenario parameters.

Let (dist_{ij}) be distance in **km**. Travel time is represented in **minutes** internally.

For leg ((i,j)) departing in bin (b), define speeds:

* **Predicted speed (used for planning):**
  [
  \hat v_{ij}(b)=\frac{dist_{ij}}{\widehat{TT}_{ij}(b)/60}.
  ]

* **Realized speed (used for evaluation/metrics):**
  [
  v^{true}*{ij}(b)=\frac{dist*{ij}}{TT^{true}_{ij}(b)/60}.
  ]

Emissions are modeled as a MEET-style speed function:
[
e(v)=\alpha v^2+\beta v+\gamma + \frac{\delta}{v}.
]

Leg-level CO₂ proxy:

* **Predicted (planner cost):**
  [
  CO2^{pred}*{ij}(b)=dist*{ij}\cdot e!\left(\hat v_{ij}(b)\right).
  ]

* **Realized (evaluation/metrics):**
  [
  CO2^{true}*{ij}(b)=dist*{ij}\cdot e!\left(v^{true}_{ij}(b)\right).
  ]

**Engine-off rule (frozen):** emissions are **zero** while waiting/servicing, including accident waiting from Section 5.3.

---

# 7. Objective Function

The **reported** objective on realized execution is:
[
J = \sum_{(i,j)\in S} CO2^{true}*{ij}(b*{dep,i}) ;+; \lambda \cdot \text{TotalRouteTime} ;+; Penalty_{shift},
]
where:

* (S) is the set of traveled arcs,
* (b_{dep,i}) is the departure bin from node (i),
* (\text{TotalRouteTime}) includes **travel + waiting + service** (minutes).

The planner minimizes the analogous **predicted** objective at each replan (replace (CO2^{true}) by (CO2^{pred}) and use predicted route time). OR-Tools minimizes an integer-scaled proxy of this predicted objective (Section 10).

## 7.1 Weight normalization (train only; frozen thereafter)

[
\lambda = \frac{\text{median}(CO2_{leg})}{\text{median}(TT_{leg})},
]
computed on **TRAIN only**, then frozen for VAL/TEST.

## 7.2 End-of-shift handling (soft)

22:00 is a **soft** end-of-shift. With time origin at 15:00:
[
T_{end}=420\text{ minutes}.
]
[
Penalty_{shift}=\beta_{shift}\cdot overtime_minutes,\quad \beta_{shift}=\lambda.
]
Overtime minutes are also reported as a metric.

---

# 8. Digital Twin Definition (Forecasting Task A)

The digital twin is an **offline scenario-replay twin** that estimates traffic state and provides predicted travel costs for planning.

## 8.1 State and forecast target

* **State:** scalar hourly multiplier (\hat m(b)) per bin (b).
* **Forecasting task (A):** predict (\hat m(b+1)) (scalar).

**Global twin:** (\hat m(b)) is **global** (shared across all OD pairs).

## 8.2 Predicted travel times used by the planner (frozen)

The planner uses predicted travel times:
[
\widehat{TT}*{ij}(b)=\hat m(b)\cdot TT^{data}*{ij}(b)\cdot (1+\rho_{TT}(b)),
]
where (\rho_{TT}(b)=0) when rain is inactive and (\rho_{TT}(b)\in{0.05,0.10,0.20}) when rain is active.

**Blockage handling in predictions (frozen):**

* The planner does **not** modify (\widehat{TT}) to represent the accident.
* Blockage is represented **only** by patching the **planning arc costs** for arcs in (\mathcal{B}) to (BIG_M_{Cost}) during (b_b) (Section 10.2).

## 8.3 Baseline predictors (must-have)

* **Bin-mean predictor:** (\hat m(b)) equals the historical mean multiplier for bin (b), estimated from **TRAIN only**.
* **Persistence:** (\hat m(b)) equals the last available estimate.
* **EWMA (primary twin baseline):** (\hat m(b)) updated online from observations.

### EWMA update rule (frozen)

For each executed leg ((i,j)) departing in bin (b), define the observed multiplier:
[
m_{obs}=\frac{TT^{obs}*{ij}}{TT^{data}*{ij}(b)\cdot (1+\rho_{TT}(b))}.
]
Update:
[
\hat m(b)\leftarrow \alpha, m_{obs} + (1-\alpha),\hat m(b).
]

**Frozen parameter:** (\alpha = 0.2).

## 8.4 DL predictor (gated add-on)

* Small GRU/TCN predicting (\hat m(b+1)).
* Trained on Colab GPU; inference on local CPU.
* Included only if it improves VAL outcomes (forecast RMSE and/or routing objective) without worsening the latency SLO; otherwise reported and excluded from runtime-critical comparisons.

---

# 9. Replanning Policy (Event-triggered; primary policy frozen)

## 9.1 Primary replanning policy (used for main experiments; frozen)

* **Always replan at every customer arrival**, and at episode start (depot).
* Execution continues after any accident-wait delay; replanning remains arrival-triggered.

## 9.2 Optional gated replanning (ablation only; frozen thresholds)

A secondary ablation may use gated replanning at arrivals:

* last-leg relative prediction error (e>\tau), where **(\tau=0.2)** (frozen), or
* multiplier shift (|\hat m(b)-\hat m(b-1)|>\delta), where **(\delta=0.1)** (frozen).

Relative prediction error definition (frozen):
[
e=\frac{|TT^{obs}*{ij}-\widehat{TT}*{ij}(b)|}{\widehat{TT}_{ij}(b)}.
]

**Oracle usage rule (frozen):** the Oracle baseline (if included) uses **full (TT^{true})** (including rain) **only at replanning times** to build the planning matrix; execution still follows (TT^{true}) and the accident-wait rule.

No DRL route selection is allowed.

---

# 10. OR-Tools Planning and Time Limits (Receding-Horizon)

* OR-Tools computes the route over remaining customers with depot return.
* **Primary solver time cap:** **500 ms** per replanning call.
* **Ablation caps:** 200 ms and 800 ms (secondary experiments).
* Caching: base matrices cached; only patch blocked arcs during the blocked bin.

## 10.1 Receding-horizon approximation (frozen)

At each `replan()` call in current bin (b_{now}), build a **static** cost matrix using values evaluated at (b_{now}). OR-Tools solves a static TSP over remaining nodes with that matrix; time dependence enters through repeated replanning over time.

**Bin lookup clipping (frozen):** if (b_{now}>6), use (b_{now}=6).

## 10.2 Combined “green” arc cost minimized by OR-Tools (frozen)

OR-Tools minimizes a single scalar arc cost combining **predicted CO₂** and **predicted travel time**:
[
C^{arc}*{ij}(b*{now})= CO2^{pred}*{ij}(b*{now}) ;+; \lambda \cdot \widehat{TT}*{ij}(b*{now}).
]

### Integer scaling for OR-Tools (frozen)

OR-Tools requires integer arc costs. Use a fixed global scale:
[
C^{arc,int}*{ij} = \left\lfloor SCALE \cdot C^{arc}*{ij}(b_{now}) \right\rceil,
]
with `SCALE` fixed globally (e.g., **1000**).

### Blockage in the planner matrix (frozen; planning-cost patch; k=3 arcs)

Let (BIG_M_{Cost}) be a fixed large integer (e.g., (10^{12})). During the blocked bin (b_b), for each ((u\to v)\in\mathcal{B}), set:
[
C^{arc,int}*{uv} = BIG_M*{Cost}.
]

## 10.3 Soft end-of-shift in solver (preferred; frozen)

* Add a Time dimension using (\widehat{TT}*{ij}(b*{now})) + service times.
* Apply a soft upper bound at (T_{end}=420) minutes with per-minute penalty:
  [
  penalty^{int}*{shift} = \left\lfloor SCALE \cdot \beta*{shift}\right\rceil,\quad \beta_{shift}=\lambda.
  ]

---

# 11. Service Level Objectives (SLOs)

**Primary SLO:** p95 end-to-end replanning latency (\le) **800 ms**, measured on local CPU.

A “latency sample” is one full `replan()` call:

1. twin predict + cost build
2. OR-Tools solve (time-limited)
3. decode + minimal overhead

If p95 > 800 ms, scope is cut rather than adding system complexity.

---

# 12. Key Metrics

## 12.1 Optimization metrics

* Total CO₂ proxy (consistent units)
* Total route time (minutes): travel + waiting + service
* Waiting time (wait_min) (minutes) due to accident closure (engine-off)
* Objective (J)
* Overtime minutes (if any)
* Number of replans per episode

## 12.2 System metrics

* p50/p95 replanning latency (ms), local CPU
* Forecast accuracy (RMSE) for predictors (EWMA vs DL if included)

---

# 13. Baselines and Minimum Evaluation Targets

## 13.1 Baselines (minimal + optional oracle ceiling)

* **B0:** plan once at start; no replans
* **B1:** replan at every customer arrival (primary policy), using EWMA twin
* **B2:** replan once at first arrival in/after (b_b), using EWMA twin
* **B3 (optional ceiling):** Oracle at replans — identical to B1 in replanning frequency, but uses **full (TT^{true})** (including rain) **only at replanning times** to build the planning matrix

## 13.2 Minimum evaluation targets

* **Quality:** ≥ 30 instances × 3 seeds = **90** test runs
* **Latency:** ≥ **1,000** replanning calls logged total (stable p95)

Optional appendix only if time remains: 40-customer scaling check.

---

# 14. Units and Reproducibility

* Distances: **km**
* Time: **minutes** (speed computed in **km/h** via division by 60)
* All RNG seeds logged (sampling, rain schedule, blocked-arc selection)
* Package versions pinned (Python, OR-Tools, torch if used)

## Seed-set split (no leakage)

Episodes are split by **disjoint RNG seed-sets**:

* **TRAIN seeds:** 0–199
* **VAL seeds:** 200–229
* **TEST seeds:** 230–259

TRAIN is used for fitting (\lambda), bin-mean statistics, and (optional) DL training.
VAL is used only for gating the DL add-on.
TEST is used only for final reporting.

---
