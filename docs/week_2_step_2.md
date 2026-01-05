# Week 2 — Step 2: CO2 proxy + integer planning costs + blockage enforcement

## Goal
Produce integer cost tensors per time bin for OR-Tools and enforce a 60-minute blockage via BIG_M on planning cost.

## Travel time with rain (observable)
For bin b:
- If rain active:
  TT_hat[b,i,j] = TT_data[b,i,j] * (1 + ρ_TT)
Else:
  TT_hat[b,i,j] = TT_data[b,i,j]

## MEET/Jabali-style CO2 proxy (speed-dependent)
Speed:
  v_kmh[b,i,j] = dist_km[i,j] / (TT_hat[b,i,j]/60)

Emissions-per-km:
  e(v) = α v^2 + β v + γ + δ/v

Leg CO2:
  CO2[b,i,j] = dist_km[i,j] * e(v_kmh[b,i,j])

Rain CO2 inflation (small to avoid double counting):
  CO2[b,i,j] *= (1 + ρ_CO2) when rain active

## Integer planning costs
Let SCALE be from `configs/ingest.json`.

- time_cost_int[b,i,j] = round(TT_hat[b,i,j] * SCALE)
- co2_cost_int[b,i,j]  = round(CO2[b,i,j] * SCALE)

Composite planning objective:
  J_cost_int = co2_cost_int + round(λ * TT_hat * SCALE)

## Blockage enforcement (SPEC-critical)
Blockage occurs in `blockage_bin` (from ingest config) on arc (u->v):
  J_cost_int[blockage_bin, u, v] = BIG_M_cost_int

BIG_M is applied to the planning cost J (not only TT), ensuring OR-Tools avoids the blocked arc.

## Verification + timing
- `verify_events_costs.py` asserts shapes, non-negativity, no NaNs, and BIG_M enforcement.
- Cost-building latency (CPU, in-memory):
  p50 = 0.078 ms
  p95 = 0.156 ms
This overhead is negligible vs the p95 800ms replanning SLO.

## Commands used
- `py scripts/build_costs_int.py --seed 230 --save`
- `py scripts/verify_events_costs.py`
- `py scripts/bench_cost_build.py`
