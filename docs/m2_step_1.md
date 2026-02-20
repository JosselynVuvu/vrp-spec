# Week 2 — Step 1: Events + objective normalization (SPEC v1.7)

## Goal
Make dynamic events reproducible and define the composite objective weight λ using TRAIN-only data.

## Inputs
- Episode files: `data/processed/vrptdt/berlin_500/episodes/{SPLIT}/seed_XXX.npz`
  - contains `dist_km (21x21)`, `TT_data_min (7x21x21)`, `node_ids`, `meta_json`
- Seed split: disjoint TRAIN/VAL/TEST (no leakage)

## Event generation (deterministic by seed)
For each seed:
- Build an initial route using NN greedy on `TT_data_min[0]`
- Choose blocked arc (u->v) from that initial route (Option B; guaranteed to matter)
- Sample rain:
  - duration L ∈ {1,2,3} bins
  - start bin uniform
  - travel time inflation: ρ_TT ∈ {0.05, 0.10, 0.20}
  - emissions inflation: ρ_CO2 ∈ {0.02, 0.05, 0.10}

Rain is observable to the planner (exogenous).

## λ fitting (TRAIN only)
We compute:
  λ = median(CO2_leg) / median(TT_leg)
using TRAIN seeds only to avoid leakage and to normalize objective terms.

Result:
- λ = 0.5914923122133638
Stored in: `configs/lambda.json`

## Commands used
- `py scripts/fit_lambda.py`
- `py scripts/apply_events.py --seed 230 --save`
