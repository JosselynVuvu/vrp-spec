## Week 3 — Step 1: Baseline policies + blockage simulation (no planning overhead)

## Goal
Implement and benchmark three baseline replanning strategies under **(i)** rain (observable), and **(ii)** a one-bin road blockage (unobservable until encountered), using the Week 2 cost tensors.

Baseline policies:
- **B0_PlanOnce**: plan once at the start, then execute the plan (no replans).
- **B2_BlockageReplan**: plan once at start; replan once at the first decision point in/after the blockage time bin.
- **B1_AlwaysReplan**: replan at every customer arrival.

## Inputs
- Episode tensors (per seed): `TT_data_min (B×N×N)`, `dist_km (N×N)`, `node_ids`
- Week 2 outputs:
  - rain generation (deterministic by seed)
  - blocked arcs (K arcs chosen early in the route)
  - planning costs: `J_cost_int[b,i,j]` (includes BIG_M on blocked arcs at `blockage_bin`)
- Config:
  - `bin_minutes`, `service_time_min` from `configs/ingest.json`
  - `λ` from `configs/lambda.json`

## Blockage execution model (SPEC-critical)
During the **blockage bin** only:
- If the vehicle attempts a blocked arc `(u→v)`:
  - it **waits until the end of the current bin** (engine off)
  - then traverses `(u→v)` in the **next bin**, so TT/CO2 are taken from that next bin.

This creates a **traffic delay** in execution even if the planner did not avoid the blocked arc.

## Simulation loop (high level)
State:
- `cur` (current node), `remaining` customers
- time accounting: `elapsed_min`, `travel_min`, `traffic_wait_min`, `service_min`
- CO2 accounting: `CO2_total`

At each step:
1. Compute current time bin: `b = start_bin + floor(elapsed_min / bin_minutes)`
2. Decide whether to replan (depends on policy).
3. If replanning, solve a path subproblem from `cur` through remaining customers back to depot using OR-Tools.
4. Pop next node from the planned queue.
5. If the next arc is blocked in the blockage bin, apply the **bin-end wait**.
6. Traverse the arc using `TT_hat_min[b]` and `CO2_hat[b]`.
7. Add service time at customer nodes.

## Output metrics (Step 1)
- `travel_min`, `traffic_wait_min`, `service_min`, `elapsed_min`
- `CO2_total`
- objective (execution-only):
  - `total_route_time_min = travel + traffic_wait + service`
  - `J_exec = CO2_total + λ * total_route_time_min`
- indicators:
  - `used_blocked_arc` (count)
  - `delay_flag` = 1 if `traffic_wait_min > 0`
- replanning stats:
  - `n_replans`
  - OR-Tools solve stats (p50/p95/max)

## Commands used (TEST split)
Run with and without rain, for the time budgets in the spec:

```bash
# rain
py scripts/run_baselines.py --split TEST --time_limit_ms 200 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60
py scripts/run_baselines.py --split TEST --time_limit_ms 500 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60
py scripts/run_baselines.py --split TEST --time_limit_ms 800 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60

# norain ablation
py scripts/run_baselines.py --split TEST --time_limit_ms 200 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60 --disable_rain
py scripts/run_baselines.py --split TEST --time_limit_ms 500 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60 --disable_rain
py scripts/run_baselines.py --split TEST --time_limit_ms 800 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60 --disable_rain
```

Outputs are written to:
`data/processed/bench/week3_results/baselines_{SPLIT}_{rain|norain}_cap{CAP}_startbin{...}_blockbin{...}_k{K}_ef{EF}.csv`
