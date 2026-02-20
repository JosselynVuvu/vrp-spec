## Milestone 3 — Step 2: Add planning overhead + wall-clock objective (J_wall)

## Goal
Make the evaluation closer to reality by accounting for **solver latency** as **vehicle idle time** that:
1) increases wall-clock route completion time, and  
2) can shift time bins, which then changes TT/CO2 for later legs.

This lets us fairly compare “replan more often” vs “plan once” under a wall-clock objective.

## Planning overhead model
For every OR-Tools solve, we observe `solve_ms` and convert it to minutes:

- `planning_wait_min += solve_ms / 60000`
- `elapsed_min += solve_ms / 60000`

Assumptions:
- vehicle is idle while planning (no travel, no service)
- **no extra CO2 during planning** (engine off by default)

We keep traffic delay separate:
- `traffic_wait_min`: only blockage-induced waiting (bin-end wait)
- `planning_wait_min`: only solver time
- `wait_min = traffic_wait_min + planning_wait_min`

## Two time notions: execution vs wall-clock
We report both:
- `exec_time_min = travel_min + traffic_wait_min + service_min`
- `wall_time_min = exec_time_min + planning_wait_min`
- In the code, `elapsed_min` equals `wall_time_min` by construction.

## Objectives committed in the thesis
- **Execution objective** (ignores planning overhead):
  - `J_exec = CO2_total + λ * exec_time_min`
- **Wall-clock objective** (includes planning overhead):
  - `J_wall = CO2_total + λ * wall_time_min`

For the thesis you decided to commit to **J_wall**, because it reflects the real trade-off:
extra replans only help if their improvement exceeds their wall-clock planning cost.

## Flags (recommended meanings)
Keep the original traffic delay meaning clean:
- `delay_flag` = 1 if `traffic_wait_min > 0` (blockage/traffic delay)
Add separate flags if needed:
- `delay_flag_planning` = 1 if `planning_wait_min > 0` (usually always true)
- `delay_flag_any` = 1 if `(traffic_wait_min + planning_wait_min) > 0`

In plots/tables, prefer the **numeric** `planning_wait_min` and `solve_ms_total` rather than a thresholded flag.

## Per-seed deltas vs B0 (for plots)
For each seed, compute:
- `delta_J_wall_vs_B0`, `delta_J_exec_vs_B0`
- `delta_wall_time_vs_B0`, `delta_exec_time_vs_B0`
- `extra_replans_vs_B0`
- efficiency:
  - `J_gain_per_extra_replan`
  - `J_gain_per_extra_plan_sec`

These make “benefit vs compute cost” visible and thesis-friendly.

## Plotting
Grid plots used in Milestone 3:
- `J_wall` vs `time_limit_ms` (rain vs norain)
- `planning_wait_min` vs `time_limit_ms`
- `solve_ms_total` vs `time_limit_ms`

Run example:
```bash
py scripts/plot_policy_eval_grid.py --split TEST --metric J_wall --k 3 --ef 0.60 --caps 200,500,800
py scripts/plot_policy_eval_grid.py --split TEST --metric planning_wait_min --k 3 --ef 0.60 --caps 200,500,800
py scripts/plot_policy_eval_grid.py --split TEST --metric solve_ms_total --k 3 --ef 0.60 --caps 200,500,800
```

## Note on the extra CSV (startbin=1)
If you have a file like:
`baselines_TEST_rain_cap500_startbin1_blockbin1_k3_ef0.60.csv`


