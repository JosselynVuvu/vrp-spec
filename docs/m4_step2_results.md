# Milestone 4 — Step 2: Results + plots + interpretation (main results)

## Files used (TEST)
From `data/processed/bench/digital_twin_eval_results/`:

Rain:
- `twin_TEST_rain_cap200_startbin0_blockbin1_k3_ef0.60.csv`
- `twin_TEST_rain_cap500_startbin0_blockbin1_k3_ef0.60.csv`
- `twin_TEST_rain_cap800_startbin0_blockbin1_k3_ef0.60.csv`

No-rain:
- `twin_TEST_norain_cap200_startbin0_blockbin1_k3_ef0.60.csv`
- `twin_TEST_norain_cap500_startbin0_blockbin1_k3_ef0.60.csv`
- `twin_TEST_norain_cap800_startbin0_blockbin1_k3_ef0.60.csv`

Policies compared (main):
- `B0_PlanOnce`
- `B2_BlockageReplan`
- `B1_AlwaysReplan`

(We do **not** include gated replanning in the main results. Gate B is reported in Appendix A only.)

---

## Figures (saved in `data/processed/bench/digital_twin_eval_results/`)

### Figure 1 — Wall-clock objective vs time limit
![](../data/processed/bench/digital_twin_eval_results/twin_grid_TEST_J_wall.png)

**Caption.** Mean wall-clock objective \(J_{wall} = CO2 + \lambda\cdot wall\_time\) across TEST seeds versus OR-Tools time limit, shown separately for rain and no-rain.

**Interpretation.** Replanning reduces \(J_{wall}\) relative to PlanOnce primarily by avoiding blockage-induced waiting (`traffic_wait_min`), while adding only small solver overhead (`planning_wait_min`). `B2_BlockageReplan` and `B1_AlwaysReplan` achieve very similar \(J_{wall}\), indicating that reacting around the blockage event captures most of the achievable benefit.

---

### Figure 2 — Wall time vs time limit
![](../data/processed/bench/digital_twin_eval_results/twin_grid_TEST_wall_time_min.png)

**Caption.** Mean total wall-clock route time (includes travel + service + blockage waiting + planning latency) versus OR-Tools time limit, for rain and no-rain.

**Interpretation.** The main wall-time reduction comes from eliminating **traffic waiting** due to blocked arcs. Differences across caps are minor because route quality changes little, and solver overhead remains small compared to travel + service time.

---

### Figure 3 — CO₂ total vs time limit
![](../data/processed/bench/digital_twin_eval_results/twin_grid_TEST_CO2_total.png)

**Caption.** Mean total CO₂ emissions across TEST seeds versus OR-Tools time limit, for rain and no-rain.

**Interpretation.** CO₂ changes only slightly with policy/time cap in these runs. Most \(J_{wall}\) improvement comes from reduced completion time (less waiting) rather than large CO₂ reductions, though replanning can slightly reduce CO₂ by avoiding more expensive arcs.

---

### Figure 4 — Planning wait (minutes) vs time limit
![](../data/processed/bench/digital_twin_eval_results/twin_grid_TEST_planning_wait_min.png)

**Caption.** Mean planning wait time in minutes (solver time converted into elapsed wall-clock time) versus OR-Tools time limit, for rain and no-rain.

**Interpretation.** Planning overhead increases with both (i) a higher per-solve time cap and (ii) replanning frequency. `B1_AlwaysReplan` has the largest planning wait because it replans at every arrival, while `B2_BlockageReplan` stays close to `B0_PlanOnce` because it triggers only one extra solve.

---

### Figure 5 — Relative prediction error (mean) vs time limit
![](../data/processed/bench/digital_twin_eval_results/twin_grid_TEST_rel_pred_err_mean.png)

**Caption.** Mean relative per-leg travel-time prediction error across executed legs versus time limit, for rain and no-rain.

**Interpretation.** Prediction error is largely insensitive to the OR-Tools time cap because it is driven by the twin-vs-truth mismatch rather than solver effort. Similar values across policies indicate that routing choice changes which arcs are executed, but not the underlying prediction quality.

---

### Figure 6 — Relative prediction error (p95) vs time limit
![](../data/processed/bench/digital_twin_eval_results/twin_grid_TEST_rel_pred_err_p95.png)

**Caption.** 95th percentile of relative per-leg travel-time prediction error versus time limit, for rain and no-rain.

**Interpretation.** The error tail remains stable across caps and policies, suggesting residual error is systematic rather than something improved by giving OR-Tools more time.

---

### Figure 7 — Total solver time vs time limit
![](../data/processed/bench/digital_twin_eval_results/twin_grid_TEST_solve_ms_total.png)

**Caption.** Mean total solver time (milliseconds summed across all replans) versus time limit, for rain and no-rain.

**Interpretation.** Total solver time grows strongly with time cap and replanning frequency. `B1_AlwaysReplan` scales roughly with `(number of replans × cap)`, while `B0_PlanOnce` and `B2_BlockageReplan` remain far smaller because they solve only 1× and ~2× per episode.

---

### Figure 8 — Solver time p95 vs time limit
![](../data/processed/bench/digital_twin_eval_results/twin_grid_TEST_solve_ms_p95.png)

**Caption.** Mean (across seeds) of the per-episode p95 single-solve runtime versus time limit, for rain and no-rain.

**Interpretation.** The p95 single-solve runtime closely tracks the time cap, indicating OR-Tools often uses most of the allowed budget. This makes the cap a direct knob controlling replanning latency.

---

### Figure 9 — Solver time max vs time limit
![](../data/processed/bench/digital_twin_eval_results/twin_grid_TEST_solve_ms_max.png)

**Caption.** Mean (across seeds) of the maximum single-solve runtime per episode versus time limit, for rain and no-rain.

**Interpretation.** The maximum single-solve runtime also tracks the cap, reinforcing that solves typically terminate due to the time limit rather than early convergence.

---

## Overall takeaway (main)
- **B0_PlanOnce** can suffer blockage waiting when a blocked arc is encountered.
- **B2_BlockageReplan** captures most of the benefit (avoids blockage delay) with minimal extra planning overhead.
- **B1_AlwaysReplan** achieves similar objectives but is much more expensive in planning time.
- Increasing the OR-Tools cap mainly increases planning latency; quality gains are small in these runs.

## Key numbers from logs (sanity check)
- **traffic_wait_min:** `B0_PlanOnce` > 0 in ~50% seeds (`delay_flag` rate ≈ 0.50); `B1/B2` ≈ 0.
- **planning_wait_min:** increases with replanning frequency; (cap=500ms sample) `B0` ≈ 0.008 min, `B2` ≈ 0.017 min, `B1` ≈ 0.167 min.

---

# Appendix A — Gate B (optional ablation; not part of main results)

Gate B (`B3_GateReplan`) runs a cheap probe solve to estimate predicted gain, and only performs a full replan if the predicted gain exceeds an estimated planning-cost threshold. This is reported as an ablation only (main submission uses B0/B1/B2).

## Appendix figures (curated)
(Keep the appendix readable: show only the core story.)

### Appendix Figure A1 — Objective (Gate B vs AlwaysReplan)
![](../data/processed/bench/digital_twin_eval_results/twin_gate_grid_TEST_J_wall.png)

### Appendix Figure A2 — Planning overhead reduction
![](../data/processed/bench/digital_twin_eval_results/twin_gate_grid_TEST_planning_wait_min.png)

### Appendix Figure A3 — Total solver time reduction
![](../data/processed/bench/digital_twin_eval_results/twin_gate_grid_TEST_solve_ms_total.png)

Optional single diagnostic (pick one):
- ![](../data/processed/bench/digital_twin_eval_results/twin_gate_grid_TEST_n_gate_full_replans.png)

**Appendix interpretation.** Gate B targets B1-level quality while reducing planning cost by skipping full replans when predicted gain is small. In the sample shown, Gate B matches `J_wall` closely while reducing `planning_wait_min` and `solve_ms_total` relative to `B1_AlwaysReplan`.

---

## Repro commands (bash)

From repo root:

### 1) Generate TEST CSVs (main)
```bash
# rain
python scripts/run_baselines.py --split TEST --time_limit_ms 200 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60
python scripts/run_baselines.py --split TEST --time_limit_ms 500 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60
python scripts/run_baselines.py --split TEST --time_limit_ms 800 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60

# no-rain
python scripts/run_baselines.py --split TEST --time_limit_ms 200 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60 --disable_rain
python scripts/run_baselines.py --split TEST --time_limit_ms 500 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60 --disable_rain
python scripts/run_baselines.py --split TEST --time_limit_ms 800 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60 --disable_rain

### 2) Generate TEST plots 
```bash

python scripts/plot_twin_eval_grid.py --split TEST 

