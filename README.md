# Lightweight Digital Twin for Green Vehicle Routing - Code Repository

## Quick Start Guide

This repository contains the implementation of a lightweight digital twin architecture for intelligent replanning in green vehicle routing under dynamic conditions.

---

## 📁 Repository Structure

```
vrp-spec/
├── data/
│   ├── raw/                    # NOT committed (in .gitignore)
│   └── processed/              # NOT committed (in .gitignore)
├── scripts/                    # Python implementation
├── configs/                    # Configuration files
└── README.md                   # This file
```

**Important:** Raw data and processed results are **NOT** committed to git to avoid file size limits.

---

## 🚀 Implementation Workflow (M1-M4)

### **📌 Important: Two Run Scripts**

The repository uses **two different scripts** for running experiments:

1. **`run_baselines_.py`** (with underscore)
   - Runs **baseline-only** experiments (B0, B1, B2)
   - Uses **base travel times** (TT_data) **without** digital twin
   - For M3 baseline comparisons

2. **`run_baselines.py`** (no underscore)  
   - Runs experiments **with EWMA digital twin** (B0, B1, B2 use predictions)
   - Optional `--include_gate` flag adds **B3_GateReplan** policy
   - For M4 digital twin and Gate B experiments

**Quick reference:**
```bash
# Baselines WITHOUT digital twin
python scripts/run_baselines_.py --split TEST --time_limit_ms 500 --blockage_bin 1 --n_blockages 3 --early_frac 0.60

# Baselines WITH digital twin
python scripts/run_baselines.py --split TEST --time_limit_ms 500 --blockage_bin 1 --n_blockages 3 --early_frac 0.60

# Digital twin + Gate B (adds B3 policy)
python scripts/run_baselines.py --split TEST --time_limit_ms 500 --blockage_bin 1 --n_blockages 3 --early_frac 0.60 --include_gate
```

---

### **M1: Dataset Setup & Canonicalization**

#### **Step 1: Setup + Dataset Acquisition**
```bash
# Verify repo root
git rev-parse --show-toplevel

# Dataset: Berlin_500 benchmark (500 potential delivery locations)
# Location: data/raw/vrptdt/vrptdt-benchmark-main/
```

**Key files:**
- `berlin_500.json` - Base instance with customer coordinates
- `.gitignore` - Excludes `/data/raw/`, `/data/processed/`, `/data/canonical/`

#### **Step 2: Data Ingestion + Canonicalization**
```bash
# Generate distance matrix and travel times
python scripts/build_canonical.py

# Generate episode bank (20 customers per episode)
python scripts/build_episode_bank.py --seeds 0-259 --n_customers 20

# Verify processed data
python scripts/verify_processed.py
```

**Outputs:**
- `data/processed/vrptdt/berlin_500/base_dist_km.npy` (501×501 distance matrix)
- `data/processed/vrptdt/berlin_500/base_TT_data_min.npy` (7×501×501 travel times)
- Episode files: `data/processed/vrptdt/berlin_500/episodes/{SPLIT}/seed_XXX.npz`

**Data splits:**
- TRAIN: seeds 0-199 (200 episodes)
- VAL: seeds 200-229 (30 episodes)
- TEST: seeds 230-259 (30 episodes)

**Time discretization:**
- 7 bins × 60 minutes = 420 minutes (15:00-22:00)
- Congestion profile: `[0.90, 0.80, 0.70, 0.60, 0.70, 0.80, 0.90]`
- Bin 3 (18:00-19:00) is peak congestion (60% of free-flow speed)

---

### **M2: Events + Objective + Cost Tensors**

#### **Step 1: Events + Lambda Fitting**
```bash
# Fit lambda on TRAIN data only (avoid leakage)
python scripts/fit_lambda.py

# Apply events (rain + blockages)
python scripts/apply_events.py --seed 230 --save
```

**Lambda fitting:**
- λ = median(CO2_leg) / median(TT_leg) = **0.5915**
- Stored in: `configs/lambda.json`

**Event generation (deterministic by seed):**
- **Rain** (observable): Duration L ∈ {1,2,3} bins, ρ_TT ∈ {0.05, 0.10, 0.20}, ρ_CO2 ∈ {0.02, 0.05, 0.10}
- **Blockages** (hidden): K=3 arcs in first 60% of route, active in bin 1 (16:00-17:00)

#### **Step 2: CO2 Proxy + Integer Costs + BIG_M**
```bash
# Build integer cost tensors
python scripts/build_costs_int.py --seed 230 --save

# Verify costs
python scripts/verify_events_costs.py

# Benchmark cost build latency
python scripts/bench_cost_build.py
```

**MEET/Jabali CO2 proxy:**
```
v_kmh[b,i,j] = dist_km[i,j] / (TT_hat[b,i,j]/60)
e(v) = α*v² + β*v + γ + δ/v
CO2[b,i,j] = dist_km[i,j] * e(v_kmh[b,i,j])
```

**Integer costs for OR-Tools:**
```
J_cost_int[b,i,j] = round((CO2[b,i,j] + λ * TT_hat[b,i,j]) * SCALE)
J_cost_int[blockage_bin, u, v] = BIG_M  # Penalize blocked arcs
```

---

### **M3: Baseline Policies + Blockage Simulation**

#### **Step 1: Baseline Policies (No Digital Twin)**

**Important:** Use `run_baselines_.py` (with underscore) for baseline-only runs:

```bash
# Run baseline policies (TEST split, rain scenario)
python scripts/run_baselines_.py --split TEST --time_limit_ms 200 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60
python scripts/run_baselines_.py --split TEST --time_limit_ms 500 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60
python scripts/run_baselines_.py --split TEST --time_limit_ms 800 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60

# No-rain ablation
python scripts/run_baselines_.py --split TEST --time_limit_ms 200 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60 --disable_rain
python scripts/run_baselines_.py --split TEST --time_limit_ms 500 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60 --disable_rain
python scripts/run_baselines_.py --split TEST --time_limit_ms 800 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60 --disable_rain
```

**Note:** `run_baselines_.py` runs B0, B1, B2 **without** the digital twin (uses base TT_data only)

**Baseline policies:**
- **B0_PlanOnce**: Plan once at start, no replanning
- **B2_BlockageReplan**: Replan once at blockage bin entry
- **B1_AlwaysReplan**: Replan at every customer arrival

**Blockage execution model:**
- Vehicle waits until bin end when blocked arc encountered
- Then traverses arc in next bin
- Wait contributes to `traffic_wait_min`

**Outputs:** 
- `data/processed/bench/week3_results/baselines_{SPLIT}_{rain|norain}_cap{CAP}.csv`

#### **Step 2: Add Planning Overhead (Wall-Clock Objective)**
```bash
# Generate plots
python scripts/plot_policy_eval_grid.py --split TEST --metric J_wall --caps 200,500,800
python scripts/plot_policy_eval_grid.py --split TEST --metric planning_wait_min --caps 200,500,800
python scripts/plot_policy_eval_grid.py --split TEST --metric solve_ms_total --caps 200,500,800
```

**Planning overhead model:**
```
planning_wait_min += solve_ms / 60000
elapsed_min += solve_ms / 60000
```

**Two objectives:**
- **J_exec** (ignores planning): `CO2_total + λ * exec_time_min`
- **J_wall** (includes planning): `CO2_total + λ * wall_time_min`

**Committed objective:** J_wall (reflects real trade-offs)

---

### **M4: EWMA Digital Twin + Gate B**

#### **Step 1: Digital Twin with EWMA Online Learning**

**Important:** Use `run_baselines.py` (no underscore) for digital twin and Gate B runs:

```bash
# Run with digital twin only (B0/B1/B2 with EWMA)
python scripts/run_baselines.py --split TEST --time_limit_ms 200 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60
python scripts/run_baselines.py --split TEST --time_limit_ms 500 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60
python scripts/run_baselines.py --split TEST --time_limit_ms 800 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60

# No-rain with digital twin
python scripts/run_baselines.py --split TEST --time_limit_ms 500 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60 --disable_rain
```

**Note:** Without `--include_gate`, this runs B0/B1/B2 with EWMA digital twin predictions

#### **Step 2: Gate B (Probe-Then-Commit Gating)**

```bash
# Run with Gate B (adds B3_GateReplan policy)
python scripts/run_baselines.py --split TEST --time_limit_ms 200 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60 --include_gate
python scripts/run_baselines.py --split TEST --time_limit_ms 500 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60 --include_gate
python scripts/run_baselines.py --split TEST --time_limit_ms 800 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60 --include_gate

# No-rain with Gate B
python scripts/run_baselines.py --split TEST --time_limit_ms 200 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60 --disable_rain --include_gate
python scripts/run_baselines.py --split TEST --time_limit_ms 500 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60 --disable_rain --include_gate
python scripts/run_baselines.py --split TEST --time_limit_ms 800 --start_bin 0 --blockage_bin 1 --n_blockages 3 --early_frac 0.60 --disable_rain --include_gate
```

**Note:** `--include_gate` adds B3_GateReplan policy (probe-then-commit)

**Digital twin prediction:**
```
TT_hat[b,i,j] = m̂(b) * TT_data[b,i,j] * (1 + ρ_TT)
```

**EWMA update rule (α = 0.2, frozen):**
```
m_obs = TT_obs / (TT_data[b,i,j] * (1 + ρ_TT))
m̂(b) ← (1-α) * m̂(b) + α * m_obs
```

**Initialization:** From TRAIN-fitted priors (`binmean_m.json`)

**Key features:**
- Only executed bin's multiplier updates
- Online learning (real-time during episode)
- No batch retraining
- Policy-insensitive accuracy (all use same EWMA)

#### **Step 3: Results + Interpretation**
```bash
# Generate digital twin performance plots
python scripts/plot_twin_eval_grid.py --split TEST --caps 200,500,800

# Generate Gate B performance plots
python scripts/plot_gate_eval_grid.py --split TEST --caps 200,500,800
```

**Gate B metrics:**
- `n_gate_probes` - Number of 50ms probe solves
- `n_gate_full_replans` - Number of triggered full replans
- `gate_gain_hat_mean` - Average estimated gain from probes

**Prediction error metrics:**
```
rel_pred_err = |TT_obs - TT_hat| / TT_hat
```
- Mean error: ~3.4%
- P95 error: ~9%
- Policy-insensitive

**Key findings:**
- B2 captures 98.8% of B1's quality at 10% computational cost
- B1 ≈ B2 > B0 in route quality
- Increasing solver cap adds latency without major quality gains

---

## Key Metrics

### **Solution Quality:**
- `J_wall` - Wall-clock objective (CO2 + λ × wall_time)
- `J_exec` - Execution objective (CO2 + λ × exec_time)
- `CO2_total` - Total emissions
- `travel_min` - Actual driving time
- `traffic_wait_min` - Blockage-induced delays
- `planning_wait_min` - Solver latency

### **Computational Cost:**
- `solve_ms_total` - Sum of all solver times
- `solve_ms_p95` - 95th percentile solve time
- `solve_ms_max` - Maximum solve time
- `n_replans` - Number of replanning events

### **Prediction Accuracy:**
- `rel_pred_err_mean` - Mean relative prediction error
- `rel_pred_err_p95` - 95th percentile error
- `rel_pred_err_max` - Maximum error

---

## Configuration Files

**configs/ingest.json:**
- `bin_minutes`: 60
- `service_time_min`: 2
- `SCALE`: 10000 (for integer costs)

**configs/lambda.json:**
- `lambda`: 0.5915 (fitted on TRAIN)

**configs/binmean_m.json:**
- EWMA initialization priors from TRAIN data

**Gate B parameters (hardcoded):**
- `probe_time_ms`: 50 (10% of standard 500ms cap)
- `eta` (threshold): 1.0 (require 100% improvement to trigger full replan)
- `alpha` (EWMA): 0.2 (fixed, not tuned)

---

## 📈 Experimental Design

**Baseline comparison (run_baselines_.py):**
- Rain: {rain, no-rain}
- Solver cap: {200ms, 500ms, 800ms}
- Policies: {B0, B1, B2} **without digital twin**
- Total: 2 × 3 = **6 configurations**

**Digital twin evaluation (run_baselines.py):**
- Rain: {rain, no-rain}
- Solver cap: {200ms, 500ms, 800ms}
- Policies: {B0, B1, B2} **with EWMA digital twin**
- Total: 2 × 3 = **6 configurations**

**Gate B evaluation (run_baselines.py --include_gate):**
- Rain: {rain, no-rain}
- Solver cap: {200ms, 500ms, 800ms}
- Policies: {B0, B1, B2, **B3_GateReplan**} with digital twin
- Total: 2 × 3 = **6 configurations**

**TEST episodes:** 30 seeds per configuration

---

## Requirements

**Software:**
- Python 3.12
- OR-Tools 9.14.6206
- NumPy, Pandas, Matplotlib
- Git for version control

**Hardware:**
- AMD Ryzen 5 PRO 4650U (or equivalent)
- 16 GB RAM
- Windows OS (Git Bash for commands)

---

##  Reproducibility

**All randomness is deterministic:**
- Same seed → same customers, same disruptions
- Fixed hyperparameters (λ, α, η)
- No VAL tuning
- Git version control for all code

**Automated checks:**
- 21 nodes per episode (20 customers + depot)
- BIG_M enforced on blocked arcs
- No seed overlap across splits
- Non-negative finite costs

---

##  NOT in Git

Per `.gitignore`:
- `/data/raw/` - Raw datasets
- `/data/processed/` - Generated episodes and results
- `/data/canonical/` - Canonical representations
- `/logs/` - Execution logs
- `*.tar.bz2` - Archive files

---



For questions about implementation details, refer to the thesis document or individual M1-M4 step files.

---

**Last updated:** Based on M1-M4 implementation steps
**Thesis:** Lightweight Digital Twin Architecture for Intelligent Replanning in Green Vehicle Routing