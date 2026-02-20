# Week 1 – Step 2: Data ingestion + canonicalization (SPEC v1.7)

## Base instance (locked)
- Instance file: `berlin_500.json`
- Content available in JSON:
  - `depot`: latitude/longitude
  - `items`: 500 customers keyed by ID with latitude/longitude
  - time-window fields exist but are NOT used in the thesis scope (no customer time windows)

## Node ordering (deterministic)
- Index 0: depot
- Indices 1..500: customers sorted by numeric item key

## Time discretization (SPEC v1.7)
- 7 bins of 60 minutes
- Bin start hours: 15,16,17,18,19,20,21 (represents 15:00–22:00 horizon)

## Distance matrix
Because no distance matrix is provided, distances are computed from coordinates:
- `dist_km[i,j]`: haversine distance (km), diagonal set to 0
- Output:
  - `data/processed/vrptdt/berlin_500/base_dist_km.npy` (501×501)

## Time-dependent travel time matrix (data-driven proxy)
Because per-bin travel times are not provided, TT is constructed via a calibrated free-flow speed and bin multipliers:

1) Calibrate free-flow speed `v_free_kmh`:
- Choose target median leg travel time = 6 minutes
- Compute median off-diagonal distance from `dist_km`
- `v_free_kmh_raw = median_dist_km / (6/60)`
- Clip to [10, 60] km/h (clip activated in this dataset)

2) Apply frozen congestion profile (Pattern B):
- `speed_mult = [0.90, 0.80, 0.70, 0.60, 0.70, 0.80, 0.90]`
- Slowest bin is b=3 (expected 18:00–19:00)

3) Construct travel times:
- `TT_data_min[b,i,j] = (dist_km[i,j] / (v_free_kmh * speed_mult[b])) * 60`
- Diagonal set to 0; off-diagonal clamped to >= 0.1 min

Outputs:
- `data/processed/vrptdt/berlin_500/base_TT_data_min.npy` (7×501×501)
- `data/processed/vrptdt/berlin_500/base_meta.json` (stores v_free_kmh, bin settings, multipliers)

## Episode bank generation (20 customers per episode)
- For each seed, sample 20 customers uniformly without replacement + depot → 21 nodes
- Save per-seed episode files:
  - `dist_km` (21×21)
  - `TT_data_min` (7×21×21)
  - `node_ids` (len 21)
  - `meta_json` (seed, split, bin config)

## Train/Val/Test split (no leakage)
- Disjoint seed ranges:
  - TRAIN: 0–199
  - VAL: 200–229
  - TEST: 230–259
- Split assignment is determined only by seed (ensures no leakage)

## Verification (scripts executed)
- `peek_instance.py`: confirmed coords and bin hours
- `inspect_base_canonical.py`: verified bin 3 slowest and no NaNs
- `verify_processed.py`: verified episode files exist for all seeds and invariants hold
