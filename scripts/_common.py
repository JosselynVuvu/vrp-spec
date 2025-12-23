import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

def load_json(rel_path: str) -> dict:
    p = REPO_ROOT / rel_path
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def parse_nodes_from_vrptdt_instance(instance_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      node_ids: (N+1,) dtype '<U...' with node_ids[0]='depot', node_ids[1:]=sorted item keys as strings
      coords_latlon: (N+1,2) float64 [lat, lon] degrees
    """
    with open(instance_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    depot = d["depot"]
    items = d["items"]  # dict keyed by strings "1".."N"

    # Stable numeric sort of item keys
    item_keys = sorted(items.keys(), key=lambda x: int(x))
    node_ids = ["depot"] + item_keys

    coords = np.zeros((len(node_ids), 2), dtype=np.float64)
    coords[0, 0] = float(depot["latitude"])
    coords[0, 1] = float(depot["longitude"])

    for idx, k in enumerate(item_keys, start=1):
        coords[idx, 0] = float(items[k]["latitude"])
        coords[idx, 1] = float(items[k]["longitude"])

    return np.array(node_ids, dtype=str), coords

def haversine_km_matrix(coords_latlon_deg: np.ndarray) -> np.ndarray:
    """
    Vectorized haversine distance matrix in km.
    coords_latlon_deg: (N,2) degrees [lat, lon]
    returns: (N,N) float64
    """
    R = 6371.0  # km
    lat = np.deg2rad(coords_latlon_deg[:, 0])
    lon = np.deg2rad(coords_latlon_deg[:, 1])

    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]

    a = np.sin(dlat / 2.0) ** 2 + (np.cos(lat)[:, None] * np.cos(lat)[None, :]) * (np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    dist = R * c
    np.fill_diagonal(dist, 0.0)
    return dist

def calibrate_v_free_kmh(dist_km: np.ndarray, target_median_leg_tt_min: float, clip_lo: float, clip_hi: float) -> Tuple[float, float, float]:
    """
    Calibrate free-flow speed so that median off-diagonal leg TT equals target_median_leg_tt_min
    at multiplier = 1.0 (neutral bin).
    Returns: (median_dist_km, v_free_raw, v_free_clipped)
    """
    n = dist_km.shape[0]
    mask = ~np.eye(n, dtype=bool)
    median_dist_km = float(np.median(dist_km[mask]))

    # v = dist / (tt_hours)
    tt_hours = target_median_leg_tt_min / 60.0
    v_free_raw = median_dist_km / tt_hours if tt_hours > 0 else 30.0
    v_free_clipped = float(np.clip(v_free_raw, clip_lo, clip_hi))
    return median_dist_km, float(v_free_raw), v_free_clipped

def build_TT_data_min(dist_km: np.ndarray, v_free_kmh: float, speed_mult: List[float]) -> np.ndarray:
    """
    TT_data_min[t,i,j] = (dist_km[i,j] / (v_free_kmh*speed_mult[t])) * 60
    returns float32 (T,N,N)
    """
    T = len(speed_mult)
    n = dist_km.shape[0]
    TT = np.zeros((T, n, n), dtype=np.float64)

    for t, m in enumerate(speed_mult):
        v = max(1e-6, v_free_kmh * float(m))
        TT[t] = (dist_km / v) * 60.0
        np.fill_diagonal(TT[t], 0.0)

        # Clamp off-diagonal to avoid zeros from numerical issues
        off = ~np.eye(n, dtype=bool)
        TT[t][off] = np.maximum(TT[t][off], 0.1)

    return TT.astype(np.float32)

def split_for_seed(seed: int, seeds_cfg: dict) -> str:
    """
    Supports two formats:
      A) range format:
         {"train_start":0,"train_end":199,"val_start":200,"val_end":229,"test_start":230,"test_end":259}
      B) explicit lists:
         {"train":[...], "val":[...], "test":[...]}
    """
    if "train_start" in seeds_cfg:
        if seeds_cfg["train_start"] <= seed <= seeds_cfg["train_end"]:
            return "TRAIN"
        if seeds_cfg["val_start"] <= seed <= seeds_cfg["val_end"]:
            return "VAL"
        if seeds_cfg["test_start"] <= seed <= seeds_cfg["test_end"]:
            return "TEST"
        raise ValueError(f"Seed {seed} not in any split range.")
    else:
        if seed in set(seeds_cfg.get("train", [])):
            return "TRAIN"
        if seed in set(seeds_cfg.get("val", [])):
            return "VAL"
        if seed in set(seeds_cfg.get("test", [])):
            return "TEST"
        raise ValueError(f"Seed {seed} not in any split list.")

def expected_seeds(seeds_cfg: dict) -> Dict[str, List[int]]:
    if "train_start" in seeds_cfg:
        return {
            "TRAIN": list(range(seeds_cfg["train_start"], seeds_cfg["train_end"] + 1)),
            "VAL": list(range(seeds_cfg["val_start"], seeds_cfg["val_end"] + 1)),
            "TEST": list(range(seeds_cfg["test_start"], seeds_cfg["test_end"] + 1)),
        }
    return {
        "TRAIN": list(seeds_cfg.get("train", [])),
        "VAL": list(seeds_cfg.get("val", [])),
        "TEST": list(seeds_cfg.get("test", [])),
    }
