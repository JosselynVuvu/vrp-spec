from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np


# -----------------------------
# Events container (SPEC v1.7)
# -----------------------------
@dataclass(frozen=True)
class Events:
    rain_mask: np.ndarray        # (B,) bool
    rho_TT: float
    rho_CO2: float
    blocked_arcs: np.ndarray     # (K,2) int32, directed OD arcs (u->v)
    init_route: np.ndarray       # (L,) int32, visiting order incl return to depot


# -----------------------------
# Episode loading
# -----------------------------
def load_episode_npz(episode_path: Path) -> Dict:
    z = np.load(episode_path, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    return {
        "node_ids": z["node_ids"],
        "dist_km": z["dist_km"].astype(np.float32),
        "TT_data_min": z["TT_data_min"].astype(np.float32),
        "meta": meta,
    }


# -----------------------------
# Deterministic initial route
# -----------------------------
def nearest_neighbor_route(TT0_min: np.ndarray) -> np.ndarray:
    """
    Deterministic nearest-neighbor route on TT bin 0.
    TT0_min: (N,N) with depot index 0.
    Returns route indices including depot return: e.g., [0, 5, 2, ..., 0]
    """
    n = int(TT0_min.shape[0])
    unvisited = set(range(1, n))
    route = [0]
    cur = 0
    while unvisited:
        nxt = min(unvisited, key=lambda j: float(TT0_min[cur, j]))
        route.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    route.append(0)
    return np.array(route, dtype=np.int32)


# -----------------------------
# Rain sampling (SPEC)
# -----------------------------
def sample_rain(rng: np.random.Generator, n_bins: int) -> Tuple[np.ndarray, float, float]:
    """
    Rain is bin-aligned, duration 1–3 bins, uniformly sampled.
    Intensities sampled uniformly from frozen sets.
    """
    L = int(rng.integers(1, 4))  # 1..3
    start = int(rng.integers(0, n_bins - L + 1))
    mask = np.zeros((n_bins,), dtype=bool)
    mask[start : start + L] = True

    rho_TT = float(rng.choice([0.05, 0.10, 0.20]))
    rho_CO2 = float(rng.choice([0.02, 0.05, 0.10]))
    return mask, rho_TT, rho_CO2


# -----------------------------
# Blockage arc selection (SPEC v1.7)
# -----------------------------
def _dedup_preserve_order(arcs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    seen = set()
    out: List[Tuple[int, int]] = []
    for uv in arcs:
        if uv not in seen:
            seen.add(uv)
            out.append(uv)
    return out


def choose_blocked_arcs_on_route(
    rng: np.random.Generator,
    route: np.ndarray,
    k: int = 3,
    skip_depot: bool = True,
    early_frac: float = 0.6,  # since blockage_bin=1, bias toward early route arcs
) -> np.ndarray:
    """
    Choose k distinct directed arcs from the initial route.
    - Excludes depot arcs if skip_depot=True.
    - Biases toward EARLY route arcs (better chance to matter at early blockage bin).
    Returns (k,2) int32 array.
    """
    arcs = [(int(route[t]), int(route[t + 1])) for t in range(len(route) - 1)]

    if skip_depot:
        arcs = [(u, v) for (u, v) in arcs if u != 0 and v != 0]

    arcs = _dedup_preserve_order(arcs)

    if len(arcs) < k:
        raise ValueError(f"Not enough eligible arcs to choose k={k}. eligible={len(arcs)}")

    cutoff = max(k, int(np.ceil(early_frac * len(arcs))))
    cutoff = min(cutoff, len(arcs))
    pool = arcs[:cutoff]

    idxs = rng.choice(len(pool), size=k, replace=False)
    chosen = [pool[i] for i in idxs]
    return np.array(chosen, dtype=np.int32)


def generate_events_for_episode(
    seed: int,
    TT_data_min: np.ndarray,
    k_blocked: int = 3,
    early_frac: float = 0.6,
) -> Events:
    """
    Deterministic events given episode seed.
    - Rain: sampled (seeded)
    - Blockage arcs: Option B (seeded), k=3 arcs from early segment of initial route
    """
    rng = np.random.default_rng(seed)
    route = nearest_neighbor_route(TT_data_min[0])
    blocked_arcs = choose_blocked_arcs_on_route(
        rng=rng,
        route=route,
        k=int(k_blocked),
        skip_depot=True,
        early_frac=float(early_frac),
    )
    rain_mask, rho_TT, rho_CO2 = sample_rain(rng, int(TT_data_min.shape[0]))
    return Events(
        rain_mask=rain_mask,
        rho_TT=rho_TT,
        rho_CO2=rho_CO2,
        blocked_arcs=blocked_arcs,
        init_route=route,
    )


# -----------------------------
# Rain application (observable)
# -----------------------------
def apply_rain_to_TT(TT_base_min: np.ndarray, rain_mask: np.ndarray, rho_TT: float) -> np.ndarray:
    TT = TT_base_min.astype(np.float32).copy()
    for b in range(TT.shape[0]):
        if bool(rain_mask[b]):
            TT[b] *= (1.0 + float(rho_TT))
    return TT


def apply_rain_to_CO2(CO2: np.ndarray, rain_mask: np.ndarray, rho_CO2: float) -> np.ndarray:
    out = CO2.astype(np.float32).copy()
    for b in range(out.shape[0]):
        if bool(rain_mask[b]):
            out[b] *= (1.0 + float(rho_CO2))
    return out


# -----------------------------
# Emissions proxy (MEET/Jabali)
# -----------------------------
def meet_emissions_proxy(
    dist_km: np.ndarray,
    TT_min: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    v_clip: Tuple[float, float] = (5.0, 130.0),
) -> np.ndarray:
    """
    CO2 proxy per bin, MEET/Jabali-style: e(v)=a v^2 + b v + c + d/v
    Returns CO2[b,i,j] in arbitrary consistent units (proxy).
    """
    B, n, _ = TT_min.shape
    dist = dist_km.astype(np.float32)
    TT = TT_min.astype(np.float32)

    TT_hours = TT / 60.0
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(dist[None, :, :] > 1e-6, dist[None, :, :] / np.maximum(TT_hours, 1e-6), 0.0)

    v = np.clip(v, v_clip[0], v_clip[1])

    e = (alpha * (v ** 2)) + (beta * v) + gamma + (delta / np.maximum(v, 1e-6))

    CO2 = dist[None, :, :] * e
    CO2 = np.where(dist[None, :, :] > 1e-6, CO2, 0.0).astype(np.float32)

    for b in range(B):
        np.fill_diagonal(CO2[b], 0.0)
    return CO2


# -----------------------------
# Integer planning costs (OR-Tools)
# -----------------------------
def build_int_costs(
    TT_hat_min: np.ndarray,
    CO2_hat: np.ndarray,
    lam: float,
    SCALE: int,
    blockage_bin: int,
    BIG_M_cost_int: int,
    blocked_arcs: Optional[np.ndarray] = None,
    blocked_u: Optional[int] = None,   # backward compatible
    blocked_v: Optional[int] = None,   # backward compatible
) -> Dict[str, np.ndarray]:
    """
    Builds int64 costs for OR-Tools:
      time_cost_int[b,i,j] = round(TT_hat_min * SCALE)
      co2_cost_int[b,i,j]  = round(CO2_hat * SCALE)
      J_cost_int           = co2_cost_int + round(lam * TT_hat_min * SCALE)

    Blockage rule (SPEC): BIG_M applied to PLANNING COST J in blockage_bin
    on ALL blocked arcs (u->v). (TT itself is not overwritten here.)
    """
    TT = TT_hat_min.astype(np.float32)
    CO2 = CO2_hat.astype(np.float32)

    time_cost = np.rint(TT * SCALE).astype(np.int64)
    co2_cost = np.rint(CO2 * SCALE).astype(np.int64)

    # J = CO2 + lam * TT
    J = co2_cost + np.rint((lam * TT) * SCALE).astype(np.int64)

    # collect blocked pairs
    pairs: List[Tuple[int, int]] = []
    if blocked_arcs is not None:
        ba = np.asarray(blocked_arcs, dtype=np.int64).reshape(-1, 2)
        pairs = [(int(u), int(v)) for (u, v) in ba]
    elif blocked_u is not None and blocked_v is not None:
        pairs = [(int(blocked_u), int(blocked_v))]

    # enforce blockage on planning cost only (all arcs)
    for (u, v) in pairs:
        J[int(blockage_bin), int(u), int(v)] = int(BIG_M_cost_int)

    # diagonals to 0
    B = TT.shape[0]
    for b in range(B):
        np.fill_diagonal(time_cost[b], 0)
        np.fill_diagonal(co2_cost[b], 0)
        np.fill_diagonal(J[b], 0)

    return {"time_cost_int": time_cost, "co2_cost_int": co2_cost, "J_cost_int": J}
