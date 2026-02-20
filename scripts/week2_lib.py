from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class Events:
    rain_mask: np.ndarray  # (B,) bool
    rho_TT: float
    rho_CO2: float
    blocked_u: int
    blocked_v: int
    init_route: np.ndarray  # (N+1,) visiting order incl return to depot


def load_episode_npz(episode_path: Path) -> Dict:
    z = np.load(episode_path, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    return {
        "node_ids": z["node_ids"],
        "dist_km": z["dist_km"].astype(np.float32),
        "TT_data_min": z["TT_data_min"].astype(np.float32),
        "meta": meta,
    }


def nearest_neighbor_route(TT0_min: np.ndarray) -> np.ndarray:
    """
    Deterministic NN route on TT bin 0.
    TT0_min: (N,N) with depot index 0.
    Returns route indices including depot return: e.g., [0, 5, 2, ..., 0]
    """
    n = TT0_min.shape[0]
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


def choose_blocked_arc_on_route(rng: np.random.Generator, route: np.ndarray) -> Tuple[int, int]:
    """
    Option B (SPEC): choose a blocked arc (u->v) from the initial planned route (guaranteed to matter).
    route includes depot return, so arcs are route[k]->route[k+1]
    """
    arcs = [(int(route[k]), int(route[k + 1])) for k in range(len(route) - 1)]
    k = int(rng.integers(0, len(arcs)))
    return arcs[k]


def generate_events_for_episode(seed: int, TT_data_min: np.ndarray) -> Events:
    """
    Deterministic events given episode seed.
    """
    rng = np.random.default_rng(seed)
    route = nearest_neighbor_route(TT_data_min[0])
    u, v = choose_blocked_arc_on_route(rng, route)
    rain_mask, rho_TT, rho_CO2 = sample_rain(rng, TT_data_min.shape[0])
    return Events(rain_mask=rain_mask, rho_TT=rho_TT, rho_CO2=rho_CO2, blocked_u=u, blocked_v=v, init_route=route)


def apply_rain_to_TT(TT_base_min: np.ndarray, rain_mask: np.ndarray, rho_TT: float) -> np.ndarray:
    TT = TT_base_min.astype(np.float32).copy()
    for b in range(TT.shape[0]):
        if bool(rain_mask[b]):
            TT[b] *= (1.0 + float(rho_TT))
    return TT


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

    # speed km/h = dist_km / (TT_hours)
    TT_hours = TT / 60.0
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(dist[None, :, :] > 1e-6, dist[None, :, :] / np.maximum(TT_hours, 1e-6), 0.0)

    v = np.clip(v, v_clip[0], v_clip[1])

    # e(v)
    e = (alpha * (v ** 2)) + (beta * v) + gamma + (delta / np.maximum(v, 1e-6))

    # CO2 = dist * e(v); if dist is ~0 => CO2=0
    CO2 = dist[None, :, :] * e
    CO2 = np.where(dist[None, :, :] > 1e-6, CO2, 0.0).astype(np.float32)
    # diagonal exactly 0
    for b in range(B):
        np.fill_diagonal(CO2[b], 0.0)
    return CO2


def apply_rain_to_CO2(CO2: np.ndarray, rain_mask: np.ndarray, rho_CO2: float) -> np.ndarray:
    out = CO2.astype(np.float32).copy()
    for b in range(out.shape[0]):
        if bool(rain_mask[b]):
            out[b] *= (1.0 + float(rho_CO2))
    return out


def build_int_costs(
    TT_hat_min: np.ndarray,
    CO2_hat: np.ndarray,
    lam: float,
    SCALE: int,
    blockage_bin: int,
    blocked_u: int,
    blocked_v: int,
    BIG_M_cost_int: int,
) -> Dict[str, np.ndarray]:
    """
    Builds int64 costs for OR-Tools:
      time_cost_int[b,i,j] = round(TT_hat_min * SCALE)
      co2_cost_int[b,i,j]  = round(CO2_hat * SCALE)
      J_cost_int           = co2_cost_int + round(lam * time_cost_float_scaled)

    Blockage rule (SPEC): BIG_M applied to PLANNING COST J in blockage_bin on (u->v).
    """
    TT = TT_hat_min.astype(np.float32)
    CO2 = CO2_hat.astype(np.float32)

    time_cost = np.rint(TT * SCALE).astype(np.int64)
    co2_cost = np.rint(CO2 * SCALE).astype(np.int64)

    # J = CO2 + lam * TT
    J = co2_cost + np.rint((lam * TT) * SCALE).astype(np.int64)

    # enforce blockage on planning cost only
    J[blockage_bin, blocked_u, blocked_v] = int(BIG_M_cost_int)

    # diagonals to 0
    B = TT.shape[0]
    for b in range(B):
        np.fill_diagonal(time_cost[b], 0)
        np.fill_diagonal(co2_cost[b], 0)
        np.fill_diagonal(J[b], 0)

    return {"time_cost_int": time_cost, "co2_cost_int": co2_cost, "J_cost_int": J}
