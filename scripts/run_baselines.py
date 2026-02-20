# scripts/run_baselines.py
# Milestone 4 — EWMA twin + receding-horizon OR-Tools (+ optional Gate B)

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import vrp_lib


# -----------------------------
# Helpers: config + paths
# -----------------------------
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(p: str | None, root: Path) -> Path | None:
    if not p:
        return None
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp)


def get_processed_base_dir(ingest_cfg: Dict[str, Any], root: Path) -> Path:
    for k in ("processed_base_dir", "out_base_dir", "processed_dir", "base_out_dir"):
        p = resolve_path(ingest_cfg.get(k), root)
        if p and p.exists():
            return p
    return root / "data" / "processed" / "vrptdt" / "berlin_500"


def episode_path(base_dir: Path, split: str, seed: int) -> Path:
    return base_dir / "episodes" / split.upper() / f"seed_{seed:03d}.npz"


def load_split_seeds_range(seeds_cfg: Dict[str, Any], split: str, base_dir: Path) -> List[int]:
    """
    configs/seeds.json uses ranges:
      train_start/train_end, val_start/val_end, test_start/test_end

    Auto-detect inclusive vs exclusive end by checking whether the episode file for
    the end seed exists.
    """
    s = split.lower()
    k_start = f"{s}_start"
    k_end = f"{s}_end"
    if k_start not in seeds_cfg or k_end not in seeds_cfg:
        raise KeyError(f"Missing {k_start}/{k_end} in configs/seeds.json")

    start = int(seeds_cfg[k_start])
    end = int(seeds_cfg[k_end])

    def ep_exists(seed: int) -> bool:
        return (base_dir / "episodes" / split.upper() / f"seed_{seed:03d}.npz").exists()

    inclusive_end = end
    if not ep_exists(end) and ep_exists(end - 1):
        inclusive_end = end - 1  # treat end as exclusive

    if inclusive_end < start:
        raise ValueError(f"Bad seed range for {split}: start={start}, end={end}")

    return list(range(start, inclusive_end + 1))


def get_emissions_params(ingest_cfg: Dict[str, Any]) -> Tuple[float, float, float, float]:
    for key in ("emissions_params", "meet_params", "co2_params"):
        if isinstance(ingest_cfg.get(key), dict):
            p = ingest_cfg[key]
            return (
                float(p.get("alpha", 0.0)),
                float(p.get("beta", 0.0)),
                float(p.get("gamma", 1.0)),
                float(p.get("delta", 50.0)),
            )
    return (
        float(ingest_cfg.get("alpha", 0.0)),
        float(ingest_cfg.get("beta", 0.0)),
        float(ingest_cfg.get("gamma", 1.0)),
        float(ingest_cfg.get("delta", 50.0)),
    )


# -----------------------------
# OR-Tools path solver (start != end)
# -----------------------------
def ortools_solve_path(
    cost_mat: np.ndarray,
    start_idx: int,
    end_idx: int,
    time_limit_ms: int,
) -> Tuple[List[int], int, float]:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    n = int(cost_mat.shape[0])
    manager = pywrapcp.RoutingIndexManager(n, 1, [start_idx], [end_idx])
    routing = pywrapcp.RoutingModel(manager)

    def transit_cb(from_index: int, to_index: int) -> int:
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(cost_mat[i, j])

    transit_idx = routing.RegisterTransitCallback(transit_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.time_limit.FromMilliseconds(int(time_limit_ms))
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.log_search = False

    t0 = time.perf_counter()
    sol = routing.SolveWithParameters(params)
    t_ms = (time.perf_counter() - t0) * 1000.0

    if sol is None:
        return [], -1, t_ms

    route: List[int] = []
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        route.append(manager.IndexToNode(idx))
        idx = sol.Value(routing.NextVar(idx))
    route.append(manager.IndexToNode(idx))
    return route, int(sol.ObjectiveValue()), t_ms


def solve_subproblem(
    cost_mat_full: np.ndarray,
    start_node: int,
    end_node: int,
    remaining_customers: List[int],
    time_limit_ms: int,
) -> Tuple[List[int], int, float]:
    """
    Solve route on nodes: {start_node} U remaining_customers U {end_node},
    returning a path from start_node to end_node in original node indices.
    """
    nodes = [start_node] + sorted([n for n in remaining_customers if n != start_node and n != end_node])
    if end_node not in nodes:
        nodes.append(end_node)

    idx = nodes
    sub = cost_mat_full[np.ix_(idx, idx)]
    start_idx = idx.index(start_node)
    end_idx = idx.index(end_node)

    route_sub, obj, solve_ms = ortools_solve_path(sub, start_idx, end_idx, time_limit_ms)
    if not route_sub:
        return [], -1, solve_ms

    route_nodes = [idx[i] for i in route_sub]
    return route_nodes, obj, solve_ms


# -----------------------------
# EWMA twin
# -----------------------------
EWMA_ALPHA = 0.2  # SPEC frozen


def _rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) * 9973 + 12345)


def sample_m_true(B: int, seed: int) -> np.ndarray:
    """
    Hidden global traffic multiplier per bin (episode-seeded).
    """
    rng = _rng_from_seed(seed)
    m = rng.normal(loc=1.0, scale=0.08, size=B).astype(np.float32)
    m = np.clip(m, 0.75, 1.35)
    return m


def compute_cost_mat_int_for_bin(
    *,
    TT_data_min_b: np.ndarray,      # (N,N)
    dist_km: np.ndarray,            # (N,N)
    rain_on: bool,
    rho_TT: float,
    rho_CO2: float,
    m_hat_b: float,
    lam: float,
    SCALE: int,
    BIG_M_cost_int: int,
    blockage_bin: int,
    b: int,
    blocked_arcs: List[Tuple[int, int]],
    emissions_params: Tuple[float, float, float, float],
) -> np.ndarray:
    """
    Build OR-Tools int arc-cost matrix for bin b:
      TT_hat = TT_data * rain_factor * m_hat
      CO2_hat = MEET(dist, TT_hat) * (1+rho_CO2 if rain)
      J_hat = CO2_hat + lam * TT_hat
      + BIG_M patch during blockage_bin on blocked arcs (planning-cost only).
    """
    rain_factor = (1.0 + float(rho_TT)) if bool(rain_on) else 1.0
    TT_hat_b = TT_data_min_b.astype(np.float32) * float(rain_factor) * float(m_hat_b)

    alpha, beta, gamma, delta = emissions_params
    CO2_hat_b = vrp_lib.meet_emissions_proxy(
        dist_km.astype(np.float32),
        TT_hat_b[None, ...],
        float(alpha), float(beta), float(gamma), float(delta),
    )[0].astype(np.float32)

    if bool(rain_on):
        CO2_hat_b = CO2_hat_b * (1.0 + float(rho_CO2))

    J_hat_b = CO2_hat_b + float(lam) * TT_hat_b
    C_int = np.rint(J_hat_b * int(SCALE)).astype(np.int64)

    np.fill_diagonal(C_int, 0)

    if int(b) == int(blockage_bin):
        for (u, v) in blocked_arcs:
            C_int[int(u), int(v)] = int(BIG_M_cost_int)

    return C_int


# -----------------------------
# Simulation / policy
# -----------------------------
@dataclass
class EpisodeCtx:
    seed: int
    node_ids: np.ndarray
    dist_km: np.ndarray                  # (N,N)

    TT_data_min: np.ndarray              # (B,N,N) dataset congestion
    TT_hat_min: np.ndarray               # (B,N,N) planner prediction (unused by policy; kept for debug)
    TT_true_min: np.ndarray              # (B,N,N) simulator truth (rain * m_true)

    CO2_hat: np.ndarray                  # (B,N,N) planner CO2 (unused by policy; kept for debug)
    CO2_true: np.ndarray                 # (B,N,N) truth CO2

    J_cost_int: np.ndarray               # (B,N,N) int64 planning costs (unused by policy; kept for debug)

    m_true: np.ndarray                   # (B,) hidden multiplier

    blockage_bin: int
    blocked_arcs: List[Tuple[int, int]]

    rain_mask: np.ndarray
    rho_TT: float
    rho_CO2: float

    bin_minutes: float
    service_time_min: float


def current_bin_from_time(elapsed_min: float, start_bin: int, B: int, bin_minutes: float) -> int:
    b = start_bin + int(elapsed_min // bin_minutes)
    return min(max(b, 0), B - 1)


def run_policy(
    policy_name: str,
    ctx: EpisodeCtx,
    lam: float,
    time_limit_ms: int,
    start_bin: int,
    *,
    SCALE: int,
    BIG_M_cost_int: int,
    emissions_params: Tuple[float, float, float, float],
    m_init: np.ndarray,
    ewma_alpha: float = EWMA_ALPHA,
    # Gate B knobs (used only by B3_GateReplan)
    eta: float = 1.0,
    probe_ms: int = 50,
) -> Dict[str, Any]:
    """
    Policies:
      - B0_PlanOnce
      - B2_BlockageReplan
      - B1_AlwaysReplan
      - B3_GateReplan (Gate B):
          Run a cheap PROBE solve (probe_ms) to estimate predicted gain vs current queue.
          Do FULL replan only if gain_hat > eta * planning_cost_hat_full.

    Conventions:
      - n_replans counts FULL plans used for execution (initial + full replans).
      - Probe solves count toward solve_ms_total + planning_wait_min, but NOT n_replans.
      - n_solve_calls counts all solver calls including probes.
    """
    N = int(ctx.dist_km.shape[0])
    depot = 0
    customers = list(range(1, N))
    B = int(ctx.TT_data_min.shape[0])

    m_hat_by_bin = np.array(m_init, dtype=np.float32).copy()
    if m_hat_by_bin.size != B:
        m_hat_by_bin = np.ones(B, dtype=np.float32)

    elapsed_min = 0.0
    travel_min = 0.0
    traffic_wait_min = 0.0
    planning_wait_min = 0.0
    service_min = 0.0
    CO2_total = 0.0
    used_blocked_arc = 0

    rel_err_list: List[float] = []

    remaining = customers.copy()
    cur = depot
    replanned_for_blockage = False

    # FULL replans used for execution
    n_replans = 0

    # All solver calls (including probe)
    solve_ms_list: List[float] = []
    solve_ms_total = 0.0

    # Gate B bookkeeping
    n_gate_probes = 0
    n_gate_full_replans = 0
    gate_gain_hat_list: List[float] = []

    blocked_set = set((int(u), int(v)) for (u, v) in ctx.blocked_arcs)

    TT_true_min = ctx.TT_true_min
    CO2_true = ctx.CO2_true

    def _apply_planning_overhead(solve_ms: float) -> None:
        nonlocal elapsed_min, planning_wait_min, solve_ms_total
        sm = float(solve_ms)
        solve_ms_total += sm
        w_min = sm / 60000.0
        planning_wait_min += w_min
        elapsed_min += w_min

    def _current_bin() -> int:
        return current_bin_from_time(elapsed_min, start_bin, B, ctx.bin_minutes)

    def _build_cost_mat_for_bin(b: int) -> np.ndarray:
        return compute_cost_mat_int_for_bin(
            TT_data_min_b=ctx.TT_data_min[b],
            dist_km=ctx.dist_km,
            rain_on=bool(ctx.rain_mask[b]),
            rho_TT=float(ctx.rho_TT),
            rho_CO2=float(ctx.rho_CO2),
            m_hat_b=float(m_hat_by_bin[b]),
            lam=float(lam),
            SCALE=int(SCALE),
            BIG_M_cost_int=int(BIG_M_cost_int),
            blockage_bin=int(ctx.blockage_bin),
            b=int(b),
            blocked_arcs=ctx.blocked_arcs,
            emissions_params=emissions_params,
        )

    def _tt_hat_for_arc(b: int, i: int, j: int) -> float:
        rain_factor = (1.0 + float(ctx.rho_TT)) if bool(ctx.rain_mask[b]) else 1.0
        return float(ctx.TT_data_min[b, i, j]) * float(rain_factor) * float(m_hat_by_bin[b])

    def _ewma_update_from_leg(b: int, i: int, j: int, tt_obs: float) -> None:
        rain_factor = (1.0 + float(ctx.rho_TT)) if bool(ctx.rain_mask[b]) else 1.0
        denom = float(ctx.TT_data_min[b, i, j]) * float(rain_factor)
        denom = max(denom, 1e-9)
        m_obs = float(tt_obs) / denom
        m_hat_by_bin[b] = (1.0 - float(ewma_alpha)) * float(m_hat_by_bin[b]) + float(ewma_alpha) * float(m_obs)

    def _J_hat_of_path(cost_mat_int: np.ndarray, path_nodes: List[int]) -> float:
        if not path_nodes or len(path_nodes) < 2:
            return 0.0
        s = 0.0
        for a, b_ in zip(path_nodes[:-1], path_nodes[1:]):
            s += float(cost_mat_int[int(a), int(b_)]) / float(SCALE)
        return float(s)

    def _J_hat_of_queue(cost_mat_int: np.ndarray, cur_node: int, queue: List[int]) -> float:
        if not queue:
            return 0.0
        path = [int(cur_node)] + [int(x) for x in queue]
        if path[-1] != depot:
            path.append(depot)
        return _J_hat_of_path(cost_mat_int, path)

    # -----------------------
    # Initial plan
    # -----------------------
    b0 = _current_bin()
    cost_mat0 = _build_cost_mat_for_bin(b0)
    route, _, solve_ms = solve_subproblem(cost_mat0, cur, depot, remaining, int(time_limit_ms))

    n_replans += 1
    solve_ms_list.append(float(solve_ms))
    _apply_planning_overhead(float(solve_ms))

    if not route or route[0] != cur or route[-1] != depot:
        return {"ok": False, "policy": policy_name, "seed": int(ctx.seed), "reason": "no_solution_initial", "n_replans": int(n_replans)}

    plan_queue = route[1:]  # includes depot at end

    # -----------------------
    # Execute
    # -----------------------
    while True:
        if cur == depot and not remaining:
            break

        b = _current_bin()
        trigger_blockage = (b >= ctx.blockage_bin) and (not replanned_for_blockage)

        do_replan = False

        if policy_name == "B1_AlwaysReplan":
            do_replan = True

        elif policy_name == "B2_BlockageReplan":
            do_replan = bool(trigger_blockage)

        elif policy_name == "B3_GateReplan":
            # Gate B: probe solve to estimate predicted gain vs current queue
            cost_mat_int = _build_cost_mat_for_bin(b)
            J_continue_hat = _J_hat_of_queue(cost_mat_int, cur, plan_queue)

            probe_cap = int(max(1, min(int(probe_ms), int(time_limit_ms))))
            route_probe, _, solve_ms_probe = solve_subproblem(cost_mat_int, cur, depot, remaining, probe_cap)

            # probe consumes time but does NOT count as a "replan used for execution"
            n_gate_probes += 1
            solve_ms_list.append(float(solve_ms_probe))
            _apply_planning_overhead(float(solve_ms_probe))

            if route_probe and route_probe[0] == cur and route_probe[-1] == depot:
                J_probe_hat = _J_hat_of_path(cost_mat_int, route_probe)
                gain_hat = float(J_continue_hat - J_probe_hat)
            else:
                gain_hat = -1e9

            gate_gain_hat_list.append(float(gain_hat))

            # planning-cost proxy in J units (simple, consistent with your Week 4 writeup)
            planning_cost_hat_full = float(lam) * (float(time_limit_ms) / 60000.0)
            do_replan = bool(gain_hat > float(eta) * planning_cost_hat_full)

        else:
            do_replan = False  # B0_PlanOnce

        if do_replan:
            # Recompute bin after any probe overhead (Gate B), because elapsed time advanced.
            b2 = _current_bin()
            cost_mat_int2 = _build_cost_mat_for_bin(b2)

            route, _, solve_ms_full = solve_subproblem(cost_mat_int2, cur, depot, remaining, int(time_limit_ms))

            n_replans += 1
            if policy_name == "B3_GateReplan":
                n_gate_full_replans += 1

            solve_ms_list.append(float(solve_ms_full))
            _apply_planning_overhead(float(solve_ms_full))

            if not route or route[0] != cur or route[-1] != depot:
                return {"ok": False, "policy": policy_name, "seed": int(ctx.seed), "reason": "no_solution_replan", "n_replans": int(n_replans)}

            plan_queue = route[1:]

            if policy_name == "B2_BlockageReplan" and trigger_blockage:
                replanned_for_blockage = True

        if not plan_queue:
            return {"ok": False, "policy": policy_name, "seed": int(ctx.seed), "reason": "empty_plan_queue", "n_replans": int(n_replans)}

        nxt = int(plan_queue.pop(0))

        # Blockage wait
        b = _current_bin()
        is_blocked_now = (b == ctx.blockage_bin) and ((cur, nxt) in blocked_set)
        if is_blocked_now:
            into_bin = elapsed_min % ctx.bin_minutes
            w = (ctx.bin_minutes - into_bin) if into_bin > 1e-9 else ctx.bin_minutes
            traffic_wait_min += w
            elapsed_min += w
            used_blocked_arc += 1
            b = _current_bin()

        # Travel (truth)
        tt_hat = _tt_hat_for_arc(b, cur, nxt)
        tt_obs = float(TT_true_min[b, cur, nxt])
        co2_obs = float(CO2_true[b, cur, nxt])

        e = abs(tt_obs - tt_hat) / max(tt_hat, 1e-9)
        rel_err_list.append(float(e))

        travel_min += tt_obs
        elapsed_min += tt_obs
        CO2_total += co2_obs

        # EWMA update
        _ewma_update_from_leg(b, cur, nxt, tt_obs)

        cur = nxt

        # Service
        if cur != depot:
            if cur in remaining:
                remaining.remove(cur)
            service_min += ctx.service_time_min
            elapsed_min += ctx.service_time_min

    # Totals
    wait_min = traffic_wait_min + planning_wait_min
    exec_time_min = travel_min + traffic_wait_min + service_min
    wall_time_min = exec_time_min + planning_wait_min

    J_exec = CO2_total + lam * exec_time_min
    J_wall = CO2_total + lam * wall_time_min

    rel_pred_err_mean = float(np.mean(rel_err_list)) if rel_err_list else 0.0
    rel_pred_err_p95 = float(np.percentile(rel_err_list, 95)) if rel_err_list else 0.0
    rel_pred_err_max = float(np.max(rel_err_list)) if rel_err_list else 0.0

    solve_ms_arr = np.array(solve_ms_list, dtype=np.float64) if solve_ms_list else np.array([np.nan])
    gate_gain_hat_mean = float(np.mean(gate_gain_hat_list)) if gate_gain_hat_list else 0.0

    return {
        "ok": True,
        "policy": policy_name,
        "seed": int(ctx.seed),

        "travel_min": float(travel_min),
        "traffic_wait_min": float(traffic_wait_min),
        "planning_wait_min": float(planning_wait_min),
        "wait_min": float(wait_min),
        "service_min": float(service_min),

        "exec_time_min": float(exec_time_min),
        "wall_time_min": float(wall_time_min),
        "elapsed_min": float(elapsed_min),
        "total_route_time_min": float(wall_time_min),

        "CO2_total": float(CO2_total),

        "J_exec": float(J_exec),
        "J_wall": float(J_wall),
        "J_total": float(J_wall),

        "delay_flag": int(traffic_wait_min > 1e-9),
        "delay_flag_planning": int(planning_wait_min > 1e-9),
        "delay_flag_any": int(wait_min > 1e-9),

        "used_blocked_arc": int(used_blocked_arc),

        # replans used for execution
        "n_replans": int(n_replans),

        # all solver calls (including probes)
        "n_solve_calls": int(len(solve_ms_list)),

        "solve_ms_total": float(solve_ms_total),
        "solve_ms_p50": float(np.percentile(solve_ms_arr, 50)),
        "solve_ms_p95": float(np.percentile(solve_ms_arr, 95)),
        "solve_ms_max": float(np.max(solve_ms_arr)),

        "rel_pred_err_mean": float(rel_pred_err_mean),
        "rel_pred_err_p95": float(rel_pred_err_p95),
        "rel_pred_err_max": float(rel_pred_err_max),

        # Gate B fields (0 if not Gate)
        "eta": float(eta),
        "probe_ms": int(probe_ms),
        "n_gate_probes": int(n_gate_probes),
        "n_gate_full_replans": int(n_gate_full_replans),
        "gate_gain_hat_mean": float(gate_gain_hat_mean),
    }


def build_episode_ctx(
    seed: int,
    ep: Dict[str, Any],
    ingest_cfg: Dict[str, Any],
    lam: float,
    SCALE: int,
    BIG_M_cost_int: int,
    blockage_bin: int,
    disable_rain: bool,
    n_blockages: int,
    early_frac: float,
) -> EpisodeCtx:
    node_ids = ep["node_ids"]
    dist_km = ep["dist_km"].astype(np.float32)

    TT_data_min = ep["TT_data_min"].astype(np.float32)
    B = int(TT_data_min.shape[0])

    events = vrp_lib.generate_events_for_episode(
        seed=seed,
        TT_data_min=TT_data_min,
        k_blocked=int(n_blockages),
        early_frac=float(early_frac),
    )

    if disable_rain:
        rain_mask = np.zeros(B, dtype=bool)
        rho_TT = 0.0
        rho_CO2 = 0.0
    else:
        rain_mask = events.rain_mask.astype(bool)
        rho_TT = float(events.rho_TT)
        rho_CO2 = float(events.rho_CO2)

    TT_rain_min = vrp_lib.apply_rain_to_TT(TT_data_min, rain_mask, float(rho_TT)).astype(np.float32)

    m_true = sample_m_true(B, seed=seed).astype(np.float32)
    TT_true_min = (TT_rain_min * m_true[:, None, None]).astype(np.float32)

    # baseline TT_hat for logging/debug (policy builds its own costs from TT_data + m_hat_by_bin)
    TT_hat_min = TT_rain_min.copy().astype(np.float32)

    alpha, beta, gamma, delta = get_emissions_params(ingest_cfg)

    CO2_hat = vrp_lib.meet_emissions_proxy(dist_km, TT_hat_min, alpha, beta, gamma, delta).astype(np.float32)
    CO2_hat = vrp_lib.apply_rain_to_CO2(CO2_hat, rain_mask, float(rho_CO2)).astype(np.float32)

    CO2_true = vrp_lib.meet_emissions_proxy(dist_km, TT_true_min, alpha, beta, gamma, delta).astype(np.float32)
    CO2_true = vrp_lib.apply_rain_to_CO2(CO2_true, rain_mask, float(rho_CO2)).astype(np.float32)

    costs = vrp_lib.build_int_costs(
        TT_hat_min=TT_hat_min,
        CO2_hat=CO2_hat,
        lam=float(lam),
        SCALE=int(SCALE),
        blockage_bin=int(blockage_bin),
        BIG_M_cost_int=int(BIG_M_cost_int),
        blocked_arcs=events.blocked_arcs,
    )

    bin_minutes = float(ingest_cfg.get("bin_minutes", 60.0))
    service_time_min = float(ingest_cfg.get("service_time_min", 2.0))

    blocked_arcs_list = [(int(u), int(v)) for (u, v) in np.asarray(events.blocked_arcs).reshape(-1, 2)]

    return EpisodeCtx(
        seed=int(seed),
        node_ids=node_ids,
        dist_km=dist_km,

        TT_data_min=TT_data_min,
        TT_hat_min=TT_hat_min,
        TT_true_min=TT_true_min,

        CO2_hat=CO2_hat,
        CO2_true=CO2_true,

        J_cost_int=costs["J_cost_int"].astype(np.int64),

        m_true=m_true,

        blockage_bin=int(blockage_bin),
        blocked_arcs=blocked_arcs_list,

        rain_mask=rain_mask.astype(bool),
        rho_TT=float(rho_TT),
        rho_CO2=float(rho_CO2),

        bin_minutes=float(bin_minutes),
        service_time_min=float(service_time_min),
    )


def get_or_compute_binmean_m(
    *,
    root: Path,
    ingest_cfg: Dict[str, Any],
    seeds_cfg: Dict[str, Any],
    base_dir: Path,
    cache_path: Path,
) -> np.ndarray:
    """
    TRAIN-only bin-mean multiplier for initialization.
    """
    if cache_path.exists():
        d = load_json(cache_path)
        arr = np.array(d.get("binmean_m", []), dtype=np.float32)
        if arr.size > 0:
            return arr

    train_seeds = load_split_seeds_range(seeds_cfg, "TRAIN", base_dir)
    if not train_seeds:
        return np.ones(7, dtype=np.float32)

    m_list = []
    for s in train_seeds:
        p = episode_path(base_dir, "TRAIN", s)
        if not p.exists():
            continue
        ep = vrp_lib.load_episode_npz(p)
        B = int(ep["TT_data_min"].shape[0])
        m_true = sample_m_true(B, seed=s)
        m_list.append(m_true)

    if not m_list:
        return np.ones(7, dtype=np.float32)

    M = np.stack(m_list, axis=0)
    binmean_m = M.mean(axis=0).astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump({"binmean_m": binmean_m.tolist()}, f, indent=2)

    return binmean_m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="TEST", choices=["TRAIN", "VAL", "TEST"])
    ap.add_argument("--time_limit_ms", type=int, default=500)
    ap.add_argument("--start_bin", type=int, default=0)
    ap.add_argument("--blockage_bin", type=int, default=-1, help="Override blockage_bin (default: from ingest.json).")
    ap.add_argument("--n_blockages", type=int, default=3)
    ap.add_argument("--early_frac", type=float, default=0.6)
    ap.add_argument("--disable_rain", action="store_true")
    ap.add_argument("--max_seeds", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="data/processed/bench/digital_twin_eval_results")

    # Gate B knobs
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--probe_ms", type=int, default=50)
    ap.add_argument("--include_gate", action="store_true", help="Include B3_GateReplan (Gate B) in the CSV.")
    args = ap.parse_args()

    if args.n_blockages < 1:
        raise ValueError("--n_blockages must be >= 1")
    if not (0.0 < args.early_frac <= 1.0):
        raise ValueError("--early_frac must be in (0, 1]")
    if args.time_limit_ms <= 0:
        raise ValueError("--time_limit_ms must be > 0")

    root = repo_root()
    ingest_cfg = load_json(root / "configs" / "ingest.json")
    lam_cfg = load_json(root / "configs" / "lambda.json")
    seeds_cfg = load_json(root / "configs" / "seeds.json")

    base_dir = get_processed_base_dir(ingest_cfg, root)
    seeds = load_split_seeds_range(seeds_cfg, args.split, base_dir)
    if args.max_seeds and args.max_seeds > 0:
        seeds = seeds[: int(args.max_seeds)]

    lam = float(lam_cfg.get("lambda", lam_cfg.get("lam", 0.0)))
    if lam <= 0:
        raise ValueError("lambda must be > 0. Check configs/lambda.json")

    SCALE = int(ingest_cfg.get("SCALE", 1000))
    BIG_M_cost_int = int(ingest_cfg.get("BIG_M_cost_int", 1_000_000_000_000))

    blockage_bin_cfg = int(ingest_cfg.get("blockage_bin", 0))
    blockage_bin = int(args.blockage_bin) if args.blockage_bin >= 0 else blockage_bin_cfg

    binmean_cache = root / "configs" / "binmean_m.json"
    m_init = get_or_compute_binmean_m(
        root=root,
        ingest_cfg=ingest_cfg,
        seeds_cfg=seeds_cfg,
        base_dir=base_dir,
        cache_path=binmean_cache,
    )

    emissions_params = get_emissions_params(ingest_cfg)

    episodes: Dict[int, Dict[str, Any]] = {}
    for s in seeds:
        p = episode_path(base_dir, args.split, s)
        if not p.exists():
            raise FileNotFoundError(f"Missing episode file for seed {s}: {p}")
        episodes[s] = vrp_lib.load_episode_npz(p)

    policies = ["B0_PlanOnce", "B2_BlockageReplan", "B1_AlwaysReplan"]
    if args.include_gate:
        policies.append("B3_GateReplan")

    rows: List[Dict[str, Any]] = []

    out_dir = root / Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag_rain = "norain" if args.disable_rain else "rain"

    out_csv = out_dir / (
        f"twin_{args.split}_{tag_rain}"
        f"_cap{int(args.time_limit_ms)}"
        f"_startbin{int(args.start_bin)}"
        f"_blockbin{int(blockage_bin)}"
        f"_k{int(args.n_blockages)}"
        f"_ef{args.early_frac:.2f}.csv"
    )

    for s in seeds:
        ep = episodes[s]
        B = int(ep["TT_data_min"].shape[0])
        if not (0 <= args.start_bin < B):
            raise ValueError(f"--start_bin must be in [0..{B-1}] got {args.start_bin}")
        if not (0 <= blockage_bin < B):
            raise ValueError(f"blockage_bin must be in [0..{B-1}] got {blockage_bin}")

        ctx = build_episode_ctx(
            seed=s,
            ep=ep,
            ingest_cfg=ingest_cfg,
            lam=lam,
            SCALE=SCALE,
            BIG_M_cost_int=BIG_M_cost_int,
            blockage_bin=blockage_bin,
            disable_rain=bool(args.disable_rain),
            n_blockages=int(args.n_blockages),
            early_frac=float(args.early_frac),
        )

        for pol in policies:
            res = run_policy(
                policy_name=pol,
                ctx=ctx,
                lam=float(lam),
                time_limit_ms=int(args.time_limit_ms),
                start_bin=int(args.start_bin),
                SCALE=int(SCALE),
                BIG_M_cost_int=int(BIG_M_cost_int),
                emissions_params=emissions_params,
                m_init=m_init,
                ewma_alpha=EWMA_ALPHA,
                eta=float(args.eta),
                probe_ms=int(args.probe_ms),
            )

            res.update({
                "split": args.split,
                "rain": tag_rain,  # <-- for plotting
                "time_limit_ms": int(args.time_limit_ms),
                "start_bin": int(args.start_bin),
                "blockage_bin": int(blockage_bin),
                "n_blockages": int(args.n_blockages),
                "early_frac": float(args.early_frac),

                "blocked_arcs": ";".join([f"{u}->{v}" for (u, v) in ctx.blocked_arcs]),
                "rain_mask": "".join("1" if bool(x) else "0" for x in ctx.rain_mask.tolist()),
                "rho_TT": float(ctx.rho_TT),
                "rho_CO2": float(ctx.rho_CO2),
                "m_true": ";".join([f"{x:.4f}" for x in ctx.m_true.tolist()]),
            })
            rows.append(res)

    # -----------------------------
    # Add per-seed deltas vs B0 (commit to J_wall)
    # -----------------------------
    b0_by_seed: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        if r.get("ok") and r.get("policy") == "B0_PlanOnce":
            b0_by_seed[int(r["seed"])] = r

    for r in rows:
        if not r.get("ok"):
            continue
        seed = int(r["seed"])
        b0 = b0_by_seed.get(seed)
        if b0 is None:
            continue

        r["delta_J_wall_vs_B0"] = float(r["J_wall"]) - float(b0["J_wall"])
        r["delta_wall_time_vs_B0"] = float(r["wall_time_min"]) - float(b0["wall_time_min"])
        r["delta_wait_vs_B0"] = float(r["wait_min"]) - float(b0["wait_min"])
        r["extra_replans_vs_B0"] = int(r["n_replans"]) - int(b0["n_replans"])

        gain_wall = float(b0["J_wall"]) - float(r["J_wall"])
        extra_replans = max(0, int(r["extra_replans_vs_B0"]))
        r["J_gain_wall_vs_B0"] = float(gain_wall)
        r["J_gain_wall_per_extra_replan"] = float(gain_wall / extra_replans) if extra_replans > 0 else 0.0

        extra_plan_sec = max(0.0, (float(r["solve_ms_total"]) - float(b0["solve_ms_total"])) / 1000.0)
        r["extra_plan_sec_vs_B0"] = float(extra_plan_sec)
        r["J_gain_wall_per_extra_plan_sec"] = float(gain_wall / extra_plan_sec) if extra_plan_sec > 1e-12 else 0.0

        r["delta_J_exec_vs_B0"] = float(r["J_exec"]) - float(b0["J_exec"])
        gain_exec = float(b0["J_exec"]) - float(r["J_exec"])
        r["J_gain_exec_vs_B0"] = float(gain_exec)
        r["J_gain_exec_per_extra_plan_sec"] = float(gain_exec / extra_plan_sec) if extra_plan_sec > 1e-12 else 0.0

    # write CSV
    fieldnames = sorted({k for rr in rows for k in rr.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rr in rows:
            w.writerow(rr)

    print(f"WROTE: {out_csv}")

    ok_rows = [rr for rr in rows if rr.get("ok")]
    if not ok_rows:
        print("No successful runs.")
        return

    def summarize(metric: str) -> None:
        for pol in policies:
            vals = [float(rr[metric]) for rr in ok_rows if rr["policy"] == pol and metric in rr]
            if vals:
                arr = np.array(vals, dtype=np.float64)
                print(f"{metric:>22} | {pol:>18}: mean={arr.mean():.3f}  median={np.median(arr):.3f}  p95={np.percentile(arr,95):.3f}")

    def summarize_rate(metric: str) -> None:
        for pol in policies:
            vals = [int(rr[metric]) for rr in ok_rows if rr["policy"] == pol and metric in rr]
            if vals:
                arr = np.array(vals, dtype=np.int32)
                print(f"{metric:>22} | {pol:>18}: rate={arr.mean():.3f}")

    print("SUMMARY (ok rows only):")
    summarize("J_wall")
    summarize("J_exec")
    summarize("CO2_total")
    summarize("travel_min")
    summarize("traffic_wait_min")
    summarize("planning_wait_min")
    summarize("wait_min")
    summarize("service_min")
    summarize("exec_time_min")
    summarize("wall_time_min")
    summarize("n_replans")
    summarize("n_solve_calls")
    summarize("solve_ms_total")
    summarize("rel_pred_err_mean")
    summarize("rel_pred_err_p95")
    summarize_rate("delay_flag")
    summarize_rate("delay_flag_planning")
    summarize_rate("used_blocked_arc")

    if "B3_GateReplan" in policies:
        summarize("n_gate_probes")
        summarize("n_gate_full_replans")
        summarize("gate_gain_hat_mean")

    summarize("delta_J_wall_vs_B0")
    summarize("delta_wall_time_vs_B0")
    summarize("J_gain_wall_vs_B0")
    summarize("J_gain_wall_per_extra_replan")
    summarize("extra_plan_sec_vs_B0")
    summarize("J_gain_wall_per_extra_plan_sec")


if __name__ == "__main__":
    main()
