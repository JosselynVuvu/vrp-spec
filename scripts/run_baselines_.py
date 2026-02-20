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
# Simulation / policy
# -----------------------------
@dataclass
class EpisodeCtx:
    seed: int
    node_ids: np.ndarray
    dist_km: np.ndarray          # (N,N)
    TT_base_min: np.ndarray      # (B,N,N)
    TT_hat_min: np.ndarray       # (B,N,N) (rain observable)
    CO2_hat: np.ndarray          # (B,N,N) (rain observable)
    J_cost_int: np.ndarray       # (B,N,N) int64 planning cost
    blockage_bin: int
    blocked_arcs: List[Tuple[int, int]]  # K blocked arcs (u->v)
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
) -> Dict[str, Any]:
    """
    Policies:
      - B0_PlanOnce: solve once at start; no replans.
      - B2_BlockageReplan: solve once at start; replan once at first arrival in/after blockage bin.
      - B1_AlwaysReplan: replan at every step (customer-to-customer decisions).

    Accident model:
      - If attempting a blocked arc during blockage_bin, WAIT until the bin ends (engine off),
        then traverse the arc in the next bin (so TT/CO2 come from that next bin).

    Planning overhead model (numeric):
      - Each OR-Tools solve consumes solve_ms.
      - Convert solve_ms -> minutes and add to elapsed_min as planning_wait_min (vehicle idle).
      - This can shift bins and affect TT/CO2 for later legs (intended).
    """
    N = int(ctx.dist_km.shape[0])
    depot = 0
    customers = list(range(1, N))

    elapsed_min = 0.0
    travel_min = 0.0
    traffic_wait_min = 0.0     # blockage waits only
    planning_wait_min = 0.0    # solver latency
    service_min = 0.0
    CO2_total = 0.0
    used_blocked_arc = 0

    remaining = customers.copy()
    cur = depot
    replanned_for_blockage = False

    n_replans = 0
    solve_ms_list: List[float] = []
    solve_ms_total = 0.0

    B = int(ctx.TT_hat_min.shape[0])
    blocked_set = set((int(u), int(v)) for (u, v) in ctx.blocked_arcs)

    def _apply_planning_overhead(solve_ms: float) -> None:
        nonlocal elapsed_min, planning_wait_min, solve_ms_total
        sm = float(solve_ms)
        solve_ms_total += sm
        w_min = sm / 60000.0
        planning_wait_min += w_min
        elapsed_min += w_min

    # -----------------------
    # Initial plan (+ overhead)
    # -----------------------
    b0 = current_bin_from_time(elapsed_min, start_bin, B, ctx.bin_minutes)
    cost_mat = ctx.J_cost_int[b0]
    route, _, solve_ms = solve_subproblem(cost_mat, cur, depot, remaining, time_limit_ms)
    n_replans += 1
    solve_ms_list.append(float(solve_ms))
    _apply_planning_overhead(solve_ms)

    if not route or route[0] != cur or route[-1] != depot:
        return {
            "ok": False,
            "policy": policy_name,
            "seed": int(ctx.seed),
            "reason": "no_solution_initial",
            "n_replans": int(n_replans),
        }

    plan_queue = route[1:]

    # -----------------------
    # Execute
    # -----------------------
    while True:
        if cur == depot and not remaining:
            break

        b = current_bin_from_time(elapsed_min, start_bin, B, ctx.bin_minutes)
        trigger_blockage = (b >= ctx.blockage_bin) and (not replanned_for_blockage)

        if policy_name == "B1_AlwaysReplan":
            do_replan = True
        elif policy_name == "B2_BlockageReplan":
            do_replan = trigger_blockage
        else:
            do_replan = False

        if do_replan:
            cost_mat = ctx.J_cost_int[b]
            route, _, solve_ms = solve_subproblem(cost_mat, cur, depot, remaining, time_limit_ms)
            n_replans += 1
            solve_ms_list.append(float(solve_ms))
            _apply_planning_overhead(solve_ms)

            if not route or route[0] != cur or route[-1] != depot:
                return {
                    "ok": False,
                    "policy": policy_name,
                    "seed": int(ctx.seed),
                    "reason": "no_solution_replan",
                    "n_replans": int(n_replans),
                }

            plan_queue = route[1:]
            if trigger_blockage:
                replanned_for_blockage = True

        if not plan_queue:
            return {
                "ok": False,
                "policy": policy_name,
                "seed": int(ctx.seed),
                "reason": "empty_plan_queue",
                "n_replans": int(n_replans),
            }

        nxt = plan_queue.pop(0)

        # -----------------------
        # Blockage wait (traffic)
        # -----------------------
        b = current_bin_from_time(elapsed_min, start_bin, B, ctx.bin_minutes)
        is_blocked_now = (b == ctx.blockage_bin) and ((cur, nxt) in blocked_set)

        if is_blocked_now:
            into_bin = elapsed_min % ctx.bin_minutes
            w = (ctx.bin_minutes - into_bin) if into_bin > 1e-9 else ctx.bin_minutes
            traffic_wait_min += w
            elapsed_min += w  # engine off during wait
            used_blocked_arc += 1
            b = current_bin_from_time(elapsed_min, start_bin, B, ctx.bin_minutes)

        # -----------------------
        # Travel (TT/CO2 from bin b)
        # -----------------------
        tt = float(ctx.TT_hat_min[b, cur, nxt])
        co2 = float(ctx.CO2_hat[b, cur, nxt])

        travel_min += tt
        elapsed_min += tt
        CO2_total += co2
        cur = nxt

        # -----------------------
        # Service
        # -----------------------
        if cur != depot:
            if cur in remaining:
                remaining.remove(cur)
            service_min += ctx.service_time_min
            elapsed_min += ctx.service_time_min

    # Totals
    wait_min = traffic_wait_min + planning_wait_min

    exec_time_min = travel_min + traffic_wait_min + service_min
    wall_time_min = exec_time_min + planning_wait_min  # includes planning overhead

    total_route_time_min = wall_time_min

    # Objectives
    J_exec = CO2_total + lam * exec_time_min
    J_wall = CO2_total + lam * wall_time_min
    J_total = J_wall  # thesis commits to wall-clock objective

    # Flags
    delay_flag = int(traffic_wait_min > 1e-9)          # traffic-only
    delay_flag_planning = int(planning_wait_min > 1e-9)
    delay_flag_any = int(wait_min > 1e-9)

    solve_ms_arr = np.array(solve_ms_list, dtype=np.float64) if solve_ms_list else np.array([np.nan])

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
        "total_route_time_min": float(total_route_time_min),

        "CO2_total": float(CO2_total),

        "J_exec": float(J_exec),
        "J_wall": float(J_wall),
        "J_total": float(J_total),

        "delay_flag": int(delay_flag),
        "delay_flag_planning": int(delay_flag_planning),
        "delay_flag_any": int(delay_flag_any),

        "used_blocked_arc": int(used_blocked_arc),
        "n_replans": int(n_replans),

        "solve_ms_total": float(solve_ms_total),
        "solve_ms_p50": float(np.percentile(solve_ms_arr, 50)),
        "solve_ms_p95": float(np.percentile(solve_ms_arr, 95)),
        "solve_ms_max": float(np.max(solve_ms_arr)),
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
    TT_base_min = ep["TT_data_min"].astype(np.float32)
    B = int(TT_base_min.shape[0])

    events = vrp_lib.generate_events_for_episode(
        seed=seed,
        TT_data_min=TT_base_min,
        k_blocked=int(n_blockages),
        early_frac=float(early_frac),
    )

    if disable_rain:
        rain_mask = np.zeros(B, dtype=bool)
        rho_TT = 0.0
        rho_CO2 = 0.0
    else:
        rain_mask = events.rain_mask
        rho_TT = float(events.rho_TT)
        rho_CO2 = float(events.rho_CO2)

    TT_hat_min = vrp_lib.apply_rain_to_TT(TT_base_min, rain_mask, rho_TT)

    alpha, beta, gamma, delta = get_emissions_params(ingest_cfg)
    CO2_hat = vrp_lib.meet_emissions_proxy(dist_km, TT_hat_min, alpha, beta, gamma, delta)
    CO2_hat = vrp_lib.apply_rain_to_CO2(CO2_hat, rain_mask, rho_CO2)

    costs = vrp_lib.build_int_costs(
        TT_hat_min=TT_hat_min,
        CO2_hat=CO2_hat,
        lam=lam,
        SCALE=SCALE,
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
        TT_base_min=TT_base_min,
        TT_hat_min=TT_hat_min.astype(np.float32),
        CO2_hat=CO2_hat.astype(np.float32),
        J_cost_int=costs["J_cost_int"].astype(np.int64),
        blockage_bin=int(blockage_bin),
        blocked_arcs=blocked_arcs_list,
        rain_mask=rain_mask.astype(bool),
        rho_TT=float(rho_TT),
        rho_CO2=float(rho_CO2),
        bin_minutes=bin_minutes,
        service_time_min=service_time_min,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="TEST", choices=["TRAIN", "VAL", "TEST"])
    ap.add_argument("--time_limit_ms", type=int, default=500)
    ap.add_argument("--start_bin", type=int, default=0)
    ap.add_argument("--blockage_bin", type=int, default=-1, help="Override blockage_bin (default: from ingest.json).")
    ap.add_argument("--n_blockages", type=int, default=3, help="Number of blocked arcs (K) in the blockage bin.")
    ap.add_argument("--early_frac", type=float, default=0.6, help="Fraction of early-route arcs eligible for blockage.")
    ap.add_argument("--disable_rain", action="store_true", help="Ablation: disable rain.")
    ap.add_argument("--max_seeds", type=int, default=0, help="If >0, only run first N seeds.")
    args = ap.parse_args()

    if args.n_blockages < 1:
        raise ValueError("--n_blockages must be >= 1")
    if not (0.0 < args.early_frac <= 1.0):
        raise ValueError("--early_frac must be in (0, 1]")

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

    # preload episodes
    episodes: Dict[int, Dict[str, Any]] = {}
    for s in seeds:
        p = episode_path(base_dir, args.split, s)
        if not p.exists():
            raise FileNotFoundError(f"Missing episode file for seed {s}: {p}")
        episodes[s] = vrp_lib.load_episode_npz(p)

    policies = ["B0_PlanOnce", "B2_BlockageReplan", "B1_AlwaysReplan"]
    rows: List[Dict[str, Any]] = []

    out_dir = root / "data" / "processed" / "bench" / "week3_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag_rain = "norain" if args.disable_rain else "rain"
    out_csv = out_dir / (
        f"baselines_{args.split}_{tag_rain}"
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
                lam=lam,
                time_limit_ms=int(args.time_limit_ms),
                start_bin=int(args.start_bin),
            )
            res.update({
                "split": args.split,
                "time_limit_ms": int(args.time_limit_ms),
                "start_bin": int(args.start_bin),
                "blockage_bin": int(blockage_bin),
                "n_blockages": int(args.n_blockages),
                "early_frac": float(args.early_frac),
                "blocked_arcs": ";".join([f"{u}->{v}" for (u, v) in ctx.blocked_arcs]),
                "rain_mask": "".join("1" if bool(x) else "0" for x in ctx.rain_mask.tolist()),
                "rho_TT": float(ctx.rho_TT),
                "rho_CO2": float(ctx.rho_CO2),
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

        # Wall objective deltas (primary for thesis)
        r["delta_J_wall_vs_B0"] = float(r["J_wall"]) - float(b0["J_wall"])
        r["delta_wall_time_vs_B0"] = float(r["wall_time_min"]) - float(b0["wall_time_min"])
        r["delta_wait_vs_B0"] = float(r["wait_min"]) - float(b0["wait_min"])
        r["extra_replans_vs_B0"] = int(r["n_replans"]) - int(b0["n_replans"])

        gain_wall = float(b0["J_wall"]) - float(r["J_wall"])  # positive = improvement
        extra_replans = max(0, int(r["extra_replans_vs_B0"]))
        r["J_gain_wall_vs_B0"] = float(gain_wall)
        r["J_gain_wall_per_extra_replan"] = float(gain_wall / extra_replans) if extra_replans > 0 else 0.0

        # Planning efficiency (per extra planning second)
        extra_plan_sec = max(0.0, (float(r["solve_ms_total"]) - float(b0["solve_ms_total"])) / 1000.0)
        r["extra_plan_sec_vs_B0"] = float(extra_plan_sec)
        r["J_gain_wall_per_extra_plan_sec"] = float(gain_wall / extra_plan_sec) if extra_plan_sec > 1e-12 else 0.0

        # (Optional) exec objective deltas (secondary)
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
    summarize("solve_ms_total")
    summarize_rate("delay_flag")
    summarize_rate("delay_flag_planning")
    summarize_rate("used_blocked_arc")

    # deltas vs B0 (primary = J_wall)
    summarize("delta_J_wall_vs_B0")
    summarize("delta_wall_time_vs_B0")
    summarize("J_gain_wall_vs_B0")
    summarize("J_gain_wall_per_extra_replan")
    summarize("extra_plan_sec_vs_B0")
    summarize("J_gain_wall_per_extra_plan_sec")


if __name__ == "__main__":
    main()
