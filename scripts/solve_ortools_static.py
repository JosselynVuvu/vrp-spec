# scripts/solve_ortools_static.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import scripts.vrp_lib as vrp_lib  # run via: py scripts/solve_ortools_static.py ...


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
    candidates = [
        ingest_cfg.get("processed_base_dir"),
        ingest_cfg.get("out_base_dir"),
        ingest_cfg.get("processed_dir"),
        ingest_cfg.get("base_out_dir"),
    ]
    for c in candidates:
        p = resolve_path(c, root)
        if p and p.exists():
            return p
    return root / "data" / "processed" / "vrptdt" / "berlin_500"


def episode_path(base_dir: Path, split: str, seed: int) -> Path:
    return base_dir / "episodes" / split.upper() / f"seed_{seed:03d}.npz"


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


def ortools_solve_tsp(cost_mat: np.ndarray, time_limit_ms: int) -> Tuple[List[int], int, float]:
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except Exception as e:
        raise RuntimeError("OR-Tools not installed. Run: py -m pip install ortools") from e

    n = int(cost_mat.shape[0])
    depot = 0

    manager = pywrapcp.RoutingIndexManager(n, 1, depot)
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


def _print_blocked_costs(label: str, mat: np.ndarray, blocked_list: List[Tuple[int, int]]) -> None:
    print(label)
    if not blocked_list:
        print("  (none)")
        return
    for (u, v) in blocked_list:
        print(f"  {u}->{v}: {int(mat[u, v])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="TEST", choices=["TRAIN", "VAL", "TEST"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--bin", type=int, default=0, help="Static planning bin (0..B-1).")
    ap.add_argument("--time_limit_ms", type=int, default=500, help="OR-Tools time cap per solve (ms).")
    ap.add_argument(
        "--show_patch_bin_costs",
        action="store_true",
        help="Also print blocked-arc costs in the blockage_bin (where BIG_M is applied), for clarity.",
    )
    args = ap.parse_args()

    root = repo_root()
    ingest_cfg = load_json(root / "configs" / "ingest.json")
    lam_cfg = load_json(root / "configs" / "lambda.json")

    base_dir = get_processed_base_dir(ingest_cfg, root)
    ep_path = episode_path(base_dir, args.split, args.seed)
    if not ep_path.exists():
        raise FileNotFoundError(f"Episode not found: {ep_path}")

    ep = vrp_lib.load_episode_npz(ep_path)
    node_ids = ep["node_ids"]
    dist_km = ep["dist_km"]
    TT_base_min = ep["TT_data_min"]  # (B,N,N)

    B = int(TT_base_min.shape[0])
    if not (0 <= args.bin < B):
        raise ValueError(f"--bin must be in [0..{B-1}] but got {args.bin}")

    SCALE = int(ingest_cfg.get("SCALE", 1000))
    blockage_bin = int(ingest_cfg.get("blockage_bin", 1))
    if not (0 <= blockage_bin < B):
        raise ValueError(f"blockage_bin must be in [0..{B-1}] but got {blockage_bin}")

    BIG_M_cost_int = int(ingest_cfg.get("BIG_M_cost_int", 1_000_000_000_000))

    lam = float(lam_cfg.get("lambda", lam_cfg.get("lam", 0.0)))
    if lam <= 0:
        raise ValueError(f"lambda must be > 0. Check configs/lambda.json. Got {lam}")

    # Events (seeded): rain + blocked arcs
    events = vrp_lib.generate_events_for_episode(args.seed, TT_base_min)

    # Normalize blocked arcs early
    blocked_list: List[Tuple[int, int]] = []
    blocked_arcs_arr = np.empty((0, 2), dtype=np.int32)

    if hasattr(events, "blocked_arcs") and events.blocked_arcs is not None:
        blocked_arcs_arr = np.asarray(events.blocked_arcs, dtype=np.int32).reshape(-1, 2)
        blocked_list = [(int(u), int(v)) for (u, v) in blocked_arcs_arr]
    elif hasattr(events, "blocked_u") and hasattr(events, "blocked_v"):
        blocked_list = [(int(events.blocked_u), int(events.blocked_v))]
        blocked_arcs_arr = np.asarray(blocked_list, dtype=np.int32)

    # Rain observable: planner uses TT_hat including rain
    TT_hat_min = vrp_lib.apply_rain_to_TT(TT_base_min, events.rain_mask, events.rho_TT)

    # CO2 proxy (speed-dependent)
    alpha, beta, gamma, delta = get_emissions_params(ingest_cfg)
    CO2_hat = vrp_lib.meet_emissions_proxy(
        dist_km=dist_km,
        TT_min=TT_hat_min,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
    )
    CO2_hat = vrp_lib.apply_rain_to_CO2(CO2_hat, events.rain_mask, events.rho_CO2)

    # Integer planning costs + BIG_M on blocked arcs (planning cost only)
    costs = vrp_lib.build_int_costs(
        TT_hat_min=TT_hat_min,
        CO2_hat=CO2_hat,
        lam=lam,
        SCALE=SCALE,
        blockage_bin=blockage_bin,
        BIG_M_cost_int=BIG_M_cost_int,
        blocked_arcs=blocked_arcs_arr,
    )

    J_cost_int = costs["J_cost_int"]  # (B,N,N)

    # Matrices
    cost_mat_plan = J_cost_int[args.bin]
    cost_mat_patch = J_cost_int[blockage_bin]

    print(f"EPISODE: {ep_path}")
    print(f"N={len(node_ids)} (incl depot), B={B}, bin={args.bin}, time_cap={args.time_limit_ms}ms")
    print(f"lambda={lam:.6f} SCALE={SCALE} blockage_bin={blockage_bin} BIG_M_cost_int={BIG_M_cost_int}")
    print(f"rain_mask={events.rain_mask.astype(int).tolist()} rho_TT={events.rho_TT} rho_CO2={events.rho_CO2}")
    print(f"blocked_arcs(k={len(blocked_list)}): {blocked_list}")

    print(f"PLANNING MATRIX BIN = {args.bin} | BIG_M PATCH BIN = {blockage_bin}")
    if args.bin != blockage_bin:
        print("NOTE: BIG_M is applied only at blockage_bin, so solving with a different --bin will not reflect blockage.")
    else:
        print("NOTE: This solve uses blockage_bin costs, so blocked arcs should show BIG_M.")

    _print_blocked_costs("Blocked arc costs in PLANNING bin:", cost_mat_plan, blocked_list)
    if args.show_patch_bin_costs and blockage_bin != args.bin:
        _print_blocked_costs("Blocked arc costs in PATCH (blockage_bin):", cost_mat_patch, blocked_list)

    # Solve with OR-Tools on the chosen planning bin
    route, obj, solve_ms = ortools_solve_tsp(cost_mat_plan, args.time_limit_ms)

    if not route:
        print(f"NO SOLUTION (solve_ms={solve_ms:.3f}ms)")
        return

    def to_id(x):
        x = x.item() if hasattr(x, "item") else x
        return str(x)

    route_node_ids = [to_id(node_ids[i]) for i in route]
    print(f"SOLVE_MS: {solve_ms:.3f}")
    print(f"OBJ(J_cost_int): {obj}")
    print("ROUTE (indices):", route)
    print("ROUTE (node_ids):", route_node_ids)

    # Reporting (static approx)
    service_time_min = float(ingest_cfg.get("service_time_min", 2.0))
    TT_bin = TT_hat_min[args.bin]
    travel_min = float(sum(float(TT_bin[a, b]) for a, b in zip(route[:-1], route[1:])))
    n_customers = len(route) - 2
    total_min = travel_min + n_customers * service_time_min
    print(
        f"REPORT (static bin approx): travel_min={travel_min:.2f}, "
        f"service_min={n_customers*service_time_min:.2f}, total_min={total_min:.2f}"
    )


if __name__ == "__main__":
    main()
