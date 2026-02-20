# scripts/solve_ortools_static.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Week 2 utilities (you already committed these)
import week2_lib  # same folder import works when running: py scripts/...


def repo_root() -> Path:
    # scripts/ -> repo root
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
    """
    Best-effort: use whatever your configs/ingest.json already has.
    Falls back to the known default you used in Week 1–2.
    """
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

    # Fallback (your repo convention from Week 1–2)
    return root / "data" / "processed" / "vrptdt" / "berlin_500"


def episode_path(base_dir: Path, split: str, seed: int) -> Path:
    # Week 1 created: .../episodes/SPLIT/seed_000.npz
    return base_dir / "episodes" / split.upper() / f"seed_{seed:03d}.npz"


def get_emissions_params(ingest_cfg: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    Read emissions parameters if present; otherwise use safe, speed-dependent defaults.
    These defaults are NOT physical calibration; they just ensure e(v) depends on v.
    """
    # Try nested keys first
    for key in ("emissions_params", "meet_params", "co2_params"):
        if isinstance(ingest_cfg.get(key), dict):
            p = ingest_cfg[key]
            return (
                float(p.get("alpha", 0.0)),
                float(p.get("beta", 0.0)),
                float(p.get("gamma", 1.0)),
                float(p.get("delta", 50.0)),
            )

    # Try flat keys
    return (
        float(ingest_cfg.get("alpha", 0.0)),
        float(ingest_cfg.get("beta", 0.0)),
        float(ingest_cfg.get("gamma", 1.0)),
        float(ingest_cfg.get("delta", 50.0)),
    )


def ortools_solve_tsp(cost_mat: np.ndarray, time_limit_ms: int) -> Tuple[List[int], int, float]:
    """
    Solve single-vehicle TSP (depot start/end) with OR-Tools.
    Returns: route (node indices including depot return), objective_cost, solve_time_ms
    """
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except Exception as e:
        raise RuntimeError(
            "OR-Tools not installed. Run: py -m pip install ortools"
        ) from e

    n = int(cost_mat.shape[0])
    depot = 0

    manager = pywrapcp.RoutingIndexManager(n, 1, depot)
    routing = pywrapcp.RoutingModel(manager)

    # Transit callback
    def transit_cb(from_index: int, to_index: int) -> int:
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(cost_mat[i, j])

    transit_idx = routing.RegisterTransitCallback(transit_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.time_limit.FromMilliseconds(int(time_limit_ms))

    # Reasonable defaults for TSP quality under a time cap
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.log_search = False

    t0 = time.perf_counter()
    solution = routing.SolveWithParameters(search_parameters)
    t_ms = (time.perf_counter() - t0) * 1000.0

    if solution is None:
        return [], -1, t_ms

    # Decode route
    route: List[int] = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    route.append(manager.IndexToNode(index))  # depot end

    obj = int(solution.ObjectiveValue())
    return route, obj, t_ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="TEST", choices=["TRAIN", "VAL", "TEST"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--bin", type=int, default=0, help="Which time bin to use as static cost matrix (0..B-1).")
    ap.add_argument("--time_limit_ms", type=int, default=500, help="OR-Tools time cap per solve (ms).")
    args = ap.parse_args()

    root = repo_root()
    ingest_cfg = load_json(root / "configs" / "ingest.json")
    lam_cfg = load_json(root / "configs" / "lambda.json")

    base_dir = get_processed_base_dir(ingest_cfg, root)
    ep_path = episode_path(base_dir, args.split, args.seed)

    if not ep_path.exists():
        raise FileNotFoundError(f"Episode not found: {ep_path}")

    # Load episode (Week 1 artifact)
    ep = week2_lib.load_episode_npz(ep_path)
    node_ids = ep["node_ids"]
    dist_km = ep["dist_km"]
    TT_base_min = ep["TT_data_min"]  # (B,N,N)

    B = int(TT_base_min.shape[0])
    if not (0 <= args.bin < B):
        raise ValueError(f"--bin must be in [0..{B-1}] but got {args.bin}")

    # Config: SCALE, blockage_bin, BIG_M_cost_int
    SCALE = int(ingest_cfg.get("SCALE", 1000))
    blockage_bin = int(ingest_cfg.get("blockage_bin", min(B - 1, 6)))
    BIG_M_cost_int = int(ingest_cfg.get("BIG_M_cost_int", 10_000_000))

    lam = float(lam_cfg.get("lambda", lam_cfg.get("lam", 0.0)))
    if lam <= 0:
        raise ValueError(f"lambda must be > 0. Check configs/lambda.json. Got {lam}")

    # Build events deterministically from seed (Week 2)
    events = week2_lib.generate_events_for_episode(args.seed, TT_base_min)

    # Rain is observable: planner uses TT_hat that includes rain effect
    TT_hat_min = week2_lib.apply_rain_to_TT(TT_base_min, events.rain_mask, events.rho_TT)

    # CO2 proxy (speed dependent)
    alpha, beta, gamma, delta = get_emissions_params(ingest_cfg)
    CO2_hat = week2_lib.meet_emissions_proxy(
        dist_km=dist_km,
        TT_min=TT_hat_min,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
    )
    CO2_hat = week2_lib.apply_rain_to_CO2(CO2_hat, events.rain_mask, events.rho_CO2)

    # Build integer costs and apply blockage BIG_M on planning cost J in blockage_bin
    costs = week2_lib.build_int_costs(
        TT_hat_min=TT_hat_min,
        CO2_hat=CO2_hat,
        lam=lam,
        SCALE=SCALE,
        blockage_bin=blockage_bin,
        blocked_u=int(events.blocked_u),
        blocked_v=int(events.blocked_v),
        BIG_M_cost_int=BIG_M_cost_int,
    )

    J_cost_int = costs["J_cost_int"]  # (B,N,N)
    cost_mat = J_cost_int[args.bin]   # (N,N)

    # Solve with OR-Tools
    route, obj, solve_ms = ortools_solve_tsp(cost_mat, args.time_limit_ms)

    print(f"EPISODE: {ep_path}")
    print(f"N={len(node_ids)} (incl depot), B={B}, bin={args.bin}, time_cap={args.time_limit_ms}ms")
    print(f"lambda={lam:.6f} SCALE={SCALE} blockage_bin={blockage_bin} BIG_M_cost_int={BIG_M_cost_int}")
    print(f"rain_mask={events.rain_mask.astype(int).tolist()} rho_TT={events.rho_TT} rho_CO2={events.rho_CO2}")
    print(f"blocked_arc=(u->v)=({int(events.blocked_u)}->{int(events.blocked_v)})")

    if not route:
        print(f"NO SOLUTION (solve_ms={solve_ms:.3f}ms)")
        return

    # Route display
    def to_id(x):
        # handles numpy scalars cleanly
        x = x.item() if hasattr(x, "item") else x
        return str(x)

    route_node_ids = [to_id(node_ids[i]) for i in route]
    print(f"SOLVE_MS: {solve_ms:.3f}")
    print(f"OBJ(J_cost_int): {obj}")
    print("ROUTE (indices):", route)
    print("ROUTE (node_ids):", route_node_ids)

    # Optional reporting: approximate travel time in chosen bin + service time
    service_time_min = float(ingest_cfg.get("service_time_min", 2.0))
    TT_bin = TT_hat_min[args.bin]  # static approx for reporting
    travel_min = 0.0
    for a, b in zip(route[:-1], route[1:]):
        travel_min += float(TT_bin[a, b])
    n_customers = len(route) - 2  # exclude depot start/end
    total_min = travel_min + n_customers * service_time_min
    print(f"REPORT (static bin approx): travel_min={travel_min:.2f}, service_min={n_customers*service_time_min:.2f}, total_min={total_min:.2f}")


if __name__ == "__main__":
    main()
