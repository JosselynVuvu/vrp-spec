# scripts/replan_once.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import week2_lib


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


def _coerce_int_list(x: Any) -> List[int]:
    if x is None:
        return []
    if isinstance(x, list):
        out = []
        for v in x:
            try:
                out.append(int(v))
            except Exception:
                pass
        return out
    return []


def load_split_seeds(seeds_cfg: Dict[str, Any], split: str, base_dir: Path | None = None) -> List[int]:
    """
    Supports range-based seeds.json:
      train_start/train_end, val_start/val_end, test_start/test_end.

    Auto-detects whether *_end is inclusive or exclusive by checking whether the
    episode file for the end seed exists (if base_dir is provided).
    """
    s = split.lower()
    k_start = f"{s}_start"
    k_end = f"{s}_end"

    if k_start not in seeds_cfg or k_end not in seeds_cfg:
        raise KeyError(f"Missing {k_start}/{k_end} in configs/seeds.json")

    start = int(seeds_cfg[k_start])
    end = int(seeds_cfg[k_end])

    # Default assumption: inclusive range [start, end]
    inclusive_end = end

    # If we can check episode existence, auto-detect exclusive end.
    if base_dir is not None:
        # expected path: .../episodes/SPLIT/seed_XXX.npz
        def ep_exists(seed: int) -> bool:
            p = base_dir / "episodes" / split.upper() / f"seed_{seed:03d}.npz"
            return p.exists()

        if not ep_exists(end) and ep_exists(end - 1):
            # likely end is exclusive (like Python range semantics)
            inclusive_end = end - 1

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


def ortools_solve_tsp(cost_mat: np.ndarray, time_limit_ms: int) -> Tuple[List[int], int, float]:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

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


def to_id(x) -> str:
    x = x.item() if hasattr(x, "item") else x
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="TEST", choices=["TRAIN", "VAL", "TEST"])
    ap.add_argument("--seed", type=int, required=True, help="Episode seed (must belong to the split seed-set).")
    ap.add_argument("--bin", type=int, default=0, help="Static planning bin used in this single replan call.")
    ap.add_argument("--time_limit_ms", type=int, default=500)
    args = ap.parse_args()

    root = repo_root()
    ingest_cfg = load_json(root / "configs" / "ingest.json")
    lam_cfg = load_json(root / "configs" / "lambda.json")
    seeds_cfg = load_json(root / "configs" / "seeds.json")

    # Enforce split membership (repro contract)
    base_dir = get_processed_base_dir(ingest_cfg, root)
    split_seeds = set(load_split_seeds(seeds_cfg, args.split, base_dir))

    if int(args.seed) not in split_seeds:
        raise ValueError(
            f"Seed {args.seed} not in split {args.split} per configs/seeds.json. "
            f"(This prevents leakage / protocol drift.)"
        )

    base_dir = get_processed_base_dir(ingest_cfg, root)
    ep_path = episode_path(base_dir, args.split, int(args.seed))
    if not ep_path.exists():
        raise FileNotFoundError(f"Episode not found: {ep_path}")

    ep = week2_lib.load_episode_npz(ep_path)
    node_ids = ep["node_ids"]
    dist_km = ep["dist_km"]
    TT_base_min = ep["TT_data_min"]  # (B,N,N)
    B = int(TT_base_min.shape[0])

    if not (0 <= args.bin < B):
        raise ValueError(f"--bin must be in [0..{B-1}]")

    SCALE = int(ingest_cfg.get("SCALE", 1000))
    blockage_bin = int(ingest_cfg.get("blockage_bin", min(B - 1, 6)))
    BIG_M_cost_int = int(ingest_cfg.get("BIG_M_cost_int", 1_000_000_000_000))

    lam = float(lam_cfg.get("lambda", lam_cfg.get("lam", 0.0)))
    if lam <= 0:
        raise ValueError("lambda must be > 0. Check configs/lambda.json")

    alpha, beta, gamma, delta = get_emissions_params(ingest_cfg)

    # -----------------------------
    # COST BUILD (timed end-to-end)
    # -----------------------------
    t0 = time.perf_counter()

    events = week2_lib.generate_events_for_episode(int(args.seed), TT_base_min)
    TT_hat_min = week2_lib.apply_rain_to_TT(TT_base_min, events.rain_mask, events.rho_TT)

    CO2_hat = week2_lib.meet_emissions_proxy(
        dist_km=dist_km,
        TT_min=TT_hat_min,
        alpha=alpha, beta=beta, gamma=gamma, delta=delta,
    )
    CO2_hat = week2_lib.apply_rain_to_CO2(CO2_hat, events.rain_mask, events.rho_CO2)

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

    t_cost_ms = (time.perf_counter() - t0) * 1000.0

    # -----------------------------
    # SOLVE (timed)
    # -----------------------------
    cost_mat = costs["J_cost_int"][args.bin]
    route, obj, t_solve_ms = ortools_solve_tsp(cost_mat, args.time_limit_ms)

    # Note: decoding time is inside this function's Python work and tiny; include it in total as overhead.
    t_total_ms = t_cost_ms + t_solve_ms

    print(f"EPISODE: {ep_path}")
    print(f"N={len(node_ids)} (incl depot), B={B}, bin={args.bin}, time_cap={args.time_limit_ms}ms")
    print(f"lambda={lam:.6f} SCALE={SCALE} blockage_bin={blockage_bin} BIG_M_cost_int={BIG_M_cost_int}")
    print(f"rain_mask={events.rain_mask.astype(int).tolist()} rho_TT={events.rho_TT} rho_CO2={events.rho_CO2}")
    print(f"blocked_arc=(u->v)=({int(events.blocked_u)}->{int(events.blocked_v)})")
    print(f"T_COST_MS: {t_cost_ms:.3f}")
    print(f"T_SOLVE_MS: {t_solve_ms:.3f}")
    print(f"T_TOTAL_MS: {t_total_ms:.3f}")

    if not route:
        print("NO SOLUTION")
        return

    route_node_ids = [to_id(node_ids[i]) for i in route]
    print(f"OBJ(J_cost_int): {obj}")
    print("ROUTE (indices):", route)
    print("ROUTE (node_ids):", route_node_ids)


if __name__ == "__main__":
    main()
