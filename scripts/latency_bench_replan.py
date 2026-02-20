# scripts/latency_bench_replan.py
from __future__ import annotations

import argparse
import csv
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


def ortools_solve_tsp(cost_mat: np.ndarray, time_limit_ms: int) -> Tuple[int, float]:
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
    obj = int(sol.ObjectiveValue()) if sol is not None else -1
    return obj, t_ms


def pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="TEST", choices=["TRAIN", "VAL", "TEST"])
    ap.add_argument("--bin", type=int, default=0)
    ap.add_argument("--time_limit_ms", type=int, default=500)
    ap.add_argument("--n_calls", type=int, default=1000)
    args = ap.parse_args()

    root = repo_root()
    ingest_cfg = load_json(root / "configs" / "ingest.json")
    lam_cfg = load_json(root / "configs" / "lambda.json")
    seeds_cfg = load_json(root / "configs" / "seeds.json")

    base_dir = get_processed_base_dir(ingest_cfg, root)
    seeds = load_split_seeds(seeds_cfg, args.split, base_dir)
    if not seeds:
        raise RuntimeError(f"No seeds for split={args.split} in configs/seeds.json")

    lam = float(lam_cfg.get("lambda", lam_cfg.get("lam", 0.0)))
    if lam <= 0:
        raise ValueError("lambda must be > 0. Check configs/lambda.json")

    SCALE = int(ingest_cfg.get("SCALE", 1000))
    BIG_M_cost_int = int(ingest_cfg.get("BIG_M_cost_int", 1_000_000_000_000))

    # Load one episode to infer B
    first_ep_path = episode_path(base_dir, args.split, seeds[0])
    if not first_ep_path.exists():
        raise FileNotFoundError(f"Episode missing for seed {seeds[0]}: {first_ep_path}")
    sample_ep = week2_lib.load_episode_npz(first_ep_path)
    TT_sample = sample_ep["TT_data_min"]
    B = int(TT_sample.shape[0])

    blockage_bin = int(ingest_cfg.get("blockage_bin", min(B - 1, 6)))
    if not (0 <= args.bin < B):
        raise ValueError(f"--bin must be in [0..{B-1}]")

    alpha, beta, gamma, delta = get_emissions_params(ingest_cfg)

    # Preload episodes (keeps disk I/O out of timing loop)
    eps = {}
    for s in seeds:
        p = episode_path(base_dir, args.split, s)
        if not p.exists():
            raise FileNotFoundError(f"Missing episode file for seed {s}: {p}")
        eps[s] = week2_lib.load_episode_npz(p)

    t_cost, t_solve, t_total, objs, used_seeds = [], [], [], [], []

    for k in range(args.n_calls):
        s = seeds[k % len(seeds)]
        ep = eps[s]
        dist_km = ep["dist_km"]
        TT_base_min = ep["TT_data_min"]

        # COST BUILD timing (includes event gen + rain + CO2 proxy + int costs)
        t0 = time.perf_counter()

        events = week2_lib.generate_events_for_episode(s, TT_base_min)
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

        # SOLVE timing
        cost_mat = costs["J_cost_int"][args.bin]
        obj, t_solve_ms = ortools_solve_tsp(cost_mat, args.time_limit_ms)

        t_tot_ms = t_cost_ms + t_solve_ms

        used_seeds.append(s)
        t_cost.append(t_cost_ms)
        t_solve.append(t_solve_ms)
        t_total.append(t_tot_ms)
        objs.append(obj)

    t_cost = np.array(t_cost, dtype=np.float64)
    t_solve = np.array(t_solve, dtype=np.float64)
    t_total = np.array(t_total, dtype=np.float64)

    print(f"BENCH: split={args.split} bin={args.bin} cap={args.time_limit_ms}ms calls={args.n_calls} seeds={len(seeds)}")
    print(f"COST  p50/p95/max ms: {pct(t_cost,50):.3f} / {pct(t_cost,95):.3f} / {t_cost.max():.3f}")
    print(f"SOLVE p50/p95/max ms: {pct(t_solve,50):.3f} / {pct(t_solve,95):.3f} / {t_solve.max():.3f}")
    print(f"TOTAL p50/p95/max ms: {pct(t_total,50):.3f} / {pct(t_total,95):.3f} / {t_total.max():.3f}")

    out_dir = root / "data" / "processed" / "bench" / "week3_latency"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"latency_{args.split}_bin{args.bin}_cap{args.time_limit_ms}_n{args.n_calls}.csv"

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["call_idx", "seed", "t_cost_ms", "t_solve_ms", "t_total_ms", "obj"])
        for i in range(args.n_calls):
            w.writerow([i, used_seeds[i], float(t_cost[i]), float(t_solve[i]), float(t_total[i]), int(objs[i])])

    print(f"WROTE: {out_csv}")


if __name__ == "__main__":
    main()
