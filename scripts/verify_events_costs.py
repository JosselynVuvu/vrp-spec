# scripts/verify_events_costs.py
from __future__ import annotations

from pathlib import Path
import sys
import time
import json

# allow "import week2_lib" when running: py scripts/verify_events_costs.py
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
from _common import REPO_ROOT, load_json, split_for_seed
from scripts.vrp_lib import (
    load_episode_npz,
    generate_events_for_episode,
    apply_rain_to_TT,
    meet_emissions_proxy,
    apply_rain_to_CO2,
    build_int_costs,
)


def _resolve_path(p: str | None, root: Path) -> Path | None:
    if not p:
        return None
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp)


def _get_processed_base_dir(ingest: dict, root: Path) -> Path:
    # best-effort: match the pattern used in your other scripts
    for k in ("processed_base_dir", "out_base_dir", "processed_dir", "base_out_dir"):
        p = _resolve_path(ingest.get(k), root)
        if p and p.exists():
            return p
    return root / "data" / "processed" / "vrptdt" / "berlin_500"


def main() -> None:
    ingest = load_json("configs/ingest.json")
    seeds_cfg = load_json("configs/seeds.json")

    lam = float(json.loads((REPO_ROOT / "configs/lambda.json").read_text(encoding="utf-8"))["lambda"])

    # emissions params defaults
    emissions_cfg_path = REPO_ROOT / "configs/emissions.json"
    if emissions_cfg_path.exists():
        e = json.loads(emissions_cfg_path.read_text(encoding="utf-8"))
        alpha, beta, gamma, delta = float(e["alpha"]), float(e["beta"]), float(e["gamma"]), float(e["delta"])
    else:
        alpha, beta, gamma, delta = 2e-4, 0.0, 0.3, 2.0

    seed = 230  # stable test seed
    split = split_for_seed(seed, seeds_cfg)

    base_dir = _get_processed_base_dir(ingest, REPO_ROOT)
    ep_path = base_dir / "episodes" / split.upper() / f"seed_{seed:03d}.npz"
    if not ep_path.exists():
        raise FileNotFoundError(f"Episode not found: {ep_path}")

    ep = load_episode_npz(ep_path)

    TT_data = ep["TT_data_min"]
    dist_km = ep["dist_km"]
    B, N, _ = TT_data.shape

    events = generate_events_for_episode(seed, TT_data)

    TT_hat = apply_rain_to_TT(TT_data, events.rain_mask, events.rho_TT)
    CO2 = meet_emissions_proxy(dist_km, TT_hat, alpha, beta, gamma, delta)
    CO2 = apply_rain_to_CO2(CO2, events.rain_mask, events.rho_CO2)

    blockage_bin = int(ingest["blockage_bin"])
    BIG_M_cost_int = int(ingest["BIG_M_cost_int"])
    SCALE = int(ingest["SCALE"])

    t0 = time.perf_counter()
    costs = build_int_costs(
        TT_hat_min=TT_hat,
        CO2_hat=CO2,
        lam=lam,
        SCALE=SCALE,
        blockage_bin=blockage_bin,
        BIG_M_cost_int=BIG_M_cost_int,
        blocked_arcs=events.blocked_arcs,  # <-- NEW (K arcs)
    )
    ms = (time.perf_counter() - t0) * 1000.0

    # checks
    assert TT_hat.shape == (B, N, N)
    assert CO2.shape == (B, N, N)
    assert costs["J_cost_int"].shape == (B, N, N)
    assert np.isfinite(TT_hat).all()
    assert np.isfinite(CO2).all()
    assert (costs["J_cost_int"] >= 0).all()

    # verify ALL blocked arcs got BIG_M in the planning-cost tensor at blockage_bin
    ba = np.asarray(events.blocked_arcs, dtype=np.int64).reshape(-1, 2)
    for (u, v) in ba:
        got = int(costs["J_cost_int"][blockage_bin, int(u), int(v)])
        assert got == BIG_M_cost_int, f"Blocked arc {u}->{v} not BIG_M at bin={blockage_bin}. got={got}"

    print("OK verify seed", seed, "split", split)
    print("episode:", ep_path)
    print("rain bins:", np.where(events.rain_mask)[0].tolist(), "rho_TT:", events.rho_TT, "rho_CO2:", events.rho_CO2)
    print("blocked_arcs:", [(int(u), int(v)) for (u, v) in ba.tolist()], "blockage_bin:", blockage_bin)
    print("cost build time (ms):", ms)


if __name__ == "__main__":
    main()
