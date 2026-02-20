from pathlib import Path
import sys
import time
import json

sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
from _common import REPO_ROOT, load_json, split_for_seed
from week2_lib import (
    load_episode_npz,
    generate_events_for_episode,
    apply_rain_to_TT,
    meet_emissions_proxy,
    apply_rain_to_CO2,
    build_int_costs,
)


def main():
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

    seed = 230  # pick a stable test seed
    split = split_for_seed(seed, seeds_cfg)
    ep = load_episode_npz(REPO_ROOT / f"data/processed/vrptdt/berlin_500/episodes/{split}/seed_{seed:03d}.npz")

    events = generate_events_for_episode(seed, ep["TT_data_min"])
    TT_hat = apply_rain_to_TT(ep["TT_data_min"], events.rain_mask, events.rho_TT)
    CO2 = meet_emissions_proxy(ep["dist_km"], TT_hat, alpha, beta, gamma, delta)
    CO2 = apply_rain_to_CO2(CO2, events.rain_mask, events.rho_CO2)

    t0 = time.perf_counter()
    costs = build_int_costs(
        TT_hat_min=TT_hat,
        CO2_hat=CO2,
        lam=lam,
        SCALE=int(ingest["SCALE"]),
        blockage_bin=int(ingest["blockage_bin"]),
        blocked_u=events.blocked_u,
        blocked_v=events.blocked_v,
        BIG_M_cost_int=int(ingest["BIG_M_cost_int"]),
    )
    ms = (time.perf_counter() - t0) * 1000.0

    # checks
    assert TT_hat.shape == (7, 21, 21)
    assert CO2.shape == (7, 21, 21)
    assert costs["J_cost_int"].shape == (7, 21, 21)
    assert np.isfinite(TT_hat).all()
    assert np.isfinite(CO2).all()
    assert (costs["J_cost_int"] >= 0).all()

    bbin = int(ingest["blockage_bin"])
    assert int(costs["J_cost_int"][bbin, events.blocked_u, events.blocked_v]) == int(ingest["BIG_M_cost_int"])

    print("OK verify seed", seed, "split", split)
    print("rain bins:", np.where(events.rain_mask)[0].tolist(), "blocked arc:", (events.blocked_u, events.blocked_v))
    print("cost build time (ms):", ms)


if __name__ == "__main__":
    main()
