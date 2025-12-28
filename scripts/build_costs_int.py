from pathlib import Path
import sys
import argparse
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    ingest = load_json("configs/ingest.json")
    seeds_cfg = load_json("configs/seeds.json")

    # Load lambda
    lam_path = REPO_ROOT / "configs/lambda.json"
    if not lam_path.exists():
        raise FileNotFoundError("configs/lambda.json missing. Run: py scripts/fit_lambda.py")
    lam = float(json.loads(lam_path.read_text(encoding="utf-8"))["lambda"])

    # Emissions params (same default as fit_lambda)
    emissions_cfg_path = REPO_ROOT / "configs/emissions.json"
    if emissions_cfg_path.exists():
        e = json.loads(emissions_cfg_path.read_text(encoding="utf-8"))
        alpha, beta, gamma, delta = float(e["alpha"]), float(e["beta"]), float(e["gamma"]), float(e["delta"])
    else:
        alpha, beta, gamma, delta = 2e-4, 0.0, 0.3, 2.0

    split = split_for_seed(args.seed, seeds_cfg)
    ep_path = REPO_ROOT / f"data/processed/vrptdt/berlin_500/episodes/{split}/seed_{args.seed:03d}.npz"
    ep = load_episode_npz(ep_path)

    events = generate_events_for_episode(args.seed, ep["TT_data_min"])

    # Rain is observable → planner sees rain-adjusted TT
    TT_hat = apply_rain_to_TT(ep["TT_data_min"], events.rain_mask, events.rho_TT)

    CO2_hat = meet_emissions_proxy(ep["dist_km"], TT_hat, alpha, beta, gamma, delta)
    CO2_hat = apply_rain_to_CO2(CO2_hat, events.rain_mask, events.rho_CO2)

    costs = build_int_costs(
        TT_hat_min=TT_hat,
        CO2_hat=CO2_hat,
        lam=lam,
        SCALE=int(ingest["SCALE"]),
        blockage_bin=int(ingest["blockage_bin"]),
        blocked_u=events.blocked_u,
        blocked_v=events.blocked_v,
        BIG_M_cost_int=int(ingest["BIG_M_cost_int"]),
    )

    bbin = int(ingest["blockage_bin"])
    print("SEED:", args.seed, "SPLIT:", split)
    print("rain_bins:", np.where(events.rain_mask)[0].tolist(), "rho_TT:", events.rho_TT, "rho_CO2:", events.rho_CO2)
    print("blocked arc (u->v):", (events.blocked_u, events.blocked_v), "blockage_bin:", bbin)
    print("J_cost_int[bbin,u,v] =", int(costs["J_cost_int"][bbin, events.blocked_u, events.blocked_v]), " BIG_M =", int(ingest["BIG_M_cost_int"]))

    if args.save:
        out_dir = REPO_ROOT / "data/processed/vrptdt/berlin_500/derived"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"seed_{args.seed:03d}_costs_int.npz"
        np.savez_compressed(
            out_path,
            time_cost_int=costs["time_cost_int"],
            co2_cost_int=costs["co2_cost_int"],
            J_cost_int=costs["J_cost_int"],
            rain_mask=events.rain_mask.astype(np.uint8),
            rho_TT=np.array([events.rho_TT], dtype=np.float32),
            rho_CO2=np.array([events.rho_CO2], dtype=np.float32),
            blocked_u=np.array([events.blocked_u], dtype=np.int32),
            blocked_v=np.array([events.blocked_v], dtype=np.int32),
        )
        print("WROTE:", out_path)


if __name__ == "__main__":
    main()
