from pathlib import Path
import sys
import argparse
import json

sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
from _common import REPO_ROOT, load_json, split_for_seed
from week2_lib import load_episode_npz, generate_events_for_episode, apply_rain_to_TT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    ingest = load_json("configs/ingest.json")
    seeds_cfg = load_json("configs/seeds.json")

    split = split_for_seed(args.seed, seeds_cfg)
    ep_path = REPO_ROOT / f"data/processed/vrptdt/berlin_500/episodes/{split}/seed_{args.seed:03d}.npz"
    ep = load_episode_npz(ep_path)

    events = generate_events_for_episode(args.seed, ep["TT_data_min"])
    TT_true = apply_rain_to_TT(ep["TT_data_min"], events.rain_mask, events.rho_TT)

    print("SEED:", args.seed, "SPLIT:", split)
    print("RAIN bins:", np.where(events.rain_mask)[0].tolist(), "rho_TT:", events.rho_TT, "rho_CO2:", events.rho_CO2)
    print("INIT ROUTE idx:", events.init_route.tolist())
    print("BLOCKED ARC (u->v):", (events.blocked_u, events.blocked_v))
    print("BLOCKAGE BIN:", ingest["blockage_bin"])

    if args.save:
        out_dir = REPO_ROOT / "data/processed/vrptdt/berlin_500/derived"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"seed_{args.seed:03d}_events.npz"
        np.savez_compressed(
            out_path,
            TT_true_min=TT_true.astype(np.float32),
            rain_mask=events.rain_mask.astype(np.uint8),
            rho_TT=np.array([events.rho_TT], dtype=np.float32),
            rho_CO2=np.array([events.rho_CO2], dtype=np.float32),
            blocked_u=np.array([events.blocked_u], dtype=np.int32),
            blocked_v=np.array([events.blocked_v], dtype=np.int32),
            init_route=events.init_route.astype(np.int32),
        )
        print("WROTE:", out_path)


if __name__ == "__main__":
    main()
