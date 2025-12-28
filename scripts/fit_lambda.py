from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
from _common import REPO_ROOT, load_json, expected_seeds, split_for_seed
from week2_lib import load_episode_npz, meet_emissions_proxy


def main():
    ingest = load_json("configs/ingest.json")
    seeds_cfg = load_json("configs/seeds.json")
    train_seeds = expected_seeds(seeds_cfg)["TRAIN"]

    # Defaults (proxy). If you later add configs/emissions.json, load from there.
    emissions_cfg_path = REPO_ROOT / "configs/emissions.json"
    if emissions_cfg_path.exists():
        with open(emissions_cfg_path, "r", encoding="utf-8") as f:
            e = json.load(f)
        alpha, beta, gamma, delta = float(e["alpha"]), float(e["beta"]), float(e["gamma"]), float(e["delta"])
    else:
        alpha, beta, gamma, delta = 2e-4, 0.0, 0.3, 2.0

    TT_legs = []
    CO2_legs = []

    for seed in train_seeds:
        split = split_for_seed(seed, seeds_cfg)
        ep_path = REPO_ROOT / f"data/processed/vrptdt/berlin_500/episodes/{split}/seed_{seed:03d}.npz"
        ep = load_episode_npz(ep_path)

        dist = ep["dist_km"]
        TT = ep["TT_data_min"]  # base (no rain) for stable normalization

        CO2 = meet_emissions_proxy(dist, TT, alpha, beta, gamma, delta)

        off = ~np.eye(dist.shape[0], dtype=bool)
        # Collect all bins, all off-diagonal legs
        TT_legs.append(TT[:, off].reshape(-1))
        CO2_legs.append(CO2[:, off].reshape(-1))

    TT_all = np.concatenate(TT_legs).astype(np.float64)
    CO2_all = np.concatenate(CO2_legs).astype(np.float64)

    TT_med = float(np.median(TT_all))
    CO2_med = float(np.median(CO2_all))
    lam = CO2_med / max(TT_med, 1e-9)

    out = {
        "lambda": lam,
        "computed_on": {
            "split": "TRAIN",
            "seeds": [int(train_seeds[0]), int(train_seeds[-1])],
            "n_seeds": int(len(train_seeds)),
            "note": "Computed on base TT_data_min (no rain/blockage) to avoid event leakage.",
        },
        "emissions_params": {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta},
    }

    out_path = REPO_ROOT / "configs/lambda.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("WROTE:", out_path)
    print("lambda =", lam)
    print("TT_med =", TT_med, "CO2_med =", CO2_med)


if __name__ == "__main__":
    main()
