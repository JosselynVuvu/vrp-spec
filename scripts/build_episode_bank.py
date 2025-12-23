import csv
import json
from pathlib import Path

import numpy as np

from _common import REPO_ROOT, load_json, ensure_dir, expected_seeds, split_for_seed

def main():
    ingest = load_json("configs/ingest.json")
    seeds_cfg = load_json("configs/seeds.json")

    base_dir = REPO_ROOT / "data/processed/vrptdt/berlin_500"
    dist_full = np.load(base_dir / "base_dist_km.npy")          # (501,501)
    TT_full = np.load(base_dir / "base_TT_data_min.npy")        # (7,501,501)

    n_total = dist_full.shape[0]
    assert n_total == TT_full.shape[1] == TT_full.shape[2], "Base matrices must match"

    # Node IDs: depot + "1".."500" (we reconstruct to avoid relying on saved node table)
    node_ids_full = np.array(["depot"] + [str(i) for i in range(1, n_total)], dtype=str)

    out_root = REPO_ROOT / "data/processed/vrptdt/berlin_500/episodes"
    ensure_dir(out_root)
    ensure_dir(out_root / "TRAIN")
    ensure_dir(out_root / "VAL")
    ensure_dir(out_root / "TEST")

    exp = expected_seeds(seeds_cfg)
    all_seeds = exp["TRAIN"] + exp["VAL"] + exp["TEST"]

    index_path = out_root / "index.csv"
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "split", "base_instance_file", "node_ids"])  # node_ids as semicolon-joined

        for seed in all_seeds:
            split = split_for_seed(seed, seeds_cfg)

            rng = np.random.default_rng(seed)
            # sample 20 customers from 1..500 (exclude depot=0)
            customers = rng.choice(np.arange(1, n_total), size=int(ingest["customers_per_episode"]), replace=False)
            # episode node indices: depot + customers
            idx = np.concatenate(([0], np.sort(customers)))  # sort for stable node order inside episode

            dist = dist_full[np.ix_(idx, idx)].astype(np.float32)
            TT = TT_full[:, idx][:, :, idx].astype(np.float32)  # (7,21,21)

            node_ids = node_ids_full[idx].astype(str)
            meta = {
                "seed": int(seed),
                "split": split,
                "base_instance_file": ingest["base_instance_file"],
                "time_origin_hour": int(ingest["time_origin_hour"]),
                "n_bins": int(ingest["n_bins"]),
                "bin_minutes": int(ingest["bin_minutes"]),
                "service_time_min": int(ingest["service_time_min"]),
                "customers_per_episode": int(ingest["customers_per_episode"]),
                "blockage_bin": int(ingest["blockage_bin"]),
                "BIG_M_TT_min": int(ingest["BIG_M_TT_min"]),
                "BIG_M_cost_int": int(ingest["BIG_M_cost_int"]),
                "SCALE": int(ingest["SCALE"]),
            }

            out_path = out_root / split / f"seed_{seed:03d}.npz"
            np.savez_compressed(
                out_path,
                node_ids=node_ids,
                dist_km=dist,
                TT_data_min=TT,
                meta_json=json.dumps(meta),
            )

            w.writerow([seed, split, ingest["base_instance_file"], ";".join(node_ids.tolist())])

    print("WROTE episode bank:", out_root)
    print("INDEX:", index_path)

if __name__ == "__main__":
    main()