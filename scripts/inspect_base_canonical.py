import json
from pathlib import Path

import numpy as np

from _common import REPO_ROOT, load_json, parse_nodes_from_vrptdt_instance

def stats(x: np.ndarray):
    return float(np.min(x)), float(np.median(x)), float(np.max(x))

def main():
    base_dir = REPO_ROOT / "data/processed/vrptdt/berlin_500"
    dist = np.load(base_dir / "base_dist_km.npy")
    TT = np.load(base_dir / "base_TT_data_min.npy")

    with open(base_dir / "base_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    print("BASE DIR:", base_dir)
    print("dist shape:", dist.shape, "dtype:", dist.dtype)
    print("TT shape:", TT.shape, "dtype:", TT.dtype)
    print("bins hours:", list(range(meta["time_origin_hour"], meta["time_origin_hour"] + meta["n_bins"])))
    print("speed_mult:", meta["speed_mult"])
    print("v_free_kmh:", meta["v_free_kmh_clipped"])

    n = dist.shape[0]
    off = ~np.eye(n, dtype=bool)

    dmin, dmed, dmax = stats(dist[off])
    print("\nDIST (km) off-diagonal: min/median/max =", (dmin, dmed, dmax))

    for t in range(TT.shape[0]):
        ttmin, ttmed, ttmax = stats(TT[t][off])
        # implied median speed for median dist/median tt
        v_med = (dmed / (ttmed / 60.0)) if ttmed > 0 else float("inf")
        print(f"BIN {t}: TT min/med/max (min) = {(ttmin, ttmed, ttmax)} | implied v_med_kmh ~ {v_med:.2f}")

    # basic invariants
    assert np.allclose(np.diag(dist), 0.0), "dist diagonal must be 0"
    for t in range(TT.shape[0]):
        assert np.allclose(np.diag(TT[t]), 0.0), "TT diagonal must be 0"
        assert np.isfinite(TT[t]).all(), "TT must be finite"
        assert (TT[t][off] > 0).all(), "TT off-diagonal must be > 0"

    # bin 3 should be slowest in pattern B (0.60)
    med_by_bin = [float(np.median(TT[t][off])) for t in range(TT.shape[0])]
    slowest_bin = int(np.argmax(med_by_bin))
    print("\nMedian TT by bin:", med_by_bin)
    print("Slowest bin (expected 3):", slowest_bin)

if __name__ == "__main__":
    main()