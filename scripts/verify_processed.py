import json
from pathlib import Path

import numpy as np

from _common import REPO_ROOT, load_json, expected_seeds

def check_episode(npz_path: Path, n_bins: int):
    z = np.load(npz_path, allow_pickle=True)
    dist = z["dist_km"]
    TT = z["TT_data_min"]
    node_ids = z["node_ids"]

    assert dist.shape == (21, 21), f"{npz_path}: dist shape {dist.shape}"
    assert TT.shape == (n_bins, 21, 21), f"{npz_path}: TT shape {TT.shape}"
    assert len(node_ids) == 21, f"{npz_path}: node_ids len {len(node_ids)}"

    assert np.isfinite(dist).all(), f"{npz_path}: dist has NaN/inf"
    assert np.isfinite(TT).all(), f"{npz_path}: TT has NaN/inf"

    assert np.allclose(np.diag(dist), 0.0), f"{npz_path}: dist diagonal not 0"
    for t in range(n_bins):
        assert np.allclose(np.diag(TT[t]), 0.0), f"{npz_path}: TT diagonal not 0 at bin {t}"

    # off-diagonal positive
    off = ~np.eye(21, dtype=bool)
    for t in range(n_bins):
        assert (TT[t][off] > 0).all(), f"{npz_path}: TT off-diagonal <=0 at bin {t}"

def main():
    ingest = load_json("configs/ingest.json")
    seeds_cfg = load_json("configs/seeds.json")
    n_bins = int(ingest["n_bins"])

    out_root = REPO_ROOT / "data/processed/vrptdt/berlin_500/episodes"

    exp = expected_seeds(seeds_cfg)
    for split, seeds in exp.items():
        split_dir = out_root / split
        assert split_dir.exists(), f"Missing split dir: {split_dir}"
        missing = []
        for s in seeds:
            p = split_dir / f"seed_{s:03d}.npz"
            if not p.exists():
                missing.append(str(p))
        assert not missing, f"Missing {len(missing)} episode files in {split}. Example: {missing[:3]}"

        # Spot-check a few episodes
        sample = seeds[:3] + seeds[-2:] if len(seeds) >= 5 else seeds
        for s in sample:
            p = split_dir / f"seed_{s:03d}.npz"
            check_episode(p, n_bins)

        print(f"OK split {split}: files={len(seeds)} spot_checked={len(sample)}")

    print("ALL OK: episode bank verified, no leakage by seed-set construction.")

if __name__ == "__main__":
    main()