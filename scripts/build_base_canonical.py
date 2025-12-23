import json
from pathlib import Path

import numpy as np

from _common import (
    REPO_ROOT,
    load_json,
    ensure_dir,
    parse_nodes_from_vrptdt_instance,
    haversine_km_matrix,
    calibrate_v_free_kmh,
    build_TT_data_min,
)

def main():
    ingest = load_json("configs/ingest.json")

    raw_dir = REPO_ROOT / ingest["raw_instances_dir"]
    base_file = ingest["base_instance_file"]
    instance_path = raw_dir / base_file

    out_dir = REPO_ROOT / "data/processed/vrptdt/berlin_500"
    ensure_dir(out_dir)

    # 1) Nodes + coords
    node_ids, coords = parse_nodes_from_vrptdt_instance(instance_path)

    # 2) Distances (km)
    dist_km = haversine_km_matrix(coords).astype(np.float32)

    # 3) Calibrate v_free_kmh once
    target_tt = float(ingest["target_median_leg_tt_min"])
    clip_lo, clip_hi = map(float, ingest["v_free_kmh_clip"])
    median_dist_km, v_free_raw, v_free_kmh = calibrate_v_free_kmh(dist_km.astype(np.float64), target_tt, clip_lo, clip_hi)

    # 4) Build TT_data_min [7,N,N] for bins 15..22 (SPEC v1.7)
    speed_mult = ingest["speed_mult"]
    assert len(speed_mult) == int(ingest["n_bins"]), "speed_mult length must equal n_bins"
    TT_data_min = build_TT_data_min(dist_km.astype(np.float64), v_free_kmh, speed_mult)

    # 5) Save arrays
    np.save(out_dir / "base_dist_km.npy", dist_km)
    np.save(out_dir / "base_TT_data_min.npy", TT_data_min)

    # 6) Save metadata
    meta = {
        "base_instance_file": base_file,
        "raw_instances_dir": str(ingest["raw_instances_dir"]),
        "time_origin_hour": int(ingest["time_origin_hour"]),
        "n_bins": int(ingest["n_bins"]),
        "bin_minutes": int(ingest["bin_minutes"]),
        "service_time_min": int(ingest["service_time_min"]),
        "customers_per_episode": int(ingest["customers_per_episode"]),
        "blockage_bin": int(ingest["blockage_bin"]),
        "BIG_M_TT_min": int(ingest["BIG_M_TT_min"]),
        "BIG_M_cost_int": int(ingest["BIG_M_cost_int"]),
        "SCALE": int(ingest["SCALE"]),
        "speed_mult": [float(x) for x in speed_mult],
        "target_median_leg_tt_min": float(target_tt),
        "median_dist_km": float(median_dist_km),
        "v_free_kmh_raw": float(v_free_raw),
        "v_free_kmh_clipped": float(v_free_kmh),
        "node_count_incl_depot": int(len(node_ids)),
    }
    with open(out_dir / "base_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Optional: store node table for debugging
    node_table = np.column_stack([node_ids, coords.astype(np.float64)])
    np.save(out_dir / "base_nodes.npy", node_table)

    print("WROTE:", out_dir)
    print(" dist:", (out_dir / "base_dist_km.npy"))
    print(" TT:", (out_dir / "base_TT_data_min.npy"))
    print(" meta:", (out_dir / "base_meta.json"))
    print(" v_free_kmh:", v_free_kmh)

if __name__ == "__main__":
    main()
