from pathlib import Path
import sys

# Make scripts/ importable when running "py scripts/peek_instance.py"
sys.path.append(str(Path(__file__).resolve().parent))

from _common import REPO_ROOT, load_json, parse_nodes_from_vrptdt_instance

def main():
    ingest = load_json("configs/ingest.json")
    raw_dir = REPO_ROOT / ingest["raw_instances_dir"]
    base_file = ingest["base_instance_file"]
    instance_path = raw_dir / base_file

    node_ids, coords = parse_nodes_from_vrptdt_instance(instance_path)

    print("INSTANCE:", str(instance_path))
    print("N_TOTAL (incl depot):", len(node_ids))
    print("N_ITEMS:", len(node_ids) - 1)
    print("DEPOT:", coords[0].tolist())
    print("FIRST 3 ITEMS:")
    for i in range(1, min(4, len(node_ids))):
        print(f"  id={node_ids[i]} latlon={coords[i].tolist()}")

    # show configured bins (SPEC v1.7)
    time_origin = ingest["time_origin_hour"]
    n_bins = ingest["n_bins"]
    print("BINS (hours):", list(range(time_origin, time_origin + n_bins)))

if __name__ == "__main__":
    main()