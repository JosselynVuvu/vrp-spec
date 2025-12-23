import os
import glob
import json

CFG = "configs/data.json"

def main():
    with open(CFG, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    raw_dir = cfg["raw_instances_dir"]
    base = cfg["base_instance_file"]
    exclude_suffix = cfg.get("exclude_suffix", "_pdp.json")

    base_path = os.path.join(raw_dir, base)

    if not os.path.isdir(raw_dir):
        raise SystemExit(f"ERROR: raw_instances_dir not found: {raw_dir}")

    if not os.path.isfile(base_path):
        raise SystemExit(f"ERROR: base instance not found: {base_path}")

    if base.endswith(exclude_suffix):
        raise SystemExit(f"ERROR: base instance must not be excluded ({exclude_suffix}): {base}")

    all_json = glob.glob(os.path.join(raw_dir, "*.json"))
    pdp_json = [p for p in all_json if p.endswith(exclude_suffix)]
    non_pdp = [p for p in all_json if not p.endswith(exclude_suffix)]

    print("OK: raw_instances_dir =", raw_dir)
    print("OK: base_instance_file =", base)
    print("json_total =", len(all_json))
    print("json_non_pdp =", len(non_pdp))
    print("json_excluded =", len(pdp_json))

if __name__ == "__main__":
    main()
