# scripts/plot_policy_eval_grid.py
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FNAME_RE = re.compile(
    r"baselines_(?P<split>TRAIN|VAL|TEST)_(?P<rain>rain|norain)_cap(?P<cap>\d+)_startbin(?P<startbin>\d+)_blockbin(?P<blockbin>\d+)_k(?P<k>\d+)_ef(?P<ef>\d+\.\d+)\.csv$"
)


def parse_filename(p: Path):
    m = FNAME_RE.match(p.name)
    if not m:
        return None
    d = m.groupdict()
    d["cap"] = int(d["cap"])
    d["startbin"] = int(d["startbin"])
    d["blockbin"] = int(d["blockbin"])
    d["k"] = int(d["k"])
    d["ef"] = float(d["ef"])
    return d


def parse_caps(s: str) -> list[int] | None:
    s = (s or "").strip()
    if not s:
        return None
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out if out else None


def load_all(in_dir: Path, split: str, k: int | None, ef: float | None, caps: list[int] | None) -> pd.DataFrame:
    rows = []
    used = []

    for f in sorted(in_dir.glob("baselines_*.csv")):
        meta = parse_filename(f)
        if meta is None:
            continue
        if meta["split"] != split:
            continue

        # --- filters to prevent mixing experiments ---
        if k is not None and meta["k"] != k:
            continue
        if ef is not None and abs(meta["ef"] - ef) > 1e-9:
            continue
        if caps is not None and meta["cap"] not in caps:
            continue

        df = pd.read_csv(f)

        # attach metadata to each row
        for kk, vv in meta.items():
            df[kk] = vv
        df["source_file"] = f.name

        rows.append(df)
        used.append(f.name)

    if not rows:
        raise FileNotFoundError(
            f"No matching baselines CSVs found in: {in_dir}\n"
            f"Filters: split={split}, k={k}, ef={ef}, caps={caps}"
        )

    print("USING FILES:")
    for name in used:
        print("  -", name)

    return pd.concat(rows, ignore_index=True)


def agg(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    ok = df[df["ok"] == True].copy()
    g = ok.groupby(["rain", "cap", "policy"], as_index=False)[metric]
    out = g.agg(mean="mean", median="median", p95=lambda x: float(np.percentile(x, 95)))
    return out.sort_values(["rain", "cap", "policy"])


def plot_grid(agg_df: pd.DataFrame, metric: str, out_path: Path) -> None:
    # fixed ordering so plots are consistent across runs
    policy_order = ["B0_PlanOnce", "B2_BlockageReplan", "B1_AlwaysReplan"]
    policies = [p for p in policy_order if p in set(agg_df["policy"].unique())]
    if not policies:
        policies = list(agg_df["policy"].unique())

    caps = sorted(agg_df["cap"].unique())
    rains = ["rain", "norain"]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, rain in zip(axs, rains):
        sub = agg_df[agg_df["rain"] == rain]
        for pol in policies:
            s = sub[sub["policy"] == pol].sort_values("cap")
            if len(s) == 0:
                continue
            ax.plot(s["cap"], s["mean"], marker="o", label=pol)

        ax.set_title(f"{metric} vs time_limit_ms ({rain})")
        ax.set_xlabel("time_limit_ms")
        ax.set_ylabel(f"{metric} (mean over seeds)")
        ax.set_xticks(caps)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.savefig(out_path, dpi=200)
    print(f"WROTE: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="TEST", choices=["TRAIN", "VAL", "TEST"])
    ap.add_argument("--metric", type=str, default="J_wall")
    ap.add_argument("--in_dir", type=str, default="data/processed/bench/week3_results")
    ap.add_argument("--out", type=str, default="")

    # Thesis-clean filters:
    ap.add_argument("--k", type=int, default=3, help="Filter by n_blockages (k) from filename.")
    ap.add_argument("--ef", type=float, default=0.60, help="Filter by early_frac (ef) from filename.")
    ap.add_argument(
        "--caps",
        type=str,
        default="200,500,800",
        help="Comma-separated time limits to include, e.g. '200,500,800'.",
    )

    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    caps = parse_caps(args.caps)

    df = load_all(in_dir=in_dir, split=args.split, k=args.k, ef=args.ef, caps=caps)

    if args.metric not in df.columns:
        raise KeyError(
            f"Metric '{args.metric}' not found in CSV columns.\n"
            f"Available columns include: {sorted(df.columns)[:60]} ..."
        )

    a = agg(df, args.metric)

    out_path = Path(args.out) if args.out else (in_dir / f"replanning_grid_{args.split}_{args.metric}.png")
    plot_grid(a, args.metric, out_path)


if __name__ == "__main__":
    main()
