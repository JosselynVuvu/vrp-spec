# scripts/plot_twin_eval_grid.py
# Plot grids for twin evaluation CSVs (rain vs norain) with optional Gate B ablation plots

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FNAME_RE = re.compile(
    r"twin_(?P<split>TRAIN|VAL|TEST)_(?P<rain>rain|norain)"
    r"_cap(?P<cap>\d+)_startbin(?P<startbin>\d+)_blockbin(?P<blockbin>\d+)"
    r"_k(?P<k>\d+)_ef(?P<ef>\d+\.\d+)\.csv$"
)


def parse_csv_filename(p: Path) -> Optional[Dict]:
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


def coerce_ok_column(df: pd.DataFrame) -> pd.DataFrame:
    if "ok" not in df.columns:
        df["ok"] = True
        return df
    if df["ok"].dtype == bool:
        return df

    def to_bool(x):
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        if isinstance(x, (int, np.integer)):
            return int(x) != 0
        if isinstance(x, (float, np.floating)):
            return float(x) != 0.0
        if isinstance(x, str):
            s = x.strip().lower()
            if s in ("true", "t", "1", "yes", "y"):
                return True
            if s in ("false", "f", "0", "no", "n", ""):
                return False
        return bool(x)

    df["ok"] = df["ok"].map(to_bool)
    return df


def parse_caps(caps_str: str) -> Optional[List[int]]:
    if not caps_str:
        return None
    return [int(x.strip()) for x in caps_str.split(",") if x.strip()]


def parse_metrics(metrics_str: str) -> List[str]:
    if not metrics_str:
        return []
    return [m.strip() for m in metrics_str.split(",") if m.strip()]


def load_all(
    in_dir: Path,
    split: str,
    k: Optional[int],
    ef: Optional[float],
    caps: Optional[List[int]],
    startbin: Optional[int],
    blockbin: Optional[int],
) -> Tuple[pd.DataFrame, List[str]]:
    rows = []
    used_files: List[str] = []

    for f in sorted(in_dir.glob("twin_*.csv")):
        meta = parse_csv_filename(f)
        if meta is None:
            continue
        if meta["split"] != split:
            continue
        if k is not None and meta["k"] != k:
            continue
        if ef is not None and abs(meta["ef"] - ef) > 1e-9:
            continue
        if caps is not None and meta["cap"] not in caps:
            continue
        if startbin is not None and meta["startbin"] != startbin:
            continue
        if blockbin is not None and meta["blockbin"] != blockbin:
            continue

        df = pd.read_csv(f)
        df = coerce_ok_column(df)

        # attach meta for grouping
        for kk, vv in meta.items():
            df[kk] = vv
        df["source_file"] = f.name

        # backwards-compat: ensure 'rain' exists even if old CSV
        if "rain" not in df.columns:
            df["rain"] = meta["rain"]

        rows.append(df)
        used_files.append(f.name)

    if not rows:
        raise FileNotFoundError(
            f"No matching twin CSVs found in {in_dir} for split={split} "
            f"(k={k}, ef={ef}, caps={caps}, startbin={startbin}, blockbin={blockbin})."
        )

    out = pd.concat(rows, ignore_index=True)
    return out, used_files


def agg_metric(df: pd.DataFrame, metric: str, policies: List[str]) -> pd.DataFrame:
    if metric not in df.columns:
        raise KeyError(
            f"Metric '{metric}' not found.\nAvailable columns include: {sorted(df.columns)[:80]} ..."
        )

    required = ["rain", "cap", "policy", "ok"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}'. Make sure run_baselines.py writes it.")

    sub = df[df["ok"] == True].copy()
    sub = sub[sub["rain"].isin(["rain", "norain"])].copy()

    # filter policies
    sub = sub[sub["policy"].isin(policies)].copy()

    g = sub.groupby(["rain", "cap", "policy"], as_index=False)[metric]
    out = g.agg(
        mean="mean",
        median="median",
        p95=lambda x: float(np.percentile(x, 95)),
        n="count",
    )
    return out.sort_values(["rain", "cap", "policy"])


def plot_grid_for_metric(agg_df: pd.DataFrame, metric: str, out_path: Path, title_prefix: str) -> None:
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

        ax.set_title(f"{title_prefix}: {metric} vs time_limit_ms ({rain})")
        ax.set_xlabel("time_limit_ms")
        ax.set_ylabel(f"{metric} (mean over seeds)")
        ax.set_xticks(caps)
        ax.grid(True, alpha=0.3)
        ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"WROTE: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="TEST", choices=["TRAIN", "VAL", "TEST"])
    ap.add_argument("--in_dir", type=str, default="data/processed/bench/digital_twin_eval_results")
    ap.add_argument("--out_dir", type=str, default="")

    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--ef", type=float, default=None)
    ap.add_argument("--caps", type=str, default="")
    ap.add_argument("--startbin", type=int, default=None)
    ap.add_argument("--blockbin", type=int, default=None)

    ap.add_argument("--metrics", type=str, default="J_wall,wall_time_min,CO2_total,planning_wait_min,solve_ms_total,solve_ms_p95,solve_ms_max,rel_pred_err_mean,rel_pred_err_p95")
    ap.add_argument("--gate_ablation", action="store_true", help="Also plot B1 vs B3 only (appendix-friendly).")

    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir

    caps = parse_caps(args.caps)
    metrics = parse_metrics(args.metrics)
    if not metrics:
        raise ValueError("No metrics provided. Use --metrics.")

    df, used_files = load_all(
        in_dir=in_dir,
        split=args.split,
        k=args.k,
        ef=args.ef,
        caps=caps,
        startbin=args.startbin,
        blockbin=args.blockbin,
    )

    print("USING FILES:")
    for f in used_files:
        print(f"  - {f}")

    policies_main = ["B0_PlanOnce", "B2_BlockageReplan", "B1_AlwaysReplan"]
    policies_gate = ["B1_AlwaysReplan", "B3_GateReplan"]

    # Main plots: keep clean (no gate line unless you want it)
    for metric in metrics:
        a = agg_metric(df, metric, policies=policies_main)
        out_path = out_dir / f"twin_grid_{args.split}_{metric}.png"
        plot_grid_for_metric(a, metric, out_path, title_prefix="Twin eval (main)")

    # Gate ablation plots: appendix
    if args.gate_ablation:
        # Only run if gate exists in data
        if "B3_GateReplan" not in set(df["policy"].unique()):
            print("NOTE: gate_ablation requested, but no B3_GateReplan rows found in CSVs.")
            return

        gate_metrics = metrics + ["n_solve_calls", "n_replans", "n_gate_probes", "n_gate_full_replans", "gate_gain_hat_mean"]
        for metric in gate_metrics:
            if metric not in df.columns:
                # skip silently for older files
                continue
            a = agg_metric(df, metric, policies=policies_gate)
            out_path = out_dir / f"twin_gate_grid_{args.split}_{metric}.png"
            plot_grid_for_metric(a, metric, out_path, title_prefix="Twin eval (Gate B ablation)")


if __name__ == "__main__":
    main()
