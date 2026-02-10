import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="./logs/yarn_resource_log.csv")
    ap.add_argument("--outdir", default="./plots")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)

    # Basic cleanup
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    df = df.dropna(subset=["round"])
    df["round"] = df["round"].astype(int)

    for c in ["free_mb", "free_vcores", "allocated_mb", "allocated_vcores", "t"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Aggregation time from "after" rows
    after = df[df["phase"] == "after"].copy()
    if "agg_seconds" not in after.columns:
        raise SystemExit("Missing agg_seconds in AFTER rows. Ensure you write it in append_csv for 'after'.")

    after["agg_seconds"] = pd.to_numeric(after["agg_seconds"], errors="coerce")
    after = after.dropna(subset=["agg_seconds"])

    # DURING stats per round
    during = df[df["phase"] == "during"].copy()
    during_stats = (
        during.groupby("round")
        .agg(
            free_mb_min=("free_mb", "min"),
            free_mb_mean=("free_mb", "mean"),
            free_vcores_min=("free_vcores", "min"),
            free_vcores_mean=("free_vcores", "mean"),
        )
        .reset_index()
    )

    summary = after[["round", "agg_seconds"]].merge(during_stats, on="round", how="left").sort_values("round")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Plot: agg_seconds vs round
    plt.figure()
    plt.plot(summary["round"], summary["agg_seconds"], marker="o")
    plt.xlabel("Round")
    plt.ylabel("Aggregation time (s)")
    plt.title("Aggregation time vs Round")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p1 = outdir / "agg_time_vs_round.png"
    plt.savefig(p1, dpi=200)
    print(f"Saved: {p1}")

    # Plot: free_mb_mean vs agg_seconds
    plt.figure()
    plt.scatter(summary["free_mb_mean"], summary["agg_seconds"])
    plt.xlabel("Mean free MB during aggregation")
    plt.ylabel("Aggregation time (s)")
    plt.title("Resource headroom vs aggregation time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = outdir / "free_mb_mean_vs_agg_time.png"
    plt.savefig(p2, dpi=200)
    print(f"Saved: {p2}")

    # Plot: free_vcores_mean vs agg_seconds
    plt.figure()
    plt.scatter(summary["free_vcores_mean"], summary["agg_seconds"])
    plt.xlabel("Mean free vcores during aggregation")
    plt.ylabel("Aggregation time (s)")
    plt.title("Free vcores vs aggregation time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p3 = outdir / "free_vcores_mean_vs_agg_time.png"
    plt.savefig(p3, dpi=200)
    print(f"Saved: {p3}")

    # Save summary CSV (useful for your paper)
    out_csv = outdir / "resource_time_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
