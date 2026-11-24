#!/usr/bin/env python
"""Turn results/timings.csv into charts and a Markdown table for the README/blog post."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ORDER = ["to_gpu", "qc_metrics", "filter", "normalize", "log1p", "hvg",
         "scale", "pca", "neighbors", "umap", "leiden"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    out = Path(p.parse_args().results)
    df = pd.read_csv(out / "timings.csv")

    totals = df.groupby(["n_cells", "backend"])["seconds"].sum().unstack("backend")
    if "gpu" in totals:
        totals["speedup"] = totals["cpu"] / totals["gpu"]

    # 1) total runtime per dataset size
    fig, ax = plt.subplots(figsize=(7, 4))
    totals[[c for c in ("cpu", "gpu") if c in totals]].plot.bar(ax=ax, color=["#7f7f7f", "#76b900"])
    ax.set_xlabel("cells"); ax.set_ylabel("total time (s)")
    ax.set_title("End-to-end scRNA-seq workflow: CPU vs GPU")
    ax.set_xticklabels([f"{n:,}" for n in totals.index], rotation=0)
    if "speedup" in totals:
        for i, (n, row) in enumerate(totals.iterrows()):
            ax.annotate(f"{row['speedup']:.0f}x", (i, row["cpu"]), ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(out / "total_time.png", dpi=150); plt.close(fig)

    # 2) per-step breakdown at the largest size
    n = df["n_cells"].max()
    steps = df[df["n_cells"] == n].pivot(index="step", columns="backend", values="seconds")
    steps = steps.reindex([s for s in ORDER if s in steps.index]).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4))
    steps.plot.barh(ax=ax, color=["#7f7f7f", "#76b900"][: steps.shape[1]])
    ax.invert_yaxis(); ax.set_xscale("log"); ax.set_xlabel("seconds (log scale)")
    ax.set_title(f"Time per step, {n:,} cells")
    fig.tight_layout(); fig.savefig(out / "per_step.png", dpi=150); plt.close(fig)

    # 3) Markdown tables to paste into the README / blog
    md = ["| cells | CPU (s) | GPU (s) | speedup |", "|---:|---:|---:|---:|"]
    for n_, r in totals.iterrows():
        md.append(f"| {n_:,} | {r['cpu']:.1f} | {r.get('gpu', float('nan')):.1f} | {r.get('speedup', float('nan')):.1f}x |")
    md += ["", f"Per step at {n:,} cells:", "", "| step | CPU (s) | GPU (s) | speedup |", "|---|---:|---:|---:|"]
    for s, r in steps.iterrows():
        cpu, gpu = r.get("cpu", 0), r.get("gpu", 0)
        sp = f"{cpu / gpu:.1f}x" if cpu and gpu else "-"
        md.append(f"| {s} | {cpu:.2f} | {gpu:.2f} | {sp} |")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
