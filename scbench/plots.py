"""Figures and Markdown tables for the README and blog post."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .data import COLOR_COL  # noqa: E402
from .workflow import STEPS  # noqa: E402

CPU_COLOR, GPU_COLOR = "#7f7f7f", "#76b900"


def plot_umaps(cpu, gpu, n_cells, out):
    """Side-by-side UMAPs, both colored by the SAME labels: the published cell types if available,
    otherwise the CPU clusters. (Coloring each panel by its own cluster numbers would make the
    colors incomparable.)"""
    if COLOR_COL in cpu.labels:
        labels, legend_title = cpu.labels[COLOR_COL], "published cell type"
    else:
        labels, legend_title = cpu.leiden, "CPU cluster"
    cats = sorted(labels.astype(str).unique())
    cmap = plt.get_cmap("tab10" if len(cats) <= 10 else "tab20")
    colors = {c: cmap(i % cmap.N) for i, c in enumerate(cats)}
    missing = (0.8, 0.8, 0.8, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, run, title in ((axes[0], cpu, "CPU (Scanpy)"), (axes[1], gpu, "GPU (rapids-singlecell)")):
        cell_labels = labels.reindex(run.leiden.index).astype(str)
        ax.scatter(run.umap[:, 0], run.umap[:, 1], c=[colors.get(l, missing) for l in cell_labels],
                   s=0.3, rasterized=True)
        ax.set_title(f"{title}: {run.leiden.nunique()} Leiden clusters")
        ax.set_xticks([])
        ax.set_yticks([])
    if len(cats) <= 20:
        handles = [plt.Line2D([], [], marker="o", ls="", color=colors[c], label=c) for c in cats]
        fig.legend(handles=handles, title=legend_title, loc="center right", fontsize=8, title_fontsize=9)
        fig.subplots_adjust(right=0.82)
    fig.suptitle(f"{n_cells:,} cells, colored by {legend_title}")
    fig.savefig(out / f"umap_{n_cells}.png", dpi=150)
    plt.close(fig)


def make_report(out):
    """Read timings.csv and write total_time.png, per_step.png and summary.md."""
    timings = pd.read_csv(out / "timings.csv")
    totals = timings.groupby(["n_cells", "backend"])["seconds"].sum().unstack("backend")
    if "gpu" in totals:
        totals["speedup"] = totals["cpu"] / totals["gpu"]
    n_max = timings["n_cells"].max()
    steps = (timings[timings["n_cells"] == n_max]
             .pivot(index="step", columns="backend", values="seconds"))
    steps = steps.reindex([s for s in STEPS if s in steps.index]).fillna(0)

    plot_totals(totals, out / "total_time.png")
    plot_steps(steps, n_max, out / "per_step.png")
    md = summary_markdown(totals, steps, n_max)
    (out / "summary.md").write_text(md)
    print(md, end="")


def plot_totals(totals, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    totals[[c for c in ("cpu", "gpu") if c in totals]].plot.bar(ax=ax, color=[CPU_COLOR, GPU_COLOR])
    ax.set_xlabel("cells")
    ax.set_ylabel("total time (s)")
    ax.set_title("End-to-end scRNA-seq workflow: CPU vs GPU")
    ax.set_xticklabels([f"{n:,}" for n in totals.index], rotation=0)
    if "speedup" in totals:
        for i, (_, row) in enumerate(totals.iterrows()):
            ax.annotate(f"{row['speedup']:.0f}x", (i, row["cpu"]), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_steps(steps, n_cells, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    steps.plot.barh(ax=ax, color=[CPU_COLOR, GPU_COLOR][: steps.shape[1]])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("seconds (log scale)")
    ax.set_title(f"Time per step, {n_cells:,} cells")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary_markdown(totals, steps, n_cells):
    nan = float("nan")
    md = ["| cells | CPU (s) | GPU (s) | speedup |", "|---:|---:|---:|---:|"]
    for n, r in totals.iterrows():
        md.append(f"| {n:,} | {r['cpu']:.1f} | {r.get('gpu', nan):.1f} | {r.get('speedup', nan):.1f}x |")
    md += ["", f"Per step at {n_cells:,} cells:", "",
           "| step | CPU (s) | GPU (s) | speedup |", "|---|---:|---:|---:|"]
    for s, r in steps.iterrows():
        cpu, gpu = r.get("cpu", 0), r.get("gpu", 0)
        speedup = f"{cpu / gpu:.1f}x" if cpu and gpu else "-"
        md.append(f"| {s} | {cpu:.2f} | {gpu:.2f} | {speedup} |")
    return "\n".join(md) + "\n"
