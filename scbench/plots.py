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
    """Read timings.csv and write total_time.png, per_step.png and summary.md.
    With several repeats, times are medians and the whiskers / ranges show min-max."""
    timings = pd.read_csv(out / "timings.csv")
    if "repeat" not in timings:  # results from before --repeats existed
        timings["repeat"] = 0
    n_repeats = timings["repeat"].nunique()
    per_run = timings.groupby(["n_cells", "backend", "repeat"])["seconds"].sum()
    by_size = per_run.groupby(["n_cells", "backend"])
    totals = by_size.median().unstack("backend")
    low, high = by_size.min().unstack("backend"), by_size.max().unstack("backend")

    n_max = timings["n_cells"].max()
    steps = (timings[timings["n_cells"] == n_max]
             .groupby(["step", "backend"])["seconds"].median().unstack("backend"))
    steps = steps.reindex([s for s in STEPS if s in steps.index]).fillna(0)

    plot_totals(totals, low, high, n_repeats, out / "total_time.png")
    plot_steps(steps, n_max, n_repeats, out / "per_step.png")
    md = summary_markdown(totals, low, high, steps, n_max, n_repeats)
    (out / "summary.md").write_text(md)
    print(md, end="")


def _median_note(n_repeats):
    return f" (median of {n_repeats} runs)" if n_repeats > 1 else ""


def plot_totals(totals, low, high, n_repeats, path):
    backends = [c for c in ("cpu", "gpu") if c in totals]
    # whiskers from min to max across repeats (zero length for a single run)
    yerr = [[(totals[b] - low[b]).values, (high[b] - totals[b]).values] for b in backends]
    fig, ax = plt.subplots(figsize=(7, 4))
    totals[backends].plot.bar(ax=ax, color=[CPU_COLOR, GPU_COLOR], yerr=yerr if n_repeats > 1 else None,
                              capsize=3, error_kw={"elinewidth": 1})
    ax.set_xlabel("cells")
    ax.set_ylabel("total time (s)")
    ax.set_title("End-to-end scRNA-seq workflow: CPU vs GPU" + _median_note(n_repeats))
    ax.set_xticklabels([f"{n:,}" for n in totals.index], rotation=0)
    if "gpu" in totals:
        for i, (_, row) in enumerate(totals.iterrows()):
            ax.annotate(f"{row['cpu'] / row['gpu']:.0f}x", (i, high["cpu"].iloc[i]), ha="center", va="bottom")
        ax.set_ylim(top=high.max().max() * 1.12)  # room for the speedup labels
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_steps(steps, n_cells, n_repeats, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    steps.plot.barh(ax=ax, color=[CPU_COLOR, GPU_COLOR][: steps.shape[1]])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("seconds (log scale)")
    ax.set_title(f"Time per step, {n_cells:,} cells" + _median_note(n_repeats))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary_markdown(totals, low, high, steps, n_cells, n_repeats):
    def cell(b, n):
        if b not in totals:
            return "-"
        med = f"{totals.at[n, b]:.1f}"
        return med if n_repeats == 1 else f"{med} ({low.at[n, b]:.1f}-{high.at[n, b]:.1f})"

    md = []
    if n_repeats > 1:
        md += [f"Median of {n_repeats} runs per size; min-max in parentheses.", ""]
    md += ["| cells | CPU (s) | GPU (s) | speedup |", "|---:|---:|---:|---:|"]
    for n in totals.index:
        speedup = f"{totals.at[n, 'cpu'] / totals.at[n, 'gpu']:.1f}x" if "gpu" in totals else "-"
        md.append(f"| {n:,} | {cell('cpu', n)} | {cell('gpu', n)} | {speedup} |")
    md += ["", f"Per step at {n_cells:,} cells{_median_note(n_repeats)}:", "",
           "| step | CPU (s) | GPU (s) | speedup |", "|---|---:|---:|---:|"]
    for s, r in steps.iterrows():
        cpu, gpu = r.get("cpu", 0), r.get("gpu", 0)
        speedup = f"{cpu / gpu:.1f}x" if cpu and gpu else "-"
        md.append(f"| {s} | {cpu:.2f} | {gpu:.2f} | {speedup} |")
    return "\n".join(md) + "\n"
