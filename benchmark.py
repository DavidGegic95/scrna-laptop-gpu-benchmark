#!/usr/bin/env python
"""Benchmark a standard single-cell RNA-seq workflow on CPU (Scanpy) vs GPU (rapids-singlecell).

Both backends run the same steps with the same parameters on the same cells:
QC metrics -> filtering -> normalize -> log1p -> HVG selection -> scale -> PCA
-> kNN graph -> UMAP -> Leiden clustering.

Outputs (in --out):
  timings.csv     one row per (n_cells, backend, step) with wall-clock seconds
  agreement.csv   Adjusted Rand Index between CPU and GPU Leiden clusters per size
  gpu_memory.csv  per size: GPU run ok/failed, failing step, peak GPU memory (sampled after each step)
  system.json     hardware/software versions, so the numbers are reproducible
  umap_<n>.png    side-by-side UMAPs (CPU vs GPU) for each size
"""
import argparse
import gc
import json
import os
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

# Published annotations, if the file has them: broad labels for coloring plots,
# detailed labels to score how well each backend's clusters recover known cell types.
COLOR_COL, SCORE_COL = "celltype.l1", "cell_type"
LABEL_COLS = [COLOR_COL, SCORE_COL]

STEPS = ["to_gpu", "qc_metrics", "filter", "normalize", "log1p", "hvg",
         "scale", "pca", "neighbors", "umap", "leiden"]


# --------------------------------------------------------------------------- data
def gene_symbols(var):
    """CELLxGENE files use Ensembl IDs as var_names and keep symbols in a column."""
    for col in ("feature_name", "gene_symbols", "gene_symbol", "gene_name"):
        if col in var:
            return var[col].astype(str)
    return pd.Series(var.index, index=var.index)


def read_h5ad_counts(path):
    """Read only the raw count matrix from an .h5ad, without loading normalized X or embeddings.
    CELLxGENE files keep normalized values in X and raw counts in raw/X; loading both doubles RAM use."""
    import h5py
    from anndata.io import read_elem
    with h5py.File(path, "r") as f:
        main_var = read_elem(f["var"])
        obs = read_elem(f["obs"])
        labels = obs[[c for c in LABEL_COLS if c in obs]]
        if "raw" in f:
            print("Using raw counts from adata.raw")
            X, var = read_elem(f["raw/X"]), read_elem(f["raw/var"])
        else:
            X, var = read_elem(f["X"]), main_var
    return X, obs.index, var.index, gene_symbols(main_var), labels


def load_counts(path):
    if path is None:
        print("No --data given: using the small PBMC 3k demo dataset (smoke test only).")
        a = sc.datasets.pbmc3k()
        X, obs_names, var_names, symbols = a.X, a.obs_names, a.var_names, gene_symbols(a.var)
        labels = None
    elif path.endswith(".h5"):
        a = sc.read_10x_h5(path)
        X, obs_names, var_names, symbols = a.X, a.obs_names, a.var_names, gene_symbols(a.var)
        labels = None
    else:
        X, obs_names, var_names, symbols, labels = read_h5ad_counts(path)

    # Keep only the counts: no precomputed results (neighbors, UMAP, PCA...) from the file,
    # so both backends start from scratch and Scanpy does not pick up a stale graph.
    obs = labels.copy() if labels is not None else pd.DataFrame(index=obs_names)
    adata = sc.AnnData(X=X, obs=obs, var=pd.DataFrame(index=var_names))
    symbols = symbols.reindex(adata.var_names).fillna(pd.Series(adata.var_names, index=adata.var_names))
    adata.var["mt"] = symbols.str.upper().str.startswith("MT-").values
    adata.var_names_make_unique()

    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)
    adata.X = adata.X.tocsr().astype(np.float32)
    sample = adata.X.data[:10000]
    if sample.size and not np.allclose(sample, np.round(sample)):
        warnings.warn("X does not look like raw integer counts; results may be meaningless.")
    print(f"Loaded {adata.n_obs:,} cells x {adata.n_vars:,} genes "
          f"({int(adata.var['mt'].sum())} mitochondrial genes)")
    return adata


def apply_filters(adata, args):
    """Identical filtering logic for both backends (works on CPU and GPU AnnData)."""
    cells = (adata.obs["n_genes_by_counts"] >= args.min_genes) & \
            (adata.obs["pct_counts_mt"] < args.max_pct_mt)
    genes = (np.asarray(adata.var["n_cells_by_counts"] >= args.min_cells)
             if "n_cells_by_counts" in adata.var else slice(None))
    return adata[cells.values, genes].copy()  # one copy instead of two (saves RAM on large data)


# ---------------------------------------------------------------------- pipelines
def timed(times, name, fn, sync=None):
    start = time.perf_counter()
    out = fn()
    if sync is not None:
        sync()  # GPU kernels are asynchronous: wait for them before stopping the clock
    times[name] = time.perf_counter() - start
    return out


def run_cpu(adata, args):
    t = {}
    timed(t, "qc_metrics", lambda: sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True))
    adata = timed(t, "filter", lambda: apply_filters(adata, args))
    timed(t, "normalize", lambda: sc.pp.normalize_total(adata, target_sum=1e4))
    timed(t, "log1p", lambda: sc.pp.log1p(adata))

    def hvg():
        sc.pp.highly_variable_genes(adata, n_top_genes=args.n_top_genes)
        return adata[:, adata.var["highly_variable"].values].copy()
    adata = timed(t, "hvg", hvg)
    timed(t, "scale", lambda: sc.pp.scale(adata, max_value=10))
    timed(t, "pca", lambda: sc.pp.pca(adata, n_comps=args.n_pcs))
    timed(t, "neighbors", lambda: sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, n_pcs=args.n_pcs))
    timed(t, "umap", lambda: sc.tl.umap(adata))
    timed(t, "leiden", lambda: sc.tl.leiden(
        adata, resolution=args.resolution, flavor="igraph", n_iterations=2,
        directed=False, random_state=0))
    return adata, t


def setup_gpu_memory(managed):
    """Optionally use CUDA managed memory, which lets data spill to system RAM when 8 GB is not enough."""
    if not managed:
        return
    import cupy as cp
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator
    rmm.reinitialize(managed_memory=True)
    cp.cuda.set_allocator(rmm_cupy_allocator)


GPU_MEM = {"baseline": 0, "peak": 0}


def gpu_used_bytes():
    import cupy as cp
    free, total = cp.cuda.runtime.memGetInfo()
    return total - free


def reset_gpu_peak():
    GPU_MEM["baseline"] = GPU_MEM["peak"] = gpu_used_bytes()


def gpu_peak_gb():
    """Peak GPU memory used by this run, sampled after every step, minus what was in use before
    (e.g. by the Windows desktop). Spikes inside a step are not captured, so this is a lower bound."""
    return round((GPU_MEM["peak"] - GPU_MEM["baseline"]) / 1e9, 2)


def free_gpu_pool():
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()


def run_gpu(adata, args, t):
    """Fills `t` step by step, so after an out-of-memory error we know which step failed."""
    import cupy as cp
    import rapids_singlecell as rsc

    to_gpu = getattr(getattr(rsc, "get", None), "anndata_to_GPU", None) or rsc.utils.anndata_to_GPU
    to_cpu = getattr(getattr(rsc, "get", None), "anndata_to_CPU", None) or rsc.utils.anndata_to_CPU
    def sync():
        cp.cuda.runtime.deviceSynchronize()
        GPU_MEM["peak"] = max(GPU_MEM["peak"], gpu_used_bytes())

    timed(t, "to_gpu", lambda: to_gpu(adata), sync)
    timed(t, "qc_metrics", lambda: rsc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], log1p=False), sync)
    adata = timed(t, "filter", lambda: apply_filters(adata, args), sync)
    timed(t, "normalize", lambda: rsc.pp.normalize_total(adata, target_sum=1e4), sync)
    timed(t, "log1p", lambda: rsc.pp.log1p(adata), sync)

    def hvg():
        rsc.pp.highly_variable_genes(adata, n_top_genes=args.n_top_genes)
        return adata[:, adata.var["highly_variable"].values].copy()
    adata = timed(t, "hvg", hvg, sync)
    timed(t, "scale", lambda: rsc.pp.scale(adata, max_value=10), sync)
    timed(t, "pca", lambda: rsc.pp.pca(adata, n_comps=args.n_pcs), sync)
    timed(t, "neighbors", lambda: rsc.pp.neighbors(adata, n_neighbors=args.n_neighbors, n_pcs=args.n_pcs), sync)
    timed(t, "umap", lambda: rsc.tl.umap(adata), sync)
    timed(t, "leiden", lambda: rsc.tl.leiden(adata, resolution=args.resolution, random_state=0), sync)
    to_cpu(adata)  # not timed: only needed for comparison/plots
    return adata, t


# ------------------------------------------------------------------------ helpers
def system_info(use_gpu):
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "cpu": platform.processor() or platform.machine(), "cpu_cores": os.cpu_count(),
            "scanpy": sc.__version__}
    if use_gpu:
        import cupy as cp
        import rapids_singlecell as rsc
        props = cp.cuda.runtime.getDeviceProperties(0)
        info.update({"gpu": props["name"].decode(),
                     "gpu_memory_gb": round(props["totalGlobalMem"] / 1e9, 1),
                     "rapids_singlecell": rsc.__version__, "cupy": cp.__version__})
    return info


def keep_result(adata):
    """Keep only what the comparison needs, so the full AnnData can be freed before the next run."""
    r = {"leiden": adata.obs["leiden"].copy(), "umap": np.asarray(adata.obsm["X_umap"]).copy()}
    for col in LABEL_COLS:
        if col in adata.obs:
            r[col] = adata.obs[col].copy()
    return r


def save_result(r, path):
    """Save labels + UMAP coordinates so plots can be redrawn without rerunning the benchmark."""
    df = pd.DataFrame(r["umap"], columns=["umap1", "umap2"], index=r["leiden"].index)
    for k, v in r.items():
        if k != "umap":
            df[k] = v.astype(str).values
    df.to_csv(path, compression="gzip")


def plot_umaps(cpu, gpu, n, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Color both panels by the SAME labels: the published cell types if available, otherwise the CPU
    # clusters. (Coloring each panel by its own cluster numbers would make the colors incomparable.)
    if COLOR_COL in cpu:
        labels, legend_title = cpu[COLOR_COL], "published cell type"
    else:
        labels, legend_title = cpu["leiden"], "CPU cluster"
    cats = sorted(labels.astype(str).unique())
    cmap = plt.get_cmap("tab10" if len(cats) <= 10 else "tab20")
    colors = {c: cmap(i % cmap.N) for i, c in enumerate(cats)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, r, title in ((axes[0], cpu, "CPU (Scanpy)"), (axes[1], gpu, "GPU (rapids-singlecell)")):
        lab = labels.reindex(r["leiden"].index).astype(str)
        xy = r["umap"]
        ax.scatter(xy[:, 0], xy[:, 1], c=[colors.get(l, (0.8, 0.8, 0.8, 1)) for l in lab],
                   s=0.3, rasterized=True)
        ax.set_title(f"{title}: {r['leiden'].nunique()} Leiden clusters")
        ax.set_xticks([]); ax.set_yticks([])
    if len(cats) <= 20:
        handles = [plt.Line2D([], [], marker="o", ls="", color=colors[c], label=c) for c in cats]
        fig.legend(handles=handles, title=legend_title, loc="center right", fontsize=8, title_fontsize=9)
        fig.subplots_adjust(right=0.82)
    fig.suptitle(f"{n:,} cells, colored by {legend_title}")
    fig.savefig(out / f"umap_{n}.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help=".h5ad or 10x .h5 with raw counts (default: PBMC 3k demo)")
    p.add_argument("--sizes", default="10000,50000,100000", help="comma-separated cell counts to subsample")
    p.add_argument("--out", default="results")
    p.add_argument("--cpu-only", action="store_true", help="skip the GPU run (for smoke tests)")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--managed-memory", action="store_true",
                   help="use CUDA managed memory so datasets larger than GPU memory can still run")
    p.add_argument("--min-genes", type=int, default=200)
    p.add_argument("--min-cells", type=int, default=3)
    p.add_argument("--max-pct-mt", type=float, default=20.0)
    p.add_argument("--n-top-genes", type=int, default=2000)
    p.add_argument("--n-pcs", type=int, default=50)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--resolution", type=float, default=1.0)
    args = p.parse_args()

    from sklearn.metrics import adjusted_rand_score

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    use_gpu = not args.cpu_only
    if use_gpu:
        setup_gpu_memory(args.managed_memory)
    full = load_counts(args.data)
    info = system_info(use_gpu) | {"managed_memory": args.managed_memory}
    (out / "system.json").write_text(json.dumps(info, indent=2))

    sizes = [int(s) for s in args.sizes.split(",")]
    if max(sizes) > full.n_obs:
        print(f"Dataset has {full.n_obs:,} cells; dropping larger sizes.")
        sizes = [s for s in sizes if s <= full.n_obs] or [full.n_obs]

    if not args.no_warmup:  # first calls include JIT/kernel compilation; keep that out of the timings
        print("Warm-up run (not recorded)...")
        warm = sc.pp.subsample(full, n_obs=min(max(min(sizes), 8000), full.n_obs), random_state=1, copy=True)
        run_cpu(warm.copy(), args)
        if use_gpu:
            run_gpu(warm.copy(), args, {})
            free_gpu_pool()

    rows, agreement, memory = [], [], []
    sizes = sorted(sizes)
    for n in sizes:
        if n == full.n_obs and n == sizes[-1]:
            sub, full = full, None  # whole dataset, last run: reuse it instead of copying
        else:
            sub = sc.pp.subsample(full, n_obs=n, random_state=0, copy=True)
        print(f"\n=== {n:,} cells ===")
        cpu_adata, t_cpu = run_cpu(sub.copy(), args)
        cpu = keep_result(cpu_adata)
        del cpu_adata
        gc.collect()
        rows += [{"n_cells": n, "backend": "cpu", "step": k, "seconds": v} for k, v in t_cpu.items()]
        print(f"CPU total: {sum(t_cpu.values()):8.2f} s")
        if not use_gpu:
            continue
        t_gpu = {}
        reset_gpu_peak()
        try:
            gpu_adata, _ = run_gpu(sub, args, t_gpu)  # last use of `sub`, so no extra copy
            gpu = keep_result(gpu_adata)
            del gpu_adata, sub
            gc.collect()
        except Exception as e:  # typically cupy/rmm out-of-memory errors
            failed = next(s for s in STEPS if s not in t_gpu)
            print(f"GPU run FAILED at step '{failed}': {type(e).__name__}: {e}")
            memory.append({"n_cells": n, "status": "failed", "failed_step": failed,
                           "error": f"{type(e).__name__}: {str(e)[:200]}", "peak_gpu_gb": gpu_peak_gb()})
            pd.DataFrame(memory).to_csv(out / "gpu_memory.csv", index=False)
            free_gpu_pool()
            continue
        memory.append({"n_cells": n, "status": "ok", "failed_step": "", "error": "",
                       "peak_gpu_gb": gpu_peak_gb()})
        free_gpu_pool()
        rows += [{"n_cells": n, "backend": "gpu", "step": k, "seconds": v} for k, v in t_gpu.items()]
        print(f"GPU total: {sum(t_gpu.values()):8.2f} s  "
              f"(speedup {sum(t_cpu.values()) / sum(t_gpu.values()):.1f}x)")

        common = cpu["leiden"].index.intersection(gpu["leiden"].index)
        ari = adjusted_rand_score(cpu["leiden"].loc[common], gpu["leiden"].loc[common])
        agreement.append({"n_cells": n, "cells_compared": len(common), "ari": ari,
                          "clusters_cpu": cpu["leiden"].nunique(),
                          "clusters_gpu": gpu["leiden"].nunique()})
        if SCORE_COL in cpu:  # how well each backend recovers the published cell types
            for name, r in (("cpu", cpu), ("gpu", gpu)):
                agreement[-1][f"ari_{name}_vs_cell_type"] = adjusted_rand_score(
                    r[SCORE_COL].astype(str), r["leiden"])
        save_result(cpu, out / f"embedding_cpu_{n}.csv.gz")
        save_result(gpu, out / f"embedding_gpu_{n}.csv.gz")
        print(f"Cluster agreement (ARI): {ari:.3f}")
        plot_umaps(cpu, gpu, n, out)

        pd.DataFrame(rows).to_csv(out / "timings.csv", index=False)  # save after each size
        pd.DataFrame(agreement).to_csv(out / "agreement.csv", index=False)
        pd.DataFrame(memory).to_csv(out / "gpu_memory.csv", index=False)

    pd.DataFrame(rows).to_csv(out / "timings.csv", index=False)
    if agreement:
        pd.DataFrame(agreement).to_csv(out / "agreement.csv", index=False)
    print(f"\nResults written to {out}/")


if __name__ == "__main__":
    main()
