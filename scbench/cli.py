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
from dataclasses import fields
from pathlib import Path

import scanpy as sc

from .backends import cpu_backend, gpu_backend
from .data import load_counts
from .plots import plot_umaps
from .results import Clustering, ResultTables, agreement
from .workflow import StepFailed, WorkflowParams, run_workflow

WARMUP_MIN_CELLS = 8000  # smaller warm-ups missed some one-time compilation (see README)


def parse_args(argv=None):
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
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    params = WorkflowParams(**{f.name: getattr(args, f.name) for f in fields(WorkflowParams)})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cpu, gpu, memory = cpu_backend(), None, None
    if not args.cpu_only:
        from . import gpu as gpu_utils
        if args.managed_memory:
            gpu_utils.use_managed_memory()
        memory = gpu_utils.GpuMemoryTracker()
        gpu = gpu_backend(memory)

    full = load_counts(args.data)
    write_system_info(out / "system.json", with_gpu=gpu is not None, managed_memory=args.managed_memory)
    sizes = usable_sizes(args.sizes, full.n_obs)
    if not args.no_warmup:
        warm_up(full, sizes, cpu, gpu, params)

    tables = ResultTables(out)
    for n in sizes:
        if n == full.n_obs and n == sizes[-1]:
            sub, full = full, None  # whole dataset, last run: reuse it instead of copying
        else:
            sub = sc.pp.subsample(full, n_obs=n, random_state=0, copy=True)
        print(f"\n=== {n:,} cells ===")

        cpu_result, cpu_times = run(sub.copy(), cpu, params)
        tables.add_timings(n, "cpu", cpu_times)
        print(f"CPU total: {sum(cpu_times.values()):8.2f} s")

        if gpu is not None:
            memory.reset()
            try:
                gpu_result, gpu_times = run(sub, gpu, params)  # last use of `sub`, so no extra copy
            except StepFailed as e:  # typically cupy/rmm out-of-memory errors
                err = f"{type(e.__cause__).__name__}: {e.__cause__}"
                print(f"GPU run FAILED at step '{e.step}': {err}")
                tables.add_gpu_memory(n, memory.peak_gb, failed_step=e.step, error=err[:200])
                gpu_result = None
            else:
                tables.add_gpu_memory(n, memory.peak_gb)
                tables.add_timings(n, "gpu", gpu_times)
                print(f"GPU total: {sum(gpu_times.values()):8.2f} s  "
                      f"(speedup {sum(cpu_times.values()) / sum(gpu_times.values()):.1f}x)")
            del sub
            gc.collect()
            gpu_utils.free_memory_pool()
            if gpu_result is not None:
                compare(n, cpu_result, gpu_result, tables, out)

        tables.save()  # after every size, so a crash later keeps what finished
    print(f"\nResults written to {out}/")


def run(adata, backend, params):
    """Run the workflow and keep only what the comparison needs, so the AnnData is freed on return."""
    result_adata, times = run_workflow(adata, backend, params)
    return Clustering.from_adata(result_adata), times


def compare(n, cpu_result, gpu_result, tables, out):
    row = agreement(cpu_result, gpu_result)
    tables.add_agreement(n, row)
    print(f"Cluster agreement (ARI): {row['ari']:.3f}")
    cpu_result.save(out / f"embedding_cpu_{n}.csv.gz")
    gpu_result.save(out / f"embedding_gpu_{n}.csv.gz")
    plot_umaps(cpu_result, gpu_result, n, out)


def usable_sizes(sizes_arg, n_available):
    sizes = sorted(int(s) for s in sizes_arg.split(","))
    if sizes[-1] > n_available:
        print(f"Dataset has {n_available:,} cells; dropping larger sizes.")
        sizes = [s for s in sizes if s <= n_available] or [n_available]
    return sizes


def warm_up(full, sizes, cpu, gpu, params):
    """First calls include JIT/kernel compilation; run once on other cells to keep that out of the timings."""
    print("Warm-up run (not recorded)...")
    n = min(max(sizes[0], WARMUP_MIN_CELLS), full.n_obs)
    warm = sc.pp.subsample(full, n_obs=n, random_state=1, copy=True)
    run_workflow(warm.copy(), cpu, params)
    if gpu is not None:
        run_workflow(warm, gpu, params)
        from .gpu import free_memory_pool
        free_memory_pool()


def write_system_info(path, with_gpu, managed_memory):
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "cpu": platform.processor() or platform.machine(), "cpu_cores": os.cpu_count(),
            "scanpy": sc.__version__}
    if with_gpu:
        from .gpu import device_info
        info |= device_info()
    info["managed_memory"] = managed_memory
    path.write_text(json.dumps(info, indent=2))
