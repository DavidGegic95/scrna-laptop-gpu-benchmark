"""The workflow being benchmarked, written once and run on either backend.

QC metrics -> filtering -> normalize -> log1p -> HVG selection -> scale -> PCA
-> kNN graph -> UMAP -> Leiden clustering.
"""
import time
from dataclasses import dataclass

import numpy as np

STEPS = ["to_gpu", "qc_metrics", "filter", "normalize", "log1p", "hvg",
         "scale", "pca", "neighbors", "umap", "leiden"]


@dataclass(frozen=True)
class WorkflowParams:
    min_genes: int = 200
    min_cells: int = 3
    max_pct_mt: float = 20.0
    n_top_genes: int = 2000
    n_pcs: int = 50
    n_neighbors: int = 15
    resolution: float = 1.0


class StepFailed(RuntimeError):
    """A workflow step raised; the original error is the __cause__ (typically GPU out-of-memory)."""

    def __init__(self, step):
        super().__init__(f"step '{step}' failed")
        self.step = step


SEED_CHECK_KEY = "leiden_seed1"


def run_workflow(adata, backend, params, seed_check=False):
    """Run every step and time it. `adata` is modified (on the GPU: moved to the device), so pass
    a copy if you still need it. Returns (result AnnData on the host, {step: seconds}).

    seed_check: also run Leiden with a different seed on the same graph (not timed), stored in
    obs[SEED_CHECK_KEY]. Shows how much clusters change from Leiden's randomness alone."""
    times = {}

    def step(name, fn):
        start = time.perf_counter()
        try:
            out = fn()
            if backend.sync:
                backend.sync()  # GPU kernels are asynchronous: wait for them before stopping the clock
        except Exception as e:
            raise StepFailed(name) from e
        times[name] = time.perf_counter() - start
        return out

    pp, tl, p = backend.pp, backend.tl, params
    if backend.to_device:
        step("to_gpu", lambda: backend.to_device(adata))
    step("qc_metrics", lambda: pp.calculate_qc_metrics(adata, qc_vars=["mt"], log1p=False, **backend.qc_kwargs))
    adata = step("filter", lambda: filter_cells_and_genes(adata, p))
    step("normalize", lambda: pp.normalize_total(adata, target_sum=1e4))
    step("log1p", lambda: pp.log1p(adata))
    adata = step("hvg", lambda: keep_highly_variable_genes(pp, adata, p.n_top_genes))
    step("scale", lambda: pp.scale(adata, max_value=10))
    step("pca", lambda: pp.pca(adata, n_comps=p.n_pcs))
    step("neighbors", lambda: pp.neighbors(adata, n_neighbors=p.n_neighbors, n_pcs=p.n_pcs))
    step("umap", lambda: tl.umap(adata))
    step("leiden", lambda: tl.leiden(adata, resolution=p.resolution, random_state=0, **backend.leiden_kwargs))
    if seed_check:
        tl.leiden(adata, resolution=p.resolution, random_state=1, key_added=SEED_CHECK_KEY,
                  **backend.leiden_kwargs)

    if backend.to_host:
        backend.to_host(adata)  # not timed: only needed for comparison and plots
    return adata, times


def filter_cells_and_genes(adata, params):
    """Same filtering code for both backends (works on CPU and GPU AnnData)."""
    cells = ((adata.obs["n_genes_by_counts"] >= params.min_genes)
             & (adata.obs["pct_counts_mt"] < params.max_pct_mt)).values
    genes = (np.asarray(adata.var["n_cells_by_counts"] >= params.min_cells)
             if "n_cells_by_counts" in adata.var else slice(None))
    return adata[cells, genes].copy()  # one copy of the subset view (saves RAM on large data)


def keep_highly_variable_genes(pp, adata, n_top_genes):
    pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    return adata[:, adata.var["highly_variable"].values].copy()
