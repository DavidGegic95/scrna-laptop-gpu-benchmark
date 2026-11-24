"""What is kept from each run, how CPU and GPU results are compared, and the CSV outputs."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .data import LABEL_COLS, SCORE_COL


@dataclass
class Clustering:
    """Only what the comparison needs, so the full AnnData can be freed before the next run."""
    leiden: pd.Series
    umap: np.ndarray
    labels: pd.DataFrame  # published annotations; may have no columns

    @classmethod
    def from_adata(cls, adata):
        labels = adata.obs[[c for c in LABEL_COLS if c in adata.obs.columns]].copy()
        return cls(adata.obs["leiden"].copy(), np.asarray(adata.obsm["X_umap"]).copy(), labels)

    def save(self, path):
        """Labels + UMAP coordinates, so plots can be redrawn without rerunning the benchmark."""
        df = pd.DataFrame(self.umap, columns=["umap1", "umap2"], index=self.leiden.index)
        df["leiden"] = self.leiden.astype(str).values
        for col in self.labels.columns:
            df[col] = self.labels[col].astype(str).values
        df.to_csv(path, compression="gzip")


def agreement(cpu, gpu):
    """Adjusted Rand Index between CPU and GPU clusters and, if available, against published cell types."""
    common = cpu.leiden.index.intersection(gpu.leiden.index)
    row = {"cells_compared": len(common),
           "ari": adjusted_rand_score(cpu.leiden.loc[common], gpu.leiden.loc[common]),
           "clusters_cpu": cpu.leiden.nunique(),
           "clusters_gpu": gpu.leiden.nunique()}
    if SCORE_COL in cpu.labels:
        for name, c in (("cpu", cpu), ("gpu", gpu)):
            row[f"ari_{name}_vs_cell_type"] = adjusted_rand_score(c.labels[SCORE_COL].astype(str), c.leiden)
    return row


class ResultTables:
    """Rows for timings.csv, agreement.csv and gpu_memory.csv, saved after each dataset size."""

    def __init__(self, out):
        self.out = out
        self.tables = {"timings": [], "agreement": [], "gpu_memory": []}

    def add_timings(self, n_cells, backend, times):
        self.tables["timings"] += [{"n_cells": n_cells, "backend": backend, "step": k, "seconds": v}
                                   for k, v in times.items()]

    def add_agreement(self, n_cells, row):
        self.tables["agreement"].append({"n_cells": n_cells, **row})

    def add_gpu_memory(self, n_cells, peak_gb, failed_step="", error=""):
        self.tables["gpu_memory"].append({"n_cells": n_cells, "status": "failed" if failed_step else "ok",
                                          "failed_step": failed_step, "error": error, "peak_gpu_gb": peak_gb})

    def save(self):
        for name, rows in self.tables.items():
            if rows:
                pd.DataFrame(rows).to_csv(self.out / f"{name}.csv", index=False)
