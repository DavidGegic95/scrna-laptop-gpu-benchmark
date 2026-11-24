"""Load raw counts into a fresh AnnData that both backends start from."""
import warnings

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from anndata.io import read_elem
from scipy import sparse

# Published annotations, if the file has them: broad labels to color plots,
# detailed labels to score how well each backend's clusters recover known cell types.
COLOR_COL = "celltype.l1"
SCORE_COL = "cell_type"
LABEL_COLS = [COLOR_COL, SCORE_COL]

# CELLxGENE files use Ensembl IDs as var_names and keep gene symbols in one of these columns.
SYMBOL_COLS = ("feature_name", "gene_symbols", "gene_symbol", "gene_name")


def load_counts(path=None):
    """Return an AnnData holding only raw counts (CSR, float32), published labels and an `mt` gene flag.

    Precomputed results in the file (neighbors, UMAP, PCA...) are dropped, so both backends start
    from scratch and Scanpy does not pick up a stale graph.
    """
    if path is None:
        print("No --data given: using the small PBMC 3k demo dataset (smoke test only).")
        parts = _counts_from_anndata(sc.datasets.pbmc3k())
    elif path.endswith(".h5"):
        parts = _counts_from_anndata(sc.read_10x_h5(path))
    elif _is_current_h5ad_format(path):
        parts = _read_h5ad_counts(path)
    else:  # files written by anndata < 0.7 can't be read piece by piece
        parts = _counts_from_anndata(sc.read_h5ad(path))
    return _build(*parts)


def _is_current_h5ad_format(path):
    with h5py.File(path, "r") as f:
        return "encoding-type" in f["obs"].attrs


def _read_h5ad_counts(path):
    """Read only the count matrix, without loading normalized X or embeddings.
    CELLxGENE files keep normalized values in X and raw counts in raw/X; loading both doubles RAM use."""
    with h5py.File(path, "r") as f:
        obs, var = read_elem(f["obs"]), read_elem(f["var"])
        if "raw" in f:
            print("Using raw counts from adata.raw")
            X, counts_var = read_elem(f["raw/X"]), read_elem(f["raw/var"])
        else:
            X, counts_var = read_elem(f["X"]), var
    return X, _labels(obs), counts_var.index, _gene_symbols(var)


def _counts_from_anndata(adata):
    if adata.raw is not None:
        print("Using raw counts from adata.raw")
        X, counts_var_names = adata.raw.X, adata.raw.var_names
    else:
        X, counts_var_names = adata.X, adata.var_names
    return X, _labels(adata.obs), counts_var_names, _gene_symbols(adata.var)


def _labels(obs):
    return obs[[c for c in LABEL_COLS if c in obs.columns]].copy()


def _gene_symbols(var):
    for col in SYMBOL_COLS:
        if col in var:
            return var[col].astype(str)
    return pd.Series(var.index, index=var.index)


def _build(X, obs, var_names, symbols):
    adata = sc.AnnData(X=X, obs=obs, var=pd.DataFrame(index=var_names))
    # Genes missing from the symbol table (e.g. only in raw) fall back to their ID.
    symbols = symbols.reindex(adata.var_names).fillna(pd.Series(adata.var_names, index=adata.var_names))
    adata.var["mt"] = symbols.str.upper().str.startswith("MT-").values
    adata.var_names_make_unique()

    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)
    adata.X = _as_float32_csr_with_int32_indices(adata.X)
    _warn_if_not_counts(adata.X)
    print(f"Loaded {adata.n_obs:,} cells x {adata.n_vars:,} genes "
          f"({int(adata.var['mt'].sum())} mitochondrial genes)")
    return adata


def _as_float32_csr_with_int32_indices(X):
    """CELLxGENE files store 64-bit sparse indices. GPU sparse matrices use 32-bit indices, so leaving them
    64-bit makes the copy to the GPU ~4x slower (an extra conversion) and changes GPU Leiden results.
    Subsampled data gets 32-bit indices anyway; this makes the full dataset match."""
    X = X.tocsr().astype(np.float32, copy=False)
    if X.nnz < np.iinfo(np.int32).max:
        X.indices = X.indices.astype(np.int32, copy=False)
        X.indptr = X.indptr.astype(np.int32, copy=False)
    return X


def _warn_if_not_counts(X):
    sample = X.data[:10000]
    if sample.size and not np.allclose(sample, np.round(sample)):
        warnings.warn("X does not look like raw integer counts; results may be meaningless.")
