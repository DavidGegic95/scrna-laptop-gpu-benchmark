# Single-cell analysis on a laptop GPU

**What can an 8 GB laptop GPU do with rapids-singlecell?**

**Status:** benchmark done; blog post in progress.

**TL;DR** On a laptop with an RTX 5070 Laptop GPU (8 GB), a standard Scanpy workflow on 161,764 cells
took **~13 s on the GPU vs. ~115 s on a 16-thread laptop CPU (about 9x faster)**. The whole dataset fit in
8 GB (peak ~7 GB), and GPU clusters matched the published cell types as well as CPU clusters did.

NVIDIA has already shown that [rapids-singlecell](https://rapids-singlecell.readthedocs.io) speeds up
single-cell RNA-seq analysis
([NVIDIA blog post](https://developer.nvidia.com/blog/gpu-accelerated-single-cell-rna-analysis-with-rapids-singlecell/)).
This repo asks the practical follow-up for students and small labs: **is a consumer laptop GPU enough?**

It runs the same workflow on a laptop's CPU ([Scanpy](https://scanpy.readthedocs.io)) and on its GPU
(rapids-singlecell) and answers three questions:

1. **Speed:** how much faster is each step on the laptop GPU vs. a 16-thread laptop CPU?
2. **Limits:** how many cells fit in 8 GB of GPU memory?
3. **Correctness:** do the CPU and GPU find the same cell populations?

It also includes **new-user notes**: what it took to get the GPU stack running on a brand-new GPU under WSL2.

## Setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU, 8 GB (driver 577.05) |
| CPU | Intel Core Ultra 9 275HX, 16 threads visible to WSL2 |
| RAM | 27 GB available to WSL2 |
| OS | Windows + WSL2 (Linux) |
| Software | Python 3.12, Scanpy 1.12.6, rapids-singlecell 0.18.1, CuPy 14.2 (see `results/system.json`) |
| Dataset | NYGC multimodal PBMC, Hao et al. 2021 (*Cell*), from [CELLxGENE](https://cellxgene.cziscience.com/): 161,764 blood immune cells x 20,264 genes, raw counts |

## What is compared

| Step | CPU | GPU |
|---|---|---|
| QC metrics, filtering | `sc.pp.calculate_qc_metrics` | `rsc.pp.calculate_qc_metrics` |
| Normalize + log1p | `sc.pp.normalize_total`, `sc.pp.log1p` | `rsc.pp.normalize_total`, `rsc.pp.log1p` |
| Highly variable genes (2,000) | `sc.pp.highly_variable_genes` | `rsc.pp.highly_variable_genes` |
| Scale, PCA (50 PCs) | `sc.pp.scale`, `sc.pp.pca` | `rsc.pp.scale`, `rsc.pp.pca` |
| kNN graph (k=15), UMAP | `sc.pp.neighbors`, `sc.tl.umap` | `rsc.pp.neighbors`, `rsc.tl.umap` |
| Leiden clustering (resolution 1.0) | `sc.tl.leiden` (igraph) | `rsc.tl.leiden` (cuGraph) |

Fairness rules:
- Same cells (fixed random subsample), same parameters, same filtering code for both backends.
- A warm-up run at the smallest benchmark size (different random cells) first, so one-time compilation is not counted.
- GPU timings wait for the GPU to finish each step (`cudaDeviceSynchronize`), since GPU work is asynchronous.
- Copying data to the GPU is timed and included in the GPU total.

## Results

### 1. Speed

| cells | CPU (s) | GPU (s) | speedup |
|---:|---:|---:|---:|
| 10,000 | 9.5 | 1.0 | 9.1x |
| 50,000 | 27.7 | 2.7 | 10.3x |
| 100,000 | 65.2 | 7.9 | 8.3x |
| 161,764 | 115.1 | 12.8 | 9.0x |

Single run shown; across two full runs, end-to-end speedups ranged from about 8x to 12x.

![Total time](results/total_time.png)

Where the time goes at 161,764 cells:

| step | CPU (s) | GPU (s) | speedup |
|---|---:|---:|---:|
| copy to GPU | - | 4.04 | - |
| QC metrics | 3.42 | 0.07 | 52x |
| filter | 5.05 | 1.19 | 4x |
| normalize | 0.49 | 0.01 | 55x |
| log1p | 0.85 | 0.96 | 0.9x |
| highly variable genes | 10.39 | 1.13 | 9x |
| scale | 1.35 | 0.08 | 18x |
| PCA | 23.49 | 0.53 | 45x |
| kNN graph | 5.37 | 1.37 | 4x |
| UMAP | 60.42 | 1.24 | 49x |
| Leiden | 4.28 | 2.18 | 2x |

![Per step](results/per_step.png)

Observations:
- **UMAP and PCA dominate on the CPU** (~73% of CPU time) and get the biggest GPU speedups (~45-50x).
- **Copying data to the GPU is now the largest single GPU cost** (~4 s of ~13 s). Loading data straight
  onto the GPU, or keeping it there across analyses, would matter more than faster kernels.
- **Leiden barely benefits on a laptop**, and at 10k cells it was slower on the GPU (0.35 s vs. 0.11 s).
- The full table for every size is in `results/timings.csv`.

### 2. How far does 8 GB go?

| cells | GPU run | peak GPU memory |
|---:|---|---:|
| 10,000 | ok | 0.4 GB |
| 50,000 | ok | 2.2 GB |
| 100,000 | ok | 4.3 GB |
| 161,764 | ok | 7.0 GB |

The whole dataset fit. Memory grew almost linearly (~43 MB per 1,000 cells for this dataset), so
extrapolating, about 180,000 cells is the ceiling for this workflow on 8 GB. That is an estimate, not
a measurement, and it depends on how many genes are detected per cell (~2,200 here). Peak memory is sampled
after each step, so short spikes inside a step may be missed. For bigger data, `--managed-memory` lets
CUDA spill to system RAM (not yet benchmarked here).

### 3. Do the results agree?

![UMAP comparison](results/umap_161764.png)

Both UMAPs are colored by the **published** cell types (not by our clusters), so the colors are directly
comparable: the CPU and GPU separate the same populations (B cells, monocytes, NK cells, CD4/CD8 T cells, DCs).

| cells | ARI CPU vs. GPU clusters | ARI CPU clusters vs. published types | ARI GPU clusters vs. published types |
|---:|---:|---:|---:|
| 10,000 | 0.91 | 0.52 | 0.52 |
| 50,000 | 0.87 | 0.52 | 0.50 |
| 100,000 | 0.84 | 0.48 | 0.53 |
| 161,764 | 0.80 | 0.47 | 0.50 |

ARI (Adjusted Rand Index) is 1.0 for identical groupings and ~0 for random ones. The CPU and GPU clusters are
not identical, because the kNN search, UMAP and Leiden implementations differ. They recover the published cell types
**equally well**, so the GPU does not lose biology. (The ARI against published types is moderate for both,
because Leiden at resolution 1.0 splits cells into a different number of groups than the 30 annotated types.)

## New-user notes

What it took to get this running on an RTX 50-series laptop GPU under WSL2:

1. **No CUDA toolkit install was needed.** Only the normal Windows NVIDIA driver; WSL2 picks it up and the
   pip packages bring the CUDA libraries.
2. **Use Python 3.12.** The system Python (3.14) was too new for the RAPIDS packages; `uv` made a 3.12 env easy.
3. **`pip install rapids-singlecell` failed**: version 0.18.1 is published only as source code, and building
   it needs a C++/CUDA compiler. The prebuilt package worked:
   `pip install 'rapids-singlecell-cu12[rapids]' --extra-index-url=https://pypi.nvidia.com`.
   A clearer pointer to the `-cu12` package for pip users would help newcomers.
4. **CELLxGENE files ship precomputed results** (neighbors, UMAP), and `sc.pp.neighbors` crashed on the stale
   graph (`KeyError: 'n_neighbors'`). Starting from the raw counts only fixed it.
5. **RAM matters as much as GPU memory.** These files store normalized values *and* raw counts, with 64-bit
   indices; loading everything briefly needed >10 GB of RAM, and extra copies pushed WSL2 into swap, which
   made timings erratic. Reading only `raw/X` and avoiding copies fixed it.
6. **Benchmarking pitfall:** the first CPU kNN call at 10k cells took 14 s, almost all one-time compilation.
   Smaller warm-ups (2,000 and 8,000 cells) did not trigger the same code path; warming up at the benchmark size did.

## Run it yourself

```bash
# CPU-only smoke test on the PBMC 3k demo data
pip install -r requirements.txt
python benchmark.py --cpu-only --sizes 2000

# Full benchmark (needs an NVIDIA GPU + rapids-singlecell, see below)
python benchmark.py --data data/nygc_pbmc_161k.h5ad --sizes 10000,50000,100000
python plot_results.py
```

**GPU setup:** with Python 3.12, the prebuilt CUDA 12 package worked:

```bash
pip install 'rapids-singlecell-cu12[rapids]' --extra-index-url=https://pypi.nvidia.com
```

See also the rapids-singlecell install guide
(<https://rapids-singlecell.readthedocs.io/en/latest/Installation.html>) and the notes above.
No local GPU? A Google Colab GPU runtime works too, but note that Colab's CPU has only 2 cores,
which makes the CPU baseline look slower than on a typical workstation.

**Data:** any `.h5ad` or 10x `.h5` file with raw counts. If the file stores normalized values in `X` and
raw counts in `.raw` (common for CELLxGENE downloads), the script uses `.raw` automatically.

## Code layout

| File | What it does |
|---|---|
| `benchmark.py`, `plot_results.py` | Command-line entry points |
| `scbench/workflow.py` | The workflow, written once for both backends, with per-step timing |
| `scbench/backends.py` | CPU (Scanpy) and GPU (rapids-singlecell) backends: which library, and the few backend-specific arguments |
| `scbench/data.py` | Loads raw counts and published cell types |
| `scbench/gpu.py` | GPU memory tracking and managed-memory setup |
| `scbench/results.py` | Cluster agreement (ARI) and the CSV outputs |
| `scbench/plots.py` | UMAP comparison, timing charts and the Markdown summary |
| `scbench/cli.py` | Command-line options and the loop over dataset sizes |

## Write-up

See [`blog_post.md`](blog_post.md).
