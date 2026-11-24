# Single-cell analysis on a laptop GPU: what an 8 GB RTX 5070 can (and can't) do with rapids-singlecell

_Draft. Reader: a biologist or student who uses Scanpy, has a laptop with an NVIDIA GPU, and wonders
whether it is worth using._

## Is the GPU in your laptop enough?

NVIDIA has shown that [rapids-singlecell](https://rapids-singlecell.readthedocs.io) can speed up
single-cell RNA-seq analysis a lot
([Dicks & Nolet, NVIDIA Technical Blog, June 2023](https://developer.nvidia.com/blog/gpu-accelerated-single-cell-rna-analysis-with-rapids-singlecell/)).
On about 90,000 cells, their full notebook (excluding decoupler) ran 21x faster on an A100 80 GB server
GPU and 16x faster on a desktop RTX 3090. Those are big machines: the A100 has 80 GB of GPU memory and
the RTX 3090 has 24 GB. Many students and small labs have a laptop with an 8 GB GPU instead. So I asked
a simpler question: is the GPU already in your laptop enough to make a difference?

At work I built RNA-seq and siRNA/ASO pipelines in Nextflow and profiled how resource-hungry steps like
STAR alignment are on CPUs. Watching jobs queue for hours made me curious how much of that time a GPU
could win back.

## Setup

- **Laptop:** NVIDIA GeForce RTX 5070 Laptop GPU (8 GB) vs. Intel Core Ultra 9 275HX (16 threads
  visible to WSL2), Windows + WSL2, 27 GB RAM available to Linux.
- **Data:** the NYGC multimodal PBMC dataset (Hao et al. 2021, *Cell*) from CELLxGENE: 161,764 blood
  immune cells, raw counts. I ran random subsets of 10k, 50k and 100k cells, plus the full dataset.
- **Workflow:** QC, filtering, normalization, log1p, 2,000 highly variable genes, scaling, PCA (50 PCs),
  kNN graph (k=15), UMAP and Leiden clustering. Same cells and parameters for Scanpy (CPU) and
  rapids-singlecell (GPU). Copying data to the GPU counts toward the GPU time.

NVIDIA's notebooks include extra steps (regress-out, Harmony, t-SNE, decoupler), so the end-to-end
numbers are not directly comparable. The per-step numbers are.

## Getting it running

No CUDA toolkit install was needed. The normal Windows NVIDIA driver is enough: WSL2 picks it up, and
the pip packages bring their own CUDA libraries. Three things did trip me up:

1. **Python version.** My system Python (3.14) was too new for RAPIDS. A Python 3.12 environment
   (created with `uv`) worked.
2. **The package name.** `pip install rapids-singlecell` failed, because that version is published only
   as source code and building it needs a C++/CUDA compiler. The prebuilt package installed cleanly:
   ```bash
   pip install 'rapids-singlecell-cu12[rapids]' --extra-index-url=https://pypi.nvidia.com
   ```
   A clearer pointer to the `-cu12` package for pip users would save newcomers time.
3. **The data file, not the GPU.** CELLxGENE files ship normalized values, raw counts and precomputed
   results. The stale neighbor graph crashed `sc.pp.neighbors` (`KeyError: 'n_neighbors'`), and loading
   everything pushed WSL2 into swap. Reading only the raw counts fixed both.

One benchmarking lesson: the first CPU kNN call took 14 s, almost all of it one-time compilation. Warm
up at the size you benchmark, or the CPU looks slower than it is.

## How much code changes?

Almost none. You move the data to the GPU once, then swap `sc` for `rsc`:

```python
import scanpy as sc                    import rapids_singlecell as rsc

                                       rsc.get.anndata_to_GPU(adata)
sc.pp.normalize_total(adata)           rsc.pp.normalize_total(adata)
sc.pp.log1p(adata)                     rsc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)     rsc.pp.highly_variable_genes(adata)
sc.pp.pca(adata)                       rsc.pp.pca(adata)
sc.pp.neighbors(adata)                 rsc.pp.neighbors(adata)
sc.tl.umap(adata)                      rsc.tl.umap(adata)
sc.tl.leiden(adata)                    rsc.tl.leiden(adata)
```

The result is a normal AnnData object, so plotting and downstream analysis stay the same.

## Speed: about 9x faster, from 10k cells up

| cells | CPU (s) | GPU (s) | speedup |
|---:|---:|---:|---:|
| 10,000 | 9.5 | 1.0 | 9.1x |
| 50,000 | 27.7 | 2.7 | 10.3x |
| 100,000 | 65.2 | 7.9 | 8.3x |
| 161,764 | 115.1 | 12.8 | 9.0x |

![Total time](results/total_time.png)

The full dataset went from almost two minutes to under 13 seconds. Across two full runs the end-to-end
speedup ranged from about 8x to 12x. The GPU already paid off at 10,000 cells, so you don't need a huge
dataset to benefit.

![Per step](results/per_step.png)

The per-step picture explains the total:

- **UMAP and PCA** take about 73% of the CPU time and get the biggest gains: 49x and 45x. These are
  the steps you rerun most when you tune parameters, so this is where the time saved adds up.
- **QC metrics, normalization and scaling** are 18-55x faster, but they were already quick on the CPU.
- **kNN graph (4x) and Leiden (2x)** gain much less. At 10,000 cells, Leiden was slower on the GPU
  (0.35 s vs. 0.11 s).
- **Copying data to the GPU** is now the largest single GPU cost: about 4 s of the 12.8 s. If you load
  once and keep the data on the GPU while you explore, you pay it once.

## The 8 GB wall

I expected to run out of memory, but I didn't. The full 161,764-cell dataset ran on the GPU with a peak
of 7.0 GB.

| cells | peak GPU memory |
|---:|---:|
| 10,000 | 0.4 GB |
| 50,000 | 2.2 GB |
| 100,000 | 4.3 GB |
| 161,764 | 7.0 GB |

Memory grew almost linearly, about 43 MB per 1,000 cells. Extrapolating, about 180,000 cells is the
ceiling for this workflow on 8 GB. That is an estimate, not a measurement. It also depends on how many
genes are detected per cell (about 2,200 here). Past that point you have three options: process the
data in batches, let CUDA spill to system RAM with managed memory (slower, and not benchmarked here),
or use a bigger GPU in the cloud.

In my runs, system RAM was as much of a limit as GPU memory. Keep the data sparse and avoid extra
copies of the matrix.

## Same biology?

![UMAP comparison](results/umap_161764.png)

Both UMAPs are colored by the published cell types, so the colors mean the same thing in both panels.
The CPU and GPU separate the same populations: B cells, monocytes, NK cells, CD4 and CD8 T cells, and
dendritic cells.

To put a number on it I used the Adjusted Rand Index (ARI), which is 1.0 when two groupings are
identical and about 0 when they are random. On the full dataset, the CPU and GPU clusters had an ARI of
0.80 (0.91 at 10k cells). They are not identical, because the nearest-neighbor search, UMAP and Leiden
are implemented differently on each side, and small differences add up as datasets grow. What matters
more is that both recover the published cell types equally well: ARI 0.47 for CPU clusters and 0.50
for GPU clusters. The GPU doesn't lose biology.

## Takeaways

- **If you have a laptop with an 8 GB NVIDIA GPU, use it.** A standard Scanpy workflow ran about 9x
  faster on datasets from 10,000 to about 160,000 cells, with only a few lines of code changed.
- **Move to a bigger GPU** when your dataset grows past roughly 150,000-180,000 cells, or when you add
  memory-hungry steps like batch correction on large data.
- **Next:** single-cell analysis starts after alignment. I'll look at GPU-accelerated alignment with
  NVIDIA Parabricks (`rna_fq2bam` vs. STAR).

_Code and full results: https://github.com/DavidGegic95/scrna-laptop-gpu-benchmark._
