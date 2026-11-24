# Single-cell analysis on a laptop GPU: what an 8 GB RTX 5070 can (and can't) do with rapids-singlecell

_Draft outline: aim for 800-1,200 words. Reader: a biologist or student who uses Scanpy, has a laptop
with an NVIDIA GPU, and wonders whether it is worth using._

## 1. The question (1 paragraph)
- NVIDIA has shown big speedups with rapids-singlecell
  ([Dicks & Nolet, NVIDIA Technical Blog, June 2023](https://developer.nvidia.com/blog/gpu-accelerated-single-cell-rna-analysis-with-rapids-singlecell/)).
  Their benchmarks ran on a server with an NVIDIA A100 80 GB GPU (2x AMD EPYC 7543, 500 GB RAM)
  and on a desktop with an RTX 3090 (AMD Ryzen 9 5950X, 64 GB RAM).
  On ~90,000 cells the full notebook (excluding decoupler) ran 21x faster on the A100 and 16x faster on the RTX 3090.
- Both are big machines: the A100 has 80 GB of GPU memory and the RTX 3090 has 24 GB.
  Many students and small labs have a laptop with an 8 GB GPU instead.
- So: is the GPU already in your laptop enough to make a difference?
- Personal angle: at work I built RNA-seq and siRNA/ASO pipelines in Nextflow and profiled how
  resource-hungry steps like STAR alignment are on CPUs, [one sentence on why GPU acceleration interests you].
- Note for the comparison: NVIDIA's notebooks include more steps than mine (e.g. regress-out, Harmony,
  t-SNE, decoupler), so compare per-step speedups, not just the end-to-end number.

## 2. Setup (short, honest)
- Laptop: RTX 5070 Laptop GPU (8 GB) vs. Intel Core Ultra 9 275HX (16 threads visible to WSL2), Windows + WSL2.
- Dataset: NYGC multimodal PBMC (Hao et al. 2021, *Cell*) from CELLxGENE, 161,764 cells.
  Sizes tested: 10k / 50k / 100k / 161,764 cells.
- Same workflow and parameters on both sides; link to the repo for the details.

## 3. Getting it running (new-user experience)
- Steps that worked to install on an RTX 50-series GPU under WSL2.
- What went wrong and how I fixed it.
- One or two suggestions that would make the first run easier.

## 4. How much code changes?
- Side-by-side snippet: `sc.pp.pca(adata)` -> `rsc.pp.pca(adata)`.
- The only extra step is moving data to the GPU (`rsc.get.anndata_to_GPU`).

## 5. Speed
- Chart: total time per dataset size and the table.
- Chart: per-step breakdown. Which steps gain the most on a laptop GPU, and which barely change?
- Small vs. large datasets: where the GPU starts paying off.

## 6. The 8 GB wall
- Largest dataset that ran fully on the GPU; which step ran out of memory first.
- What helped (fewer genes, keeping data sparse, ...), and when you need a bigger GPU or the cloud.

## 7. Same biology?
- UMAPs side by side; Adjusted Rand Index between clusterings: [value]. One sentence on what ARI means.
- Why small differences happen (approximate nearest neighbors, different Leiden implementations).

## 8. Takeaways (3 bullets)
- Who should use their laptop GPU for single-cell analysis, and up to what dataset size.
- When to move to a bigger GPU.
- Next: GPU-accelerated alignment with Parabricks (`rna_fq2bam` vs. STAR).

_Code and full results: https://github.com/DavidGegic95/scrna-laptop-gpu-benchmark._
