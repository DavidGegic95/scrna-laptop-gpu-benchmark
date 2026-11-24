"""CPU and GPU backends: the same workflow, different libraries."""
from dataclasses import dataclass, field
from types import ModuleType
from typing import Callable, Optional

import scanpy as sc


@dataclass(frozen=True)
class Backend:
    name: str
    pp: ModuleType
    tl: ModuleType
    qc_kwargs: dict = field(default_factory=dict)      # arguments only one library accepts
    leiden_kwargs: dict = field(default_factory=dict)
    to_device: Optional[Callable] = None  # move data to the device before the first step (timed)
    to_host: Optional[Callable] = None    # move results back after the last step (not timed)
    sync: Optional[Callable] = None       # wait for asynchronous work at the end of each step


def cpu_backend():
    return Backend(
        name="cpu", pp=sc.pp, tl=sc.tl,
        qc_kwargs={"percent_top": None, "inplace": True},
        leiden_kwargs={"flavor": "igraph", "n_iterations": 2, "directed": False},
    )


def gpu_backend(memory):
    """`memory` is a GpuMemoryTracker, sampled after every step."""
    import cupy as cp
    import rapids_singlecell as rsc

    def sync():
        cp.cuda.runtime.deviceSynchronize()
        memory.sample()

    return Backend(
        name="gpu", pp=rsc.pp, tl=rsc.tl,
        to_device=rsc.get.anndata_to_GPU, to_host=rsc.get.anndata_to_CPU, sync=sync,
    )
