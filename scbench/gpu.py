"""GPU memory setup and tracking. Only imported when the GPU run is enabled."""
import cupy as cp


def use_managed_memory():
    """CUDA managed memory lets data spill to system RAM when GPU memory is not enough."""
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator

    rmm.reinitialize(managed_memory=True)
    cp.cuda.set_allocator(rmm_cupy_allocator)


def free_memory_pool():
    cp.get_default_memory_pool().free_all_blocks()


def device_info():
    import rapids_singlecell as rsc

    props = cp.cuda.runtime.getDeviceProperties(0)
    return {"gpu": props["name"].decode(),
            "gpu_memory_gb": round(props["totalGlobalMem"] / 1e9, 1),
            "rapids_singlecell": rsc.__version__, "cupy": cp.__version__}


def _used_bytes():
    free, total = cp.cuda.runtime.memGetInfo()
    return total - free


class GpuMemoryTracker:
    """Peak GPU memory used by one run, minus what was in use before it (e.g. by the Windows desktop).
    Sampled after each step, so spikes inside a step are missed: the result is a lower bound."""

    def __init__(self):
        self.baseline = self.peak = 0

    def reset(self):
        self.baseline = self.peak = _used_bytes()

    def sample(self):
        self.peak = max(self.peak, _used_bytes())

    @property
    def peak_gb(self):
        return round((self.peak - self.baseline) / 1e9, 2)
