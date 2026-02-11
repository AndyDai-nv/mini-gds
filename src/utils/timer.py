"""Performance profiling utilities."""

import time
import psutil
import torch
from contextlib import contextmanager
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Timer:
    """High-precision timer for benchmarking."""

    name: str = "Timer"
    elapsed: float = 0.0
    _start_time: Optional[float] = field(default=None, repr=False)
    _start_gpu_mem: Optional[int] = field(default=None, repr=False)

    def __enter__(self):
        """Start timing."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self._start_gpu_mem = torch.cuda.memory_allocated()

        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        """Stop timing and compute elapsed time."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.elapsed = time.perf_counter() - self._start_time

    def __str__(self) -> str:
        return f"{self.name}: {self.elapsed:.4f}s"


@contextmanager
def profile_memory(device: str = "cuda:0"):
    """Profile GPU memory usage.

    Args:
        device: CUDA device to profile

    Yields:
        Dict containing memory statistics
    """
    if not torch.cuda.is_available():
        yield {"error": "CUDA not available"}
        return

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    start_mem = torch.cuda.memory_allocated(device)

    stats = {
        "start_allocated": start_mem,
        "start_reserved": torch.cuda.memory_reserved(device),
    }

    try:
        yield stats
    finally:
        torch.cuda.synchronize(device)
        stats["end_allocated"] = torch.cuda.memory_allocated(device)
        stats["end_reserved"] = torch.cuda.memory_reserved(device)
        stats["peak_allocated"] = torch.cuda.max_memory_allocated(device)
        stats["peak_reserved"] = torch.cuda.max_memory_reserved(device)
        stats["delta_allocated"] = stats["end_allocated"] - stats["start_allocated"]


def get_gpu_memory_info(device: str = "cuda:0") -> Dict[str, Any]:
    """Get current GPU memory information.

    Args:
        device: CUDA device to query

    Returns:
        Dictionary with memory statistics in MB
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    total = torch.cuda.get_device_properties(device).total_memory / 1024**2

    return {
        "allocated_mb": allocated,
        "reserved_mb": reserved,
        "total_mb": total,
        "free_mb": total - allocated,
    }


def get_cpu_memory_info() -> Dict[str, float]:
    """Get current CPU memory information.

    Returns:
        Dictionary with memory statistics in MB
    """
    process = psutil.Process()
    memory_info = process.memory_info()

    return {
        "rss_mb": memory_info.rss / 1024**2,  # Resident Set Size
        "vms_mb": memory_info.vms / 1024**2,  # Virtual Memory Size
        "percent": process.memory_percent(),
    }
