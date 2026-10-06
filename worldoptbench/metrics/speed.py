"""Speed metrics — latency, throughput, VRAM peak (outline §3.2, Axis 1).

VRAM tracking needs torch+CUDA; degrades to `None` on machines without it
(e.g. local dev on a laptop) rather than failing the whole run — the point
is this module still works before Phase 0's GPU access exists.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass
class SpeedMetrics:
    latency_seconds: float
    fps: float | None  # None if the result has no countable frames
    vram_peak_gb: float | None  # None if torch/CUDA isn't available
    # Memory the process holds on the GPU after the call, incl. persistent pools
    # (e.g. CUDA graphs) that `vram_peak_gb` can't see because they're allocated
    # at capture time, outside the timed call. Compare this one across
    # optimizations; `vram_peak_gb` is per-call transient allocation only.
    vram_reserved_gb: float | None = None
    speedup: float = 1.0  # baseline_latency / this latency; filled in by the caller
    # Seconds until the first frame is available. For models that return only when the whole rollout is done
    # (every wrapper here) this equals the latency; a streaming model can report a smaller number by setting
    # `time_to_first_frame_seconds` in its Rollout metadata.
    time_to_first_frame_seconds: float | None = None


def _peak_vram_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / (1024**3)
    except ImportError:
        return None


def _reserved_vram_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.memory_reserved() / (1024**3)
    except ImportError:
        return None


def _reset_vram_tracking() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def measure_speed(fn: Callable[[], T]) -> tuple[T, SpeedMetrics]:
    """Times a call to `fn` (typically `model.generate(...)`) and captures
    peak VRAM alongside it.

    Returns (fn's return value, SpeedMetrics) so the caller doesn't have to
    call fn twice — the first thing runner.py does with the returned value
    is read `.frames` off it for the visual/physics metrics.
    """
    _reset_vram_tracking()

    start = time.perf_counter()
    result = fn()
    latency = time.perf_counter() - start

    num_frames = len(getattr(result, "frames", None) or [])
    fps = (num_frames / latency) if (num_frames and latency > 0) else None

    reported = (getattr(result, "metadata", None) or {}).get("time_to_first_frame_seconds")
    return result, SpeedMetrics(
        latency_seconds=latency,
        fps=fps,
        vram_peak_gb=_peak_vram_gb(),
        vram_reserved_gb=_reserved_vram_gb(),
        time_to_first_frame_seconds=reported if reported is not None else latency,
    )
