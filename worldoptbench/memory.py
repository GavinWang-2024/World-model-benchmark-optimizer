"""Will the model fit, and in what precision? (outline sections 3.4 and 7.10,
"CUDA memory management".)

A deliberately rough, parameter-count-based advisor. It estimates *weights* from
the parameter count and dtype and applies a flat overhead factor for everything
else (activations, caches, CUDA context, workspaces). That overhead varies a lot —
a diffusion video model's activations can dwarf its weights, while the 15.7M
parameter Dreamer world model needed 0.2 GB in total — so treat a "fits" verdict
as a first filter, not a guarantee, and a "doesn't fit" as a reason to try a
smaller dtype or quantization before an out-of-memory error finds out for you.

Pure functions plus one thin wrapper (`check_vram`) that asks torch for the device;
no GPU is needed to test the logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BYTES_PER_PARAM: dict[str, float] = {
    "float32": 4.0,
    "float16": 2.0,
    "bfloat16": 2.0,
    "float8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
}

# Fraction of device memory we are willing to plan to use; the rest covers the
# CUDA context, fragmentation, and anything else on the card.
USABLE_FRACTION = 0.90
# Flat multiplier on weight memory for activations, caches and workspaces.
DEFAULT_OVERHEAD = 1.5

# Preference order: highest fidelity that fits wins. Each entry is (dtype, quantization-or-None).
_LADDER: tuple[tuple[str, str | None], ...] = (
    ("float32", None),
    ("bfloat16", None),
    ("float16", None),
    ("int8", "int8_weight_only"),
    ("int4", "int4_weight_only"),
)


@dataclass
class Recommendation:
    fits: bool
    dtype: str | None  # None when nothing fits
    quantization: str | None  # a QuantizationModule scheme, when the dtype needs one
    needed_gb: float | None
    available_gb: float
    notes: list[str] = field(default_factory=list)


def estimate_weights_gb(param_count: int, dtype: str) -> float:
    if dtype not in BYTES_PER_PARAM:
        raise ValueError(f"unknown dtype {dtype!r}; choose from {sorted(BYTES_PER_PARAM)}")
    return param_count * BYTES_PER_PARAM[dtype] / 1024**3


def recommend(
    param_count: int,
    vram_gb: float,
    compute_capability: tuple[int, int] = (8, 0),
    overhead: float = DEFAULT_OVERHEAD,
) -> Recommendation:
    """Picks the highest-fidelity precision whose estimated footprint fits.

    Args:
        param_count: model parameters.
        vram_gb: device memory available to this process.
        compute_capability: GPU (major, minor). bfloat16 needs >= 8.0 (Ampere);
            float8 weights need >= 8.9, so they are never recommended below that.
        overhead: multiplier on weight memory for non-weight memory.
    """
    budget = vram_gb * USABLE_FRACTION
    notes: list[str] = []
    for dtype, quantization in _LADDER:
        if dtype == "bfloat16" and compute_capability < (8, 0):
            notes.append("bfloat16 skipped: needs compute capability 8.0+")
            continue
        needed = estimate_weights_gb(param_count, dtype) * overhead
        if needed <= budget:
            if quantization:
                notes.append(f"{dtype} needs quantization ({quantization}); expect some physics loss — re-measure PAES")
            return Recommendation(True, dtype, quantization, needed, vram_gb, notes)
        notes.append(f"{dtype}: ~{needed:.1f} GB needed > {budget:.1f} GB usable")
    smallest = estimate_weights_gb(param_count, "int4") * overhead
    notes.append("nothing fits on one device: consider tensor/pipeline parallelism or CPU offload")
    return Recommendation(False, None, None, smallest, vram_gb, notes)


def check_vram(info: Any, device: int = 0, overhead: float = DEFAULT_OVERHEAD) -> Recommendation:
    """`recommend` for a model (anything with `.param_count`, e.g. `ModelInfo`) on a
    real CUDA device. Needs torch; raises RuntimeError without a CUDA device.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("check_vram needs a CUDA device; none is available")
    free_bytes, _total = torch.cuda.mem_get_info(device)
    capability = torch.cuda.get_device_capability(device)
    return recommend(info.param_count, free_bytes / 1024**3, capability, overhead)
