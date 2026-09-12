"""Visual quality metrics — FVD, PSNR, SSIM, temporal consistency
(outline §3.2, Axis 2).

Important nuance the outline glosses over: PSNR/SSIM/FVD all require
something to compare against. That's natural for Video2World rollouts where
you hold out the true continuation as ground truth, but there's no such
reference for pure Text2World generation. So here: temporal consistency is
always computed (no reference needed), while PSNR/SSIM/FVD are left `None`
unless `reference_frames` is supplied.

FVD specifically is normally computed over a *distribution* of generated vs.
real videos via a feature extractor (e.g. I3D) — not a single pair. Treat
`fvd` here as a TODO until Phase 2 has enough samples to compare
distributions properly rather than a single rollout.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class VisualMetrics:
    temporal_consistency: float  # 1.0 = identical consecutive frames, 0.0 = maximally different
    psnr: float | None = None
    ssim: float | None = None
    fvd: float | None = None  # TODO: needs a real reference distribution, see module docstring


def _temporal_consistency(frames: Sequence[Any]) -> float:
    """1 - normalized mean frame-to-frame delta. Pure numpy — no torch needed,
    so this half of the module works even before the `ml` extra is installed.
    """
    import numpy as np

    arrs = [np.asarray(f, dtype=np.float64) for f in frames]
    if len(arrs) < 2:
        return 1.0

    deltas = [float(np.abs(arrs[i + 1] - arrs[i]).mean()) for i in range(len(arrs) - 1)]
    mean_delta = sum(deltas) / len(deltas)
    max_possible = 255.0  # assumes uint8-range pixel values
    return max(0.0, 1.0 - mean_delta / max_possible)


def compute_visual_metrics(
    frames: Sequence[Any],
    reference_frames: Sequence[Any] | None = None,
) -> VisualMetrics:
    temporal_consistency = _temporal_consistency(frames)

    psnr = ssim = fvd = None
    if reference_frames is not None:
        # Lazy import: torch/torchmetrics only needed when a reference is
        # actually supplied, so importing this module never requires the
        # `ml` extra.
        import torch
        from torchmetrics.functional import (
            peak_signal_noise_ratio,
            structural_similarity_index_measure,
        )

        gen = torch.stack([torch.as_tensor(f, dtype=torch.float32) for f in frames])
        ref = torch.stack([torch.as_tensor(f, dtype=torch.float32) for f in reference_frames])
        # torchmetrics expects (N, C, H, W); frames are typically (H, W, C).
        if gen.ndim == 4 and gen.shape[-1] in (1, 3):
            gen = gen.permute(0, 3, 1, 2)
            ref = ref.permute(0, 3, 1, 2)

        psnr = float(peak_signal_noise_ratio(gen, ref, data_range=255.0))
        ssim = float(structural_similarity_index_measure(gen, ref, data_range=255.0))
        # fvd intentionally left None — see module docstring.

    return VisualMetrics(temporal_consistency=temporal_consistency, psnr=psnr, ssim=ssim, fvd=fvd)
