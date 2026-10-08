"""Reference-free video statistics, and how far an optimized video's statistics sit from the
unoptimized video's.

Why: agreement with the baseline video (PSNR/SSIM) saturates at a noise floor on diffusion
models. Generation is chaotic, so a 1e-4 perturbation of the initial latents already gives 14-20 dB
against the unperturbed video (DESIGN_DIFFUSION.md). Below that, PSNR cannot tell "a different
but equally good video" from "a degraded one". These statistics ask a different question: does
the video still look like the baseline's *kind* of video: as sharp, as noisy, as contrasty, as
colourful, as much motion, as steady, as smooth in time. Caching and step reduction fail in
characteristic ways (blur, flicker, noise, lost motion, jerky motion) that move these numbers
even when pixel agreement is already at the floor.

They are pixel statistics, not a quality model: a video can be bad and match the baseline's
statistics, and the statistics do not see content errors (a wrong object, broken anatomy). They
need no downloaded model. Use them paired (same prompt and seed as the baseline), and calibrate
against a numerical-perturbation control run through the same pipeline: a configuration only
shows a real shift if it deviates by more than the perturbation does.

Pure numpy; frames are (H, W, 3) uint8-range arrays.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

_EPS = 1e-8

# The statistics, in a stable order. Each is a positive number per video.
STATS = ("sharpness", "noise", "contrast", "colorfulness", "motion", "flicker", "jerk")


def _gray(frame: Any) -> Any:
    import numpy as np

    a = np.asarray(frame, dtype=np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def _laplacian(gray: Any) -> Any:
    return gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * gray[1:-1, 1:-1]


def _immerkaer_noise(gray: Any) -> float:
    """Immerkaer's fast noise-sigma estimate: mean absolute response to a Laplacian-difference mask that
    cancels image structure."""
    h, w = gray.shape
    response = (
        gray[:-2, :-2] - 2.0 * gray[:-2, 1:-1] + gray[:-2, 2:]
        - 2.0 * gray[1:-1, :-2] + 4.0 * gray[1:-1, 1:-1] - 2.0 * gray[1:-1, 2:]
        + gray[2:, :-2] - 2.0 * gray[2:, 1:-1] + gray[2:, 2:]
    )
    return math.sqrt(math.pi / 2.0) / (6.0 * (w - 2) * (h - 2)) * float(abs(response).sum())


def _colorfulness(frame: Any) -> float:
    """Hasler and Suesstrunk's colourfulness measure."""
    import numpy as np

    a = np.asarray(frame, dtype=np.float64)
    rg = a[..., 0] - a[..., 1]
    yb = 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    return float(math.hypot(rg.std(), yb.std()) + 0.3 * math.hypot(rg.mean(), yb.mean()))


def video_stats(frames: Sequence[Any]) -> dict[str, float]:
    """The seven reference-free statistics of one video (at least 3 frames)."""
    import numpy as np

    if len(frames) < 3:
        raise ValueError("video_stats needs at least 3 frames (jerk uses a second difference)")
    grays = [_gray(f) for f in frames]
    stack = np.stack(grays)
    first_diff = np.abs(np.diff(stack, axis=0)).mean()
    second_diff = np.abs(stack[2:] - 2.0 * stack[1:-1] + stack[:-2]).mean()
    luminance = stack.mean(axis=(1, 2))
    return {
        "sharpness": float(np.mean([_laplacian(g).var() for g in grays])),  # variance of the Laplacian: blur lowers it
        "noise": float(np.mean([_immerkaer_noise(g) for g in grays])),
        "contrast": float(np.mean([g.std() for g in grays])),
        "colorfulness": float(np.mean([_colorfulness(f) for f in frames])),
        "motion": float(first_diff),  # mean absolute frame-to-frame change ("dynamic degree")
        "flicker": float(np.abs(luminance[2:] - 2.0 * luminance[1:-1] + luminance[:-2]).mean()),  # brightness pumping
        "jerk": float(second_diff / (first_diff + _EPS)),  # temporal roughness relative to the motion present
    }


# Below these a statistic is imperceptible (gray levels on a 0-255 scale, or a dimensionless ratio), so
# a change from "nothing" to "nearly nothing" must not read as a huge ratio. Ratios are also clamped.
_FLOORS = {
    "sharpness": 1e-2, "noise": 1e-2, "contrast": 1e-2, "colorfulness": 1e-2,
    "motion": 1e-2, "flicker": 5e-2, "jerk": 1e-3,
}
MAX_RATIO = 16.0


def stat_ratios(optimized: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    """optimized / reference per statistic (1.0 = same as the baseline video), with a per-statistic floor
    on both so near-zero values don't give absurd ratios, and the result clamped to [1/16, 16]."""
    ratios = {}
    for k in STATS:
        ratio = max(optimized[k], _FLOORS[k]) / max(reference[k], _FLOORS[k])
        ratios[k] = min(max(ratio, 1.0 / MAX_RATIO), MAX_RATIO)
    return ratios


def style_deviation(ratios: dict[str, float]) -> float:
    """Mean absolute natural-log ratio over the statistics: 0 = same statistics as the baseline;
    log(2) = 0.69 would be a factor of two on average. One number per clip for ranking configurations;
    look at the individual ratios to see *what* shifted."""
    return sum(abs(math.log(ratios[k])) for k in STATS) / len(STATS)
