"""Latent-forecast metrics, for models that predict representations instead of pixels (V-JEPA 2).

There is no decoder, so PSNR / SSIM / physics scores do not apply. What can be measured is how close the predicted
latents are to the encoder's latents of the frames that actually followed. Two questions, kept separate:

  * `forecast_scores`   - fidelity to the truth (the encoder's latents of the real future), with a *persistence*
                          baseline (repeat the last context step's latents) and a skill score against it. This is the
                          analogue of the Dreamer simulator skill score: 0 = no better than "nothing changes",
                          1 = perfect. It says how good the forecast is.
  * `forecast_agreement`- fidelity of one forecast to another (an optimized run against the unoptimized run, same
                          clip). This is the analogue of the "agreement with its own unoptimized output" used for the
                          text-to-video models: it says what the optimization changed, not whether the forecast is right.

Both are about latent geometry, not about pixels or physics, and say nothing about either. A step is a
(tokens, hidden) array; every function takes lists of steps.

Cosine is per token, averaged over tokens. Relative L2 is ||pred - target|| / ||target|| over the whole step, so it is
scale-free but, unlike cosine, sees changes in magnitude.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

_EPS = 1e-12


def split_steps(tokens: Any, tokens_per_step: int) -> list[np.ndarray]:
    """(n_steps * tokens_per_step, hidden) -> a list of (tokens_per_step, hidden) arrays, time-major."""
    arr = np.asarray(tokens, dtype=np.float64)
    if arr.ndim != 2 or tokens_per_step <= 0 or arr.shape[0] % tokens_per_step:
        raise ValueError(
            f"cannot split {arr.shape} into steps of {tokens_per_step} tokens (need 2-D, evenly divisible)"
        )
    return [arr[i : i + tokens_per_step] for i in range(0, arr.shape[0], tokens_per_step)]


def _steps(x: Sequence[Any], name: str) -> list[np.ndarray]:
    steps = [np.asarray(s, dtype=np.float64) for s in x]
    if not steps:
        raise ValueError(f"{name} is empty")
    if any(s.ndim != 2 for s in steps):
        raise ValueError(f"{name}: every step must be a (tokens, hidden) array")
    return steps


def _check_pair(a: list[np.ndarray], b: list[np.ndarray], names: str) -> None:
    if len(a) != len(b) or any(x.shape != y.shape for x, y in zip(a, b)):
        raise ValueError(f"{names} must have the same number of steps and the same (tokens, hidden) shapes")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Mean over tokens of the cosine similarity between matching rows."""
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return float(np.mean(num / np.maximum(den, _EPS)))


def relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    """||pred - target|| / ||target||, over the whole step."""
    return float(np.linalg.norm(pred - target) / max(np.linalg.norm(target), _EPS))


def _slope(values: Sequence[float]) -> float | None:
    """Least-squares slope of a per-step curve (change per step); None when there is only one step."""
    if len(values) < 2:
        return None
    x = np.arange(len(values), dtype=np.float64)
    return float(np.polyfit(x, np.asarray(values, dtype=np.float64), 1)[0])


def forecast_scores(
    pred: Sequence[Any], target: Sequence[Any], persistence: Any, steps_per_second: float | None = None
) -> dict[str, Any]:
    """Score a forecast against the encoder's latents of the real future.

    Args:
        pred: predicted latents, one (tokens, hidden) array per future step.
        target: the encoder's latents of the real future, same shapes.
        persistence: the last *context* step's latents (tokens, hidden), the "nothing changes" baseline. Take it
            from the unoptimized encoder, so every configuration is judged against the same reference.
        steps_per_second: when given, the drift slope is also reported per second of forecast.

    Returns:
        cosine / relative-L2 per step for the forecast and for persistence; their means; `skill` = 1 - mean forecast
        relative-L2 / mean persistence relative-L2 (None when persistence is exact); and `drift_per_step`, the
        least-squares slope of the forecast's cosine over the horizon (negative = it loses fidelity the further out
        it looks; None for a one-step forecast).
    """
    p, t = _steps(pred, "pred"), _steps(target, "target")
    _check_pair(p, t, "pred and target")
    base = np.asarray(persistence, dtype=np.float64)
    if base.shape != t[0].shape:
        raise ValueError(f"persistence has shape {base.shape}, steps have {t[0].shape}")

    cos_pred = [cosine(a, b) for a, b in zip(p, t)]
    err_pred = [relative_l2(a, b) for a, b in zip(p, t)]
    cos_base = [cosine(base, b) for b in t]
    err_base = [relative_l2(base, b) for b in t]
    mean_err_base = float(np.mean(err_base))
    drift = _slope(cos_pred)
    out: dict[str, Any] = {
        "steps": len(p),
        "cosine_per_step": cos_pred,
        "relative_l2_per_step": err_pred,
        "persistence_cosine_per_step": cos_base,
        "persistence_relative_l2_per_step": err_base,
        "cosine": float(np.mean(cos_pred)),
        "relative_l2": float(np.mean(err_pred)),
        "persistence_cosine": float(np.mean(cos_base)),
        "persistence_relative_l2": mean_err_base,
        "skill": None if mean_err_base <= _EPS else 1.0 - float(np.mean(err_pred)) / mean_err_base,
        "drift_per_step": drift,
    }
    if steps_per_second is not None:
        out["drift_per_second"] = None if drift is None else drift * steps_per_second
    return out


def forecast_agreement(a: Sequence[Any], b: Sequence[Any]) -> dict[str, Any]:
    """How closely forecast `a` (an optimized run) matches forecast `b` (the unoptimized run, same clip).

    Identical forecasts give cosine 1 and relative L2 0. This measures what the optimization changed; it does not
    say whether either forecast is correct (that is `forecast_scores`).
    """
    x, y = _steps(a, "a"), _steps(b, "b")
    _check_pair(x, y, "a and b")
    cos = [cosine(i, j) for i, j in zip(x, y)]
    err = [relative_l2(i, j) for i, j in zip(x, y)]
    return {
        "steps": len(x),
        "cosine": float(np.mean(cos)),
        "min_cosine": float(np.min(cos)),
        "relative_l2": float(np.mean(err)),
        "max_relative_l2": float(np.max(err)),
    }
