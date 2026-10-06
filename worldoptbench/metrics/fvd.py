"""Fréchet distance between two sets of videos: the distributional metric behind FVD (outline section 3.2).

Standard FVD embeds each video with an I3D network trained on Kinetics and compares the two sets' feature
distributions with the Fréchet distance. This module has the distance itself and uses a different, honest
embedding: a *video descriptor* from DINOv2 (`metrics/perceptual.py`): the mean and the standard deviation,
over evenly spaced frames, of the frame embeddings (content and how much it varies in time). That makes
the number a DINO-space Fréchet distance, NOT comparable with published FVD values, and it should be called
by that name.

Two small-sample facts matter more than the choice of network:
* The distance needs a covariance estimate. With n videos and d feature dimensions the covariance has rank
  at most n - 1, so for n = 32 and d = 768 it is badly rank deficient, and the plug-in distance is biased
  upward by an amount that depends on n. Use `reduce` (PCA fitted on the reference set to a few components)
  and `shrink` (blend towards a scaled identity) so the estimate is stable, and ALWAYS compare against a
  control computed at the same n: the distance between the reference set and a numerically-perturbed copy
  of itself is the floor, and only the excess over that floor is a real distribution shift.
* The distance between a set and itself is exactly 0; between independent samples of the same distribution it is
  positive from sampling noise alone. Nothing here has a "good" absolute value.

Pure numpy/scipy; the embedding step needs the models (`PerceptualScorer(keep_descriptor=True)`).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _covariance(features: np.ndarray, shrink: float) -> np.ndarray:
    centred = features - features.mean(axis=0, keepdims=True)
    cov = centred.T @ centred / max(len(features) - 1, 1)
    if shrink > 0:
        d = cov.shape[0]
        cov = (1.0 - shrink) * cov + shrink * (np.trace(cov) / d) * np.eye(d)
    return cov


def frechet_distance(a: Any, b: Any, shrink: float = 0.0) -> float:
    """||mu_a - mu_b||^2 + tr(S_a + S_b - 2 (S_a S_b)^(1/2)) for two (n, d) feature sets.

    `shrink` in [0, 1) blends each covariance towards (trace / d) * I, which keeps the matrix square root
    well defined when n < d. Raises if the two sets do not share a dimension or either has fewer than 2 rows.
    """
    from scipy import linalg  # noqa: PLC0415

    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("expected two (n, d) arrays with the same d")
    if len(a) < 2 or len(b) < 2:
        raise ValueError("need at least 2 videos in each set to estimate a covariance")
    if not 0.0 <= shrink < 1.0:
        raise ValueError("shrink must be in [0, 1)")
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    cov_a, cov_b = _covariance(a, shrink), _covariance(b, shrink)
    root = linalg.sqrtm(cov_a @ cov_b)
    if np.iscomplexobj(root):
        root = root.real  # tiny imaginary parts from numerical error
    value = float((mu_a - mu_b) @ (mu_a - mu_b) + np.trace(cov_a) + np.trace(cov_b) - 2.0 * np.trace(root))
    return max(value, 0.0)  # the formula can dip a hair below 0 numerically for identical sets


def fit_pca(reference: Any, components: int) -> tuple[np.ndarray, np.ndarray]:
    """(mean, basis) of the top `components` principal directions of the reference features; apply with
    `project`. Fitted on the reference (baseline) set only, so the optimized set cannot shape the space."""
    reference = np.asarray(reference, dtype=np.float64)
    if components < 1 or components > min(reference.shape):
        raise ValueError(f"components must be in [1, {min(reference.shape)}]")
    mean = reference.mean(axis=0)
    _, _, vt = np.linalg.svd(reference - mean, full_matrices=False)
    return mean, vt[:components]


def project(features: Any, pca: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mean, basis = pca
    return (np.asarray(features, dtype=np.float64) - mean) @ basis.T


def reduced_frechet_distance(reference: Any, other: Any, components: int = 8, shrink: float = 0.1) -> float:
    """Fréchet distance in the PCA space of `reference` (fitted on it alone), with covariance shrinkage."""
    pca = fit_pca(reference, components)
    return frechet_distance(project(reference, pca), project(other, pca), shrink=shrink)
