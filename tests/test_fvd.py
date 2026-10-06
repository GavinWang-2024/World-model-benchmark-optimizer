"""Frechet distance between video sets (numpy/scipy) and the descriptor the scorer exposes for it."""

import numpy as np
import pytest

from worldoptbench.metrics.fvd import fit_pca, frechet_distance, project, reduced_frechet_distance
from worldoptbench.metrics.perceptual import PerceptualScorer


def _features(n=60, d=6, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d))


def test_a_set_is_at_distance_zero_from_itself():
    features = _features()
    assert frechet_distance(features, features) == pytest.approx(0.0, abs=1e-8)


def test_a_pure_mean_shift_gives_exactly_the_squared_shift():
    # identical covariances cancel, leaving ||mu_a - mu_b||^2
    features = _features()
    shift = np.array([1.0, -2.0, 0.5, 0.0, 0.0, 3.0])
    assert frechet_distance(features, features + shift) == pytest.approx(float(shift @ shift), rel=1e-6)


def test_doubling_the_scale_gives_the_trace_of_the_covariance():
    # S_b = 4 S_a  =>  tr(S_a + 4 S_a - 2 * 2 S_a) = tr(S_a)
    features = _features()
    centred = features - features.mean(axis=0)
    expected = float(np.trace(np.cov(centred, rowvar=False)))
    assert frechet_distance(centred, 2.0 * centred) == pytest.approx(expected, rel=1e-6)


def test_the_distance_is_symmetric_and_not_negative():
    a, b = _features(seed=1), _features(seed=2) * 1.5 + 0.3
    assert frechet_distance(a, b) == pytest.approx(frechet_distance(b, a), rel=1e-6)
    assert frechet_distance(a, b) >= 0.0


def test_shrinkage_makes_a_rank_deficient_estimate_usable():
    # 5 videos in 20 dimensions: the covariances are rank <= 4, the plain distance is still computable
    # but shrinkage keeps the square root well conditioned; both must return finite numbers
    a, b = _features(n=5, d=20, seed=3), _features(n=5, d=20, seed=4)
    assert np.isfinite(frechet_distance(a, b, shrink=0.0))
    assert np.isfinite(frechet_distance(a, b, shrink=0.2))


def test_arguments_are_validated():
    good = _features()
    with pytest.raises(ValueError, match="same d"):
        frechet_distance(good, _features(d=7))
    with pytest.raises(ValueError, match="at least 2"):
        frechet_distance(good[:1], good)
    with pytest.raises(ValueError, match="shrink"):
        frechet_distance(good, good, shrink=1.0)
    with pytest.raises(ValueError, match="components"):
        fit_pca(good, 0)
    with pytest.raises(ValueError, match="components"):
        fit_pca(good, 100)


def test_pca_projection_has_the_requested_dimension_and_centres_the_reference():
    reference = _features(n=40, d=10)
    pca = fit_pca(reference, 3)
    projected = project(reference, pca)
    assert projected.shape == (40, 3)
    assert np.allclose(projected.mean(axis=0), 0.0, atol=1e-9)


def test_reduced_distance_is_zero_for_identical_sets_and_grows_with_a_shift():
    reference = _features(n=40, d=10, seed=5)
    assert reduced_frechet_distance(reference, reference) == pytest.approx(0.0, abs=1e-8)
    near = reduced_frechet_distance(reference, reference + 0.2, components=4)
    far = reduced_frechet_distance(reference, reference + 2.0, components=4)
    assert 0.0 < near < far


# ---- the descriptor ---------------------------------------------------------------------------------------


class FakeScorer(PerceptualScorer):
    def dino_embed(self, frames):
        return np.array([[float(np.mean(f)), float(np.std(f)) + 1.0, 1.0] for f in frames])


def _video(level, frames=12):
    return [np.full((6, 8, 3), level + i, dtype=np.uint8) for i in range(frames)]


def test_descriptor_is_the_mean_and_std_of_unit_normalized_frame_embeddings():
    embeddings = np.array([[3.0, 4.0], [0.0, 2.0]])
    d = PerceptualScorer.descriptor(embeddings)
    assert len(d) == 4
    units = np.array([[0.6, 0.8], [0.0, 1.0]])
    assert np.allclose(d[:2], units.mean(axis=0)) and np.allclose(d[2:], units.std(axis=0))


def test_scores_carry_descriptors_only_when_asked_for():
    plain = FakeScorer().score(_video(50), _video(80))
    assert "descriptor" not in plain and "descriptor_ref" not in plain
    kept = FakeScorer(keep_descriptor=True).score(_video(50), _video(80))
    assert len(kept["descriptor"]) == len(kept["descriptor_ref"]) == 6  # 3-dim embeddings: mean and std
    assert kept["descriptor"] != kept["descriptor_ref"]
    assert "descriptor_ref" not in FakeScorer(keep_descriptor=True).score(_video(50))  # no reference, no reference descriptor
