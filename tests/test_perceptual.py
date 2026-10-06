"""Perceptual scoring: the arithmetic and the wiring, with fake embedders (no DINO/CLIP download).
The real models are exercised by the contact-sheet / sweep scripts, not here.
"""

import numpy as np
import pytest

from worldoptbench.metrics.perceptual import (
    PerceptualScorer,
    consecutive_consistency,
    cosine_rows,
    sample_indices,
)
from worldoptbench.metrics.visual import compute_visual_metrics
from worldoptbench.runner import load_standard_set, run_benchmark


class FakeScorer(PerceptualScorer):
    """Embeddings computed from pixel statistics, so similar frames embed similarly and nothing is downloaded."""

    def dino_embed(self, frames):
        return np.array([[float(np.mean(f)), float(np.std(f)) + 1.0, 1.0] for f in frames])

    def clip_image_embed(self, frames):
        return np.array([[float(np.mean(f)), 1.0] for f in frames])

    def clip_text_embed(self, text):
        return np.array([[float(len(text)), 1.0]])


def _video(level=100, frames=12, flicker=0.0, size=(6, 8, 3)):
    return [np.full(size, level * (1.0 + flicker * (-1) ** i), dtype=np.float64).clip(0, 255).astype(np.uint8) for i in range(frames)]


# ---- helpers ----------------------------------------------------------------------------------------


def test_sample_indices_are_evenly_spaced_include_the_ends_and_never_exceed_the_count():
    assert sample_indices(33, 8) == [0, 5, 9, 14, 18, 23, 27, 32]
    assert sample_indices(3, 8) == [0, 1, 2]
    assert sample_indices(1, 8) == [0]
    with pytest.raises(ValueError, match="at least one frame"):
        sample_indices(0, 8)


def test_cosine_rows_and_consecutive_consistency():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(cosine_rows(a, a), [1.0, 1.0])
    assert np.allclose(cosine_rows(a, a[::-1]), [0.0, 0.0], atol=1e-9)
    assert consecutive_consistency(np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])) == pytest.approx(1.0)
    assert consecutive_consistency(np.array([[1.0, 0.0], [0.0, 1.0]])) == pytest.approx(0.0, abs=1e-9)
    assert consecutive_consistency(np.array([[1.0, 2.0]])) == 1.0  # one frame: nothing to compare


# ---- scores -------------------------------------------------------------------------------------------


def test_a_video_is_perfectly_similar_to_itself():
    video = _video()
    scores = FakeScorer().score(video, video)
    assert scores["dino_similarity"] == pytest.approx(1.0)
    assert scores["dino_consistency"] == pytest.approx(scores["dino_consistency_ref"])
    assert "clip_score" not in scores  # no prompt, no CLIP


def test_a_different_video_scores_lower_similarity():
    reference = _video(level=60)
    assert FakeScorer().score(_video(level=60), reference)["dino_similarity"] > FakeScorer().score(_video(level=200), reference)["dino_similarity"]


def test_flicker_lowers_consistency_but_not_the_reference_consistency():
    steady = _video(level=100)
    flickering = _video(level=100, flicker=0.6)
    scores = FakeScorer().score(flickering, steady)
    assert scores["dino_consistency"] < scores["dino_consistency_ref"] == pytest.approx(1.0)


def test_no_reference_gives_only_the_reference_free_score():
    scores = FakeScorer().score(_video())
    assert set(scores) == {"dino_consistency"}


def test_clip_scores_need_a_prompt_and_the_delta_needs_both_videos():
    video, reference = _video(level=150), _video(level=40)
    with_prompt = FakeScorer().score(video, reference, "a red ball")
    assert with_prompt["clip_delta"] == pytest.approx(with_prompt["clip_score"] - with_prompt["clip_score_ref"])
    no_reference = FakeScorer().score(video, None, "a red ball")
    assert "clip_score" in no_reference and "clip_score_ref" not in no_reference and "clip_delta" not in no_reference


def test_a_reference_with_a_different_number_of_sampled_frames_skips_similarity_only():
    scores = FakeScorer(frames_sampled=8).score(_video(frames=33), _video(frames=3))
    assert "dino_similarity" not in scores and "dino_consistency_ref" in scores


def test_frames_sampled_is_validated():
    with pytest.raises(ValueError, match="frames_sampled"):
        PerceptualScorer(frames_sampled=1)


# ---- wiring -----------------------------------------------------------------------------------------------


def test_visual_metrics_carry_perceptual_scores_only_when_a_scorer_is_given():
    video, reference = _video(), _video(level=120)
    assert compute_visual_metrics(video, reference_frames=reference).perceptual is None
    metrics = compute_visual_metrics(video, reference_frames=reference, prompt="a ball", scorer=FakeScorer())
    assert {"dino_similarity", "dino_consistency", "clip_score", "clip_delta"} <= set(metrics.perceptual)


def test_run_benchmark_adds_perceptual_scores_after_timing_and_passes_the_prompt():
    from test_runner import FakeWorldModel

    seen = []

    class Recording(FakeScorer):
        def score(self, frames, reference_frames=None, prompt=None):
            seen.append(prompt)
            return super().score(frames, reference_frames, prompt)

    results = run_benchmark(FakeWorldModel(), perceptual_scorer=Recording())
    assert results and all(r.visual["perceptual"] is not None for r in results)
    expected = {p["prompt"] for p in load_standard_set()["prompts"]}
    assert set(seen) == expected  # each entry's own prompt reached the scorer


def test_run_benchmark_without_a_scorer_has_no_perceptual_field_value():
    from test_runner import FakeWorldModel

    assert all(r.visual["perceptual"] is None for r in run_benchmark(FakeWorldModel()))
