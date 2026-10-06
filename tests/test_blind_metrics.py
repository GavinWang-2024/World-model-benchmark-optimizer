"""Reference-free video statistics (numpy only)."""

import math

import numpy as np
import pytest

from worldoptbench.metrics.blind import MAX_RATIO, STATS, stat_ratios, style_deviation, video_stats
from worldoptbench.metrics.visual import compute_visual_metrics


def _scene(frames=9, shift_per_frame=1, seed=0):
    """A smooth textured image that translates one pixel per frame (so there is real motion)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(128, 40, (48, 64, 3))
    for _ in range(4):  # smooth it so it is not already pure noise
        base = (base + np.roll(base, 1, 0) + np.roll(base, 1, 1) + np.roll(base, -1, 1)) / 4
    return [np.clip(np.roll(base, i * shift_per_frame, 1), 0, 255).astype(np.uint8) for i in range(frames)]


def _blur(frames):
    out = []
    for f in frames:
        a = f.astype(float)
        for _ in range(3):
            a = (a + np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1)) / 5
        out.append(a.astype(np.uint8))
    return out


def test_stats_are_finite_positive_and_complete():
    stats = video_stats(_scene())
    assert set(stats) == set(STATS)
    assert all(math.isfinite(v) and v >= 0 for v in stats.values())


def test_a_video_has_ratio_one_to_itself_and_zero_style_deviation():
    stats = video_stats(_scene())
    ratios = stat_ratios(stats, stats)
    assert all(r == pytest.approx(1.0) for r in ratios.values())
    assert style_deviation(ratios) == pytest.approx(0.0)


def test_blur_lowers_sharpness_and_noise():
    sharp = video_stats(_scene())
    soft = video_stats(_blur(_scene()))
    ratios = stat_ratios(soft, sharp)
    assert ratios["sharpness"] < 0.5
    assert ratios["noise"] < 1.0
    assert style_deviation(ratios) > 0.1


def test_added_noise_raises_the_noise_estimate():
    clean = _scene()
    rng = np.random.default_rng(3)
    noisy = [np.clip(f.astype(float) + rng.normal(0, 10, f.shape), 0, 255).astype(np.uint8) for f in clean]
    ratios = stat_ratios(video_stats(noisy), video_stats(clean))
    assert ratios["noise"] > 2.0 and ratios["sharpness"] > 2.0


def test_a_frozen_video_has_no_motion_and_the_ratio_says_so():
    moving = _scene()
    frozen = [moving[0]] * len(moving)
    stats = video_stats(frozen)
    assert stats["motion"] == 0.0 and stats["flicker"] == 0.0
    assert stat_ratios(stats, video_stats(moving))["motion"] < 0.1


def test_brightness_pumping_shows_up_as_flicker():
    steady = _scene()
    pumping = [np.clip(f.astype(float) * (1.0 + 0.2 * (-1) ** i), 0, 255).astype(np.uint8) for i, f in enumerate(steady)]
    assert video_stats(pumping)["flicker"] > 5 * video_stats(steady)["flicker"] + 1.0


def test_ratios_are_clamped_and_near_zero_values_do_not_explode():
    huge = {k: 1e9 for k in STATS}
    tiny = {k: 0.0 for k in STATS}
    assert all(r == MAX_RATIO for r in stat_ratios(huge, tiny).values())
    assert all(r == pytest.approx(1.0) for r in stat_ratios(tiny, tiny).values())  # 0 vs 0 is "the same", not 0/0
    assert style_deviation(stat_ratios(huge, tiny)) == pytest.approx(math.log(MAX_RATIO))


def test_too_few_frames_is_rejected():
    with pytest.raises(ValueError, match="3 frames"):
        video_stats(_scene(frames=2))


def test_visual_metrics_carry_the_blind_statistics_only_with_a_reference():
    frames = _scene()
    without = compute_visual_metrics(frames)
    assert without.blind is None and without.blind_ratio is None and without.style_deviation is None

    with_ref = compute_visual_metrics(_blur(frames), reference_frames=frames)
    assert set(with_ref.blind) == set(STATS) and set(with_ref.blind_ratio) == set(STATS)
    assert with_ref.blind_ratio["sharpness"] < 0.5
    assert with_ref.style_deviation > 0.1
