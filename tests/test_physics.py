import numpy as np
import pytest

from worldoptbench.metrics.physics import (
    compute_drift_curve,
    compute_pai_bench_score,
    compute_physics_score,
    compute_sim_fidelity,
    compute_worldroambench_score,
)


def test_pai_bench_not_wired_up_yet():
    with pytest.raises(NotImplementedError):
        compute_pai_bench_score(frames=[])


def test_worldroambench_not_wired_up_yet():
    with pytest.raises(NotImplementedError):
        compute_worldroambench_score(frames=[])


def test_compute_physics_score_not_wired_up_yet():
    # Confirms the runner's try/except NotImplementedError path still works
    # end to end even though the underlying repos aren't wired up.
    with pytest.raises(NotImplementedError):
        compute_physics_score(frames=[])


def test_drift_curve_flat_score_has_zero_drift():
    curve = compute_drift_curve({4: 0.9, 16: 0.9, 60: 0.9, 120: 0.9})
    assert curve.drift_rate == pytest.approx(0.0, abs=1e-9)
    assert curve.drift_onset is None  # never crosses default threshold of 0.75


def test_drift_curve_detects_onset():
    curve = compute_drift_curve({4: 0.95, 16: 0.90, 60: 0.71, 120: 0.55}, threshold=0.75)
    assert curve.drift_onset == 60
    assert curve.drift_rate < 0  # score decreasing over time


def test_drift_curve_needs_at_least_two_horizons():
    with pytest.raises(ValueError):
        compute_drift_curve({4: 0.9})


def test_sim_fidelity_perfect_match_scores_one():
    frames = [np.full((4, 4, 3), 100, dtype=np.uint8)] * 3
    result = compute_sim_fidelity(frames, frames)
    assert result.score == pytest.approx(1.0)
    assert result.error_curve == [0.0, 0.0, 0.0]


def test_sim_fidelity_error_curve_tracks_per_step_drift():
    reference = [np.zeros((4, 4, 3), dtype=np.uint8)] * 3
    generated = [np.full((4, 4, 3), v, dtype=np.uint8) for v in (0, 51, 255)]
    result = compute_sim_fidelity(generated, reference)
    assert result.error_curve == pytest.approx([0.0, 0.2, 1.0])
    assert result.score == pytest.approx(1.0 - 1.2 / 3)


def test_sim_fidelity_score_is_clamped_at_zero():
    result = compute_sim_fidelity(
        [np.full((2, 2, 3), 255, dtype=np.uint8)], [np.zeros((2, 2, 3), dtype=np.uint8)]
    )
    assert result.score == 0.0


def test_sim_fidelity_rejects_mismatched_or_empty_input():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        compute_sim_fidelity([frame], [frame, frame])
    with pytest.raises(ValueError):
        compute_sim_fidelity([], [])


def test_physics_score_with_reference_uses_sim_fidelity_and_skips_stubs():
    # Unlike the no-reference case above, this must NOT raise NotImplementedError.
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    score = compute_physics_score(frames=[frame], reference_frames=[frame])
    assert score.sim_fidelity_score == pytest.approx(1.0)
    assert score.sim_error_curve == [0.0]
    assert score.pai_bench_score is None


def _frames(*values):
    return [np.full((4, 4, 3), v, dtype=np.uint8) for v in values]


def test_skill_score_is_zero_for_a_model_no_better_than_freezing_the_scene():
    reference = _frames(0, 100, 200)
    baseline = np.zeros((4, 4, 3), dtype=np.uint8)  # "nothing moves" = the last context frame
    frozen = [baseline] * 3  # a model that just repeats it
    result = compute_sim_fidelity(frozen, reference, baseline_frame=baseline)
    assert result.score == pytest.approx(0.0)
    assert result.persistence_error_curve == pytest.approx(result.error_curve)


def test_skill_score_is_one_for_a_perfect_model_and_half_for_half_the_baseline_error():
    reference = _frames(100, 100)
    baseline = np.zeros((4, 4, 3), dtype=np.uint8)
    assert compute_sim_fidelity(reference, reference, baseline).score == pytest.approx(1.0)

    half = compute_sim_fidelity(_frames(50, 50), reference, baseline)
    assert half.score == pytest.approx(0.5)


def test_skill_score_clamps_models_worse_than_the_baseline_to_zero():
    reference = _frames(10, 10)
    baseline = np.zeros((4, 4, 3), dtype=np.uint8)  # off by 10; the model below is off by 200
    assert compute_sim_fidelity(_frames(210, 210), reference, baseline).score == 0.0


def test_skill_score_separates_what_pixel_score_cannot():
    # The failure that motivated the skill score: with a mostly-static scene a
    # useless prediction still gets a pixel score near 1, but ~0 skill.
    reference = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
    for r in reference:
        r[:1, :1] = 255  # one moving pixel out of 64
    baseline = np.zeros((8, 8, 3), dtype=np.uint8)
    useless = [baseline] * 3

    result = compute_sim_fidelity(useless, reference, baseline_frame=baseline)
    assert result.pixel_score > 0.98
    assert result.score == pytest.approx(0.0)


def test_skill_score_falls_back_to_pixel_score_when_baseline_is_already_perfect():
    reference = _frames(77, 77)
    baseline = np.full((4, 4, 3), 77, dtype=np.uint8)  # static scene: nothing to beat
    result = compute_sim_fidelity(_frames(70, 70), reference, baseline)
    assert result.score == pytest.approx(result.pixel_score)


def test_physics_score_threads_baseline_frame_through():
    reference = _frames(100, 100)
    baseline = np.zeros((4, 4, 3), dtype=np.uint8)
    score = compute_physics_score(
        frames=_frames(50, 50), reference_frames=reference, baseline_frame=baseline
    )
    assert score.sim_fidelity_score == pytest.approx(0.5)
    assert score.sim_pixel_score > score.sim_fidelity_score
    assert score.sim_persistence_error_curve is not None
