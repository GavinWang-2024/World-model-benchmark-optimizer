import pytest

from worldoptbench.metrics.physics import (
    compute_drift_curve,
    compute_pai_bench_score,
    compute_physics_score,
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
