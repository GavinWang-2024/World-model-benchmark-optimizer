import pytest

from worldoptbench.metrics.paes import compute_paes


def test_no_drift_no_optimization_gives_physics_score():
    # speedup=1, drift_rate=0 -> PAES should just equal physics_score
    assert compute_paes(speedup=1.0, physics_score=0.9, drift_rate=0.0, horizon_t=60) == 0.9


def test_speedup_increases_paes_all_else_equal():
    low = compute_paes(speedup=1.0, physics_score=0.9, drift_rate=0.0, horizon_t=60)
    high = compute_paes(speedup=2.0, physics_score=0.9, drift_rate=0.0, horizon_t=60)
    assert high > low


def test_decaying_physics_score_is_penalized():
    # drift_rate is score change per second: negative = the score falls as the rollout lengthens.
    no_drift = compute_paes(speedup=2.0, physics_score=0.9, drift_rate=0.0, horizon_t=60)
    decaying = compute_paes(speedup=2.0, physics_score=0.9, drift_rate=-0.01, horizon_t=60)
    assert decaying < no_drift
    assert decaying == pytest.approx(2.0 * 0.9 / (1 + 0.01 * 60))


def test_default_does_not_penalize_a_rising_score():
    steady = compute_paes(speedup=1.0, physics_score=0.9, drift_rate=0.0, horizon_t=60)
    rising = compute_paes(speedup=1.0, physics_score=0.9, drift_rate=+0.01, horizon_t=60)
    assert rising == steady


def test_abs_mode_keeps_the_original_sign_blind_behavior():
    positive = compute_paes(1.0, 0.9, drift_rate=0.01, horizon_t=60, drift_mode="abs")
    negative = compute_paes(1.0, 0.9, drift_rate=-0.01, horizon_t=60, drift_mode="abs")
    assert positive == negative
    assert positive < compute_paes(1.0, 0.9, drift_rate=0.0, horizon_t=60, drift_mode="abs")


def test_decay_and_abs_agree_when_the_score_decays():
    kwargs = {"speedup": 1.5, "physics_score": 0.8, "drift_rate": -0.004, "horizon_t": 30}
    assert compute_paes(**kwargs, drift_mode="decay") == compute_paes(**kwargs, drift_mode="abs")


def test_unknown_drift_mode_is_rejected():
    with pytest.raises(ValueError, match="drift_mode"):
        compute_paes(1.0, 0.9, 0.0, 60, drift_mode="signed")
