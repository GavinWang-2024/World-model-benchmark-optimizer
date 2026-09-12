from worldoptbench.metrics.paes import compute_paes


def test_no_drift_no_optimization_gives_physics_score():
    # speedup=1, drift_rate=0 -> PAES should just equal physics_score
    assert compute_paes(speedup=1.0, physics_score=0.9, drift_rate=0.0, horizon_t=60) == 0.9


def test_speedup_increases_paes_all_else_equal():
    low = compute_paes(speedup=1.0, physics_score=0.9, drift_rate=0.0, horizon_t=60)
    high = compute_paes(speedup=2.0, physics_score=0.9, drift_rate=0.0, horizon_t=60)
    assert high > low


def test_drift_penalizes_score():
    no_drift = compute_paes(speedup=2.0, physics_score=0.9, drift_rate=0.0, horizon_t=60)
    with_drift = compute_paes(speedup=2.0, physics_score=0.9, drift_rate=0.01, horizon_t=60)
    assert with_drift < no_drift


def test_drift_rate_sign_does_not_matter():
    positive = compute_paes(speedup=1.0, physics_score=0.9, drift_rate=0.01, horizon_t=60)
    negative = compute_paes(speedup=1.0, physics_score=0.9, drift_rate=-0.01, horizon_t=60)
    assert positive == negative
