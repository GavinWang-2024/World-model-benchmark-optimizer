"""Latent-forecast metrics: exact values on small constructed cases (numpy only)."""

import math

import numpy as np
import pytest

from worldoptbench.metrics.latent import (
    cosine,
    forecast_agreement,
    forecast_scores,
    relative_l2,
    split_steps,
)

RNG = np.random.default_rng(0)


def steps(n=4, tokens=3, hidden=5, seed=0):
    return [np.random.default_rng(seed + i).normal(size=(tokens, hidden)) for i in range(n)]


# ---- primitives ----------------------------------------------------------------------------------------


def test_cosine_is_one_for_parallel_zero_for_orthogonal_and_minus_one_for_opposite():
    a = np.array([[1.0, 0.0], [0.0, 2.0]])
    assert cosine(a, 3 * a) == pytest.approx(1.0)
    assert cosine(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])) == pytest.approx(0.0)
    assert cosine(a, -a) == pytest.approx(-1.0)


def test_cosine_ignores_scale_but_relative_l2_does_not():
    a = RNG.normal(size=(4, 8))
    assert cosine(a, 2 * a) == pytest.approx(1.0)
    assert relative_l2(2 * a, a) == pytest.approx(1.0)  # ||2a - a|| / ||a|| = 1
    assert relative_l2(a, a) == 0.0


def test_a_zero_vector_does_not_divide_by_zero():
    z = np.zeros((2, 3))
    assert cosine(z, z) == 0.0
    assert relative_l2(np.ones((2, 3)), z) > 0 and math.isfinite(relative_l2(np.ones((2, 3)), z))


def test_split_steps_is_time_major_and_validates():
    tokens = np.arange(12).reshape(6, 2)
    out = split_steps(tokens, 2)
    assert len(out) == 3 and np.array_equal(out[1], tokens[2:4])
    with pytest.raises(ValueError):
        split_steps(tokens, 4)  # 6 rows do not divide into 4-token steps
    with pytest.raises(ValueError):
        split_steps(tokens.ravel(), 2)


# ---- forecast_scores -----------------------------------------------------------------------------------


def test_a_perfect_forecast_has_cosine_one_zero_error_and_skill_one():
    truth = steps()
    s = forecast_scores(truth, truth, persistence=truth[0] + 1.0)
    assert s["cosine"] == pytest.approx(1.0) and s["relative_l2"] == pytest.approx(0.0)
    assert s["skill"] == pytest.approx(1.0)
    assert s["steps"] == 4 and len(s["cosine_per_step"]) == 4


def test_a_forecast_no_better_than_persistence_has_skill_zero():
    truth = steps()
    base = truth[0] * 0.5
    s = forecast_scores([base] * 4, truth, persistence=base)
    assert s["skill"] == pytest.approx(0.0)
    assert s["cosine"] == pytest.approx(s["persistence_cosine"])


def test_a_forecast_worse_than_persistence_has_negative_skill():
    truth = steps()
    s = forecast_scores([-t for t in truth], truth, persistence=truth[0])
    assert s["skill"] < 0
    assert s["cosine"] == pytest.approx(-1.0)


def test_skill_is_none_when_persistence_is_already_exact():
    flat = [np.ones((2, 3))] * 3
    s = forecast_scores(flat, flat, persistence=np.ones((2, 3)))
    assert s["skill"] is None


def test_drift_slope_is_negative_when_fidelity_decays_with_the_horizon():
    truth = [np.ones((1, 4)) * np.array([1.0, 0.0, 0.0, 0.0]) for _ in range(5)]
    # rotate the prediction further from the truth at every step
    pred = [np.array([[math.cos(0.3 * k), math.sin(0.3 * k), 0.0, 0.0]]) for k in range(5)]
    s = forecast_scores(pred, truth, persistence=truth[0], steps_per_second=2.0)
    assert s["drift_per_step"] < 0
    assert s["drift_per_second"] == pytest.approx(s["drift_per_step"] * 2.0)
    assert s["cosine_per_step"][0] == pytest.approx(1.0)


def test_a_single_step_forecast_has_no_drift_slope():
    truth = steps(1)
    assert forecast_scores(truth, truth, persistence=truth[0])["drift_per_step"] is None


def test_shape_mismatches_are_rejected():
    truth = steps()
    with pytest.raises(ValueError, match="same number of steps"):
        forecast_scores(truth[:3], truth, persistence=truth[0])
    with pytest.raises(ValueError, match="persistence"):
        forecast_scores(truth, truth, persistence=np.zeros((9, 9)))
    with pytest.raises(ValueError, match="empty"):
        forecast_scores([], [], persistence=truth[0])


# ---- forecast_agreement --------------------------------------------------------------------------------


def test_identical_forecasts_agree_perfectly():
    a = steps()
    g = forecast_agreement(a, [x.copy() for x in a])
    assert g["cosine"] == pytest.approx(1.0) and g["relative_l2"] == 0.0 and g["max_relative_l2"] == 0.0


def test_agreement_reports_the_worst_step_as_well_as_the_mean():
    a = steps()
    b = [x.copy() for x in a]
    b[2] = -b[2]  # one step completely flipped
    g = forecast_agreement(a, b)
    assert g["min_cosine"] == pytest.approx(-1.0)
    assert g["max_relative_l2"] == pytest.approx(2.0)
    assert g["cosine"] == pytest.approx((3 * 1.0 + -1.0) / 4)


def test_agreement_is_symmetric_in_cosine():
    a, b = steps(seed=0), steps(seed=50)
    assert forecast_agreement(a, b)["cosine"] == pytest.approx(forecast_agreement(b, a)["cosine"])
