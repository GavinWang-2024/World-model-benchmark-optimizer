"""Cost per generated second and the cheapest-above-a-floor query."""

import pytest

from worldoptbench.cost import (
    OUTLINE_H100_SPOT_USD_PER_HOUR,
    CostRow,
    cheapest_meeting,
    cost_pareto,
    cost_rows,
    usd_per_generated_second,
)


def _rows(latency, horizon=2.0, speedup=1.0, n=2):
    return [{"horizon": horizon, "speed": {"latency_seconds": latency, "speedup": speedup}} for _ in range(n)]


def test_cost_per_second_is_latency_per_horizon_times_the_hourly_rate():
    # 10 s of compute for 2 s of video at $3.60/hour = $0.001/s of compute => 5 s of compute per video-second
    assert usd_per_generated_second(10.0, 2.0, 3.6) == pytest.approx(5.0 * 3.6 / 3600.0)
    assert usd_per_generated_second(0.0, 2.0, 3.6) == 0.0
    assert usd_per_generated_second(10.0, 2.0, 0.0) == 0.0  # your own hardware at no marginal cost


def test_invalid_inputs_are_rejected():
    for bad in ((-1.0, 2.0, 1.0), (1.0, 0.0, 1.0), (1.0, 2.0, -0.5)):
        with pytest.raises(ValueError):
            usd_per_generated_second(*bad)


def test_the_outline_rate_is_the_midpoint_of_its_quoted_range():
    assert OUTLINE_H100_SPOT_USD_PER_HOUR == pytest.approx((1.5 + 2.0) / 2)


def test_rows_are_sorted_cheapest_first_and_a_faster_config_costs_less():
    rows = cost_rows({"slow": _rows(12.0), "fast": _rows(6.0, speedup=2.0)}, usd_per_hour=3.6)
    assert [r.name for r in rows] == ["fast", "slow"]
    assert rows[0].usd_per_second == pytest.approx(rows[1].usd_per_second / 2)
    assert rows[0].speedup == pytest.approx(2.0)


def test_rows_without_a_horizon_are_skipped():
    assert cost_rows({"x": [{"horizon": 0, "speed": {"latency_seconds": 1.0}}]}, 1.0) == []


def test_cheapest_meeting_a_quality_floor_ignores_rows_with_no_quality():
    rows = [CostRow("cheap_bad", 1.0, 3.0, 0.4, {}), CostRow("mid_ok", 2.0, 2.0, 0.8, {}),
            CostRow("dear_good", 3.0, 1.0, 0.95, {}), CostRow("unscored", 0.5, 5.0, None, {})]
    assert cheapest_meeting(rows, 0.75).name == "mid_ok"
    assert cheapest_meeting(rows, 0.9).name == "dear_good"
    assert cheapest_meeting(rows, 0.99) is None


def test_cost_pareto_keeps_rows_nobody_beats_on_both_cost_and_quality():
    rows = [CostRow("a", 1.0, 3.0, 0.5, {}), CostRow("b", 2.0, 2.0, 0.9, {}),
            CostRow("c", 2.5, 1.5, 0.7, {}),  # dearer and worse than b
            CostRow("d", 3.0, 1.0, 0.95, {})]
    assert [r.name for r in cost_pareto(rows)] == ["a", "b", "d"]
