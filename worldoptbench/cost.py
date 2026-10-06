"""Cost-aware comparison: dollars per generated second of video (outline section 12-E).

LLM serving has mature $-per-token Pareto frontiers; for world models the equivalent question is "given a quality
floor, which configuration is cheapest per generated second". The arithmetic is trivial; the honest parts are what
the inputs mean:

* Latency is measured on ONE machine (here an RTX 5070 Ti laptop GPU). The hourly rate you pass is that machine's
  cost: a rented equivalent, or your own hardware amortized, or just electricity. This module does not translate a
  latency measured on one GPU into the latency on another (that would be a guess), so it cannot recommend hardware;
  to compare machines, benchmark on each and pass each its own rate.
* The cost is for the model's compute only: no queueing, batching gains, storage or egress.
* `usd_per_hour` has no default. `OUTLINE_H100_SPOT_USD_PER_HOUR` is the midpoint of the range quoted in the outline
  (about $1.50-2.00 for an H100 spot instance, from build_plan.md), not a current price; check the provider.

    usd_per_generated_second = latency_seconds_per_rollout / horizon_seconds * usd_per_hour / 3600
"""

from __future__ import annotations

import statistics as st
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

OUTLINE_H100_SPOT_USD_PER_HOUR = 1.75


def usd_per_generated_second(latency_seconds: float, horizon_seconds: float, usd_per_hour: float) -> float:
    """What one second of generated video costs when a rollout takes `latency_seconds` for `horizon_seconds` of output."""
    if latency_seconds < 0 or horizon_seconds <= 0 or usd_per_hour < 0:
        raise ValueError("latency and rate must be non-negative and the horizon positive")
    return latency_seconds / horizon_seconds * usd_per_hour / 3600.0


@dataclass
class CostRow:
    name: str
    usd_per_second: float
    speedup: float | None
    quality: float | None  # whatever quality number the caller supplied (higher = better)
    extra: dict[str, Any]


def cost_rows(
    configs: dict[str, Sequence[Any]], usd_per_hour: float, quality: dict[str, float] | None = None
) -> list[CostRow]:
    """One row per configuration from its rollouts (RunResult dicts: `speed.latency_seconds`, `horizon`,
    `speed.speedup`). `quality` optionally maps configuration name to a quality score to rank by."""
    rows = []
    for name, results in configs.items():
        per_second = [
            usd_per_generated_second(r["speed"]["latency_seconds"], r["horizon"], usd_per_hour)
            for r in results if r.get("horizon")
        ]
        if not per_second:
            continue
        speedups = [r["speed"]["speedup"] for r in results if r["speed"].get("speedup") is not None]
        rows.append(CostRow(name, st.mean(per_second), st.mean(speedups) if speedups else None,
                            (quality or {}).get(name), {}))
    return sorted(rows, key=lambda r: r.usd_per_second)


def cheapest_meeting(rows: Sequence[CostRow], min_quality: float) -> CostRow | None:
    """The cheapest row whose quality is at least `min_quality` (rows without a quality number never qualify)."""
    eligible = [r for r in rows if r.quality is not None and r.quality >= min_quality]
    return min(eligible, key=lambda r: r.usd_per_second) if eligible else None


def cost_pareto(rows: Sequence[CostRow]) -> list[CostRow]:
    """Rows no other row beats on both cost (lower) and quality (higher); rows with no quality are left out."""
    scored = [r for r in rows if r.quality is not None]
    front = [
        r for r in scored
        if not any(o is not r and o.usd_per_second <= r.usd_per_second and o.quality >= r.quality
                   and (o.usd_per_second < r.usd_per_second or o.quality > r.quality) for o in scored)
    ]
    return sorted(front, key=lambda r: r.usd_per_second)
