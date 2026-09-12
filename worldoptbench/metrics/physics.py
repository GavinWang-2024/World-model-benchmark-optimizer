"""Physics consistency metrics — PAI-Bench, WorldRoamBench, and the
long-horizon drift curve derived from them (outline §3.2, Axis 3).

PAI-Bench (github.com/SHI-Labs/physical-ai-bench) and WorldRoamBench don't
have a documented standalone Python API as of 2026-09-12 — both are research
codebases meant to be run from their own repos (build_plan.md Phase 3 itself
warns "expect rough edges"). Rather than guess at function names that would
silently do the wrong thing, `compute_pai_bench_score` and
`compute_worldroambench_score` are explicit NotImplementedError stubs — fill
them in once each repo is actually cloned in Phase 3 and its real entrypoint
is visible.

`compute_drift_curve` doesn't depend on either repo — it's pure curve
fitting over whatever physics scores you already have at each horizon — so
it's implemented and tested now, not stubbed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class PhysicsScore:
    pai_bench_score: float | None = None
    worldroambench_score: float | None = None
    drift_onset: float | None = None  # seconds into the rollout where score first crosses threshold
    drift_rate: float | None = None  # score change per second, from compute_drift_curve


@dataclass
class DriftCurve:
    drift_onset: float | None
    drift_rate: float


def compute_pai_bench_score(frames: Sequence[Any]) -> float:
    """TODO (Phase 3): wire up against github.com/SHI-Labs/physical-ai-bench
    once cloned — see module docstring for why this isn't guessed at.
    """
    raise NotImplementedError(
        "PAI-Bench wrapper isn't wired up yet — clone SHI-Labs/physical-ai-bench "
        "in Phase 3 and implement this against its real entrypoint"
    )


def compute_worldroambench_score(
    frames: Sequence[Any], actions: Sequence[Any] | None = None
) -> float:
    """TODO (Phase 3): wire up against WorldRoamBench (arxiv:2606.31672) once
    its code repo is located and cloned — no public GitHub found as of
    2026-09-12, see module docstring.
    """
    raise NotImplementedError(
        "WorldRoamBench wrapper isn't wired up yet — locate/clone its repo in "
        "Phase 3 and implement this against its real entrypoint"
    )


def compute_drift_curve(
    scores_by_horizon: dict[float, float], threshold: float = 0.75
) -> DriftCurve:
    """Fits a simple linear drift rate across horizons and finds the first
    horizon where the score drops below `threshold` ("drift onset").

    Args:
        scores_by_horizon: physics score at each rollout horizon, e.g.
            {4: 0.95, 16: 0.90, 60: 0.71, 120: 0.55} — matches the drift
            curve outline §3.2 describes (4s/16s/60s/120s).
        threshold: score below which we call it "drifted."

    Raises:
        ValueError: if fewer than 2 horizons are given — can't fit a slope.
    """
    if len(scores_by_horizon) < 2:
        raise ValueError("Need scores at 2+ horizons to compute a drift rate")

    horizons = sorted(scores_by_horizon)
    scores = [scores_by_horizon[h] for h in horizons]

    import numpy as np

    # Simple least-squares slope — swap for something fancier once there's
    # enough real data to justify it (this is intentionally the boilerplate
    # version, not the research version).
    slope, _intercept = np.polyfit(horizons, scores, 1)

    drift_onset = next((h for h, s in zip(horizons, scores) if s < threshold), None)

    return DriftCurve(drift_onset=drift_onset, drift_rate=float(slope))


def compute_physics_score(
    frames: Sequence[Any], actions: Sequence[Any] | None = None
) -> PhysicsScore:
    """Convenience wrapper combining both benchmark scores.

    Raises NotImplementedError until Phase 3 wires up the two repos above —
    callers (runner.py) catch that and degrade to a speed+visual-only run.
    """
    pai_score = compute_pai_bench_score(frames)
    wrb_score = compute_worldroambench_score(frames, actions)
    return PhysicsScore(pai_bench_score=pai_score, worldroambench_score=wrb_score)
