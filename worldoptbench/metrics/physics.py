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
    # Set instead of the two above when ground-truth reference frames exist
    # (simulator-backed models) — see compute_sim_fidelity.
    sim_fidelity_score: float | None = None
    sim_pixel_score: float | None = None  # the raw pixel-error score alongside it, for reference
    sim_error_curve: list[float] | None = None
    sim_persistence_error_curve: list[float] | None = None  # error of the "nothing moves" baseline
    drift_onset: float | None = None  # seconds into the rollout where score first crosses threshold
    drift_rate: float | None = None  # score change per second, from compute_drift_curve


@dataclass
class DriftCurve:
    drift_onset: float | None
    drift_rate: float


@dataclass
class SimFidelity:
    score: float  # 0-1, higher is better: the skill score if a baseline was usable, else pixel_score
    pixel_score: float  # 1 - mean normalized pixel error; compresses into a narrow band near 1.0
    error_curve: list[float]  # normalized per-step error, 0-1 — rises as the rollout drifts
    persistence_error_curve: list[float] | None = None  # same, for the "nothing moves" baseline


# Below this, the "nothing moves" baseline is already essentially perfect (a
# static scene), so a ratio against it would be dominated by noise.
_MIN_PERSISTENCE_ERROR = 1e-3


def _mean_abs_error(a: Any, b: Any) -> float:
    import numpy as np

    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).mean()) / 255.0


def compute_sim_fidelity(
    frames: Sequence[Any],
    reference_frames: Sequence[Any],
    baseline_frame: Any | None = None,
) -> SimFidelity:
    """How closely a generated rollout tracks the simulator's true rollout
    under the same actions.

    Raw pixel error alone is nearly useless here: most pixels are static
    background, so on a real Dreamer rollout an undertrained model's blurry
    blob scored 0.93 while a predictor that just repeats the last context
    frame scored 0.92 (perfect = 1.0). So when `baseline_frame` — the last
    frame the model saw — is given, the score is a skill score against that
    "nothing moves" predictor:

        score = clip(1 - sum(model error) / sum(baseline error), 0, 1)

    0 means no better than freezing the scene (or worse); 1 is a perfect
    match. If the baseline is itself ~perfect (a static scene) the ratio is
    meaningless, so the score falls back to the raw pixel score.

    Raises:
        ValueError: if the two sequences are empty or differ in length.
    """
    if len(frames) == 0 or len(frames) != len(reference_frames):
        raise ValueError(
            f"Need equal, non-empty frame sequences (got {len(frames)} generated, "
            f"{len(reference_frames)} reference)"
        )

    errors = [_mean_abs_error(g, r) for g, r in zip(frames, reference_frames)]
    pixel_score = max(0.0, 1.0 - sum(errors) / len(errors))

    if baseline_frame is None:
        return SimFidelity(score=pixel_score, pixel_score=pixel_score, error_curve=errors)

    baseline_errors = [_mean_abs_error(baseline_frame, r) for r in reference_frames]
    baseline_total = sum(baseline_errors)
    if baseline_total / len(baseline_errors) < _MIN_PERSISTENCE_ERROR:
        score = pixel_score
    else:
        score = min(1.0, max(0.0, 1.0 - sum(errors) / baseline_total))
    return SimFidelity(
        score=score,
        pixel_score=pixel_score,
        error_curve=errors,
        persistence_error_curve=baseline_errors,
    )


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
    frames: Sequence[Any],
    actions: Sequence[Any] | None = None,
    reference_frames: Sequence[Any] | None = None,
    baseline_frame: Any | None = None,
) -> PhysicsScore:
    """Convenience wrapper: simulator fidelity if ground-truth
    `reference_frames` exist, otherwise the PAI-Bench + WorldRoamBench scores.

    With a reference, PAI-Bench/WorldRoamBench are skipped — they judge
    plausibility of text-to-video output, which isn't what a simulator-backed
    model is being compared on.

    Without one, raises NotImplementedError until Phase 3 wires up the two
    repos above — callers (runner.py) catch that and degrade to a
    speed+visual-only run.
    """
    if reference_frames is not None:
        fidelity = compute_sim_fidelity(frames, reference_frames, baseline_frame)
        return PhysicsScore(
            sim_fidelity_score=fidelity.score,
            sim_pixel_score=fidelity.pixel_score,
            sim_error_curve=fidelity.error_curve,
            sim_persistence_error_curve=fidelity.persistence_error_curve,
        )

    pai_score = compute_pai_bench_score(frames)
    wrb_score = compute_worldroambench_score(frames, actions)
    return PhysicsScore(pai_bench_score=pai_score, worldroambench_score=wrb_score)
