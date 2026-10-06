"""Physics-Aware Efficiency Score (outline §3.3):

    PAES = Speedup x Physics_Score / Drift_Penalty
    Drift_Penalty = 1 + decay_rate x horizon_T

where decay_rate is how fast the physics score *falls* per second, and a score
that holds steady or rises over the rollout is not penalized.

Deliberately a single small pure function — see outline §3.3 for why this
specific shape (a method that trades physics accuracy for speed should score
lower than one that doesn't) and §7.5 for the plan to validate/ablate the
formula's weights once real numbers exist.

History: the first version penalized |drift_rate|, i.e. a rising score cost as
much as a falling one. On the first real runs (an under-trained Dreamer whose
score fluctuated around a flat level, with per-scenario drift rates of -0.006,
+0.001 and +0.005 per second) that penalized noise in both directions and
treated improvement as drift. `drift_mode="abs"` keeps the old behavior for
comparison with earlier results; `"decay"` (the default) is what the metric is
meant to measure.
"""

from __future__ import annotations

from typing import Literal

DriftMode = Literal["decay", "abs"]


def compute_paes(
    speedup: float,
    physics_score: float,
    drift_rate: float,
    horizon_t: float,
    drift_mode: DriftMode = "decay",
) -> float:
    """
    Args:
        speedup: baseline_latency / optimized_latency (1.0 if no optimization applied).
        physics_score: the 0-1 physics score for this rollout (PAI-Bench, or
            simulator fidelity for simulator-backed models).
        drift_rate: physics score change per second (from
            worldoptbench.metrics.physics.compute_drift_curve). Negative means
            the score is decaying as the rollout gets longer.
        horizon_t: rollout horizon in seconds the drift_rate was measured over.
        drift_mode: "decay" penalizes only a falling score (drift_rate < 0);
            "abs" penalizes the magnitude in either direction (the original
            definition, kept so older results stay reproducible).

    Returns:
        A single comparable score — higher is better.

    Raises:
        ValueError: on an unknown drift_mode.
    """
    if drift_mode == "decay":
        penalized_rate = max(0.0, -drift_rate)
    elif drift_mode == "abs":
        penalized_rate = abs(drift_rate)
    else:
        raise ValueError(f"unknown drift_mode {drift_mode!r}; use 'decay' or 'abs'")
    drift_penalty = 1 + penalized_rate * horizon_t
    return (speedup * physics_score) / drift_penalty
