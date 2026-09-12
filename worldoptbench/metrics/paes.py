"""Physics-Aware Efficiency Score (outline §3.3):

    PAES = Speedup x Physics_Score / Drift_Penalty
    Drift_Penalty = 1 + |drift_rate| x horizon_T

Deliberately a single small pure function — see outline §3.3 for why this
specific shape (a method that trades physics accuracy for speed should score
lower than one that doesn't) and §7.5 for the plan to validate/ablate the
formula's weights once real numbers exist.
"""

from __future__ import annotations


def compute_paes(speedup: float, physics_score: float, drift_rate: float, horizon_t: float) -> float:
    """
    Args:
        speedup: baseline_latency / optimized_latency (1.0 if no optimization applied).
        physics_score: PAI-Bench or WorldRoamBench physics subscore, 0-1.
        drift_rate: physics score change per second (from
            worldoptbench.metrics.physics.compute_drift_curve) — only the
            magnitude is penalized, sign doesn't matter.
        horizon_t: rollout horizon in seconds the drift_rate was measured over.

    Returns:
        A single comparable score — higher is better.
    """
    drift_penalty = 1 + abs(drift_rate) * horizon_t
    return (speedup * physics_score) / drift_penalty
