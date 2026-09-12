"""Speed, visual, physics, and PAES metrics. Populated in Phases 2-4 (see build_plan.md)."""

from worldoptbench.metrics.paes import compute_paes
from worldoptbench.metrics.physics import DriftCurve, PhysicsScore, compute_drift_curve
from worldoptbench.metrics.speed import SpeedMetrics
from worldoptbench.metrics.visual import VisualMetrics

__all__ = [
    "DriftCurve",
    "PhysicsScore",
    "SpeedMetrics",
    "VisualMetrics",
    "compute_drift_curve",
    "compute_paes",
]

# measure_speed / compute_visual_metrics / compute_pai_bench_score /
# compute_worldroambench_score / compute_physics_score are NOT re-exported
# here — they lazy-import torch/torchmetrics internally, but importing this
# __init__ should stay possible with zero ML dependencies installed. Import
# them directly from their submodules instead.
