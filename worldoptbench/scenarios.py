"""Scenario — per-rollout inputs and ground truth supplied by something other
than the model.

For text-to-video models (Cosmos) a rollout is fully described by a prompt and
horizon, so the runner needs nothing else. Simulator-backed models (Dreamer)
need context frames and an action sequence, and the simulator can also produce
the *true* continuation for those same actions — ground truth for PSNR/SSIM
and for the simulator-fidelity physics score.

Scenarios are built outside `model.generate()` on purpose: the runner times
only `generate()`, so simulator time never gets counted as model latency.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Scenario:
    # Extra kwargs for model.generate() (e.g. init_video=..., actions=...).
    generate_kwargs: dict[str, Any] = field(default_factory=dict)
    # True continuation for exactly the frames generate() should return.
    reference_frames: list[Any] | None = None
    # The last frame the model was shown, i.e. what a "nothing moves"
    # predictor would output. Lets the physics score be a skill score against
    # that trivial baseline — see metrics.physics.compute_sim_fidelity.
    baseline_frame: Any | None = None


# (standard-set prompt entry, horizon in seconds) -> Scenario
ScenarioFn = Callable[[dict[str, Any], float], Scenario]
