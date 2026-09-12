"""WorldModelInterface — the abstraction every world model wrapper implements.

See world_model_project_outline.md §3.1 and §12 item G. The benchmark runner
and optimization stack only ever call .generate() and .get_info() on this —
they never know or care what architecture is underneath.

Conditioning inputs (prompt / init_frame / init_video) are all optional so a
caller can do Text2World, Image2World, or Video2World depending on what the
concrete model actually supports — check ModelInfo.supports_*_conditioning
before relying on image/video conditioning doing something meaningful.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

Architecture = Literal["diffusion", "autoregressive", "jepa"]


@dataclass
class ModelInfo:
    """Static facts about a model.

    Used by OptimizationStack (§3.4) to filter compatible modules by
    architecture, and by the benchmark runner (§3.2) for reporting.
    """

    name: str
    architecture: Architecture
    param_count: int  # e.g. 7_000_000_000 for the 7B checkpoint
    supported_optimizations: list[str] = field(default_factory=list)
    supports_image_conditioning: bool = False
    supports_video_conditioning: bool = False


@dataclass
class Rollout:
    """Standard output of a .generate() call.

    `frames` is deliberately untyped (not `np.ndarray`) so importing this
    module never requires numpy/torch to be installed — only .generate()
    implementations need those, and only when actually called. The benchmark
    runner (Phase 2) is responsible for normalizing whatever a concrete
    wrapper returns into a consistent array type before computing metrics.
    """

    frames: list[Any]
    fps: float
    metadata: dict[str, Any] = field(default_factory=dict)


class WorldModelInterface(ABC):
    """Any world model — diffusion, autoregressive, JEPA — implements this."""

    @abstractmethod
    def generate(
        self,
        prompt: str | None = None,
        init_frame: Any | None = None,
        init_video: Any | None = None,
        actions: list[Any] | None = None,
        horizon: float = 4.0,
        **kwargs: Any,
    ) -> Rollout:
        """Generate a rollout.

        Args:
            prompt: text conditioning. Optional if init_frame/init_video is given.
            init_frame: a single starting frame — Image2World conditioning.
            init_video: a short starting clip — Video2World conditioning.
            actions: action sequence conditioning, one entry per rollout step.
            horizon: rollout length in seconds.
            **kwargs: architecture-specific extras (e.g. num_inference_steps,
                guidance_scale for diffusion models). Concrete wrappers
                document whatever they accept here.

        Returns:
            A Rollout with the generated frames.

        Raises:
            ValueError: if none of prompt/init_frame/init_video is given —
                every implementation needs at least one conditioning input.
        """
        raise NotImplementedError

    @abstractmethod
    def get_info(self) -> ModelInfo:
        """Static info about this model: architecture, size, what it supports."""
        raise NotImplementedError
