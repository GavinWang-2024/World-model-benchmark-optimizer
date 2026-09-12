"""Cosmos-Predict2 wrapper implementing WorldModelInterface.

Requires the `ml` extra (`pip install -e ".[ml]"`) PLUS NVIDIA's own
cosmos_predict2 package, which isn't on PyPI as a normal dependency — clone
and install it per NVIDIA's instructions as part of build_plan.md Phase 0.
Needs an HF token with access to the gated nvidia/Cosmos-Predict2* repos.

API confirmed vs. inferred (2026-09-12, no GPU access yet to verify — see
build_plan.md Phase 0):
  - CONFIRMED from nvidia-cosmos/cosmos-predict2 docs: Video2World is called
    via `Video2WorldPipeline.from_config(config=..., dit_path=...)` then
    `pipe(input_path=..., prompt=...)`. CLI equivalent:
    `python -m examples.video2world --model_size 2B --input_path ... --prompt ...`
  - CONFIRMED CLI-level: Text2World has its own script
    (`python -m examples.text2world --model_size 2B --prompt ...`), and per
    NVIDIA's docs it internally chains Text2Image -> Video2World.
  - INFERRED BY ANALOGY, NOT CONFIRMED: a standalone `Text2WorldPipeline`
    class + `get_cosmos_predict2_text2world_pipeline` /
    `_checkpoint` config helpers, mirroring the confirmed Video2World
    pattern. If these names are wrong, this will raise ImportError
    immediately and loudly — fix against the actual installed package once
    Phase 0's stock example run confirms the real names.
"""

from __future__ import annotations

from typing import Any

from worldoptbench.models.base import ModelInfo, Rollout, WorldModelInterface

_PARAM_COUNTS = {
    "2B": 2_000_000_000,
    "7B": 7_000_000_000,
}


class CosmosPredict(WorldModelInterface):
    """Shared base for the Cosmos-Predict2 checkpoints used in v1 (outline §3.1).

    Loading is lazy (first .generate() call, not __init__) so importing this
    module — or even constructing a wrapper to call .get_info() — never
    requires torch/cosmos_predict2 to be installed.
    """

    _SIZE: str  # set by subclasses below: "2B" or "7B"

    def __init__(self, device: str = "cuda", **pipeline_kwargs: Any):
        self._device = device
        self._pipeline_kwargs = pipeline_kwargs
        self._text2world_pipe = None
        self._video2world_pipe = None

    def _load_text2world(self):
        if self._text2world_pipe is not None:
            return self._text2world_pipe

        # INFERRED — see module docstring. Verify against the installed
        # package; the CLI path (examples.text2world) is confirmed to exist
        # even if this exact class layout isn't.
        from cosmos_predict2.configs.base.config_text2world import (
            get_cosmos_predict2_text2world_pipeline,
        )
        from cosmos_predict2.pipelines.text2world import Text2WorldPipeline

        from imaginaire.constants import get_cosmos_predict2_text2world_checkpoint

        self._text2world_pipe = Text2WorldPipeline.from_config(
            config=get_cosmos_predict2_text2world_pipeline(model_size=self._SIZE),
            dit_path=get_cosmos_predict2_text2world_checkpoint(model_size=self._SIZE),
            **self._pipeline_kwargs,
        )
        return self._text2world_pipe

    def _load_video2world(self):
        if self._video2world_pipe is not None:
            return self._video2world_pipe

        # CONFIRMED pattern from NVIDIA docs — see module docstring.
        from cosmos_predict2.configs.base.config_video2world import (
            get_cosmos_predict2_video2world_pipeline,
        )
        from cosmos_predict2.pipelines.video2world import Video2WorldPipeline

        from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint

        self._video2world_pipe = Video2WorldPipeline.from_config(
            config=get_cosmos_predict2_video2world_pipeline(model_size=self._SIZE),
            dit_path=get_cosmos_predict2_video2world_checkpoint(model_size=self._SIZE),
            **self._pipeline_kwargs,
        )
        return self._video2world_pipe

    def generate(
        self,
        prompt: str | None = None,
        init_frame: Any | None = None,
        init_video: Any | None = None,
        actions: list[Any] | None = None,
        horizon: float = 4.0,
        **kwargs: Any,
    ) -> Rollout:
        if prompt is None and init_frame is None and init_video is None:
            raise ValueError("generate() needs at least one of prompt, init_frame, init_video")

        if actions is not None:
            # Cosmos-Predict2's public pipelines are text/image/video
            # conditioned, not action-conditioned out of the box — action
            # conditioning would need a fine-tuned checkpoint. Not needed
            # for the v1 baseline runs, so fail loudly rather than silently
            # ignoring the argument.
            raise NotImplementedError(
                "Cosmos-Predict2 wrapper doesn't support action conditioning yet"
            )

        conditioning_input = init_video if init_video is not None else init_frame

        if conditioning_input is not None:
            pipe = self._load_video2world()
            # num_conditional_frames=1 for a single image, matches NVIDIA's
            # documented image2world usage; video conditioning would pass
            # more frames — left as a kwarg override for callers who need it.
            kwargs.setdefault("num_conditional_frames", 1 if init_frame is not None else None)
            output = pipe(input_path=conditioning_input, prompt=prompt, **kwargs)
        else:
            pipe = self._load_text2world()
            output = pipe(prompt=prompt, **kwargs)

        # NOTE: exact output shape (frames list vs. tensor vs. wrapper object,
        # fps field name) needs confirming against the real pipeline once it
        # runs — adjust this unpacking then. Structured as a single place to
        # fix rather than scattered across callers.
        return Rollout(
            frames=list(output.frames) if hasattr(output, "frames") else list(output),
            fps=getattr(output, "fps", kwargs.get("fps", 16)),
            metadata={"prompt": prompt, "horizon": horizon, "model": f"Cosmos-Predict2-{self._SIZE}"},
        )

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            name=f"Cosmos-Predict2-{self._SIZE}",
            architecture="diffusion",
            param_count=_PARAM_COUNTS[self._SIZE],
            supported_optimizations=["worldcache", "adacache", "fp8_quantization"],
            supports_image_conditioning=True,
            supports_video_conditioning=True,
        )


class CosmosPredict2B(CosmosPredict):
    """Lightweight checkpoint for fast experiments (outline §3.1)."""

    _SIZE = "2B"


class CosmosPredict7B(CosmosPredict):
    """Primary v1 target (outline §3.1). NOTE: as of 2026-09-12 the confirmed
    public sizes are 2B and 14B, not 7B — double-check the exact repo id
    during Phase 0 and adjust _PARAM_COUNTS / _SIZE here if NVIDIA's naming
    has shifted.
    """

    _SIZE = "7B"
