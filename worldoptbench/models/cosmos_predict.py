"""Cosmos-Predict2.5-2B (diffusers) wrapper implementing WorldModelInterface.

Uses the diffusers `Cosmos2_5_PredictBasePipeline` and the `diffusers/base/post-trained` branch of
`nvidia/Cosmos-Predict2.5-2B` (gated: accept the NVIDIA Open Model License on the model page, and put a read token in
`.env`; see `worldoptbench/env.py`). Unlike Wan this IS a world foundation model: it generates from text alone
(Text2World), from one image (Image2World, `init_frame`) or from the last frames of a video (Video2World,
`init_video`). It is not action-conditioned (that is a separate `robot/action-cond` checkpoint in the same repo and
a different pipeline), so `actions=` raises.

Prompts are not encoded here. The Qwen2.5-VL-7B text encoder is ~16.6 GB in bf16 and does not fit on a 12 GB GPU beside
the transformer, so embeddings are computed once on CPU by `scripts/encode_cosmos_prompts.py` and loaded from a file;
the pipeline is built without a text encoder. Embeddings are large (512 tokens x 100,352 features, ~100 MB per prompt).

Safety checker (NVIDIA Open Model License section 2.1): the license ends if the model's safety guardrail is bypassed or
disabled without a substantially similar one, and the pipeline itself refuses to run without a checker. This wrapper
therefore never disables it. By default it builds the real `cosmos_guardrail.CosmosSafetyChecker`; if that package is not
installed it raises with instructions. Pass your own `safety_checker=` to substitute one. Two consequences worth knowing:
the checker runs inside the pipeline call, so its time is part of every measured latency (a constant that dilutes
speedups), and because prompts arrive as pre-computed embeddings the pipeline's text check is skipped (the video check on
the output still runs). Run the text check at encode time if the prompts need it.

Measured on the RTX 5070 Ti laptop GPU (12 GB), 480x832, 17 frames, 36 steps, one prompt: 113 s per clip, 7.3 GB peak with VAE
tiling, deterministic for a fixed seed. The real guardrail (face pixelation on the video, text check skipped for embeddings)
costs ~0.2 s per 33-frame clip and 1.2 GB of GPU memory.

Differences from Wan that matter for the optimization modules:
* The two guidance passes run as two separate transformer calls (`cfg_batched = False`), same as Wan.
* Guidance is `pos + s * (pos - neg)`, i.e. an effective scale of s + 1 compared with Wan's `neg + s * (pos - neg)`. The
  default `guidance_scale=7.0` is the model's own.
* `CosmosTransformer3DModel` has `transformer_blocks` (not `blocks`), no diffusers cache mixin (`cache_context`,
  `enable_cache`), no entry in diffusers' block registry, and its pipeline never enters `cache_context`. `make_cacheable`
  and `register_cosmos_blocks` add all of that, so the hook-based caches attach; whether each one WORKS is measured, not
  assumed (`library_report` only checks capabilities).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from worldoptbench.models.base import ModelInfo, Rollout, WorldModelInterface
from worldoptbench.models.wan_video import (
    WanReferenceScenarios,
    compose_callbacks,
    frames_for_horizon,
)

REPO_ID = "nvidia/Cosmos-Predict2.5-2B"
REVISION = "diffusers/base/post-trained"
DEFAULT_EMBEDDINGS = Path(__file__).resolve().parent.parent.parent / "results" / "cosmos_prompts.pt"
FPS = 16  # Cosmos-Predict2.5 generates 16 fps video
LICENSE_NOTICE = "Built on NVIDIA Cosmos. Licensed by NVIDIA Corporation under the NVIDIA Open Model License."

GUARDRAIL_HELP = (
    "Cosmos-Predict2.5's license requires its safety guardrail, and the pipeline will not run without one. Install the "
    "real one with `pip install cosmos_guardrail` (it downloads nvidia/Cosmos-1.0-Guardrail, ~35 GB, and upgrades numpy "
    "and opencv in the environment), or pass `safety_checker=` with a checker of your own."
)


def default_safety_checker() -> Any:
    """The real NVIDIA guardrail, or a RuntimeError saying how to get it. Never a stub."""
    try:
        from cosmos_guardrail import CosmosSafetyChecker
    except ImportError as exc:
        raise RuntimeError(GUARDRAIL_HELP) from exc

    class DeviceAwareSafetyChecker(CosmosSafetyChecker):
        """NVIDIA's checker unchanged, plus the `.device` that diffusers reads from every pipeline component (it sorts
        components by name, so the checker is asked first and the stock class raises AttributeError)."""

        @property
        def device(self):
            import torch

            for model in self.nn_models:
                for parameter in model.parameters():
                    return parameter.device
            return torch.device("cpu")

    return DeviceAwareSafetyChecker()


def register_cosmos_blocks() -> None:
    """Teaches diffusers' hook system what a `CosmosTransformerBlock` returns (a tensor, the hidden states), the same entry
    it has for Wan's block. Without it first-block cache, TaylorSeer, MagCache and the other block-level caches refuse the
    model ("Model class ... not registered"). Idempotent."""
    from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
    from diffusers.models.transformers.transformer_cosmos import CosmosTransformerBlock

    try:
        TransformerBlockRegistry.get(CosmosTransformerBlock)
    except ValueError:
        TransformerBlockRegistry.register(
            CosmosTransformerBlock,
            TransformerBlockMetadata(return_hidden_states_index=0, return_encoder_hidden_states_index=None),
        )


def make_cacheable(transformer: Any) -> Any:
    """Gives a `CosmosTransformer3DModel` what diffusers' caching hooks expect of a transformer, in place.

    Two things are missing from the stock model. It has no `CacheMixin` (`enable_cache`, `cache_context`), and the Cosmos
    pipeline never enters `cache_context`, which the Wan and CogVideoX pipelines do around each guidance pass. Without
    contexts a stateful cache would compare each pass with the *other* pass's state (conditional vs unconditional). So a
    torch forward pre-hook sets the context to "uncond" when the transformer is called with the negative prompt's
    embeddings (`transformer._uncond_embeds`, set by `CosmosPredict.generate`) and to "cond" otherwise, and a forward hook
    clears it. These are torch module hooks, not diffusers hooks, so they run OUTSIDE every diffusers hook, including ones
    on the transformer itself (WorldCache's), which need the context to already be set. The object keeps its weights; only
    its class changes. Returns the transformer.
    """
    from diffusers.hooks import HookRegistry
    from diffusers.models.cache_utils import CacheMixin
    from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel

    register_cosmos_blocks()
    if isinstance(transformer, CacheMixin):
        return transformer

    class CacheableCosmosTransformer(CacheMixin, CosmosTransformer3DModel):
        _uncond_embeds: Any = None

    def enter_context(module: Any, args: tuple, kwargs: dict) -> None:
        embeds = kwargs.get("encoder_hidden_states")
        context = "uncond" if embeds is not None and embeds is module._uncond_embeds else "cond"
        HookRegistry.check_if_exists_or_initialize(module)._set_context(context)

    def leave_context(module: Any, args: tuple, kwargs: dict, output: Any) -> None:
        HookRegistry.check_if_exists_or_initialize(module)._set_context(None)

    transformer.__class__ = CacheableCosmosTransformer
    transformer.register_forward_pre_hook(enter_context, with_kwargs=True)
    transformer.register_forward_hook(leave_context, with_kwargs=True)
    return transformer


def _as_uint8(frame: Any) -> np.ndarray:
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = (array * 255.0).round().clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


class CosmosPredict(WorldModelInterface):
    def __init__(
        self,
        embeddings_path: str | Path = DEFAULT_EMBEDDINGS,
        model_path: str | Path | None = None,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 36,
        guidance_scale: float = 7.0,
        num_latent_conditional_frames: int = 2,
        safety_checker: Any = None,
        vae_tiling: bool = True,
        device: str = "cuda",
    ):
        """
        Args:
            embeddings_path: file written by scripts/encode_cosmos_prompts.py.
            model_path: local diffusers directory; default: the Hugging Face cache copy of the post-trained branch.
            height, width: output size in pixels; multiples of 16. Default 480x832 (the model's 480p). Measured (one prompt,
                seed 0, 17 frames): 480x832 is a clean, coherent video; 384x672 and 320x576 are incoherent blobs and 256x448
                is noise, so the model has a minimum usable resolution and this is the smallest tested size that works.
            num_inference_steps, guidance_scale: sampler settings (the model's defaults are 36 and 7.0).
                `num_inference_steps` is read at generate time, so a module (fewer_steps) can change it.
            num_latent_conditional_frames: 1 or 2, how many latent frames a conditioning video contributes (1 = a single
                frame, 2 = the last 5 frames).
            safety_checker: see the module docstring. None builds the real NVIDIA guardrail.
            vae_tiling: decode in tiles. Without it 480x832 peaks at 13.8 GB, over a 12 GB GPU, and slows 2x from spilling
                into shared memory; with it the peak is ~7.3 GB (measured).
        """
        if height % 16 or width % 16:
            raise ValueError("height and width must be multiples of 16")
        if num_latent_conditional_frames not in (1, 2):
            raise ValueError("num_latent_conditional_frames must be 1 or 2")
        import torch
        from diffusers import AutoencoderKLWan, Cosmos2_5_PredictBasePipeline
        from huggingface_hub import snapshot_download

        self.height, self.width = height, width
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.num_latent_conditional_frames = num_latent_conditional_frames
        self.cfg_batched = False
        self.step_callbacks: list = []
        self._device = torch.device(device)

        saved = torch.load(Path(embeddings_path), map_location="cpu")
        self._prompt_embeds: dict[str, Any] = saved["prompts"]
        self._negative_embeds = saved["negative"]

        checker = safety_checker if safety_checker is not None else default_safety_checker()
        path = str(model_path) if model_path else snapshot_download(
            REPO_ID, revision=REVISION, allow_patterns=["model_index.json", "scheduler/*", "transformer/*", "vae/*"]
        )
        vae = AutoencoderKLWan.from_pretrained(path, subfolder="vae", torch_dtype=torch.float32)
        self.pipeline = Cosmos2_5_PredictBasePipeline.from_pretrained(
            path, vae=vae, text_encoder=None, tokenizer=None, safety_checker=checker, torch_dtype=torch.bfloat16
        ).to(self._device)
        self.transformer = make_cacheable(self.pipeline.transformer)
        self.pipeline.set_progress_bar_config(disable=True)
        if vae_tiling:
            self.pipeline.vae.enable_tiling()

    @property
    def prompts(self) -> list[str]:
        """The prompts that have embeddings and can be generated."""
        return list(self._prompt_embeds)

    def torch_module(self) -> Any:
        """The transformer (satisfies models.base.TorchBacked)."""
        return self.transformer

    def _embeddings(self, prompt: str) -> tuple[Any, Any]:
        if prompt not in self._prompt_embeds:
            raise KeyError(
                f"no embedding for prompt {prompt!r}. Encode it once with "
                f"`python scripts/encode_cosmos_prompts.py --prompt \"...\"`. Available: {self.prompts}"
            )
        return self._prompt_embeds[prompt].to(self._device), self._negative_embeds.to(self._device)

    def generate(
        self,
        prompt: str | None = None,
        init_frame: Any | None = None,
        init_video: Any | None = None,
        actions: list[Any] | None = None,
        horizon: float = 2.0,
        **kwargs: Any,
    ) -> Rollout:
        """
        Args:
            prompt: must be one of `self.prompts`. Required (conditioning images and videos still take a prompt).
            init_frame: one image (HxWx3 uint8) to continue from (Image2World).
            init_video: a list of frames (HxWx3 uint8) whose last 1 or 5 frames condition the rollout (Video2World).
            horizon: video length in seconds at 16 fps, rounded to a valid frame count (4k + 1).
            seed (kwarg): seeds the initial noise, default 0.
            guidance_scale (kwarg): overrides the constructor's value for this call.
        """
        if prompt is None:
            raise ValueError("CosmosPredict needs a prompt (one of its encoded prompts)")
        if init_frame is not None and init_video is not None:
            raise ValueError("pass init_frame or init_video, not both")
        if actions is not None:
            raise NotImplementedError(
                "Cosmos-Predict2.5 base is not action-conditioned (the action-conditioned model is a separate checkpoint)"
            )
        import torch

        seed = int(kwargs.pop("seed", 0))
        guidance = float(kwargs.pop("guidance_scale", self.guidance_scale))
        positive, negative = self._embeddings(prompt)
        self.transformer._uncond_embeds = negative  # lets the transformer tell the two guidance passes apart
        n_frames = frames_for_horizon(horizon, FPS)

        call_kwargs: dict[str, Any] = {}
        if init_frame is not None:
            call_kwargs["image"] = _as_uint8(init_frame)
        elif init_video is not None:
            call_kwargs["video"] = [_as_uint8(f) for f in init_video]
            call_kwargs["num_latent_conditional_frames"] = self.num_latent_conditional_frames
        if self.step_callbacks:
            call_kwargs["callback_on_step_end"] = compose_callbacks(self.step_callbacks)
        call_kwargs.update(kwargs)
        with torch.no_grad():
            output = self.pipeline(
                prompt_embeds=positive,
                negative_prompt_embeds=negative,
                height=self.height,
                width=self.width,
                num_frames=n_frames,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=guidance,
                generator=torch.Generator(device=self._device).manual_seed(seed),
                output_type="np",
                **call_kwargs,
            )
        video = np.asarray(output.frames[0])  # (frames, H, W, 3) float in [0, 1]
        return Rollout(
            frames=[_as_uint8(f) for f in video],
            fps=float(FPS),
            metadata={
                "prompt": prompt, "horizon": horizon, "model": self.get_info().name, "seed": seed,
                "num_frames": n_frames, "num_inference_steps": self.num_inference_steps, "guidance_scale": guidance,
                "conditioning": "image" if init_frame is not None else "video" if init_video is not None else "text",
                "notice": LICENSE_NOTICE,
            },
        )

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            name="Cosmos-Predict2.5-2B",
            architecture="diffusion",
            param_count=int(sum(p.numel() for p in self.transformer.parameters())),
            supported_optimizations=[],
            supports_image_conditioning=True,
            supports_video_conditioning=True,
        )


class CosmosReferenceScenarios(WanReferenceScenarios):
    """ScenarioFn for Cosmos quality: the reference is the unoptimized model's video for the same prompt, seed and length
    (see `WanReferenceScenarios`, whose caching and scoring this reuses). Each Cosmos rollout takes ~2 minutes and the model
    plus guardrail fill 7 GB of a 12 GB GPU, so build the references BEFORE loading the model under test (the sweep script
    does, in its own process): two models on one 12 GB GPU spill into shared memory and distort every timing.
    """

    def __init__(self, cache_dir: str | Path, baseline_factory: Any = None, model_kwargs: dict[str, Any] | None = None):
        kwargs = dict(model_kwargs or {})
        super().__init__(cache_dir, baseline_factory or (lambda: CosmosPredict(**kwargs)), kwargs)

    def _settings_tag(self) -> str:
        kw = self._model_kwargs
        return "cosmos_{}x{}_s{}_g{}".format(
            kw.get("height", 480), kw.get("width", 832), kw.get("num_inference_steps", 36), kw.get("guidance_scale", 7.0)
        )
