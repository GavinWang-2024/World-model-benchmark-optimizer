"""Wan2.1-T2V-1.3B (diffusers) wrapper implementing WorldModelInterface.

The first diffusion model in the benchmark, chosen because it is public (Apache 2.0),
runs on a 12 GB GPU, and shares its architecture class (a video DiT with a flow-matching
sampler) with Cosmos-Predict2, which is gated. It is a text-to-video model, NOT a world
model: it exercises the diffusion optimization machinery (caching, attention kernels, step
reduction) and says nothing about physics. See DESIGN_DIFFUSION.md, including how quality
is measured (agreement with the unoptimized output on the same seed).

Prompts are not encoded here. The UMT5-XXL text encoder (~11 GB in bf16) does not fit on a
12 GB GPU beside the transformer, so embeddings are computed once on CPU by
`scripts/encode_prompts.py` and loaded from a file; the pipeline is built without a text
encoder. A prompt that wasn't encoded raises a clear error telling you how to add it.

Loads at construction, not lazily: optimization modules discover hooks with `hasattr`, and a
lazy `transformer` property would silently load multiple GB whenever `library_report` or
`autotune` merely asks which modules apply.

Opt-in attributes the diffusion modules (optimizations/diffusion.py) look for:
    transformer, pipeline, step_callbacks, num_inference_steps, cfg_batched (False: Wan runs
    the conditional and unconditional guidance passes as two separate calls).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from worldoptbench.models.base import ModelInfo, Rollout, WorldModelInterface

REPO_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_EMBEDDINGS = Path(__file__).resolve().parent.parent.parent / "results" / "wan_prompts.pt"
FPS = 16  # Wan2.1 generates 16 fps video
TEMPORAL_COMPRESSION = 4  # the VAE compresses time 4x, so frame counts are 4k + 1


def frames_for_horizon(horizon: float, fps: int = FPS) -> int:
    """Seconds -> a valid Wan frame count (4k + 1, at least 5), the nearest to horizon * fps."""
    wanted = max(1, round(horizon * fps))
    k = max(1, round((wanted - 1) / TEMPORAL_COMPRESSION))
    return TEMPORAL_COMPRESSION * k + 1


def compose_callbacks(callbacks: list) -> Any:
    """Chain `callback_on_step_end`-style callables into one: each gets (pipe, step, timestep,
    callback_kwargs) and returns the kwargs dict the next one receives."""

    def chained(pipe: Any, step: int, timestep: Any, callback_kwargs: dict) -> dict:
        for callback in callbacks:
            callback_kwargs = callback(pipe, step, timestep, callback_kwargs)
        return callback_kwargs

    return chained


class WanVideo(WorldModelInterface):
    def __init__(
        self,
        embeddings_path: str | Path = DEFAULT_EMBEDDINGS,
        model_path: str | Path | None = None,
        height: int = 192,
        width: int = 320,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        device: str = "cuda",
    ):
        """
        Args:
            embeddings_path: file written by scripts/encode_prompts.py.
            model_path: local diffusers directory; default: the Hugging Face cache copy of
                `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (must already be downloaded; see scripts/fetch_wan.py).
            height, width: output size in pixels; multiples of 16. The defaults are small on purpose
                (full 480p x 81 frames takes minutes per video on a laptop GPU).
            num_inference_steps, guidance_scale: sampler settings. `num_inference_steps` is read at
                generate time, so a module (fewer_steps) can change it after construction.
        """
        if height % 16 or width % 16:
            raise ValueError("height and width must be multiples of 16 (VAE 8x spatial compression x patch size 2)")
        import torch
        from diffusers import AutoencoderKLWan, WanPipeline
        from huggingface_hub import snapshot_download

        self.height, self.width = height, width
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.cfg_batched = False
        self.step_callbacks: list = []
        self._device = torch.device(device)

        saved = torch.load(Path(embeddings_path), map_location="cpu")
        self._prompt_embeds: dict[str, Any] = saved["prompts"]
        self._negative_embeds = saved["negative"]

        path = str(model_path) if model_path else snapshot_download(
            REPO_ID, allow_patterns=["model_index.json", "scheduler/*", "transformer/*", "vae/*"]
        )
        # The VAE stays float32 (diffusers' recommendation for Wan); the transformer runs in bf16.
        vae = AutoencoderKLWan.from_pretrained(path, subfolder="vae", torch_dtype=torch.float32)
        self.pipeline = WanPipeline.from_pretrained(
            path, vae=vae, text_encoder=None, tokenizer=None, torch_dtype=torch.bfloat16
        ).to(self._device)
        self.transformer = self.pipeline.transformer
        self.pipeline.set_progress_bar_config(disable=True)

    @property
    def prompts(self) -> list[str]:
        """The prompts that have embeddings and can be generated."""
        return list(self._prompt_embeds)

    @property
    def mag_ratios(self) -> list[float] | None:
        """MagCache's per-step magnitude ratios for this model, size, step count and sampler, from the table bundled in
        `worldoptbench/data/wan_mag_ratios.json` (measured with `optimizations.diffusion_more.calibrate_mag_ratios`), or None
        when this setup was never calibrated. Read at call time, so it follows `fewer_steps` and a swapped scheduler."""
        import json

        path = Path(__file__).resolve().parent.parent / "data" / "wan_mag_ratios.json"
        if not path.exists():
            return None
        key = f"{self.height}x{self.width}_s{self.num_inference_steps}_{type(self.pipeline.scheduler).__name__}"
        return json.loads(path.read_text(encoding="utf-8")).get(key)

    def torch_module(self) -> Any:
        """The transformer (satisfies models.base.TorchBacked), so architecture-agnostic modules
        such as quantization can modify it."""
        return self.transformer

    def _embeddings(self, prompt: str) -> tuple[Any, Any]:
        if prompt not in self._prompt_embeds:
            raise KeyError(
                f"no embedding for prompt {prompt!r}. Encode it once with "
                f"`python scripts/encode_prompts.py --prompt \"...\"`. Available: {self.prompts}"
            )
        return (
            self._prompt_embeds[prompt].to(self._device),
            self._negative_embeds.to(self._device),
        )

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
            prompt: must be one of `self.prompts` (see the module docstring). Required.
            horizon: video length in seconds, rounded to the nearest valid frame count at 16 fps.
            seed (kwarg): seeds the initial noise, default 0. The same seed gives (nearly) the
                same video; compare optimizations on identical seeds.
            guidance_scale (kwarg): overrides the constructor's value for this call.
        """
        if prompt is None:
            raise ValueError("WanVideo needs a prompt (one of its encoded prompts)")
        if init_frame is not None or init_video is not None:
            raise NotImplementedError("WanVideo is text-to-video; frame/video conditioning isn't supported")
        if actions is not None:
            raise NotImplementedError("WanVideo isn't action-conditioned")
        import torch

        seed = int(kwargs.pop("seed", 0))
        guidance = float(kwargs.pop("guidance_scale", self.guidance_scale))
        positive, negative = self._embeddings(prompt)
        n_frames = frames_for_horizon(horizon)

        call_kwargs: dict[str, Any] = {}
        if self.step_callbacks:
            call_kwargs["callback_on_step_end"] = compose_callbacks(self.step_callbacks)
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
        frames = [np.ascontiguousarray((f * 255.0).round().clip(0, 255).astype(np.uint8)) for f in video]
        return Rollout(
            frames=frames,
            fps=float(FPS),
            metadata={
                "prompt": prompt, "horizon": horizon, "model": self.get_info().name, "seed": seed,
                "num_frames": n_frames, "num_inference_steps": self.num_inference_steps, "guidance_scale": guidance,
            },
        )

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            name="Wan2.1-T2V-1.3B",
            architecture="diffusion",
            param_count=int(sum(p.numel() for p in self.transformer.parameters())),
            supported_optimizations=[],
            supports_image_conditioning=False,
            supports_video_conditioning=False,
        )


class WanReferenceScenarios:
    """ScenarioFn for diffusion quality: the reference is the *unoptimized* model's video for
    the same prompt, seed and length, so "fidelity" means agreement with the baseline, which is
    how caching papers report quality (there is no physics scorer for Wan here).

    `baseline_frame` is the reference's first frame, so the physics score is a skill against
    "just repeat the first frame": 0 means no better than a frozen video, 1 means identical.
    That keeps it comparable in spirit to the Dreamer skill score and, like it, avoids a raw
    pixel score that sits near 1.0 for any video with a static background.

    References are generated by a separate unoptimized WanVideo on first use and cached on
    disk (keyed by prompt, seed, length and the sampler settings), so repeated sweeps and
    every optimized configuration compare against the same videos. Build all scenarios before
    timing anything (the runner does), since generating a reference is slow and uses the GPU.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        baseline_factory: Any = None,
        model_kwargs: dict[str, Any] | None = None,
    ):
        self._cache_dir = Path(cache_dir)
        self._model_kwargs = dict(model_kwargs or {})
        self._baseline_factory = baseline_factory or (lambda: WanVideo(**self._model_kwargs))
        self._baseline: Any = None

    def _settings_tag(self) -> str:
        kw = self._model_kwargs
        return "{}x{}_s{}_g{}".format(
            kw.get("height", 192), kw.get("width", 320), kw.get("num_inference_steps", 30), kw.get("guidance_scale", 5.0)
        )

    def _path(self, prompt: str, seed: int, horizon: float) -> Path:
        import hashlib

        digest = hashlib.sha1(prompt.encode()).hexdigest()[:10]
        return self._cache_dir / f"{digest}_seed{seed}_h{horizon:g}_{self._settings_tag()}.npz"

    def __call__(self, entry: dict[str, Any], horizon: float):
        from worldoptbench.scenarios import Scenario

        prompt, seed = entry["prompt"], int(entry.get("seed", 0))
        path = self._path(prompt, seed, horizon)
        if path.exists():
            with np.load(path) as saved:
                frames = list(saved["frames"])
        else:
            if self._baseline is None:
                self._baseline = self._baseline_factory()
            frames = self._baseline.generate(prompt=prompt, horizon=horizon, seed=seed).frames
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, frames=np.stack(frames))
        return Scenario(
            generate_kwargs={"seed": seed},
            reference_frames=frames,
            baseline_frame=frames[0],
        )
