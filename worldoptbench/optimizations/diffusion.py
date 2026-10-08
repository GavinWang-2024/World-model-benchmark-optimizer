"""Diffusion optimizations: thin wrappers over what diffusers already implements,
plus the two step-level ones it doesn't (CFG truncation, fewer steps).

Scope note: diffusers 0.40 ships the caching techniques (first-block / TeaCache-style,
Pyramid Attention Broadcast, TaylorSeer, FasterCache), layer skipping, attention
backends and fp8 weight storage, and its transformers (including Wan's) expose them
through `enable_cache`, `apply_layer_skip`, `set_attention_backend` and
`enable_layerwise_casting`. This project's value here is not reimplementing them but
putting them behind one contract, so they can be stacked, skipped when they don't
apply, measured with PAES, and chosen by `autotune`.

What a diffusion model wrapper must expose (opt-in attributes, like the Dreamer hooks):
    transformer       the diffusers transformer (a CacheMixin model)
    pipeline          the diffusers pipeline (for `current_timestep`, `num_timesteps`, guidance)
    step_callbacks    a list of `callback_on_step_end`-style callables the wrapper
                      composes into every pipeline call
    num_inference_steps   int the wrapper passes to the pipeline
    cfg_batched       bool: True if classifier-free guidance runs the conditional and
                      unconditional passes as ONE batch, False if as two calls (Wan)

Every module here starts `experimental`: each was probed on a tiny random-weight Wan
transformer on CPU (see tests/test_diffusion_modules.py), which proves the mechanics
and the module contract, not speed or quality on the real model.

Caches share one slot: diffusers allows a single cache technique on a transformer at
a time (`enable_cache` raises otherwise), so they declare `exclusive_group =
"diffusion_cache"` and the stack keeps only the first.
"""

from __future__ import annotations

from typing import Any

from worldoptbench.models.base import Architecture, WorldModelInterface
from worldoptbench.optimizations.base import OptimizationModule, register_module

_ARCH: tuple[Architecture, ...] = ("diffusion",)
CACHE_GROUP = "diffusion_cache"


def refresh_hook_registries(transformer: Any) -> None:
    """Make diffusers' hook registries re-discover their child registries.

    `HookRegistry._get_child_registries` caches its list the first time `cache_context(...)` runs and
    never invalidates it. A cache applied (or removed) after the transformer has already run once
    is therefore invisible to later `cache_context` calls: its state never gets a context and the
    first forward raises "No context is set". Call this after adding or removing hooks."""
    for module in transformer.modules():
        registry = getattr(module, "_diffusers_hook", None)
        if registry is not None and hasattr(registry, "_child_registries_cache"):
            registry._child_registries_cache = None


class _TransformerModule(OptimizationModule):
    """Shared plumbing: needs the model's diffusers transformer."""

    supported_architectures = _ARCH
    requires = ("transformer",)
    needs_packages = ("diffusers",)
    maturity = "experimental"

    def _transformer(self, model: WorldModelInterface) -> Any:
        transformer = getattr(model, "transformer", None)
        if transformer is None:
            raise TypeError(f"{type(model).__name__} has no `transformer` to optimize")
        return transformer


class _CacheModule(_TransformerModule):
    """A technique applied through `transformer.enable_cache(config)`."""

    exclusive_group = CACHE_GROUP

    def _config(self, model: WorldModelInterface) -> Any:
        raise NotImplementedError

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        self._applied_to = self._transformer(model)
        self._applied_to.enable_cache(self._config(model))
        refresh_hook_registries(self._applied_to)
        return model

    def restore(self) -> None:
        """Turn the cache off again (idempotent)."""
        transformer = getattr(self, "_applied_to", None)
        if transformer is not None and transformer.is_cache_enabled:
            transformer.disable_cache()
            refresh_hook_registries(transformer)


_M4 = "Measured on Wan2.1-1.3B (4 prompts, 33 frames, 192x320, 30 steps; fidelity = agreement with the unoptimized video): "
_M32 = "Measured on Wan2.1-1.3B (16 prompts x 2 seeds = 32 clips, 33 frames, 192x320, 30 steps; PSNR is agreement with the unoptimized video, mean +- sem): "

def _timestep_callback(model: WorldModelInterface):
    pipeline = getattr(model, "pipeline", None)
    if pipeline is None:
        raise TypeError(f"{type(model).__name__} has no `pipeline`, which this cache needs for the current timestep")
    return lambda: pipeline.current_timestep


@register_module
class FirstBlockCacheModule(_CacheModule):
    """Skip the rest of the transformer when the first block's output barely changed
    since the previous step (TeaCache-style). One knob: `threshold` — larger skips more
    and drifts more. At 0 it never skips (identical to no cache)."""

    name = "first_block_cache"
    maturity = "measured"
    summary = (
        "First-block (TeaCache-style) feature cache via diffusers. " + _M32 + "threshold 0.05: 1.23x, PSNR 21.5 +- 0.8; 0.1: 1.72x, PSNR 14.6 +- 0.8 (0.2: 2.62x, PSNR ~10 on 4 clips). 0.05 is statistically indistinguishable from a numerical perturbation (PSNR at the 21.5 dB floor, blind-statistics deviation 0.089 vs 0.081); 0.1 shifts the video's statistics clearly (sharpness -23%, flicker +27%). WorldCache is clearly better at ~1.7x."
    )

    def __init__(self, threshold: float = 0.05):
        if threshold < 0:
            raise ValueError("threshold must be >= 0")
        self.threshold = threshold

    @property
    def label(self) -> str:
        return f"{self.name}_{self.threshold:g}"

    def _config(self, model):
        from diffusers.hooks import FirstBlockCacheConfig

        return FirstBlockCacheConfig(threshold=self.threshold)


@register_module
class PyramidAttentionBroadcastModule(_CacheModule):
    """Reuse attention outputs across steps within a timestep range (Pyramid Attention
    Broadcast). `block_skip_range=k` recomputes self-attention every k-th step and reuses
    it in between."""

    name = "pab"
    requires = ("transformer", "pipeline")
    maturity = "measured"
    summary = "Pyramid Attention Broadcast via diffusers. " + _M32 + "skip range 2: 1.07x, PSNR 26.6 +- 0.6. Small speedup, good fidelity; adds ~5% on top of cfg_truncation at no measurable fidelity cost (cfg 0.4 + pab 2: 1.43x vs 1.36x, PSNR 26.3 vs 25.9)."

    def __init__(self, block_skip_range: int = 2, timestep_skip_range: tuple[int, int] = (100, 800)):
        if block_skip_range < 2:
            raise ValueError("block_skip_range must be >= 2 (1 would recompute every step)")
        self.block_skip_range = block_skip_range
        self.timestep_skip_range = tuple(timestep_skip_range)

    @property
    def label(self) -> str:
        return f"{self.name}_{self.block_skip_range}"

    def _config(self, model):
        from diffusers.hooks import PyramidAttentionBroadcastConfig

        return PyramidAttentionBroadcastConfig(
            spatial_attention_block_skip_range=self.block_skip_range,
            spatial_attention_timestep_skip_range=self.timestep_skip_range,
            current_timestep_callback=_timestep_callback(model),
        )


@register_module
class TaylorSeerCacheModule(_CacheModule):
    """Forecast skipped features with a Taylor expansion instead of reusing them stale."""

    name = "taylorseer"
    maturity = "measured"
    summary = "TaylorSeer feature-forecasting cache via diffusers. " + _M32 + "interval 3: 1.34x, PSNR 14.5 +- 0.6 (interval 5 on 4 clips: 1.43x, PSNR ~11). Dominated by cfg_truncation 0.4 (1.36x, PSNR 25.9); a real degradation signature (flicker x1.9, motion +20% vs the baseline video)."

    def __init__(
        self,
        cache_interval: int = 5,
        disable_cache_before_step: int = 3,
        max_order: int = 1,
        cache_identifiers: list[str] | None = None,
    ):
        """`cache_identifiers`: regexes (full-matched against module names) for the modules to
        forecast. Default None: diffusers' patterns if any module matches them, else every
        self/cross-attention block (`blocks.N.attnK`). diffusers' own defaults end in `attn`, so they
        match nothing on Wan (`attn1`/`attn2`) and the cache silently did nothing."""
        if cache_interval < 2:
            raise ValueError("cache_interval must be >= 2")
        self.cache_interval = cache_interval
        self.disable_cache_before_step = disable_cache_before_step
        self.max_order = max_order
        self.cache_identifiers = list(cache_identifiers) if cache_identifiers is not None else None

    @property
    def label(self) -> str:
        return f"{self.name}_{self.cache_interval}"

    def _config(self, model):
        import re

        from diffusers.hooks import TaylorSeerCacheConfig
        from diffusers.hooks.taylorseer_cache import _TRANSFORMER_BLOCK_IDENTIFIERS

        identifiers = self.cache_identifiers
        if identifiers is None:
            names = [n for n, _ in self._transformer(model).named_modules()]
            if not any(re.fullmatch(p, n) for p in _TRANSFORMER_BLOCK_IDENTIFIERS for n in names):
                identifiers = [r"^[A-Za-z_]*blocks\.\d+\.attn\d*"]
                if not any(re.fullmatch(identifiers[0], n) for n in names):
                    raise ValueError("taylorseer found no attention blocks to cache; pass cache_identifiers=[...]")
        return TaylorSeerCacheConfig(
            cache_interval=self.cache_interval,
            disable_cache_before_step=self.disable_cache_before_step,
            max_order=self.max_order,
            cache_identifiers=identifiers,
        )


@register_module
class FasterCacheModule(_CacheModule):
    """FasterCache via diffusers (attention reuse plus skipping the unconditional pass).

    Declares itself incompatible with pipelines that run classifier-free guidance as two
    separate transformer calls: probing a Wan transformer showed it works on a batch of 2
    and fails with an IndexError on a batch of 1, i.e. it expects the conditional and
    unconditional inputs concatenated. Wan's pipeline calls them separately, so it is
    skipped there rather than crashing mid-generation. diffusers itself labels
    FasterCache "purely experimental".
    """

    name = "fastercache"
    requires = ("transformer", "pipeline")
    summary = "FasterCache via diffusers; needs batched CFG, so unavailable on Wan. Unmeasured."

    def __init__(self, block_skip_range: int = 2, unconditional_skip_range: int = 5):
        self.block_skip_range = block_skip_range
        self.unconditional_skip_range = unconditional_skip_range

    @property
    def label(self) -> str:
        return f"{self.name}_{self.block_skip_range}"

    def incompatibility(self, model: WorldModelInterface) -> str | None:
        reason = super().incompatibility(model)
        if reason is None and getattr(model, "cfg_batched", True) is False:
            return (
                f"{self.name} needs the conditional and unconditional passes batched together, but "
                f"{type(model).__name__} runs them as separate calls"
            )
        return reason

    def _config(self, model):
        from diffusers.hooks import FasterCacheConfig

        return FasterCacheConfig(
            spatial_attention_block_skip_range=self.block_skip_range,
            unconditional_batch_skip_range=self.unconditional_skip_range,
            current_timestep_callback=_timestep_callback(model),
        )


@register_module
class LayerSkipModule(_TransformerModule):
    """Always skip the attention and feed-forward of some transformer blocks. A fixed, lossy
    shortcut (not input-dependent like the caches). Not reversible here: diffusers' layer-skip
    hooks have no documented undo, so apply it to a fresh model.

    By default it skips `num_blocks` blocks spread evenly through the *middle* of the stack
    (the first and last blocks matter most), chosen at apply time from the transformer's
    depth; pass explicit `indices` to override.
    """

    name = "layer_skip"
    exclusive_group = "diffusion_layer_skip"
    maturity = "measured"
    summary = "Skip fixed transformer blocks via diffusers. " + _M4 + "2 blocks: 1.06x, 4 blocks: 1.14x, PSNR ~10. Wrecks quality for little gain."

    def __init__(self, num_blocks: int = 2, indices: list[int] | None = None):
        if not isinstance(num_blocks, int):
            raise TypeError("num_blocks must be an int; pass a list of block indices as indices=[...]")
        if indices is not None and not indices:
            raise ValueError("indices must name at least one block to skip")
        if indices is None and num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.num_blocks = num_blocks
        self.indices = list(indices) if indices is not None else None

    @property
    def label(self) -> str:
        return f"{self.name}_{len(self.indices) if self.indices is not None else self.num_blocks}"

    @staticmethod
    def middle_indices(depth: int, count: int) -> list[int]:
        """`count` indices spread evenly over the middle half of `depth` blocks."""
        low, high = depth // 4, (3 * depth) // 4
        if count > high - low + 1:
            raise ValueError(f"cannot skip {count} blocks from the middle of a {depth}-block transformer")
        if count == 1:
            return [(low + high) // 2]
        return sorted({low + round(i * (high - low) / (count - 1)) for i in range(count)})

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        from diffusers.hooks import LayerSkipConfig, apply_layer_skip

        transformer = self._transformer(model)
        blocks = getattr(transformer, "blocks", None) or transformer.transformer_blocks  # Wan names them `blocks`, Cosmos `transformer_blocks`
        indices = self.indices if self.indices is not None else self.middle_indices(len(blocks), self.num_blocks)
        apply_layer_skip(transformer, LayerSkipConfig(indices=indices))
        self.applied_indices = indices
        return model


@register_module
class AttentionBackendModule(_TransformerModule):
    """Switch the attention kernel (`native`, `flex`, cuDNN, flash, sage, xformers).
    Availability varies by machine: `flash`/`sage` need packages that may not build on
    Windows, and the cuDNN/efficient SDPA backends need a CUDA device. A backend that
    isn't usable raises when first applied, with diffusers' own message."""

    name = "attention_backend"
    maturity = "measured"
    summary = "Select the attention kernel via diffusers. " + _M4 + "native/efficient identical, cuDNN 1.10x with a different output (PSNR 15: a numerically different kernel is enough to reach the metric's noise floor, not a bug), flex 0.34x, flash unavailable here. No clear win on this machine."

    def __init__(self, backend: str = "native"):
        self.backend = backend

    @property
    def label(self) -> str:
        return f"attention_{self.backend}"

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        from diffusers.models.attention_dispatch import AttentionBackendName

        valid = [b.value for b in AttentionBackendName]
        if self.backend not in valid:
            raise ValueError(f"unknown attention backend {self.backend!r}; diffusers offers {valid}")
        self._applied_to = self._transformer(model)
        self._applied_to.set_attention_backend(self.backend)
        return model

    def restore(self) -> None:
        transformer = getattr(self, "_applied_to", None)
        if transformer is not None:
            transformer.reset_attention_backend()


@register_module
class LayerwiseCastingModule(_TransformerModule):
    """Store weights in fp8 and upcast layer by layer for compute (diffusers'
    layer-wise casting). Cuts weight memory roughly in half versus bf16 at a small
    numerical cost; it does not make the arithmetic faster."""

    name = "layerwise_casting"
    maturity = "measured"
    summary = "fp8 weight storage with per-layer upcast via diffusers. " + _M4 + "0.91x (slower), peak memory 4.4 -> 3.1 GB. A memory option, not a speed one."
    _DTYPES = ("float8_e4m3fn", "float8_e5m2")

    def __init__(self, storage_dtype: str = "float8_e4m3fn", compute_dtype: str | None = None):
        """`compute_dtype=None` (the default) upcasts to the transformer's own dtype, which is
        what its activations use; forcing a different one gives a dtype mismatch in the matmuls
        (found when a float32 model was given bfloat16)."""
        if storage_dtype not in self._DTYPES:
            raise ValueError(f"storage_dtype must be one of {self._DTYPES}")
        self.storage_dtype = storage_dtype
        self.compute_dtype = compute_dtype

    @property
    def label(self) -> str:
        return f"{self.name}_{self.storage_dtype}"

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        import torch

        transformer = self._transformer(model)
        # `transformer.dtype` (diffusers' own logic) skips the fp32 modules a model keeps on purpose;
        # `next(parameters())` picked up Wan's fp32 `scale_shift_table` and set a float32 compute dtype.
        compute = (
            getattr(torch, self.compute_dtype)
            if self.compute_dtype is not None
            else getattr(transformer, "dtype", None) or next(transformer.parameters()).dtype
        )
        transformer.enable_layerwise_casting(storage_dtype=getattr(torch, self.storage_dtype), compute_dtype=compute)
        return model


@register_module
class CfgTruncationModule(OptimizationModule):
    """Turn classifier-free guidance off for the last part of denoising.

    Each CFG step runs the transformer twice (conditional + unconditional); late steps
    change the image little, so dropping the unconditional pass there roughly halves their
    cost. `after_fraction=0.6` keeps guidance for the first 60% of steps. Implemented as a
    `callback_on_step_end` that sets the pipeline's guidance scale to 1.0 (which makes its
    `do_classifier_free_guidance` false); the pipeline resets it at the start of every call.
    """

    name = "cfg_truncation"
    supported_architectures = _ARCH
    requires = ("pipeline", "step_callbacks")
    maturity = "measured"
    summary = "Drop classifier-free guidance after a fraction of steps. " + _M32 + "after 60%: 1.21x, PSNR 30.8 +- 0.6, SSIM 0.965; after 40%: 1.36x, PSNR 25.9 +- 0.6. Best fidelity per speed at <= 1.4x; composes with caches (with worldcache 0.04: 1.98x vs 1.65x alone at the same PSNR)."

    def __init__(self, after_fraction: float = 0.6):
        if not 0.0 < after_fraction < 1.0:
            raise ValueError("after_fraction must be in (0, 1)")
        self.after_fraction = after_fraction

    @property
    def label(self) -> str:
        return f"{self.name}_{self.after_fraction:g}"

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        fraction = self.after_fraction

        def truncate_guidance(pipe: Any, step: int, timestep: Any, callback_kwargs: dict) -> dict:
            if step + 1 >= int(fraction * pipe.num_timesteps):
                pipe._guidance_scale = 1.0  # CFG off from the next step on
            return callback_kwargs

        model.step_callbacks.append(truncate_guidance)
        return model


@register_module
class FewerStepsModule(OptimizationModule):
    """Run fewer denoising steps. The simplest step reduction: quality falls as steps drop,
    and how fast depends on the sampler (the pipeline's default scheduler is kept)."""

    name = "fewer_steps"
    supported_architectures = _ARCH
    requires = ("num_inference_steps",)
    maturity = "measured"
    summary = "Fewer denoising steps with the default sampler. " + _M32 + "20 steps: 1.43x, PSNR 13.3 +- 0.7 (15 steps on 4 clips: 1.83x, PSNR ~10). Reference-free check: sharpness and noise -17% and flicker +38% on average vs the baseline, beyond a numerical-perturbation control; on the one contact sheet viewed it is a coherent, differently composed video, not an obviously blurrier one."

    def __init__(self, steps: int = 25):
        if steps < 1:
            raise ValueError("steps must be >= 1")
        self.steps = steps

    @property
    def label(self) -> str:
        return f"{self.name}_{self.steps}"

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        model.num_inference_steps = self.steps
        return model
