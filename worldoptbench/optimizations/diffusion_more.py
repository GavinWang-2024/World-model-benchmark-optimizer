"""More diffusion optimizations for the Wan pipeline (catalog D1, D2, D4, E10, H1, H4).

    scheduler            swap the sampler (UniPC / DPM-Solver++ / Euler) and tune its timestep shift and order  (D1, D2)
    uncond_reuse         run the unconditional guidance pass only every `period` steps, reusing it in between      (D4)
    cross_attn_kv_cache  compute the text keys and values of each cross-attention once per video instead of per step (E10)
    vae                  VAE tiling, slicing, and a lower-precision decode                                          (H1, H4)

All of them assume the diffusers WanPipeline shape (`pipeline.scheduler`, `pipeline.vae`, a `transformer` called once per
guidance branch inside `cache_context("cond")` / `cache_context("uncond")`), the same assumptions as the other modules in
diffusion.py. Their measured effect on Wan2.1-1.3B is in each module's `summary` and in DESIGN_DIFFUSION.md.
"""

from __future__ import annotations

from typing import Any

from worldoptbench.models.base import Architecture, WorldModelInterface
from worldoptbench.optimizations.base import OptimizationModule, register_module
from worldoptbench.optimizations.diffusion import _TransformerModule, refresh_hook_registries

_ARCH: tuple[Architecture, ...] = ("diffusion",)
_M32 = (
    "Measured on Wan2.1-1.3B (16 prompts x 2 seeds = 32 clips, 33 frames, 192x320, 30 steps; DINOv2 similarity to the unoptimized "
    "video, noise controls 0.88): "
)

# ---- scheduler -----------------------------------------------------------------------------------------------

_SCHEDULERS = {
    "unipc": "UniPCMultistepScheduler",
    "dpmpp": "DPMSolverMultistepScheduler",
    "euler": "FlowMatchEulerDiscreteScheduler",
}


@register_module
class SchedulerModule(OptimizationModule):
    """Replace the pipeline's sampler, keeping its flow-matching setup, and optionally change the timestep `flow_shift`
    (how steps are spread between noisy and clean) or the solver order. Wan's default is UniPC with order 2 and shift 3,
    which is already a strong solver, so this mostly matters in combination with fewer steps (`fewer_steps`)."""

    name = "scheduler"
    supported_architectures = _ARCH
    requires = ("pipeline",)
    needs_packages = ("diffusers",)
    maturity = "measured"
    summary = (
        "Swap the sampler and tune flow shift / solver order. " + _M32 + "at 20 steps (1.43x): default UniPC 0.633, DPM-Solver++ 0.644, "
        "Euler 0.639, UniPC order 3 0.643, so the solver makes no difference; the timestep shift does: shift 5 gives 0.764 (prompt match "
        "unchanged), shift 8 0.650, shift 1.5 0.554; at 15 steps shift 5 gives 0.688. Still well below CFG truncation + PAB at the same "
        "speed (0.930)."
    )

    def __init__(self, kind: str = "unipc", flow_shift: float | None = None, solver_order: int | None = None):
        if kind not in _SCHEDULERS:
            raise ValueError(f"kind must be one of {sorted(_SCHEDULERS)}")
        if flow_shift is not None and flow_shift <= 0:
            raise ValueError("flow_shift must be positive")
        if solver_order is not None and solver_order not in (1, 2, 3):
            raise ValueError("solver_order must be 1, 2 or 3")
        self.kind, self.flow_shift, self.solver_order = kind, flow_shift, solver_order
        self._original: Any = None
        self._pipeline: Any = None

    @property
    def label(self) -> str:
        parts = [self.name, self.kind]
        if self.flow_shift is not None:
            parts.append(f"shift{self.flow_shift:g}")
        if self.solver_order is not None:
            parts.append(f"order{self.solver_order}")
        return "_".join(parts)

    def incompatibility(self, model: WorldModelInterface) -> str | None:
        reason = super().incompatibility(model)
        if reason is None and not hasattr(model.pipeline, "scheduler"):
            return "the pipeline has no `scheduler` to swap"
        return reason

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        import diffusers  # noqa: PLC0415

        pipeline = model.pipeline
        config = dict(pipeline.scheduler.config)
        shift = self.flow_shift if self.flow_shift is not None else config.get("flow_shift", config.get("shift", 3.0))
        if self.kind == "euler":
            config["shift"] = shift
        else:
            config["flow_shift"] = shift
            config["use_flow_sigmas"] = True
            config["prediction_type"] = "flow_prediction"
            if self.solver_order is not None:
                config["solver_order"] = self.solver_order
            if self.kind == "dpmpp":
                config["algorithm_type"] = "dpmsolver++"
        self._pipeline, self._original = pipeline, pipeline.scheduler
        pipeline.scheduler = getattr(diffusers, _SCHEDULERS[self.kind]).from_config(config)
        return model

    def restore(self) -> None:
        if self._pipeline is not None:
            self._pipeline.scheduler = self._original


# ---- unconditional-branch reuse -----------------------------------------------------------------------------------


def _build_uncond_hook():
    from diffusers.hooks.hooks import BaseState, ModelHook  # noqa: PLC0415

    class State(BaseState):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0
            self.cached: Any = None

        def reset(self):
            self.__init__()

    class UncondReuseHook(ModelHook):
        _is_stateful = True

        def __init__(self, state_manager, period: int, target_context: str):
            super().__init__()
            self.state_manager, self.period, self.target = state_manager, period, target_context

        def new_forward(self, module, *args, **kwargs):
            if self.state_manager._current_context != self.target:
                return self.fn_ref.original_forward(*args, **kwargs)  # the conditional branch always runs
            state = self.state_manager.get_state()
            index = state.calls
            state.calls += 1
            if state.cached is not None and index % self.period != 0:
                return state.cached  # reuse the unconditional prediction of an earlier step
            state.cached = self.fn_ref.original_forward(*args, **kwargs)
            return state.cached

        def reset_state(self, module):
            self.state_manager.reset()
            return module

    return State, UncondReuseHook


@register_module
class UncondReuseModule(_TransformerModule):
    """Classifier-free guidance runs the transformer twice per step (conditional and unconditional). The unconditional
    prediction changes slowly across steps, so recompute it only every `period` steps and reuse the last one in between:
    with period 2 the unconditional half of every other step is skipped (up to about 25% less compute). Complements
    `cfg_truncation`, which drops the unconditional pass for the last steps instead of thinning it throughout."""

    name = "uncond_reuse"
    requires = ("transformer", "pipeline")
    maturity = "measured"
    summary = (
        "Recompute the unconditional guidance pass only every `period` steps. " + _M32 + "HARMFUL: period 2 gives 1.29x but DINO similarity "
        "0.36 (31 of 32 clips beyond the noise floor, prompt match -0.08); period 3 gives 1.43x at 0.22. The unconditional prediction does "
        "not change slowly enough to reuse; cfg_truncation is the safe way to thin it."
    )

    _HOOK = "uncond_reuse_hook"

    def __init__(self, period: int = 2, context: str = "uncond"):
        if period < 2:
            raise ValueError("period must be >= 2 (1 would recompute every step)")
        self.period, self.context = period, context
        self._applied_to: Any = None

    @property
    def label(self) -> str:
        return f"{self.name}_{self.period}"

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        from diffusers.hooks import HookRegistry  # noqa: PLC0415
        from diffusers.hooks.hooks import StateManager  # noqa: PLC0415

        transformer = self._transformer(model)
        if not hasattr(transformer, "cache_context"):
            raise TypeError("uncond_reuse needs a diffusers transformer with cache_context (a CacheMixin model)")
        state_cls, hook_cls = _build_uncond_hook()
        HookRegistry.check_if_exists_or_initialize(transformer).register_hook(
            hook_cls(StateManager(state_cls, (), {}), self.period, self.context), self._HOOK
        )
        refresh_hook_registries(transformer)
        self._applied_to = transformer
        return model

    def restore(self) -> None:
        if self._applied_to is not None and hasattr(self._applied_to, "_diffusers_hook"):
            try:
                self._applied_to._diffusers_hook.remove_hook(self._HOOK, recurse=False)
            except KeyError:
                pass
            refresh_hook_registries(self._applied_to)
        self._applied_to = None


# ---- cross-attention key/value cache ------------------------------------------------------------------------------


def _build_kv_hook():
    from diffusers.hooks.hooks import BaseState, ModelHook  # noqa: PLC0415

    class State(BaseState):
        def __init__(self) -> None:
            super().__init__()
            self.value: Any = None

        def reset(self):
            self.__init__()

    class KVCacheHook(ModelHook):
        _is_stateful = True

        def __init__(self, state_manager):
            super().__init__()
            self.state_manager = state_manager

        def new_forward(self, module, *args, **kwargs):
            state = self.state_manager.get_state()
            inputs = args[0] if args else next(iter(kwargs.values()))
            if state.value is None or state.value.shape != (*inputs.shape[:-1], state.value.shape[-1]):
                state.value = self.fn_ref.original_forward(*args, **kwargs)
            return state.value

        def reset_state(self, module):
            self.state_manager.reset()
            return module

    return State, KVCacheHook


@register_module
class CrossAttentionKVCacheModule(_TransformerModule):
    """The text conditioning of a video does not change between denoising steps, but every step recomputes the key and value
    projections of each cross-attention from it. This computes them once per guidance branch and reuses them, which is
    exact (the same numbers, not an approximation). It assumes the text embedding is constant within one pipeline call;
    the diffusers pipeline resets the cached state when a call ends, so the next video recomputes."""

    name = "cross_attn_kv_cache"
    requires = ("transformer",)
    maturity = "measured"
    summary = (
        "Compute the text keys/values of every cross-attention once per video. " + _M32 + "exact (output identical, DINO similarity 1.000) "
        "but only 1.02x (about 2% of the compute), at +0.2 GB of cache. Free, small."
    )

    _HOOK = "cross_attn_kv_cache_hook"
    _PATTERN = r"^blocks\.\d+\.attn2\.(to_k|to_v)$"

    def __init__(self) -> None:
        self._hooked: list = []
        self._applied_to: Any = None

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        import re  # noqa: PLC0415

        from diffusers.hooks import HookRegistry  # noqa: PLC0415
        from diffusers.hooks.hooks import StateManager  # noqa: PLC0415

        transformer = self._transformer(model)
        if not hasattr(transformer, "cache_context"):
            raise TypeError("cross_attn_kv_cache needs a diffusers transformer with cache_context")
        targets = [m for name, m in transformer.named_modules() if re.fullmatch(self._PATTERN, name)]
        if not targets:
            raise ValueError("cross_attn_kv_cache found no `blocks.N.attn2.to_k` / `to_v` projections")
        state_cls, hook_cls = _build_kv_hook()
        for module in targets:
            HookRegistry.check_if_exists_or_initialize(module).register_hook(hook_cls(StateManager(state_cls, (), {})), self._HOOK)
            self._hooked.append(module)
        refresh_hook_registries(transformer)
        self._applied_to = transformer
        return model

    def restore(self) -> None:
        for module in self._hooked:
            try:
                module._diffusers_hook.remove_hook(self._HOOK, recurse=False)
            except (AttributeError, KeyError):
                pass
        self._hooked = []
        if self._applied_to is not None:
            refresh_hook_registries(self._applied_to)
        self._applied_to = None


# ---- VAE -------------------------------------------------------------------------------------------------------------


@register_module
class VaeModule(OptimizationModule):
    """VAE options: `tiling` and `slicing` decode in pieces (less memory, usually a little slower), `dtype="bfloat16"` runs the
    decode in lower precision (the pipelines keep Wan's VAE in float32 by default, as its authors recommend)."""

    name = "vae"
    supported_architectures = _ARCH
    requires = ("pipeline",)
    needs_packages = ("diffusers",)
    maturity = "measured"
    summary = (
        "VAE tiling / slicing / lower-precision decode. " + _M32 + "dtype bfloat16: 1.04x, DINO similarity 0.999 (PSNR ~51 dB), peak GPU "
        "memory 3.58 GB against 4.40 GB; tiling: 0.98x (slower), peak 4.15 GB. The bf16 VAE is a near-free win; tiling is for memory only."
    )

    def __init__(self, tiling: bool = True, slicing: bool = False, dtype: str | None = None):
        if dtype is not None and dtype not in ("bfloat16", "float16"):
            raise ValueError("dtype must be None, 'bfloat16' or 'float16'")
        if not (tiling or slicing or dtype):
            raise ValueError("turn on at least one of tiling, slicing, dtype")
        self.tiling, self.slicing, self.dtype = tiling, slicing, dtype
        self._vae: Any = None
        self._original_dtype: Any = None

    @property
    def label(self) -> str:
        parts = [self.name] + (["tiling"] if self.tiling else []) + (["slicing"] if self.slicing else []) + ([self.dtype] if self.dtype else [])
        return "_".join(parts)

    def incompatibility(self, model: WorldModelInterface) -> str | None:
        reason = super().incompatibility(model)
        if reason is None and not hasattr(model.pipeline, "vae"):
            return "the pipeline has no `vae`"
        return reason

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        import torch  # noqa: PLC0415

        vae = model.pipeline.vae
        self._vae = vae
        self._original_dtype = next(vae.parameters()).dtype
        if self.tiling:
            vae.enable_tiling()
        if self.slicing:
            vae.enable_slicing()
        if self.dtype:
            vae.to(getattr(torch, self.dtype))
        return model

    def restore(self) -> None:
        if self._vae is None:
            return
        if self.tiling:
            self._vae.disable_tiling()
        if self.slicing:
            self._vae.disable_slicing()
        if self.dtype:
            self._vae.to(self._original_dtype)


# ---- MagCache (diffusers) -----------------------------------------------------------------------------------------


def calibrate_mag_ratios(model: Any, run: Any, num_inference_steps: int | None = None) -> list[float]:
    """The per-step magnitude ratios MagCache needs, measured on this model. diffusers' calibration mode skips nothing and
    prints the ratios at the end of each classifier-free-guidance context; this runs `run()` (a callable that does one
    full generation on `model`) with calibration switched on, captures that output, and returns the conditional branch's
    list (the first printed). The ratios are checkpoint- and scheduler-dependent: calibrate with the sampler and step
    count you will use."""
    import contextlib  # noqa: PLC0415
    import io  # noqa: PLC0415
    import re  # noqa: PLC0415

    from diffusers.hooks import MagCacheConfig  # noqa: PLC0415

    transformer = model.transformer
    steps = int(num_inference_steps or model.num_inference_steps)
    transformer.enable_cache(MagCacheConfig(num_inference_steps=steps, calibrate=True))
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            run()
    finally:
        transformer.disable_cache()
        refresh_hook_registries(transformer)
    lists = re.findall(r"\[MagCache\] Calibration Complete[^\n]*\n\[([^\]]+)\]", buffer.getvalue())
    if not lists:
        raise RuntimeError("MagCache calibration printed no ratios; run() must perform a full generation through the transformer")
    return [float(x) for x in lists[0].split(",")]


from worldoptbench.optimizations.diffusion import _CacheModule  # noqa: E402


@register_module
class MagCacheModule(_CacheModule):
    """MagCache (diffusers): skip transformer blocks on steps where the residual's magnitude is predicted, from per-step ratios
    measured once on the model (`calibrate_mag_ratios`), to have changed little; `threshold` bounds the accumulated predicted
    error, `max_skip_steps` the consecutive skips, `retention_ratio` the fraction of early steps that always compute.
    `mag_ratios` are model-dependent: with none given the module is not usable on a model until it has been calibrated."""

    name = "magcache"
    requires = ("transformer", "num_inference_steps")
    maturity = "measured"
    summary = (
        "MagCache (diffusers) with ratios calibrated on the model. " + _M32 + "the best cache tried: threshold 0.06 gives 1.49x at DINO 0.885; "
        "0.24 gives 2.03x at 0.855; both statistically the same as the noise control (0.876), 2-4 of 32 clips below the floor, prompt match "
        "unchanged. Thresholds above 0.24 add nothing because max_skip_steps (3) and retention_ratio (0.2) bound the skipping; "
        "max_skip_steps 5 gives 2.15x at 0.833; retention_ratio 0.1 collapses it (0.646): the early steps must be computed. With "
        "cfg_truncation 0.6 + cross_attn_kv_cache + a bf16 VAE: 2.54x at 0.856, 2.63x at 0.847 (threshold 0.5). Needs per-setup ratios."
    )

    def __init__(self, threshold: float = 0.06, max_skip_steps: int = 3, retention_ratio: float = 0.2,
                 mag_ratios: list[float] | None = None):
        if threshold < 0 or max_skip_steps < 1 or not 0.0 <= retention_ratio < 1.0:
            raise ValueError("threshold must be >= 0, max_skip_steps >= 1 and retention_ratio in [0, 1)")
        self.threshold, self.max_skip_steps, self.retention_ratio = threshold, max_skip_steps, retention_ratio
        self.mag_ratios = list(mag_ratios) if mag_ratios is not None else None

    @property
    def label(self) -> str:
        return f"{self.name}_{self.threshold:g}"

    def _ratios(self, model) -> list[float] | None:
        """The ratios given at construction, else the model's own (`model.mag_ratios`, for a setup it was calibrated on)."""
        return self.mag_ratios if self.mag_ratios is not None else getattr(model, "mag_ratios", None)

    def incompatibility(self, model):
        reason = super().incompatibility(model)
        if reason is None and self._ratios(model) is None:
            return (
                "magcache needs mag_ratios measured on this model, size, step count and sampler (calibrate_mag_ratios); "
                "none were given and the model has none for this setup"
            )
        return reason

    def _config(self, model):
        from diffusers.hooks import MagCacheConfig  # noqa: PLC0415

        ratios = self._ratios(model)
        if ratios is None:
            raise ValueError("magcache needs mag_ratios for this model: measure them with calibrate_mag_ratios(model, run)")
        return MagCacheConfig(
            threshold=self.threshold, max_skip_steps=self.max_skip_steps, retention_ratio=self.retention_ratio,
            num_inference_steps=int(model.num_inference_steps), mag_ratios=list(ratios),
        )
