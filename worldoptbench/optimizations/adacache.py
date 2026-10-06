"""AdaCache-style adaptive residual caching for diffusion transformers (Kahatapitiya et al.,
arXiv 2411.02397), implemented from the paper's description (no code was available).

What it does. Each transformer block's attention, cross-attention and MLP *outputs* (the residual
branches) are cached. At every step that is fully computed, a distance between the new outputs and
the previously computed ones, divided by the number of steps between them, says how fast the
representation is changing. A codebook maps that distance to a cache rate `tau`: the next
`tau - 1` steps reuse the cached outputs, and the step after that recomputes. Slow change gives a
long rate, fast change a short one, so the schedule adapts to each video (unlike PAB's fixed
period). The decision is global: one distance and one rate for every layer.

Codebooks are the paper's (30-step, "slow" and "fast"; 100-step "fast"), mapping a distance
threshold to a rate: the first threshold the distance is under wins, and anything beyond the last
threshold gets rate 1 (no caching). Those thresholds were tuned for Open-Sora, Open-Sora-Plan and
Latte, whose activation scales differ from Wan's, so `distance_scale` multiplies the measured
distance (set it from `trace()` on your own model before trusting a codebook).

Optional motion regularization (`moreg=True`): the distance is multiplied by (m + mg), where m is
the mean absolute frame-to-frame difference of the transformer's input latents and mg the change of
m per step since the last computation, so higher motion gives a larger distance and a shorter rate.

Optional drift guard (`guard=<threshold>`, not in the paper): AdaCache can only see drift at the steps it
recomputes, so a long cache rate runs blind between them. With a guard, the first `guard_blocks` blocks are
computed on every step as a probe, and their relative change since the last full computation (mean absolute
difference over mean absolute value, averaged over the probe modules) is compared with the threshold: if it
is larger, the scheduled reuse is vetoed and the step recomputes everything (and the schedule restarts from
the codebook as for any computed step). It is a portable safety net in the sense that it needs nothing from
the caching rule underneath except "reuse is scheduled".

Choices where the description was ambiguous (deviations to know about):
- The distance is the mean absolute difference (L1 divided by element count) of each hooked
  module's new output against its previously computed output, averaged over modules; the paper
  says L1 and a common metric across layers but not how layers are combined or normalized.
- Rates count steps in each classifier-free-guidance context independently (cond and uncond keep
  separate schedules and caches).
- Step 0 always computes, and step 1 too (a distance needs two computed steps).
"""

from __future__ import annotations

import re
from typing import Any

from worldoptbench.models.base import Architecture, WorldModelInterface
from worldoptbench.optimizations.base import register_module
from worldoptbench.optimizations.diffusion import CACHE_GROUP, _TransformerModule, refresh_hook_registries

_HOOK = "adacache_hook"
_MODULE_PATTERN = r"^[A-Za-z_]*blocks\.\d+\.(attn\d*|ffn|ff)$"

# distance threshold -> cache rate (steps between full computations); a trailing 1.00 -> 1 entry means
# "anything larger: no caching"
PRESETS: dict[str, dict[float, int]] = {
    "fast30": {0.08: 6, 0.16: 5, 0.24: 4, 0.32: 3, 0.40: 2, 1.00: 1},
    "slow30": {0.08: 3, 0.16: 2, 1.00: 1},
    "fast100": {0.03: 12, 0.05: 10, 0.07: 8, 0.09: 6, 0.11: 4, 1.00: 3},
}


def codebook_rate(distance: float, codebook: dict[float, int]) -> int:
    """The rate of the first threshold the distance is under; 1 (recompute next step) beyond the last."""
    for threshold in sorted(codebook):
        if distance < threshold:
            return int(codebook[threshold])
    return 1


def motion_score(latent: Any, frame_step: int = 1) -> float:
    """Mean absolute difference between latent frames `frame_step` apart; (B, C, F, H, W) in, 0 if the
    latent has too few frames."""
    if latent.ndim != 5 or latent.shape[2] <= frame_step:
        return 0.0
    return float((latent[:, :, frame_step:].float() - latent[:, :, :-frame_step].float()).abs().mean().item())


def _build_hook():
    import torch  # noqa: PLC0415
    from diffusers.hooks.hooks import BaseState, ModelHook  # noqa: PLC0415

    class AdaState(BaseState):
        def __init__(self) -> None:
            super().__init__()
            self.step = 0
            self.next_compute = 0  # the first step at which a full computation is due
            self.reuse = False  # this step reuses the caches
            self.last_compute = None  # step index of the previous full computation
            self.cache: dict[int, Any] = {}  # module index -> output at the last full computation
            self.distance_sum: Any = None  # sum over modules of mean|new - old| in the current full step
            self.distance_count = 0
            self.latent: Any = None
            self.last_motion: float | None = None
            # drift guard
            self.scheduled_reuse = False  # reuse was scheduled, before any veto
            self.vetoed = False
            self.probe_drift: float | None = None
            self.probe_drift_sum: Any = None
            self.probe_drift_count = 0
            self.probe_distance_sum: Any = None
            self.probe_distance_count = 0
            self.probe_pending: dict[int, Any] = {}

        def reset(self):
            self.__init__()

    class AdaCacheHook(ModelHook):
        _is_stateful = True

        def __init__(self, state_manager, controller: dict, index: int, last_index: int):
            super().__init__()
            self.state_manager = state_manager
            self.controller = controller  # shared: config + recorder
            self.index = index
            self.last_index = last_index

        def new_forward(self, module: torch.nn.Module, *args, **kwargs):
            state: AdaState = self.state_manager.get_state()
            cfg = self.controller["config"]
            probes = self.controller["probe_modules"]  # leading modules computed on every step (0 = no guard)
            if self.index == 0:
                state.reuse = state.step < state.next_compute
                state.scheduled_reuse = state.reuse
                state.vetoed = False
                state.probe_drift = None
                state.probe_drift_sum, state.probe_drift_count = None, 0
                state.probe_distance_sum, state.probe_distance_count = None, 0
                state.probe_pending = {}

            if self.index < probes:
                # a probe: always computed; its change since the last full computation feeds the guard, and its
                # new output only becomes the reference if this step ends up being a full computation
                output = self.fn_ref.original_forward(*args, **kwargs)
                if isinstance(output, torch.Tensor):
                    old = state.cache.get(self.index)
                    if old is not None and old.shape == output.shape:
                        change = (output.detach() - old).abs().mean()
                        relative = change / (old.abs().mean() + 1e-8)
                        state.probe_drift_sum = relative if state.probe_drift_sum is None else state.probe_drift_sum + relative
                        state.probe_drift_count += 1
                        state.probe_distance_sum = change if state.probe_distance_sum is None else state.probe_distance_sum + change
                        state.probe_distance_count += 1
                    state.probe_pending[self.index] = output.detach()
            else:
                if probes and self.index == probes and state.reuse and state.probe_drift_count:
                    state.probe_drift = float((state.probe_drift_sum / state.probe_drift_count).item())
                    if state.probe_drift > cfg["guard"]:
                        state.reuse = False  # veto: the probe says the cached features are too stale
                        state.vetoed = True
                if state.reuse and self.index in state.cache:
                    output = state.cache[self.index]
                else:
                    output = self.fn_ref.original_forward(*args, **kwargs)
                    if isinstance(output, torch.Tensor):
                        old = state.cache.get(self.index)
                        if old is not None and old.shape == output.shape:
                            distance = (output.detach() - old).abs().mean()
                            state.distance_sum = distance if state.distance_sum is None else state.distance_sum + distance
                            state.distance_count += 1
                        state.cache[self.index] = output.detach()

            if self.index == self.last_index:
                self._finish_step(state)
            return output

        def _finish_step(self, state: AdaState) -> None:
            cfg = self.controller["config"]
            recorder = self.controller["recorder"]
            context = self.state_manager._current_context
            entry = {"step": state.step, "compute": not state.reuse, "distance": None, "rate": None}
            if self.controller["probe_modules"]:
                entry["probe_drift"] = state.probe_drift
                entry["vetoed"] = state.vetoed
                if not state.reuse:  # a full computation: the probes' outputs become the new references
                    state.cache.update(state.probe_pending)
                    if state.probe_distance_count:
                        state.distance_sum = (
                            state.probe_distance_sum if state.distance_sum is None else state.distance_sum + state.probe_distance_sum
                        )
                        state.distance_count += state.probe_distance_count
                state.probe_pending = {}
            if state.step == 0:
                recorder[context] = []  # a new video for this context
            recorder.setdefault(context, []).append(entry)

            if not state.reuse:
                rate = 1
                if state.last_compute is not None and state.distance_count:
                    k = state.step - state.last_compute
                    distance = float(state.distance_sum.item()) / state.distance_count / k * cfg["distance_scale"]
                    if cfg["moreg"] and state.latent is not None:
                        motion = motion_score(state.latent, cfg["moreg_frame_step"])
                        gradient = 0.0 if state.last_motion is None else (motion - state.last_motion) / k
                        distance *= max(motion + gradient, 0.0)
                        entry["motion"] = motion
                    rate = min(codebook_rate(distance, cfg["codebook"]), cfg["max_rate"])
                    entry["distance"] = distance
                if cfg["moreg"] and state.latent is not None and "motion" not in entry:
                    entry["motion"] = motion_score(state.latent, cfg["moreg_frame_step"])
                state.last_motion = entry.get("motion", state.last_motion)
                entry["rate"] = rate
                state.next_compute = state.step + rate
                state.last_compute = state.step
                state.distance_sum, state.distance_count = None, 0
            state.step += 1

        def reset_state(self, module):
            self.state_manager.reset()
            return module

    return AdaState, AdaCacheHook


def apply_adacache(transformer: Any, config: dict) -> tuple[list, dict, Any]:
    """Hook the transformer's attention / cross-attention / MLP modules. Returns (undo list, recorder,
    state manager)."""
    from diffusers.hooks import HookRegistry  # noqa: PLC0415
    from diffusers.hooks.hooks import StateManager  # noqa: PLC0415

    AdaState, Hook = _build_hook()
    state_manager = StateManager(AdaState, (), {})
    named = [(name, m) for name, m in transformer.named_modules() if re.fullmatch(_MODULE_PATTERN, name)]
    targets = [m for _, m in named]
    if not targets:
        raise ValueError("adacache found no attention/MLP modules named like `blocks.N.attnK` / `blocks.N.ffn`")
    probe_modules = 0
    if config["guard"] is not None:
        block_of = [int(re.search(r"blocks\.(\d+)\.", name).group(1)) for name, _ in named]
        probe_modules = sum(1 for b in block_of if b < config["guard_blocks"])
        if probe_modules == 0 or probe_modules >= len(targets):
            raise ValueError(f"guard_blocks={config['guard_blocks']} leaves no probe or no cached module among {len(targets)} modules")

    recorder: dict = {}
    controller = {"config": config, "recorder": recorder, "probe_modules": probe_modules}
    undo: list = []
    for index, module in enumerate(targets):
        hook = Hook(state_manager, controller, index, len(targets) - 1)
        HookRegistry.check_if_exists_or_initialize(module).register_hook(hook, _HOOK)
        undo.append((module, _HOOK))

    if config["moreg"]:

        def read_input(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is not None and hidden.ndim == 5:
                state_manager.get_state().latent = hidden.detach()
            return None

        undo.append(("pre", transformer.register_forward_pre_hook(read_input, with_kwargs=True)))
    return undo, recorder, state_manager


@register_module
class AdaCacheModule(_TransformerModule):
    """AdaCache-style adaptive residual caching (see this file's docstring). Mutually exclusive with
    the other diffusion caches. Needs `distance_scale` calibrated per model."""

    name = "adacache"
    supported_architectures: tuple[Architecture, ...] = ("diffusion",)
    exclusive_group = CACHE_GROUP
    maturity = "measured"
    summary = (
        "AdaCache-style adaptive residual caching, implemented from the paper's description. Measured on Wan2.1-1.3B "
        "(32 clips; style deviation = reference-free distance from the baseline video, noise control 0.081): the "
        "paper's codebooks are too aggressive for Wan (slow30: 1.73x, deviation 0.318; fast30: 2.51x, 0.775); a "
        "larger distance_scale gives gentler points (slow30 x4: 1.15x, 0.051, within noise). At equal speed it is "
        "clearly worse than WorldCache (1.65x: 0.126) and worse than cfg_truncation + PAB below 1.5x; motion "
        "regularization showed no measurable benefit. guard=<threshold> (not in the paper) keeps the first block as a "
        "per-step probe and vetoes scheduled reuse when it has drifted: at matched speed it lowers the deviation "
        "(1.54x: 0.197 vs 0.318 unguarded) but stays behind WorldCache. Needs distance_scale calibrated per model."
    )

    def __init__(
        self,
        preset: str = "fast30",
        codebook: dict[float, int] | None = None,
        distance_scale: float = 1.0,
        max_rate: int = 12,
        moreg: bool = False,
        moreg_frame_step: int = 1,
        guard: float | None = None,
        guard_blocks: int = 1,
    ):
        if codebook is None and preset not in PRESETS:
            raise ValueError(f"preset must be one of {sorted(PRESETS)} (or pass a codebook)")
        if distance_scale <= 0 or max_rate < 1 or moreg_frame_step < 1:
            raise ValueError("distance_scale must be > 0, max_rate >= 1, moreg_frame_step >= 1")
        if (guard is not None and guard <= 0) or guard_blocks < 1:
            raise ValueError("guard must be > 0 (or None for no guard) and guard_blocks >= 1")
        book = {float(k): int(v) for k, v in (codebook if codebook is not None else PRESETS[preset]).items()}
        if not book or any(v < 1 for v in book.values()):
            raise ValueError("a codebook needs at least one threshold and every rate must be >= 1")
        self.preset = preset if codebook is None else "custom"
        self.config = {
            "codebook": book, "distance_scale": float(distance_scale), "max_rate": int(max_rate),
            "moreg": moreg, "moreg_frame_step": moreg_frame_step, "guard": guard, "guard_blocks": guard_blocks,
        }
        self._undo: list = []

    @property
    def label(self) -> str:
        suffix = "" if self.config["distance_scale"] == 1.0 else f"_x{self.config['distance_scale']:g}"
        guard = "" if self.config["guard"] is None else f"_guard{self.config['guard']:g}"
        return f"{self.name}_{self.preset}{suffix}" + ("_moreg" if self.config["moreg"] else "") + guard

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        transformer = self._transformer(model)
        if getattr(transformer, "is_cache_enabled", False) or getattr(transformer, "_adacache_state_manager", None):
            raise RuntimeError("a cache is already applied to this transformer; restore it first")
        self._applied_to = transformer
        self._undo, self._recorder, manager = apply_adacache(transformer, self.config)
        transformer._adacache_state_manager = manager
        refresh_hook_registries(transformer)
        return model

    def trace(self, context: str = "cond") -> list[dict]:
        """Per-step records for one CFG context of the last video: step, compute (False = reused the
        caches), distance (None where there was none yet), rate (None on reused steps). For calibrating
        `distance_scale`: look at the distances."""
        return list(getattr(self, "_recorder", {}).get(context, []))

    def vetoed_steps(self) -> int:
        """Scheduled reuses the drift guard vetoed in the last video, summed over CFG contexts."""
        return sum(bool(e.get("vetoed")) for entries in getattr(self, "_recorder", {}).values() for e in entries)

    def reused_steps(self) -> int:
        """Steps that reused caches in the last video, summed over CFG contexts (0 = never engaged)."""
        return sum(not e["compute"] for entries in getattr(self, "_recorder", {}).values() for e in entries)

    def restore(self) -> None:
        for target, hook in self._undo:
            if target == "pre":
                hook.remove()
            else:
                target._diffusers_hook.remove_hook(hook, recurse=False)
        self._undo = []
        transformer = getattr(self, "_applied_to", None)
        if transformer is not None:
            if hasattr(transformer, "_adacache_state_manager"):
                del transformer._adacache_state_manager
            refresh_hook_registries(transformer)
