"""WorldCache-style caching for diffusion transformers (Nawaz et al., arXiv 2603.22286).

Implemented from the paper's description (no official code was available), as hooks on a
diffusers transformer, in the same shape as diffusers' first-block cache: the first block is a
cheap *probe*; if the probe says the denoising state has barely moved since the previous step,
the remaining blocks are skipped and their effect is approximated from the previous steps.
What WorldCache adds to a plain first-block cache are four ideas, each switchable here so they
can be ablated:

1. Motion-adaptive threshold (CFC): tau = tau0 / (1 + alpha * v), where v is how much the
   transformer's *input latents* changed over the last two steps. Fast dynamics tighten the
   threshold, slow ones loosen it.
2. Saliency-weighted drift (SWD): the probe's drift is a spatially weighted relative L1 change,
   weighting each location by (1 + beta_s * S), S being the normalized channel variance of the
   probe features, so changes in salient regions count more.
3. Optimal interpolation (OSI): when skipping, instead of reusing the last deep residual,
   extrapolate along the last two with a least-squares gain gamma in [0, gamma_max]:
   r_hat = r_{t-2} + gamma * (r_{t-1} - r_{t-2}).
4. Adaptive threshold schedule (ATS): tau(t) = tau_CFC * (1 + beta_a * t / T), relaxing the
   threshold late in denoising, where the network only makes fine corrections.

5. Motion-compensated warping (optional, `warp=True`): the paper estimates a displacement between the
   latents of consecutive steps and warps the cached deep features before reusing them. Here the
   displacement is one global translation per latent frame, found by multi-scale phase correlation of
   the transformer's input latents (`estimate_shift`), and it is applied to the cached deep residuals
   with a bilinear shift (`warp_tokens`); off for the first `warp_start_step` (5) steps, as in the paper.
   The paper's motion model may be richer (dense or local) than a global per-frame translation.

Defaults are the paper's: tau0 0.08, alpha 2, beta_s 0.12, beta_a 4, gamma_max 2.

Where the description was ambiguous I chose, and these are deviations a reader should know:
- "r_tilde_t" in OSI is not specified. I estimate the gain from the *probe* outputs, the one thing
  known at step t: gamma = <z_t - z_{t-2}, z_{t-1} - z_{t-2}> / ||z_{t-1} - z_{t-2}||^2, clamped.
- Deep residuals are measured from the probe output (r = z_N - z_probe), as in diffusers' first-block
  cache, so the approximation is z_probe_t + r_hat; the paper writes it from the block input.
- Drift is normalized by the weighted L1 norm of the previous probe output (a relative drift), which
  reduces to the paper's Eq. 1 when beta_s = 0.
- `drift_reference`: the paper's formula compares each probe with the previous *step's* probe
  ("previous_step"). Taken literally that never sees the error from reusing a stale deep residual, so
  with the paper's defaults it skipped 28 of 30 steps on Wan and destroyed the video (PSNR ~5 dB,
  see DESIGN_DIFFUSION.md). The default here is "last_computed" (drift since the last full forward,
  which is what diffusers' first-block cache and TeaCache do, so the error accumulates into the
  decision); "previous_step" is kept so the literal reading can be reproduced.
  `max_consecutive_skips` (default off) is an extra guard on staleness.
- The saliency map and the velocity proxy use the transformer's input latents and token grid, read
  by a forward pre-hook on the transformer.
"""

from __future__ import annotations

from typing import Any

from worldoptbench.models.base import Architecture, WorldModelInterface
from worldoptbench.optimizations.base import register_module
from worldoptbench.optimizations.diffusion import CACHE_GROUP, _TransformerModule, refresh_hook_registries

_HEAD_HOOK = "worldcache_head_hook"
_BLOCK_HOOK = "worldcache_block_hook"
_EPS = 1e-8


# ---- the maths, kept free of hooks so it can be tested on plain tensors ---------------------------------


def relative_l1(current: Any, previous: Any) -> float:
    """||current - previous||_1 / (||previous||_1 + eps): the paper's Eq. 1 drift / velocity proxy."""
    return float(((current - previous).abs().sum() / (previous.abs().sum() + _EPS)).item())


def saliency_map(features: Any, grid: tuple[int, int, int]) -> Any:
    """Normalized [0, 1] spatial saliency, shape (H, W): channel variance of the features averaged
    over batch and time. `features` is (B, F*H*W, D) in the transformer's token order."""
    batch, _, dim = features.shape
    frames, height, width = grid
    spatial = features.float().reshape(batch, frames, height, width, dim).mean(dim=(0, 1))  # (H, W, D)
    variance = spatial.var(dim=-1, unbiased=False)
    low, high = variance.min(), variance.max()
    return (variance - low) / (high - low + _EPS)


def weighted_drift(current: Any, previous: Any, grid: tuple[int, int, int] | None, beta_s: float) -> float:
    """Saliency-weighted relative L1 drift between consecutive probe outputs. With no grid or
    beta_s == 0 this is exactly `relative_l1`."""
    if grid is None or beta_s == 0.0:
        return relative_l1(current, previous)
    batch, tokens, dim = current.shape
    frames, height, width = grid
    if frames * height * width != tokens:
        return relative_l1(current, previous)  # tokens aren't a plain grid (e.g. extra tokens): no spatial weighting
    weights = 1.0 + beta_s * saliency_map(current, grid)  # (H, W)
    change = (current.float() - previous.float()).abs().sum(dim=-1).reshape(batch, frames, height, width)
    base = previous.float().abs().sum(dim=-1).reshape(batch, frames, height, width)
    return float(((change * weights).sum() / ((base * weights).sum() + _EPS)).item())


def motion_threshold(tau0: float, alpha: float, velocity: float) -> float:
    return tau0 / (1.0 + alpha * velocity)


def scheduled_threshold(tau: float, beta_a: float, step: int, total_steps: int) -> float:
    return tau * (1.0 + beta_a * step / max(total_steps, 1))


def interpolation_gain(probe_now: Any, probe_prev: Any, probe_prev2: Any, gamma_max: float) -> float:
    """Least-squares gain gamma* = <d_tgt, d_src> / ||d_src||^2 in [0, gamma_max], with
    d_tgt = z_t - z_{t-2} and d_src = z_{t-1} - z_{t-2}."""
    d_tgt = (probe_now - probe_prev2).float()
    d_src = (probe_prev - probe_prev2).float()
    gamma = ((d_tgt * d_src).sum() / ((d_src * d_src).sum() + _EPS)).item()
    return min(max(gamma, 0.0), gamma_max)


# ---- motion estimation and warping (torch) ------------------------------------------------------------


def translate(images: Any, dy: Any, dx: Any) -> Any:
    """Shift each image by (dy, dx) pixels (may be fractional): out(y, x) = in(y - dy, x - dx), bilinear,
    border-padded. images: (N, C, H, W); dy, dx: (N,) tensors. Content moves by +d."""
    import torch  # noqa: PLC0415
    import torch.nn.functional as functional  # noqa: PLC0415

    n, _, h, w = images.shape
    ys = torch.arange(h, device=images.device, dtype=torch.float32)[None, :, None]
    xs = torch.arange(w, device=images.device, dtype=torch.float32)[None, None, :]
    gy = (ys - dy.float().to(images.device)[:, None, None]).expand(n, h, w)
    gx = (xs - dx.float().to(images.device)[:, None, None]).expand(n, h, w)
    grid = torch.stack((2.0 * gx / max(w - 1, 1) - 1.0, 2.0 * gy / max(h - 1, 1) - 1.0), dim=-1)
    out = functional.grid_sample(images.float(), grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out.to(images.dtype)


def _hann(h: int, w: int, device: Any) -> Any:
    import torch  # noqa: PLC0415

    rows = torch.hann_window(h, periodic=False, device=device)[:, None]
    cols = torch.hann_window(w, periodic=False, device=device)[None, :]
    return rows * cols


def phase_shift(a: Any, b: Any) -> tuple[float, float]:
    """(dy, dx) with a(y, x) ~= b(y - dy, x - dx), by phase correlation summed over channels, with a
    Hann window against edge effects and a parabolic fit for sub-pixel accuracy. a, b: (C, H, W)."""
    import torch  # noqa: PLC0415

    _, h, w = a.shape
    window = _hann(h, w, a.device)
    fa = torch.fft.rfft2(a.float() * window)
    fb = torch.fft.rfft2(b.float() * window)
    cross = (fa * fb.conj()).sum(dim=0)
    cross = cross / (cross.abs() + 1e-8)
    corr = torch.fft.irfft2(cross, s=(h, w))
    idx = int(torch.argmax(corr))
    y, x = divmod(idx, w)

    def refine(index: int, size: int, minus: float, centre: float, plus: float) -> float:
        denominator = minus - 2.0 * centre + plus
        offset = 0.0 if abs(denominator) < 1e-12 else 0.5 * (minus - plus) / denominator
        position = index + max(-0.5, min(0.5, offset))
        return position if position <= size / 2 else position - size

    centre = float(corr[y, x])
    dy = refine(y, h, float(corr[(y - 1) % h, x]), centre, float(corr[(y + 1) % h, x]))
    dx = refine(x, w, float(corr[y, (x - 1) % w]), centre, float(corr[y, (x + 1) % w]))
    return dy, dx


def estimate_shift(current: Any, previous: Any, levels: int = 2, max_shift: float = 3.0) -> Any:
    """One global translation per frame, (F, 2) = (dy, dx) in latent pixels, with current(y, x) ~=
    previous(y - dy, x - dx). Coarse to fine over `levels` pyramid levels (2x average pooling each).
    Estimates larger than `max_shift` in either axis are treated as unreliable and zeroed.
    current, previous: (B, C, F, H, W) latents."""
    import torch  # noqa: PLC0415
    import torch.nn.functional as functional  # noqa: PLC0415

    b, c, frames, h, w = current.shape
    cur = current.float().permute(2, 0, 1, 3, 4).reshape(frames, b * c, h, w)
    prev = previous.float().permute(2, 0, 1, 3, 4).reshape(frames, b * c, h, w)
    out = torch.zeros(frames, 2)
    for f in range(frames):
        total = torch.zeros(2)
        for level in range(levels - 1, -1, -1):
            scale = 2**level
            a = functional.avg_pool2d(cur[f : f + 1], scale)[0] if scale > 1 else cur[f]
            p = functional.avg_pool2d(prev[f : f + 1], scale)[0] if scale > 1 else prev[f]
            if min(a.shape[-2:]) < 4:
                continue
            guess = (total / scale).to(p.device)
            p = translate(p[None], guess[0:1], guess[1:2])[0]  # apply what the coarser levels found
            dy, dx = phase_shift(a, p)
            total = total + torch.tensor([dy, dx]) * scale
        if float(total.abs().max()) > max_shift:
            total = torch.zeros(2)
        out[f] = total
    return out


def warp_tokens(features: Any, grid: tuple[int, int, int], shifts: Any) -> Any:
    """Translate token features (B, F*H*W, D) by per-frame shifts (F, 2) = (dy, dx) in *token* units."""
    batch, tokens, dim = features.shape
    frames, height, width = grid
    if frames * height * width != tokens or not bool((shifts != 0).any()):
        return features
    images = features.reshape(batch, frames, height, width, dim).permute(0, 1, 4, 2, 3)
    images = images.reshape(batch * frames, dim, height, width)
    moved = translate(images, shifts[:, 0].repeat(batch), shifts[:, 1].repeat(batch))
    return moved.reshape(batch, frames, dim, height, width).permute(0, 1, 3, 4, 2).reshape(batch, tokens, dim)


# ---- hooks ---------------------------------------------------------------------------------------------


def _build_hooks():
    """diffusers imports are deferred so this module imports without diffusers installed."""
    import torch  # noqa: PLC0415
    from diffusers.hooks.first_block_cache import FBCBlockHook, FBCHeadBlockHook, FBCSharedBlockState  # noqa: PLC0415

    class WorldCacheState(FBCSharedBlockState):
        def __init__(self) -> None:
            super().__init__()
            self.step = 0
            self.pending_compute = False  # the previous step ran the full stack; its residual isn't recorded yet
            self.consecutive_skips = 0
            self.grid: tuple[int, int, int] | None = None
            self.patch: tuple[int, int, int] = (1, 1, 1)
            self.latent: Any = None
            self.last_computed_probe: Any = None  # probe output at the last full forward
            self.latents: list = []  # input latents of the last two steps
            self.probes: list = []  # probe outputs of the last two steps (newest last)
            self.residuals: list = []  # deep residuals of the last two steps (newest last)

        def reset(self):
            super().reset()
            self.__init__()

    class WorldCacheHeadHook(FBCHeadBlockHook):
        _is_stateful = True

        def __init__(self, state_manager, config: dict, total_steps, recorder: dict):
            super().__init__(state_manager, threshold=0.0)
            self.config = config
            self.total_steps = total_steps
            # context name -> one dict per step of the current/last video. Lives outside the per-video
            # state because the pipeline resets that state when a call ends (maybe_free_model_hooks).
            self.recorder = recorder

        def _decide(self, state: WorldCacheState, probe: Any) -> tuple[bool, float | None, dict]:
            """(should_compute_remaining_blocks, gain used for interpolation, what the decision saw)."""
            cfg = self.config
            seen: dict = {"step": state.step, "drift": None, "velocity": None, "threshold": None}
            if not state.residuals or not state.probes:
                return True, None, seen  # nothing to approximate from yet
            if cfg["max_consecutive_skips"] is not None and state.consecutive_skips >= cfg["max_consecutive_skips"]:
                return True, None, seen

            velocity = 0.0
            if cfg["motion_adaptive"] and len(state.latents) >= 2 and state.latent is not None:
                velocity = relative_l1(state.latent, state.latents[-2])
            tau = motion_threshold(cfg["tau0"], cfg["alpha"], velocity)
            if cfg["schedule"]:
                tau = scheduled_threshold(tau, cfg["beta_a"], state.step, int(self.total_steps()))
            reference = state.last_computed_probe if cfg["drift_reference"] == "last_computed" else state.probes[-1]
            drift = weighted_drift(probe, reference, state.grid if cfg["saliency"] else None, cfg["beta_s"])
            seen.update(drift=drift, velocity=velocity, threshold=tau)
            if drift >= tau:
                return True, None, seen

            gain = None
            if cfg["interpolate"] and len(state.residuals) >= 2 and len(state.probes) >= 2:
                gain = interpolation_gain(probe, state.probes[-1], state.probes[-2], cfg["gamma_max"])
            return False, gain, seen

        def new_forward(self, module: torch.nn.Module, *args, **kwargs):
            meta = self._metadata
            original_hidden = meta._get_parameter_from_args_kwargs("hidden_states", args, kwargs)
            output = self.fn_ref.original_forward(*args, **kwargs)
            is_tuple = isinstance(output, tuple)
            probe = output[meta.return_hidden_states_index] if is_tuple else output

            state: WorldCacheState = self.state_manager.get_state()
            if state.pending_compute and state.tail_block_residuals is not None:
                state.residuals = (state.residuals + [state.tail_block_residuals[0]])[-2:]
            state.pending_compute = False

            should_compute, gain, seen = self._decide(state, probe)
            state.should_compute = should_compute
            seen["skip"] = not should_compute
            context = self.state_manager._current_context
            if state.step == 0:
                self.recorder[context] = []  # a new video for this context
            self.recorder.setdefault(context, []).append(seen)

            previous_latents = state.latents  # before this step's latent is appended
            if state.latent is not None:
                state.latents = (state.latents + [state.latent])[-2:]
            state.probes = (state.probes + [probe.detach()])[-2:]
            state.step += 1

            if should_compute:
                state.consecutive_skips = 0
                state.pending_compute = True
                if is_tuple:
                    head_output = [None] * len(output)
                    head_output[0] = output[meta.return_hidden_states_index]
                    head_output[1] = output[meta.return_encoder_hidden_states_index]
                else:
                    head_output = output
                state.head_block_output = head_output
                state.head_block_residual = probe - original_hidden
                state.last_computed_probe = probe.detach()
                return output

            # skip the remaining blocks: approximate their effect from the previous deep residuals
            state.consecutive_skips += 1
            newest = state.residuals[-1]
            older = state.residuals[-2] if len(state.residuals) >= 2 else None
            if (
                self.config["warp"] and state.step - 1 >= self.config["warp_start_step"]
                and state.grid is not None and state.latent is not None and state.latent.ndim == 5
            ):
                # cached residuals belong to earlier steps: bring them into this step's spatial layout
                shift_sizes = []

                def aligned(residual, earlier_latent):
                    shift = estimate_shift(state.latent, earlier_latent, max_shift=self.config["max_shift"])
                    shift_sizes.append(float(shift.abs().mean()))
                    token_shift = shift / torch.tensor(state.patch[1:], dtype=shift.dtype)
                    return warp_tokens(residual, state.grid, token_shift.to(residual.device))

                if previous_latents:
                    newest = aligned(newest, previous_latents[-1])
                if older is not None and len(previous_latents) >= 2:
                    older = aligned(older, previous_latents[-2])
                if shift_sizes:
                    seen["shift"] = sum(shift_sizes) / len(shift_sizes)
            if gain is not None and older is not None:
                residual = older + gain * (newest - older)
            else:
                residual = newest
            state.residuals = (state.residuals + [residual])[-2:]
            encoder_residual = state.tail_block_residuals[1] if state.tail_block_residuals is not None else None
            state.tail_block_residuals = (residual, encoder_residual)

            hidden = residual + probe
            if not is_tuple:
                return hidden
            result = [None] * len(output)
            result[meta.return_hidden_states_index] = hidden
            if meta.return_encoder_hidden_states_index is not None:
                result[meta.return_encoder_hidden_states_index] = (
                    encoder_residual + output[meta.return_encoder_hidden_states_index]
                    if encoder_residual is not None
                    else output[meta.return_encoder_hidden_states_index]
                )
            return tuple(result)

        def reset_state(self, module):
            self.state_manager.reset()
            return module

    return WorldCacheState, WorldCacheHeadHook, FBCBlockHook


def apply_worldcache(transformer: Any, config: dict, total_steps, recorder: dict) -> list:
    """Hook `transformer`'s blocks; returns the (module, hook name) pairs to undo, plus the
    transformer pre-hook handle as ("pre", handle)."""
    from diffusers.hooks import HookRegistry  # noqa: PLC0415
    from diffusers.hooks._common import _ALL_TRANSFORMER_BLOCK_IDENTIFIERS  # noqa: PLC0415
    from diffusers.hooks.hooks import StateManager  # noqa: PLC0415
    import torch  # noqa: PLC0415

    WorldCacheState, HeadHook, BlockHook = _build_hooks()
    state_manager = StateManager(WorldCacheState, (), {})

    blocks = []
    for name, child in transformer.named_children():
        if name in _ALL_TRANSFORMER_BLOCK_IDENTIFIERS and isinstance(child, torch.nn.ModuleList):
            blocks.extend(child)
    if len(blocks) < 3:
        raise ValueError(f"worldcache needs a transformer with at least 3 blocks, found {len(blocks)}")

    undo: list = []
    head, tail = blocks[0], blocks[-1]
    HookRegistry.check_if_exists_or_initialize(head).register_hook(HeadHook(state_manager, config, total_steps, recorder), _HEAD_HOOK)
    undo.append((head, _HEAD_HOOK))
    for block in blocks[1:]:
        hook = BlockHook(state_manager, is_tail=block is tail)
        HookRegistry.check_if_exists_or_initialize(block).register_hook(hook, _BLOCK_HOOK)
        undo.append((block, _BLOCK_HOOK))

    patch = tuple(getattr(getattr(transformer, "config", None), "patch_size", (1, 1, 1)))

    def read_input(module, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if hidden is None or hidden.ndim != 5:
            return None
        state = state_manager.get_state()
        _, _, frames, height, width = hidden.shape
        state.grid = (frames // patch[0], height // patch[1], width // patch[2])
        state.patch = patch
        state.latent = hidden.detach()
        return None

    handle = transformer.register_forward_pre_hook(read_input, with_kwargs=True)
    undo.append(("pre", handle))
    transformer._worldcache_state_manager = state_manager
    refresh_hook_registries(transformer)
    return undo


# ---- the module ----------------------------------------------------------------------------------------


@register_module
class WorldCacheModule(_TransformerModule):
    """WorldCache-style caching (see this file's docstring); a training-free drop-in for
    first_block_cache with a motion-adaptive, saliency-weighted, schedule-relaxed threshold and
    an extrapolating approximation. Mutually exclusive with the diffusers caches."""

    name = "worldcache"
    supported_architectures: tuple[Architecture, ...] = ("diffusion",)
    requires = ("transformer", "num_inference_steps")
    exclusive_group = CACHE_GROUP
    maturity = "measured"
    summary = (
        "WorldCache-style adaptive caching, implemented from the paper's description. Measured on Wan2.1-1.3B "
        "(32 clips): tau0 0.02: 1.21x, below a numerical-perturbation control on both PSNR (27.0) and reference-free "
        "statistics; 0.04: 1.65x, a small but measurable shift (flicker ~+17%); 0.08: 2.40x, a large one (motion +65%, "
        "flicker x2.2). Clearly better than first_block_cache at ~1.7x (style deviation 0.126 vs 0.296), worse than "
        "cfg_truncation at <= 1.4x. On learned scores (DINOv2 similarity to the baseline video; controls 0.88) it keeps 0.83 at 1.65-1.98x against 0.68 for first_block_cache at 1.73x and 0.57-0.69 for AdaCache at 1.7-2x. With cfg_truncation 0.6: 1.99x at the same shift as 0.04 alone. "
        "The speedup comes almost entirely from the threshold schedule; interpolation and saliency weighting showed "
        "no measurable benefit; the paper's literal previous-step drift destroys the video (drift_reference). "
        "Motion-compensated warping is implemented as an option (warp=True, global per-frame phase correlation) and measured as a no-op: the displacement between consecutive-step latents is ~0.001 latent px, quality is unchanged and it costs ~11% speed, so it is off by default. Does not reproduce the paper's quality claim: different metric."
    )

    def __init__(
        self,
        tau0: float = 0.08,
        alpha: float = 2.0,
        beta_s: float = 0.12,
        beta_a: float = 4.0,
        gamma_max: float = 2.0,
        motion_adaptive: bool = True,
        saliency: bool = True,
        interpolate: bool = True,
        schedule: bool = True,
        max_consecutive_skips: int | None = None,
        drift_reference: str = "last_computed",
        warp: bool = False,
        warp_start_step: int = 5,
        max_shift: float = 3.0,
    ):
        if tau0 < 0:
            raise ValueError("tau0 must be >= 0 (0 never skips)")
        if gamma_max < 0:
            raise ValueError("gamma_max must be >= 0")
        if warp_start_step < 0 or max_shift <= 0:
            raise ValueError("warp_start_step must be >= 0 and max_shift > 0")
        if drift_reference not in ("last_computed", "previous_step"):
            raise ValueError("drift_reference must be 'last_computed' or 'previous_step'")
        if max_consecutive_skips is not None and max_consecutive_skips < 1:
            raise ValueError("max_consecutive_skips must be >= 1 or None")
        self.config = {
            "tau0": tau0, "alpha": alpha, "beta_s": beta_s, "beta_a": beta_a, "gamma_max": gamma_max,
            "motion_adaptive": motion_adaptive, "saliency": saliency, "interpolate": interpolate,
            "schedule": schedule, "max_consecutive_skips": max_consecutive_skips,
            "drift_reference": drift_reference, "warp": warp, "warp_start_step": warp_start_step,
            "max_shift": max_shift,
        }
        self._undo: list = []

    @property
    def label(self) -> str:
        off = [k for k in ("motion_adaptive", "saliency", "interpolate", "schedule") if not self.config[k]]
        suffix = ("_no_" + "_no_".join(k.split("_")[0] for k in off)) if off else ""
        if self.config["drift_reference"] != "last_computed":
            suffix += "_prevstep"
        if self.config["warp"]:
            suffix += "_warp"
        return f"{self.name}_{self.config['tau0']:g}{suffix}"

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        transformer = self._transformer(model)
        if getattr(transformer, "is_cache_enabled", False) or getattr(transformer, "_worldcache_state_manager", None):
            raise RuntimeError("a cache is already applied to this transformer; restore it first")
        self._applied_to = transformer
        self._recorder: dict = {}
        self._undo = apply_worldcache(transformer, self.config, lambda: model.num_inference_steps, self._recorder)
        return model

    def trace(self, context: str = "cond") -> list[dict]:
        """Per-step decisions for one CFG context in the last video: step, drift, velocity, threshold
        (None where there was nothing to compare yet), skip. For tuning: compare `drift` to `threshold`."""
        return list(getattr(self, "_recorder", {}).get(context, []))

    def skipped_steps(self) -> int:
        """Steps whose deep blocks were skipped in the last video, summed over the CFG contexts
        (a diagnostic: 0 means the cache never engaged)."""
        return sum(entry["skip"] for entries in getattr(self, "_recorder", {}).values() for entry in entries)

    def restore(self) -> None:
        for target, hook in self._undo:
            if target == "pre":
                hook.remove()
            else:
                target._diffusers_hook.remove_hook(hook, recurse=False)
        self._undo = []
        transformer = getattr(self, "_applied_to", None)
        if transformer is not None:
            if hasattr(transformer, "_worldcache_state_manager"):
                del transformer._worldcache_state_manager
            refresh_hook_registries(transformer)
