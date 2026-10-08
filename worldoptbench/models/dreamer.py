"""DreamerV3 wrapper implementing WorldModelInterface, built against
github.com/NM512/dreamerv3-torch (archived by its author in favor of
NM512/r2dreamer, but self-contained and still the PyTorch port with the
simplest checkpoint format).

No Hugging Face, no gated weights: you train a checkpoint yourself, e.g.

    git clone https://github.com/NM512/dreamerv3-torch
    cd dreamerv3-torch && pip install -r requirements.txt
    python dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/walker

then point `DreamerRepo` at the clone and `DreamerWorldModel` at that logdir.
The repo is a set of scripts, not an installable package, so it's put on
`sys.path` at first use.

API status (read from the repo source on 2026-10-03; NOT run — no Python or
torch on the dev machine yet):
  - Mirrors dreamer.py's `__main__` config parsing, `make_env`, and checkpoint
    format (`latest.pt` -> {"agent_state_dict", "optims_state_dict"}), and
    models.WorldModel / networks.RSSM's observe / imagine_with_action.
  - Only the world model is loaded (the `_wm.*` weights); the actor/critic
    aren't needed because rollouts are driven by supplied actions.
  - INFERRED, not confirmed: that the fully wrapped env (UUID -> SelectAction
    -> TimeLimit -> NormalizeActions -> DeepMindControl) forwards
    `action_space` / `observation_space` (main() reads them off a `Damy`
    wrapper around it); that `env.step({"action": vec})` returns
    (obs, reward, done, info); and DMC walker's 20 steps/s. If the first real
    run trips on one of these, `DreamerRepo` is the one place to fix.

How this differs from the Cosmos wrapper:
  - Not text-conditioned. A rollout needs context frames (`init_video`) and an
    action sequence (`actions`); `prompt` is accepted but only recorded.
  - Frames come from imagining forward in latent space — no diffusion, no
    denoising loop — so diffusion-only optimizations (WorldCache etc.) don't
    apply. Architecture is labelled "autoregressive" (recurrent state-space
    model, one latent step per env step).
  - Context frames, actions, and ground truth come from a simulator via
    `DreamerSimScenarios`, kept outside `generate()` so simulator time isn't
    counted as model latency (see worldoptbench/scenarios.py).
  - Continuous-action tasks only (DMC). Discrete/one-hot envs would need
    different action sampling in `DreamerSimScenarios`.
"""

from __future__ import annotations

import argparse
import copy
import os
import pathlib
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

from worldoptbench.models.base import ModelInfo, Rollout, WorldModelInterface, eager_executor
from worldoptbench.scenarios import Scenario


def horizon_to_steps(horizon: float, steps_per_second: float) -> int:
    """Rollout length in seconds -> number of agent (action-repeated) steps."""
    return max(1, round(horizon * steps_per_second))


def overrides_to_argv(overrides: dict[str, Any]) -> list[str]:
    """Config overrides -> the argv list dreamer.py's argparse expects.

    Scalars only (str/int/float/bool): the repo's own `tools.args_type`
    parses each value from a string, and its handling of lists/dicts isn't
    something worth guessing at. Bool becomes "True"/"False", which is what
    `args_type` expects for bool defaults.
    """
    argv: list[str] = []
    for key, value in overrides.items():
        if not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"Override {key!r} must be a scalar, got {type(value).__name__}")
        argv += [f"--{key}", str(value)]
    return argv


def extract_world_model_state(agent_state_dict: dict[str, Any]) -> dict[str, Any]:
    """Pulls the WorldModel weights out of a Dreamer agent state dict.

    Dreamer stores the world model as `self._wm`, so keys look like
    `_wm.encoder...`. With `compile: True` they can also carry torch.compile's
    `_orig_mod.` segment, which is stripped.
    """
    prefix = "_wm."
    state = {
        key[len(prefix):].replace("_orig_mod.", ""): value
        for key, value in agent_state_dict.items()
        if key.startswith(prefix)
    }
    if not state:
        raise KeyError(
            "No '_wm.*' keys in agent_state_dict — is this a Dreamer checkpoint "
            "from dreamerv3-torch?"
        )
    return state


def _recursive_update(base: dict, update: dict) -> None:
    # Same merge dreamer.py's __main__ uses to layer named configs.
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            _recursive_update(base[key], value)
        else:
            base[key] = value


class DreamerRepo:
    """Handle on a dreamerv3-torch clone: builds its config, its simulator
    environments, and its observation/action spaces. Shared by
    DreamerWorldModel and DreamerSimScenarios so they can't disagree about
    config. Loading is lazy; constructing this imports nothing heavy.
    """

    def __init__(
        self,
        repo_path: str | pathlib.Path,
        task: str = "dmc_walker_walk",
        configs: Sequence[str] = ("dmc_vision",),
        device: str | None = None,
        steps_per_second: float = 20.0,
        overrides: dict[str, Any] | None = None,
    ):
        """
        Args:
            repo_path: the dreamerv3-torch clone.
            task: "<suite>_<task>", as passed to dreamer.py's --task.
            configs: named configs from configs.yaml (layered after `defaults`).
            device: torch device string; defaults to cuda:0 if available else cpu.
            steps_per_second: simulated seconds -> agent steps. The default,
                20, assumes DMC walker: 0.025 s control step x action_repeat 2
                (INFERRED from dm_control's walker default — check it, and
                change it for other tasks or action_repeat values).
            overrides: scalar config overrides (see `overrides_to_argv`).
        """
        self.repo_path = pathlib.Path(repo_path).expanduser().resolve()
        self.task = task
        self.configs = tuple(configs)
        self.device = device
        self.steps_per_second = steps_per_second
        self.overrides = dict(overrides or {})
        self._modules: tuple[Any, Any, Any] | None = None
        self._config: argparse.Namespace | None = None
        self._observation_space: Any = None
        self._action_space: Any = None

    def modules(self) -> tuple[Any, Any, Any]:
        """(dreamer, models, tools) from the clone, imported on first use."""
        if self._modules is not None:
            return self._modules

        if not (self.repo_path / "dreamer.py").exists():
            raise FileNotFoundError(f"No dreamer.py in {self.repo_path} — wrong repo_path?")
        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))

        # dreamer.py unconditionally sets MUJOCO_GL="osmesa" on import, which
        # doesn't exist on Windows. Keep a value the user set; otherwise leave
        # the repo's osmesa default off-Windows and unset it on Windows.
        user_gl = os.environ.get("MUJOCO_GL")
        import dreamer

        if user_gl is not None:
            os.environ["MUJOCO_GL"] = user_gl
        elif sys.platform == "win32":
            os.environ.pop("MUJOCO_GL", None)

        import models
        import tools

        self._modules = (dreamer, models, tools)
        return self._modules

    @property
    def config(self) -> argparse.Namespace:
        if self._config is not None:
            return self._config

        from ruamel import yaml

        _, _, tools = self.modules()
        raw = yaml.safe_load((self.repo_path / "configs.yaml").read_text())
        defaults: dict[str, Any] = {}
        for name in ["defaults", *self.configs]:
            _recursive_update(defaults, raw[name])

        parser = argparse.ArgumentParser()
        for key, value in sorted(defaults.items(), key=lambda kv: kv[0]):
            arg_type = tools.args_type(value)
            parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

        device = self.device
        if device is None:
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        argv = overrides_to_argv(
            {
                "task": self.task,
                "device": device,
                "compile": False,  # not needed for inference; also unsupported on Windows
                **self.overrides,
            }
        )
        config = parser.parse_args(argv)
        config.time_limit //= config.action_repeat  # as main() does: env steps -> agent steps

        # main() also sets num_actions from the env's action space.
        env = self._make_env(config)
        try:
            acts = env.action_space
            config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
            self._observation_space = env.observation_space
            self._action_space = acts
        finally:
            _close(env)

        self._config = config
        return config

    @property
    def observation_space(self):
        _ = self.config  # populates the spaces
        return self._observation_space

    @property
    def action_space(self):
        _ = self.config
        return self._action_space

    @property
    def max_steps(self) -> int:
        """Agent steps in one episode — the ceiling on context + horizon,
        because the simulator can't provide ground truth past episode end.
        """
        return int(self.config.time_limit)

    def _make_env(self, config: argparse.Namespace):
        dreamer, _, _ = self.modules()
        return dreamer.make_env(config, "eval", 0)

    def make_env(self, seed: int):
        """A fresh, fully wrapped simulator env (observations are dicts with
        "image"/"is_first"/"is_terminal"; actions are `{"action": vec}` with
        vec in [-1, 1]) — exactly what dreamer.py's make_env builds. A new
        env per seed keeps scenarios reproducible regardless of call order.
        """
        config = copy.copy(self.config)
        config.seed = seed
        return self._make_env(config)


def _own(frame: Any) -> np.ndarray:
    """A contiguous copy. dm_control renders flipped views with negative
    strides (which torch rejects) that may also alias a reused render buffer.
    """
    return np.array(frame, dtype=np.uint8, copy=True, order="C")


def _close(env: Any) -> None:
    try:
        env.close()
    except Exception:  # noqa: BLE001, S110  (closing is best effort: some wrapped envs have no close)
        pass


def _autocast(device_type: str, dtype: Any) -> Any:
    import torch

    return torch.autocast(device_type=device_type, dtype=dtype, enabled=dtype is not None)


def _observe_core(wm: Any, image: Any, prev_actions: Any) -> tuple[Any, Any]:
    """Runs the context frames through the encoder + RSSM posterior and
    returns the latent after the last frame as (stoch, deter).

    This re-implements the loop in the repo's `RSSM.observe`/`obs_step` instead
    of calling it, because `obs_step` branches on tensor values
    (`if torch.sum(is_first) == len(is_first)`), which forces a host sync and
    makes it impossible to capture in a CUDA graph. Here `is_first` is known
    statically (only t=0), so the branch becomes a plain Python `if t == 0`.
    Same ops in the same order, so the RNG stream — and the output — match the
    repo's version exactly (checked against saved baseline results).

    Uses RSSM's private `_obs_out_layers` / `_suff_stats_layer`, which is fine
    for the pinned clone but would need revisiting if the repo changes.
    """
    import torch

    dyn = wm.dynamics
    embed = wm.encoder({"image": image})  # (B, C, E)
    batch, n_ctx = embed.shape[:2]
    posterior = None
    for t in range(n_ctx):
        if t == 0:  # is_first: reset to the learned initial state, zero previous action
            prev_state = dyn.initial(batch)
            prev_action = torch.zeros_like(prev_actions[:, 0])
        else:
            prev_state, prev_action = posterior, prev_actions[:, t]
        prior = dyn.img_step(prev_state, prev_action)
        x = dyn._obs_out_layers(torch.cat([prior["deter"], embed[:, t]], -1))
        stats = dyn._suff_stats_layer("obs", x)
        stoch = dyn.get_dist(stats).sample()
        posterior = {"stoch": stoch, "deter": prior["deter"], **stats}
    return posterior["stoch"], posterior["deter"]


def _keyframe_positions(n: int, stride: int, device: Any) -> Any:
    """Frame indices that get decoded when decoding every `stride`-th frame:
    0, stride, 2*stride, ... plus the last frame, so interpolation never has to
    extrapolate past the end. Built with device-side ops only (arange/full) —
    uploading a Python list would be a host-to-device copy, which CUDA-graph
    capture forbids.
    """
    import torch

    keys = torch.arange(0, n, stride, device=device)
    if (n - 1) % stride != 0:
        keys = torch.cat([keys, torch.full((1,), n - 1, device=device, dtype=keys.dtype)])
    return keys


def _interpolate_keyframes(decoded_keys: Any, n: int, stride: int) -> Any:
    """Linearly interpolates decoded key frames (B, K, ...) back to n frames
    (B, n, ...), exact at the key frames themselves. Pure tensor ops (searchsorted
    and gathers), so it is capture-safe.
    """
    import torch

    keys = _keyframe_positions(n, stride, decoded_keys.device)
    t = torch.arange(n, device=decoded_keys.device)
    hi = torch.searchsorted(keys, t)  # first key >= t
    lo = (hi - 1).clamp(min=0)
    span = (keys[hi] - keys[lo]).clamp(min=1)  # hi == lo only at t == 0, where the weight is 0
    w = ((t - keys[lo]).to(decoded_keys.dtype) / span.to(decoded_keys.dtype)).view(
        1, n, *([1] * (decoded_keys.dim() - 2))
    )
    return decoded_keys[:, lo] * (1.0 - w) + decoded_keys[:, hi] * w


def _decode_features(wm: Any, features: Any, decode_backend: Any = None) -> Any:
    """Latent features (B, T, F) -> float images (B, T, H, W, C) in [0, 1]."""
    if decode_backend is not None:  # a swapped-in engine (e.g. TensorRT)
        return decode_backend(wm, features)
    return wm.heads["decoder"](features)["image"].mode()


def _imagine_features(wm: Any, stoch: Any, deter: Any, future_actions: Any, lean: bool, backend: Any = None) -> Any:
    """Imagines forward under `future_actions` (B, N, A) and returns the latent features (B, N, F), before any decoding.
    The imagination half of `_imagine_decode`, shared with `generate_latents` (consumers that never look at pixels)."""
    import torch

    dyn = wm.dynamics
    if backend is not None:  # a swapped-in engine (TensorRT, Gumbel-torch) runs the whole loop
        prior = backend(wm, stoch, deter, future_actions)
    elif lean:
        state = {"stoch": stoch, "deter": deter}
        stochs, deters = [], []
        for t in range(future_actions.shape[1]):
            state = dyn.img_step(state, future_actions[:, t])
            stochs.append(state["stoch"])
            deters.append(state["deter"])
        prior = {"stoch": torch.stack(stochs, 1), "deter": torch.stack(deters, 1)}
    else:
        prior = dyn.imagine_with_action(future_actions, {"stoch": stoch, "deter": deter})
    return dyn.get_feat(prior)


def _imagine_decode(
    wm: Any,
    stoch: Any,
    deter: Any,
    future_actions: Any,
    lean: bool,
    backend: Any = None,
    decode_stride: int = 1,
    decode_backend: Any = None,
) -> Any:
    """Imagines forward under `future_actions` (B, N, A) from a latent state
    and decodes to uint8 frames (B, N, H, W, 3).

    The default path calls the repo's `imagine_with_action`, whose `static_scan`
    grows every output with `torch.cat` each step (quadratic copying) and keeps
    stats (mean/std/logit) nothing downstream reads. `lean=True` does the same
    steps but collects only stoch/deter and stacks once — identical numerics.
    `backend`, if given, replaces the loop entirely (and takes precedence over
    `lean`); see models.base.HasImagineBackend.

    `decode_stride > 1` decodes only every stride-th frame (plus the last) and
    interpolates the rest — cheaper decoding for some visual error.
    `decode_backend` replaces the decoder call (e.g. with a TensorRT engine).
    """
    import torch

    features = _imagine_features(wm, stoch, deter, future_actions, lean, backend)
    n = features.shape[1]
    if decode_stride > 1 and n > 1:
        keys = _keyframe_positions(n, decode_stride, features.device)
        decoded = _interpolate_keyframes(_decode_features(wm, features[:, keys], decode_backend), n, decode_stride)
    else:
        decoded = _decode_features(wm, features, decode_backend)
    # .float() first: under autocast the decoder can return bf16, whose 8-bit
    # mantissa would quantize pixel values by whole grey levels.
    return (decoded.float().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


def synthetic_request(
    horizon: float,
    steps_per_second: float,
    n_actions: int,
    frame_shape: tuple[int, ...] = (64, 64, 3),
    context_steps: int = 5,
) -> dict[str, Any]:
    """A zero-valued request with the right shapes for `horizon`, for warming up
    (capturing graphs, building engines) without needing a simulator.
    """
    steps = horizon_to_steps(horizon, steps_per_second)
    return {
        "init_video": [np.zeros(frame_shape, dtype=np.uint8) for _ in range(context_steps)],
        "context_actions": [np.zeros(n_actions, dtype=np.float32) for _ in range(context_steps - 1)],
        "actions": [np.zeros(n_actions, dtype=np.float32) for _ in range(steps)],
        "horizon": horizon,
    }


class DreamerWorldModel(WorldModelInterface):
    """The trained DreamerV3 world model, rolled out by imagination.

    `generate()` needs `init_video` (context frames) and `actions` (one per
    rollout step); `horizon` must agree with `len(actions)` via the repo's
    steps_per_second. Build matching inputs + ground truth with
    `DreamerSimScenarios`.
    """

    def __init__(self, repo: DreamerRepo, checkpoint_dir: str | pathlib.Path):
        self._repo = repo
        self._checkpoint = pathlib.Path(checkpoint_dir).expanduser() / "latest.pt"
        self._wm = None
        self._rollout_fn = None  # built once at load: executors cache on its identity
        self._latents_fn = None  # the same without the decoder (generate_latents)
        self._chunk_fns = None  # (observe, chunk, decode, noise) functions, built at load
        # Extension points that optimization modules set (see models.base):
        self.tensor_executor = eager_executor  # HasTensorExecutor: how the rollout runs (e.g. CUDA graphs)
        self.autocast_dtype = None  # HasAutocast: torch dtype to autocast the rollout in, or None
        self.lean_scan = False  # HasLeanScan: use the collect-and-stack imagination loop
        self.imagine_backend = None  # HasImagineBackend: a replacement imagination loop (e.g. TensorRT)
        self.decode_stride = 1  # decode every k-th frame and interpolate the rest (1 = decode all)
        self.decode_backend = None  # a replacement decoder call (e.g. TensorRT)
        self.chunk_steps = None  # HasChunkedRollout: run the rollout in chunks of this many steps (see models/chunked.py)

    def _load(self):
        if self._wm is not None:
            return self._wm

        import torch

        _, models, _ = self._repo.modules()
        config = self._repo.config
        wm = models.WorldModel(
            self._repo.observation_space, self._repo.action_space, 0, config
        ).to(config.device)
        # weights_only=False: the checkpoint also holds optimizer state and is
        # one you trained yourself, so it's trusted; the repo itself loads it
        # the same way.
        checkpoint = torch.load(self._checkpoint, map_location=config.device, weights_only=False)
        wm.load_state_dict(extract_world_model_state(checkpoint["agent_state_dict"]))
        wm.requires_grad_(False)
        wm.eval()
        self._wm = wm
        device_type = torch.device(config.device).type

        def rollout(image: Any, prev_actions: Any, future_actions: Any) -> Any:
            # Everything from context frames to decoded uint8 frames, as one
            # pure tensor function with no host syncs (so it can be graphed).
            # The knobs are read per call, but a captured graph bakes in
            # whatever they were at capture time — set them before first use.
            with _autocast(device_type, self.autocast_dtype):
                stoch, deter = _observe_core(wm, image, prev_actions)
                return _imagine_decode(
                    wm, stoch, deter, future_actions, self.lean_scan, self.imagine_backend,
                    self.decode_stride, self.decode_backend,
                )

        def rollout_latents(image: Any, prev_actions: Any, future_actions: Any) -> Any:
            # Same context phase and imagination as `rollout`, without the decoder: latent features only.
            with _autocast(device_type, self.autocast_dtype):
                stoch, deter = _observe_core(wm, image, prev_actions)
                return _imagine_features(wm, stoch, deter, future_actions, self.lean_scan, self.imagine_backend).float()

        self._rollout_fn = rollout
        self._latents_fn = rollout_latents
        self._chunk_fns = self._build_chunk_fns(wm)
        return wm

    def _build_chunk_fns(self, wm: Any) -> tuple[Any, Any, Any, Any] | None:
        """The observe / chunk / decode / noise functions `models.chunked.run_chunked` needs, built once so
        executors cache on their identity. The latent state travels packed in one float32 tensor
        (flattened stochastic state, then deterministic state), since an executor passes single tensors.
        None for models whose latent is not discrete (the explicit-noise backends need discrete latents)."""
        import torch

        dyn = wm.dynamics
        if not getattr(dyn, "_discrete", 0):
            return None
        stoch_n, classes = int(dyn._stoch), int(dyn._discrete)
        stoch_dim = stoch_n * classes

        def unpack(state: Any) -> tuple[Any, Any]:
            return state[:, :stoch_dim].reshape(-1, stoch_n, classes), state[:, stoch_dim:]

        def pack(stoch: Any, deter: Any) -> Any:
            return torch.cat([stoch.reshape(stoch.shape[0], -1).float(), deter.float()], dim=1)

        def observe_fn(image: Any, prev_actions: Any) -> Any:
            stoch, deter = _observe_core(wm, image, prev_actions)
            return pack(stoch, deter)

        def chunk_fn(state: Any, actions: Any, noise: Any) -> Any:
            stoch, deter = unpack(state)
            prior = self.imagine_backend(wm, stoch, deter, actions, noise)
            features = dyn.get_feat(prior)
            last = pack(prior["stoch"][:, -1], prior["deter"][:, -1])
            return torch.cat([last, features.reshape(state.shape[0], -1).float()], dim=1)

        def decode_fn(features: Any) -> Any:
            decoded = _decode_features(wm, features, self.decode_backend)
            return (decoded.float().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)

        def noise_fn(state: Any, steps: int) -> Any:
            return self.imagine_backend.draw_noise(wm, unpack(state)[0], steps)

        return observe_fn, chunk_fn, decode_fn, noise_fn

    def _run_chunked(self, wm: Any, image: Any, prev_actions: Any, futures: Any) -> Any:
        from worldoptbench.models.chunked import run_chunked

        backend = self.imagine_backend
        if backend is None or not hasattr(backend, "draw_noise"):
            raise RuntimeError(
                "chunked rollouts need an imagination backend that takes pre-drawn noise (the gumbel_sampling "
                "or tensorrt module); apply one before chunked_rollout, or turn chunking off"
            )
        if self.decode_stride != 1:
            raise RuntimeError("chunked rollouts decode every frame; sparse decoding (decode_stride > 1) is not supported with them")
        if self.autocast_dtype is not None:
            raise RuntimeError("chunked rollouts pack the latent state in float32; run them without autocast")
        if self._chunk_fns is None:
            raise RuntimeError("this world model's latent is not discrete, so it cannot be run in chunks")
        observe_fn, chunk_fn, decode_fn, noise_fn = self._chunk_fns
        return run_chunked(
            self.tensor_executor, observe_fn, chunk_fn, decode_fn, noise_fn,
            image, prev_actions, futures, int(self.chunk_steps),
        )

    def torch_module(self):
        """The loaded WorldModel (satisfies models.base.TorchBacked), so
        modules like quantization can modify it in place. Loads the
        checkpoint if that hasn't happened yet.
        """
        return self._load()

    def generate(
        self,
        prompt: str | None = None,
        init_frame: Any | None = None,
        init_video: Any | None = None,
        actions: list[Any] | None = None,
        horizon: float = 4.0,
        **kwargs: Any,
    ) -> Rollout:
        """
        Args:
            prompt: ignored apart from being recorded (Dreamer isn't text-conditioned).
            init_video: context frames, a sequence of (H, W, 3) uint8 arrays.
                The model observes these to build its latent state. Required.
            actions: the actions to imagine forward under, one per rollout
                step, each a vector in [-1, 1]. Required.
            horizon: rollout length in seconds; must equal len(actions) steps.
            context_actions (kwarg): the C-1 actions taken between the C
                context frames. If omitted, zeros are used, which degrades
                the inferred latent state — pass them whenever you have them.
            seed (kwarg): seeds the latent sampling, default 0. Required for
                meaningful comparisons: Dreamer samples its latent state
                randomly, and on a real run two unseeded generations from
                *identical weights* differed by ~12/255 pixels and their
                fidelity scores ranged 0.06-0.27 — larger than any
                optimization's effect. Same seed => essentially identical output
                (rare +-1/255 pixel jitter from non-deterministic GPU kernels
                still appears, even between identical runs).

        Returns:
            A Rollout of len(actions) uint8 frames — the imagined
            continuation only, not the context.
        """
        if prompt is None and init_frame is None and init_video is None:
            raise ValueError("generate() needs at least one of prompt, init_frame, init_video")
        if init_video is None:
            raise ValueError(
                "DreamerWorldModel needs init_video (context frames) — use "
                "DreamerSimScenarios to build context frames + actions"
            )
        if actions is None:
            raise ValueError("DreamerWorldModel needs `actions` to imagine under")

        context_actions = kwargs.pop("context_actions", None)
        seed = int(kwargs.pop("seed", 0))
        context, prev_actions, future, n_steps = self._prepare(init_video, actions, context_actions, horizon)

        frames = self._run([context], [prev_actions], [future], seed)[0]

        return Rollout(
            frames=list(frames),
            fps=self._repo.steps_per_second,
            metadata={
                "prompt": prompt,
                "horizon": horizon,
                "model": self.get_info().name,
                "n_steps": n_steps,
            },
        )

    def _prepare(self, init_video, actions, context_actions, horizon):
        """Validates one request and turns it into numpy arrays:
        (context uint8 (C, H, W, 3), prev_actions float32 (C, A), future float32 (N, A), n_steps).
        """
        n_steps = horizon_to_steps(horizon, self._repo.steps_per_second)
        if len(actions) != n_steps:
            raise ValueError(
                f"horizon={horizon}s is {n_steps} steps at {self._repo.steps_per_second} "
                f"steps/s but {len(actions)} actions were given"
            )
        n_ctx = len(init_video)
        n_act = self._repo.config.num_actions
        context = np.stack([np.asarray(f, dtype=np.uint8) for f in init_video])
        # Dreamer convention: action[t] is the action that LED TO obs[t], and
        # is zeros at t=0 (is_first).
        prev_actions = np.zeros((n_ctx, n_act), dtype=np.float32)
        if context_actions is not None:
            if len(context_actions) != n_ctx - 1:
                raise ValueError(
                    f"context_actions must have {n_ctx - 1} entries for {n_ctx} context "
                    f"frames, got {len(context_actions)}"
                )
            prev_actions[1:] = np.asarray(context_actions, dtype=np.float32)
        future = np.asarray(actions, dtype=np.float32)
        return context, prev_actions, future, n_steps

    def _run(self, contexts, prev_actions, futures, seed, latents: bool = False):
        """Runs B same-shaped prepared requests as one batch; returns uint8 frames (B, N, H, W, 3), or with
        `latents=True` the float latent features (B, N, F) of the decoder-free rollout."""
        import torch

        wm = self._load()
        device = torch.device(self._repo.config.device)
        rng_devices = [device.index or 0] if device.type == "cuda" else []

        def to_device(arrays: Any) -> Any:
            return torch.tensor(np.asarray(np.stack(arrays), dtype=np.float32), device=device)

        image = to_device(contexts) / 255.0
        # fork_rng restores the caller's global RNG state afterwards, so seeding
        # here doesn't leak into anything else in the process.
        with torch.no_grad(), torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(seed)
            if latents:
                frames = self.tensor_executor(self._latents_fn, image, to_device(prev_actions), to_device(futures))
            elif self.chunk_steps:
                frames = self._run_chunked(wm, image, to_device(prev_actions), to_device(futures))
            else:
                frames = self.tensor_executor(
                    self._rollout_fn, image, to_device(prev_actions), to_device(futures)
                )
            return frames.cpu().numpy()  # also the sync point that ends the timed work

    def generate_latents(
        self, init_video: Any, actions: Any, horizon: float, context_actions: Any | None = None, seed: int = 0
    ) -> np.ndarray:
        """The imagined rollout as latent features (N, F), skipping the decoder: for consumers that never look at pixels (a
        planner or policy evaluator, whose reward and value heads read the latents). Same inputs and seeding as `generate`, so
        the same seed gives the latents that `generate` decodes, and every optimization that changes the dynamics (weights,
        precision, sampler, TensorRT step, CUDA graphs) applies unchanged; the decoder-side ones (sparse and TensorRT decoding)
        have nothing to act on. Chunked rollouts are not supported here."""
        context, prev, future, _ = self._prepare(init_video, actions, context_actions, horizon)
        self._load()
        return self._run([context], [prev], [future], int(seed), latents=True)[0]

    def generate_batch(self, requests: Sequence[dict[str, Any]], seed: int = 0) -> list[Rollout]:
        """Runs several rollouts in one pass (one batched forward), for throughput.

        Each request is a dict with `init_video`, `actions`, `horizon` and optionally
        `context_actions`, exactly as for `generate()`. All requests must have the
        same shapes (same number of context frames, steps, and frame size) — group
        mixed traffic first with `worldoptbench.scheduling.bucket_requests`.

        One `seed` covers the whole batch, so a request's output is NOT the same as
        calling `generate()` on it alone (the latent noise is drawn jointly); it is
        statistically equivalent. The TensorRT step backend is built for batch size 1
        and refuses a larger batch.

        Returns one Rollout per request, in order.
        """
        if not requests:
            return []
        prepared = [
            self._prepare(r["init_video"], r["actions"], r.get("context_actions"), r["horizon"])
            for r in requests
        ]
        shapes = {(c.shape, f.shape) for c, _, f, _ in prepared}
        if len(shapes) != 1:
            raise ValueError(
                "generate_batch needs requests with identical shapes (context frames, steps, "
                f"frame size); got {len(shapes)} different shapes — bucket them first with "
                "worldoptbench.scheduling.bucket_requests"
            )
        frames = self._run([c for c, _, _, _ in prepared], [p for _, p, _, _ in prepared], [f for _, _, f, _ in prepared], int(seed))
        name = self.get_info().name
        return [
            Rollout(
                frames=list(frames[i]),
                fps=self._repo.steps_per_second,
                metadata={"model": name, "horizon": r["horizon"], "n_steps": prepared[i][3], "batch_size": len(requests)},
            )
            for i, r in enumerate(requests)
        ]

    def warmup(self, horizons: Sequence[float], batch_size: int = 1, context_steps: int = 5) -> None:
        """Runs a throwaway rollout at each horizon so graph capture and engine
        builds happen now instead of inside the first request of each shape
        (catalog A12/L8; the benchmark runner already does this for itself).
        """
        frame_shape = tuple(self._repo.observation_space.spaces["image"].shape)
        for horizon in horizons:
            request = synthetic_request(
                horizon, self._repo.steps_per_second, self._repo.config.num_actions, frame_shape, context_steps
            )
            if batch_size == 1:
                self.generate(prompt="warmup", **request)
            else:
                self.generate_batch([request] * batch_size)

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            name=f"DreamerV3-{self._repo.task}",
            architecture="autoregressive",
            # 0 until the checkpoint has been loaded (get_info shouldn't force a load).
            param_count=(
                sum(p.numel() for p in self._wm.parameters()) if self._wm is not None else 0
            ),
            supported_optimizations=[],
            supports_image_conditioning=False,
            supports_video_conditioning=True,
        )


class DreamerSimScenarios:
    """ScenarioFn for DreamerWorldModel: steps the real simulator to produce
    context frames, an action sequence, and the *true* continuation under
    those same actions (the ground truth for PSNR/SSIM and sim fidelity).

    Actions are random in [-1, 1], seeded by the entry seed alone, so (a) a
    baseline run and an optimized run see identical scenarios, and (b) the
    horizons of one entry are nested: the 2 s scenario is the first part of
    the 20 s one (same start state, same actions, same ground truth). That
    makes a score-vs-horizon drift curve measure drift, instead of mixing in
    the variance between unrelated trajectories. Random actions
    mostly flail the walker around rather than doing anything purposeful —
    fine for stressing the dynamics model, but not representative of a trained
    policy's state distribution; a policy-driven variant would need the actor.

    Pass as `run_benchmark(..., scenario_fn=DreamerSimScenarios(repo))` with
    the prompt set in prompts/dreamer_dmc_set.json (entries need a "seed").
    """

    def __init__(self, repo: DreamerRepo, context_steps: int = 5):
        if context_steps < 1:
            raise ValueError("context_steps must be >= 1")
        self._repo = repo
        self._context_steps = context_steps

    def __call__(self, entry: dict[str, Any], horizon: float) -> Scenario:
        seed = int(entry.get("seed", 0))
        n_steps = horizon_to_steps(horizon, self._repo.steps_per_second)
        env_steps = (self._context_steps - 1) + n_steps
        if env_steps > self._repo.max_steps:
            raise ValueError(
                f"horizon={horizon}s needs {env_steps} env steps but an episode is only "
                f"{self._repo.max_steps} steps — pick shorter horizons"
            )

        env = self._repo.make_env(seed)
        if hasattr(env.action_space, "n"):
            _close(env)
            raise NotImplementedError("DreamerSimScenarios supports continuous actions only")
        n_act = self._repo.config.num_actions
        rng = np.random.default_rng(seed)

        try:
            obs = env.reset()
            context_frames = [_own(obs["image"])]
            context_actions: list[np.ndarray] = []
            future_actions: list[np.ndarray] = []
            reference_frames: list[np.ndarray] = []

            for step in range(env_steps):
                action = rng.uniform(-1.0, 1.0, size=n_act).astype(np.float32)
                obs, _reward, done, _info = env.step({"action": action})
                if step < self._context_steps - 1:
                    context_actions.append(action)
                    context_frames.append(_own(obs["image"]))
                else:
                    future_actions.append(action)
                    reference_frames.append(_own(obs["image"]))
                if done and step < env_steps - 1:
                    raise RuntimeError(f"Simulator episode ended early at step {step + 1}")
        finally:
            _close(env)

        return Scenario(
            generate_kwargs={
                "init_video": context_frames,
                "actions": future_actions,
                "context_actions": context_actions,
                # Same entry seed => same latent sampling, so a baseline run
                # and an optimized run are directly comparable.
                "seed": seed,
            },
            reference_frames=reference_frames,
            baseline_frame=context_frames[-1],
        )


class DreamerReplayScenarios:
    """ScenarioFn that reads scenarios from stored episodes instead of stepping a
    simulator: context frames, actions, and the true continuation all come from a
    recorded episode (dreamerv3-torch's `.npz` files with `image` and `action`).

    Why this exists: `DreamerSimScenarios` drives the simulator with *random*
    actions, and on a real run that turned out to measure something other than what
    matters. A flailing walker falls over and lies still, which is easy to predict
    (skill ~0.33-0.38 against a freeze-the-scene baseline), whereas on held-out
    episodes of the competent trained policy walking, the same models scored 0.04-0.09
    at 2 s and ~0.02 at 5 s. Competent behaviour is what a world model is used for, so
    that is what to score.

    Use *held-out* episodes: the `eval_eps` directory the repo writes during training
    holds episodes the model was never trained on (they never enter the training
    replay buffer), whereas `train_eps` episodes are data it has already fit.

    Selection is deterministic in the entry's `seed`, and independent of the
    horizon, so the horizons of one entry are nested prefixes of the same window
    (exactly like the simulator scenarios): episode = eligible[seed % n], start =
    a seed-derived offset that leaves room for the longest horizon you intend to use
    (`max_horizon`, seconds).
    """

    def __init__(
        self,
        episodes_dir: str | pathlib.Path,
        steps_per_second: float = 20.0,
        context_steps: int = 5,
        max_horizon: float = 5.0,
        min_return: float | None = None,
    ):
        if context_steps < 1:
            raise ValueError("context_steps must be >= 1")
        self._steps_per_second = steps_per_second
        self._context_steps = context_steps
        self._max_steps = horizon_to_steps(max_horizon, steps_per_second)
        self._max_horizon = max_horizon

        paths = sorted(pathlib.Path(episodes_dir).expanduser().glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"no .npz episodes in {episodes_dir}")
        if min_return is not None:
            kept = []
            for path in paths:
                with np.load(path) as ep:
                    if float(np.sum(ep["reward"])) >= min_return:
                        kept.append(path)
            paths = kept
            if not paths:
                raise ValueError(f"no episode in {episodes_dir} has return >= {min_return}")
        self._paths = paths

    @property
    def num_episodes(self) -> int:
        return len(self._paths)

    def __call__(self, entry: dict[str, Any], horizon: float) -> Scenario:
        if horizon > self._max_horizon:
            raise ValueError(
                f"horizon={horizon}s exceeds max_horizon={self._max_horizon}s this provider was built for"
            )
        seed = int(entry.get("seed", 0))
        n_steps = horizon_to_steps(horizon, self._steps_per_second)
        path = self._paths[seed % len(self._paths)]
        with np.load(path) as ep:
            images, actions = ep["image"], ep["action"]

        need = self._context_steps + self._max_steps  # the window must fit the *longest* horizon, so shorter ones nest inside it
        slack = len(images) - need
        if slack < 0:
            raise ValueError(f"{path.name} has {len(images)} frames; need {need} for max_horizon={self._max_horizon}s")
        start = (seed * 37) % (slack + 1)

        c = self._context_steps
        context = [np.array(images[start + i], dtype=np.uint8, copy=True) for i in range(c)]
        # Dreamer convention: actions[t] is the action that led TO frame t. The context actions are the
        # c-1 that produced frames 1..c-1 of the context; the future actions are those leading to the
        # frames the model must predict.
        context_actions = [np.array(actions[start + i], dtype=np.float32) for i in range(1, c)]
        future_actions = [np.array(actions[start + c + i], dtype=np.float32) for i in range(n_steps)]
        reference = [np.array(images[start + c + i], dtype=np.uint8, copy=True) for i in range(n_steps)]
        return Scenario(
            generate_kwargs={
                "init_video": context,
                "actions": future_actions,
                "context_actions": context_actions,
                "seed": seed,
            },
            reference_frames=reference,
            baseline_frame=context[-1],
        )
