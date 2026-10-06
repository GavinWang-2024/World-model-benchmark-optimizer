"""Policies for the Dreamer walker, and their returns in the real simulator and inside a (possibly optimized) world model.

This is the machinery for the functional-utility check (`worldoptbench/utility.py`, outline section 12-D): do an optimized
world model's imagined returns still rank a set of policies the way the original model's do, and the way the simulator does?

Policies are variants of the trained actor, so they all work on the same latent features of one world model (an actor from
a different checkpoint reads a different latent space and cannot be evaluated in this model). They span good to terrible
play, so the ranking has real structure:

  actor            the trained actor's mode (the best the checkpoint can do)
  noise_*          the mode plus Gaussian action noise of the given standard deviation (clipped to [-1, 1])
  half             the mode scaled by 0.5
  hold2            a new action only every second step, repeating it in between
  inverted         the negated mode (should fall over)
  zero / random    no action / uniform random actions

*True* return: run the policy in the real simulator with the baseline world model doing the state estimation (encoder and
posterior), exactly as the agent acts; it defines the ground-truth ranking. *Imagined* return: from start states observed in
held-out episodes, roll the policy forward inside the world model and sum the reward head's prediction. The optimized models
differ only in how that imagination is computed (weights, precision, sampler, TensorRT step).

Uses the repo's private pieces (`ImagBehavior.actor`, `RSSM.obs_step`/`img_step`, the reward head), pinned like the rest of
dreamer.py.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PolicySpec:
    name: str
    kind: str  # actor | noise | scale | hold | inverted | zero | random
    value: float = 0.0


DEFAULT_POLICIES: tuple[PolicySpec, ...] = (
    PolicySpec("actor", "actor"),
    PolicySpec("noise_0.2", "noise", 0.2),
    PolicySpec("noise_0.5", "noise", 0.5),
    PolicySpec("noise_1.0", "noise", 1.0),
    PolicySpec("half", "scale", 0.5),
    PolicySpec("hold2", "hold", 2),
    PolicySpec("inverted", "inverted"),
    PolicySpec("zero", "zero"),
    PolicySpec("random", "random"),
)


# A harder test: nine versions of the same policy that differ only in how much action noise is added, so their true returns
# are close together and ranking them takes a model that resolves small differences, not just good play from terrible play.
FINE_POLICIES: tuple[PolicySpec, ...] = tuple(
    PolicySpec("actor" if sigma == 0.0 else f"noise_{sigma:g}", "actor" if sigma == 0.0 else "noise", sigma)
    for sigma in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
)

POLICY_SETS = {"coarse": DEFAULT_POLICIES, "fine": FINE_POLICIES}


def load_actor(repo: Any, checkpoint_dir: str | pathlib.Path, wm: Any) -> Any:
    """The trained actor network from a dreamerv3-torch checkpoint, in eval mode with gradients off."""
    import torch  # noqa: PLC0415

    _, models, _ = repo.modules()
    config = repo.config
    behavior = models.ImagBehavior(config, wm).to(config.device)
    checkpoint = torch.load(pathlib.Path(checkpoint_dir).expanduser() / "latest.pt", map_location=config.device, weights_only=False)
    prefix = "_task_behavior.actor."
    state = {
        key.replace("_orig_mod.", "")[len(prefix):]: value
        for key, value in checkpoint["agent_state_dict"].items()
        if key.replace("_orig_mod.", "").startswith(prefix)
    }
    if not state:
        raise KeyError("no actor weights (`_task_behavior.actor.*`) in the checkpoint")
    behavior.actor.load_state_dict(state)
    behavior.actor.eval().requires_grad_(False)
    return behavior.actor


class PolicyActor:
    """Turns latent features into actions for one PolicySpec. Stateful (the hold policy keeps its last action) and seeded,
    so the same seed gives the same random draws whatever world model it is run in."""

    def __init__(self, actor: Any, spec: PolicySpec, num_actions: int, device: Any, seed: int):
        import torch  # noqa: PLC0415

        self._actor, self.spec, self._n, self._device = actor, spec, num_actions, torch.device(device)
        self._generator = torch.Generator(device=self._device).manual_seed(int(seed))
        self.reset()

    def reset(self) -> None:
        self._step, self._held = 0, None

    def act(self, feat: Any) -> Any:
        """`feat`: (1, F) latent features -> (1, A) action in [-1, 1]."""
        import torch  # noqa: PLC0415

        kind, value = self.spec.kind, self.spec.value
        step = self._step
        self._step += 1
        if kind == "zero":
            return torch.zeros(1, self._n, device=self._device)
        if kind == "random":
            return torch.rand(1, self._n, device=self._device, generator=self._generator) * 2.0 - 1.0
        mode = self._actor(feat.float()).mode()
        if kind == "actor":
            return mode
        if kind == "inverted":
            return -mode
        if kind == "scale":
            return (value * mode).clamp(-1.0, 1.0)
        if kind == "noise":
            noise = torch.randn(1, self._n, device=self._device, generator=self._generator)
            return (mode + value * noise).clamp(-1.0, 1.0)
        if kind == "hold":
            if step % int(value) == 0 or self._held is None:
                self._held = mode
            return self._held
        raise ValueError(f"unknown policy kind {kind!r}")


def true_returns(
    repo: Any, wm: Any, actor: Any, specs: Sequence[PolicySpec], episodes: int, seed: int = 0, max_steps: int | None = None
) -> dict[str, list[float]]:
    """Episode returns of each policy in the real simulator. The baseline world model does the state estimation
    (encoder, then the RSSM posterior), as the agent does when it acts; `episodes` episodes per policy, each on its own seed
    (the same seeds for every policy)."""
    import torch  # noqa: PLC0415

    from worldoptbench.models.dreamer import _close, _own  # noqa: PLC0415

    config = repo.config
    device = torch.device(config.device)
    dyn = wm.dynamics
    limit = max_steps or repo.max_steps
    out: dict[str, list[float]] = {}
    for spec in specs:
        returns = []
        for episode in range(episodes):
            env = repo.make_env(seed + 1000 * episode)
            policy = PolicyActor(actor, spec, config.num_actions, device, seed + episode)
            total, latent, action = 0.0, None, None
            try:
                obs = env.reset()
                for _ in range(limit):
                    batch = {
                        "image": _own(obs["image"])[None],  # the simulator's frames have negative strides; torch rejects those
                        "is_first": np.array([obs["is_first"]]),
                        "is_terminal": np.array([obs["is_terminal"]]),
                    }
                    with torch.no_grad():
                        processed = wm.preprocess(batch)
                        embed = wm.encoder(processed)
                        latent, _ = dyn.obs_step(latent, action, embed, processed["is_first"])
                        action = policy.act(dyn.get_feat(latent))
                    obs, reward, done, _info = env.step({"action": action[0].cpu().numpy().astype(np.float32)})
                    total += float(reward)
                    if done:
                        break
            finally:
                _close(env)
            returns.append(total)
        out[spec.name] = returns
    return out


def load_starts(
    episodes_dir: str | pathlib.Path, count: int, context_steps: int = 5, min_return: float | None = None, seed: int = 0
) -> list[dict[str, Any]]:
    """`count` start states for imagination: `context_steps` consecutive frames (and the actions between them) taken at random
    positions of recorded episodes (dreamerv3-torch .npz files), optionally only from episodes whose return is at least
    `min_return`. Returns dicts with `init_video` (C, H, W, 3 uint8) and `context_actions` (C-1, A)."""
    files = sorted(pathlib.Path(episodes_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz episodes in {episodes_dir}")
    episodes = []
    for path in files:
        with np.load(path) as data:
            reward = data["reward"] if "reward" in data else None
            if min_return is not None and (reward is None or float(reward.sum()) < min_return):
                continue
            if len(data["image"]) > context_steps + 1:
                episodes.append((np.asarray(data["image"]), np.asarray(data["action"])))
    if not episodes:
        raise ValueError("no episodes passed the filters")
    rng = np.random.default_rng(seed)
    starts = []
    for _ in range(count):
        images, actions = episodes[int(rng.integers(len(episodes)))]
        t = int(rng.integers(0, len(images) - context_steps))
        starts.append({
            "init_video": images[t : t + context_steps],
            "context_actions": actions[t + 1 : t + context_steps].astype(np.float32),  # action[t] led to obs[t]
        })
    return starts


def imagined_returns(
    model: Any, actor: Any, specs: Sequence[PolicySpec], starts: Sequence[dict[str, Any]], horizon: int, seed: int = 0
) -> dict[str, list[float]]:
    """For each policy, the summed reward the (possibly optimized) world model predicts over `horizon` imagined steps from each
    start state. `model` is a DreamerWorldModel with its optimization stack applied: the weights are whatever the stack left,
    `autocast_dtype` is honoured, and a set `imagine_backend` (Gumbel sampler, TensorRT) does each imagination step."""
    import torch  # noqa: PLC0415

    from worldoptbench.models.dreamer import _autocast, _observe_core  # noqa: PLC0415

    wm = model._load()
    config = model._repo.config
    device = torch.device(config.device)
    dyn = wm.dynamics
    backend, dtype = model.imagine_backend, model.autocast_dtype
    out: dict[str, list[float]] = {spec.name: [] for spec in specs}
    n_actions = config.num_actions

    with torch.no_grad():
        for index, start in enumerate(starts):
            image = torch.tensor(np.asarray(start["init_video"], dtype=np.float32), device=device)[None] / 255.0
            prev = np.zeros((len(start["init_video"]), n_actions), dtype=np.float32)
            prev[1:] = start["context_actions"]
            prev = torch.tensor(prev, device=device)[None]
            with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
                torch.manual_seed(seed + 7919 * index)
                with _autocast(device.type, dtype):
                    stoch0, deter0 = _observe_core(wm, image, prev)
            for p, spec in enumerate(specs):
                policy = PolicyActor(actor, spec, n_actions, device, seed + 31 * index + p)
                stoch, deter = stoch0.clone(), deter0.clone()
                total = torch.zeros((), device=device)
                with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
                    torch.manual_seed(seed + 104729 * index + 13 * p)
                    for _ in range(horizon):
                        with _autocast(device.type, dtype):
                            action = policy.act(dyn.get_feat({"stoch": stoch, "deter": deter}))
                            if backend is not None:
                                prior = backend(wm, stoch, deter, action[:, None, :])
                                stoch, deter = prior["stoch"][:, 0], prior["deter"][:, 0]
                            else:
                                state = dyn.img_step({"stoch": stoch, "deter": deter}, action)
                                stoch, deter = state["stoch"], state["deter"]
                            reward = wm.heads["reward"](dyn.get_feat({"stoch": stoch, "deter": deter})).mode()
                        total = total + reward.float().reshape(-1)[0]
                out[spec.name].append(float(total.item()))
    return out
