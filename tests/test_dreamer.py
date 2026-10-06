"""DreamerWorldModel / DreamerSimScenarios — the parts that don't need torch
or a dreamerv3-torch clone: pure helpers, argument validation, and the
scenario builder driven by a fake simulator. The torch rollout itself and the
repo's config/env loading are only exercised by a real run.
"""

import types

import numpy as np
import pytest

from worldoptbench.models.dreamer import (
    DreamerSimScenarios,
    DreamerWorldModel,
    extract_world_model_state,
    horizon_to_steps,
    overrides_to_argv,
)

# ---- pure helpers ---------------------------------------------------------


def test_horizon_to_steps_rounds_and_has_a_floor_of_one():
    assert horizon_to_steps(2, 20.0) == 40
    assert horizon_to_steps(0.51, 10.0) == 5
    assert horizon_to_steps(0.0, 20.0) == 1


def test_overrides_to_argv_stringifies_scalars():
    argv = overrides_to_argv({"task": "dmc_walker_walk", "compile": False, "seed": 3})
    assert argv == ["--task", "dmc_walker_walk", "--compile", "False", "--seed", "3"]


def test_overrides_to_argv_rejects_non_scalars():
    with pytest.raises(TypeError):
        overrides_to_argv({"size": [64, 64]})


def test_extract_world_model_state_strips_prefix_and_compile_segment():
    state = extract_world_model_state(
        {
            "_wm.encoder.w": 1,
            "_wm._orig_mod.dynamics.w": 2,
            "_task_behavior.actor.w": 3,  # actor weights are not needed
        }
    )
    assert state == {"encoder.w": 1, "dynamics.w": 2}


def test_extract_world_model_state_rejects_foreign_checkpoints():
    with pytest.raises(KeyError):
        extract_world_model_state({"something_else.w": 1})


# ---- generate() argument validation (all fails before torch is needed) -----


def _model():
    repo = types.SimpleNamespace(task="dmc_walker_walk", steps_per_second=20.0)
    return DreamerWorldModel(repo, checkpoint_dir=".")


def test_generate_requires_some_conditioning():
    with pytest.raises(ValueError):
        _model().generate()


def test_generate_requires_context_video_and_actions():
    with pytest.raises(ValueError, match="init_video"):
        _model().generate(prompt="dmc_walker_walk", horizon=1)
    with pytest.raises(ValueError, match="actions"):
        _model().generate(init_video=[np.zeros((64, 64, 3), dtype=np.uint8)], horizon=1)


def test_generate_rejects_actions_that_disagree_with_horizon():
    with pytest.raises(ValueError, match="actions were given"):
        _model().generate(
            init_video=[np.zeros((64, 64, 3), dtype=np.uint8)],
            actions=[np.zeros(6)] * 3,
            horizon=1,  # 20 steps, not 3
        )


def test_get_info_is_autoregressive_and_does_not_load_the_model():
    info = _model().get_info()
    assert info.architecture == "autoregressive"
    assert info.name == "DreamerV3-dmc_walker_walk"
    assert info.param_count == 0  # not loaded yet
    assert info.supports_video_conditioning and not info.supports_image_conditioning


# ---- DreamerSimScenarios against a fake simulator ---------------------------


class _FakeEnv:
    """Image value = number of steps taken so far, so tests can tell frames apart."""

    def __init__(self, n_act=6, done_at=None):
        self.action_space = types.SimpleNamespace(shape=(n_act,))
        self.steps = 0
        self.done_at = done_at
        self.closed = False
        self.actions_seen = []

    def reset(self):
        self.steps = 0
        return {"image": np.full((2, 2, 3), 0, dtype=np.uint8)}

    def step(self, act):
        self.steps += 1
        self.actions_seen.append(act["action"])
        done = self.done_at is not None and self.steps >= self.done_at
        return {"image": np.full((2, 2, 3), self.steps, dtype=np.uint8)}, 0.0, done, {}

    def close(self):
        self.closed = True


class _FakeRepo:
    steps_per_second = 2.0
    max_steps = 20
    config = types.SimpleNamespace(num_actions=6)

    def __init__(self, **env_kwargs):
        self.env_kwargs = env_kwargs
        self.envs = []

    def make_env(self, seed):
        env = _FakeEnv(**self.env_kwargs)
        self.envs.append((seed, env))
        return env


def test_scenario_splits_steps_into_context_and_ground_truth():
    repo = _FakeRepo()
    scenario = DreamerSimScenarios(repo, context_steps=3)({"id": "x", "seed": 7}, horizon=4)

    kwargs = scenario.generate_kwargs
    # 4s * 2 steps/s = 8 future steps; 3 context frames need 2 context actions.
    assert len(kwargs["init_video"]) == 3
    assert len(kwargs["context_actions"]) == 2
    assert len(kwargs["actions"]) == 8
    assert len(scenario.reference_frames) == 8

    # The fake env's image value is the step count: context frames are steps
    # 0,1,2 and the ground truth continues 3..10 — it follows on directly.
    assert [int(f[0, 0, 0]) for f in kwargs["init_video"]] == [0, 1, 2]
    assert [int(f[0, 0, 0]) for f in scenario.reference_frames] == list(range(3, 11))
    # Latent sampling is seeded per (entry seed, horizon), distinctly per scenario.
    other = DreamerSimScenarios(repo, context_steps=3)({"id": "x", "seed": 8}, horizon=4)
    assert kwargs["seed"] != other.generate_kwargs["seed"]
    # The "nothing moves" baseline is the last frame the model is shown.
    assert int(scenario.baseline_frame[0, 0, 0]) == 2

    seed, env = repo.envs[0]
    assert seed == 7 and env.closed
    # Every action the simulator received is the one handed to the model, in order.
    expected = [*kwargs["context_actions"], *kwargs["actions"]]
    assert all(np.array_equal(a, b) for a, b in zip(env.actions_seen, expected))
    assert all(-1.0 <= a.min() and a.max() <= 1.0 for a in kwargs["actions"])


def test_scenario_is_deterministic_per_seed_and_horizon():
    def build(seed):
        scenario = DreamerSimScenarios(_FakeRepo(), context_steps=2)(
            {"id": "x", "seed": seed}, horizon=2
        )
        return scenario.generate_kwargs["actions"]

    same_a, same_b, other = build(1), build(1), build(2)
    assert all(np.array_equal(x, y) for x, y in zip(same_a, same_b))
    assert not np.array_equal(same_a[0], other[0])


def test_scenario_rejects_horizons_longer_than_an_episode():
    with pytest.raises(ValueError, match="episode"):
        DreamerSimScenarios(_FakeRepo(), context_steps=1)({"id": "x"}, horizon=100)


def test_scenario_errors_if_simulator_ends_early_and_still_closes_env():
    repo = _FakeRepo(done_at=2)
    with pytest.raises(RuntimeError, match="ended early"):
        DreamerSimScenarios(repo, context_steps=2)({"id": "x"}, horizon=4)
    assert repo.envs[0][1].closed


def test_scenario_rejects_discrete_action_spaces():
    repo = _FakeRepo()
    repo.make_env = lambda seed: types.SimpleNamespace(
        action_space=types.SimpleNamespace(n=4), close=lambda: None
    )
    with pytest.raises(NotImplementedError):
        DreamerSimScenarios(repo)({"id": "x"}, horizon=1)


def test_horizons_of_one_entry_are_nested_prefixes_of_the_same_trajectory():
    short = DreamerSimScenarios(_FakeRepo(), context_steps=3)({"id": "x", "seed": 4}, horizon=2)
    long = DreamerSimScenarios(_FakeRepo(), context_steps=3)({"id": "x", "seed": 4}, horizon=5)

    n = len(short.reference_frames)
    assert n < len(long.reference_frames)
    # Same context, same actions, same ground truth, same model sampling seed.
    assert all(np.array_equal(a, b) for a, b in zip(short.generate_kwargs["init_video"], long.generate_kwargs["init_video"]))
    assert all(np.array_equal(a, b) for a, b in zip(short.generate_kwargs["actions"], long.generate_kwargs["actions"][:n]))
    assert all(np.array_equal(a, b) for a, b in zip(short.reference_frames, long.reference_frames[:n]))
    assert short.generate_kwargs["seed"] == long.generate_kwargs["seed"]
