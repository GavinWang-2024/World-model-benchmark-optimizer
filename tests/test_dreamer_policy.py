"""Policy variants, start-state loading, and the returns loops, against a tiny fake world model (CPU, no Dreamer repo)."""

import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from worldoptbench.models.dreamer_policy import (  # noqa: E402
    DEFAULT_POLICIES,
    PolicyActor,
    PolicySpec,
    imagined_returns,
    load_starts,
    true_returns,
)

A = 2


class FakeActor:
    """actor(feat).mode() -> a fixed action (0.6, -0.4) whatever the features."""

    def __call__(self, feat):
        return types.SimpleNamespace(mode=lambda: torch.tensor([[0.6, -0.4]]))


def _policy(kind, value=0.0, seed=0):
    return PolicyActor(FakeActor(), PolicySpec("p", kind, value), A, "cpu", seed)


FEAT = torch.zeros(1, 3)


# ---- policy variants -----------------------------------------------------------------------------------


def test_the_default_policy_set_spans_the_kinds_and_has_unique_names():
    names = [s.name for s in DEFAULT_POLICIES]
    assert len(names) == len(set(names)) >= 8
    assert {s.kind for s in DEFAULT_POLICIES} >= {"actor", "noise", "scale", "hold", "inverted", "zero", "random"}


def test_each_policy_kind_produces_the_action_it_describes():
    assert torch.allclose(_policy("actor").act(FEAT), torch.tensor([[0.6, -0.4]]))
    assert torch.allclose(_policy("inverted").act(FEAT), torch.tensor([[-0.6, 0.4]]))
    assert torch.allclose(_policy("scale", 0.5).act(FEAT), torch.tensor([[0.3, -0.2]]))
    assert torch.equal(_policy("zero").act(FEAT), torch.zeros(1, A))


def test_noise_and_random_actions_stay_in_range_and_are_reproducible_per_seed():
    noisy = _policy("noise", 5.0, seed=3)
    values = torch.cat([noisy.act(FEAT) for _ in range(50)])
    assert values.min() >= -1.0 and values.max() <= 1.0  # clipped even at huge noise
    again = _policy("noise", 5.0, seed=3)
    assert torch.equal(torch.cat([again.act(FEAT) for _ in range(50)]), values)  # same seed, same draws
    other = _policy("noise", 5.0, seed=4)
    assert not torch.equal(torch.cat([other.act(FEAT) for _ in range(50)]), values)
    uniform = torch.cat([_policy("random", seed=1).act(FEAT) for _ in range(5)])
    assert uniform.min() >= -1.0 and uniform.max() <= 1.0


def test_hold_repeats_each_action_for_its_period_and_reset_clears_it():
    class Counter:
        def __init__(self):
            self.n = 0

        def __call__(self, feat):
            self.n += 1
            return types.SimpleNamespace(mode=lambda n=self.n: torch.tensor([[float(n), 0.0]]))

    policy = PolicyActor(Counter(), PolicySpec("hold2", "hold", 2), A, "cpu", 0)
    seen = [float(policy.act(FEAT)[0, 0]) for _ in range(6)]
    assert seen[0] == seen[1] and seen[2] == seen[3] and seen[4] == seen[5] and seen[0] != seen[2]
    policy.reset()
    assert float(policy.act(FEAT)[0, 0]) > seen[-1]  # a reset starts a fresh period: a new action immediately, not the old held one


def test_unknown_policy_kind_is_an_error():
    with pytest.raises(ValueError, match="unknown policy kind"):
        _policy("telepathy").act(FEAT)


# ---- start states --------------------------------------------------------------------------------------------


def _write_episode(path, steps, ret):
    # frame t is filled with the value t and action t is (2t, 2t+1), so a start's position can be read back
    np.savez(path, image=np.stack([np.full((4, 4, 3), t, dtype=np.uint8) for t in range(steps)]),
             action=np.array([[2.0 * t, 2.0 * t + 1] for t in range(steps)], dtype=np.float32),
             reward=np.full(steps, ret / steps, dtype=np.float32))


def test_load_starts_cuts_consecutive_frames_with_the_actions_between_them(tmp_path):
    _write_episode(tmp_path / "e.npz", 30, 100.0)
    starts = load_starts(tmp_path, count=5, context_steps=4, seed=0)
    assert len(starts) == 5
    for s in starts:
        assert s["init_video"].shape == (4, 4, 4, 3) and s["context_actions"].shape == (3, A)
        t = int(s["init_video"][0, 0, 0, 0])  # the first frame's value is its index
        assert [int(f[0, 0, 0]) for f in s["init_video"]] == [t, t + 1, t + 2, t + 3]  # consecutive frames
        # action[t] led to obs[t], so the actions between the context frames are action[t+1 .. t+3]
        assert np.array_equal(s["context_actions"], np.array([[2.0 * k, 2.0 * k + 1] for k in (t + 1, t + 2, t + 3)], dtype=np.float32))


def test_load_starts_filters_by_return_and_is_seeded(tmp_path):
    _write_episode(tmp_path / "low.npz", 30, 10.0)
    _write_episode(tmp_path / "high.npz", 30, 900.0)
    a = load_starts(tmp_path, 4, min_return=500.0, seed=1)
    b = load_starts(tmp_path, 4, min_return=500.0, seed=1)
    assert all(np.array_equal(x["init_video"], y["init_video"]) for x, y in zip(a, b))
    with pytest.raises(ValueError, match="passed the filters"):
        load_starts(tmp_path, 1, min_return=5000.0)
    with pytest.raises(FileNotFoundError):
        load_starts(tmp_path / "missing_dir_xyz", 1)


# ---- imagined returns ----------------------------------------------------------------------------------------


class FakeDyn:
    def get_feat(self, state):
        return torch.cat([state["stoch"].reshape(state["stoch"].shape[0], -1), state["deter"]], dim=-1)

    def img_step(self, state, action):
        # deterministic: deter accumulates the action sum; stoch stays
        return {"stoch": state["stoch"], "deter": state["deter"] + action.sum(-1, keepdim=True)}


def _fake_model(backend=None):
    reward = lambda feat: types.SimpleNamespace(mode=lambda: feat[:, -1:])  # reward = the deter value
    wm = types.SimpleNamespace(dynamics=FakeDyn(), heads={"reward": reward})
    repo = types.SimpleNamespace(config=types.SimpleNamespace(device="cpu", num_actions=A))
    return types.SimpleNamespace(_load=lambda: wm, _repo=repo, imagine_backend=backend, autocast_dtype=None)


@pytest.fixture()
def fixed_observation(monkeypatch):
    import worldoptbench.models.dreamer as dreamer

    monkeypatch.setattr(dreamer, "_observe_core", lambda wm, image, prev: (torch.zeros(1, 2, 3), torch.zeros(1, 1)))


def _starts(n=2):
    return [{"init_video": np.zeros((3, 4, 4, 3), dtype=np.uint8), "context_actions": np.zeros((2, A), dtype=np.float32)} for _ in range(n)]


def test_imagined_return_sums_the_predicted_reward_over_the_horizon(fixed_observation):
    # actor action (0.6, -0.4) sums to 0.2 per step: deter after k steps is 0.2k, reward = deter => sum_{k=1..H} 0.2k
    horizon = 5
    result = imagined_returns(_fake_model(), FakeActor(), [PolicySpec("actor", "actor")], _starts(), horizon)
    expected = sum(0.2 * k for k in range(1, horizon + 1))
    assert result["actor"] == [pytest.approx(expected), pytest.approx(expected)]


def test_better_actions_give_a_higher_imagined_return_and_the_ranking_follows(fixed_observation):
    specs = [PolicySpec("actor", "actor"), PolicySpec("inverted", "inverted"), PolicySpec("zero", "zero")]
    result = imagined_returns(_fake_model(), FakeActor(), specs, _starts(1), horizon=4)
    assert result["actor"][0] > result["zero"][0] > result["inverted"][0]  # +0.2/step, 0, -0.2/step


def test_a_set_imagine_backend_runs_each_step_and_its_output_is_used(fixed_observation):
    calls = []

    def backend(wm, stoch, deter, actions):
        calls.append(actions.shape)
        return {"stoch": stoch[:, None], "deter": (deter + 10.0)[:, None]}  # a world where every step adds 10

    result = imagined_returns(_fake_model(backend), FakeActor(), [PolicySpec("actor", "actor")], _starts(1), horizon=3)
    assert calls == [(1, 1, A)] * 3  # one backend call per step, batch 1, one action
    assert result["actor"][0] == pytest.approx(10.0 + 20.0 + 30.0)


def test_the_same_seed_gives_the_same_imagined_returns_and_a_different_seed_differs_for_stochastic_policies(fixed_observation):
    specs = [PolicySpec("noise_1.0", "noise", 1.0)]
    a = imagined_returns(_fake_model(), FakeActor(), specs, _starts(2), 6, seed=0)
    b = imagined_returns(_fake_model(), FakeActor(), specs, _starts(2), 6, seed=0)
    c = imagined_returns(_fake_model(), FakeActor(), specs, _starts(2), 6, seed=1)
    assert a == b and a != c


# ---- true returns -------------------------------------------------------------------------------------------------


class FakeEnv:
    def __init__(self, steps=5):
        self.steps, self.t, self.closed = steps, 0, False

    def reset(self):
        self.t = 0
        return {"image": np.zeros((4, 4, 3), dtype=np.uint8), "is_first": True, "is_terminal": False}

    def step(self, action):
        self.t += 1
        reward = float(action["action"].sum())  # rewards the sum of the action
        return {"image": np.zeros((4, 4, 3), dtype=np.uint8), "is_first": False, "is_terminal": False}, reward, self.t >= self.steps, {}

    def close(self):
        self.closed = True


def _fake_world_for_env():
    dyn = types.SimpleNamespace(
        obs_step=lambda latent, action, embed, is_first: ({"stoch": torch.zeros(1, 2), "deter": torch.zeros(1, 1)}, None),
        get_feat=lambda latent: torch.zeros(1, 3),
    )
    wm = types.SimpleNamespace(
        dynamics=dyn, encoder=lambda p: torch.zeros(1, 4),
        preprocess=lambda batch: {k: torch.tensor(v, dtype=torch.float32) for k, v in batch.items()},
    )
    envs = []

    def make_env(seed):
        envs.append(FakeEnv())
        return envs[-1]

    repo = types.SimpleNamespace(config=types.SimpleNamespace(device="cpu", num_actions=A), max_steps=100, make_env=make_env)
    return repo, wm, envs


def test_true_return_is_the_summed_simulator_reward_of_the_policys_actions():
    repo, wm, envs = _fake_world_for_env()
    result = true_returns(repo, wm, FakeActor(), [PolicySpec("actor", "actor"), PolicySpec("inverted", "inverted")], episodes=2)
    assert result["actor"] == [pytest.approx(5 * 0.2)] * 2  # 5 steps of reward 0.6 - 0.4
    assert result["inverted"] == [pytest.approx(-5 * 0.2)] * 2
    assert len(envs) == 4 and all(e.closed for e in envs)  # a fresh env per episode, always closed


def test_true_return_stops_at_max_steps():
    repo, wm, _ = _fake_world_for_env()
    result = true_returns(repo, wm, FakeActor(), [PolicySpec("actor", "actor")], episodes=1, max_steps=3)
    assert result["actor"] == [pytest.approx(3 * 0.2)]


def test_the_fine_policy_set_is_graded_action_noise_on_one_actor_with_unique_names():
    from worldoptbench.models.dreamer_policy import FINE_POLICIES, POLICY_SETS

    assert POLICY_SETS["fine"] is FINE_POLICIES and POLICY_SETS["coarse"] is DEFAULT_POLICIES
    names = [s.name for s in FINE_POLICIES]
    assert len(names) == len(set(names)) == 9 and names[0] == "actor"
    sigmas = [s.value for s in FINE_POLICIES]
    assert sigmas == sorted(sigmas) and sigmas[0] == 0.0
    assert all(s.kind == "noise" for s in FINE_POLICIES[1:])  # only the noise level differs
