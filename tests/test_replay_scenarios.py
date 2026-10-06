"""DreamerReplayScenarios against small synthetic episode files. In each fake episode the
frame at time t is filled with the value t (plus a per-episode offset) and the action at
time t is [t, -t], so every alignment claim can be checked by reading values back.
"""

import numpy as np
import pytest

from worldoptbench.models.dreamer import DreamerReplayScenarios

LENGTH = 60  # frames per fake episode: room for 5 context + 40 steps (a 2 s max horizon at 20 steps/s)


def _write_episode(directory, name, offset=0, reward_per_step=0.0, length=LENGTH):
    image = np.stack([np.full((4, 4, 3), (t + offset) % 256, dtype=np.uint8) for t in range(length)])
    action = np.stack([[float(t), -float(t)] for t in range(length)]).astype(np.float32)
    np.savez(directory / name, image=image, action=action, reward=np.full(length, reward_per_step, dtype=np.float32))


@pytest.fixture
def episodes(tmp_path):
    for i in range(3):
        _write_episode(tmp_path, f"ep{i}.npz", offset=i * 50, reward_per_step=float(i))  # returns 0, 60, 120
    return tmp_path


def _provider(directory, **kwargs):
    return DreamerReplayScenarios(directory, steps_per_second=20.0, max_horizon=2.0, **kwargs)


def test_scenario_alignment_follows_the_dreamer_convention(episodes):
    scenario = _provider(episodes)({"id": "x", "seed": 0}, horizon=1)  # seed 0 -> episode 0, start 0
    kw = scenario.generate_kwargs

    assert [int(f[0, 0, 0]) for f in kw["init_video"]] == [0, 1, 2, 3, 4]  # 5 context frames
    # actions[t] is the action that led TO frame t: context actions produced frames 1..4
    assert [float(a[0]) for a in kw["context_actions"]] == [1.0, 2.0, 3.0, 4.0]
    # future actions are those leading to the frames the model must predict (5, 6, ...)
    assert [float(a[0]) for a in kw["actions"]] == [float(t) for t in range(5, 25)]  # 1 s = 20 steps
    assert [int(f[0, 0, 0]) for f in scenario.reference_frames] == list(range(5, 25))
    assert int(scenario.baseline_frame[0, 0, 0]) == 4  # the last frame the model is shown
    assert kw["seed"] == 0


def test_horizons_of_one_entry_are_nested_prefixes_of_one_window(episodes):
    provider = _provider(episodes)
    short = provider({"id": "x", "seed": 4}, horizon=0.5)
    long = provider({"id": "x", "seed": 4}, horizon=2)

    n = len(short.reference_frames)
    assert n == 10 and len(long.reference_frames) == 40
    for a, b in zip(short.generate_kwargs["init_video"], long.generate_kwargs["init_video"]):
        assert np.array_equal(a, b)
    for a, b in zip(short.generate_kwargs["actions"], long.generate_kwargs["actions"][:n]):
        assert np.array_equal(a, b)
    for a, b in zip(short.reference_frames, long.reference_frames[:n]):
        assert np.array_equal(a, b)
    assert short.generate_kwargs["seed"] == long.generate_kwargs["seed"]


def test_episode_and_start_depend_only_on_the_seed_and_wrap_around(episodes):
    provider = _provider(episodes)
    assert provider.num_episodes == 3

    def first_value(seed):
        return int(provider({"id": "x", "seed": seed}, horizon=0.25).generate_kwargs["init_video"][0][0, 0, 0])

    again = [first_value(s) for s in range(6)]
    assert again == [first_value(s) for s in range(6)]  # deterministic
    # seed 3 wraps back to episode 0 but with a different start offset than seed 0
    assert first_value(0) == 0 and first_value(3) == (3 * 37) % (LENGTH - 45 + 1)


def test_min_return_keeps_only_competent_episodes(episodes):
    assert _provider(episodes, min_return=60.0).num_episodes == 2  # returns are 0, 60, 120
    only_best = _provider(episodes, min_return=100.0)
    assert only_best.num_episodes == 1
    # every seed now lands on the one remaining episode (offset 100 -> frame value 100 at its t = 0)
    assert int(only_best({"id": "x", "seed": 0}, 0.25).generate_kwargs["init_video"][0][0, 0, 0]) == 100


def test_clear_errors_for_empty_dirs_unmatched_filters_and_bad_horizons(episodes, tmp_path_factory):
    with pytest.raises(FileNotFoundError, match="no .npz"):
        DreamerReplayScenarios(tmp_path_factory.mktemp("empty"))
    with pytest.raises(ValueError, match="return >="):
        _provider(episodes, min_return=1e6)
    with pytest.raises(ValueError, match="max_horizon"):
        _provider(episodes)({"id": "x", "seed": 0}, horizon=3)  # provider was built for 2 s
    with pytest.raises(ValueError, match="context_steps"):
        DreamerReplayScenarios(episodes, context_steps=0)


def test_an_episode_too_short_for_the_longest_horizon_is_rejected(tmp_path):
    _write_episode(tmp_path, "short.npz", length=30)  # needs 45 for max_horizon 2 s
    with pytest.raises(ValueError, match="frames; need"):
        _provider(tmp_path)({"id": "x", "seed": 0}, horizon=0.25)


def test_returned_arrays_are_copies_not_views_of_the_file(episodes):
    scenario = _provider(episodes)({"id": "x", "seed": 0}, horizon=0.25)
    scenario.generate_kwargs["init_video"][0][...] = 99
    again = _provider(episodes)({"id": "x", "seed": 0}, horizon=0.25)
    assert int(again.generate_kwargs["init_video"][0][0, 0, 0]) == 0
    assert scenario.reference_frames[0].flags["C_CONTIGUOUS"] and scenario.reference_frames[0].dtype == np.uint8
