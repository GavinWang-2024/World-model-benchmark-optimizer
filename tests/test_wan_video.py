"""The parts of the Wan wrapper that need no model: frame-count arithmetic, callback chaining,
and the disk-cached reference scenarios (with a fake baseline model). numpy only; the real
model was verified separately (loads in ~8 s, 33 frames at 192x320 in ~12 s, same seed =>
identical video).
"""

import numpy as np
import pytest

from worldoptbench.models.base import Rollout
from worldoptbench.models.wan_video import WanReferenceScenarios, compose_callbacks, frames_for_horizon


# ---- frame counts ----------------------------------------------------------------------------------


def test_frames_for_horizon_matches_known_wan_lengths():
    # 81 frames (5 s at 16 fps) is Wan's standard clip; the others follow the same 4k + 1 rule.
    assert {h: frames_for_horizon(h) for h in (0.25, 0.5, 1, 2, 3, 5)} == {0.25: 5, 0.5: 9, 1: 17, 2: 33, 3: 49, 5: 81}


def test_frames_for_horizon_is_always_a_valid_wan_length_and_monotonic():
    previous = 0
    for tenth in range(1, 100):
        n = frames_for_horizon(tenth / 10)
        assert n % 4 == 1 and n >= 5  # the VAE compresses time 4x, so lengths are 4k + 1
        assert n >= previous
        previous = n


# ---- callback chaining ---------------------------------------------------------------------------------


def test_compose_callbacks_threads_the_kwargs_through_in_order():
    seen = []

    def first(pipe, step, timestep, kwargs):
        seen.append(("first", step, dict(kwargs)))
        return {**kwargs, "a": 1}

    def second(pipe, step, timestep, kwargs):
        seen.append(("second", step, dict(kwargs)))
        return {**kwargs, "b": 2}

    result = compose_callbacks([first, second])("pipe", 3, "t", {"latents": "x"})

    assert result == {"latents": "x", "a": 1, "b": 2}
    assert [s[0] for s in seen] == ["first", "second"]
    assert seen[1][2] == {"latents": "x", "a": 1}  # the second saw the first's output


def test_compose_callbacks_with_nothing_is_the_identity():
    assert compose_callbacks([])("pipe", 0, "t", {"latents": 1}) == {"latents": 1}


# ---- reference scenarios ---------------------------------------------------------------------------------


class FakeBaseline:
    def __init__(self):
        self.calls = []

    def generate(self, prompt=None, horizon=2.0, seed=0, **kwargs):
        self.calls.append((prompt, horizon, seed))
        base = (len(prompt) + seed + int(horizon * 10)) % 200
        frames = [np.full((4, 6, 3), base + i, dtype=np.uint8) for i in range(5)]
        return Rollout(frames=frames, fps=16.0)


def _provider(tmp_path, fake, **model_kwargs):
    return WanReferenceScenarios(tmp_path, baseline_factory=lambda: fake, model_kwargs=model_kwargs)


def test_references_are_generated_once_then_loaded_from_disk(tmp_path):
    fake = FakeBaseline()
    entry = {"id": "x", "prompt": "a ball", "seed": 3}

    first = _provider(tmp_path, fake)(entry, 2)
    assert fake.calls == [("a ball", 2, 3)]
    assert len(list(tmp_path.glob("*.npz"))) == 1

    second_fake = FakeBaseline()
    second = _provider(tmp_path, second_fake)(entry, 2)  # a new provider, e.g. the next sweep
    assert second_fake.calls == []  # served from the cache; the baseline was never built or run
    for a, b in zip(first.reference_frames, second.reference_frames):
        assert np.array_equal(a, b)


def test_scenario_carries_the_seed_the_reference_and_a_first_frame_baseline(tmp_path):
    fake = FakeBaseline()
    scenario = _provider(tmp_path, fake)({"id": "x", "prompt": "a ball", "seed": 7}, 1)

    assert scenario.generate_kwargs == {"seed": 7}  # the optimized run must use the same seed
    assert len(scenario.reference_frames) == 5
    assert np.array_equal(scenario.baseline_frame, scenario.reference_frames[0])  # skill vs a frozen first frame


def test_seed_defaults_to_zero_and_distinct_requests_get_distinct_cache_files(tmp_path):
    fake = FakeBaseline()
    provider = _provider(tmp_path, fake)
    provider({"id": "x", "prompt": "a ball"}, 2)
    provider({"id": "x", "prompt": "a ball", "seed": 1}, 2)
    provider({"id": "x", "prompt": "a ball", "seed": 0}, 3)
    provider({"id": "y", "prompt": "a cube", "seed": 0}, 2)
    assert [c[2] for c in fake.calls][0] == 0
    assert len(list(tmp_path.glob("*.npz"))) == 4  # seed, horizon and prompt each change the reference


def test_changing_the_sampler_settings_invalidates_the_cached_reference(tmp_path):
    entry = {"id": "x", "prompt": "a ball", "seed": 0}
    fake_a, fake_b = FakeBaseline(), FakeBaseline()
    _provider(tmp_path, fake_a, num_inference_steps=30)(entry, 2)
    _provider(tmp_path, fake_b, num_inference_steps=50)(entry, 2)  # a different baseline: must not reuse the 30-step video
    assert len(fake_b.calls) == 1 and len(list(tmp_path.glob("*.npz"))) == 2


def test_the_baseline_model_is_built_lazily_and_only_once(tmp_path):
    built = []

    def factory():
        built.append(1)
        return FakeBaseline()

    provider = WanReferenceScenarios(tmp_path, baseline_factory=factory)
    assert built == []  # constructing the provider builds nothing
    provider({"id": "x", "prompt": "p", "seed": 0}, 2)
    provider({"id": "y", "prompt": "q", "seed": 0}, 2)
    assert built == [1]
    provider({"id": "x", "prompt": "p", "seed": 0}, 2)  # cached: still one
    assert built == [1]

