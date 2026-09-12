"""Contract tests for WorldModelInterface — no torch/GPU needed.

Uses a fake in-memory implementation rather than CosmosPredict so these run
everywhere, including CI, before Phase 0's compute/HF access exists.
"""

import pytest

from worldoptbench.models.base import ModelInfo, Rollout, WorldModelInterface


class FakeWorldModel(WorldModelInterface):
    """Minimal concrete implementation used only to exercise the interface."""

    def generate(self, prompt=None, init_frame=None, init_video=None, actions=None, horizon=4.0, **kwargs):
        if prompt is None and init_frame is None and init_video is None:
            raise ValueError("generate() needs at least one of prompt, init_frame, init_video")
        return Rollout(frames=["frame"] * int(horizon), fps=1.0)

    def get_info(self):
        return ModelInfo(name="fake", architecture="diffusion", param_count=0)


def test_cannot_instantiate_interface_directly():
    with pytest.raises(TypeError):
        WorldModelInterface()


def test_fake_model_generates_from_prompt_only():
    rollout = FakeWorldModel().generate(prompt="a robot arm picking up a red cube", horizon=4)
    assert isinstance(rollout, Rollout)
    assert len(rollout.frames) == 4


def test_fake_model_generates_from_init_frame_only():
    rollout = FakeWorldModel().generate(init_frame=object(), horizon=2)
    assert len(rollout.frames) == 2


def test_generate_requires_some_conditioning():
    with pytest.raises(ValueError):
        FakeWorldModel().generate()


def test_get_info_shape():
    info = FakeWorldModel().get_info()
    assert info.architecture == "diffusion"
    assert info.supports_image_conditioning is False  # default
