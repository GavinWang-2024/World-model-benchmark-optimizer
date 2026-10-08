"""CosmosPredict wrapper wiring with a fake pipeline: argument handling, conditioning inputs, the guardrail rule. CPU, no weights."""

import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from worldoptbench.models import cosmos_predict as cp


class FakePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        frames = np.full((1, kwargs["num_frames"], 16, 32, 3), 0.5, dtype=np.float32)
        return types.SimpleNamespace(frames=frames)


def _model():
    model = object.__new__(cp.CosmosPredict)
    model.height, model.width = 16, 32
    model.num_inference_steps, model.guidance_scale = 4, 7.0
    model.num_latent_conditional_frames = 2
    model.cfg_batched, model.step_callbacks = False, []
    model._device = torch.device("cpu")
    model._prompt_embeds = {"a ball": torch.zeros(1, 3, 5)}
    model._negative_embeds = torch.ones(1, 3, 5)
    model.pipeline = FakePipeline()
    model.transformer = torch.nn.Linear(2, 2)
    return model


def test_text_to_world_passes_embeddings_not_text_and_returns_uint8_frames_at_16_fps():
    model = _model()
    out = model.generate(prompt="a ball", horizon=2.0, seed=3)
    call = model.pipeline.calls[0]
    assert "prompt" not in call and "image" not in call and "video" not in call
    assert call["num_frames"] == 33 and call["output_type"] == "np" and call["guidance_scale"] == 7.0
    assert torch.equal(call["negative_prompt_embeds"], torch.ones(1, 3, 5))
    assert len(out.frames) == 33 and out.frames[0].dtype == np.uint8 and out.fps == 16.0
    assert out.metadata["conditioning"] == "text" and "Built on NVIDIA Cosmos" in out.metadata["notice"]


def test_a_frame_or_a_video_becomes_image_or_video_conditioning():
    model = _model()
    frame = np.zeros((16, 32, 3), dtype=np.uint8)
    model.generate(prompt="a ball", init_frame=frame, horizon=1.0)
    assert model.pipeline.calls[-1]["image"].shape == (16, 32, 3) and "video" not in model.pipeline.calls[-1]
    out = model.generate(prompt="a ball", init_video=[frame] * 6, horizon=1.0)
    call = model.pipeline.calls[-1]
    assert len(call["video"]) == 6 and call["num_latent_conditional_frames"] == 2 and "image" not in call
    assert out.metadata["conditioning"] == "video"


def test_refusals_are_loud():
    model = _model()
    with pytest.raises(ValueError, match="needs a prompt"):
        model.generate()
    with pytest.raises(KeyError, match="encode_cosmos_prompts"):
        model.generate(prompt="never encoded")
    with pytest.raises(NotImplementedError, match="not action-conditioned"):
        model.generate(prompt="a ball", actions=[np.zeros(2)])
    frame = np.zeros((16, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="not both"):
        model.generate(prompt="a ball", init_frame=frame, init_video=[frame])


def test_step_callbacks_are_chained_into_the_pipeline_call():
    model = _model()
    model.step_callbacks.append(lambda pipe, step, t, kw: kw)
    model.generate(prompt="a ball", horizon=1.0)
    assert callable(model.pipeline.calls[0]["callback_on_step_end"])


def test_without_the_guardrail_package_the_default_is_an_error_never_a_silent_stub(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "cosmos_guardrail":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(RuntimeError, match="safety guardrail"):
        cp.default_safety_checker()


def test_construction_validates_sizes_before_loading_anything():
    with pytest.raises(ValueError, match="multiples of 16"):
        cp.CosmosPredict(height=250, width=448)
    with pytest.raises(ValueError, match="1 or 2"):
        cp.CosmosPredict(num_latent_conditional_frames=3)
