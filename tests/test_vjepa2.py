"""The V-JEPA 2 wrapper, on a tiny randomly initialised model (CPU, no download).

What is checked is the wrapper's own logic: token layout, no future leakage, the interface contract, and that the
optimization hooks are wired. Whether the real checkpoint forecasts well is measured by scripts/sweep_vjepa2.py.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from worldoptbench.models.base import HasAutocast, TorchBacked
from worldoptbench.models.vjepa2 import (
    VJEPA2Latent,
    horizon_to_steps,
    preprocess_video,
)
from worldoptbench.optimizations.base import library_report


@pytest.fixture(scope="module")
def tiny():
    from transformers import VJEPA2Config, VJEPA2Model

    cfg = VJEPA2Config(
        crop_size=32, patch_size=16, frames_per_clip=8, tubelet_size=2,
        hidden_size=96, num_attention_heads=2, num_hidden_layers=2, mlp_ratio=2.0,
        pred_hidden_size=48, pred_num_attention_heads=2, pred_num_hidden_layers=1, num_pooler_layers=1,
    )
    torch.manual_seed(0)
    return VJEPA2Model(cfg).eval()


def make(tiny):
    return VJEPA2Latent(model=tiny, device="cpu", fps=4.0)


def video(n_frames, seed=0, size=40):
    return np.random.default_rng(seed).integers(0, 256, size=(n_frames, size, size + 8, 3), dtype=np.uint8)


# ---- preprocessing and arithmetic ---------------------------------------------------------------------


def test_preprocess_gives_a_normalized_batch_of_the_model_input_size():
    x = preprocess_video(video(4), crop_size=32)
    assert tuple(x.shape) == (1, 4, 3, 32, 32)
    assert x.dtype == torch.float32
    assert x.abs().max() < 3.0  # ImageNet-normalized pixels live within a few units of zero


def test_preprocess_accepts_unit_range_floats_the_same_as_bytes():
    frames = video(2)
    a = preprocess_video(frames, 32)
    b = preprocess_video(frames.astype(np.float32) / 255.0, 32)
    assert torch.allclose(a, b, atol=1e-5)


def test_preprocess_rejects_wrong_shapes():
    with pytest.raises(ValueError, match="T, H, W, 3"):
        preprocess_video(np.zeros((4, 32, 32), dtype=np.uint8), 32)


def test_horizon_to_steps_converts_seconds_at_the_given_rate():
    assert horizon_to_steps(4.0, fps=4.0, tubelet_size=2) == 8
    assert horizon_to_steps(1.0, fps=4.0, tubelet_size=2) == 2
    assert horizon_to_steps(0.01, fps=4.0, tubelet_size=2) == 1  # never zero
    with pytest.raises(ValueError):
        horizon_to_steps(0, 4.0, 2)


# ---- generate ------------------------------------------------------------------------------------------


def test_generate_returns_one_latent_array_per_future_step(tiny):
    model = make(tiny)
    rollout = model.generate(init_video=video(4), horizon=1.0)  # 4 frames = 2 context steps; 1 s = 2 future steps
    assert rollout.metadata["kind"] == "latent"
    assert rollout.metadata["context_steps"] == 2
    assert rollout.metadata["future_steps"] == 2
    assert rollout.metadata["tokens_per_step"] == 4  # (32 / 16) ** 2
    assert len(rollout.frames) == 2
    for step in rollout.frames:
        assert step.shape == (4, 96) and step.dtype == np.float32 and np.isfinite(step).all()
    assert rollout.metadata["context_last"].shape == (4, 96)
    assert rollout.fps == 2.0  # 4 fps / tubelet 2


def test_future_steps_kwarg_overrides_the_horizon(tiny):
    rollout = make(tiny).generate(init_video=video(4), horizon=100.0, future_steps=3)
    assert len(rollout.frames) == 3


def test_generation_is_deterministic_and_depends_on_the_context(tiny):
    model = make(tiny)
    a = model.generate(init_video=video(4, seed=1), horizon=1.0)
    b = model.generate(init_video=video(4, seed=1), horizon=1.0)
    c = model.generate(init_video=video(4, seed=2), horizon=1.0)
    assert all(np.array_equal(x, y) for x, y in zip(a.frames, b.frames))
    assert not np.allclose(a.frames[0], c.frames[0])


def test_the_forecast_equals_the_predictor_run_on_context_only_tokens(tiny):
    """No leakage: reproduce the forecast by hand from an encoder pass over the context frames alone."""
    model = make(tiny)
    frames = video(4, seed=3)
    rollout = model.generate(init_video=frames, future_steps=2)

    pixels = preprocess_video(frames, 32)
    with torch.no_grad():
        context = tiny.encoder(pixel_values_videos=pixels).last_hidden_state
        n_ctx_tokens = context.shape[1]
        assert n_ctx_tokens == 2 * 4  # two context steps of four tokens
        expected = tiny.predictor(
            encoder_hidden_states=context,
            context_mask=[torch.arange(n_ctx_tokens).unsqueeze(0)],
            target_mask=[torch.arange(n_ctx_tokens, n_ctx_tokens + 8).unsqueeze(0)],
        ).last_hidden_state[0]
    got = np.concatenate(rollout.frames)
    assert np.allclose(got, expected.numpy(), atol=1e-5)


def test_the_tubelet_step_token_layout_is_time_major(tiny):
    """Tokens of step s are the contiguous block [s*per_step, (s+1)*per_step), time slowest: the wrapper's
    `context_last` (the persistence baseline) must be exactly the final block of the encoder output."""
    model = make(tiny)
    frames = video(4, seed=4)
    rollout = model.generate(init_video=frames, future_steps=1)
    full = model.encode_clip(frames)  # (2 steps * 4 tokens, hidden)
    assert full.shape == (8, 96)
    # context_last is exactly the last step's block of the same encoder output
    assert np.allclose(rollout.metadata["context_last"], full[-4:].numpy(), atol=1e-5)


def test_a_prepared_tensor_gives_the_same_forecast_as_raw_frames(tiny):
    model = make(tiny)
    frames = video(4, seed=5)
    prepared = model.prepare(frames)
    assert tuple(prepared.shape) == (1, 4, 3, 32, 32)
    raw = model.generate(init_video=frames, future_steps=2)
    fast = model.generate(init_video=prepared, future_steps=2)
    assert all(np.array_equal(a, b) for a, b in zip(raw.frames, fast.frames))
    assert np.array_equal(model.encode_clip(frames).numpy(), model.encode_clip(prepared).numpy())


def test_encode_clip_covers_every_step_and_is_float32_cpu(tiny):
    out = make(tiny).encode_clip(video(8))
    assert tuple(out.shape) == (4 * 4, 96)
    assert out.dtype == torch.float32 and out.device.type == "cpu"


# ---- the interface contract ----------------------------------------------------------------------------


def test_generate_without_a_context_clip_raises(tiny):
    with pytest.raises(ValueError, match="init_video"):
        make(tiny).generate(horizon=1.0)


@pytest.mark.parametrize("kwargs", [{"prompt": "a ball"}, {"actions": [0, 1]}])
def test_unsupported_conditioning_raises_instead_of_being_ignored(tiny, kwargs):
    with pytest.raises(ValueError):
        make(tiny).generate(init_video=video(4), **kwargs)


def test_a_frame_count_that_is_not_a_tubelet_multiple_raises(tiny):
    with pytest.raises(ValueError, match="multiple of the tubelet size"):
        make(tiny).generate(init_video=video(3), horizon=1.0)


def test_unknown_kwargs_raise(tiny):
    with pytest.raises(TypeError, match="bogus"):
        make(tiny).generate(init_video=video(4), bogus=1)


def test_get_info_describes_a_jepa_video_conditioned_model(tiny):
    info = make(tiny).get_info()
    assert info.architecture == "jepa"
    assert info.supports_video_conditioning and not info.supports_image_conditioning
    assert info.param_count == sum(p.numel() for p in tiny.parameters())


# ---- optimization hooks --------------------------------------------------------------------------------


def test_the_optional_capabilities_are_exposed(tiny):
    model = make(tiny)
    assert isinstance(model, TorchBacked) and model.torch_module() is tiny
    assert isinstance(model, HasAutocast) and model.autocast_dtype is None


def test_autocast_changes_the_numerics_slightly_but_not_the_shapes(tiny):
    model = make(tiny)
    base = model.generate(init_video=video(4), future_steps=2)
    model.autocast_dtype = torch.bfloat16
    try:
        low = model.generate(init_video=video(4), future_steps=2)
    finally:
        model.autocast_dtype = None
    assert [f.shape for f in low.frames] == [f.shape for f in base.frames]
    assert not np.array_equal(low.frames[0], base.frames[0])
    assert np.isfinite(low.frames[0]).all()


def test_modules_that_need_hooks_this_model_lacks_are_skipped_with_a_reason(tiny):
    report = {r.name: r for r in library_report(make(tiny))}
    assert report["precision"].applicable and report["quantization"].applicable
    assert not report["worldcache"].applicable and "diffusion" in report["worldcache"].reason
    assert not report["cuda_graphs"].applicable and "tensor_executor" in report["cuda_graphs"].reason
