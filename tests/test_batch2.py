"""Sparse decoding, latent noise scaling, TensorRT decoder/precision wiring,
batched generation checks, request scheduling + cache, and the default stack.
CPU-only: models are fakes, CUDA/TensorRT checks are monkeypatched, and tests that
touch tensors are skipped without torch.
"""

import sys
import types

import pytest

from worldoptbench import scheduling
from worldoptbench.defaults import recommended_config, recommended_stack
from worldoptbench.models.base import ModelInfo, Rollout, WorldModelInterface
from worldoptbench.models.dreamer import DreamerWorldModel, synthetic_request
from worldoptbench.optimizations import (
    GumbelSamplingModule,
    LatentNoiseModule,
    SparseDecodeModule,
    TensorRTDecoderModule,
    TensorRTModule,
    base,
    get_module_class,
)
from worldoptbench.optimizations.base import OptimizationModule
from worldoptbench.optimizations.tensorrt_backend import TensorRTImagination
from worldoptbench.optimizations.tensorrt_decoder import TensorRTDecoder


class PlainModel(WorldModelInterface):
    architecture = "autoregressive"

    def generate(self, prompt=None, init_frame=None, init_video=None, actions=None, horizon=4.0, **kwargs):
        return Rollout(frames=[], fps=1.0)

    def get_info(self):
        return ModelInfo(name="plain", architecture=self.architecture, param_count=0)


class KnobModel(PlainModel):
    def __init__(self):
        self.imagine_backend = None
        self.decode_stride = 1
        self.decode_backend = None


# ---- sparse decoding: keyframes and interpolation ------------------------------------------------


def test_keyframe_positions_include_the_last_frame():
    pytest.importorskip("torch")
    from worldoptbench.models.dreamer import _keyframe_positions

    assert _keyframe_positions(5, 2, "cpu").tolist() == [0, 2, 4]
    assert _keyframe_positions(6, 4, "cpu").tolist() == [0, 4, 5]  # last frame added so we never extrapolate
    assert _keyframe_positions(1, 4, "cpu").tolist() == [0]


def test_interpolation_is_exact_at_keyframes_and_linear_between():
    torch = pytest.importorskip("torch")
    from worldoptbench.models.dreamer import _interpolate_keyframes

    out = _interpolate_keyframes(torch.tensor([[0.0, 10.0, 20.0]]), 5, 2)  # keys at t = 0, 2, 4
    assert out.tolist() == [[0.0, 5.0, 10.0, 15.0, 20.0]]

    uneven = _interpolate_keyframes(torch.tensor([[0.0, 8.0, 10.0]]), 6, 4)  # keys at t = 0, 4, 5
    assert uneven.tolist() == [[0.0, 2.0, 4.0, 6.0, 8.0, 10.0]]


def test_interpolation_keeps_trailing_dimensions():
    torch = pytest.importorskip("torch")
    from worldoptbench.models.dreamer import _interpolate_keyframes

    keys = torch.rand(2, 3, 4, 4, 3)
    out = _interpolate_keyframes(keys, 5, 2)
    assert out.shape == (2, 5, 4, 4, 3)
    assert torch.equal(out[:, 0], keys[:, 0]) and torch.equal(out[:, 4], keys[:, 2])


def _fake_decode_world(n_steps):
    """A fake world whose 'decoder' paints each frame with its own time index, so a linear
    signal makes sparse+interpolated decoding exactly equal to dense decoding."""
    torch = pytest.importorskip("torch")
    calls = []
    features = torch.arange(n_steps, dtype=torch.float32).view(1, n_steps, 1)

    def decoder(feat):
        calls.append(feat.shape[1])
        image = (feat / (n_steps - 1)).view(feat.shape[0], feat.shape[1], 1, 1, 1).expand(-1, -1, 2, 2, 3)
        return {"image": types.SimpleNamespace(mode=lambda: image)}

    dyn = types.SimpleNamespace(get_feat=lambda prior: prior["feat"])
    wm = types.SimpleNamespace(dynamics=dyn, heads={"decoder": decoder})
    backend = lambda wm_, stoch, deter, future: {"feat": features}  # noqa: E731
    return wm, backend, calls, torch


def test_sparse_decoding_decodes_fewer_frames_but_matches_dense_on_a_linear_signal():
    from worldoptbench.models.dreamer import _imagine_decode

    n = 9
    wm, backend, calls, torch = _fake_decode_world(n)
    args = (torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, n, 1), False, backend)

    dense = _imagine_decode(wm, *args)
    sparse = _imagine_decode(wm, *args, decode_stride=4)

    assert calls == [n, 3]  # the decoder saw 9 frames, then only the 3 key frames (t = 0, 4, 8)
    assert dense.shape == sparse.shape == (1, n, 2, 2, 3)
    assert torch.equal(dense, sparse)


def test_a_decode_backend_replaces_the_decoder_call():
    from worldoptbench.models.dreamer import _imagine_decode

    wm, backend, calls, torch = _fake_decode_world(4)
    seen = []

    def decode_backend(wm_, features):
        seen.append(tuple(features.shape))
        return torch.full((1, features.shape[1], 2, 2, 3), 0.5)

    frames = _imagine_decode(
        wm, torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 4, 1), False, backend, decode_backend=decode_backend
    )
    assert seen == [(1, 4, 1)] and calls == []  # the repo decoder was never called
    assert int(frames[0, 0, 0, 0, 0]) == 128


def test_sparse_decode_module():
    with pytest.raises(ValueError, match="stride"):
        SparseDecodeModule(stride=1)
    model = KnobModel()
    assert SparseDecodeModule(stride=3).apply(model) is model and model.decode_stride == 3
    assert SparseDecodeModule(stride=3).label == "sparse_decode_3"
    with pytest.raises(TypeError, match="decode_stride"):
        SparseDecodeModule().apply(PlainModel())


# ---- latent noise scaling --------------------------------------------------------------------------------------


def _fake_rssm(logits, stoch=1):
    torch = pytest.importorskip("torch")
    from torch import nn

    classes = len(logits)

    class Cell(nn.Module):
        def forward(self, x, state):
            return state[0], [state[0]]

    fake = nn.Module()
    fake._discrete, fake._rec_depth, fake._stoch, fake._unimix_ratio = classes, 1, stoch, 0.01
    fake._num_actions, fake._deter = 2, 4
    fake._img_in_layers = nn.Linear(stoch * classes + 2, 4)
    fake._cell, fake._img_out_layers = Cell(), nn.Identity()
    fake._imgs_stat_layer = nn.Linear(4, stoch * classes)
    with torch.no_grad():
        fake._imgs_stat_layer.weight.zero_()
        fake._imgs_stat_layer.bias.copy_(torch.tensor(logits, dtype=torch.float32).repeat(stoch))
    return fake


def _run_backend(scale, n=600):
    torch = pytest.importorskip("torch")
    from worldoptbench.models.dreamer_step import TorchGumbelImagination

    backend = TorchGumbelImagination()
    backend.noise_scale = scale
    wm = types.SimpleNamespace(dynamics=_fake_rssm([2.0, 1.0, 0.0, -1.0, -2.0]))
    stoch = torch.nn.functional.one_hot(torch.zeros(n, 1, dtype=torch.long), 5).float()
    torch.manual_seed(0)
    return backend(wm, stoch, torch.zeros(n, 4), torch.zeros(n, 2, 2))["stoch"]


def test_zero_noise_scale_is_the_deterministic_mode():
    out = _run_backend(0.0)
    assert (out.argmax(-1) == 0).all()  # always the most likely class (logit 2.0)


def test_full_noise_scale_samples_the_whole_distribution():
    out = _run_backend(1.0)
    assert len(set(out.argmax(-1).flatten().tolist())) > 1


def test_latent_noise_module_installs_a_backend_and_validates():
    pytest.importorskip("torch")
    from worldoptbench.models.dreamer_step import TorchGumbelImagination

    model = KnobModel()
    assert LatentNoiseModule(scale=0.3).apply(model) is model
    assert isinstance(model.imagine_backend, TorchGumbelImagination) and model.imagine_backend.noise_scale == 0.3
    assert LatentNoiseModule(scale=0.0).label == "latent_noise_0"
    with pytest.raises(ValueError):
        LatentNoiseModule(scale=-1.0)
    with pytest.raises(TypeError, match="imagine_backend"):
        LatentNoiseModule().apply(PlainModel())


def test_latent_noise_reuses_an_existing_backend_and_refuses_one_without_a_scale():
    existing = types.SimpleNamespace(noise_scale=1.0)
    model = KnobModel()
    model.imagine_backend = existing
    LatentNoiseModule(scale=0.5).apply(model)
    assert model.imagine_backend is existing and existing.noise_scale == 0.5

    model.imagine_backend = types.SimpleNamespace()  # no noise_scale attribute
    with pytest.raises(TypeError, match="noise_scale"):
        LatentNoiseModule().apply(model)


def test_backend_modules_applied_later_keep_the_noise_scale(monkeypatch):
    pytest.importorskip("torch")
    model = KnobModel()
    model.imagine_backend = types.SimpleNamespace(noise_scale=0.0)
    GumbelSamplingModule().apply(model)
    assert model.imagine_backend.noise_scale == 0.0  # not reset to 1.0 by the new backend

    import torch

    monkeypatch.setitem(sys.modules, "tensorrt", types.ModuleType("tensorrt"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model.imagine_backend = types.SimpleNamespace(noise_scale=0.25)
    TensorRTModule().apply(model)
    assert model.imagine_backend.noise_scale == 0.25


# ---- TensorRT precision / decoder wiring ---------------------------------------------------------------------


def test_tensorrt_precision_option_and_label():
    assert TensorRTModule().label == "tensorrt"
    assert TensorRTModule(precision="fp16").label == "tensorrt_fp16"
    assert TensorRTImagination(precision="fp16").precision == "fp16"
    with pytest.raises(ValueError, match="precision"):
        TensorRTModule(precision="int8")
    with pytest.raises(ValueError, match="precision"):
        TensorRTImagination(precision="bf16")


def test_tensorrt_decoder_module_requirements(monkeypatch):
    assert get_module_class("tensorrt_decoder") is TensorRTDecoderModule
    with pytest.raises(TypeError, match="decode_backend"):
        TensorRTDecoderModule().apply(PlainModel())

    monkeypatch.setitem(sys.modules, "tensorrt", None)
    with pytest.raises(ImportError, match="tensorrt-cu12"):
        TensorRTDecoderModule().apply(KnobModel())


def test_tensorrt_decoder_module_installs_the_backend(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    monkeypatch.setitem(sys.modules, "tensorrt", types.ModuleType("tensorrt"))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        TensorRTDecoderModule().apply(KnobModel())

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = KnobModel()
    TensorRTDecoderModule(cache_dir=tmp_path).apply(model)
    assert isinstance(model.decode_backend, TensorRTDecoder)
    assert model.decode_backend._cache_dir == tmp_path


def test_decoder_core_needs_a_convolutional_head():
    pytest.importorskip("torch")
    from worldoptbench.models.dreamer_step import DecoderCore

    with pytest.raises(NotImplementedError, match="convolutional"):
        DecoderCore(types.SimpleNamespace())


# ---- batched generation: validation (no model needed) ------------------------------------------------------------


def _fake_batch_model():
    repo = types.SimpleNamespace(
        task="t", steps_per_second=20.0, config=types.SimpleNamespace(num_actions=2)
    )
    return DreamerWorldModel(repo, ".")


def _request(horizon, frame=(4, 4, 3)):
    return synthetic_request(horizon, 20.0, 2, frame_shape=frame, context_steps=3)


def test_generate_batch_of_nothing_is_nothing():
    assert _fake_batch_model().generate_batch([]) == []


def test_generate_batch_rejects_mixed_shapes_and_points_at_bucketing():
    with pytest.raises(ValueError, match="bucket_requests"):
        _fake_batch_model().generate_batch([_request(1), _request(2)])  # 20 vs 40 steps
    with pytest.raises(ValueError, match="bucket_requests"):
        _fake_batch_model().generate_batch([_request(1), _request(1, frame=(8, 8, 3))])


def test_synthetic_request_has_the_shapes_a_real_request_needs():
    request = synthetic_request(2, 20.0, 6)
    assert len(request["init_video"]) == 5 and len(request["context_actions"]) == 4
    assert len(request["actions"]) == 40 and request["horizon"] == 2
    assert request["init_video"][0].shape == (64, 64, 3) and request["actions"][0].shape == (6,)


# ---- scheduling and cache -------------------------------------------------------------------------------------------


def test_bucket_requests_groups_by_shape_preserving_order():
    requests = [_request(1), _request(2), _request(1), _request(1, frame=(8, 8, 3)), _request(2)]
    assert scheduling.bucket_requests(requests) == [[0, 2], [1, 4], [3]]  # buckets in first-seen order


def test_bucket_requests_splits_large_buckets_and_validates():
    requests = [_request(1)] * 5
    assert scheduling.bucket_requests(requests, max_batch=2) == [[0, 1], [2, 3], [4]]
    with pytest.raises(ValueError, match="max_batch"):
        scheduling.bucket_requests(requests, max_batch=0)
    assert scheduling.bucket_requests([]) == []


def test_request_key_depends_on_everything_that_changes_the_result():
    base_request = _request(1)
    key = scheduling.request_key(base_request, seed=0)
    assert key == scheduling.request_key(_request(1), seed=0)  # equal content, equal key

    assert key != scheduling.request_key(base_request, seed=1)
    assert key != scheduling.request_key(_request(2), seed=0)

    changed_frame = _request(1)
    changed_frame["init_video"][0] = changed_frame["init_video"][0] + 1
    assert key != scheduling.request_key(changed_frame, seed=0)

    changed_action = _request(1)
    changed_action["actions"][0] = changed_action["actions"][0] + 0.5
    assert key != scheduling.request_key(changed_action, seed=0)

    no_context_actions = dict(_request(1))
    no_context_actions.pop("context_actions")
    assert key != scheduling.request_key(no_context_actions, seed=0)


def test_result_cache_is_least_recently_used_and_counts_hits():
    cache = scheduling.ResultCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1  # touching "a" makes "b" the oldest
    cache.put("c", 3)
    assert cache.get("b") is None and cache.get("a") == 1 and cache.get("c") == 3
    assert len(cache) == 2 and cache.hits == 3 and cache.misses == 1
    with pytest.raises(ValueError):
        scheduling.ResultCache(capacity=0)


def test_result_cache_namespaces_keep_stacks_apart():
    a, b = scheduling.ResultCache(namespace="baseline"), scheduling.ResultCache(namespace="tensorrt")
    a.put("same-request", "frames-from-baseline")
    assert b.get("same-request") is None
    assert a._full("k") == "baseline|k" and b._full("k") == "tensorrt|k"


# ---- default stack ---------------------------------------------------------------------------------------------------


def _fake_module(name, maturity="measured", needs_cuda=False, requires=()):
    return type(
        f"Fake_{name}",
        (OptimizationModule,),
        {
            "name": name,
            "supported_architectures": ("autoregressive",),
            "maturity": maturity,
            "needs_cuda": needs_cuda,
            "requires": requires,
            "summary": "fake",
            "apply": lambda self, model: model,
        },
    )


@pytest.fixture
def fake_library(monkeypatch):
    registry = {}
    monkeypatch.setattr(base, "_REGISTRY", registry)
    monkeypatch.setattr(base, "_cuda_available", lambda: True)
    return registry


def test_default_stack_prefers_tensorrt_then_degrades(fake_library):
    for name in ("tensorrt", "gumbel_sampling", "cuda_graphs", "no_dist_validation"):
        fake_library[name] = _fake_module(name)
    model = PlainModel()
    assert recommended_stack(model) == ["tensorrt", "cuda_graphs"]

    del fake_library["tensorrt"]  # e.g. the package isn't installed / module unavailable
    assert recommended_stack(model) == ["gumbel_sampling", "cuda_graphs"]

    del fake_library["gumbel_sampling"]
    assert recommended_stack(model) == ["cuda_graphs"]


def test_default_stack_skips_a_tier_when_any_member_does_not_apply(fake_library, monkeypatch):
    fake_library["tensorrt"] = _fake_module("tensorrt", requires=("imagine_backend",))
    fake_library["cuda_graphs"] = _fake_module("cuda_graphs", needs_cuda=True)
    fake_library["no_dist_validation"] = _fake_module("no_dist_validation")
    model = PlainModel()  # has no imagine_backend, so tensorrt can't apply
    assert recommended_stack(model) == ["cuda_graphs"]

    monkeypatch.setattr(base, "_cuda_available", lambda: False)  # and no GPU: graphs can't either
    assert recommended_stack(model) == ["no_dist_validation"]


def test_default_stack_never_recommends_an_experimental_module(fake_library):
    fake_library["cuda_graphs"] = _fake_module("cuda_graphs", maturity="experimental")
    assert recommended_stack(PlainModel()) == []  # nothing measured applies, so run unoptimized


def test_recommended_config_returns_the_kwargs_the_top_tier_was_measured_with(fake_library):
    for name in ("tensorrt", "cuda_graphs"):
        fake_library[name] = _fake_module(name)
    names, kwargs = recommended_config(PlainModel())
    assert names == ["tensorrt", "cuda_graphs"]
    assert kwargs == {"tensorrt": {"precision": "fp16"}}  # fp16 is the measured configuration, not fp32

    # the returned kwargs are a copy: mutating them must not corrupt the next call
    kwargs["tensorrt"]["precision"] = "fp32"
    assert recommended_config(PlainModel())[1] == {"tensorrt": {"precision": "fp16"}}


def test_recommended_config_with_nothing_applicable_is_empty(fake_library):
    assert recommended_config(PlainModel()) == ([], {})
