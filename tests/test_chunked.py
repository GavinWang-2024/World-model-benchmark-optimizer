"""Chunked rollouts: the driver on fake tensor functions, the noise-up-front backends, and the module.
CPU only; no Dreamer repo, GPU or TensorRT. Needs torch.
"""

import types

import pytest

torch = pytest.importorskip("torch")

from test_tensorrt_gumbel import _fake_rssm

from worldoptbench.models.base import (
    HasChunkedRollout,
    ModelInfo,
    Rollout,
    WorldModelInterface,
)
from worldoptbench.models.chunked import run_chunked
from worldoptbench.models.dreamer import DreamerWorldModel
from worldoptbench.models.dreamer_step import TorchGumbelImagination
from worldoptbench.optimizations import ChunkedRolloutModule
from worldoptbench.optimizations.tensorrt_backend import TensorRTImagination
from worldoptbench.stack import OptimizationStack

# ---- the driver on fake functions -----------------------------------------------------------------------

STATE, FEATURES = 3, 2


def _step(state, action, noise):
    """One fake imagination step: a state update depending on the previous state, the action and the noise."""
    new = 0.9 * state + action.sum(-1, keepdim=True) + noise.reshape(noise.shape[0], -1).sum(-1, keepdim=True)
    return new, torch.cat([new[:, :1], new[:, :1] * 2], dim=1)


def observe_fn(image, prev_actions):
    return image.reshape(image.shape[0], -1)[:, :STATE] + prev_actions.sum(dim=(1, 2)).unsqueeze(1)


def chunk_fn(state, actions, noise):
    features = []
    for t in range(actions.shape[1]):
        state, feature = _step(state, actions[:, t], noise[t])
        features.append(feature)
    return torch.cat([state, torch.stack(features, 1).reshape(state.shape[0], -1)], dim=1)


def decode_fn(features):
    return (features * 10).round().to(torch.int32)  # stands in for the decoder: (B, K, F) -> (B, K, F)


def noise_fn(state, steps):
    torch.manual_seed(5)
    return torch.randn(steps, state.shape[0], 2)


def _reference(image, prev, actions):
    """The same rollout with no chunking at all."""
    state = observe_fn(image, prev)
    noise = noise_fn(state, actions.shape[1])
    features = []
    for t in range(actions.shape[1]):
        state, feature = _step(state, actions[:, t], noise[t])
        features.append(feature)
    return decode_fn(torch.stack(features, 1))


def _inputs(steps, batch=2, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch, 4, 3), torch.randn(batch, 2, 2), torch.randn(batch, steps, 2)


class RecordingExecutor:
    def __init__(self):
        self.calls = []

    def __call__(self, fn, *tensors):
        self.calls.append((fn, tuple(tuple(t.shape) for t in tensors)))
        return fn(*tensors)


@pytest.mark.parametrize("steps,chunk", [(12, 4), (12, 5), (12, 12), (12, 20), (7, 1), (1, 3), (13, 6)])
def test_chunked_equals_unchunked_for_any_chunk_size_including_padding(steps, chunk):
    image, prev, actions = _inputs(steps)
    expected = _reference(image, prev, actions)
    got = run_chunked(lambda fn, *t: fn(*t), observe_fn, chunk_fn, decode_fn, noise_fn, image, prev, actions, chunk)
    assert got.shape == expected.shape == (2, steps, FEATURES)
    assert torch.equal(got, expected)


def test_every_chunk_call_has_the_same_shapes_so_one_graph_serves_any_horizon():
    image, prev, actions = _inputs(13)
    executor = RecordingExecutor()
    run_chunked(executor, observe_fn, chunk_fn, decode_fn, noise_fn, image, prev, actions, 5)
    chunk_calls = [shapes for fn, shapes in executor.calls if fn is chunk_fn]
    decode_calls = [shapes for fn, shapes in executor.calls if fn is decode_fn]
    assert len(chunk_calls) == len(decode_calls) == 3  # ceil(13 / 5), the last one padded
    assert len(set(chunk_calls)) == 1 and len(set(decode_calls)) == 1
    assert chunk_calls[0] == ((2, STATE), (2, 5, 2), (5, 2, 2))


def test_the_executor_always_sees_the_same_function_objects():
    # executors cache captured graphs on the identity of the function, so these must not be rebuilt per call
    image, prev, actions = _inputs(10)
    executor = RecordingExecutor()
    run_chunked(executor, observe_fn, chunk_fn, decode_fn, noise_fn, image, prev, actions, 4)
    run_chunked(executor, observe_fn, chunk_fn, decode_fn, noise_fn, image, prev, actions, 4)
    assert {fn for fn, _ in executor.calls} == {observe_fn, chunk_fn, decode_fn}


def test_noise_is_drawn_once_after_observing_and_before_any_chunk():
    image, prev, actions = _inputs(9)
    order = []

    def observing(*a):
        order.append("observe")
        return observe_fn(*a)

    def noising(state, steps):
        order.append(("noise", steps))
        return noise_fn(state, steps)

    def chunking(*a):
        order.append("chunk")
        return chunk_fn(*a)

    run_chunked(lambda fn, *t: fn(*t), observing, chunking, decode_fn, noising, image, prev, actions, 4)
    assert order == ["observe", ("noise", 9), "chunk", "chunk", "chunk"]  # all 9 steps at once, not per chunk


def test_the_padded_steps_cannot_change_the_real_ones():
    image, prev, actions = _inputs(10)
    base = run_chunked(lambda fn, *t: fn(*t), observe_fn, chunk_fn, decode_fn, noise_fn, image, prev, actions, 4)
    # a different chunk size pads differently (10 -> 12 vs 10 -> 10) but must agree on the 10 real steps
    other = run_chunked(lambda fn, *t: fn(*t), observe_fn, chunk_fn, decode_fn, noise_fn, image, prev, actions, 5)
    assert torch.equal(base, other)


def test_the_driver_validates_its_arguments():
    image, prev, actions = _inputs(4)
    with pytest.raises(ValueError, match="chunk_steps"):
        run_chunked(lambda fn, *t: fn(*t), observe_fn, chunk_fn, decode_fn, noise_fn, image, prev, actions, 0)
    with pytest.raises(ValueError, match="at least one step"):
        run_chunked(lambda fn, *t: fn(*t), observe_fn, chunk_fn, decode_fn, noise_fn, image, prev, actions[:, :0], 2)


# ---- backends with noise drawn up front ---------------------------------------------------------------


def _stoch(batch=1):
    return torch.nn.functional.one_hot(torch.randint(0, 5, (batch, 4)), 5).float()


def test_torch_backend_draw_noise_has_the_right_shape_and_scale():
    wm = types.SimpleNamespace(dynamics=_fake_rssm())
    backend = TorchGumbelImagination()
    noise = backend.draw_noise(wm, _stoch(2), 6)
    assert noise.shape == (6, 2, 4, 5)
    backend.noise_scale = 0.0
    assert torch.count_nonzero(backend.draw_noise(wm, _stoch(2), 6)) == 0


def test_torch_backend_with_pre_drawn_noise_matches_drawing_inside():
    wm = types.SimpleNamespace(dynamics=_fake_rssm())
    stoch, deter, actions = _stoch(), torch.zeros(1, 8), torch.randn(1, 6, 3)
    backend = TorchGumbelImagination()

    torch.manual_seed(11)
    inside = backend(wm, stoch, deter, actions)
    torch.manual_seed(11)
    noise = backend.draw_noise(wm, stoch, 6)
    outside = backend(wm, stoch, deter, actions, noise)

    assert torch.equal(inside["stoch"], outside["stoch"]) and torch.equal(inside["deter"], outside["deter"])


def test_running_the_backend_in_chunks_with_sliced_noise_equals_one_call():
    # the property the whole chunked scheme rests on, at the backend level
    wm = types.SimpleNamespace(dynamics=_fake_rssm())
    stoch, deter, actions = _stoch(), torch.zeros(1, 8), torch.randn(1, 6, 3)
    backend = TorchGumbelImagination()
    torch.manual_seed(3)
    noise = backend.draw_noise(wm, stoch, 6)

    whole = backend(wm, stoch, deter, actions, noise)
    first = backend(wm, stoch, deter, actions[:, :4], noise[:4])
    second = backend(wm, first["stoch"][:, -1], first["deter"][:, -1], actions[:, 4:], noise[4:])

    assert torch.equal(torch.cat([first["stoch"], second["stoch"]], 1), whole["stoch"])
    assert torch.equal(torch.cat([first["deter"], second["deter"]], 1), whole["deter"])


def test_tensorrt_backend_draws_noise_in_its_working_dtype_without_needing_tensorrt():
    wm = types.SimpleNamespace(dynamics=_fake_rssm())
    stoch = _stoch()
    assert TensorRTImagination(precision="fp32").draw_noise(wm, stoch, 3).dtype == torch.float32
    half = TensorRTImagination(precision="fp16")
    assert half.draw_noise(wm, stoch, 3).dtype == torch.float16
    half.noise_scale = 0.0
    assert torch.count_nonzero(half.draw_noise(wm, stoch, 3)) == 0


# ---- the module and the model -------------------------------------------------------------------------


class ChunkableModel(WorldModelInterface):
    architecture = "autoregressive"

    def __init__(self, backend=None):
        self.chunk_steps = None
        self.imagine_backend = backend

    def generate(self, prompt=None, init_frame=None, init_video=None, actions=None, horizon=4.0, **kwargs):
        return Rollout(frames=[], fps=1.0)

    def get_info(self):
        return ModelInfo(name="chunkable", architecture="autoregressive", param_count=0)


def test_module_sets_and_restores_chunk_steps():
    model = ChunkableModel(backend=TorchGumbelImagination())
    module = ChunkedRolloutModule(chunk_steps=25)
    module.apply(model)
    assert model.chunk_steps == 25
    module.restore()
    assert model.chunk_steps is None


def test_module_refuses_a_backend_that_cannot_take_pre_drawn_noise():
    with pytest.raises(RuntimeError, match="pre-drawn noise"):
        ChunkedRolloutModule().apply(ChunkableModel(backend=lambda *a: None))


def test_module_validates_and_labels_and_needs_the_hook():
    with pytest.raises(ValueError, match="chunk_steps"):
        ChunkedRolloutModule(chunk_steps=0)
    assert ChunkedRolloutModule(chunk_steps=50).label == "chunked_rollout_50"
    plain = ChunkableModel()
    del plain.chunk_steps
    stack = OptimizationStack(plain, [ChunkedRolloutModule()])
    assert stack.modules == [] and "chunk_steps" in stack.skipped[0].reason


def test_dreamer_model_exposes_chunking_off_by_default():
    model = DreamerWorldModel(types.SimpleNamespace(task="t", steps_per_second=20.0), ".")
    assert isinstance(model, HasChunkedRollout)
    assert model.chunk_steps is None


def test_chunked_rollout_is_not_offered_to_autotune_as_a_candidate():
    from worldoptbench.autotune import default_candidates
    from worldoptbench.optimizations.base import get_module_class

    assert get_module_class("chunked_rollout").autotune_candidate is False
    assert "chunked_rollout" not in default_candidates(ChunkableModel(backend=TorchGumbelImagination()))
