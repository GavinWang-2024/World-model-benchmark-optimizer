"""CudaGraphsModule / CudaGraphExecutor — no CUDA needed: capture is replaced
by a recording fake, so these test the caching, buffer-copy, and module logic,
not that CUDA graphs themselves work (that was verified on a real GPU: see the
results table in DREAMER_SETUP.md). Tests that use tensors need torch.
"""

import types

import pytest

from worldoptbench.models.base import (
    HasTensorExecutor,
    ModelInfo,
    Rollout,
    WorldModelInterface,
    eager_executor,
)
from worldoptbench.models.dreamer import DreamerWorldModel
from worldoptbench.optimizations import CudaGraphsModule, get_module_class
from worldoptbench.optimizations.cuda_graphs import CudaGraphExecutor, _Captured
from worldoptbench.stack import OptimizationStack


class PlainModel(WorldModelInterface):
    def generate(self, prompt=None, init_frame=None, init_video=None, actions=None, horizon=4.0, **kwargs):
        return Rollout(frames=[], fps=1.0)

    def get_info(self):
        return ModelInfo(name="plain", architecture="autoregressive", param_count=0)


class ExecutorModel(PlainModel):
    def __init__(self):
        self.tensor_executor = eager_executor


def test_eager_executor_just_calls_the_function():
    assert eager_executor(lambda a, b: a + b, 2, 3) == 5


def test_dreamer_model_exposes_an_eager_tensor_executor_by_default():
    repo = types.SimpleNamespace(task="dmc_walker_walk", steps_per_second=20.0)
    model = DreamerWorldModel(repo, checkpoint_dir=".")
    assert isinstance(model, HasTensorExecutor)
    assert model.tensor_executor is eager_executor


def test_registered_labelled_and_supports_every_architecture(monkeypatch):
    from worldoptbench.optimizations import base

    assert get_module_class("cuda_graphs") is CudaGraphsModule
    assert CudaGraphsModule().label == "cuda_graphs"
    # The stack only keeps modules whose requirements are met: the model's hook, and a CUDA device.
    monkeypatch.setattr(base, "_cuda_available", lambda: True)
    assert OptimizationStack(ExecutorModel(), [CudaGraphsModule()]).name == "cuda_graphs"
    assert OptimizationStack(PlainModel(), [CudaGraphsModule()]).name == "baseline"  # no tensor_executor hook


def test_models_without_a_tensor_executor_are_rejected():
    assert not isinstance(PlainModel(), HasTensorExecutor)
    with pytest.raises(TypeError, match="tensor_executor"):
        CudaGraphsModule().apply(PlainModel())


def test_apply_requires_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        CudaGraphsModule().apply(ExecutorModel())


def test_apply_installs_a_graph_executor(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = ExecutorModel()

    result = CudaGraphsModule(warmup_iters=7).apply(model)

    assert result is model
    assert isinstance(model.tensor_executor, CudaGraphExecutor)
    assert model.tensor_executor._warmup_iters == 7


class _FakeGraph:
    def __init__(self, entry_ref):
        self.replays = 0
        self._entry_ref = entry_ref

    def replay(self):
        # What a real graph does: read the static inputs, write the static output.
        self.replays += 1
        entry = self._entry_ref[0]
        entry.output.copy_(entry.inputs[0] * 2)


def _fake_capture_executor(captures):
    """A CudaGraphExecutor whose _capture builds a fake graph instead of using CUDA."""
    torch = pytest.importorskip("torch")
    executor = CudaGraphExecutor()

    def fake_capture(fn, tensors):
        captures.append((fn, tuple(tuple(t.shape) for t in tensors)))
        ref = []
        entry = _Captured(
            graph=_FakeGraph(ref),
            inputs=[t.detach().clone() for t in tensors],
            output=torch.zeros_like(tensors[0]),
        )
        ref.append(entry)
        return entry

    executor._capture = fake_capture
    return executor


def test_executor_captures_once_per_shape_and_replays_every_call():
    torch = pytest.importorskip("torch")
    captures = []
    executor = _fake_capture_executor(captures)
    fn = object()

    first = executor(fn, torch.ones(3))
    second = executor(fn, torch.full((3,), 5.0))

    assert len(captures) == 1 and executor.num_graphs == 1
    assert first.tolist() == [2.0, 2.0, 2.0]  # inputs were copied into the static buffer
    assert second.tolist() == [10.0, 10.0, 10.0]  # ...and refreshed on the next call


def test_executor_output_is_a_clone_not_the_static_buffer():
    torch = pytest.importorskip("torch")
    executor = _fake_capture_executor([])
    fn = object()

    out = executor(fn, torch.ones(2))
    out.zero_()  # a caller scribbling on its result must not corrupt the next replay
    assert executor(fn, torch.ones(2)).tolist() == [2.0, 2.0]


def test_executor_captures_separately_per_shape_and_per_function():
    torch = pytest.importorskip("torch")
    captures = []
    executor = _fake_capture_executor(captures)
    fn_a, fn_b = object(), object()

    executor(fn_a, torch.ones(3))
    executor(fn_a, torch.ones(5))  # new shape (e.g. a different horizon)
    executor(fn_b, torch.ones(3))  # new function
    executor(fn_a, torch.ones(3))  # cached

    assert len(captures) == 3 and executor.num_graphs == 3
