"""OptimizationModule registry + OptimizationStack — fake models/modules only,
no torch/GPU needed.
"""

import numpy as np
import pytest

from worldoptbench.models.base import ModelInfo, Rollout, WorldModelInterface
from worldoptbench.optimizations import base
from worldoptbench.optimizations.base import (
    OptimizationModule,
    available_modules,
    get_module_class,
    register_module,
)
from worldoptbench.stack import OptimizationStack


class FakeModel(WorldModelInterface):
    def __init__(self, architecture="diffusion"):
        self.architecture = architecture
        self.applied: list[str] = []  # order modules touched this model in

    def generate(self, prompt=None, init_frame=None, init_video=None, actions=None, horizon=4.0, **kwargs):
        return Rollout(frames=[np.zeros((4, 4, 3), dtype=np.uint8)] * 2, fps=1.0)

    def get_info(self):
        return ModelInfo(name="fake", architecture=self.architecture, param_count=0)


class DiffusionOnly(OptimizationModule):
    name = "diffusion_only"
    supported_architectures = ("diffusion",)

    def apply(self, model):
        model.applied.append(self.name)
        return model


class AnyArch(OptimizationModule):
    name = "any_arch"
    supported_architectures = ("diffusion", "autoregressive", "jepa")

    def __init__(self, level=1):
        self.level = level

    def apply(self, model):
        model.applied.append(f"{self.name}:{self.level}")
        return model


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    # Every test starts with an empty registry and can't leak registrations.
    monkeypatch.setattr(base, "_REGISTRY", {})


def test_cannot_instantiate_module_base_directly():
    with pytest.raises(TypeError):
        OptimizationModule()


def test_register_and_lookup_by_name():
    register_module(DiffusionOnly)
    assert get_module_class("diffusion_only") is DiffusionOnly
    assert available_modules() == ["diffusion_only"]


def test_register_same_class_twice_is_fine_but_name_clash_is_not():
    register_module(DiffusionOnly)
    register_module(DiffusionOnly)

    class Impostor(DiffusionOnly):
        pass

    with pytest.raises(ValueError):
        register_module(Impostor)


def test_unknown_module_name_raises_loudly():
    with pytest.raises(KeyError, match="nope"):
        OptimizationStack(FakeModel(), ["nope"])


def test_stack_skips_architecture_incompatible_modules():
    model = FakeModel(architecture="autoregressive")
    stack = OptimizationStack(model, [DiffusionOnly(), AnyArch()])

    assert [m.name for m in stack.modules] == ["any_arch"]
    assert [s.name for s in stack.skipped] == ["diffusion_only"]
    assert "autoregressive" in stack.skipped[0].reason
    assert stack.name == "any_arch"


def test_stack_applies_modules_in_order():
    model = FakeModel()
    stack = OptimizationStack(model, [AnyArch(level=2), DiffusionOnly()])

    result = stack.apply()

    assert result is model
    assert model.applied == ["any_arch:2", "diffusion_only"]
    assert stack.name == "any_arch+diffusion_only"


def test_stack_builds_registered_modules_by_name_with_kwargs():
    register_module(AnyArch)
    model = FakeModel()

    OptimizationStack(model, ["any_arch"], module_kwargs={"any_arch": {"level": 7}}).apply()

    assert model.applied == ["any_arch:7"]


def test_stack_with_no_applicable_modules_is_baseline():
    model = FakeModel(architecture="jepa")
    stack = OptimizationStack(model, [DiffusionOnly()])

    assert stack.name == "baseline"
    assert stack.apply() is model
    assert model.applied == []


def test_stack_passes_wrapper_returned_by_one_module_to_the_next():
    class Wrapped(FakeModel):
        def __init__(self, inner):
            super().__init__(inner.architecture)
            self.inner = inner

    class Wrapper(OptimizationModule):
        name = "wrapper"
        supported_architectures = ("diffusion",)

        def apply(self, model):
            return Wrapped(model)

    class Recorder(OptimizationModule):
        name = "recorder"
        supported_architectures = ("diffusion",)
        seen = None

        def apply(self, model):
            Recorder.seen = model
            return model

    result = OptimizationStack(FakeModel(), [Wrapper(), Recorder()]).apply()

    assert isinstance(result, Wrapped)
    assert Recorder.seen is result


def test_stack_apply_twice_is_an_error():
    stack = OptimizationStack(FakeModel(), [DiffusionOnly()])
    stack.apply()
    with pytest.raises(RuntimeError):
        stack.apply()


def test_stack_name_works_as_runner_label():
    from worldoptbench.runner import run_benchmark

    stack = OptimizationStack(FakeModel(), [AnyArch()])
    results = run_benchmark(stack.apply(), optimization=stack.name)

    assert all(r.optimization == "any_arch" for r in results)
