"""PrecisionModule / Tf32Module / LeanScanModule and the Dreamer model's opt-in
knobs. Pure-Python wherever possible; tests that touch torch settings are
skipped without torch, and CUDA checks are monkeypatched so no GPU is needed.
"""

import types

import pytest

from worldoptbench.metrics.speed import SpeedMetrics
from worldoptbench.models.base import (
    HasAutocast,
    HasTensorExecutor,
    ModelInfo,
    Rollout,
    WorldModelInterface,
)
from worldoptbench.models.dreamer import DreamerWorldModel
from worldoptbench.optimizations import (
    LeanScanModule,
    PrecisionModule,
    Tf32Module,
    get_module_class,
)
from worldoptbench.optimizations.lean_scan import HasLeanScan
from worldoptbench.stack import OptimizationStack


class PlainModel(WorldModelInterface):
    architecture = "autoregressive"

    def generate(self, prompt=None, init_frame=None, init_video=None, actions=None, horizon=4.0, **kwargs):
        return Rollout(frames=[], fps=1.0)

    def get_info(self):
        return ModelInfo(name="plain", architecture=self.architecture, param_count=0)


class KnobModel(PlainModel):
    def __init__(self, architecture="autoregressive"):
        self.architecture = architecture
        self.autocast_dtype = None
        self.lean_scan = False


# ---- the Dreamer model opts into all three extension points --------------------


def test_dreamer_model_exposes_all_optimization_knobs_with_safe_defaults():
    model = DreamerWorldModel(types.SimpleNamespace(task="t", steps_per_second=20.0), ".")
    assert isinstance(model, HasTensorExecutor)
    assert isinstance(model, HasAutocast) and model.autocast_dtype is None
    assert isinstance(model, HasLeanScan) and model.lean_scan is False


def test_speed_metrics_has_a_reserved_memory_field_defaulting_to_none():
    metrics = SpeedMetrics(latency_seconds=1.0, fps=2.0, vram_peak_gb=None)
    assert metrics.vram_reserved_gb is None


# ---- registration ---------------------------------------------------------------


def test_modules_are_registered_by_name():
    assert get_module_class("precision") is PrecisionModule
    assert get_module_class("tf32") is Tf32Module
    assert get_module_class("lean_scan") is LeanScanModule


# ---- precision ------------------------------------------------------------------


def test_precision_label_includes_dtype_and_unknown_dtype_is_rejected():
    assert PrecisionModule().label == "precision_bfloat16"
    assert PrecisionModule(dtype="float16").label == "precision_float16"
    with pytest.raises(ValueError, match="float8"):
        PrecisionModule(dtype="float8")


def test_precision_rejects_models_without_autocast_dtype():
    assert not isinstance(PlainModel(), HasAutocast)
    with pytest.raises(TypeError, match="autocast_dtype"):
        PrecisionModule().apply(PlainModel())


def test_precision_sets_the_models_autocast_dtype():
    torch = pytest.importorskip("torch")
    model = KnobModel()
    assert PrecisionModule(dtype="float16").apply(model) is model
    assert model.autocast_dtype is torch.float16
    PrecisionModule(dtype="bfloat16").apply(model)
    assert model.autocast_dtype is torch.bfloat16


def test_precision_refuses_bfloat16_on_a_gpu_without_support(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(RuntimeError, match="float16"):
        PrecisionModule(dtype="bfloat16").apply(KnobModel())


# ---- tf32 -----------------------------------------------------------------------


def test_tf32_requires_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        Tf32Module().apply(PlainModel())


def test_tf32_requires_ampere_or_newer(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 5))
    with pytest.raises(RuntimeError, match="Ampere"):
        Tf32Module().apply(PlainModel())


def test_tf32_apply_sets_process_wide_precision_and_restore_undoes_it(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (12, 0))
    before = torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32
    module = Tf32Module()
    try:
        model = PlainModel()
        assert module.apply(model) is model
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        module.restore()
    assert (torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32) == before
    module.restore()  # restoring twice is harmless


# ---- lean scan ------------------------------------------------------------------


def test_lean_scan_rejects_models_without_the_switch():
    assert not isinstance(PlainModel(), HasLeanScan)
    with pytest.raises(TypeError, match="lean_scan"):
        LeanScanModule().apply(PlainModel())


def test_lean_scan_turns_the_switch_on():
    model = KnobModel()
    assert LeanScanModule().apply(model) is model
    assert model.lean_scan is True


def test_lean_scan_is_skipped_for_non_recurrent_architectures():
    stack = OptimizationStack(KnobModel(architecture="diffusion"), [LeanScanModule()])
    assert stack.modules == [] and stack.skipped[0].name == "lean_scan"
    assert stack.name == "baseline"


def test_all_modules_compose_in_one_stack_regardless_of_order():
    model = KnobModel()
    torch = pytest.importorskip("torch")
    del torch
    stack = OptimizationStack(model, [LeanScanModule(), PrecisionModule(dtype="float16")])
    stack.apply()
    assert model.lean_scan is True and str(model.autocast_dtype) == "torch.float16"
    assert stack.name == "lean_scan+precision_float16"
