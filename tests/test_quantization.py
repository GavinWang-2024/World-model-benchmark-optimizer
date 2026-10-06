"""QuantizationModule — TorchAO is replaced by a recording fake, so these check
the module's own logic (scheme handling, layer filtering, refusal to no-op),
not that TorchAO quantizes correctly. Tests that need real nn.Linear layers
are skipped unless torch is installed.
"""

import sys
import types

import pytest

from worldoptbench.models.base import ModelInfo, Rollout, TorchBacked, WorldModelInterface
from worldoptbench.optimizations import QuantizationModule, get_module_class
from worldoptbench.optimizations.quantization import SCHEMES
from worldoptbench.stack import OptimizationStack


class PlainModel(WorldModelInterface):
    """Has no torch_module(), so nothing for quantization to act on."""

    def generate(self, prompt=None, init_frame=None, init_video=None, actions=None, horizon=4.0, **kwargs):
        return Rollout(frames=[], fps=1.0)

    def get_info(self):
        return ModelInfo(name="plain", architecture="autoregressive", param_count=0)


class TorchModel(PlainModel):
    def __init__(self, module):
        self._module = module

    def torch_module(self):
        return self._module


def _install_fake_torchao(monkeypatch, config_names=tuple(SCHEMES.values())):
    """Registers a fake `torchao.quantization` whose quantize_ just records its call."""
    calls = []
    fake = types.ModuleType("torchao.quantization")
    fake.quantize_ = lambda module, config, filter_fn=None: calls.append((module, config, filter_fn))
    for name in config_names:
        setattr(fake, name, type(name, (), {}))
    monkeypatch.setitem(sys.modules, "torchao", types.ModuleType("torchao"))
    monkeypatch.setitem(sys.modules, "torchao.quantization", fake)
    return calls


def test_registered_under_its_name_and_supports_every_architecture():
    assert get_module_class("quantization") is QuantizationModule
    info = PlainModel().get_info()
    for arch in ("diffusion", "autoregressive", "jepa"):
        assert QuantizationModule().supports(ModelInfo(name="m", architecture=arch, param_count=0))
    assert QuantizationModule().supports(info)


def test_label_includes_scheme_so_result_files_stay_distinguishable():
    assert QuantizationModule().label == "quantization_int8_weight_only"
    assert QuantizationModule(scheme="float8_weight_only").label == "quantization_float8_weight_only"
    # a model that exposes torch_module(), else the stack would (correctly) skip the module
    stack = OptimizationStack(TorchModel(object()), [QuantizationModule(scheme="int4_weight_only")])
    assert stack.name == "quantization_int4_weight_only"


def test_unknown_scheme_is_rejected():
    with pytest.raises(ValueError, match="nope"):
        QuantizationModule(scheme="nope")


def test_plain_models_without_torch_module_are_rejected():
    assert not isinstance(PlainModel(), TorchBacked)
    with pytest.raises(TypeError, match="torch_module"):
        QuantizationModule().apply(PlainModel())


def test_missing_torchao_gives_an_actionable_error(monkeypatch):
    # A None entry makes the import raise ImportError even if torchao is installed.
    monkeypatch.setitem(sys.modules, "torchao", None)
    monkeypatch.setitem(sys.modules, "torchao.quantization", None)
    with pytest.raises(ImportError, match="torchao"):
        QuantizationModule().apply(TorchModel(object()))


def test_torchao_missing_the_config_class_gives_an_actionable_error(monkeypatch):
    _install_fake_torchao(monkeypatch, config_names=())
    with pytest.raises(ImportError, match="Int8WeightOnlyConfig"):
        QuantizationModule().apply(TorchModel(object()))


def test_apply_quantizes_only_linear_layers_with_the_schemes_config(monkeypatch):
    nn = pytest.importorskip("torch").nn
    calls = _install_fake_torchao(monkeypatch)
    module = nn.Sequential(nn.Linear(8, 16), nn.Conv2d(1, 1, 3), nn.Linear(16, 2))
    model = TorchModel(module)

    quant = QuantizationModule(scheme="float8_weight_only")
    result = quant.apply(model)

    assert result is model
    assert quant.quantized_layers == 2  # the two Linears; the Conv2d is left alone
    (called_module, config, keep), = calls
    assert called_module is module
    assert type(config).__name__ == SCHEMES["float8_weight_only"]
    assert [keep(m, name) for name, m in module.named_modules() if name] == [True, False, True]


def test_min_features_skips_small_layers(monkeypatch):
    nn = pytest.importorskip("torch").nn
    _install_fake_torchao(monkeypatch)
    module = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 2))

    quant = QuantizationModule(min_features=4)
    quant.apply(TorchModel(module))

    assert quant.quantized_layers == 1  # Linear(16, 2) has out_features < 4


def test_refuses_to_report_a_noop_as_quantized(monkeypatch):
    nn = pytest.importorskip("torch").nn
    calls = _install_fake_torchao(monkeypatch)
    module = nn.Sequential(nn.Linear(8, 16))

    with pytest.raises(ValueError, match="No Linear layers matched"):
        QuantizationModule(min_features=1000).apply(TorchModel(module))
    assert calls == []  # quantize_ was never called
