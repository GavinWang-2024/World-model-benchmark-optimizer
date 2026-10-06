"""cudnn_benchmark / channels_last / low_rank, graph-pool sharing, the VRAM advisor,
and library-wide hygiene. CPU-only: CUDA checks are monkeypatched, and tests that
touch tensors are skipped without torch.
"""

import pytest

from worldoptbench import memory
from worldoptbench.models.base import ModelInfo, Rollout, WorldModelInterface
from worldoptbench.optimizations import (
    ChannelsLastModule,
    CudaGraphsModule,
    CudnnBenchmarkModule,
    LowRankModule,
    available_modules,
    get_module_class,
)


class PlainModel(WorldModelInterface):
    def generate(self, prompt=None, init_frame=None, init_video=None, actions=None, horizon=4.0, **kwargs):
        return Rollout(frames=[], fps=1.0)

    def get_info(self):
        return ModelInfo(name="plain", architecture="autoregressive", param_count=0)


class TorchModel(PlainModel):
    def __init__(self, module):
        self._module = module

    def torch_module(self):
        return self._module


# ---- library hygiene ----------------------------------------------------------------------


def test_every_registered_module_declares_a_summary_and_a_valid_maturity():
    names = available_modules()
    assert len(names) >= 11  # the library should not silently shrink
    for name in names:
        cls = get_module_class(name)
        assert cls.summary, f"{name} has no summary"
        assert cls.maturity in {"measured", "experimental"}, f"{name} has maturity {cls.maturity!r}"
        assert cls.supported_architectures, f"{name} declares no architectures"


def test_every_registered_module_is_constructible_with_defaults():
    # library_report() and autotune's default_candidates() instantiate every module with no
    # arguments, so one that demands an argument would break "hand the stack the whole library".
    for name in available_modules():
        module = get_module_class(name)()
        assert module.name == name and module.label


def test_a_measured_module_does_not_claim_to_be_unmeasured():
    # "measured" must mean a result is recorded, so the summary must not contradict it.
    for name in available_modules():
        cls = get_module_class(name)
        if cls.maturity == "measured":
            assert "unmeasured" not in cls.summary.lower(), f"{name} is marked measured but its summary says unmeasured"


def test_modules_swept_on_hardware_are_marked_measured_and_state_the_verdict():
    # These three were swept; their summaries must carry the finding, not a promise.
    for cls in (CudnnBenchmarkModule, ChannelsLastModule, LowRankModule):
        assert cls.maturity == "measured"
        assert "Measured" in cls.summary or "measured" in cls.summary
    assert "harms physics" in LowRankModule.summary  # the one result that should stop someone using it


# ---- cudnn_benchmark ----------------------------------------------------------------------------


def test_cudnn_benchmark_requires_cuda_and_restores(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        CudnnBenchmarkModule().apply(PlainModel())

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    before = torch.backends.cudnn.benchmark
    module = CudnnBenchmarkModule()
    try:
        assert module.apply(PlainModel()) is not None
        assert torch.backends.cudnn.benchmark is True
    finally:
        module.restore()
    assert torch.backends.cudnn.benchmark == before
    module.restore()  # harmless twice


# ---- channels_last -----------------------------------------------------------------------------------


def test_channels_last_converts_conv_weights_without_changing_outputs():
    torch = pytest.importorskip("torch")
    from torch import nn

    net = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Conv2d(8, 4, 3), nn.Flatten(), nn.Linear(4 * 4 * 4, 2)).eval()
    x = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        before = net(x)

    module = ChannelsLastModule()
    module.apply(TorchModel(net))

    assert module.converted_parameters == 2  # the two conv kernels; biases and the Linear weight aren't 4-D
    conv_weights = [m.weight for m in net.modules() if isinstance(m, nn.Conv2d)]
    assert all(w.is_contiguous(memory_format=torch.channels_last) for w in conv_weights)
    with torch.no_grad():
        assert torch.allclose(net(x), before, atol=1e-6)


def test_channels_last_refuses_a_model_with_no_convolutions_and_a_model_without_the_hook():
    torch = pytest.importorskip("torch")
    from torch import nn

    with pytest.raises(ValueError, match="no 4-D"):
        ChannelsLastModule().apply(TorchModel(nn.Linear(4, 4)))
    with pytest.raises(TypeError, match="torch_module"):
        ChannelsLastModule().apply(PlainModel())
    del torch


# ---- low_rank --------------------------------------------------------------------------------------------


def _low_rank_linear(in_features=64, out_features=128, true_rank=8):
    """A Linear whose weight genuinely has low rank, so truncation at a larger rank is lossless."""
    import torch
    from torch import nn

    torch.manual_seed(0)
    layer = nn.Linear(in_features, out_features)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(out_features, true_rank) @ torch.randn(true_rank, in_features))
    return layer


def test_low_rank_is_lossless_when_the_weight_really_is_low_rank_and_shrinks_the_model():
    torch = pytest.importorskip("torch")
    from torch import nn

    layer = _low_rank_linear()
    net = nn.Sequential(layer, nn.ReLU())
    x = torch.randn(5, 64)
    with torch.no_grad():
        before = net(x)

    module = LowRankModule(rank_fraction=0.25)  # keeps rank 16 >= the true rank 8
    module.apply(TorchModel(net))

    assert module.factored_layers == 1
    assert module.parameters_after < module.parameters_before  # 16*(64+128) < 64*128
    assert isinstance(net[0], nn.Sequential) and len(net[0]) == 2
    with torch.no_grad():
        assert torch.allclose(net(x), before, atol=1e-2)


def test_low_rank_preserves_the_bias():
    torch = pytest.importorskip("torch")
    from torch import nn

    layer = _low_rank_linear()
    with torch.no_grad():
        layer.bias.fill_(3.0)
    net = nn.Sequential(layer)
    LowRankModule().apply(TorchModel(net))
    assert torch.allclose(net[0][1].bias, torch.full((128,), 3.0))


def test_low_rank_skips_small_layers_and_layers_it_could_not_shrink():
    torch = pytest.importorskip("torch")
    from torch import nn

    small = nn.Linear(16, 16)  # below min_features
    square = nn.Linear(64, 64)  # at rank_fraction 0.75, rank 48: 48*128 >= 64*64, no saving
    with pytest.raises(ValueError, match="no Linear layer was eligible"):
        LowRankModule(rank_fraction=0.75).apply(TorchModel(nn.Sequential(small, square)))
    assert isinstance(square, nn.Linear)  # untouched
    del torch


def test_low_rank_validates_its_arguments_and_labels_by_fraction():
    with pytest.raises(ValueError):
        LowRankModule(rank_fraction=1.0)
    with pytest.raises(ValueError):
        LowRankModule(rank_fraction=0.0)
    assert LowRankModule(rank_fraction=0.5).label == "low_rank_0.5"
    with pytest.raises(TypeError, match="torch_module"):
        LowRankModule().apply(PlainModel())


# ---- graph pool sharing --------------------------------------------------------------------------------------


def test_cuda_graphs_module_passes_pool_sharing_to_the_executor(monkeypatch):
    torch = pytest.importorskip("torch")
    from worldoptbench.models.base import eager_executor

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    class ExecutorModel(PlainModel):
        tensor_executor = eager_executor

    off, on = ExecutorModel(), ExecutorModel()
    CudaGraphsModule().apply(off)
    CudaGraphsModule(share_pool=True).apply(on)
    assert off.tensor_executor._share_pool is False
    assert on.tensor_executor._share_pool is True


# ---- VRAM advisor ---------------------------------------------------------------------------------------------


def test_weights_estimate_matches_byte_arithmetic():
    assert memory.estimate_weights_gb(1024**3, "float32") == pytest.approx(4.0)
    assert memory.estimate_weights_gb(1024**3, "bfloat16") == pytest.approx(2.0)
    assert memory.estimate_weights_gb(1024**3, "int4") == pytest.approx(0.5)
    with pytest.raises(ValueError, match="float7"):
        memory.estimate_weights_gb(1, "float7")


def test_recommend_picks_the_highest_fidelity_that_fits():
    assert memory.recommend(100_000_000, vram_gb=24).dtype == "float32"  # tiny model, plenty of room
    big = memory.recommend(7_000_000_000, vram_gb=24)  # fp32 ~28 GB*1.5 no; bf16 14*1.5=21 <= 21.6
    assert (big.fits, big.dtype, big.quantization) == (True, "bfloat16", None)
    huge = memory.recommend(14_000_000_000, vram_gb=24)  # int8: 14*1.5 = 21 <= 21.6
    assert (huge.fits, huge.dtype, huge.quantization) == (True, "int8", "int8_weight_only")
    assert any("physics" in note for note in huge.notes)  # warns that quantization needs re-measuring


def test_recommend_reports_when_nothing_fits_and_respects_compute_capability():
    nothing = memory.recommend(70_000_000_000, vram_gb=24)
    assert nothing.fits is False and nothing.dtype is None
    assert any("parallelism" in note for note in nothing.notes)

    old_gpu = memory.recommend(7_000_000_000, vram_gb=24, compute_capability=(7, 5))
    assert old_gpu.dtype == "float16"  # no bfloat16 below compute capability 8.0
    assert any("bfloat16 skipped" in note for note in old_gpu.notes)


def test_check_vram_needs_cuda_and_reads_the_device(monkeypatch):
    torch = pytest.importorskip("torch")
    info = ModelInfo(name="m", architecture="autoregressive", param_count=7_000_000_000)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        memory.check_vram(info)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=0: (24 * 1024**3, 24 * 1024**3))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=0: (12, 0))
    assert memory.check_vram(info).dtype == "bfloat16"
