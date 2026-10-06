"""TensorRT / Gumbel-sampling / no-dist-validation modules and the RSSM step they
share. No GPU, TensorRT, or Dreamer repo needed: the RSSM is replaced by a tiny
fake with the same attribute names, and TensorRT is faked in `sys.modules`.
Tests that touch torch are skipped without it.
"""

import sys
import types

import pytest

from worldoptbench.models.base import HasImagineBackend, ModelInfo, Rollout, WorldModelInterface
from worldoptbench.models.dreamer import DreamerWorldModel
from worldoptbench.optimizations import (
    GumbelSamplingModule,
    NoDistValidationModule,
    TensorRTModule,
    get_module_class,
)
from worldoptbench.optimizations.tensorrt_backend import TensorRTImagination
from worldoptbench.stack import OptimizationStack


class PlainModel(WorldModelInterface):
    architecture = "autoregressive"

    def generate(self, prompt=None, init_frame=None, init_video=None, actions=None, horizon=4.0, **kwargs):
        return Rollout(frames=[], fps=1.0)

    def get_info(self):
        return ModelInfo(name="plain", architecture=self.architecture, param_count=0)


class BackendModel(PlainModel):
    def __init__(self, architecture="autoregressive"):
        self.architecture = architecture
        self.imagine_backend = None


# ---- registration / wiring ----------------------------------------------------


def test_modules_are_registered_and_only_target_recurrent_models():
    for name, cls in (("tensorrt", TensorRTModule), ("gumbel_sampling", GumbelSamplingModule)):
        assert get_module_class(name) is cls
        stack = OptimizationStack(BackendModel(architecture="diffusion"), [cls()])
        assert stack.modules == [] and stack.skipped[0].name == name
    assert get_module_class("no_dist_validation") is NoDistValidationModule


def test_dreamer_model_exposes_an_unset_imagine_backend():
    model = DreamerWorldModel(types.SimpleNamespace(task="t", steps_per_second=20.0), ".")
    assert isinstance(model, HasImagineBackend)
    assert model.imagine_backend is None


def test_backend_modules_reject_models_without_the_hook():
    assert not hasattr(PlainModel(), "imagine_backend")
    with pytest.raises(TypeError, match="imagine_backend"):
        GumbelSamplingModule().apply(PlainModel())
    with pytest.raises(TypeError, match="imagine_backend"):
        TensorRTModule().apply(PlainModel())


def test_gumbel_module_installs_the_torch_backend():
    pytest.importorskip("torch")
    from worldoptbench.models.dreamer_step import TorchGumbelImagination

    model = BackendModel()
    assert GumbelSamplingModule().apply(model) is model
    assert isinstance(model.imagine_backend, TorchGumbelImagination)


# ---- TensorRT module: dependency / device checks ---------------------------------


def test_tensorrt_module_gives_an_actionable_error_when_tensorrt_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorrt", None)  # makes `import tensorrt` raise
    with pytest.raises(ImportError, match="tensorrt-cu12"):
        TensorRTModule().apply(BackendModel())


def test_tensorrt_module_requires_cuda_then_installs_the_backend(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    monkeypatch.setitem(sys.modules, "tensorrt", types.ModuleType("tensorrt"))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        TensorRTModule().apply(BackendModel())

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = BackendModel()
    assert TensorRTModule(cache_dir=tmp_path).apply(model) is model
    assert isinstance(model.imagine_backend, TensorRTImagination)
    assert model.imagine_backend._cache_dir == tmp_path


def test_tensorrt_backend_only_supports_batch_size_one():
    torch = pytest.importorskip("torch")
    backend = TensorRTImagination()
    with pytest.raises(NotImplementedError, match="batch size 1"):
        backend(object(), torch.zeros(2, 4, 4), torch.zeros(2, 8), torch.zeros(2, 3, 5))


# ---- the shared RSSM step ----------------------------------------------------------


def _fake_rssm(stoch=4, classes=5, deter=8, actions=3, logits=None, discrete=True, rec_depth=1):
    """Same attribute names as dreamerv3-torch's RSSM, but tiny and deterministic:
    the output logits are fixed (zero weights + a bias), so the sampling
    distribution is known exactly.
    """
    torch = pytest.importorskip("torch")
    from torch import nn

    class Cell(nn.Module):
        def forward(self, x, state):
            out = torch.tanh(x[:, : state[0].shape[1]] + state[0])
            return out, [out]

    fake = nn.Module()
    fake._discrete = classes if discrete else 0
    fake._rec_depth = rec_depth
    fake._stoch = stoch
    fake._unimix_ratio = 0.01
    fake._num_actions = actions
    fake._deter = deter
    fake._img_in_layers = nn.Linear(stoch * classes + actions, deter)
    fake._cell = Cell()
    fake._img_out_layers = nn.Identity()
    fake._imgs_stat_layer = nn.Linear(deter, stoch * classes)
    with torch.no_grad():
        fake._imgs_stat_layer.weight.zero_()
        bias = torch.tensor(logits if logits is not None else [0.0] * classes).repeat(stoch)
        fake._imgs_stat_layer.bias.copy_(bias)
    return fake


def test_gumbel_noise_is_finite_with_the_right_mean():
    torch = pytest.importorskip("torch")
    from worldoptbench.models.dreamer_step import gumbel_noise

    torch.manual_seed(0)
    noise = gumbel_noise((200_000,), "cpu")
    assert torch.isfinite(noise).all()  # an unclamped version once produced NaNs
    assert noise.mean().item() == pytest.approx(0.5772, abs=0.02)  # Euler-Mascheroni


def test_step_core_outputs_one_hot_rows_and_the_cells_new_state():
    torch = pytest.importorskip("torch")
    from worldoptbench.models.dreamer_step import RSSMStepCore, gumbel_noise

    dyn = _fake_rssm()
    core = RSSMStepCore(dyn)
    stoch = torch.nn.functional.one_hot(torch.randint(0, 5, (2, 4)), 5).float()
    deter = torch.randn(2, 8)
    action = torch.randn(2, 3)

    new_stoch, new_deter = core(stoch, action, deter, gumbel_noise((2, 4, 5), "cpu"))

    assert new_stoch.shape == (2, 4, 5) and new_deter.shape == (2, 8)
    assert torch.equal(new_stoch.sum(-1), torch.ones(2, 4))
    expected, _ = dyn._cell(dyn._img_in_layers(torch.cat([stoch.reshape(2, -1), action], -1)), [deter])
    assert torch.allclose(new_deter, expected)


def test_gumbel_max_sampling_matches_the_unimixed_categorical():
    torch = pytest.importorskip("torch")
    from worldoptbench.models.dreamer_step import RSSMStepCore, gumbel_noise

    logits = [2.0, 1.0, 0.0, -1.0, -2.0]
    core = RSSMStepCore(_fake_rssm(stoch=1, classes=5, logits=logits))
    n = 60_000
    stoch = torch.nn.functional.one_hot(torch.zeros(n, 1, dtype=torch.long), 5).float()

    torch.manual_seed(0)
    sampled, _ = core(stoch, torch.zeros(n, 3), torch.zeros(n, 8), gumbel_noise((n, 1, 5), "cpu"))

    expected = torch.softmax(torch.tensor(logits), -1) * 0.99 + 0.01 / 5
    observed = sampled.mean(0).squeeze(0)
    assert torch.allclose(observed, expected, atol=0.01)  # same distribution as torch.multinomial would give


def test_step_core_refuses_configurations_it_would_get_wrong():
    pytest.importorskip("torch")
    from worldoptbench.models.dreamer_step import RSSMStepCore

    with pytest.raises(NotImplementedError, match="discrete"):
        RSSMStepCore(_fake_rssm(discrete=False))
    with pytest.raises(NotImplementedError, match="rec_depth"):
        RSSMStepCore(_fake_rssm(rec_depth=2))


def test_torch_gumbel_backend_shapes_and_seed_determinism():
    torch = pytest.importorskip("torch")
    from worldoptbench.models.dreamer_step import TorchGumbelImagination

    wm = types.SimpleNamespace(dynamics=_fake_rssm())
    stoch = torch.nn.functional.one_hot(torch.randint(0, 5, (1, 4)), 5).float()
    deter, actions = torch.zeros(1, 8), torch.randn(1, 6, 3)
    backend = TorchGumbelImagination()

    torch.manual_seed(7)
    first = backend(wm, stoch, deter, actions)
    torch.manual_seed(7)
    second = backend(wm, stoch, deter, actions)

    assert first["stoch"].shape == (1, 6, 4, 5) and first["deter"].shape == (1, 6, 8)
    assert torch.equal(first["stoch"], second["stoch"]) and torch.equal(first["deter"], second["deter"])


def test_imagine_decode_uses_the_backend_when_one_is_set():
    torch = pytest.importorskip("torch")
    from worldoptbench.models.dreamer import _imagine_decode

    seen = {}

    def backend(wm, stoch, deter, future):
        seen["called"] = True
        return {"stoch": torch.zeros(1, 3, 2, 2), "deter": torch.zeros(1, 3, 4)}

    decoded = types.SimpleNamespace(mode=lambda: torch.full((1, 3, 2, 2, 3), 0.5))
    dyn = types.SimpleNamespace(
        get_feat=lambda prior: prior["deter"],  # content irrelevant to this wiring test
        imagine_with_action=lambda *a: pytest.fail("the repo's loop must not run when a backend is set"),
    )
    wm = types.SimpleNamespace(dynamics=dyn, heads={"decoder": lambda feat: {"image": decoded}})

    frames = _imagine_decode(wm, torch.zeros(1, 2, 2), torch.zeros(1, 4), torch.zeros(1, 3, 6), False, backend)

    assert seen["called"]
    assert frames.dtype == torch.uint8 and tuple(frames.shape) == (1, 3, 2, 2, 3)  # (batch, steps, H, W, C)
    assert int(frames[0, 0, 0, 0, 0]) == 128  # 0.5 * 255 rounded


# ---- no_dist_validation -------------------------------------------------------------


def test_no_dist_validation_turns_validation_off_and_restore_undoes_it():
    torch = pytest.importorskip("torch")
    before = torch.distributions.Distribution._validate_args
    module = NoDistValidationModule()
    try:
        model = PlainModel()
        assert module.apply(model) is model
        assert torch.distributions.Distribution._validate_args is False
    finally:
        module.restore()
    assert torch.distributions.Distribution._validate_args == before
    module.restore()  # harmless twice
