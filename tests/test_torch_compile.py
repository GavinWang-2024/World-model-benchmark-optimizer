"""The torch_compile module: wiring and failure modes, with the `eager` backend so no compiler is needed.

That inductor speeds a model up is measured by scripts/sweep_vjepa2.py on a GPU; here we check that the right submodules
get compiled in place, that nothing changes numerically under a pass-through backend, and that mistakes fail loudly.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from worldoptbench.models.base import ModelInfo, Rollout, WorldModelInterface
from worldoptbench.models.vjepa2 import VJEPA2Latent
from worldoptbench.optimizations.base import available_modules
from worldoptbench.optimizations.torch_compile import TorchCompileModule


def tiny_model():
    from transformers import VJEPA2Config, VJEPA2Model

    cfg = VJEPA2Config(
        crop_size=32, patch_size=16, frames_per_clip=8, tubelet_size=2,
        hidden_size=96, num_attention_heads=2, num_hidden_layers=2, mlp_ratio=2.0,
        pred_hidden_size=48, pred_num_attention_heads=2, pred_num_hidden_layers=1, num_pooler_layers=1,
    )
    torch.manual_seed(0)
    return VJEPA2Model(cfg).eval()


def clip():
    return np.random.default_rng(0).integers(0, 256, size=(4, 40, 48, 3), dtype=np.uint8)


def test_the_module_is_registered_and_covers_the_architectures_it_is_meant_for():
    assert "torch_compile" in available_modules()
    assert set(TorchCompileModule.supported_architectures) == {"diffusion", "autoregressive", "jepa"}


def test_named_submodules_are_compiled_in_place_and_the_rest_is_left_alone():
    model = VJEPA2Latent(model=tiny_model(), device="cpu")
    module = TorchCompileModule(backend="eager", targets=("encoder", "predictor"))
    assert module.apply(model) is model
    root = model.torch_module()
    assert module.compiled == ["encoder", "predictor"]
    assert root.encoder._compiled_call_impl is not None
    assert root.predictor._compiled_call_impl is not None
    assert root._compiled_call_impl is None  # the parent was not asked for


def test_no_targets_compiles_the_whole_module():
    model = VJEPA2Latent(model=tiny_model(), device="cpu")
    module = TorchCompileModule(backend="eager")
    module.apply(model)
    assert module.compiled == ["<whole model>"]
    assert model.torch_module()._compiled_call_impl is not None


def test_a_pass_through_backend_leaves_the_forecast_unchanged():
    base = VJEPA2Latent(model=tiny_model(), device="cpu").generate(init_video=clip(), future_steps=2)
    model = VJEPA2Latent(model=tiny_model(), device="cpu")
    TorchCompileModule(backend="eager", targets=("encoder", "predictor")).apply(model)
    compiled = model.generate(init_video=clip(), future_steps=2)
    assert all(np.allclose(a, b, atol=1e-6) for a, b in zip(base.frames, compiled.frames))


def test_a_typo_in_a_target_fails_loudly_instead_of_reporting_a_compiled_model():
    model = VJEPA2Latent(model=tiny_model(), device="cpu")
    with pytest.raises(ValueError, match="no submodule 'encodr'"):
        TorchCompileModule(backend="eager", targets=("encodr",)).apply(model)


def test_a_model_without_a_torch_module_is_rejected():
    class Bare(WorldModelInterface):
        def generate(self, *args, **kwargs):
            return Rollout(frames=[], fps=1.0)

        def get_info(self):
            return ModelInfo(name="bare", architecture="jepa", param_count=0)

    with pytest.raises(TypeError, match="torch_module"):
        TorchCompileModule(backend="eager").apply(Bare())
    assert "torch_module" in TorchCompileModule().incompatibility(Bare())


def test_the_label_names_the_mode_only_for_inductor():
    assert TorchCompileModule(mode="max-autotune").label == "torch_compile_max-autotune"
    assert TorchCompileModule(backend="eager").label == "torch_compile"
