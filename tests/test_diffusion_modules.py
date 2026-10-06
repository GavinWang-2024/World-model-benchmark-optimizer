"""Diffusion optimization modules against a tiny random-weight Wan transformer on CPU
(17k parameters; no download, no GPU). These prove the mechanics and the module
contract — a cache engages, restores exactly, conflicts are reported — not speed or
quality on the real model, which is measured separately.

Behaviours asserted here were first probed by hand against diffusers 0.40 and written
down only after being observed (see DESIGN_DIFFUSION.md).
"""

import types

import pytest

pytest.importorskip("diffusers")
torch = pytest.importorskip("torch")

from diffusers.models.transformers.transformer_wan import WanTransformer3DModel  # noqa: E402

from worldoptbench.models.base import ModelInfo, Rollout, WorldModelInterface  # noqa: E402
from worldoptbench.optimizations import (  # noqa: E402
    AttentionBackendModule,
    CfgTruncationModule,
    FasterCacheModule,
    FewerStepsModule,
    FirstBlockCacheModule,
    LayerSkipModule,
    LayerwiseCastingModule,
    PyramidAttentionBroadcastModule,
    TaylorSeerCacheModule,
)
from worldoptbench.stack import OptimizationStack  # noqa: E402


def tiny_transformer():
    torch.manual_seed(0)
    return WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=8, in_channels=4, out_channels=4,
        text_dim=16, freq_dim=16, ffn_dim=32, num_layers=4, cross_attn_norm=True,
        qk_norm="rms_norm_across_heads", eps=1e-6, image_dim=None, added_kv_proj_dim=None, rope_max_seq_len=64,
    ).eval()


class FakeDiffusion(WorldModelInterface):
    """Exposes the opt-in attributes a diffusion model wrapper must (see optimizations/diffusion.py)."""

    def __init__(self, cfg_batched=False):
        self.transformer = tiny_transformer()
        self.pipeline = types.SimpleNamespace(current_timestep=500, num_timesteps=10, _guidance_scale=5.0)
        self.step_callbacks = []
        self.num_inference_steps = 50
        self.cfg_batched = cfg_batched

    def generate(self, prompt=None, init_frame=None, init_video=None, actions=None, horizon=4.0, **kwargs):
        return Rollout(frames=[], fps=1.0)

    def get_info(self):
        return ModelInfo(name="fake-wan", architecture="diffusion", param_count=0)


_INPUTS = None


def _forward_sequence(model):
    """Four forward passes at falling timesteps, each in the 'cond' cache context (as the
    Wan pipeline does), returning the outputs."""
    global _INPUTS
    if _INPUTS is None:
        torch.manual_seed(1)
        _INPUTS = ([torch.randn(1, 4, 3, 8, 8) for _ in range(4)], torch.randn(1, 6, 16))
    xs, txt = _INPUTS
    outs = []
    with torch.no_grad():
        for x, t in zip(xs, (900, 700, 500, 300)):
            with model.transformer.cache_context("cond"):
                outs.append(
                    model.transformer(hidden_states=x, timestep=torch.tensor([t]), encoder_hidden_states=txt, return_dict=False)[0]
                )
    return outs


def _max_diff(a, b):
    return max((x - y).abs().max().item() for x, y in zip(a, b))


@pytest.fixture(scope="module")
def baseline():
    return _forward_sequence(FakeDiffusion())


# ---- caches --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [FirstBlockCacheModule(threshold=1e9), PyramidAttentionBroadcastModule(block_skip_range=2)],
    ids=["first_block_cache", "pab"],
)
def test_a_cache_changes_the_output_when_it_engages_and_restores_it_exactly(module, baseline):
    model = FakeDiffusion()
    assert module.apply(model) is model
    assert model.transformer.is_cache_enabled
    assert _max_diff(_forward_sequence(model), baseline) > 0.1  # it really skipped work

    module.restore()
    assert not model.transformer.is_cache_enabled
    assert _max_diff(_forward_sequence(model), baseline) == 0.0  # and turning it off is exact


def test_first_block_cache_with_a_zero_threshold_never_skips(baseline):
    model = FakeDiffusion()
    FirstBlockCacheModule(threshold=0.0).apply(model)
    assert _max_diff(_forward_sequence(model), baseline) == 0.0


def test_cache_state_does_not_leak_between_videos():
    # diffusers resets cache state when a pipeline call ends (DiffusionPipeline.maybe_free_model_hooks calls
    # transformer._reset_stateful_cache()); calling the transformer directly, as here, has to do the same.
    # Without it the second 'video' would start from the first one's cached features.
    model = FakeDiffusion()
    FirstBlockCacheModule(threshold=1e9).apply(model)
    first = _forward_sequence(model)
    model.transformer._reset_stateful_cache()
    second = _forward_sequence(model)
    assert _max_diff(first, second) == 0.0  # a second 'video' starts clean, not from the first one's cache


def test_taylorseer_applies_and_restores(baseline):
    model = FakeDiffusion()
    module = TaylorSeerCacheModule(cache_interval=2, disable_cache_before_step=1)
    module.apply(model)
    assert model.transformer.is_cache_enabled
    _forward_sequence(model)  # runs without error
    module.restore()
    assert not model.transformer.is_cache_enabled and _max_diff(_forward_sequence(model), baseline) == 0.0


def _taylorseer_hooked(transformer):
    return [
        name
        for name, sub in transformer.named_modules()
        if getattr(sub, "_diffusers_hook", None) is not None and sub._diffusers_hook.get_hook("taylorseer_cache") is not None
    ]


def test_taylorseer_hooks_wan_attention_blocks_by_default():
    # diffusers' default patterns are full-matched and end in `attn`; Wan's modules are `attn1`/`attn2`,
    # so with the defaults nothing was hooked and the cache silently did nothing (measured: 1.00x,
    # output identical to baseline).
    model = FakeDiffusion()
    TaylorSeerCacheModule().apply(model)
    hooked = _taylorseer_hooked(model.transformer)
    assert hooked, "taylorseer hooked nothing"
    assert all(".attn" in name for name in hooked)


def test_taylorseer_explicit_identifiers_override_the_default():
    model = FakeDiffusion()
    TaylorSeerCacheModule(cache_identifiers=[r"^blocks\.0\.attn1"]).apply(model)
    assert _taylorseer_hooked(model.transformer) == ["blocks.0.attn1"]


def test_taylorseer_refuses_to_do_nothing_when_no_block_matches():
    model = FakeDiffusion()
    model.transformer = torch.nn.Sequential(torch.nn.Linear(2, 2))  # no attention blocks at all
    with pytest.raises(Exception):  # no cache support or no blocks: either way, not a silent no-op
        TaylorSeerCacheModule().apply(model)


def test_restore_is_idempotent_and_safe_before_apply():
    module = FirstBlockCacheModule()
    module.restore()  # never applied: nothing to undo
    model = FakeDiffusion()
    module.apply(model)
    module.restore()
    module.restore()


def test_caches_are_mutually_exclusive_in_the_stack():
    model = FakeDiffusion()
    stack = OptimizationStack(
        model, [FirstBlockCacheModule(), PyramidAttentionBroadcastModule(), TaylorSeerCacheModule(), LayerSkipModule(indices=[1])]
    )
    assert [m.name for m in stack.modules] == ["first_block_cache", "layer_skip"]  # layer skip is a different slot
    assert {s.name for s in stack.skipped} == {"pab", "taylorseer"}
    assert all("diffusion_cache" in s.reason and "first_block_cache" in s.reason for s in stack.skipped)
    stack.apply()  # and applying what remains does not crash


def test_applying_two_caches_directly_surfaces_diffusers_own_error():
    model = FakeDiffusion()
    FirstBlockCacheModule().apply(model)
    with pytest.raises(ValueError, match="already been enabled"):
        PyramidAttentionBroadcastModule().apply(model)


def test_pab_needs_a_pipeline_for_the_timestep():
    model = FakeDiffusion()
    del model.pipeline
    assert "pipeline" in PyramidAttentionBroadcastModule().incompatibility(model)


# ---- FasterCache: declared incompatible with unbatched CFG ------------------------------------------


def test_fastercache_is_skipped_when_cfg_runs_as_separate_calls_like_wan():
    reason = FasterCacheModule().incompatibility(FakeDiffusion(cfg_batched=False))
    assert "batched" in reason and "separate calls" in reason
    assert FasterCacheModule().incompatibility(FakeDiffusion(cfg_batched=True)) is None
    stack = OptimizationStack(FakeDiffusion(cfg_batched=False), [FasterCacheModule()])
    assert stack.modules == [] and stack.skipped[0].name == "fastercache"


# ---- layer skip / attention backend / layer-wise casting ---------------------------------------------


def test_layer_skip_changes_the_output(baseline):
    model = FakeDiffusion()
    LayerSkipModule(indices=[1, 2]).apply(model)
    assert _max_diff(_forward_sequence(model), baseline) > 0.1


def test_layer_skip_picks_evenly_spaced_middle_blocks_by_default():
    assert LayerSkipModule.middle_indices(30, 2) == [7, 22]
    assert LayerSkipModule.middle_indices(30, 1) == [14]
    assert LayerSkipModule.middle_indices(4, 2) == [1, 3]
    with pytest.raises(ValueError, match="cannot skip"):
        LayerSkipModule.middle_indices(4, 5)
    model = FakeDiffusion()
    module = LayerSkipModule(num_blocks=2)
    module.apply(model)
    assert module.applied_indices == LayerSkipModule.middle_indices(4, 2)


def test_layer_skip_validates_its_arguments():
    with pytest.raises(TypeError, match="indices"):
        LayerSkipModule([1, 2])  # a list where the block count belongs
    with pytest.raises(ValueError):
        LayerSkipModule(indices=[])
    with pytest.raises(ValueError):
        LayerSkipModule(num_blocks=0)
    assert LayerSkipModule(num_blocks=3).label == "layer_skip_3"
    assert LayerSkipModule(indices=[1, 2]).label == "layer_skip_2"


def test_native_attention_backend_matches_the_default_and_restores(baseline):
    model = FakeDiffusion()
    module = AttentionBackendModule("native")
    module.apply(model)
    assert _max_diff(_forward_sequence(model), baseline) < 1e-6
    module.restore()


def test_unknown_attention_backend_is_rejected_with_the_valid_names():
    with pytest.raises(ValueError, match="native"):
        AttentionBackendModule("warp-drive").apply(FakeDiffusion())
    assert AttentionBackendModule("flex").label == "attention_flex"


def test_layerwise_casting_stores_fp8_and_runs_close_to_baseline(baseline):
    model = FakeDiffusion()
    LayerwiseCastingModule().apply(model)  # compute dtype defaults to the model's own (float32 here)
    dtypes = {p.dtype for p in model.transformer.blocks[0].parameters()}
    assert torch.float8_e4m3fn in dtypes
    assert _max_diff(_forward_sequence(model), baseline) < 0.2  # fp8 rounding error, not a different answer


def test_layerwise_casting_default_compute_dtype_ignores_fp32_kept_params():
    # Real Wan checkpoints load as bf16 but keep `scale_shift_table` in fp32, and it is the FIRST
    # parameter. The default compute dtype used to be next(parameters()).dtype, i.e. float32, so the
    # pipeline cast its latents to float32 and the bf16 patch embedding rejected them.
    model = FakeDiffusion()
    model.transformer.to(torch.bfloat16)
    model.transformer.scale_shift_table.data = model.transformer.scale_shift_table.data.float()
    assert next(model.transformer.parameters()).dtype == torch.float32  # the trap

    LayerwiseCastingModule().apply(model)

    assert model.transformer.dtype == torch.bfloat16  # what the pipeline casts its latents to


def test_layerwise_casting_rejects_non_fp8_storage():
    with pytest.raises(ValueError, match="storage_dtype"):
        LayerwiseCastingModule(storage_dtype="float16")


# ---- step-level modules --------------------------------------------------------------------------------


def test_cfg_truncation_switches_guidance_off_after_the_chosen_fraction():
    model = FakeDiffusion()
    CfgTruncationModule(after_fraction=0.6).apply(model)
    (callback,) = model.step_callbacks
    guidance = []
    for step in range(10):  # the pipeline sets _guidance_scale once at the start of a call, then calls this each step
        result = callback(model.pipeline, step, 0, {"latents": "unchanged"})
        assert result == {"latents": "unchanged"}  # it must hand the callback kwargs back untouched
        guidance.append(model.pipeline._guidance_scale)
    assert guidance == [5.0] * 5 + [1.0] * 5  # off after step index 5, i.e. for the last 40% of steps


def test_cfg_truncation_validates_and_labels():
    for bad in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError):
            CfgTruncationModule(after_fraction=bad)
    assert CfgTruncationModule(after_fraction=0.7).label == "cfg_truncation_0.7"


def test_fewer_steps_sets_the_step_count():
    model = FakeDiffusion()
    assert FewerStepsModule(steps=20).apply(model) is model and model.num_inference_steps == 20
    with pytest.raises(ValueError):
        FewerStepsModule(steps=0)
    assert FewerStepsModule(steps=20).label == "fewer_steps_20"


# ---- contract -------------------------------------------------------------------------------------------


def test_diffusion_modules_do_not_apply_to_a_non_diffusion_model():
    class Recurrent(FakeDiffusion):
        def get_info(self):
            return ModelInfo(name="r", architecture="autoregressive", param_count=0)

    stack = OptimizationStack(Recurrent(), [FirstBlockCacheModule(), CfgTruncationModule(), FewerStepsModule()])
    assert stack.modules == [] and len(stack.skipped) == 3
    assert all("diffusion" in s.reason for s in stack.skipped)


def test_recommended_config_for_a_diffusion_model_is_the_measured_pab_plus_cfg_truncation_tier():
    from worldoptbench.defaults import recommended_config

    names, kwargs = recommended_config(FakeDiffusion())
    assert names == ["pab", "cfg_truncation"]
    assert kwargs == {"pab": {"block_skip_range": 2}, "cfg_truncation": {"after_fraction": 0.4}}
    # and the stack it describes builds without skipping anything
    stack = OptimizationStack(FakeDiffusion(), names, module_kwargs=kwargs)
    assert [m.name for m in stack.modules] == names and not stack.skipped


def test_recommended_config_never_offers_a_module_the_diffusion_model_cannot_run():
    from worldoptbench.defaults import recommended_config

    model = FakeDiffusion()
    del model.step_callbacks  # cfg_truncation needs it, so neither diffusion tier applies
    names, _ = recommended_config(model)
    assert "cfg_truncation" not in names
    assert "cuda_graphs" not in names and "tensorrt" not in names  # Dreamer tiers refuse a diffusion model
