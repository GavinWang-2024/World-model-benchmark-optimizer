"""scheduler, uncond_reuse, cross_attn_kv_cache and vae modules (tiny random-weight Wan transformer on CPU)."""

import types

import pytest

pytest.importorskip("diffusers")
torch = pytest.importorskip("torch")

import diffusers  # noqa: E402
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel  # noqa: E402

from worldoptbench.models.base import ModelInfo, Rollout, WorldModelInterface  # noqa: E402
from worldoptbench.optimizations import (  # noqa: E402
    CrossAttentionKVCacheModule,
    SchedulerModule,
    UncondReuseModule,
    VaeModule,
)


def tiny_transformer():
    torch.manual_seed(0)
    return WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=8, in_channels=4, out_channels=4,
        text_dim=16, freq_dim=16, ffn_dim=32, num_layers=4, cross_attn_norm=True,
        qk_norm="rms_norm_across_heads", eps=1e-6, image_dim=None, added_kv_proj_dim=None, rope_max_seq_len=64,
    ).eval()


class FakeVae(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)
        self.tiling = self.slicing = False

    def enable_tiling(self):
        self.tiling = True

    def disable_tiling(self):
        self.tiling = False

    def enable_slicing(self):
        self.slicing = True

    def disable_slicing(self):
        self.slicing = False


class FakeDiffusion(WorldModelInterface):
    def __init__(self):
        self.transformer = tiny_transformer()
        scheduler = diffusers.UniPCMultistepScheduler(
            prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0, solver_order=2)
        self.pipeline = types.SimpleNamespace(current_timestep=500, scheduler=scheduler, vae=FakeVae())
        self.step_callbacks = []
        self.num_inference_steps = 8
        self.cfg_batched = False

    def generate(self, prompt=None, init_frame=None, init_video=None, actions=None, horizon=4.0, **kwargs):
        return Rollout(frames=[], fps=1.0)

    def get_info(self):
        return ModelInfo(name="fake", architecture="diffusion", param_count=0)


def _run(model, steps=8, contexts=("cond", "uncond")):
    """`steps` denoising steps, each calling the transformer once per context with that context's text."""
    torch.manual_seed(1)
    base = torch.randn(1, 4, 3, 8, 8)
    texts = {"cond": torch.randn(1, 6, 16), "uncond": torch.randn(1, 6, 16)}
    outputs = {c: [] for c in contexts}
    with torch.no_grad():
        for i in range(steps):
            for context in contexts:
                with model.transformer.cache_context(context):
                    outputs[context].append(model.transformer(
                        hidden_states=base + 0.01 * i, timestep=torch.tensor([900 - 100 * i]),
                        encoder_hidden_states=texts[context], return_dict=False)[0])
    return outputs


def _count_blocks(model):
    counter = {"n": 0}
    model.transformer.blocks[0].register_forward_hook(lambda *a: counter.__setitem__("n", counter["n"] + 1))
    return counter


def _max_diff(a, b):
    return max((x - y).abs().max().item() for x, y in zip(a, b))


# ---- scheduler -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind,cls", [("unipc", "UniPCMultistepScheduler"), ("dpmpp", "DPMSolverMultistepScheduler"),
                                       ("euler", "FlowMatchEulerDiscreteScheduler")])
def test_each_scheduler_kind_is_built_from_the_pipelines_config_and_can_set_timesteps(kind, cls):
    model = FakeDiffusion()
    SchedulerModule(kind=kind).apply(model)
    scheduler = model.pipeline.scheduler
    assert type(scheduler).__name__ == cls
    scheduler.set_timesteps(10)
    assert len(scheduler.timesteps) == 10


def test_flow_shift_and_solver_order_are_applied_and_restore_puts_the_original_back():
    model = FakeDiffusion()
    original = model.pipeline.scheduler
    module = SchedulerModule(kind="unipc", flow_shift=7.0, solver_order=3)
    module.apply(model)
    assert model.pipeline.scheduler is not original
    assert model.pipeline.scheduler.config.flow_shift == 7.0 and model.pipeline.scheduler.config.solver_order == 3
    module.restore()
    assert model.pipeline.scheduler is original


def test_euler_gets_its_shift_from_the_flow_shift():
    model = FakeDiffusion()
    SchedulerModule(kind="euler", flow_shift=5.0).apply(model)
    assert model.pipeline.scheduler.config.shift == 5.0


def test_scheduler_validates_and_labels():
    with pytest.raises(ValueError, match="kind"):
        SchedulerModule(kind="rk4")
    with pytest.raises(ValueError, match="flow_shift"):
        SchedulerModule(flow_shift=0)
    with pytest.raises(ValueError, match="solver_order"):
        SchedulerModule(solver_order=5)
    assert SchedulerModule(kind="dpmpp", flow_shift=5, solver_order=3).label == "scheduler_dpmpp_shift5_order3"


# ---- uncond_reuse -----------------------------------------------------------------------------------------


def test_uncond_reuse_runs_the_unconditional_branch_only_every_period_steps():
    plain, reused = FakeDiffusion(), FakeDiffusion()
    plain_count, reused_count = _count_blocks(plain), _count_blocks(reused)
    plain_out = _run(plain, steps=8)
    UncondReuseModule(period=2).apply(reused)
    reused_out = _run(reused, steps=8)
    assert plain_count["n"] == 16 and reused_count["n"] == 8 + 4  # 8 conditional + uncond at steps 0, 2, 4, 6
    assert _max_diff(reused_out["cond"], plain_out["cond"]) == 0.0  # the conditional branch is untouched
    # a skipped step returns exactly the previous computed unconditional output
    assert torch.equal(reused_out["uncond"][1], reused_out["uncond"][0])
    assert torch.equal(reused_out["uncond"][2], plain_out["uncond"][2])  # a recomputed step matches the unoptimized one


def test_uncond_reuse_period_three_and_a_fresh_video_start_clean():
    model = FakeDiffusion()
    count = _count_blocks(model)
    UncondReuseModule(period=3).apply(model)
    _run(model, steps=9)
    assert count["n"] == 9 + 3  # uncond at steps 0, 3, 6
    model.transformer._reset_stateful_cache()  # what the pipeline does at the end of a call
    count["n"] = 0
    first = _run(model, steps=2, contexts=("uncond",))
    assert count["n"] == 1  # the first unconditional call of a new video is computed, the second reused
    assert torch.equal(first["uncond"][0], first["uncond"][1])


def test_uncond_reuse_restore_is_exact_and_validates():
    baseline = _run(FakeDiffusion())
    model = FakeDiffusion()
    module = UncondReuseModule(period=2)
    module.apply(model)
    _run(model)
    module.restore()
    model.transformer._reset_stateful_cache()
    assert _max_diff(_run(model)["uncond"], baseline["uncond"]) == 0.0
    with pytest.raises(ValueError, match="period"):
        UncondReuseModule(period=1)
    assert UncondReuseModule(period=3).label == "uncond_reuse_3"


# ---- cross_attn_kv_cache --------------------------------------------------------------------------------


def _count_projections(model):
    counter = {"n": 0}
    for name, module in model.transformer.named_modules():
        if name.endswith("attn2.to_k") or name.endswith("attn2.to_v"):
            original = module.forward

            def counted(*a, _orig=original, **k):
                counter["n"] += 1
                return _orig(*a, **k)

            module.forward = counted
    return counter


def test_the_text_key_value_projections_are_computed_once_per_branch_and_the_output_is_identical():
    plain, cached = FakeDiffusion(), FakeDiffusion()
    plain_count, cached_count = _count_projections(plain), _count_projections(cached)
    plain_out = _run(plain, steps=6)
    CrossAttentionKVCacheModule().apply(cached)
    cached_out = _run(cached, steps=6)
    layers = 4 * 2  # to_k and to_v in each of 4 blocks
    assert plain_count["n"] == 6 * 2 * layers  # every step, both branches
    assert cached_count["n"] == 2 * layers  # once per branch
    for context in ("cond", "uncond"):
        assert _max_diff(cached_out[context], plain_out[context]) == 0.0  # exact, not approximate


def test_the_kv_cache_does_not_leak_into_the_next_video_and_restores_cleanly():
    model = FakeDiffusion()
    count = _count_projections(model)
    module = CrossAttentionKVCacheModule()
    module.apply(model)
    _run(model, steps=3)
    model.transformer._reset_stateful_cache()
    count["n"] = 0
    _run(model, steps=3)
    assert count["n"] == 2 * 8  # recomputed once per branch for the new video
    module.restore()
    count["n"] = 0
    _run(model, steps=3)
    assert count["n"] == 3 * 2 * 8  # back to every step


def test_kv_cache_needs_a_diffusers_transformer_with_cross_attention():
    model = FakeDiffusion()
    model.transformer = torch.nn.Sequential(torch.nn.Linear(2, 2))
    with pytest.raises(TypeError, match="cache_context"):
        CrossAttentionKVCacheModule().apply(model)


# ---- vae ---------------------------------------------------------------------------------------------------


def test_vae_options_are_applied_and_restored():
    model = FakeDiffusion()
    module = VaeModule(tiling=True, slicing=True, dtype="bfloat16")
    module.apply(model)
    vae = model.pipeline.vae
    assert vae.tiling and vae.slicing and next(vae.parameters()).dtype == torch.bfloat16
    module.restore()
    assert not vae.tiling and not vae.slicing and next(vae.parameters()).dtype == torch.float32


def test_vae_validates_and_is_constructible_with_defaults():
    assert VaeModule().tiling is True  # every registered module must work with no arguments
    with pytest.raises(ValueError, match="at least one"):
        VaeModule(tiling=False)
    with pytest.raises(ValueError, match="dtype"):
        VaeModule(dtype="float64")
    assert VaeModule(tiling=True, dtype="bfloat16").label == "vae_tiling_bfloat16"


# ---- magcache ---------------------------------------------------------------------------------------------


def _count_block(model, index=1):
    """Counts how often a block's own computation runs. Wraps its `forward` (before a cache is applied, so the cache's hook
    wraps the counter): a torch forward hook would still fire when a diffusers cache skips the computation."""
    counter = {"n": 0}
    block = model.transformer.blocks[index]
    original = block.forward

    def counted(*a, **k):
        counter["n"] += 1
        return original(*a, **k)

    block.forward = counted
    return counter


def test_calibration_returns_one_ratio_per_step_for_the_conditional_branch():
    from worldoptbench.optimizations.diffusion_more import calibrate_mag_ratios

    model = FakeDiffusion()
    ratios = calibrate_mag_ratios(model, lambda: _run(model, steps=8))
    assert len(ratios) == 8 and all(isinstance(r, float) and r > 0 for r in ratios)
    assert not model.transformer.is_cache_enabled  # calibration switches itself off


def test_calibration_without_a_full_generation_is_an_error_not_an_empty_list():
    from worldoptbench.optimizations.diffusion_more import calibrate_mag_ratios

    with pytest.raises(RuntimeError, match="printed no ratios"):
        calibrate_mag_ratios(FakeDiffusion(), lambda: None)


def test_magcache_skips_blocks_with_calibrated_ratios_and_restores_exactly():
    from worldoptbench.optimizations import MagCacheModule
    from worldoptbench.optimizations.diffusion_more import calibrate_mag_ratios

    baseline = _run(FakeDiffusion())
    model = FakeDiffusion()
    ratios = calibrate_mag_ratios(model, lambda: _run(model, steps=8))
    count = _count_block(model)
    module = MagCacheModule(threshold=1e9, max_skip_steps=3, retention_ratio=0.0, mag_ratios=ratios)
    module.apply(model)
    assert model.transformer.is_cache_enabled
    _run(model)
    assert count["n"] < 16  # blocks after the first were skipped on some steps (16 = every step, both branches)
    module.restore()
    model.transformer._reset_stateful_cache()
    assert not model.transformer.is_cache_enabled
    assert max((x - y).abs().max().item() for c in ("cond", "uncond") for x, y in zip(_run(model)[c], baseline[c])) == 0.0


def test_magcache_without_ratios_is_skipped_by_the_stack_with_a_reason_instead_of_crashing():
    from worldoptbench.optimizations import MagCacheModule
    from worldoptbench.stack import OptimizationStack

    stack = OptimizationStack(FakeDiffusion(), [MagCacheModule()])
    assert stack.modules == [] and "mag_ratios" in stack.skipped[0].reason


def test_magcache_validates_and_shares_the_cache_slot_with_the_other_caches():
    from worldoptbench.optimizations import FirstBlockCacheModule, MagCacheModule
    from worldoptbench.stack import OptimizationStack

    with pytest.raises(ValueError, match="threshold"):
        MagCacheModule(threshold=-1)
    with pytest.raises(ValueError, match="retention_ratio"):
        MagCacheModule(retention_ratio=1.0)
    stack = OptimizationStack(FakeDiffusion(), [FirstBlockCacheModule(), MagCacheModule(mag_ratios=[1.0] * 8)])
    assert [m.name for m in stack.modules] == ["first_block_cache"] and stack.skipped[0].name == "magcache"
    assert MagCacheModule(threshold=0.12).label == "magcache_0.12"


def test_a_model_that_carries_its_own_mag_ratios_makes_magcache_usable_without_passing_any():
    from worldoptbench.optimizations import MagCacheModule
    from worldoptbench.stack import OptimizationStack

    model = FakeDiffusion()
    model.mag_ratios = [1.0, 0.97, 1.02, 0.97, 1.04, 1.0, 1.0, 1.0]
    assert MagCacheModule().incompatibility(model) is None
    stack = OptimizationStack(model, [MagCacheModule(threshold=0.3)])
    assert [m.name for m in stack.modules] == ["magcache"]
    stack.apply()
    assert model.transformer.is_cache_enabled
    explicit = OptimizationStack(FakeDiffusion(), [MagCacheModule(mag_ratios=[1.0] * 8)])  # explicit ratios still work without a model table
    assert [m.name for m in explicit.modules] == ["magcache"]


def test_wan_exposes_bundled_ratios_only_for_the_setup_they_were_measured_on():
    from worldoptbench.models.wan_video import WanVideo

    def wan(height, width, steps):
        model = object.__new__(WanVideo)
        model.height, model.width, model.num_inference_steps = height, width, steps
        model.pipeline = types.SimpleNamespace(scheduler=diffusers.UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True))
        return model

    measured = wan(192, 320, 30).mag_ratios
    assert measured is not None and len(measured) == 30 and all(r > 0 for r in measured)
    assert wan(480, 832, 30).mag_ratios is None  # another resolution was never calibrated
    assert wan(192, 320, 20).mag_ratios is None  # nor another step count


# ---- recommended defaults for diffusion ---------------------------------------------------------------------------


def test_the_default_diffusion_tier_adds_the_lossless_kv_cache_and_bf16_vae_when_the_pipeline_has_a_vae():
    from worldoptbench.defaults import recommended_config
    from worldoptbench.optimizations import available_modules  # noqa: F401
    from worldoptbench.stack import OptimizationStack

    model = FakeDiffusion()
    names, kwargs = recommended_config(model)
    assert names == ["pab", "cfg_truncation", "cross_attn_kv_cache", "vae"]
    assert kwargs["vae"] == {"tiling": False, "dtype": "bfloat16"} and kwargs["cfg_truncation"] == {"after_fraction": 0.4}
    stack = OptimizationStack(model, names, module_kwargs=kwargs)
    assert [m.name for m in stack.modules] == names and not stack.skipped


def test_without_a_vae_the_default_falls_back_to_the_plain_pab_plus_cfg_truncation_tier():
    from worldoptbench.defaults import recommended_config

    model = FakeDiffusion()
    del model.pipeline.vae
    assert recommended_config(model)[0] == ["pab", "cfg_truncation"]


def test_aggressive_uses_magcache_only_when_the_model_has_ratios_for_its_setup():
    from worldoptbench.defaults import recommended_config

    plain = FakeDiffusion()
    assert recommended_config(plain, aggressive=True)[0] == ["pab", "cfg_truncation", "cross_attn_kv_cache", "vae"]  # no ratios: falls back

    calibrated = FakeDiffusion()
    calibrated.mag_ratios = [1.0] * 8
    names, kwargs = recommended_config(calibrated, aggressive=True)
    assert names == ["magcache", "cfg_truncation", "cross_attn_kv_cache", "vae"]
    assert kwargs["magcache"] == {"threshold": 0.24} and kwargs["cfg_truncation"] == {"after_fraction": 0.6}
    assert recommended_config(calibrated)[0] == ["pab", "cfg_truncation", "cross_attn_kv_cache", "vae"]  # not aggressive: the default
