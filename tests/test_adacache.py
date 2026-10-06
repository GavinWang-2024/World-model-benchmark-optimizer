"""AdaCache module: the codebook and motion maths on plain values, then the hooks on a tiny
random-weight Wan transformer (CPU, no download). The real-model behaviour is measured by the sweep.
"""

import types

import pytest

torch = pytest.importorskip("torch")

from worldoptbench.optimizations.adacache import PRESETS, AdaCacheModule, codebook_rate, motion_score  # noqa: E402

# ---- the maths ---------------------------------------------------------------------------------------


def test_codebook_rate_picks_the_first_threshold_the_distance_is_under():
    book = PRESETS["fast30"]  # {0.08: 6, 0.16: 5, 0.24: 4, 0.32: 3, 0.40: 2, 1.00: 1}
    assert codebook_rate(0.0, book) == 6
    assert codebook_rate(0.079, book) == 6
    assert codebook_rate(0.08, book) == 5  # strictly under: a distance equal to a threshold moves to the next
    assert codebook_rate(0.3, book) == 3
    assert codebook_rate(0.9, book) == 1


def test_codebook_rate_is_one_beyond_the_last_threshold_and_ignores_dict_order():
    assert codebook_rate(5.0, {0.5: 4}) == 1  # nothing matched: recompute next step
    assert codebook_rate(0.1, {1.0: 1, 0.2: 3}) == 3  # sorted by threshold, not insertion order


def test_presets_are_well_formed():
    for name, book in PRESETS.items():
        assert book and all(rate >= 1 for rate in book.values()), name
        rates = [book[t] for t in sorted(book)]
        assert rates == sorted(rates, reverse=True), f"{name}: larger distances must not get longer rates"


def test_motion_score_is_zero_for_a_static_latent_and_the_mean_frame_difference_otherwise():
    static = torch.ones(1, 2, 4, 3, 3)
    assert motion_score(static) == 0.0
    ramp = torch.zeros(1, 1, 4, 2, 2)
    for f in range(4):
        ramp[:, :, f] = 2.0 * f  # every consecutive pair differs by 2
    assert motion_score(ramp) == pytest.approx(2.0)
    assert motion_score(ramp, frame_step=2) == pytest.approx(4.0)
    assert motion_score(torch.ones(1, 1, 1, 2, 2)) == 0.0  # one frame: nothing to difference
    assert motion_score(torch.ones(4, 4)) == 0.0  # not a video latent


# ---- construction -----------------------------------------------------------------------------------


def test_constructor_validates_and_labels():
    with pytest.raises(ValueError, match="preset"):
        AdaCacheModule(preset="turbo")
    with pytest.raises(ValueError, match="distance_scale"):
        AdaCacheModule(distance_scale=0)
    with pytest.raises(ValueError, match="rate"):
        AdaCacheModule(codebook={0.1: 0})
    assert AdaCacheModule().label == "adacache_fast30"
    assert AdaCacheModule(preset="slow30", distance_scale=0.5, moreg=True).label == "adacache_slow30_x0.5_moreg"
    assert AdaCacheModule(codebook={1.0: 2}).label == "adacache_custom"


# ---- hooks on a tiny Wan transformer ------------------------------------------------------------------

pytest.importorskip("diffusers")

from diffusers.models.transformers.transformer_wan import WanTransformer3DModel  # noqa: E402

from worldoptbench.models.base import ModelInfo, Rollout, WorldModelInterface  # noqa: E402
from worldoptbench.optimizations import FirstBlockCacheModule  # noqa: E402
from worldoptbench.stack import OptimizationStack  # noqa: E402


def tiny_transformer():
    torch.manual_seed(0)
    return WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=8, in_channels=4, out_channels=4,
        text_dim=16, freq_dim=16, ffn_dim=32, num_layers=4, cross_attn_norm=True,
        qk_norm="rms_norm_across_heads", eps=1e-6, image_dim=None, added_kv_proj_dim=None, rope_max_seq_len=64,
    ).eval()


class FakeDiffusion(WorldModelInterface):
    def __init__(self):
        self.transformer = tiny_transformer()
        self.pipeline = types.SimpleNamespace(current_timestep=500)
        self.step_callbacks = []
        self.num_inference_steps = 8
        self.cfg_batched = False

    def generate(self, prompt=None, init_frame=None, init_video=None, actions=None, horizon=4.0, **kwargs):
        return Rollout(frames=[], fps=1.0)

    def get_info(self):
        return ModelInfo(name="fake-wan", architecture="diffusion", param_count=0)


def _run_video(model, steps=8, contexts=("cond",)):
    torch.manual_seed(1)
    base = torch.randn(1, 4, 3, 8, 8)
    text = torch.randn(1, 6, 16)
    outputs = []
    with torch.no_grad():
        for i in range(steps):
            for context in contexts:
                with model.transformer.cache_context(context):
                    outputs.append(
                        model.transformer(
                            hidden_states=base + 0.01 * i, timestep=torch.tensor([900 - 100 * i]),
                            encoder_hidden_states=text, return_dict=False,
                        )[0]
                    )
    return outputs


def _max_diff(a, b):
    return max((x - y).abs().max().item() for x, y in zip(a, b))


def _pattern(module, context="cond"):
    return "".join("C" if e["compute"] else "." for e in module.trace(context))


@pytest.fixture(scope="module")
def baseline():
    return _run_video(FakeDiffusion())


def test_a_fixed_rate_codebook_gives_the_expected_compute_and_reuse_pattern(baseline):
    # steps 0 and 1 always compute (a distance needs two computed steps); then rate 3 = compute, reuse, reuse
    model = FakeDiffusion()
    module = AdaCacheModule(codebook={1e9: 3})
    module.apply(model)
    outputs = _run_video(model)
    assert _pattern(module) == "CC..C..C"
    assert module.reused_steps() == 4
    assert _max_diff(outputs, baseline) > 1e-4  # reuse really changed the output


def test_a_codebook_that_never_caches_matches_the_baseline_exactly(baseline):
    model = FakeDiffusion()
    module = AdaCacheModule(codebook={1e-12: 3, 1e9: 1})  # every real distance falls through to rate 1
    module.apply(model)
    assert _max_diff(_run_video(model), baseline) == 0.0
    assert module.reused_steps() == 0 and _pattern(module) == "C" * 8


def test_restore_removes_every_hook_and_is_exact(baseline):
    model = FakeDiffusion()
    module = AdaCacheModule(codebook={1e9: 3})
    module.apply(model)
    _run_video(model)
    module.restore()
    assert not any(
        getattr(m, "_diffusers_hook", None) is not None and m._diffusers_hook.hooks for m in model.transformer.modules()
    )
    assert _max_diff(_run_video(model), baseline) == 0.0
    module.restore()  # idempotent


def test_state_resets_between_videos():
    model = FakeDiffusion()
    AdaCacheModule(codebook={1e9: 3}).apply(model)
    first = _run_video(model)
    model.transformer._reset_stateful_cache()
    assert _max_diff(first, _run_video(model)) == 0.0


def test_max_rate_caps_what_the_codebook_asks_for():
    model = FakeDiffusion()
    module = AdaCacheModule(codebook={1e9: 6}, max_rate=2)
    module.apply(model)
    _run_video(model)
    rates = [e["rate"] for e in module.trace() if e["rate"] is not None]
    assert rates[0] == 1 and max(rates) == 2


def test_the_distance_is_recorded_and_scaled_by_distance_scale():
    plain, scaled = FakeDiffusion(), FakeDiffusion()
    module_plain = AdaCacheModule(codebook={1e9: 3})
    module_scaled = AdaCacheModule(codebook={1e9: 3}, distance_scale=10.0)
    module_plain.apply(plain)
    module_scaled.apply(scaled)
    _run_video(plain)
    _run_video(scaled)
    first_plain = next(e["distance"] for e in module_plain.trace() if e["distance"] is not None)
    first_scaled = next(e["distance"] for e in module_scaled.trace() if e["distance"] is not None)
    assert first_plain > 0 and first_scaled == pytest.approx(10.0 * first_plain, rel=1e-4)


def test_cfg_contexts_keep_separate_schedules_and_caches():
    model = FakeDiffusion()
    module = AdaCacheModule(codebook={1e9: 3})
    module.apply(model)
    _run_video(model, contexts=("cond", "uncond"))
    assert _pattern(module, "cond") == _pattern(module, "uncond") == "CC..C..C"
    assert module.reused_steps() == 8  # four per context


def test_moreg_runs_and_records_the_motion_score():
    model = FakeDiffusion()
    module = AdaCacheModule(codebook={1e9: 3}, moreg=True)
    module.apply(model)
    outputs = _run_video(model)
    assert all(torch.isfinite(o).all() for o in outputs)
    assert all("motion" in e for e in module.trace() if e["compute"])


def test_it_will_not_stack_on_another_cache_and_the_stack_skips_it():
    model = FakeDiffusion()
    FirstBlockCacheModule().apply(model)
    with pytest.raises(RuntimeError, match="already applied"):
        AdaCacheModule().apply(model)

    again = FakeDiffusion()
    AdaCacheModule().apply(again)
    with pytest.raises(RuntimeError, match="already applied"):
        AdaCacheModule().apply(again)

    stack = OptimizationStack(FakeDiffusion(), [AdaCacheModule(), FirstBlockCacheModule()])
    assert [m.name for m in stack.modules] == ["adacache"]
    assert [s.name for s in stack.skipped] == ["first_block_cache"]


def test_a_transformer_without_blocks_is_refused_loudly():
    model = FakeDiffusion()
    model.transformer = torch.nn.Sequential(torch.nn.Linear(2, 2))
    with pytest.raises(Exception):  # no cache_context support or no matching modules: never a silent no-op
        AdaCacheModule().apply(model)


# ---- the drift guard ---------------------------------------------------------------------------------


def test_a_guard_that_never_trips_keeps_the_schedule_and_only_freshens_the_probe_block(baseline):
    plain, guarded = FakeDiffusion(), FakeDiffusion()
    module_plain = AdaCacheModule(codebook={1e9: 3})
    module_guarded = AdaCacheModule(codebook={1e9: 3}, guard=1e9)
    module_plain.apply(plain)
    module_guarded.apply(guarded)
    out_plain, out_guarded = _run_video(plain), _run_video(guarded)
    assert _pattern(module_guarded) == _pattern(module_plain) == "CC..C..C"
    assert module_guarded.vetoed_steps() == 0
    # the probe block is recomputed on every step instead of reused, so the output differs a little from plain reuse
    assert 0.0 < _max_diff(out_plain, out_guarded) < 1.0
    drifts = [e["probe_drift"] for e in module_guarded.trace() if not e["compute"]]
    assert drifts and all(d is not None and d >= 0 for d in drifts)


def test_a_guard_that_always_trips_vetoes_every_scheduled_reuse_and_reproduces_the_baseline(baseline):
    model = FakeDiffusion()
    module = AdaCacheModule(codebook={1e9: 3}, guard=1e-12)
    module.apply(model)
    outputs = _run_video(model)
    assert _pattern(module) == "C" * 8  # nothing was reused
    assert module.vetoed_steps() == 6  # steps 2..7 were all scheduled to reuse and all vetoed
    assert _max_diff(outputs, baseline) == 0.0  # recomputing everything is exactly the unoptimized model


def test_the_guard_only_vetoes_steps_that_were_scheduled_to_reuse():
    model = FakeDiffusion()
    module = AdaCacheModule(codebook={1e9: 3}, guard=1e-12)
    module.apply(model)
    _run_video(model)
    for entry in module.trace():
        if entry["vetoed"]:
            assert entry["step"] >= 2 and entry["compute"] and entry["probe_drift"] is not None
    assert [e["step"] for e in module.trace() if e["vetoed"]] == [2, 3, 4, 5, 6, 7]


def test_the_guard_keeps_separate_state_per_cfg_context():
    model = FakeDiffusion()
    module = AdaCacheModule(codebook={1e9: 3}, guard=1e-12)
    module.apply(model)
    _run_video(model, contexts=("cond", "uncond"))
    assert module.vetoed_steps() == 12 and _pattern(module, "cond") == _pattern(module, "uncond") == "C" * 8


def test_guard_options_are_validated_and_labelled():
    with pytest.raises(ValueError, match="guard"):
        AdaCacheModule(guard=0)
    with pytest.raises(ValueError, match="guard"):
        AdaCacheModule(guard=0.1, guard_blocks=0)
    assert AdaCacheModule(guard=0.25).label == "adacache_fast30_guard0.25"
    with pytest.raises(ValueError, match="guard_blocks"):
        AdaCacheModule(guard=0.1, guard_blocks=4).apply(FakeDiffusion())  # all 4 blocks probed: nothing left to cache


def test_without_a_guard_there_are_no_probe_fields_in_the_trace():
    model = FakeDiffusion()
    module = AdaCacheModule(codebook={1e9: 3})
    module.apply(model)
    _run_video(model)
    assert all("probe_drift" not in e and "vetoed" not in e for e in module.trace())
    assert module.vetoed_steps() == 0
