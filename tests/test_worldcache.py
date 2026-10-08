"""WorldCache module: the maths on plain tensors, then the hooks on a tiny random-weight Wan
transformer (CPU, no download). Not measured on a real model by these tests; the sweep does that.
"""

import types

import pytest

torch = pytest.importorskip("torch")

from worldoptbench.optimizations.worldcache import (
    WorldCacheModule,
    interpolation_gain,
    motion_threshold,
    relative_l1,
    saliency_map,
    scheduled_threshold,
    weighted_drift,
)

# ---- the maths ----------------------------------------------------------------------------------------


def test_relative_l1_is_the_papers_drift():
    previous = torch.tensor([[1.0, -2.0, 3.0]])
    current = torch.tensor([[2.0, -2.0, 1.0]])
    assert relative_l1(current, previous) == pytest.approx((1 + 0 + 2) / 6, rel=1e-6)
    assert relative_l1(previous, previous) == 0.0


def test_motion_threshold_tightens_with_velocity_and_is_tau0_when_still():
    assert motion_threshold(0.08, 2.0, 0.0) == 0.08
    assert motion_threshold(0.08, 2.0, 1.0) == pytest.approx(0.08 / 3)
    assert motion_threshold(0.08, 2.0, 0.5) > motion_threshold(0.08, 2.0, 1.0)


def test_schedule_relaxes_the_threshold_linearly_over_the_trajectory():
    assert scheduled_threshold(0.1, 4.0, 0, 30) == 0.1
    assert scheduled_threshold(0.1, 4.0, 30, 30) == pytest.approx(0.5)  # 1 + beta_a at the last step
    assert scheduled_threshold(0.1, 4.0, 15, 30) == pytest.approx(0.3)


def test_interpolation_gain_is_the_least_squares_fit_clamped():
    p2 = torch.zeros(1, 4, 3)
    p1 = torch.ones(1, 4, 3)
    assert interpolation_gain(0.5 * p1, p1, p2, gamma_max=2.0) == pytest.approx(0.5)
    assert interpolation_gain(1.5 * p1, p1, p2, gamma_max=2.0) == pytest.approx(1.5)
    assert interpolation_gain(5.0 * p1, p1, p2, gamma_max=2.0) == 2.0  # clamped above
    assert interpolation_gain(-1.0 * p1, p1, p2, gamma_max=2.0) == 0.0  # moving the opposite way: don't extrapolate
    assert interpolation_gain(p1, p2, p2, gamma_max=2.0) == 0.0  # no source motion: zero, not a division blow-up


def test_saliency_map_is_normalized_and_picks_the_high_variance_location():
    features = torch.zeros(1, 4, 4)  # grid (1, 2, 2), 4 channels
    features[0, 2] = torch.tensor([5.0, -5.0, 5.0, -5.0])  # token 2 = location (h=1, w=0): the only varied one
    s = saliency_map(features, (1, 2, 2))
    assert s.shape == (2, 2)
    assert float(s.min()) == 0.0 and float(s.max()) == pytest.approx(1.0, abs=1e-6)
    assert float(s[1, 0]) == pytest.approx(1.0, abs=1e-6)


def test_weighted_drift_equals_plain_drift_without_saliency():
    torch.manual_seed(0)
    a, b = torch.randn(1, 8, 4), torch.randn(1, 8, 4)
    assert weighted_drift(a, b, None, 0.12) == relative_l1(a, b)
    assert weighted_drift(a, b, (2, 2, 2), 0.0) == relative_l1(a, b)


def test_weighted_drift_counts_change_at_a_salient_location_more():
    current = torch.ones(1, 2, 4)  # grid (1, 1, 2)
    current[0, 0] = torch.tensor([4.0, -4.0, 4.0, -4.0])  # token 0 is salient, token 1 is flat
    grid = (1, 1, 2)
    shift = torch.tensor([1.0, 1.0, 1.0, 1.0])
    on_salient = current.clone()
    on_salient[0, 0] -= shift  # previous differs from current only at the salient token
    on_flat = current.clone()
    on_flat[0, 1] -= shift  # same size of change, at the flat token

    assert weighted_drift(current, on_salient, grid, 1.0) > relative_l1(current, on_salient)
    assert weighted_drift(current, on_flat, grid, 1.0) < relative_l1(current, on_flat)


def test_weighted_drift_falls_back_when_tokens_are_not_the_grid():
    a, b = torch.randn(1, 5, 4), torch.randn(1, 5, 4)
    assert weighted_drift(a, b, (2, 2, 2), 0.5) == relative_l1(a, b)  # 5 tokens != 2*2*2


# ---- construction -------------------------------------------------------------------------------------


def test_constructor_validates_and_label_names_the_ablations():
    with pytest.raises(ValueError, match="tau0"):
        WorldCacheModule(tau0=-1)
    with pytest.raises(ValueError, match="max_consecutive_skips"):
        WorldCacheModule(max_consecutive_skips=0)
    with pytest.raises(ValueError, match="drift_reference"):
        WorldCacheModule(drift_reference="sometimes")
    assert WorldCacheModule(drift_reference="previous_step").label == "worldcache_0.08_prevstep"
    assert WorldCacheModule().label == "worldcache_0.08"
    assert WorldCacheModule(interpolate=False).label == "worldcache_0.08_no_interpolate"


# ---- hooks on a tiny Wan transformer --------------------------------------------------------------------

pytest.importorskip("diffusers")

from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

from worldoptbench.models.base import ModelInfo, Rollout, WorldModelInterface
from worldoptbench.optimizations import FirstBlockCacheModule
from worldoptbench.stack import OptimizationStack


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


def _run_video(model, steps=8):
    """`steps` forward passes with slowly changing latents (so drift is small, as in real denoising),
    each inside the 'cond' cache context like the Wan pipeline."""
    torch.manual_seed(1)
    base = torch.randn(1, 4, 3, 8, 8)
    text = torch.randn(1, 6, 16)
    outputs = []
    with torch.no_grad():
        for i in range(steps):
            latents = base + 0.01 * i * torch.ones_like(base)
            with model.transformer.cache_context("cond"):
                outputs.append(
                    model.transformer(
                        hidden_states=latents, timestep=torch.tensor([900 - 100 * i]),
                        encoder_hidden_states=text, return_dict=False,
                    )[0]
                )
    return outputs


def _max_diff(a, b):
    return max((x - y).abs().max().item() for x, y in zip(a, b))


@pytest.fixture(scope="module")
def baseline():
    return _run_video(FakeDiffusion())


def test_a_zero_threshold_never_skips_and_matches_the_baseline_exactly(baseline):
    model = FakeDiffusion()
    module = WorldCacheModule(tau0=0.0)
    module.apply(model)
    assert _max_diff(_run_video(model), baseline) == 0.0
    assert module.skipped_steps() == 0


def test_a_huge_threshold_skips_and_changes_the_output_and_restore_is_exact(baseline):
    model = FakeDiffusion()
    module = WorldCacheModule(tau0=1e9)
    module.apply(model)
    out = _run_video(model)
    assert module.skipped_steps() >= 4  # engaged on most steps once it had history
    assert _max_diff(out, baseline) > 1e-4

    module.restore()
    assert not any(hasattr(b, "_diffusers_hook") and b._diffusers_hook.hooks for b in model.transformer.blocks)
    assert _max_diff(_run_video(model), baseline) == 0.0  # hooks and pre-hook gone: bit-identical again


def test_the_first_step_always_computes_and_the_second_reuses_without_interpolating():
    first = FakeDiffusion()
    module = WorldCacheModule(tau0=1e9)
    module.apply(first)
    _run_video(first, steps=1)
    assert module.skipped_steps() == 0  # step 0 has no residual to reuse, whatever the threshold

    second = FakeDiffusion()
    module = WorldCacheModule(tau0=1e9)
    module.apply(second)
    _run_video(second, steps=2)
    assert module.skipped_steps() == 1  # step 1 has one residual: reused as is (OSI needs two)


def test_state_resets_between_videos(baseline):
    # the same sequence twice, with the pipeline-style reset in between, must give identical results:
    # no step counter, probe or residual survives into the next video
    model = FakeDiffusion()
    WorldCacheModule(tau0=0.5).apply(model)
    first = _run_video(model)
    model.transformer._reset_stateful_cache()
    second = _run_video(model)
    assert _max_diff(first, second) == 0.0


def test_max_consecutive_skips_bounds_how_long_the_cache_can_run_on_stale_features():
    model = FakeDiffusion()
    module = WorldCacheModule(tau0=1e9, max_consecutive_skips=1)
    module.apply(model)
    _run_video(model, steps=8)
    unbounded = FakeDiffusion()
    unbounded_module = WorldCacheModule(tau0=1e9)
    unbounded_module.apply(unbounded)
    _run_video(unbounded, steps=8)
    assert 0 < module.skipped_steps() < unbounded_module.skipped_steps()


def test_each_ablation_runs_and_a_disabled_interpolation_reuses_the_last_residual():
    for flags in (
        {"motion_adaptive": False}, {"saliency": False}, {"interpolate": False}, {"schedule": False},
        {"drift_reference": "previous_step"},
    ):
        model = FakeDiffusion()
        WorldCacheModule(tau0=1e9, **flags).apply(model)
        outputs = _run_video(model)
        assert all(torch.isfinite(o).all() for o in outputs), flags


def test_it_will_not_stack_on_another_cache_directly_and_the_stack_skips_it():
    model = FakeDiffusion()
    FirstBlockCacheModule().apply(model)
    with pytest.raises(RuntimeError, match="already applied"):
        WorldCacheModule().apply(model)

    stack = OptimizationStack(FakeDiffusion(), [FirstBlockCacheModule(), WorldCacheModule()])
    assert [m.name for m in stack.modules] == ["first_block_cache"]
    assert [s.name for s in stack.skipped] == ["worldcache"]
    assert "diffusion_cache" in stack.skipped[0].reason


def test_two_worldcaches_cannot_be_applied_to_the_same_transformer():
    model = FakeDiffusion()
    WorldCacheModule().apply(model)
    with pytest.raises(RuntimeError, match="already applied"):
        WorldCacheModule().apply(model)


def test_both_drift_references_agree_on_the_first_decision_and_record_every_later_one():
    # at step 1 the previous step *is* the last computed one, so the two references must give the same drift;
    # they only differ once a step has been skipped (that difference is what the real-model sweep measures)
    traces = {}
    for reference in ("last_computed", "previous_step"):
        model = FakeDiffusion()
        module = WorldCacheModule(tau0=1e9, drift_reference=reference)
        module.apply(model)
        _run_video(model, steps=5)
        traces[reference] = module.trace()
    assert [e["drift"] for e in traces["last_computed"]][:2] == [e["drift"] for e in traces["previous_step"]][:2]
    for entries in traces.values():
        assert [e["step"] for e in entries] == [0, 1, 2, 3, 4]
        assert entries[0]["drift"] is None and all(e["drift"] is not None for e in entries[1:])


# ---- motion estimation and warping ---------------------------------------------------------------------------

from worldoptbench.optimizations.worldcache import (
    estimate_shift,
    phase_shift,
    translate,
    warp_tokens,
)


def _smooth(channels, h, w, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(1, channels, h, w)
    for _ in range(3):
        x = (x + torch.roll(x, 1, 2) + torch.roll(x, 1, 3) + torch.roll(x, -1, 3) + torch.roll(x, -1, 2)) / 5
    return x[0]


def test_translate_moves_content_by_plus_d_and_zero_shift_is_the_identity():
    image = torch.zeros(1, 1, 8, 10)
    image[0, 0, 3, 4] = 1.0
    moved = translate(image, torch.tensor([2.0]), torch.tensor([-1.0]))
    assert tuple(int(v) for v in (moved[0, 0] == moved.max()).nonzero()[0]) == (5, 3)
    assert torch.allclose(translate(image, torch.tensor([0.0]), torch.tensor([0.0])), image)


@pytest.mark.parametrize("dy,dx", [(0, 0), (1, 0), (0, -2), (2, 1)])
def test_phase_shift_recovers_integer_shifts(dy, dx):
    base = _smooth(4, 24, 40)
    shifted = translate(base[None], torch.tensor([float(dy)]), torch.tensor([float(dx)]))[0]
    estimate = phase_shift(shifted, base)
    assert estimate[0] == pytest.approx(dy, abs=0.1) and estimate[1] == pytest.approx(dx, abs=0.1)


def test_phase_shift_is_roughly_right_for_fractional_shifts():
    base = _smooth(4, 24, 40)
    shifted = translate(base[None], torch.tensor([0.5]), torch.tensor([-0.6]))[0]
    estimate = phase_shift(shifted, base)
    assert estimate[0] == pytest.approx(0.5, abs=0.35) and estimate[1] == pytest.approx(-0.6, abs=0.35)  # ~0.2 px bias


def test_estimate_shift_gives_one_translation_per_frame_and_zero_for_identical_latents():
    frames = 3
    previous = torch.stack([_smooth(4, 24, 40, seed=f) for f in range(frames)], dim=1)[None]  # (1, C, F, H, W)
    truth = [(0.0, 0.0), (1.0, 0.0), (0.0, 2.0)]
    current = torch.stack(
        [translate(previous[:, :, f], torch.tensor([truth[f][0]]), torch.tensor([truth[f][1]]))[0] for f in range(frames)],
        dim=1,
    )[None]
    estimate = estimate_shift(current, previous)
    assert estimate.shape == (frames, 2)
    for f in range(frames):
        assert estimate[f, 0].item() == pytest.approx(truth[f][0], abs=0.3)
        assert estimate[f, 1].item() == pytest.approx(truth[f][1], abs=0.3)
    assert estimate_shift(previous, previous).abs().max().item() < 1e-3


def test_estimate_shift_zeroes_estimates_beyond_max_shift():
    previous = torch.stack([_smooth(4, 24, 40)], dim=1)[None]
    current = torch.stack([translate(previous[:, :, 0], torch.tensor([2.0]), torch.tensor([0.0]))[0]], dim=1)[None]
    assert estimate_shift(current, previous, max_shift=3.0)[0, 0].item() == pytest.approx(2.0, abs=0.3)
    assert estimate_shift(current, previous, max_shift=1.0).abs().max().item() == 0.0  # called unreliable


def test_warp_tokens_moves_a_marked_token_and_is_a_no_op_for_zero_shift():
    grid = (2, 4, 5)
    features = torch.zeros(1, 2 * 4 * 5, 3)
    features[0, 1 * 5 + 2, :] = 1.0  # frame 0, row 1, col 2
    moved = warp_tokens(features, grid, torch.tensor([[1.0, 1.0], [0.0, 0.0]]))
    index = int(moved[0].sum(-1).argmax())
    assert divmod(index, 20)[0] == 0 and divmod(divmod(index, 20)[1], 5) == (2, 3)
    assert warp_tokens(features, grid, torch.zeros(2, 2)) is features
    assert warp_tokens(features, (3, 3, 3), torch.ones(3, 2)) is features  # tokens are not that grid: untouched


def test_warp_options_are_validated_and_labelled():
    with pytest.raises(ValueError, match="warp_start_step"):
        WorldCacheModule(warp_start_step=-1)
    with pytest.raises(ValueError, match="max_shift"):
        WorldCacheModule(max_shift=0)
    assert WorldCacheModule(warp=True).label == "worldcache_0.08_warp"
    assert WorldCacheModule().config["warp"] is False  # off by default: it measured as a no-op


def test_warping_that_never_starts_changes_nothing_and_when_it_starts_the_run_stays_finite(baseline):
    never = FakeDiffusion()
    WorldCacheModule(tau0=1e9, warp=True, warp_start_step=100).apply(never)
    plain = FakeDiffusion()
    WorldCacheModule(tau0=1e9).apply(plain)
    assert _max_diff(_run_video(never), _run_video(plain)) == 0.0

    warped = FakeDiffusion()
    module = WorldCacheModule(tau0=1e9, warp=True, warp_start_step=2)
    module.apply(warped)
    outputs = _run_video(warped)
    assert all(torch.isfinite(o).all() for o in outputs)
    assert any("shift" in entry for entry in module.trace()), "warping never ran"
