"""Constraints, their use in the stack and in autotune, time to first frame, and the profile block."""

import pytest

from worldoptbench.autotune import Evaluation, autotune
from worldoptbench.constraints import Constraints, Measured, check_constraints, measure_results
from worldoptbench.metrics.speed import measure_speed
from worldoptbench.models.base import ModelInfo, Rollout, WorldModelInterface
from worldoptbench.reporting import format_profile
from worldoptbench.stack import OptimizationStack


def _row(speedup=2.0, physics=0.9, vram_peak=3.0, vram_reserved=4.0, latency=2.0, horizon=2.0, paes=1.5):
    return {
        "model_name": "fake", "architecture": "diffusion", "prompt_id": "p", "domain": "d", "horizon": horizon,
        "speed": {"speedup": speedup, "latency_seconds": latency, "vram_peak_gb": vram_peak, "vram_reserved_gb": vram_reserved},
        "visual": {"psnr": 20.0, "temporal_consistency": 0.9},
        "physics": {"sim_fidelity_score": physics}, "paes": paes, "optimization": "stack",
    }


# ---- Constraints and measuring ---------------------------------------------------------------------


def test_constraints_validate_their_values():
    with pytest.raises(ValueError, match="target_speedup"):
        Constraints(target_speedup=0)
    with pytest.raises(ValueError, match="min_physics_score"):
        Constraints(min_physics_score=1.5)
    assert Constraints() == Constraints(None, None, None, None)


def test_measure_results_means_speedup_and_physics_and_takes_the_largest_memory():
    measured = measure_results([_row(speedup=1.0, physics=0.8, vram_reserved=2.0, vram_peak=5.0),
                                _row(speedup=3.0, physics=0.6, vram_reserved=7.0, vram_peak=1.0)])
    assert measured.speedup == pytest.approx(2.0) and measured.physics == pytest.approx(0.7)
    assert measured.vram_gb == 7.0  # the most any rollout needed, reserved or peak
    assert measured.latency_seconds == pytest.approx(2.0)


def test_measure_results_accepts_objects_as_well_as_dicts():
    from worldoptbench.runner import RunResult

    result = RunResult(model_name="m", architecture="a", prompt_id="p", domain="d", horizon=1.0,
                       speed={"speedup": 2.5, "latency_seconds": 1.0, "vram_peak_gb": 1.0}, visual={}, physics={"sim_fidelity_score": 0.5})
    measured = measure_results([result])
    assert measured.speedup == 2.5 and measured.physics == 0.5 and measured.vram_gb == 1.0


def test_unmeasured_quantities_are_none_not_zero():
    measured = measure_results([{"speed": {"latency_seconds": 1.0}, "horizon": 1.0}])
    assert measured.physics is None and measured.speedup is None and measured.vram_gb is None
    assert measure_results([]) == Measured()


# ---- checking -------------------------------------------------------------------------------------------


def test_a_run_that_meets_everything_passes():
    report = check_constraints(Constraints(min_physics_score=0.85, target_speedup=2.0, max_vram_gb=8),
                               Measured(physics=0.9, speedup=2.4, vram_gb=6.0, latency_seconds=1.0))
    assert report.ok and report.violations == [] and "met" in report.format()


def test_each_broken_constraint_is_reported_with_its_numbers():
    report = check_constraints(Constraints(min_physics_score=0.85, target_speedup=2.0, max_vram_gb=8, max_latency_seconds=1.0),
                               Measured(physics=0.7, speedup=1.5, vram_gb=9.0, latency_seconds=2.0))
    assert not report.ok
    assert {v.name for v in report.violations} == {"min_physics_score", "target_speedup", "max_vram_gb", "max_latency_seconds"}
    text = report.format()
    assert "NOT met" in text and "physics 0.7" in text and "9.0 GB" in text


def test_a_constraint_that_could_not_be_measured_is_a_violation_not_a_pass():
    report = check_constraints(Constraints(min_physics_score=0.85, max_vram_gb=8), Measured(speedup=2.0))
    assert not report.ok
    assert all("could not be verified" in v.message for v in report.violations)


def test_constraints_you_did_not_set_are_not_checked():
    assert check_constraints(Constraints(), Measured()).ok


# ---- the stack ---------------------------------------------------------------------------------------------


class Plain(WorldModelInterface):
    def generate(self, prompt=None, init_frame=None, init_video=None, actions=None, horizon=4.0, **kwargs):
        return Rollout(frames=[], fps=1.0)

    def get_info(self):
        return ModelInfo(name="plain", architecture="diffusion", param_count=0)


def test_stack_checks_a_measured_run_against_its_constraints():
    stack = OptimizationStack(Plain(), [], constraints=Constraints(target_speedup=2.0))
    assert stack.check([_row(speedup=2.5)]).ok
    assert not stack.check([_row(speedup=1.2)]).ok


def test_stack_without_constraints_refuses_to_check():
    with pytest.raises(ValueError, match="no constraints"):
        OptimizationStack(Plain(), []).check([_row()])


# ---- autotune under constraints --------------------------------------------------------------------------


def _evaluator(table):
    """table: tuple of module names -> Evaluation; the empty tuple is the baseline."""
    return lambda modules: table[tuple(modules)]


def test_autotune_rejects_stacks_over_the_vram_limit():
    table = {
        (): Evaluation(paes=1.0, physics=0.9, speedup=1.0, vram_gb=2.0),
        ("a",): Evaluation(paes=3.0, physics=0.9, speedup=3.0, vram_gb=20.0),  # best PAES but too big
        ("b",): Evaluation(paes=1.6, physics=0.9, speedup=1.6, vram_gb=4.0),
        ("b", "a"): Evaluation(paes=3.5, physics=0.9, speedup=3.5, vram_gb=22.0),  # adding the big one is rejected too
    }
    result = autotune(["a", "b"], _evaluator(table), constraints=Constraints(max_vram_gb=8.0))
    assert result.best_modules == ["b"]
    assert any("vram 20.00 GB exceeds" in step.note for step in result.trace)
    assert result.report.ok and result.report.violations == []  # the chosen stack fits, and nothing else was required


def test_autotune_stops_adding_modules_once_the_speedup_target_is_met():
    table = {
        (): Evaluation(paes=1.0, physics=0.9, speedup=1.0),
        ("a",): Evaluation(paes=2.2, physics=0.9, speedup=2.2),
        ("a", "b"): Evaluation(paes=3.0, physics=0.9, speedup=3.0),
        ("b",): Evaluation(paes=1.5, physics=0.9, speedup=1.5),
    }
    unconstrained = autotune(["a", "b"], _evaluator(table))
    assert unconstrained.best_modules == ["a", "b"]
    stopped = autotune(["a", "b"], _evaluator(table), constraints=Constraints(target_speedup=2.0))
    assert stopped.best_modules == ["a"] and stopped.report.ok  # 2.2x clears the target; no reason to push on


def test_autotune_reports_which_constraint_the_best_stack_still_misses():
    table = {
        (): Evaluation(paes=1.0, physics=0.9, speedup=1.0),
        ("a",): Evaluation(paes=1.4, physics=0.9, speedup=1.4),
    }
    result = autotune(["a"], _evaluator(table), constraints=Constraints(target_speedup=3.0))
    assert result.best_modules == ["a"] and not result.report.ok
    assert [v.name for v in result.report.violations] == ["target_speedup"]


def test_autotune_applies_the_constraints_physics_floor():
    table = {
        (): Evaluation(paes=1.0, physics=0.9, speedup=1.0),
        ("a",): Evaluation(paes=3.0, physics=0.5, speedup=3.0),
    }
    result = autotune(["a"], _evaluator(table), max_physics_drop=1.0, constraints=Constraints(min_physics_score=0.8))
    assert result.best_modules == [] and any("below the floor" in s.note for s in result.trace)


def test_autotune_without_constraints_is_unchanged_and_has_no_report():
    table = {(): Evaluation(paes=1.0, physics=0.9), ("a",): Evaluation(paes=2.0, physics=0.9)}
    result = autotune(["a"], _evaluator(table))
    assert result.best_modules == ["a"] and result.report is None


# ---- time to first frame ------------------------------------------------------------------------------------


def test_time_to_first_frame_defaults_to_the_latency_for_a_model_that_returns_at_the_end():
    _, speed = measure_speed(lambda: Rollout(frames=[1, 2, 3], fps=1.0))
    assert speed.time_to_first_frame_seconds == speed.latency_seconds


def test_a_streaming_model_can_report_a_smaller_time_to_first_frame():
    _, speed = measure_speed(lambda: Rollout(frames=[1], fps=1.0, metadata={"time_to_first_frame_seconds": 0.001}))
    assert speed.time_to_first_frame_seconds == 0.001


# ---- the profile block ----------------------------------------------------------------------------------------


def test_profile_block_has_the_sections_of_the_outline_and_says_n_a_for_what_is_missing():
    text = format_profile([_row(speedup=2.0, physics=0.8, paes=1.7), _row(speedup=3.0, physics=0.6, paes=2.1)], hardware="TestGPU")
    for expected in ("Model:        fake", "Architecture: diffusion", "Optimization: stack", "Hardware:     TestGPU",
                     "Speed:        2.50x speedup", "FVD n/a", "PSNR 20.0", "Physics:      0.700", "Drift onset: n/a",
                     "PAES Score:   1.900"):
        assert expected in text, expected
    assert format_profile([]) == "(no results)"


def test_profile_reports_per_generated_second_latency_and_perceptual_scores_when_present():
    row = _row(latency=4.0, horizon=2.0)
    row["visual"]["perceptual"] = {"dino_similarity": 0.87}
    row["visual"]["style_deviation"] = 0.12
    text = format_profile([row], hardware="G")
    assert "2.00 s/s generation" in text and "DINO similarity 0.870" in text and "style deviation 0.120" in text
