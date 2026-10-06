"""The library layer: module requirements / compatibility, `library_report`,
`OptimizationStack` skipping with reasons and restoring global state, and the
greedy `autotune` search. All CPU-only: models and modules are fakes, and the
search runs against a scripted `evaluate`.
"""

import pytest

from worldoptbench.autotune import Evaluation, autotune, default_candidates
from worldoptbench.models.base import ModelInfo, Rollout, WorldModelInterface
from worldoptbench.optimizations import base
from worldoptbench.optimizations.base import OptimizationModule, library_report, register_module
from worldoptbench.stack import OptimizationStack


class FakeModel(WorldModelInterface):
    def __init__(self, architecture="autoregressive", **attrs):
        self.architecture = architecture
        for key, value in attrs.items():
            setattr(self, key, value)

    def generate(self, prompt=None, init_frame=None, init_video=None, actions=None, horizon=4.0, **kwargs):
        return Rollout(frames=[], fps=1.0)

    def get_info(self):
        return ModelInfo(name="fake", architecture=self.architecture, param_count=0)


class NeedsHook(OptimizationModule):
    name = "needs_hook"
    supported_architectures = ("autoregressive", "diffusion")
    requires = ("special_hook",)
    maturity = "measured"
    summary = "needs a hook"

    def apply(self, model):
        return model


class NeedsCuda(OptimizationModule):
    name = "needs_cuda"
    supported_architectures = ("autoregressive",)
    needs_cuda = True

    def apply(self, model):
        return model


class NeedsPackage(OptimizationModule):
    name = "needs_package"
    supported_architectures = ("autoregressive",)
    needs_packages = ("definitely_not_an_installed_package_xyz",)

    def apply(self, model):
        return model


class DiffusionOnly(OptimizationModule):
    name = "diffusion_only"
    supported_architectures = ("diffusion",)

    def apply(self, model):
        return model


class Plain(OptimizationModule):
    name = "plain"
    supported_architectures = ("autoregressive", "diffusion")

    def apply(self, model):
        return model


class WithRestore(Plain):
    name = "with_restore"

    def __init__(self):
        self.restored = 0

    def restore(self):
        self.restored += 1


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(base, "_REGISTRY", {})


# ---- compatibility -------------------------------------------------------------------


def test_compatible_module_has_no_incompatibility():
    assert Plain().incompatibility(FakeModel()) is None
    assert NeedsHook().incompatibility(FakeModel(special_hook=object())) is None


def test_architecture_mismatch_is_reported_with_both_architectures():
    reason = DiffusionOnly().incompatibility(FakeModel(architecture="autoregressive"))
    assert "diffusion" in reason and "autoregressive" in reason


def test_missing_hook_is_named():
    reason = NeedsHook().incompatibility(FakeModel())
    assert "special_hook" in reason and "needs_hook" in reason


def test_cuda_requirement(monkeypatch):
    monkeypatch.setattr(base, "_cuda_available", lambda: False)
    assert "CUDA" in NeedsCuda().incompatibility(FakeModel())
    monkeypatch.setattr(base, "_cuda_available", lambda: True)
    assert NeedsCuda().incompatibility(FakeModel()) is None


def test_package_requirement_is_checked_without_importing():
    reason = NeedsPackage().incompatibility(FakeModel())
    assert "definitely_not_an_installed_package_xyz" in reason and "installed" in reason


def test_architecture_is_checked_before_other_requirements():
    # A diffusion-only module on a model with no hooks should say "architecture", the most basic problem.
    class BothWrong(DiffusionOnly):
        requires = ("special_hook",)

    assert "supports" in BothWrong().incompatibility(FakeModel(architecture="autoregressive"))


# ---- library_report -----------------------------------------------------------------------


def test_library_report_lists_every_registered_module_with_applicability(monkeypatch):
    monkeypatch.setattr(base, "_cuda_available", lambda: False)
    for cls in (NeedsHook, NeedsCuda, Plain):
        register_module(cls)

    reports = {r.name: r for r in library_report(FakeModel())}

    assert set(reports) == {"needs_hook", "needs_cuda", "plain"}
    assert reports["plain"].applicable is True and reports["plain"].reason is None
    assert reports["needs_hook"].applicable is False and "special_hook" in reports["needs_hook"].reason
    assert reports["needs_cuda"].applicable is False
    assert reports["needs_hook"].maturity == "measured" and reports["needs_hook"].summary == "needs a hook"
    assert reports["plain"].maturity == "experimental"  # the default until a module is swept


def test_library_report_without_a_model_does_not_claim_applicability():
    register_module(Plain)
    (report,) = library_report()
    assert report.applicable is None and report.reason is None


def test_default_candidates_are_only_the_applicable_modules(monkeypatch):
    monkeypatch.setattr(base, "_cuda_available", lambda: False)
    for cls in (NeedsHook, NeedsCuda, Plain, DiffusionOnly):
        register_module(cls)
    assert default_candidates(FakeModel(special_hook=1)) == ["needs_hook", "plain"]


# ---- the stack uses compatibility ------------------------------------------------------------


def test_stack_skips_a_module_missing_its_hook_instead_of_crashing():
    stack = OptimizationStack(FakeModel(), [NeedsHook(), Plain()])
    assert [m.name for m in stack.modules] == ["plain"]
    assert stack.skipped[0].name == "needs_hook" and "special_hook" in stack.skipped[0].reason
    stack.apply()  # nothing to trip over


def test_whole_library_can_be_handed_to_the_stack_safely(monkeypatch):
    monkeypatch.setattr(base, "_cuda_available", lambda: False)
    for cls in (NeedsHook, NeedsCuda, NeedsPackage, DiffusionOnly, Plain):
        register_module(cls)
    stack = OptimizationStack(FakeModel(), base.available_modules())
    assert [m.name for m in stack.modules] == ["plain"]
    assert {s.name for s in stack.skipped} == {"needs_hook", "needs_cuda", "needs_package", "diffusion_only"}


def test_stack_restore_calls_restore_on_modules_that_define_it():
    with_restore = WithRestore()
    stack = OptimizationStack(FakeModel(), [with_restore, Plain()])
    stack.apply()
    stack.restore()
    assert with_restore.restored == 1


# ---- autotune --------------------------------------------------------------------------------------


def scripted(scores, physics=None):
    """An `evaluate` backed by a table: stack (as a frozenset) -> PAES, with optional physics."""
    calls = []
    physics = physics or {}

    def evaluate(stack):
        calls.append(list(stack))
        key = frozenset(stack)
        return Evaluation(paes=scores[key], physics=physics.get(key, 0.80))

    evaluate.calls = calls
    return evaluate


def test_autotune_with_no_candidates_returns_the_baseline():
    result = autotune([], scripted({frozenset(): 1.0}))
    assert result.best_modules == [] and result.best.paes == 1.0 and result.trace == []


def test_autotune_builds_the_best_stack_one_module_at_a_time():
    scores = {
        frozenset(): 1.0,
        frozenset("A"): 2.0, frozenset("B"): 1.05, frozenset("C"): 1.5,
        frozenset("AB"): 2.05, frozenset("AC"): 2.6,
        frozenset("ACB"): 2.62,
    }
    result = autotune(["A", "B", "C"], scripted(scores))

    assert result.best_modules == ["A", "C"]  # B never clears the 10% margin over A+C
    assert result.best.paes == 2.6 and result.baseline.paes == 1.0
    accepted = [step.modules for step in result.trace if step.accepted]
    assert accepted == [["A"], ["A", "C"]]


def test_autotune_requires_the_gain_to_beat_the_margin():
    scores = {frozenset(): 1.0, frozenset("A"): 1.05}
    assert autotune(["A"], scripted(scores)).best_modules == []  # +5% < 10%
    assert autotune(["A"], scripted(scores), min_gain=0.02).best_modules == ["A"]


def test_autotune_rejects_a_speedup_that_costs_physics():
    scores = {frozenset(): 1.0, frozenset("A"): 5.0, frozenset("B"): 1.5, frozenset("AB"): 6.0}
    physics = {frozenset("A"): 0.50, frozenset("AB"): 0.50}  # A wrecks physics wherever it appears; baseline is 0.80
    result = autotune(["A", "B"], scripted(scores, physics))

    assert result.best_modules == ["B"]
    rejected = [step for step in result.trace if step.modules == ["A"]]
    assert rejected and "rejected" in rejected[0].note and not rejected[0].accepted


def test_autotune_tolerates_small_physics_loss_and_honors_an_absolute_floor():
    scores = {frozenset(): 1.0, frozenset("A"): 2.0}
    small_loss = {frozenset("A"): 0.79}  # 0.01 below baseline: within the default 0.02
    assert autotune(["A"], scripted(scores, small_loss)).best_modules == ["A"]
    assert autotune(["A"], scripted(scores, small_loss), min_physics=0.795).best_modules == []


def test_autotune_stops_at_max_modules_and_ignores_duplicates():
    scores = {
        frozenset(): 1.0,
        frozenset("A"): 2.0, frozenset("B"): 1.4,
        frozenset("AB"): 4.0,
    }
    evaluate = scripted(scores)
    assert autotune(["A", "B", "A"], evaluate, max_modules=1).best_modules == ["A"]
    assert autotune(["A", "B", "A"], scripted(scores)).best_modules == ["A", "B"]
    # duplicates are evaluated once per round, not twice
    assert sum(1 for stack in evaluate.calls if stack == ["A"]) == 1


def test_autotune_trace_explains_a_near_miss():
    scores = {frozenset(): 1.0, frozenset("A"): 1.04}
    (step,) = autotune(["A"], scripted(scores)).trace
    assert not step.accepted and "gain" in step.note
